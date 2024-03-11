import os
import torch
import wandb
import pyrallis
from tqdm import tqdm
from typing import Dict
from pathlib import Path
from dataclasses import asdict
from torch.utils.data import DataLoader
from model.regressor import M2MRegressor
from human_feedback.config import TrainRegressorConfig
from human_feedback.dataset import MotionPairsSplit
from human_feedback.motion_utils import RecoverInput, HML2XYZ
from human_feedback.viz import save_viz
import torch.nn.functional as F
import torch.nn as nn


class EarlyStopper:
    def __init__(self, patience: int = 3, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = float('inf')

    def stop(self, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        elif loss > (self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

def geomteric_loss(pos_true: torch.Tensor, pos_pred: torch.Tensor, weights: Dict[str, float]):
    # xyz loss
    pos_true = pos_true.permute(0, 2, 3, 1)
    pos_pred = pos_pred.permute(0, 2, 3, 1)
    pos_loss = F.mse_loss(pos_pred, pos_true)
    # velocities loss (TODO: consider removing root joint for this)
    vel_true = (pos_true[..., 1:] - pos_true[..., :-1])
    vel_pred = (pos_pred[..., 1:] - pos_pred[..., :-1])
    vel_loss = F.mse_loss(vel_pred, vel_true)
    # foot contact loss
    l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx = 7, 8, 10, 11
    relevant_joints = [l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx]
    gt_joint_xyz = pos_true[:, relevant_joints, :, :]  # [BatchSize, 4, 3, Frames]
    gt_joint_vel = torch.linalg.norm(gt_joint_xyz[:, :, :, 1:] - gt_joint_xyz[:, :, :, :-1], axis=2)  # [BatchSize, 4, Frames]
    fc_mask = torch.unsqueeze((gt_joint_vel <= 0.01), dim=2).repeat(1, 1, 3, 1)
    pred_joint_xyz = pos_pred[:, relevant_joints, :, :]  # [BatchSize, 4, 3, Frames]
    pred_vel = pred_joint_xyz[:, :, :, 1:] - pred_joint_xyz[:, :, :, :-1]
    pred_vel[~fc_mask] = 0
    fc_loss = F.mse_loss(pred_vel, torch.zeros_like(pred_vel, device=pred_vel.device))
    # geometric loss
    loss = weights["pos"] * pos_loss + weights["vel"] * vel_loss + weights["foot"] * fc_loss
    return loss


@pyrallis.wrap()
def main(cfg: TrainRegressorConfig):
    
    wandb.login()
    wandb.init(project="m2m-regressor", config=asdict(cfg))

    save_dir = cfg.save_dir
    transform = RecoverInput(cfg.model.njoints, cfg.model.data_rep)
    train_ds = MotionPairsSplit(**asdict(cfg.dataset), split=cfg.train_split_file, sample_window=True, transform=transform)
    val_ds = MotionPairsSplit(**asdict(cfg.dataset), split=cfg.val_split_file, random_choice=False, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    model = M2MRegressor(**asdict(cfg.model))
    if cfg.model_path is not None:
        model.load_state_dict(torch.load(cfg.model_path))
    device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.rot2xyz.smpl_model.eval()
    hml2xyz = HML2XYZ()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    early_stopper = EarlyStopper()

    best_val_loss = float("inf")
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader):
            gt_motion, pert_motion, _ = batch
            gt_motion = gt_motion.to(device)
            pert_motion = pert_motion.to(device)
            optimizer.zero_grad()
        
            output_motion = model(pert_motion)

            # forward kinematics for loss
            gt_pos = hml2xyz(gt_motion)
            output_pos = hml2xyz(output_motion)
            gloss = geomteric_loss(gt_pos, output_pos, weights=cfg.geometric_loss_weights)
            loss = F.mse_loss(output_motion, gt_motion) + gloss
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * gt_motion.size(0)
        train_loss /= len(train_ds)

        # Validation Loop
        model.eval()
        val_loss = 0.0
        for i, val_batch in enumerate(val_loader):
            gt_motion, pert_motion, seq_len = val_batch
            gt_motion = gt_motion.to(device)
            pert_motion = pert_motion.to(device)
        
            with torch.no_grad():
                output_motion = model(pert_motion)
                batch_val_gloss = geomteric_loss(gt_pos, output_pos, weights=cfg.geometric_loss_weights)
                batch_val_loss = F.mse_loss(output_motion, gt_motion) + batch_val_gloss
            val_loss += batch_val_loss
                
        val_loss /= len(val_ds)

        wandb.log({"loss": train_loss, "val_loss": val_loss})
        print(f"Epoch [{epoch+1}/{cfg.epochs}], Train-Loss: {train_loss}, Validation-Loss: {val_loss}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = save_dir / f"{cfg.save_dir.name}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"Saved model {save_path}")
        
        if early_stopper.stop(val_loss):
            # visualize motions
            gt_motion_pos = hml2xyz(gt_motion)
            pert_motion_pos = hml2xyz(pert_motion)
            output_motion_pos = hml2xyz(output_motion)
            for j in range(len(gt_motion)):
                if j+1 <= cfg.viz_samples_per_batch:
                    idx = f"epoch{epoch}_batch{i}_sample{j}"
                    sample_seq_len = seq_len[j]
                    vid_path = save_viz(gt_motion_pos[j, :sample_seq_len, :, :],
                                        pert_motion_pos[j, :sample_seq_len, :, :], 
                                        output_motion_pos[j, :sample_seq_len, :, :], 
                                        save_dir=save_dir, 
                                        idx=idx)
                    wandb.log({"video": wandb.Video(str(vid_path))})
            break



if __name__ == "__main__":
    main()