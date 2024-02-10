import os
import torch
import pyrallis
from tqdm import tqdm
from typing import List
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from torch.utils.data import DataLoader
from model.regressor import M2MRegressor
from human_feedback.config import TrainRegressorConfig
from human_feedback.dataset import MotionPairsSplit
from human_feedback.viz import plot_3d_motion, save_vid_list
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import torch.nn as nn
import torch.nn.functional as F


class RecoverPositions:
    # HumanML3D recover joint positions from vec
    def __init__(self, num_joints: int):
        self.num_joints = num_joints

    def __call__(self, sample):
        xyz = recover_from_ric(sample, self.num_joints).permute(1, 2, 0)
        return xyz
    

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
    

def geomteric_loss(pos_true: torch.Tensor, pos_pred: torch.Tensor, weights: List[float]):
    # xyz loss
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
    pos_lambda, vel_lambda, fc_lambda = weights
    loss = pos_lambda * pos_loss + vel_lambda * vel_loss + fc_lambda * fc_loss
    return loss


def save_viz(gt: torch.Tensor, perturbated: torch.Tensor, pred: torch.Tensor, 
             save_dir: Path, idx: str, fps: float = 40):
    save_dir.mkdir(exist_ok=True, parents=True)
    gt_save_path = save_dir / f"gt_{idx}.mp4"
    gt_np = gt.detach().cpu().permute(2, 0, 1).numpy()
    plot_3d_motion(gt_save_path, gt_np, title="GT", fps=fps)

    pert_save_path = save_dir / f"pert_{idx}.mp4"
    pert_np = perturbated.detach().cpu().permute(2, 0, 1).numpy()
    plot_3d_motion(pert_save_path, pert_np, title="Perturbed", fps=fps)

    pred_save_path = save_dir / f"pred_{idx}.mp4"
    pred_np = pred.detach().cpu().permute(2, 0, 1).numpy()
    plot_3d_motion(pred_save_path, pred_np, title="Pred", fps=fps)

    saved_files = [gt_save_path, pert_save_path, pred_save_path]
    save_path = save_dir / f"{idx}.mp4"
    save_vid_list(saved_files=saved_files, save_path=save_path)
    [os.remove(filepath) for filepath in saved_files]


@pyrallis.wrap()
def main(cfg: TrainRegressorConfig):
    run_time = datetime.now().strftime("%Y%m%d_%H%M")
    model = M2MRegressor(**asdict(cfg.model))

    transform = RecoverPositions(cfg.model.njoints)
    train_ds = MotionPairsSplit(**asdict(cfg.dataset), split="train", transform=transform)
    val_ds = MotionPairsSplit(**asdict(cfg.dataset), split="val", random_choice=False, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    device = torch.device("cuda:" + str(cfg.device) if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.rot2xyz.smpl_model.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    early_stopper = EarlyStopper()

    best_val_loss = float("inf")
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader):
            gt_motion, pert_motion = batch
            gt_motion = gt_motion.to(device)
            pert_motion = pert_motion.to(device)
            optimizer.zero_grad()
        
            output_motion = model(pert_motion)
            loss = geomteric_loss(gt_motion, output_motion, weights=cfg.geometric_loss_weights)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * gt_motion.size(0)
        train_loss /= len(train_ds)

        # Validation Loop
        model.eval()
        val_loss = 0.0
        for i, val_batch in enumerate(val_loader):
            gt_motion, pert_motion = val_batch
            gt_motion = gt_motion.to(device)
            pert_motion = pert_motion.to(device)
        
            with torch.no_grad():
                output_motion = model(pert_motion)
                batch_val_loss = geomteric_loss(gt_motion, output_motion, weights=cfg.geometric_loss_weights)
            val_loss += batch_val_loss

            for j in range(len(gt_motion)):
                if j < 1:
                    idx = f"epoch{epoch}_batch{i}_sample{j}"
                    save_dir_viz = cfg.save_dir / f"{cfg.save_dir.name}_{run_time}" / f"epoch_{epoch}"
                    save_viz(gt_motion[j], pert_motion[j], output_motion[j], save_dir=save_dir_viz, idx=idx)
        val_loss /= len(val_ds)

        print(f"Epoch [{epoch+1}/{cfg.epochs}], Train-Loss: {train_loss:.4f}, Validation-Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = cfg.save_dir / f"{cfg.save_dir.name}_{run_time}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"Saved model {save_path}")
        
        if early_stopper.stop(val_loss):
            print("Early Stopping")
            break



if __name__ == "__main__":
    main()