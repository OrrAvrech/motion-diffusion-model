import os
import torch
import wandb
import pyrallis
from tqdm import tqdm
from typing import Dict
from pathlib import Path
from dataclasses import asdict
from torch.utils.data import DataLoader
from model.m2mdm import M2MDM
from model.diffusion import GaussianDiffusion
from diffusion.resample import UniformStepsSampler
from human_feedback.config import TrainDiffusionConfig
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
def main(cfg: TrainDiffusionConfig):
    
    # wandb.login()
    # wandb.init(project="m2m-regressor", config=asdict(cfg))

    save_dir = cfg.save_dir
    transform = RecoverInput(cfg.model.data_rep, cfg.model.njoints)
    train_ds = MotionPairsSplit(**asdict(cfg.dataset), split=cfg.train_split_file, sample_window=True, transform=transform)
    val_ds = MotionPairsSplit(**asdict(cfg.dataset), split=cfg.val_split_file, random_choice=False, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    model = M2MDM(**asdict(cfg.model))
    diffusion = GaussianDiffusion(model=model,
                                  diffusion_steps=cfg.diffusion.nsteps,
                                  noise_schedule=cfg.diffusion.noise_schedule)
    
    # if cfg.model_path is not None:
    #     model.load_state_dict(torch.load(cfg.model_path))
    
    device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    diffusion.to(device)

    model.smpl.smpl_model.eval()
    hml2xyz = HML2XYZ(cfg.model.data_rep)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    early_stopper = EarlyStopper()

    best_val_loss = float("inf")
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader):
            optimizer.zero_grad()

            gt_motion, pert_motion, _ = batch
            gt_motion = gt_motion.to(device)
            pert_motion = pert_motion.to(device)
            schedule_sampler = UniformStepsSampler(cfg.diffusion.nsteps)
            t, weights = schedule_sampler.sample(cfg.batch_size, device)

            loss = diffusion(gt_motion, t, pert_motion)
            loss = loss * weights 

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * gt_motion.size(0)
        train_loss /= len(train_ds)


if __name__ == "__main__":
    main()