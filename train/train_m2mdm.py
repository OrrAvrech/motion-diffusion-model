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
    

def lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(1).unsqueeze(1)
    return mask


@pyrallis.wrap()
def main(cfg: TrainDiffusionConfig):
    
    # wandb.login()
    # wandb.init(project="m2m-regressor", config=asdict(cfg))

    # save_dir = cfg.save_dir
    transform = RecoverInput(cfg.model.data_rep, cfg.model.njoints)
    train_ds = MotionPairsSplit(**asdict(cfg.dataset), split=cfg.train_split_file, sample_window=True, transform=transform)
    # val_ds = MotionPairsSplit(**asdict(cfg.dataset), split=cfg.val_split_file, random_choice=False, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    # val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    max_len = train_ds.max_frames

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
    # hml2xyz = HML2XYZ(cfg.model.data_rep)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # early_stopper = EarlyStopper()

    # best_val_loss = float("inf")
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader):
            optimizer.zero_grad()

            gt_motion, pert_motion, seq_len = batch
            curr_batch_size = gt_motion.shape[0]
            gt_motion = gt_motion.to(device)
            pert_motion = pert_motion.to(device)
            seq_len = seq_len.to(device)

            schedule_sampler = UniformStepsSampler(cfg.diffusion.nsteps)
            t, weights = schedule_sampler.sample(curr_batch_size, device)

            mask = lengths_to_mask(seq_len, max_len)
            cond = {"y": {"mask": mask, 
                          "motion": pert_motion,
                          "lengths": seq_len}}
            loss = diffusion(gt_motion, t, cond)
            loss = (loss * weights).mean()

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * gt_motion.size(0)

        train_loss /= len(train_ds)
        print(f"Epoch [{epoch+1}/{cfg.epochs}], Train-Loss: {train_loss}")


if __name__ == "__main__":
    main()