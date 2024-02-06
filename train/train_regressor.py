import torch
import pyrallis
from dataclasses import asdict
from torch.utils.data import DataLoader
from model.regressor import M2MRegressor
from config.humanfeedback_config import TrainRegressorConfig
from data_loaders.humanfeedback.dataset import MotionPairsSplit
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import torch.nn as nn


class RecoverPositions:
    # HumanML3D recover joint positions from vec
    def __init__(self, num_joints: int):
        self.num_joints = num_joints

    def __call__(self, sample):
        xyz = recover_from_ric(sample, self.num_joints).permute(1, 2, 0)
        return xyz
    

@pyrallis.wrap()
def main(cfg: TrainRegressorConfig):
    model = M2MRegressor(**asdict(cfg.model))
    transform = RecoverPositions(cfg.model.njoints)
    ds = MotionPairsSplit(**asdict(cfg.dataset), split="train", transform=transform)
    data_loader = DataLoader(ds, batch_size=cfg.batch_size)

    device = torch.device("cuda:" + str(cfg.device) if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.rot2xyz.smpl_model.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    for epoch in range(cfg.epochs):
        for batch in data_loader:
            gt_motion, pert_motion = batch
            gt_motion = gt_motion.to(device)
            pert_motion = pert_motion.to(device)
            optimizer.zero_grad()
        
            output_motion = model(pert_motion)
            loss = loss_fn(gt_motion, output_motion)
            
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{cfg.epochs}], Loss: {loss.item():.4f}")



if __name__ == "__main__":
    main()