import torch
import pyrallis
from tqdm import tqdm
from datetime import datetime
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
    loss_fn = nn.MSELoss()
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
            loss = loss_fn(gt_motion, output_motion)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * gt_motion.size(0)
        train_loss /= len(train_ds)

        # Validation Loop
        model.eval()
        val_loss = 0.0
        for val_batch in val_loader:
            gt_motion, pert_motion = val_batch
            gt_motion = gt_motion.to(device)
            pert_motion = pert_motion.to(device)
        
            with torch.no_grad():
                output_motion = model(pert_motion)
                batch_val_loss = loss_fn(gt_motion, output_motion)
            val_loss += batch_val_loss
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