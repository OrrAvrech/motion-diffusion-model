import os
import torch
import wandb
import pyrallis
from tqdm import tqdm
from dataclasses import asdict
from torch.utils.data import DataLoader
from model.m2mdm import select_model
from model.diffusion import GaussianDiffusion
from diffusion.resample import UniformStepsSampler
from human_feedback.config import TrainDiffusionConfig
from human_feedback.dataset import MotionPairsSplit, IdentityPairsSplit
from human_feedback.motion_utils import RecoverInput, HML2XYZ
from human_feedback.metrics import compute_mpjpe, compute_pa_mpjpe
from human_feedback.viz import save_viz
import torch.nn.functional as F


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
    
    wandb.login()
    # run = wandb.init(project="m2mdm", config=asdict(cfg), tags=["identity"])
    run = wandb.init(project="m2mdm", config=asdict(cfg))

    save_dir = cfg.save_dir
    transform = RecoverInput(cfg.model.data_rep, cfg.model.njoints)
    train_ds = MotionPairsSplit(**asdict(cfg.dataset), split=cfg.train_split_file, sample_window=True, transform=transform)
    val_ds = MotionPairsSplit(**asdict(cfg.dataset), split=cfg.val_split_file, random_choice=False, transform=transform)

    # train_ds = IdentityPairsSplit(**asdict(cfg.dataset), split=cfg.train_split_file, sample_window=True, transform=transform)
    # val_ds = IdentityPairsSplit(**asdict(cfg.dataset), split=cfg.val_split_file, random_choice=False, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_data_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    max_len = train_ds.max_frames

    model = select_model(cfg.model.arch)(**asdict(cfg.model), seq_len=cfg.dataset.max_frames)
    diffusion = GaussianDiffusion(model=model, **asdict(cfg.diffusion))
    
    device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
    if cfg.model_path is not None:
        print(f"load pre-trained model from {cfg.model_path}")
        model.load_state_dict(torch.load(cfg.model_path, map_location=device))
    
    model.to(device)
    diffusion.to(device)

    model.smpl.smpl_model.eval()
    hml2xyz = HML2XYZ(cfg.model.data_rep)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    early_stopper = EarlyStopper(cfg.patience)

    best_val_metric = float("inf")
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
            mask = lengths_to_mask(seq_len, max_len)

            schedule_sampler = UniformStepsSampler(cfg.diffusion.nsteps)
            t, weights = schedule_sampler.sample(curr_batch_size, device)

            cond = {"y": {"mask": mask, 
                          "motion": pert_motion,
                          "lengths": seq_len}}
            loss = diffusion(gt_motion, t, cond)
            loss = (loss * weights).mean()

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * curr_batch_size

        train_loss /= len(train_ds)
        logs = {"train/loss": train_loss}
        print(f"Epoch [{epoch+1}/{cfg.epochs}], Train-Loss: {train_loss}")

        if epoch % cfg.log_interval == 0:
            # Validation Loop
            model.eval()
            val_mse, val_mpjpe, val_mpjpe_align = [0.0 for _ in range(3)]
            for i, val_batch in enumerate(val_loader):
                gt_motion, pert_motion, seq_len = val_batch
                curr_batch_size = gt_motion.shape[0]
                gt_motion = gt_motion.to(device)
                pert_motion = pert_motion.to(device)

                cond = {"y": {"motion": pert_motion}}
                with torch.no_grad():
                    sample = diffusion.p_sample_loop(shape=gt_motion.shape,
                                                    model_kwargs=cond,
                                                    progress=True)
                    
                # extract joint positions
                gt_motion_pos = hml2xyz(val_ds.denormalize(gt_motion.detach().cpu()))
                pert_motion_pos = hml2xyz(val_ds.denormalize(pert_motion.detach().cpu()))
                output_motion_pos = hml2xyz(val_ds.denormalize(sample.detach().cpu()))

                val_mse += F.mse_loss(gt_motion, sample) * curr_batch_size
                val_mpjpe += compute_mpjpe(gt_motion_pos, output_motion_pos, root_align=False) * curr_batch_size
                val_mpjpe_align += compute_mpjpe(gt_motion_pos, output_motion_pos, root_align=True) * curr_batch_size

            val_mse /= len(val_ds)
            val_mpjpe /= len(val_ds)
            val_mpjpe_align /= len(val_ds)
            logs.update({"val/mse": val_mse, "val/mpjpe": val_mpjpe, "val/mpjpe-align": val_mpjpe_align})
            print(f"Validation-MSE: {val_mse}, Validation-MPJPE: {val_mpjpe}, Validation-MPJPE-Aligned: {val_mpjpe_align}")

            if val_mpjpe < best_val_metric:
                best_val_metric = val_mpjpe
                save_path = save_dir / f"{cfg.save_dir.name}.pt"
                torch.save(model.state_dict(), save_path)
                print(f"Saved model {save_path}")

        if epoch % int(cfg.log_interval*2) == 0:
            save_dir_viz = save_dir / f"viz_{save_dir.name}"
            print(f"Visualize motions saved in {save_dir_viz}")
            examples = []
            for j in range(len(gt_motion)):
                if j+1 <= cfg.viz_samples_per_batch:
                    idx = f"epoch{epoch}_batch{i}_sample{j}"
                    sample_seq_len = seq_len[j]
                    vid_path = save_viz(gt_motion_pos[j, :sample_seq_len, :, :],
                                        pert_motion_pos[j, :sample_seq_len, :, :], 
                                        output_motion_pos[j, :sample_seq_len, :, :], 
                                        save_dir=save_dir_viz, 
                                        idx=idx)
                    examples.append(wandb.Video(str(vid_path)))
            logs.update({"videos": examples})
        run.log(logs)
        
        if early_stopper.stop(val_mpjpe):
            print(f"Early stop, no improvement for {early_stopper.patience} epochs")
            break


if __name__ == "__main__":
    main()