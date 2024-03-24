import torch
import random
from pathlib import Path
from torch.utils import data
from typing import Optional, Tuple
import numpy as np


class MotionPairs(data.Dataset):
    def __init__(self, input_dir: Path, pert_dir: Path, max_frames: Optional[int] = None,
                 random_choice: bool = True, sample_window: bool = False, transform = None) -> None:
        self.input_dir = input_dir
        self.pert_dir = pert_dir
        self.max_frames = max_frames
        self.random_choice = random_choice
        self.sample_window = sample_window
        self.transform = transform
        self.motion_files = list(self.input_dir.rglob("*.npy"))

        self.mean = 0.0
        mean_npy_path = input_dir.parent / "Mean_all.npy"
        if mean_npy_path.exists():
            print(f"load mean from {mean_npy_path}")
            self.mean = np.load(mean_npy_path).astype(np.float32)
        
        mean_pert_npy_path = input_dir.parent / "Mean_pert_all.npy"
        if mean_pert_npy_path.exists():
            print(f"load pert mean from {mean_pert_npy_path}")
            self.pert_mean = np.load(mean_pert_npy_path).astype(np.float32)
        else:
            self.pert_mean = self.mean

        self.std = 1.0
        std_npy_path = input_dir.parent / "Std_all.npy"
        if std_npy_path.exists():
            print(f"load std from {std_npy_path}")
            self.std = np.load(std_npy_path).astype(np.float32)

        std_pert_npy_path = input_dir.parent / "Std_pert_all.npy"
        if std_pert_npy_path.exists():
            print(f"load pert std from {std_pert_npy_path}")
            self.pert_std = np.load(std_pert_npy_path).astype(np.float32)
        else:
            self.pert_std = self.std

    def process_motion(self, motion: torch.Tensor, seq_len: int, 
                       mean: torch.Tensor, std: torch.Tensor,
                       start_idx: Optional[int] = None) -> Tuple[torch.Tensor]:
        if seq_len < self.max_frames:
            # zero padding
            zeros = torch.zeros((self.max_frames, *motion.shape[1:]))
            zeros[:seq_len, ...] = motion
            motion_proc = zeros
        else:
            if self.sample_window is True:
                # randomly sample a window of max-frames size
                start_idx = start_idx if start_idx is not None else random.randint(0, seq_len - self.max_frames - 10)
                motion_proc = motion[start_idx:start_idx + self.max_frames, ...]
            else:
                motion_proc = motion[:self.max_frames, ...]
            seq_len = motion_proc.shape[0]

        # Z Normalization
        motion_proc = (motion_proc - mean) / std
        return motion_proc, seq_len, start_idx

    def denormalize(self, motion: torch.Tensor):
        mean = (torch.Tensor(self.mean) * torch.ones(motion.shape[1])).view(1, -1, 1, 1)
        std = (torch.Tensor(self.std) * torch.ones(motion.shape[1])).view(1, -1, 1, 1)
        return motion * std + mean

    def __getitem__(self, idx) -> Tuple[torch.Tensor]:
        gt_filepath = self.motion_files[idx]
        gt_motion = torch.Tensor(np.load(gt_filepath))
        pert_motion_dir = self.pert_dir / gt_filepath.stem
        pert_motions_files = list(pert_motion_dir.rglob("*.npy"))
        if self.random_choice is True:
            # randomly choose an element if more than one perturbations exist
            pert_filepath = random.choice(pert_motions_files)
        else:
            pert_filepath = pert_motions_files[0]
        pert_motion = torch.Tensor(np.load(pert_filepath))
        min_seq_len = min(gt_motion.shape[0], pert_motion.shape[0])

        gt_motion, seq_len, start_idx = self.process_motion(gt_motion,
                                                            min_seq_len, 
                                                            mean=self.mean, 
                                                            std=self.std)
        pert_motion, _, _ = self.process_motion(pert_motion, 
                                                min_seq_len, 
                                                mean=self.pert_mean,
                                                std=self.pert_std,
                                                start_idx=start_idx)

        if self.transform is not None:
            gt_motion = self.transform(gt_motion)
            pert_motion = self.transform(pert_motion)
        return gt_motion, pert_motion, seq_len

    def __len__(self):
        return len(self.motion_files)
    

class MotionPairsSplit(MotionPairs):
    def __init__(self, input_dir: Path, pert_dir: str, split: str, max_frames: Optional[int] = None,
                 random_choice: bool = True, sample_window: bool = False, transform = None) -> None:
        super().__init__(input_dir=input_dir, pert_dir=pert_dir, random_choice=random_choice,
                         max_frames=max_frames, sample_window=sample_window, transform=transform)
        self.split_file = input_dir.parent / f"{split}.txt"
        id_list = []
        with open(self.split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        self.motion_files = [p for p in self.motion_files if p.stem in id_list]


class IdentityPairsSplit(MotionPairsSplit):
    def __init__(self, input_dir: Path, pert_dir: str, split: str, max_frames: Optional[int] = None,
                 random_choice: bool = True, sample_window: bool = False, transform = None) -> None:
        super().__init__(input_dir=input_dir, pert_dir=pert_dir, random_choice=random_choice, split=split,
                         max_frames=max_frames, sample_window=sample_window, transform=transform)
        
    def __getitem__(self, idx) -> Tuple[torch.Tensor]:
        gt_filepath = self.motion_files[idx]
        gt_motion = torch.Tensor(np.load(gt_filepath))
        pert_motion = torch.Tensor(np.load(gt_filepath))

        gt_motion, seq_len = self.process_motion(gt_motion)
        pert_motion, _ = self.process_motion(pert_motion)

        if self.transform is not None:
            gt_motion = self.transform(gt_motion)
            pert_motion = self.transform(pert_motion)
        return gt_motion, pert_motion, seq_len
        


