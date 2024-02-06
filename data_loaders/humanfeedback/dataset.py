import torch
import random
from pathlib import Path
from torch.utils import data
from typing import Optional, Tuple
import numpy as np


class MotionPairs(data.Dataset):
    def __init__(self, input_dir: Path, pert_dir: Path, max_frames: Optional[int] = None,
                 transform = None) -> None:
        self.input_dir = input_dir
        self.pert_dir = pert_dir
        self.max_frames = max_frames
        self.transform = transform
        self.motion_files = list(self.input_dir.rglob("*.npy"))

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        input_filepath = self.motion_files[idx]
        input_motion = torch.Tensor(np.load(input_filepath))[:self.max_frames, ...]
        pert_motion_dir = self.pert_dir / input_filepath.stem
        pert_motions_files = list(pert_motion_dir.glob("*.npy"))
        # randomly choose an element if more than one perturbations exist
        pert_filepath = random.choice(pert_motions_files)
        pert_motion = torch.Tensor(np.load(pert_filepath))[:self.max_frames, ...]
        if self.transform is not None:
            input_motion = self.transform(input_motion)
            pert_motion = self.transform(pert_motion)
        return input_motion, pert_motion

    def __len__(self):
        return len(self.motion_files)
    

class MotionPairsSplit(MotionPairs):
    def __init__(self, input_dir: Path, pert_dir: str, split: str, max_frames: Optional[int] = None,
                 transform = None) -> None:
        super().__init__(input_dir=input_dir, pert_dir=pert_dir, max_frames=max_frames, transform=transform)
        self.split_file = input_dir.parent / f"{split}.txt"
        id_list = []
        with open(self.split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        self.motion_files = [p for p in self.motion_files if p.stem in id_list]


