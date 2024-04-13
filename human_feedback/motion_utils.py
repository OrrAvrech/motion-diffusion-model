import torch
from typing import Optional, Union

from torch import Tensor
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import numpy as np


class RecoverInput:
    # recover joint positions or HumanML3D vec
    def __init__(self, data_rep: str, njoints: int):
        self.data_rep = data_rep
        self.njoints = njoints

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        # vec -> [njoints, nfeats, nframes]
        if self.data_rep == "xyz":
            vec = recover_from_ric(sample, self.njoints).permute(1, 2, 0)
        else:
            vec = sample.unsqueeze(1).permute(2, 1, 0)
        return vec
    

class HML2XYZ:
    def __init__(self, data_rep: str, njoints: Optional[int] = 22):
        self.data_rep = data_rep
        self.njoints = njoints

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        if self.data_rep == "xyz":
            xyz = sample
        else:
            hml = sample.squeeze(2).permute(0, 2, 1)
            xyz = recover_from_ric(hml, self.njoints)
        return xyz


def subsample_motion(motion: np.array, subsample_factor: int) -> np.array:
    new_indices = np.arange(0, len(motion), subsample_factor)
    # Round the indices to the nearest integer to match array indices
    new_indices = np.round(new_indices).astype(int)
    # Ensure that indices are within the bounds of the original array
    new_indices = np.clip(new_indices, 0, len(motion) - 1)
    # Subsample the array using the new indices
    new_motion = motion[new_indices]
    return new_motion


def align_by_root(x: Tensor) -> Tensor:
    root = x[:, 0, :].unsqueeze(1)
    return x - root


def align_by_root_np(x: np.array) -> np.array:
    root = np.expand_dims(x[:, 0, :], axis=1)
    return x - root