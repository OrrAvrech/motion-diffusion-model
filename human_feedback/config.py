from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Union


@dataclass
class MotionDatasetConfig:
    input_dir: Path
    pert_dir: Path
    max_frames: Optional[int] = None


@dataclass
class TransformerConfig:
    njoints: int
    nfeats: int
    latent_dim: int = 256
    ff_size: int = 1024
    num_layers: int = 8 
    num_heads: int = 4
    dropout: float = 0.1
    activation: str = "gelu"
    data_rep: str = "rot6d" 
    arch: str = "trans_enc"


@dataclass
class TrainRegressorConfig:
    dataset: MotionDatasetConfig
    model: TransformerConfig
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    geometric_loss_weights: Dict[str, float]
    save_dir: Path
    patience: Optional[int] = None
    log_interval: Optional[int] = None
    model_path: Optional[Path] = None
    viz_samples_per_batch: Optional[int] = 0
    device: Optional[Union[List[int], int]] = 0
    train_split_file: Optional[str] = "train"
    val_split_file: Optional[str] = "val"

    def __post_init__(self):
        self.save_dir.mkdir(exist_ok=True, parents=True)
        if self.log_interval is None:
            self.log_interval = (self.epochs - 1) // 2 # record first, middle, last epochs by default
        if self.patience is None:
            self.patience = self.epochs # no early-stop by default


@dataclass
class DiffusionConfig:
    nsteps: int
    noise_schedule: str
    cond_mask_prob: float
    guidance_weight: float


@dataclass
class TrainDiffusionConfig:
    dataset: MotionDatasetConfig
    model: TransformerConfig
    diffusion: DiffusionConfig
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    geometric_loss_weights: Dict[str, float]
    save_dir: Path
    patience: Optional[int] = None
    log_interval: Optional[int] = None
    model_path: Optional[Path] = None
    viz_samples_per_batch: Optional[int] = 0
    device: Optional[Union[List[int], int]] = 0
    train_split_file: Optional[str] = "train"
    val_split_file: Optional[str] = "val"

    def __post_init__(self):
        self.save_dir.mkdir(exist_ok=True, parents=True)
        if self.log_interval is None:
            self.log_interval = (self.epochs - 1) // 2 # record first, middle, last epochs by default
        if self.patience is None:
            self.patience = self.epochs # no early-stop by default


@dataclass
class EvalConfig:
    dataset: MotionDatasetConfig
    model: TransformerConfig
    model_path: Path
    batch_size: int
    viz_samples_per_batch: Optional[int] = 0
    device: Optional[Union[List[int], int]] = 0
    train_split_file: Optional[str] = "train"
    val_split_file: Optional[str] = "val"