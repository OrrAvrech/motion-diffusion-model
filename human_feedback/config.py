from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class MotionDatasetConfig:
    input_dir: Path
    pert_dir: Path
    max_frames: Optional[int] = None


@dataclass
class EncoderConfig:
    njoints: int
    nfeats: int
    latent_dim: int = 256
    ff_size: int = 1024
    num_layers: int = 8 
    num_heads: int = 4
    dropout: float = 0.1
    activation: str = "gelu"
    data_rep: str = "rot6d" 
    dataset: str = "amass"
    arch: str = "trans_enc"


@dataclass
class TrainRegressorConfig:
    dataset: MotionDatasetConfig
    model: EncoderConfig
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    geometric_loss_weights: List[float]
    save_dir: Path
    device: Optional[int] = 0

    def __post_init__(self):
        self.save_dir.mkdir(exist_ok=True, parents=True)

