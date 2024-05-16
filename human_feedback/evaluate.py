import torch
import random
import pyrallis
from tqdm import tqdm
from dataclasses import asdict
from torch.utils.data import DataLoader
from model.diffusion import GaussianDiffusion
from model.regressor import M2MRegressor
from model.m2mdm import select_model
from human_feedback.config import EvalConfig
from human_feedback.dataset import GoldenPairs
from human_feedback.motion_utils import RecoverInput, HML2XYZ
from human_feedback.viz import save_viz
import numpy as np

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

@pyrallis.wrap()
def main(cfg: EvalConfig):
    model_dir = cfg.model_path.parent
    save_dir = model_dir / "golden"
    save_dir.mkdir(exist_ok=True, parents=True)

    transform = RecoverInput(cfg.model.data_rep, cfg.model.njoints)
    test_ds = GoldenPairs(**asdict(cfg.dataset), transform=transform)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

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

    # Test Loop
    model.eval()
    for i, test_batch in enumerate(test_loader):
        correct_motion, wrong_motion, seq_len = test_batch
        correct_motion = correct_motion.to(device)
        wrong_motion = wrong_motion.to(device)
    
        cond = {"y": {"motion": wrong_motion}}
        with torch.no_grad():
            sample = diffusion.p_sample_loop(shape=correct_motion.shape,
                                            model_kwargs=cond,
                                            progress=True)
            
        # extract joint positions
        correct_motion_pos = hml2xyz(test_ds.denormalize(correct_motion.detach().cpu()))
        wrong_motion_pos = hml2xyz(test_ds.denormalize(wrong_motion.detach().cpu()))
        output_motion_pos = hml2xyz(test_ds.denormalize(sample.detach().cpu()))
        
        # visualize motions
        print("Visualize motions...")
        for j in tqdm(range(len(correct_motion))):
            if j+1 <= cfg.viz_samples_per_batch:
                idx = f"batch{i}_sample{j}"
                sample_seq_len = seq_len[j]
                save_viz(correct_motion_pos[j, :sample_seq_len, :, :],
                         wrong_motion_pos[j, :sample_seq_len, :, :], 
                         output_motion_pos[j, :sample_seq_len, :, :], 
                         save_dir=save_dir, 
                         idx=idx)
        # break after a single batch TEMP
        # break


if __name__ == "__main__":
    main()