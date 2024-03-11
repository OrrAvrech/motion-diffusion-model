import torch
import pyrallis
from tqdm import tqdm
from dataclasses import asdict
from torch.utils.data import DataLoader
from model.regressor import M2MRegressor
from human_feedback.config import EvalConfig
from human_feedback.dataset import MotionPairsSplit
from human_feedback.motion_utils import RecoverInput, HML2XYZ
from human_feedback.viz import save_viz
     

@pyrallis.wrap()
def main(cfg: EvalConfig):
    model_dir = cfg.model_path.parent
    save_dir = model_dir / "viz"
    save_dir.mkdir(exist_ok=True, parents=True)

    transform = RecoverInput(cfg.model.njoints, cfg.model.data_rep)
    val_ds = MotionPairsSplit(**asdict(cfg.dataset), split=cfg.val_split_file, random_choice=False, transform=transform)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    model = M2MRegressor(**asdict(cfg.model))
    model.load_state_dict(torch.load(cfg.model_path))
    device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.rot2xyz.smpl_model.eval()
    hml2xyz = HML2XYZ()

    # Validation Loop
    model.eval()
    for i, val_batch in enumerate(val_loader):
        gt_motion, pert_motion, seq_len = val_batch
        gt_motion = gt_motion.to(device)
        pert_motion = pert_motion.to(device)
    
        with torch.no_grad():
            output_motion = model(pert_motion)
        
        # visualize motions
        print("Visualize motions...")
        gt_motion_pos = hml2xyz(gt_motion)
        pert_motion_pos = hml2xyz(pert_motion)
        output_motion_pos = hml2xyz(output_motion)
        for j in tqdm(range(len(gt_motion))):
            if j+1 <= cfg.viz_samples_per_batch:
                idx = f"batch{i}_sample{j}"
                sample_seq_len = seq_len[j]
                save_viz(gt_motion_pos[j, :sample_seq_len, :, :],
                            pert_motion_pos[j, :sample_seq_len, :, :], 
                            output_motion_pos[j, :sample_seq_len, :, :], 
                            save_dir=save_dir, 
                            idx=idx)
        # break after a single batch TEMP
        break


if __name__ == "__main__":
    main()