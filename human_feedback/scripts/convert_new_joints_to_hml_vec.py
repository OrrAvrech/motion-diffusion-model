import torch
from pathlib import Path
import numpy as np
from human_feedback.motion_process import extract_features, recover_from_ric


def align_by_root(x: torch.Tensor):
    root = x[:, 0, :].unsqueeze(1)
    return x - root


def main():
    data_dir = Path("/proj/vondrick2/orr/motion-diffusion-model/dataset/MOYO/human_feedback/vecs_12")
    output_dir = Path("/proj/vondrick2/orr/motion-diffusion-model/dataset/MOYO/human_feedback/vecs_12_hml_vec")
    output_dir.mkdir(exist_ok=True, parents=True)
    feet_thresh = 0.002
    num_joints = 22
    for npy_path in data_dir.rglob("*.npy"):
        parent = npy_path.parent.name
        pos = np.load(npy_path)
        hml_vec1 = extract_features(positions=pos, feet_thre=feet_thresh)
        pos1 = recover_from_ric(torch.Tensor(hml_vec1), num_joints)
        pos_tensor = torch.Tensor(pos)[:-1, ...]
        print(torch.mean(torch.norm(align_by_root(pos_tensor) - align_by_root(pos1), dim=-1)))

        save_dir = output_dir / parent
        save_dir.mkdir(exist_ok=True)
        np.save(save_dir / npy_path.name, hml_vec1)
        
        # mdm_output_path = data_dir.parent / "mdm_output.mp4"
        # extract_features_path = data_dir.parent / "extract_features.mp4"
        # process_file_path = data_dir.parent / "process_file.mp4"
        # plot_3d_motion(mdm_output_path, joints=pos, title="MDM output")
        # plot_3d_motion(extract_features_path, joints=pos1.cpu().numpy(), title="extarct features")
        # plot_3d_motion(process_file_path, joints=pos2.cpu().numpy(), title="process file")
        # all_path = data_dir.parent / "all.mp4"
        # save_vid_list([mdm_output_path, extract_features_path, process_file_path], all_path)


if __name__ == "__main__":
    main()
