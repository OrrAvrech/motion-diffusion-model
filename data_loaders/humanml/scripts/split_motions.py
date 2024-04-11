from pathlib import Path
import numpy as np
from shutil import copy


def main():
    subsample_factor = 2
    num_frames = int(196*subsample_factor)

    dataset_dir = Path("/proj/vondrick2/orr/motion-diffusion-model/dataset/MotionX")
    vecs_dir = dataset_dir / "vector_263"
    output_dir = dataset_dir / "human_feedback" / "new_joint_vecs_split"
    output_dir.mkdir(exist_ok=True, parents=True)
    # text_dir = dataset_dir / "texts"
    # new_text_dir = dataset_dir / "texts_split"
    # new_text_dir.mkdir(exist_ok=True, parents=True)
    split_filepath = dataset_dir / "vecs_12.txt"

    filenames = []
    for npy_path in vecs_dir.rglob("*.npy"):
        full_motion = np.load(npy_path)
        for i in range(0, len(full_motion) - num_frames, num_frames):
            curr_motion = full_motion[i: i+num_frames]
            # MOYO is in 40fps, HumanML3D 20fps, MotionX 30fps
            curr_motion = curr_motion[::subsample_factor]
            if len(curr_motion) <= 60:
                 # skip short motions
                continue

            motion_filename = f"{npy_path.stem}_{i}_{i+num_frames}.npy"
            filenames.append(motion_filename)
            np.save(output_dir / motion_filename, curr_motion)

            # copy(text_dir / f"{npy_path.stem}.txt", new_text_dir / f"{Path(motion_filename).stem}.txt")
    
    with open(split_filepath, "a") as fp:
        [fp.write(f"{x}\n") for x in filenames]


if __name__ == "__main__":
    main()