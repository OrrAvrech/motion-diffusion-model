from pathlib import Path
from shutil import move


def main():
    vecs_dir = Path("/proj/vondrick2/orr/motion-diffusion-model/dataset/MOYO/human_feedback/vecs_12_new_joint_vecs")
    for npy_path in vecs_dir.rglob("*.npy"):
        npy_dirname = npy_path.stem[:-2]
        npy_dir = npy_path.parent / npy_dirname
        npy_dir.mkdir(exist_ok=True, parents=True)
        move(str(npy_path), str(npy_dir))


if __name__ == "__main__":
    main()