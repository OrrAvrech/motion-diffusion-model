import random
from pathlib import Path


def main():
    dataset_dir = Path("/proj/vondrick2/orr/motion-diffusion-model/dataset/MotionX/human_feedback/vecs_12_enc50_ft/hml_vec")
    train_split_file = Path("/proj/vondrick2/orr/motion-diffusion-model/dataset/MotionX/human_feedback/vecs_12_train.txt")
    val_split_file = Path("/proj/vondrick2/orr/motion-diffusion-model/dataset/MotionX/human_feedback/vecs_12_val.txt")
    val_split = 0.2
    dirs_list = [d for d in dataset_dir.rglob("*") if d.is_dir() and "subset" in str(d)]
    random.shuffle(dirs_list)

    split_index = int(len(dirs_list) * (1-val_split))
    train_data = dirs_list[:split_index]
    val_data = dirs_list[split_index:]

    for npy_path in dataset_dir.rglob("*.npy"):
        filestem = npy_path.stem
        # TODO: remove after
        split_str = filestem.rsplit("_", 1)
        gt_filestem = split_str[0]
        num = int(split_str[1])
        if num > 0:
            continue
        filename = str(npy_path.relative_to(dataset_dir).parent)
        if npy_path.parent in train_data:
            with open(train_split_file, "a") as fp:
                fp.write(f"{filename}\n")
        elif npy_path.parent in val_data:
            with open(val_split_file, "a") as fp:
                fp.write(f"{filename}\n")
        else:
            print(str(npy_path))




if __name__ == "__main__":
    main()