from pathlib import Path


def main():
    input_files_path = Path("/proj/vondrick2/orr/motion-diffusion-model/dataset/MotionX/vecs_12.txt")
    saved_files_dir = Path("/proj/vondrick2/orr/motion-diffusion-model/dataset/MotionX/human_feedback/vecs_12/hml_vec")
    with open(input_files_path, "r") as fp:
        files_list = fp.readlines()
    files_list = [line.strip() for line in files_list]
    
    existing_dirnames = list(set(x.relative_to(saved_files_dir).parent for x in saved_files_dir.rglob("*.npy")))
    existing_filenames = []
    for dirname in existing_dirnames:
        pert_dir = saved_files_dir / dirname
        num_files = len(list(pert_dir.glob("*.npy")))
        if num_files == 4:
            existing_filenames.append(str(dirname))

    remaining_filenames = []
    for filename in files_list:
        if filename.split(".npy")[0] in existing_filenames:
            continue
        remaining_filenames.append(filename)
    
    with open(input_files_path.parent / f"{input_files_path.stem}_remaining.txt", "w") as fp:
        [fp.write(f"{x}\n") for x in remaining_filenames]



if __name__ == "__main__":
    main()