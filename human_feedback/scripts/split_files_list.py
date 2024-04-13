from pathlib import Path


def split_files(input_file: Path, k: int, output_dir: Path):
    output_dir.mkdir(exist_ok=True, parents=True)
    # Read the list of files from the input text file
    with open(input_file, 'r') as f:
        files = f.read().splitlines()
    files = [name.split(".npy")[0] for name in files]

    # Calculate the number of files in each split
    num_files_per_split = len(files) // k
    remainder = len(files) % k

    # Split the list of files into K different parts
    split_files = [files[i * num_files_per_split:(i + 1) * num_files_per_split] for i in range(k)]
    for i in range(remainder):
        split_files[i % k].append(files[k * num_files_per_split + i])

    # Write each part into separate output files
    print(f"writing {k} files, with approx {num_files_per_split} files each")
    for i, file_list in enumerate(split_files):
        with open(output_dir / f'output_file_{i+1}.txt', 'w') as f:
            f.write('\n'.join(file_list))


def main():
    input_files_list = Path("/proj/vondrick2/orr/motion-diffusion-model/dataset/MotionX/vecs_12_remaining.txt")
    output_dir = Path("/proj/vondrick2/orr/motion-diffusion-model/dataset/MotionX/human_feedback/split_files")
    num_splits = 48
    split_files(input_files_list, num_splits, output_dir)



if __name__ == "__main__":
    main()