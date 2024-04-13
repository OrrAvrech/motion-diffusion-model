from pathlib import Path


def main():
    split_file = Path("/proj/vondrick2/orr/motion-diffusion-model/dataset/MOYO/train.txt")
    filenames_file = Path("/proj/vondrick2/orr/motion-diffusion-model/dataset/MOYO/vecs_12.txt")
    output_file = Path("/proj/vondrick2/orr/motion-diffusion-model/dataset/MOYO/vecs_12_train.txt")

    with open(split_file, "r") as f:
        prefixes = f.read().splitlines()

    with open(filenames_file, "r") as f:
        filenames = f.read().splitlines()
        filenames = [x.split(".npy")[0] for x in filenames]

    output_files = []
    for filename in filenames:
        for p in prefixes:
            if p in filename:
                output_files.append(filename)

    for name in output_files:
        with open(output_file, 'a') as f:
            f.write(f"{name}\n")


if __name__ == "__main__":
    main()