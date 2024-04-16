import typer
from pathlib import Path


def main(output_path: Path, files_dir: Path, ext: str):
    if output_path.exists():
        raise f"{output_path} already exists will be appended by this script"
    with open(output_path, "a") as file:
        for filepath in files_dir.rglob(f"*.{ext}"):
            filename = str(filepath.relative_to(files_dir).parent / f"{filepath.stem}.npy")
            file.write(filename + "\n")


if __name__ == "__main__":
    typer.run(main)