import subprocess
import typer
from pathlib import Path
from typing import Optional
from functools import partial


app = typer.Typer()


def run_instance(timeline_path: Path, 
                 output_dir: Path, 
                 num_samples: int,
                 program_py: str, 
                 model_path: Path,  
                 dataset: str,
                 split_file: str,
                 device: int):
    command = f"python {program_py} --model_path {model_path} --timeline_path {timeline_path} --num_samples {num_samples} --device {device} --output_dir {output_dir} --dataset {dataset} --split_file {split_file}"
    process = subprocess.Popen(command, shell=True)


@app.command()
def edit_long(timeline_path: Path, 
              output_dir: Path,
              split_files_dir: Path,
              device: Optional[int] = None, 
              num_samples: Optional[int] = 1000,
              program_py: Optional[str] = "/proj/vondrick2/orr/motion-diffusion-model/sample/edit_long.py", 
              model_path: Optional[Path] = Path("./save/humanml_trans_enc_512/model000200000.pt"),  
              dataset: Optional[str] = "humanfeedback",
              num_gpus: Optional[int] = 8):
    '''
    "--model_path",
    "./save/humanml_trans_enc_512/model000200000.pt",
    "--timeline_path",
    "./dataset/MOYO/human_feedback/perturb_t2m",
    "--num_samples",
    "2",
    "--dataset",
    "moyo",
    "--device",
    "1",
    "--output_dir",
    "./dataset/MOYO",
    "--split_file",
    "output_file"
    '''
    func_to_pool = partial(run_instance, 
                           timeline_path, 
                           output_dir,
                           num_samples,
                           program_py,
                           model_path,
                           dataset)

    for i, split_file in enumerate(split_files_dir.glob("*.txt")):
        device = int(i % num_gpus)
        print(f"process {i}: run {split_file.name} on device {device}")
        func_to_pool(split_file.stem, device)


if __name__ == "__main__":
    app()