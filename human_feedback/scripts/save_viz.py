import os
import json
import typer
import torch
from pathlib import Path
from typing import Optional
from human_feedback.viz import plot_3d_motion, plot_3d_motion_pairs, save_vid_list
from human_feedback.motion_process import recover_from_ric
import numpy as np

app = typer.Typer()


@app.command()
def folder(motion_dir: Path, output_dir: Path, text_dir: Optional[Path] = None, fps: Optional[int] = 40):
    # motion_dir -> npy paths of HumanML3D vecs
    for motion_path in motion_dir.rglob("*.npy"):
        motion_np = np.load(motion_path)
        joints = recover_from_ric(torch.Tensor(motion_np))
        joints_np = joints.detach().cpu().numpy()

        relative_motion_path = motion_path.relative_to(motion_dir).parent
        content = motion_path.stem
        if text_dir is not None:
            with open(text_dir / relative_motion_path / f"{motion_path.stem}.txt") as fp:
                content = fp.read()

        save_dir = output_dir / relative_motion_path
        save_dir.mkdir(exist_ok=True, parents=True)
        njoints_save_path = save_dir / f"{motion_path.stem}.mp4"
        plot_3d_motion(njoints_save_path, joints_np, title=content, fps=fps)
        print(f"saved new joints viz {njoints_save_path}")


@app.command()
def conversion(
    joints_dir: Path, new_joints_dir: Path, output_dir: Path, fps: Optional[int] = 40
):
    output_dir.mkdir(exist_ok=True)
    for npy_path in joints_dir.glob("*.npy"):
        npy_name = npy_path.name
        new_joints_path = new_joints_dir / npy_name

        joints_np = np.load(npy_path)
        new_joints_np = np.load(new_joints_path)

        joints_save_path = output_dir / f"joints_{npy_path.stem}.mp4"
        plot_3d_motion(joints_save_path, joints_np, title="SMPL", fps=fps)
        print(f"saved joints viz {joints_save_path}")

        njoints_save_path = output_dir / f"new_joints_{npy_path.stem}.mp4"
        plot_3d_motion(njoints_save_path, new_joints_np, title="HumanML3D", fps=fps)
        print(f"saved new joints viz {njoints_save_path}")

        saved_files = [joints_save_path, njoints_save_path]
        save_path = output_dir / f"{npy_path.stem}.mp4"
        save_vid_list(saved_files=saved_files, save_path=save_path)
        print(f"save conversion visualization {save_path}")
        [os.remove(filepath) for filepath in saved_files]


@app.command()
def perturbations(
    input_dir: Path, pert_dir: Path, output_dir: Path, fps: Optional[int] = 40
):
    output_dir.mkdir(exist_ok=True)
    for npy_path in input_dir.glob("*.npy"):
        input_motion = np.load(npy_path)

        saved_files = []
        input_save_path = output_dir / f"input_{npy_path.stem}.mp4"
        plot_3d_motion(input_save_path, input_motion, title="input-motion", fps=fps)
        print(f"saved joints viz {input_save_path}")
        saved_files.append(input_save_path)

        pert_sample_dir = pert_dir / npy_path.stem
        for i, pert_npy in enumerate(pert_sample_dir.glob("*.npy")):
            pert_save_path = output_dir / f"{pert_npy.stem}.mp4"
            pert_motion_vec = torch.Tensor(np.load(pert_npy))
            pert_joints = recover_from_ric(pert_motion_vec)
            pert_joints_np = pert_joints.detach().cpu().numpy()
            plot_3d_motion(pert_save_path, pert_joints_np, title=f"iter-{i}", fps=fps)
            saved_files.append(pert_save_path)

        save_path = output_dir / f"{npy_path.stem}.mp4"
        save_vid_list(saved_files=saved_files, save_path=save_path)
        print(f"save perturbations visualization {save_path}")
        [os.remove(filepath) for filepath in saved_files]


@app.command()
def data_overlay(input_dir: Path, pert_dir: Path, instructions_dir: Path, output_dir: Path,
                 fps: Optional[int] = 40, scale: Optional[float] = 1.0):
    output_dir.mkdir(exist_ok=True)
    for npy_path in input_dir.glob("*.npy"):
        input_motion = torch.Tensor(np.load(npy_path))
        input_joints = recover_from_ric(input_motion)
        input_joints_np = input_joints.detach().cpu().numpy()

        with open(instructions_dir / f"{npy_path.stem}.json") as fp:
            instructions = json.load(fp)

        saved_files = []
        pert_sample_dir = pert_dir / npy_path.stem
        for i, pert_npy in enumerate(pert_sample_dir.glob("*.npy")):
            pert_save_path = output_dir / f"{pert_npy.stem}.mp4"
            pert_motion_vec = torch.Tensor(np.load(pert_npy))
            pert_joints = recover_from_ric(pert_motion_vec)
            pert_joints_np = pert_joints.detach().cpu().numpy()
            
            text = instructions[i]["text"]
            body_part = instructions[i]["body_part"].lower().replace(" ", "_")
            caption = f"Edit [{body_part}]: {text}"
            plot_3d_motion_pairs(pert_save_path, input_joints_np, pert_joints_np, 
                                 title=caption, fps=fps, scale=scale)
            saved_files.append(pert_save_path)

        save_path = output_dir / f"{npy_path.stem}.mp4"
        save_vid_list(saved_files=saved_files, save_path=save_path)
        [os.remove(filepath) for filepath in saved_files]


if __name__ == "__main__":
    app()
