import os
import torch
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from textwrap import wrap
from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain
from human_feedback.motion_utils import align_by_root


def save_vid_list(saved_files: List[Path], save_path: Path):
    ffmpeg_rep_files = [f' -i {str(f)} ' for f in saved_files]
    hstack_args = f' -filter_complex hstack=inputs={len(saved_files)}'
    ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {str(save_path)}'
    os.system(ffmpeg_rep_cmd)


def align_data(data, scale):
    data *= scale
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    frame_number = data.shape[0]
    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]
    return data, frame_number, trajec, MINS, MAXS


def plot_xzPlane(minx, maxx, miny, minz, maxz):
    verts = [
        [minx, miny, minz],
        [minx, miny, maxz],
        [maxx, miny, maxz],
        [maxx, miny, minz],
    ]
    xz_plane = Poly3DCollection([verts])
    xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
    return xz_plane


def plot_3d_motion(
    save_path,
    joints,
    title,
    kinematic_tree=t2m_kinematic_chain,
    figsize=(3, 3),
    fps=120,
    radius=3,
    vis_mode="default",
    pert_frames=None,
):
    # matplotlib.use("Agg")

    title = "\n".join(wrap(title, 20))
    pert_frames = [] if pert_frames is None else pert_frames

    def init():
        # ax.set_xlim3d([-radius / 2, radius / 2])
        # ax.set_ylim3d([0, radius])
        # ax.set_zlim3d([-radius / 3.0, radius * 2 / 3.0])
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    data = joints.copy().reshape(len(joints), -1, 3)

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    # ax = fig.add_subplot(111, projection="3d")
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = [
        "#DD5A37",
        "#D69E00",
        "#B75A39",
        "#FF6D00",
        "#DDB50E",
    ]  # Generation color
    colors = colors_orange
    if vis_mode == "upper_body":  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == "gt":
        colors = colors_blue

    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        # ax.view_init(elev=-90, azim=-90)
        ax.dist = 7.5
        xz_plane = plot_xzPlane(
            MINS[0] - trajec[index, 0],
            MAXS[0] - trajec[index, 0],
            0,
            MINS[2] - trajec[index, 1],
            MAXS[2] - trajec[index, 1],
        )
        ax.add_collection3d(xz_plane)

        used_colors = colors_orange if index in pert_frames else colors_blue
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(
                data[index, chain, 0],
                data[index, chain, 1],
                data[index, chain, 2],
                linewidth=linewidth,
                color=color,
            )

        plt.axis("off")
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_zticklabels([])

    ani = FuncAnimation(
        fig, update, frames=frame_number, interval=1000 / fps, repeat=False
    )

    ani.save(save_path, fps=fps)
    plt.close()


def plot_3d_motion_pairs(save_path,
                         joints1,
                         joints2,
                         title,
                         color1="blue",
                         color2="red",
                         kinematic_tree=t2m_kinematic_chain,
                         figsize=(5, 5),
                         fps=120,
                         scale=1.0,
                         radius=3):
    title = "\n".join(wrap(title, 20))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    data1 = joints1.copy().reshape(len(joints1), -1, 3)
    data2 = joints2.copy().reshape(len(joints2), -1, 3)

    data1, frame_number1, trajec, MINS, MAXS = align_data(data=data1 ,scale=scale)
    data2, frame_number2, _, _, _ = align_data(data=data2, scale=scale)
    frame_number = min(frame_number1, frame_number2)
    data1 = data1[:frame_number, ...]
    data2 = data2[:frame_number, ...]

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    # ax = fig.add_subplot(111, projection="3d")
    init()
    colors1 = [color1 for _ in range(5)]
    colors2 = [color2 for _ in range(5)]

    def update(index):
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        # ax.view_init(elev=-90, azim=-90)
        ax.dist = 7.5
        xz_plane = plot_xzPlane(
            MINS[0] - trajec[index, 0],
            MAXS[0] - trajec[index, 0],
            0,
            MINS[2] - trajec[index, 1],
            MAXS[2] - trajec[index, 1],
        )
        ax.add_collection3d(xz_plane)

        for i, (chain, color) in enumerate(zip(kinematic_tree, colors2)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(
                data2[index, chain, 0],
                data2[index, chain, 1],
                data2[index, chain, 2],
                linewidth=linewidth,
                color=color,
            )


        for i, (chain, color) in enumerate(zip(kinematic_tree, colors1)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(
                data1[index, chain, 0],
                data1[index, chain, 1],
                data1[index, chain, 2],
                linewidth=linewidth,
                color=color,
            )

        plt.axis("off")

    ani = FuncAnimation(
        fig, update, frames=frame_number, interval=1000 / fps, repeat=False
    )

    ani.save(save_path, fps=fps)
    plt.close()


def save_viz(gt: torch.Tensor, perturbated: torch.Tensor, pred: torch.Tensor, 
             save_dir: Path, idx: str, fps: float = 20) -> Path:
    save_dir.mkdir(exist_ok=True, parents=True)
    gt_save_path = save_dir / f"gt_{idx}.mp4"
    gt_np = gt.detach().cpu().numpy()
    plot_3d_motion(gt_save_path, gt_np, title="GT", fps=fps)

    pert_save_path = save_dir / f"pert_{idx}.mp4"
    pert_np = perturbated.detach().cpu().numpy()
    plot_3d_motion(pert_save_path, pert_np, title="Perturbed", fps=fps)

    pred_save_path = save_dir / f"pred_{idx}.mp4"
    pred_np = pred.detach().cpu().numpy()
    plot_3d_motion(pred_save_path, pred_np, title="Pred", fps=fps)

    saved_files = [gt_save_path, pert_save_path, pred_save_path]
    save_path = save_dir / f"{idx}.mp4"
    save_vid_list(saved_files=saved_files, save_path=save_path)
    [os.remove(filepath) for filepath in saved_files]
    return save_path


def save_overlays_viz(gt: torch.Tensor, perturbed: torch.Tensor, pred: torch.Tensor,
                      save_dir: Path, idx: str, fps: float = 20) -> Path:
    save_dir.mkdir(exist_ok=True, parents=True)

    gt = align_by_root(gt)
    perturbed = align_by_root(perturbed)
    pred = align_by_root(pred)

    gt_np = gt.detach().cpu().numpy()
    pert_np = perturbed.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()

    gt_pert_save_path = save_dir / f"gt_pert_{idx}.mp4"
    # pert_pred_save_path = save_dir / f"pert_pred_{idx}.mp4"
    gt_pred_save_path = save_dir / f"gt_pred_{idx}.mp4"

    plot_3d_motion_pairs(gt_pert_save_path, gt_np, pert_np, title="GT vs. Perturbed", fps=fps)
    # plot_3d_motion_pairs(pert_pred_save_path, pert_np, pred_np, title="Perturbed vs. Pred", fps=fps,
    #                      color1="red", color2="green")
    plot_3d_motion_pairs(gt_pred_save_path, gt_np, pred_np, title="GT vs. Pred", fps=fps,
                         color2="green")

    saved_files = [gt_pert_save_path, gt_pred_save_path]
    save_path = save_dir / f"{idx}.mp4"
    save_vid_list(saved_files=saved_files, save_path=save_path)
    [os.remove(filepath) for filepath in saved_files]
    return save_path