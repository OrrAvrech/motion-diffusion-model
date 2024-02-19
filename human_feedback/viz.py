import os
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from textwrap import wrap
from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain


def save_vid_list(saved_files: List[Path], save_path: Path):
    ffmpeg_rep_files = [f' -i {str(f)} ' for f in saved_files]
    hstack_args = f' -filter_complex hstack=inputs={len(saved_files)}'
    ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {str(save_path)}'
    os.system(ffmpeg_rep_cmd)



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