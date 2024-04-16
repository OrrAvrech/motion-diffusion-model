import typer
import torch
import joblib
import pickle
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Tuple
from utils.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle
from human_feedback.smpl import get_smpl_model, get_smpl_output
from human_feedback.motion_process import process_file, recover_from_ric
import numpy as np

app = typer.Typer()

NUM_JOINTS_SMPL = 23
NUM_BETAS = 10
NUM_JOINTS_HUMANML = 22


def load_4d_humans(
    pkl_path: Path,
    smpl_dir: Path,
    num_joints: int,
    max_frames: int,
    transform: bool = False,
):
    data4d = joblib.load(pkl_path)
    predictions = list(data4d.values())

    smpl_output, smpl_model = get_smpl_output(smpl_dir, predictions, transform)
    positions = smpl_output.joints[:, :num_joints, :]
    parents = smpl_model.parents[:num_joints]
    faces = smpl_model.faces

    positions_np = positions.detach().cpu().numpy()
    positions_np = positions_np[:max_frames, ...]
    vertices_np = smpl_output.vertices.detach().cpu().numpy()
    vertices_np = vertices_np[:max_frames, ...]

    return positions_np, vertices_np, parents, faces


def compute_canonical_transform(global_orient):
    rotation_matrix = torch.tensor([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ], dtype=global_orient.dtype)
    global_orient_matrix = axis_angle_to_matrix(global_orient)
    global_orient_matrix = torch.matmul(rotation_matrix, global_orient_matrix)
    global_orient = matrix_to_axis_angle(global_orient_matrix)
    return global_orient


def transform_translation(trans):
    trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    trans = np.dot(trans, trans_matrix)  # exchange the y and z axis
    trans[:, 2] = trans[:, 2] * (-1)
    return trans


@app.command()
def moyo(
    dataset_dir: Path,
    joints_dir: Optional[Path] = None,
    humanml_dir: Optional[Path] = None,
    smpl_dir: Optional[Path] = None
) -> Tuple[list, list, list]:
    positions_list, data_list, rec_ric_data_list = [], [], []
    for pkl_path in tqdm(dataset_dir.rglob("*.pkl")):
        try:
            with open(pkl_path, "rb") as fp:
                moyo_smpl = pickle.load(fp)
        except pickle.UnpicklingError:
            print(f"Error loading {pkl_path}, skipping...")
            continue
        filename_npy = f"{pkl_path.stem}.npy"


        global_orient = compute_canonical_transform(torch.Tensor(moyo_smpl["global_orient"]))
        smpl_params = dict()
        # smpl_params["global_orient"] = axis_angle_to_matrix(
        #     torch.Tensor(moyo_smpl["global_orient"])
        # ).unsqueeze(1)
        smpl_params["global_orient"] = axis_angle_to_matrix(global_orient).unsqueeze(1)
        smpl_params["body_pose"] = axis_angle_to_matrix(
            torch.Tensor(moyo_smpl["body_pose"]).reshape(-1, NUM_JOINTS_SMPL, 3)
        )
        smpl_params["betas"] = torch.Tensor(moyo_smpl["betas"])[:, :NUM_BETAS]

        transl = transform_translation(moyo_smpl["transl"])
        # smpl_params["transl"] = torch.Tensor(moyo_smpl["transl"])
        smpl_params["transl"] = torch.Tensor(transl)

        smpl_model = get_smpl_model(smpl_dir)
        smpl_output = smpl_model(**smpl_params, pose2rot=False)
        positions = smpl_output.joints.detach().cpu().numpy()
        positions = positions[:, :NUM_JOINTS_HUMANML, :]

        # from top-view to side-view
        # as in amass_data to pose_data conversion
        # https://github.com/EricGuo5513/HumanML3D/blob/main/raw_pose_processing.ipynb
        # trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        # positions = np.dot(positions, trans_matrix)[:100] #TODO
        positions = positions[:100]

        # save joints (positions)
        if joints_dir is not None:
            joints_dir.mkdir(exist_ok=True, parents=True)
            np.save(joints_dir / filename_npy, positions)

        # save new_joint_vec (HumanML full rep.)
        data, _, _, _ = process_file(positions, feet_thre=0.002)
        # save new_joints (HumanML rep. joint positions only)
        rec_ric_data = recover_from_ric(
            torch.from_numpy(data).unsqueeze(0).float(), NUM_JOINTS_HUMANML
        )

        if humanml_dir is not None:
            new_joints_dir = humanml_dir / "new_joints"
            new_joint_vec_dir = humanml_dir / "new_joints_vec"
            new_joints_dir.mkdir(exist_ok=True, parents=True)
            new_joint_vec_dir.mkdir(exist_ok=True)
            np.save(new_joints_dir / filename_npy, rec_ric_data.squeeze().numpy())
            np.save(new_joint_vec_dir / filename_npy, data)

        positions_list.append(positions)
        data_list.append(data)
        rec_ric_data_list.append(rec_ric_data)

    return positions_list, data_list, rec_ric_data_list


@app.command()
def moyo_filenames_to_text(dataset_dir: Path, text_dir: Path):
    text_dir.mkdir(exist_ok=True, parents=True)
    for pkl_path in dataset_dir.rglob("*.pkl"):
        filename = pkl_path.stem
        action_name = filename.split("03596_")[-1].split("-")[0]
        action_name = action_name.replace("_", " ").replace("(", "").replace(")", "").rstrip()
        action_splits = action_name.split(" or ")
        text = f"a person does "
        tokens = "a/DET person/NOUN does/VERB "
        for i, action in enumerate(action_splits):
            if i == 0:
                text += action.replace(" ", "-")
                tokens += f"{action.replace(' ', '-')}/NOUN"
            else:
                text += f" or {action.replace(' ', '-')}"
                tokens += f" or/CCONJ {action.rstrip().replace(' ', '-')}/NOUN"

        text_with_tokens = f"{text}.#{tokens}#0.0#0.0"
        txt_path = text_dir / f"{filename}.txt"
        with open(txt_path, "w") as f:
            f.write(text_with_tokens)


@app.command()
def humans4d(
    dataset_dir: Path,
    joints_dir: Optional[Path] = None,
    humanml_dir: Optional[Path] = None,
    smpl_dir: Optional[Path] = None,
    max_frames: Optional[int] = 196,
) -> Tuple[list, list, list]:
    positions_list, data_list, rec_ric_data_list = [], [], []
    for pkl_path in tqdm(dataset_dir.rglob("*.pkl")):
        positions, _, _, _ = load_4d_humans(
            pkl_path, smpl_dir, NUM_JOINTS_HUMANML, max_frames, transform=None
        )
        filename_npy = f"{pkl_path.stem}.npy"

        # save joints (positions)
        if joints_dir is not None:
            joints_dir.mkdir(exist_ok=True, parents=True)
            np.save(joints_dir / filename_npy, positions)

        # save new_joint_vec (HumanML full rep.)
        data, _, _, _ = process_file(positions, 0.002)
        # save new_joints (HumanML rep. joint positions only)
        rec_ric_data = recover_from_ric(
            torch.from_numpy(data).unsqueeze(0).float(), NUM_JOINTS_HUMANML
        )

        if humanml_dir is not None:
            new_joints_dir = humanml_dir / "new_joints"
            new_joint_vec_dir = humanml_dir / "new_joints_vec"
            new_joints_dir.mkdir(exist_ok=True, parents=True)
            new_joint_vec_dir.mkdir(exist_ok=True)
            np.save(new_joints_dir / filename_npy, rec_ric_data.squeeze().numpy())
            np.save(new_joint_vec_dir / filename_npy, data)

        positions_list.append(positions)
        data_list.append(data)
        rec_ric_data_list.append(rec_ric_data)

    return positions_list, data_list, rec_ric_data_list


@app.command()
def motionx(dataset_dir: Path,
            joints_dir: Optional[Path] = None,
            humanml_dir: Optional[Path] = None,
            smpl_dir: Optional[Path] = None,
            hands_detect: Optional[bool] = False):
    
    positions_list, data_list, rec_ric_data_list = [], [], []
    for npy_path in tqdm(dataset_dir.rglob("*.npy")):
        try:
            with open(npy_path, "rb") as fp:
                motion_smplx = np.load(fp)
        except pickle.UnpicklingError:
            print(f"Error loading {npy_path}, skipping...")
            continue
        filename_npy = npy_path.name

        motion = torch.tensor(motion_smplx).float()
        motion_params = {
                    'root_orient': motion[:, :3],  # controls the global root orientation
                    'pose_body': motion[:, 3:3+63],  # controls the body
                    'pose_hand': motion[:, 66:66+90],  # controls the finger articulation
                    'pose_jaw': motion[:, 66+90:66+93],  # controls the yaw pose
                    'face_expr': motion[:, 159:159+50],  # controls the face expression
                    'face_shape': motion[:, 209:209+100],  # controls the face shape
                    'trans': motion[:, 309:309+3],  # controls the global body position
                    'betas': motion[:, 312:],  # controls the body shape. Body shape is static
                }

        smpl_params = dict()
        smpl_params["global_orient"] = axis_angle_to_matrix(
            torch.Tensor(motion_params["root_orient"])
        ).unsqueeze(1)
        smpl_params["body_pose"] = axis_angle_to_matrix(
            torch.Tensor(motion_params["pose_body"]).reshape(-1, NUM_JOINTS_SMPL, 3)
        )
        smpl_params["betas"] = torch.Tensor(motion["betas"])[:, :NUM_BETAS]
        smpl_params["transl"] = torch.Tensor(motion["trans"])

        smpl_model = get_smpl_model(smpl_dir)
        smpl_output = smpl_model(**smpl_params, pose2rot=False)
        positions = smpl_output.joints.detach().cpu().numpy()
        positions = positions[:, :NUM_JOINTS_HUMANML, :]

        # from top-view to side-view
        # as in amass_data to pose_data conversion
        # https://github.com/EricGuo5513/HumanML3D/blob/main/raw_pose_processing.ipynb
        # trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        # positions = np.dot(positions, trans_matrix)

        # save joint positions
        if joints_dir is not None:
            joints_dir.mkdir(exist_ok=True, parents=True)
            np.save(joints_dir / filename_npy, positions)

        # save new_joint_vec (HumanML vector rep.)
        data, _, _, _ = process_file(positions, feet_thre=0.002, hands_detect=hands_detect)
        # save new_joints (HumanML rep. joint positions only)
        rec_ric_data = recover_from_ric(
            torch.from_numpy(data).unsqueeze(0).float(), NUM_JOINTS_HUMANML
        )

        if humanml_dir is not None:
            new_joints_dir = humanml_dir / "new_joints"
            new_joint_vec_dir = humanml_dir / "new_joints_vec"
            new_joints_dir.mkdir(exist_ok=True, parents=True)
            new_joint_vec_dir.mkdir(exist_ok=True)
            np.save(new_joints_dir / filename_npy, rec_ric_data.squeeze().numpy())
            np.save(new_joint_vec_dir / filename_npy, data)

        positions_list.append(positions)
        data_list.append(data)
        rec_ric_data_list.append(rec_ric_data)

    return positions_list, data_list, rec_ric_data_list


if __name__ == "__main__":
    app()
