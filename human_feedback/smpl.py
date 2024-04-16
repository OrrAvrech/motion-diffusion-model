import numpy as np
from smplx import SMPL
import torch
import pickle
from typing import Optional, List
import smplx
from smplx.lbs import vertices2joints
from smplx.utils import SMPLOutput
from dataclasses import asdict, dataclass
from pathlib import Path

from utils.rotation_conversions import euler_angles_to_matrix, matrix_to_euler_angles


@dataclass
class SMPLConfig:
    gender: str
    mean_params: Path
    model_path: Path
    num_body_joints: int
    joint_regressor_extra: Optional[Path] = None


class SMPL4DHumans(smplx.SMPLLayer):
    def __init__(
        self,
        *args,
        joint_regressor_extra: Optional[str] = None,
        update_hips: bool = False,
        **kwargs
    ):
        """
        Extension of the official SMPL implementation to support more joints.
        Args:
            Same as SMPLLayer.
            joint_regressor_extra (str): Path to extra joint regressor.
        """
        super().__init__(*args, **kwargs)
        smpl_to_openpose = [
            24,
            12,
            17,
            19,
            21,
            16,
            18,
            20,
            0,
            2,
            5,
            8,
            1,
            4,
            7,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
        ]

        if joint_regressor_extra is not None:
            self.register_buffer(
                "joint_regressor_extra",
                torch.tensor(
                    pickle.load(open(joint_regressor_extra, "rb"), encoding="latin1"),
                    dtype=torch.float32,
                ),
            )
        self.register_buffer(
            "joint_map", torch.tensor(smpl_to_openpose, dtype=torch.long)
        )
        self.update_hips = update_hips

    def forward(self, *args, **kwargs) -> SMPLOutput:
        """
        Run forward pass. Same as SMPL and also append an extra set of joints if joint_regressor_extra is specified.
        """
        smpl_output = super().forward(*args, **kwargs)
        joints = smpl_output.joints[:, self.joint_map, :]
        if self.update_hips:
            joints[:, [9, 12]] = (
                joints[:, [9, 12]]
                + 0.25 * (joints[:, [9, 12]] - joints[:, [12, 9]])
                + 0.5
                * (joints[:, [8]] - 0.5 * (joints[:, [9, 12]] + joints[:, [12, 9]]))
            )
        if hasattr(self, "joint_regressor_extra"):
            extra_joints = vertices2joints(
                self.joint_regressor_extra, smpl_output.vertices
            )
            joints = torch.cat([joints, extra_joints], dim=1)
        smpl_output.joints = joints
        return smpl_output


def get_smpl_model(smpl_dir: Optional[Path] = None, num_joints: int = 23):
    if smpl_dir is None:
        smpl_dir = Path(__file__).resolve().parent / "data"
        
    smpl_cfg = SMPLConfig(
        gender="neutral",
        # joint_regressor_extra=smpl_dir / "SMPL_to_J19.pkl",
        joint_regressor_extra=None,
        mean_params=smpl_dir / "smpl_mean_params.npz",
        model_path=smpl_dir / "smpl",
        num_body_joints=num_joints,
    )
    smpl_model = SMPL(**asdict(smpl_cfg))
    return smpl_model


def get_smpl_output(smpl_dir: Path, predictions: List[dict], transform: bool = False):
    smpl_model = get_smpl_model(smpl_dir)

    pred_smpl_params = [pred["smpl"][0] for pred in predictions]
    global_orient = torch.Tensor(
        np.array(list(map(lambda x: x["global_orient"], pred_smpl_params)))
    )
    global_orient = torch.Tensor(
        np.array(list(map(lambda x: x["global_orient"], pred_smpl_params)))
    )

    if transform:
        global_orient_euler = matrix_to_euler_angles(
            global_orient.squeeze(1), convention="XYZ"
        )
        global_orient_euler[:, 0] += np.pi
        global_orient = euler_angles_to_matrix(
            global_orient_euler, convention="XYZ"
        ).unsqueeze(1)

    body_pose = torch.Tensor(
        np.array(list(map(lambda x: x["body_pose"], pred_smpl_params)))
    )
    betas = torch.Tensor(np.array(list(map(lambda x: x["betas"], pred_smpl_params))))

    smpl_params = dict()
    smpl_params["global_orient"] = global_orient
    smpl_params["body_pose"] = body_pose
    smpl_params["betas"] = betas

    smpl_output = smpl_model(**smpl_params, pose2rot=False)
    return smpl_output, smpl_model
