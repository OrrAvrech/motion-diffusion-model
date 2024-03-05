import numpy as np

HML_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
]

NUM_HML_JOINTS = len(HML_JOINT_NAMES)  # 22 SMPLH body joints

HML_LOWER_BODY_JOINTS = [HML_JOINT_NAMES.index(name) for name in ['pelvis', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_foot', 'right_foot',]]
SMPL_UPPER_BODY_JOINTS = [i for i in range(len(HML_JOINT_NAMES)) if i not in HML_LOWER_BODY_JOINTS]


# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
HML_ROOT_BINARY = np.array([True] + [False] * (NUM_HML_JOINTS-1))
HML_ROOT_MASK = np.concatenate(([True]*(1+2+1),
                                HML_ROOT_BINARY[1:].repeat(3),
                                HML_ROOT_BINARY[1:].repeat(6),
                                HML_ROOT_BINARY.repeat(3),
                                [False] * 4))
HML_LOWER_BODY_JOINTS_BINARY = np.array([i in HML_LOWER_BODY_JOINTS for i in range(NUM_HML_JOINTS)])
HML_LOWER_BODY_MASK = np.concatenate(([True]*(1+2+1),
                                     HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(3),
                                     HML_LOWER_BODY_JOINTS_BINARY[1:].repeat(6),
                                     HML_LOWER_BODY_JOINTS_BINARY.repeat(3),
                                     [True]*4))
HML_UPPER_BODY_MASK = ~HML_LOWER_BODY_MASK

# Human-Feedback
HML_LOWER_BACK_JOINTS = [HML_JOINT_NAMES.index(name) for name in ['pelvis', 'spine1', 'spine2', 'spine3', 'left_hip', 'right_hip']]
HML_UPPER_BACK_JOINTS = [HML_JOINT_NAMES.index(name) for name in ['spine1', 'spine2', 'spine3', 'neck', 'right_collar', 'left_collar', 'right_shoulder', 'left_shoulder']]
HML_KNEE_JOINTS = [HML_JOINT_NAMES.index(name) for name in ['right_knee', 'left_knee']]
HML_HIP_JOINTS = [HML_JOINT_NAMES.index(name) for name in ['right_hip', 'left_hip', 'pelvis']]
HML_FEET_JOINTS = [HML_JOINT_NAMES.index(name) for name in ['right_foot', 'left_foot', 'right_ankle', 'left_ankle']]

HML_LOWER_BACK_JOINTS_BINARY = np.array([i in HML_LOWER_BACK_JOINTS for i in range(NUM_HML_JOINTS)])
HML_UPPER_BACK_JOINTS_BINARY = np.array([i in HML_UPPER_BACK_JOINTS for i in range(NUM_HML_JOINTS)])
HML_KNEE_JOINTS_BINARY = np.array([i in HML_KNEE_JOINTS for i in range(NUM_HML_JOINTS)])
HML_HIP_JOINTS_BINARY = np.array([i in HML_HIP_JOINTS for i in range(NUM_HML_JOINTS)])
HML_FEET_JOINTS_BINARY = np.array([i in HML_FEET_JOINTS for i in range(NUM_HML_JOINTS)])

HML_LOWER_BACK_MASK = ~np.concatenate(([True]*(1+2+1),
                                     HML_LOWER_BACK_JOINTS_BINARY[1:].repeat(3),
                                     HML_LOWER_BACK_JOINTS_BINARY[1:].repeat(6),
                                     HML_LOWER_BACK_JOINTS_BINARY.repeat(3),
                                     [True]*4))
HML_UPPER_BACK_MASK = ~np.concatenate(([True]*(1+2+1),
                                     HML_UPPER_BACK_JOINTS_BINARY[1:].repeat(3),
                                     HML_UPPER_BACK_JOINTS_BINARY[1:].repeat(6),
                                     HML_UPPER_BACK_JOINTS_BINARY.repeat(3),
                                     [True]*4))
HML_KNEE_MASK = ~np.concatenate(([True]*(1+2+1),
                                HML_KNEE_JOINTS_BINARY[1:].repeat(3),
                                HML_KNEE_JOINTS_BINARY[1:].repeat(6),
                                HML_KNEE_JOINTS_BINARY.repeat(3),
                                [True]*4))
HML_HIP_MASK = ~np.concatenate(([True]*(1+2+1),
                                HML_HIP_JOINTS_BINARY[1:].repeat(3),
                                HML_HIP_JOINTS_BINARY[1:].repeat(6),
                                HML_HIP_JOINTS_BINARY.repeat(3),
                                [True]*4))
HML_FEET_MASK = ~np.concatenate(([True]*(1+2+1),
                                HML_FEET_JOINTS_BINARY[1:].repeat(3),
                                HML_FEET_JOINTS_BINARY[1:].repeat(6),
                                HML_FEET_JOINTS_BINARY.repeat(3),
                                [True]*4))



