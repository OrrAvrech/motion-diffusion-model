import numpy as np
from pathlib import Path


def mean_variance(data_dir: Path, save_dir: Path, joints_num: int):
    file_list = list(data_dir.rglob("*.npy"))
    data_list = []

    for filepath in file_list:
        data = np.load(filepath)
        if np.isnan(data).any():
            print(filepath.name)
            continue
        data_list.append(data)

    data = np.concatenate(data_list, axis=0)
    print(data.shape)
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)
    Std[0:1] = Std[0:1].mean() / 1.0
    Std[1:3] = Std[1:3].mean() / 1.0
    Std[3:4] = Std[3:4].mean() / 1.0
    Std[4: 4+(joints_num - 1) * 3] = Std[4: 4+(joints_num - 1) * 3].mean() / 1.0
    Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9] = Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9].mean() / 1.0
    Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3] = Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3].mean() / 1.0
    Std[4 + (joints_num - 1) * 9 + joints_num * 3: ] = Std[4 + (joints_num - 1) * 9 + joints_num * 3: ].mean() / 1.0

    assert 8 + (joints_num - 1) * 9 + joints_num * 3 == Std.shape[-1]

    np.save(save_dir / 'Mean.npy', Mean)
    np.save(save_dir / 'Std.npy', Std)

    return Mean, Std


if __name__ == "__main__":
    data_dir = Path("/proj/vondrick2/orr/motion-diffusion-model/dataset/MOYO/human_feedback/vecs_12_all")
    save_dir = Path("/proj/vondrick2/orr/motion-diffusion-model/dataset/MOYO/human_feedback/")
    joints_num = 22
    mean_variance(data_dir, save_dir, joints_num)