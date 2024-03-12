from typing import Optional
from data_loaders.humanml.scripts.motion_process import recover_from_ric


class RecoverInput:
    # recover joint positions or HumanML3D vec
    def __init__(self, data_rep: str, njoints: int):
        self.data_rep = data_rep
        self.njoints = njoints

    def __call__(self, sample):
        # vec -> [njoints, nfeats, nframes]
        if self.data_rep == "xyz":
            vec = recover_from_ric(sample, self.njoints).permute(1, 2, 0)
        else:
            vec = sample.unsqueeze(1).permute(2, 1, 0)
        return vec
    

class HML2XYZ:
    def __init__(self, data_rep: str, njoints: Optional[int] = 22):
        self.data_rep = data_rep
        self.njoints = njoints

    def __call__(self, sample):
        if self.data_rep == "xyz":
            xyz = sample
        else:
            hml = sample.squeeze(2).permute(0, 2, 1)
            xyz = recover_from_ric(hml, self.njoints)
        return xyz
 