import torch.nn as nn
from model.regressor import M2MRegressor
from model.diffusion import GaussianDiffusion
from model.rotation2xyz import Rotation2xyz


class M2MDM(nn.Module):
    def __init__(self, njoints, nfeats, diffusion_steps, noise_schedule, 
                 latent_dim=256, ff_size=1024, num_layers=8, 
                 num_heads=4, dropout=0.1, activation="gelu", 
                 data_rep='rot6d', dataset='amass', arch='trans_enc',
                 lambda_rcxyz=0., lambda_vel=0., lambda_fc=0., **kargs):
        super().__init__()

        model = M2MRegressor(njoints=njoints,
                             nfeats=nfeats,
                             latent_dim=latent_dim,
                             ff_size=ff_size,
                             num_layers=num_layers,
                             num_heads=num_heads,
                             dropout=dropout,
                             activation=activation,
                             data_rep=data_rep,
                             dataset=dataset,
                             arch=arch,
                             **kargs)
        diffusion = GaussianDiffusion(model=model,
                                      diffusion_steps=diffusion_steps,
                                      noise_schedule=noise_schedule,
                                      data_rep=data_rep,
                                      lambda_rcxyz=lambda_rcxyz,
                                      lambda_vel=lambda_vel,
                                      lambda_fc=lambda_fc)
        smpl = Rotation2xyz(device='cpu', dataset=dataset)

        self.model = model
        self.diffusion = diffusion
        self.smpl = smpl

    def train(self):
        self.diffusion.train()

    def eval(self):
        self.diffusion.eval()
