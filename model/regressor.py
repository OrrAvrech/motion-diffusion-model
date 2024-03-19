import torch
import torch.nn as nn
from model.mdm import InputProcess, OutputProcess, PositionalEncoding
from model.rotation2xyz import Rot2xyz


class M2MRegressor(nn.Module):
    def __init__(self, njoints, nfeats, latent_dim=256, ff_size=1024, num_layers=8, 
                 num_heads=4, dropout=0.1, activation="gelu", data_rep='rot6d', arch='trans_enc'):
        super().__init__()

        self.njoints = njoints
        self.nfeats = nfeats
        self.data_rep = data_rep

        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation
        self.input_feats = self.njoints * self.nfeats

        self.arch = arch
        self.input_process = InputProcess(self.data_rep, self.input_feats, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.num_layers)

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)

        self.rot2xyz = Rot2xyz()

    def forward(self, x):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        x = self.input_process(x)

        xseq = self.sequence_pos_encoder(x)
        output = self.seqTransEncoder(xseq) # [seqlen, bs, d]

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output

    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)
