import torch
import torch.nn as nn
from model.regressor import M2MRegressor
from model.mdm import InputProcess, OutputProcess, PositionalEncoding, TimestepEmbedder, MotionEmbedder
from model.diffusion import GaussianDiffusion
from model.rotation2xyz import Rotation2xyz


class M2MDM(nn.Module):
    def __init__(self, njoints, nfeats, latent_dim=256, ff_size=1024, num_layers=8, 
                 num_heads=4, dropout=0.1, activation="gelu", data_rep='rot6d', dataset='amass', 
                 cond_dim=263, arch='trans_enc', emb_trans_dec=False):
        super().__init__()

        # data params
        self.njoints = njoints
        self.nfeats = nfeats
        self.data_rep = data_rep
        self.dataset = dataset

        # input dimensions
        self.cond_dim = cond_dim
        self.input_feats = self.njoints * self.nfeats

        # hidden dimensions
        self.latent_dim = latent_dim

        # transformer params
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

        # arch name
        self.arch = arch

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.input_process = InputProcess(self.data_rep, self.input_feats, self.latent_dim)

        self.cond_projection = nn.Linear(cond_dim, latent_dim)
        self.cond_encoder = nn.Sequential()
        for _ in range(2):
            self.cond_encoder.append(
                nn.TransformerEncoderLayer(
                    d_model=self.latent_dim,
                    nhead=self.num_heads,
                    dim_feedforward=self.ff_size,
                    dropout=self.dropout,
                    activation=self.activation,
                )
            )

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.embed_motion = MotionEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.emb_trans_dec = emb_trans_dec

        if self.arch == 'trans_enc':
            print("TRANS_ENC init")
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)

            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers)
        elif self.arch == 'trans_dec':
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=activation)
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                            num_layers=self.num_layers)
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)
      
        self.smpl = Rotation2xyz(device='cpu', dataset=dataset)

    def mask_cond(self, cond, cond_mask_prob=0.0, force_mask=False):
        bs, _ = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond


    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, cond_embed: torch.Tensor, cond_mask_prob: float = 0.0):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        # timestemp MLP
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        # encode cond
        cond_tokens = self.cond_projection(cond_embed)
        cond_tokens = self.sequence_pos_encoder(cond_tokens)
        cond_tokens = self.cond_encoder(cond_tokens)

        # mask and project cond
        emb += self.embed_motion(self.mask_cond(cond_tokens, cond_mask_prob))

        x = self.input_process(x)

        if self.arch == 'trans_enc':
            # adding the cond + t embed
            xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        elif self.arch == 'trans_dec':
            if self.emb_trans_dec:
                xseq = torch.cat((emb, x), axis=0)
            else:
                xseq = x
            xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
            if self.emb_trans_dec:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)[1:] # [seqlen, bs, d] # FIXME - maybe add a causal mask
            else:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output

