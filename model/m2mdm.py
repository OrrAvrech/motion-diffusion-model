import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict
from abc import ABC, abstractmethod
from model.mdm import InputProcess, CondProjection, OutputProcess, PositionalEncoding, TimestepEmbedder
from model.rotation2xyz import Rot2xyz


class BaseM2MDM(nn.Module, ABC):
    def __init__(self, njoints, nfeats, latent_dim=256, ff_size=1024, num_layers=8, 
                 num_heads=4, dropout=0.1, activation="gelu", data_rep='rot6d',
                 cond_dim=263, arch='trans_enc', emb_trans_dec=False):
        super().__init__()

        # data params
        self.njoints = njoints
        self.nfeats = nfeats
        self.data_rep = data_rep

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
        self.seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                               nhead=self.num_heads,
                                                               dim_feedforward=self.ff_size,
                                                               dropout=self.dropout,
                                                               activation=self.activation)

        self.cond_projection = CondProjection(self.data_rep, self.input_feats, self.latent_dim)
        self.cond_encoder = nn.TransformerEncoder(self.seqTransEncoderLayer, num_layers=2)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.embed_motion = nn.Linear(self.latent_dim, self.latent_dim)
        self.emb_trans_dec = emb_trans_dec

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)
      
        self.smpl = Rot2xyz()

    def mask_cond(self, cond: Tensor, cond_mask_prob: float):
        bs = cond.shape[1]
        if cond_mask_prob == 1.0:
            # force null cond
            return torch.zeros_like(cond)
        elif self.training and cond_mask_prob > 0.:
            # if cond_mask_prob = 1.0 -> bernoulli will draw 1s w.p 1 -> null cond
            # if cond_mask_prob = 0.0 -> bernoulli will draw 0s w.p 1 -> real cond (no guidance)
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    @abstractmethod
    def forward(self, x: Tensor, timesteps: Tensor, y: Dict, cond_mask_prob: float):
        pass

    def guided_forward(self, x: Tensor, timesteps: Tensor, y: Dict, guidance_weight: float):
        unc = self.forward(x=x, timesteps=timesteps, y=y, cond_mask_prob=1.0)
        conditioned = self.forward(x=x, timesteps=timesteps, y=y, cond_mask_prob=0.0)
        return unc + (conditioned - unc) * guidance_weight


class EncoderM2MDM(BaseM2MDM):
    def __init__(self, njoints, nfeats, latent_dim=256, ff_size=1024, num_layers=8, 
                 num_heads=4, dropout=0.1, activation="gelu", data_rep='rot6d', 
                 cond_dim=263, arch='trans_enc', emb_trans_dec=False):
        super().__init__(njoints, nfeats, latent_dim, ff_size, num_layers, 
                         num_heads, dropout, activation, data_rep, 
                         cond_dim, arch, emb_trans_dec)
        self.seqTransEncoder = nn.TransformerEncoder(self.seqTransEncoderLayer,
                                                     num_layers=self.num_layers)
        
    def forward(self, x: Tensor, timesteps: Tensor, y: Dict, cond_mask_prob: float):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        # timestemp MLP
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        # encode cond
        cond_embed = y["motion"]
        cond_tokens = self.cond_projection(cond_embed)
        cond_tokens = self.sequence_pos_encoder(cond_tokens)
        cond_tokens = self.cond_encoder(cond_tokens)

        # mask and project cond
        cond_mask = self.mask_cond(cond_tokens, cond_mask_prob)
        cond_mask_pooled = cond_mask.mean(dim=0).unsqueeze(0)
        emb += self.embed_motion(cond_mask_pooled)

        x = self.input_process(x)

        # adding the cond + t embed
        xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        # encoder-only
        output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output


class NullCondEncoderM2MDM(BaseM2MDM):
    def __init__(self, njoints, nfeats, seq_len, latent_dim=256, ff_size=1024, num_layers=8, 
                 num_heads=4, dropout=0.1, activation="gelu", data_rep='rot6d', 
                 cond_dim=263, arch='trans_enc', emb_trans_dec=False):
        super().__init__(njoints, nfeats, latent_dim, ff_size, num_layers, 
                         num_heads, dropout, activation, data_rep, 
                         cond_dim, arch, emb_trans_dec)
        self.seq_len = seq_len
        self.null_cond_embed = nn.Parameter(torch.randn(self.seq_len, 1, self.latent_dim))
        self.seqTransEncoder = nn.TransformerEncoder(self.seqTransEncoderLayer,
                                                     num_layers=self.num_layers)
        
    def mask_cond(self, cond: Tensor, cond_mask_prob: float):
        # return a keep-mask as follows: 1-> real-cond, 0-> null-cond, by applying:
        # cond_tokens = torch.where(keep_mask, cond_tokens, null_cond_embed)
        # cond_mask_prob == 1 means that the null-cond is always chosen (uncond)
        # cond_mask_prob == 0 means that the real-cond is always chosen (unguided)
        bs = cond.shape[1]
        if cond_mask_prob == 1:
            keep_mask =  torch.zeros(bs, dtype=torch.bool)
        elif cond_mask_prob == 0:
            keep_mask = torch.ones(bs, dtype=torch.bool)
        else:
            keep_mask = torch.zeros(bs).float().uniform_(0, 1) > cond_mask_prob
        keep_mask = keep_mask.unsqueeze(-1).unsqueeze(0)
        keep_mask = keep_mask.to(cond.device)
        return keep_mask    
    
    def forward(self, x: Tensor, timesteps: Tensor, y: Dict, cond_mask_prob: float):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        # timestemp MLP
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        # encode cond
        cond_embed = y["motion"]
        cond_tokens = self.cond_projection(cond_embed)
        cond_tokens = self.sequence_pos_encoder(cond_tokens)
        cond_tokens = self.cond_encoder(cond_tokens)

        # mask and project cond
        null_cond_embed = self.null_cond_embed.to(cond_tokens.dtype)
        keep_mask = self.mask_cond(cond_tokens, cond_mask_prob)
        cond_mask = torch.where(keep_mask, cond_tokens, null_cond_embed)
        cond_mask_pooled = cond_mask.mean(dim=0).unsqueeze(0)
        emb += self.embed_motion(cond_mask_pooled)

        x = self.input_process(x)

        # adding the cond + t embed
        xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        # encoder-only
        output = self.seqTransEncoder(xseq)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output

        
class DecoderM2MDM(BaseM2MDM):
    def __init__(self, njoints, nfeats, latent_dim=256, ff_size=1024, num_layers=8, 
                 num_heads=4, dropout=0.1, activation="gelu", data_rep='rot6d', 
                 cond_dim=263, arch='trans_enc', emb_trans_dec=False):
        super().__init__(njoints, nfeats, latent_dim, ff_size, num_layers, 
                         num_heads, dropout, activation, data_rep, 
                         cond_dim, arch, emb_trans_dec)
        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=activation)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=self.num_layers)
        
    def forward(self, x: Tensor, timesteps: Tensor, y: Dict, cond_mask_prob: float):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        # timestemp MLP
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        # encode cond
        cond_embed = y["motion"]
        cond_tokens = self.cond_projection(cond_embed)
        cond_tokens = self.sequence_pos_encoder(cond_tokens)
        cond_tokens = self.cond_encoder(cond_tokens)

        # mask and project cond
        cond_mask = self.mask_cond(cond_tokens, cond_mask_prob)
        cond_mask_pooled = cond_mask.mean(dim=0).unsqueeze(0)
        emb += self.embed_motion(cond_mask_pooled)

        x = self.input_process(x)

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

        
