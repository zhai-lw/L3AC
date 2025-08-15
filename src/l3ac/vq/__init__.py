from typing import Any
import torch
from torch import nn


class VQEmbed(nn.Module):
    def __init__(self, vq, feature_dim, codebook_dim):
        super().__init__()
        self.vq = vq
        if feature_dim == codebook_dim:
            self.project_in = nn.Identity()
            self.project_out = nn.Identity()
        else:
            self.project_in = nn.Linear(feature_dim, codebook_dim)
            self.project_out = nn.Linear(codebook_dim, feature_dim)

    def to_indices(self, features):
        raise NotImplementedError

    def to_features(self, indices):
        # latents = self.vq.get_output_from_indices(indices)
        latents = self.vq.indices_to_codes(indices)
        return self.project_out(latents)

    def forward(self, x):
        latents = self.project_in(x)
        q_latents, indices, *ret = self.vq(latents)
        q_features = self.project_out(q_latents)
        vq_loss = ret[0].sum() if len(ret) > 0 else torch.Tensor([0.]).to(dtype=x.dtype, device=x.device)
        return q_features, indices, vq_loss


def build_vq(name="vq", feature_dim: int = 256, **vq_args):
    """
    return a VQ module which takes (B, T, feature_dim) input and output (B, T)or (B, T, codebook_num)
    """
    match name:
        case "super_fsq":
            codebook_num = vq_args.get("codebook_num", 1)
            assert codebook_num == 1
            from .fsq import SuperFSQ
            levels = vq_args.get("levels")
            noise_rate = vq_args.get("noise_rate", 0.5)
            codebook_dim = len(levels)
            vq = SuperFSQ(levels=levels, noise_rate=noise_rate)
        case _:
            raise ValueError(f"Unknown vq name: {name}")

    return VQEmbed(vq, feature_dim, codebook_dim)
