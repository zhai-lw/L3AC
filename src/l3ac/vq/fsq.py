import math

import torch
from torch import nn

from .fsq_act import build_act


class SuperFSQ(nn.Module):
    def __init__(self, levels, special_edge: bool = True,
                 act_func: str = 'tanh', inv_act: bool = False, straight_through: bool = False,
                 noise_rate: float = 0., noise_filter: bool = False):
        super().__init__()
        self.levels = nn.Buffer(torch.tensor(levels, dtype=torch.int32), persistent=False)
        basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32)
        self.basis = nn.Buffer(basis, persistent=False)
        self.special_edge = special_edge

        self.act, self.inv_act = build_act(act_func)
        if not inv_act:
            self.inv_act = lambda x: (x * 2 - 1)  # * math.sqrt(3)  # ! to make q_z.var() -> 1.0
            assert straight_through is False
            self.straight_through = straight_through

        self.noise_rate = noise_rate

        self.codebook_dim = self.levels.numel()
        self.codebook_size = torch.prod(self.levels).item()

    def forward(self, z: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        noise_rate = self.noise_rate if self.training else 0.0
        ori_shape = z.shape
        assert ori_shape[-1] == self.codebook_dim
        z = z.reshape(-1, self.codebook_dim)

        act_z = self.act(z)
        q_act_z, level_indices = self.quantize_act_value(act_z)
        indices = self.level_indices_to_indices(level_indices)

        if noise_rate > 0.:
            noise_mask = torch.rand_like(act_z) <= noise_rate
            noises = (torch.rand_like(act_z) - 0.5) / self.levels
            q_act_z[noise_mask] = act_z[noise_mask] + noises[noise_mask]

        q_z = self.inv_act(q_act_z)

        if self.straight_through:
            q_z = set_grad(q_z, grad_from=z)

        return (q_z.reshape(ori_shape),
                {
                    "indices": indices.reshape(ori_shape[:-1]),
                    "level_indices": level_indices.reshape(ori_shape),
                })

    def quantize_act_value(self, act_z):
        # [0, 1] -> {0, 1, 2, ..., (l-1)}
        if self.special_edge:
            level_indices = (act_z * (self.levels - 1)).round()
            q_act_z = level_indices / (self.levels - 1)
        else:
            level_indices = (act_z * self.levels * 0.999).floor()
            q_act_z = (level_indices + 0.5) / self.levels
        q_act_z = set_grad(q_act_z, grad_from=act_z)
        return q_act_z, level_indices.detach()

    def level_indices_to_indices(self, level_indices):
        return (level_indices * self.basis).sum(dim=-1).to(torch.int32)

    def indices_to_level_indices(self, indices):
        return (indices.unsqueeze(-1) // self.basis) % self.levels

    def indices_to_act_value(self, indices: torch.Tensor):
        level_indices = self.indices_to_level_indices(indices)
        if self.special_edge:
            return level_indices / (self.levels - 1)
        else:
            return (level_indices + 0.5) / self.levels

    def indices_to_codes(self, indices: torch.Tensor):
        return self.inv_act(self.indices_to_act_value(indices))


def set_grad(value_from: torch.Tensor, grad_from: torch.Tensor):
    return grad_from + (value_from - grad_from).detach()
    # return grad_from * (value_from / (grad_from + 1e-5)).detach()
