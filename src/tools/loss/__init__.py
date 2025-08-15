__all__ = ['focal', 'multi_label']

import torch
from torch import nn, Tensor

from . import weight


class Zero(nn.Module):
    @staticmethod
    def forward(*args):
        return torch.zeros_like(args[0]).sum()


class Ignore(nn.Module):
    def __init__(self, loss_func, *ignored_labels):
        super().__init__()
        self.loss_func = loss_func
        self.register_buffer('ignored_labels', torch.tensor(ignored_labels), persistent=False)

    def forward(self, predicts: Tensor, targets: Tensor):
        """
        Args:
            predicts: (batch_size, num_classes)
            targets: (batch_size)
        Returns:
            loss
        """
        mask = ~torch.isin(targets, self.ignored_labels)
        if mask.any():
            return self.loss_func(predicts[mask], targets[mask])
        else:
            return Zero.forward(predicts, targets)
