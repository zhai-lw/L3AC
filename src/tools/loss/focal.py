from typing import Optional

import torch
from torch import Tensor
from torch import nn


class Loss(nn.Module):
    """ Focal Loss https://arxiv.org/abs/1708.02002.
    """

    def __init__(self,
                 gamma: float = 0.,
                 weight: Optional[Tensor] = None,
                 label_smoothing: float = 0.0,
                 reduction: str = 'mean'):
        """Constructor.

        Args:
            gamma (float, optional): A constant, as described in the paper. Defaults to 0. Recommend to set to 2.
        """
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        assert reduction in ('mean', 'sum', 'none'), 'invalid reduction'
        self.reduction = reduction

        self.ce_loss = nn.CrossEntropyLoss(weight=self.weight, reduction='none', label_smoothing=label_smoothing)

    def __repr__(self):
        return f"{type(self).__name__}" \
               f"(weight={self.weight}, " \
               f"gamma={self.gamma}, " \
               f"label_smoothing={self.label_smoothing}, " \
               f"reduction={self.reduction})"

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        ce_loss = self.ce_loss(x, target)
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
