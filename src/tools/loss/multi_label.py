# refs: [ZLPR: A Novel Loss for Multi-label Classification](https://arxiv.org/abs/2208.02955)

import torch


def cross_entropy(y_true, y_pred):
    """
    Args:
        y_true: (batch_size, ..., num_classes) one-hot
        y_pred: (batch_size, ..., num_classes) (-inf, inf) (the prediction is the index where y_pred[index]>0)
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return neg_loss + pos_loss


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        if self.reduction == 'mean':
            return cross_entropy(y_true, y_pred).mean()
        elif self.reduction == 'sum':
            return cross_entropy(y_true, y_pred).sum()
        elif self.reduction == 'none':
            return cross_entropy(y_true, y_pred)
