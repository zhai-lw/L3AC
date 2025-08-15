import torch
import torch.nn.functional as F

import tools.loss

from .base import LossWrapper, NetworkLoss

from .audio import ElementWise, MultiStft, MultiMel, PerceptualLoss


def loss_builder(name: str, sample_rate: int, asr_weight_path) -> torch.nn.Module:
    match name.lower():
        case 'nll_loss':
            loss_nn = LossWrapper(F.nll_loss)
        case 'cross_entropy':
            loss_nn = LossWrapper(torch.nn.CrossEntropyLoss())
        case 'element_wise':
            loss_nn = ElementWise(ignore_small_values=False, mse_or_l1='l1')
        case 'multi_stft':
            loss_nn = MultiStft()
        case 'multi_mel':
            loss_nn = MultiMel(sample_rate=sample_rate)
        case 'perception':
            loss_nn = PerceptualLoss(sample_rate=sample_rate, weight_path=asr_weight_path)
        case loss_name if loss_name.startswith('network_'):
            loss_nn = NetworkLoss(loss_name.removeprefix('network_'))
        case _:
            raise NotImplementedError(f"{name} not implemented")

    return loss_nn


class Losses(torch.nn.Module):
    def __init__(self, sample_rate: int, asr_weight_path, loss_weights: dict):
        super().__init__()
        self.loss_nns = torch.nn.ModuleDict()
        self.loss_weights = {}
        for loss_name, loss_weight in loss_weights.items():
            self.loss_nns[loss_name] = loss_builder(loss_name, sample_rate=sample_rate, asr_weight_path=asr_weight_path)
            self.loss_weights[loss_name] = tools.loss.weight.builder(loss_weight)

    def forward(self, nn_output, ref_input) -> (torch.Tensor, dict[str, torch.Tensor]):
        loss_dict = {name: loss_nn(nn_output, ref_input) for name, loss_nn in self.loss_nns.items()}
        weighted_sum = sum(loss_dict[name] * weight for name, weight in self.loss_weights.items())
        return weighted_sum, loss_dict
