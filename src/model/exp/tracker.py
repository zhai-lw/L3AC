from typing import Callable

import torch

from ..network import Network


def param_tracker_maker(network: Network) -> Callable[[], dict[str, torch.Tensor]]:
    # return lambda: {
    #     'fc_weight': network.fc[-1].weight.data.data[0, 0],
    #     'fc_bias': network.fc[-1].bias.data[0],
    # }
    return dict
