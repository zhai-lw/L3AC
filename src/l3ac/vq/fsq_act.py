import math

import torch

Sqrt2 = math.sqrt(2)


def build_act(act_name: str):
    """
    Args:
        act_name:
    Returns:
        act_func: (-∞, +∞) -> [0, 1]
        inv_act_func: (0, 1) -> (-∞, +∞)
    """
    match act_name:
        case "cdf":
            act_func, inv_act_func = cdf_act, cdf_inv_act
        case "tanh":
            act_func, inv_act_func = tanh_act, tanh_inv_act
        case "sigmoid":
            act_func, inv_act_func = sigmoid_act, sigmoid_inv_act
        case _:
            raise NotImplementedError(f"act_func({act_name}) has not been implemented yet.")
    return act_func, inv_act_func


def cdf_act(z: torch.Tensor) -> torch.Tensor:
    # (-∞, +∞) -> [0, 1]
    return (1 + torch.erf(z / Sqrt2)) / 2


def cdf_inv_act(act_z: torch.Tensor) -> torch.Tensor:
    # (0, 1) -> (-∞, +∞)
    return torch.erfinv(2 * act_z - 1) * Sqrt2


def tanh_act(z: torch.Tensor) -> torch.Tensor:
    return (torch.tanh(z) + 1) / 2


def tanh_inv_act(z: torch.Tensor) -> torch.Tensor:
    return torch.arctanh(z * 2 - 1)


def sigmoid_act(z: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(z)


def sigmoid_inv_act(z: torch.Tensor) -> torch.Tensor:
    return torch.logit(z)


def act_test():
    x = torch.randn(64, 100)
    for act_name in ('cdf', 'tanh', 'sigmoid'):
        act_func, inv_act_func = build_act(act_name)

        y = act_func(x)
        x_hat = inv_act_func(y)

        print(f"{act_name} error: {(x_hat - x).sum()}")


if __name__ == '__main__':
    act_test()
