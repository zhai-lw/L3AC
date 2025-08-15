import functools

import torch

import utils


def scale_to(value: torch.Tensor, scale: float):
    return value / value.detach().abs() * scale


@functools.cache
def builder(weight):
    if not isinstance(weight, str):
        return weight

    match weight.split('-'):
        case ['scale', scale_value]:
            min_scale = max_scale = float(scale_value)
        case ['clamp', 'min', min_value]:
            min_scale, max_scale = float(min_value), None
        case ['clamp', 'max', max_value]:
            min_scale, max_scale = None, float(max_value)
        case ['clamp', 'min', min_value, 'max', max_value]:
            min_scale, max_scale = float(min_value), float(max_value)
        case _:
            raise ValueError(f'Can not recognize {weight}')
    return AutoScaleWeight(min_scale=min_scale, max_scale=max_scale)


class AutoScaleWeight:
    def __init__(self, min_scale=None, max_scale=None):
        self.min = utils.default(min_scale, utils.number.MIN)
        self.max = utils.default(max_scale, utils.number.MAX)

    def __mul__(self, other):
        return other * self

    def __rmul__(self, value):
        if value >= self.max:
            return scale_to(value, self.max)
        elif value <= self.min:
            return scale_to(value, self.min)
        else:
            return value
