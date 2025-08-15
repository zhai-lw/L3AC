import math
import random
from typing import Iterable

import numpy
import numpy as np
from scipy.interpolate import interp1d

from .reduce_dim import DimReducer
from .welford import WelFord


def normalization(data: numpy.ndarray):
    return (data - data.min()) / (data.max() - data.min())


def smooth_data(data, smooth_level=(2, 5)):
    rounds, size = smooth_level
    d = data
    s_d = d.copy()
    for i in range(rounds):
        for j in range(1, size):
            d += np.concatenate((s_d[j:], np.full(j, s_d[-j])))
        d /= size
        s_d = d.copy()
    return d


def re_sampling(array, sam_num, way='interp', unwrap=False):
    if unwrap:
        array = np.unwrap(array)
    if way == "interp":
        return interp1d(np.arange(len(array)), array)(
            np.arange(0, len(array), len(array) / sam_num))
    elif (way == "avg_ds") | (way == "mid_ds"):
        if way == "mid_ds":
            def way2get_value(x):
                return np.median(x)
        else:
            def way2get_value(x):
                return np.mean(x)
        if sam_num > len(array):
            raise ValueError("sam_num > len(array)")
        avg_num = len(array) // sam_num
        remainder = len(array) % sam_num
        result = np.empty(sam_num)
        for i in range(sam_num):
            if remainder > 0:
                result[i] = way2get_value(array[(i * avg_num):((i + 1) * avg_num + 1)])
                remainder -= 1
            else:
                result[i] = way2get_value(array[i * avg_num:(i + 1) * avg_num])
            i += 1
        return result
    elif way == 'nearest':
        sample_index = np.around(np.linspace(0, len(array) - 1, sam_num, endpoint=True), decimals=0)
        return array[sample_index.astype(int)]


def group_by(array: np.ndarray, indexes):
    if not isinstance(indexes, Iterable):
        indexes = [indexes]
    results = [array]
    for index in indexes:
        results = [r for sr in results for r in group_by_(sr, index)]
    return results


def group_by_(array: np.ndarray, index):
    array = array[(array[:, index]).argsort()]
    return np.split(array, np.cumsum(np.unique(array[:, index], return_counts=True)[1])[:-1])


def isvalid(d):
    return not ((d is None) or (isinstance(d, float) and math.isnan(d)))


def pad_or_trim(array_list: list[np.ndarray], target_length: int, random_start=False):
    target_list = []
    for array in array_list:
        if len(array) > target_length:
            start_index = random.randint(0, len(array) - target_length) if random_start else 0
            target_list.append(array[start_index:start_index + target_length])
        else:
            target_list.append(np.pad(array, (0, target_length - len(array))))
    return target_list


def pad_list_array(array_list: list[np.ndarray], target_length=None):
    target_length = target_length or max(len(array) for array in array_list)
    pad_list = [np.pad(array, (0, target_length - len(array))) for array in array_list]
    return pad_list


def reduce_dim(*x: numpy.ndarray, way="ICA", dim=2, **kwargs):
    return DimReducer(dim, way, **kwargs).fit_transform(*x)
