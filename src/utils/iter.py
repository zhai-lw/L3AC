from itertools import *
from typing import Iterable, Protocol, Iterator
import more_itertools
import numpy


class SizedIterable(Protocol):
    def __len__(self) -> int:
        pass

    def __iter__(self) -> Iterator:
        pass


# count down enumerate
def cd_enumerate(iterable: SizedIterable):
    return enumerate(iterable, start=-len(iterable) + 1)


class Rearranger:
    def __init__(self, iterable: Iterable, num: int, re_iter_every_time=True):
        self.iterable = iterable
        self.num = num
        self._iterator = None
        self.re_iter_every_time = re_iter_every_time

    @property
    def iterator(self):
        if self.re_iter_every_time:
            return chain.from_iterable(repeat(self.iterable))
        else:
            if self._iterator is None:
                self._iterator = chain.from_iterable(repeat(self.iterable))
            return self._iterator

    def __iter__(self):
        return islice(self.iterator, self.num)

    def __len__(self):
        return self.num


class UpSampler:
    def __init__(self, iterable: SizedIterable, num: int, fill_value=None):
        self.target_num = num
        self.iterable = iterable
        self.fill_value = fill_value
        assert self.target_num >= len(self.iterable)
        indicator = (numpy.linspace(0, len(iterable), self.target_num + 1)).astype(int)
        self.indicator = indicator[:-1] < indicator[1:]

    def __iter__(self):
        iterator = iter(self.iterable)
        for flag in self.indicator:
            if flag:
                yield next(iterator)
            else:
                yield self.fill_value

    def __len__(self):
        return self.target_num


class DownSampler:
    def __init__(self, iterable: SizedIterable, num: int):
        self.target_num = num
        self.iterable = iterable
        assert self.target_num <= len(self.iterable)
        indicator = (numpy.linspace(0, self.target_num, len(iterable) + 1)).astype(int)
        self.indicator = indicator[:-1] < indicator[1:]

    def __iter__(self):
        iterator = iter(self.iterable)
        for flag in self.indicator:
            item = next(iterator)
            if flag:
                yield item

    def __len__(self):
        return self.target_num


def resample(iterable: SizedIterable, num: int, fill_value=None):
    if num > len(iterable):
        return UpSampler(iterable, num, fill_value)
    else:
        return DownSampler(iterable, num)


# supported in python 3.12 [itertools.batched]
def batched(iterable, n):
    # batched('ABCDEFG', 3) → ABC DEF G
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


def consume(iterator: Iterator, num: int = None) -> int:
    if num is None:
        return sum(1 for _ in iterator)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, num, num), None)
        return num


class RoundRobin:
    def __init__(self, *iterables):
        self.iterables = iterables

    def __iter__(self):
        return more_itertools.roundrobin(*self.iterables)

    def __len__(self):
        return sum(len(it) for it in self.iterables)
