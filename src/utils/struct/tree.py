from collections import defaultdict
from functools import reduce
from typing import Iterable


def tree(): return defaultdict(tree)


DEFAULT_VALUE_NAME = '__value__'


class Tree:
    def __init__(self, default=None):
        self.root = tree()
        self.default = default

    def _set(self, keys: Iterable, value: object):
        cursor = reduce((lambda d, k: d[k]), keys, self.root)
        cursor[DEFAULT_VALUE_NAME] = value

    def _get(self, keys: Iterable, default=None):
        cursor = reduce(lambda d, k: d[k], keys, self.root)
        if DEFAULT_VALUE_NAME in cursor:
            return cursor[DEFAULT_VALUE_NAME]
        elif default is not None:
            return default
        elif self.default is not None:
            return self.default
        else:
            raise KeyError(keys)

    def get(self, key, default=None):
        if isinstance(key, (list, tuple)):
            return self._get(key, default)
        else:
            return self._get([key], default)

    def __getitem__(self, key):
        if isinstance(key, (list, tuple)):
            return self._get(key)
        else:
            return self._get([key])

    def __setitem__(self, key, value):
        if isinstance(key, (list, tuple)):
            self._set(key, value)
        else:
            self._set([key], value)

    def items(self, deep=65535):
        yield from iter_tree(self.root, [], deep)


def iter_tree(cursor: defaultdict, cursor_index: list, max_deep=65535):
    max_deep -= 1
    for key, value in cursor.items():
        if key is DEFAULT_VALUE_NAME:
            continue
        keys = cursor_index + [key]
        if DEFAULT_VALUE_NAME in value:
            yield keys, value[DEFAULT_VALUE_NAME]
        if max_deep > 0:
            yield from iter_tree(value, keys, max_deep)
