import re
from collections import defaultdict
from typing import Any, Iterable, Mapping

# requirements: pip install scipy more-itertools
from . import args, context, data, file, iter, log, module, number, output, struct
# requirements: pip install scipy seaborn
from . import plot


def remove_special_char(s, mode='ascii'):
    if mode == 'ascii':
        return s.encode("ascii", "ignore").decode()
    elif mode == 'abc+n':
        return re.sub('[^A-Za-z0-9 ]+', '', s)
    else:
        raise ValueError(f"mode {mode} not supported")


def none_equal(a, b):
    return a != a and b != b or a == b


def default(var, default_value):
    return var if var is not None else default_value


def list_dict_to_dict_list(list_dict: list[dict]):
    dict_list = defaultdict(list)
    for dict_obj in list_dict:
        for key, value in dict_obj.items():
            dict_list[key].append(value)
    return dict_list


def collect_iter_by_key(iterable: Iterable[tuple[Any, Iterable]]) -> dict[Any, list]:
    dict_list = defaultdict(list)
    for key, li in iterable:
        dict_list[key].extend(li)
    return dict_list


def flatten(dictionary: Mapping, key_formatter='{parent_key}.{key}', parent_key=None):
    items = []
    for key, value in dictionary.items():
        new_key = key_formatter.format(parent_key=parent_key, key=key) if (parent_key is not None) else key
        if isinstance(value, Mapping):
            items.extend(flatten(value, key_formatter=key_formatter, parent_key=new_key))
        else:
            items.append((new_key, value))
    return items


def flatten_dict(dictionary: Mapping, nest_key=True) -> dict:
    if nest_key:
        flatten_items = flatten(dictionary, key_formatter='{parent_key}.{key}')
        return dict(flatten_items)
    else:
        flatten_items = flatten(dictionary, key_formatter='{key}')
        flatten_dictionary = dict(flatten_items)
        assert len(flatten_dictionary) == len(flatten_items)
        return flatten_dictionary
