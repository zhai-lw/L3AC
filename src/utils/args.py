import argparse
from functools import reduce
from pathlib import Path
from typing import Any


class Parser(argparse.ArgumentParser):
    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)
        for key, value in args.__dict__.items():
            if isinstance(value, str) and '~' in value:
                args.__dict__[key] = value.replace('~', str(Path.home()))
        return args


class AutoSigner:
    def __call__(self, func_obj):
        import inspect
        args_name = inspect.getfullargspec(func_obj).args

        if args_name[0] == "self":
            def wrapper(obj, *args: dict[str: Any], **kwargs):
                return func_obj(obj, **self.get_kwargs(args_name[1:], kwargs, *args))
        else:
            def wrapper(*args: dict[str: Any], **kwargs):
                return func_obj(**self.get_kwargs(args_name, kwargs, *args))

        return wrapper

    def get_kwargs(self, args_name: list[str], *args_dict: dict[str: Any]) -> dict[str, Any]:
        args_dict = reduce(lambda a, b: a | b, args_dict)
        kwargs = {name: self.format_args(name, args_dict[name]) for name in args_name}
        return kwargs

    @staticmethod
    def format_args(arg_name, arg_value):
        return arg_value
