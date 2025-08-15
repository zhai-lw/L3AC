import contextlib
import logging
import pathlib
from collections import defaultdict

import torch

log = logging.getLogger("L3AC")


class Module(torch.nn.Module):
    def __init__(self, name: str = None, ):
        super().__init__()
        self.name = name or self.default_name()

    @classmethod
    def default_name(cls):
        return cls.__module__ + "." + cls.__name__

    @property
    def trainable_modules(self) -> dict[str, torch.nn.Module]:
        # return {"model": self}
        raise NotImplementedError

    @property
    def trainable_parameters(self):
        for module in self.trainable_modules.values():
            yield from module.parameters()

    def train(self, mode: bool = True):
        self.training = mode
        for module in self.trainable_modules.values():
            module.train(mode=mode)
        return self

    def save_model(self, model_dir=None, model_path=None):
        model_path = model_path or pathlib.Path(model_dir).joinpath(self.name)
        model_path.mkdir(exist_ok=True)
        log.info(f"Saving model to {model_path}")
        for name, module in self.trainable_modules.items():
            torch.save(module.state_dict(), model_path / f"{name}.pt")

    def load_model(self, model_dir=None, model_path=None):
        model_path = model_path or pathlib.Path(model_dir).joinpath(self.name)
        if model_path.exists():
            for name, module in self.trainable_modules.items():
                module_path = model_path / f"{name}.pt"
                if module_path.exists():
                    log.info(f"Loading module({name}) from ({module_path})")
                    module.load_state_dict(torch.load(module_path, map_location='cpu', weights_only=True))
                else:
                    log.info(f"Module({name})'s path: ({module_path}) does not exist.")
        else:
            log.warning(f"Model path ({model_path}) does not exist.")


@contextlib.contextmanager
def training(*networks: torch.nn.Module):
    for network in networks:
        network.train(mode=True)
    yield


@contextlib.contextmanager
def inferencing(*networks: torch.nn.Module):
    for network in networks:
        network.eval()
    with torch.inference_mode():  # instead of torch.no_grad()
        yield


def set_requires_grad(network: torch.nn.Module, requires_grad_map: dict):
    original_map = {}
    for name, param in network.named_parameters():
        original_map[name] = param.requires_grad
        param.requires_grad = requires_grad_map[name]
    return original_map


def freeze(network: torch.nn.Module, set_eval: bool = True):
    set_requires_grad(network, defaultdict(lambda: False))
    if set_eval:
        network = network.eval()
    return network


@contextlib.contextmanager
def without_autograd(*networks: torch.nn.Module):
    original_map = defaultdict(dict)
    for ni, network in enumerate(networks):
        original_map[ni] = set_requires_grad(network, defaultdict(lambda: False))
    yield
    for ni, network in enumerate(networks):
        set_requires_grad(network, original_map[ni])


def print_all_parameters(network: torch.nn.Module, network_name='model', out_func: callable = print):
    out_func(f"{network_name}: ")
    network_attrs = network.__dict__.copy()
    # print all modules
    module_dict = network_attrs.pop('_modules', {})
    for name, param in module_dict.items():
        print_all_parameters(param, network_name=f"Module-{name}", out_func=lambda s: out_func(f"\t{s}"))
    # print all parameters
    param_dict = network_attrs.pop('_parameters', {})
    for name, param in param_dict.items():
        out_func(f"\tParameter-{name} ({param.shape}) requires_grad={param.requires_grad}")
    # print all buffers
    buffer_dict = network_attrs.pop('_buffers', {})
    for name, param in buffer_dict.items():
        out_func(f"\tBuffer-{name} ({param.shape}) requires_grad={param.requires_grad}")
    # print other tensors
    for name, param in network_attrs.items():
        if isinstance(param, torch.Tensor):
            out_func(f"\tTensor-{name}: ({param.shape}) requires_grad={param.requires_grad}")
