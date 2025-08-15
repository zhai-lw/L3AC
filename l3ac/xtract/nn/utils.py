import torch
from torch import nn, Tensor


def get_torch_precision(data_precision: str) -> torch.dtype:
    match data_precision:
        case 'float32' | 'fp32':
            return torch.float32
        case 'bfloat16' | 'bf16':
            return torch.bfloat16
        case 'float16' | 'fp16':
            return torch.float16
        case 'float64' | 'fp64':
            return torch.float64
        # case 'float8' | 'fp8':
        #     return torch.float8
        case _:
            raise ValueError(f"Unsupported data precision: {data_precision}")


def get_eps(data_type: torch.dtype):
    match data_type:
        case torch.float32 | torch.float64:
            return 1e-8
        case torch.bfloat16:
            return 1e-7
        case torch.float16:
            return 1e-5
        case _:
            raise NotImplementedError(f"Unsupported data type: {data_type}")


EPS = 1e-8


class FuncLayer(nn.Module):
    def __init__(self, lambda_fun):
        super().__init__()
        self.lambda_func = lambda_fun

    def forward(self, x: Tensor):
        return self.lambda_func(x)


def seed_everything(seed: int):
    import random
    random.seed(seed)
    import numpy
    numpy.random.seed(seed)
    import torch
    torch.manual_seed(seed)


def t2n(tensor_data: torch.Tensor):
    return tensor_data.detach().cpu().numpy()


def get_lr(optimizer: torch.optim.Optimizer):
    return optimizer.param_groups[0]['lr']


def get_lrs(optimizer: torch.optim.Optimizer):
    return [param_group['lr'] for param_group in optimizer.param_groups]


class FreeCacheContext:
    def __enter__(self):
        torch.cuda.empty_cache()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()
