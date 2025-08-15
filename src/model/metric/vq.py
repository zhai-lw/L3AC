import torch

from utils.args import AutoSigner
from xtract import ddp
from . import base


class CodebookRecoder:
    def __init__(self, size: int, device='cpu'):
        self.size = size
        self.record: torch.Tensor = torch.zeros(self.size, device=device)

    def reset(self):
        self.record.zero_()

    def update(self, indices: torch.Tensor):
        idx, counts = indices.unique(return_counts=True)
        self.record.scatter_add_(0, idx.to(dtype=torch.int64), counts.float())

    @property
    def usage_probs(self):
        return (self.record > 0).float().mean()

    @property
    def entropy(self):
        avg_probs = self.record / self.record.sum()
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        return entropy

    @property
    def perplexity(self):
        return torch.exp(self.entropy)

    @property
    def valid_probs(self):
        return self.perplexity / self.size

    def __repr__(self):
        return (f"CodebookRecoder("
                f"size={self.size}, "
                f"usage_probs={self.usage_probs.item():.4f}, "
                f"valid_probs={self.valid_probs.item():.4f}, "
                f")")


class CodebookUsage(base.BaseMetric):
    def __init__(self, codebook_size: int, cuda_device):
        super().__init__()
        self.metric = CodebookRecoder(codebook_size, device=cuda_device)

    def reset(self):
        self.metric.reset()

    @AutoSigner()
    def update(self, indices: torch.Tensor):
        self.metric.update(indices)

    @ddp.tensor_reducer(reduction='sum')
    def compute_internal_results(self) -> torch.Tensor:
        return self.metric.record

    @classmethod
    def get_results(cls, record: torch.Tensor):
        usage_probs = (record > 0).float().mean()
        avg_probs = record / record.sum()
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        perplexity = torch.exp(entropy)
        valid_probs = perplexity / record.numel()
        return {
            "usage_probs": usage_probs.item(),
            "valid_probs": valid_probs.item(),
        }
