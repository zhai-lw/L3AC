from typing import Iterator

import torch
from pydantic import BaseModel, Field

import xtract


class NonScheduler(torch.optim.lr_scheduler.LRScheduler):
    def get_lr(self):
        return self.base_lrs


class OptimizerConfig(BaseModel):
    lr: float = 3e-4
    eps: float = xtract.nn.EPS

    scheduler_step_num: int

    optimizer_name: str = 'AdamW'
    optimizer_args: dict = Field(default_factory=dict)

    scheduler_name: str = 'OneCycleLR'
    scheduler_args: dict = Field(default_factory=dict)

    def build_optimizer(self, parameters) -> torch.optim.Optimizer:
        optimizer_cls = getattr(torch.optim, self.optimizer_name)
        optimizer = optimizer_cls(
            parameters, lr=self.lr, eps=self.eps,
            **self.optimizer_args
        )
        return optimizer

    def build_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        match self.scheduler_name:
            case 'None':
                scheduler = NonScheduler(optimizer)
            case "OneCycleLR":
                max_lr = self.scheduler_args.get('max_lr', 3e-3)
                min_lr = self.scheduler_args.get('min_lr', 1e-6)
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer=optimizer, total_steps=self.scheduler_step_num,
                    max_lr=max_lr,
                    div_factor=max_lr / self.lr,
                    final_div_factor=self.lr / min_lr,
                    pct_start=0.4,
                    three_phase=True,
                )
            case _:
                scheduler_cls = getattr(torch.optim.lr_scheduler, self.scheduler_name)
                scheduler = scheduler_cls(optimizer, **self.scheduler_args)
        return scheduler

    def build_all(self, parameters: Iterator[torch.nn.Parameter], ):
        optimizer = self.build_optimizer(parameters)
        scheduler = self.build_scheduler(optimizer)
        return optimizer, scheduler
