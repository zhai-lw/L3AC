import contextlib
from functools import cached_property

import torch
from pydantic import field_validator, ValidationInfo, Field, computed_field
from pydantic_settings import SettingsConfigDict

import utils.iter
from xtract.config import FileConfig
from xtract import nn as xnn

from .acc_runtime import RS, ACC
from . import mlogging, tracker

from model.data import DataLoaderBuilder
from model.loss import Losses
from model.optimizer import OptimizerConfig
from model.metric import Metrics
from model.network import NetworkConfig, Network
from model.network.discriminator import Discriminator


class Config(FileConfig):
    sample_rate: int = 16000

    model_config = SettingsConfigDict(env_prefix='ST_', extra='forbid')

    steps_per_epoch_num: int = 100
    train_epoch_num: int = 40
    target_batch_size: int = 192
    train_dis_every_n_batches: int = 10

    grad_max_norm: float = 10
    dis_grad_max_norm: float = 10

    train_data: DataLoaderBuilder
    eval_data: DataLoaderBuilder
    test_data: DataLoaderBuilder

    nn_config: NetworkConfig
    dis_nn_config: dict = dict(fft_size=(126, 542, 1418, 2296)),

    opt_config: OptimizerConfig
    dis_opt_config: OptimizerConfig

    loss_config: dict = dict(loss_weights={})

    metric_config: dict = Field(default_factory=dict)

    @computed_field
    @cached_property
    def gradient_accumulation_steps(self) -> int:
        assert self.target_batch_size % (self.train_data.batch_size * RS.gpu_num) == 0
        return self.target_batch_size // (self.train_data.batch_size * RS.gpu_num)

    @field_validator('train_data', 'eval_data', 'test_data', mode='before')
    @classmethod
    def complete_data_config(cls, data_config: dict, info: ValidationInfo):
        return data_config | dict(
            sample_rate=info.data['sample_rate'],
            num_workers=RS.cpu_num,
        )

    @field_validator('opt_config', mode='before')
    @classmethod
    def complete_opt_config(cls, opt_config: dict, info: ValidationInfo):
        return opt_config | dict(
            scheduler_step_num=info.data['train_epoch_num'] * info.data['steps_per_epoch_num'],
        )

    @field_validator('dis_opt_config', mode='before')
    @classmethod
    def complete_dis_opt_config(cls, dis_opt_config: dict, info: ValidationInfo):
        opt_config = info.data['opt_config'].model_dump(exclude_unset=True)
        return opt_config | dis_opt_config

    @field_validator('metric_config', mode='before')
    @classmethod
    def complete_metric_config(cls, metric_config: dict):
        return metric_config | dict(
            cpu_num=RS.cpu_num,
            cuda_device=ACC.device,
        )

    @field_validator('dis_nn_config', 'loss_config', 'metric_config', mode='before')
    @classmethod
    def complete_other_config(cls, other_config: dict, info: ValidationInfo):
        return other_config | dict(
            sample_rate=info.data['sample_rate'],
        )


class Model:
    def __init__(self, config: Config):
        self.mc = config

        self.train_loader = ACC.prepare(config.train_data.get_dataloader(prefetch_size=self.mc.target_batch_size))
        self.eval_loader = ACC.prepare(config.eval_data.get_dataloader())
        self.test_loader = ACC.prepare(config.test_data.get_dataloader())

        network = Network(config.nn_config)
        optimizer, scheduler = config.opt_config.build_all(network.trainable_parameters)
        self.network, self.optimizer, self.scheduler = ACC.prepare(
            network, optimizer, scheduler
        )

        dis_nn = Discriminator(**config.dis_nn_config)
        dis_optimizer, dis_scheduler = config.dis_opt_config.build_all(dis_nn.trainable_parameters)
        self.dis_nn, self.dis_optimizer, self.dis_scheduler = ACC.prepare(
            dis_nn, dis_optimizer, dis_scheduler
        )

        self.loss_nn = ACC.prepare(Losses(**config.loss_config))

        self.metric = Metrics(network, **config.metric_config)

        self.param_tracker = tracker.param_tracker_maker(network)

    def estimate_progress(self) -> (int, int):
        assert self.scheduler.scheduler.last_epoch % self.mc.steps_per_epoch_num == 0, "There is a PROBLEM!"
        total_epochs = self.mc.train_epoch_num
        current_epoch = self.scheduler.scheduler.last_epoch // self.mc.steps_per_epoch_num
        return current_epoch, total_epochs

    @property
    def train_progress(self):
        del self.train_loader
        self.train_loader = ACC.prepare(self.mc.train_data.get_dataloader(prefetch_size=self.mc.target_batch_size))
        return mlogging.progress_bar(self.train_loader,
                                     flag_frequency=self.mc.steps_per_epoch_num,
                                     desc="training")

    @contextlib.contextmanager
    def grad_sync_context(self, grad_sync=True):
        if grad_sync:
            context = contextlib.nullcontext()
        else:
            context = utils.context.nested_context(
                lambda: ACC.no_sync(self.network),
                lambda: ACC.no_sync(self.dis_nn),
                lambda: ACC.no_sync(self.loss_nn),  # !
            )
        with context:
            yield

    @staticmethod
    def step_scheduler(scheduler: torch.optim.lr_scheduler.LRScheduler) -> float:
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        return current_lr

    def step_train_discriminator(self, accumulated_batch: list[dict], accumulate_num: int,
                                 recorder: mlogging.MainProcessLogger):
        # update discriminator
        step_flags = []
        self.network.eval()
        self.dis_nn.train()
        self.dis_optimizer.zero_grad(set_to_none=True)
        for cd, (batch_input, step_flag) in utils.iter.cd_enumerate(accumulated_batch):
            with self.grad_sync_context(cd == 0):
                with torch.no_grad():
                    nn_output = self.network(batch_input['audio'])
                weighted_sum, loss_dict = self.dis_nn(fake=nn_output['generated_audio'],
                                                      real=batch_input['audio'],
                                                      get_dis_loss=True)
                ACC.backward(weighted_sum / accumulate_num)
                param_norm = ACC.clip_grad_norm_(self.dis_nn.parameters(), self.mc.dis_grad_max_norm)
                if step_flag:
                    step_flags.append(step_flag)
                    recorder.variables(loss_dict, namespace='training/dis_', prog_bar=True)
                    recorder.variable("training/dis_loss_sum", weighted_sum, prog_bar=True)
                    recorder.variable(f"training/dis_param_norm", param_norm, prog_bar=True)
        self.dis_optimizer.step()

        if len(step_flags) > 0:
            assert len(step_flags) == 1, "You should set smaller steps_per_epoch_num! "
            learning_rate = self.step_scheduler(self.dis_scheduler)
            recorder.variable(f"training/dis_lr", learning_rate, prog_bar=False)

    def step_train_generator(self, accumulated_batch: list[dict], accumulate_num: int,
                             recorder: mlogging.MainProcessLogger):
        step_flags = []
        self.network.train()
        self.dis_nn.eval()
        self.optimizer.zero_grad(set_to_none=True)
        with xnn.without_autograd(self.dis_nn):  # !
            for cd, (batch_input, step_flag) in utils.iter.cd_enumerate(accumulated_batch):
                with self.grad_sync_context(cd == 0):
                    nn_output = self.network(batch_input['audio'])
                    gen_weighted_sum, gen_loss_dict = self.dis_nn(fake=nn_output['generated_audio'],
                                                                  real=batch_input['audio'],
                                                                  get_gen_loss=True)
                    nn_output['gen_loss'] = gen_weighted_sum
                    weighted_sum, loss_dict = self.loss_nn(nn_output, batch_input)
                    ACC.backward(weighted_sum / accumulate_num)
                    param_norm = ACC.clip_grad_norm_(self.network.parameters(), self.mc.grad_max_norm)
                    if step_flag:
                        step_flags.append(step_flag)
                        recorder.variables(loss_dict, namespace='training/', prog_bar=True)
                        recorder.variable("training/loss_sum", weighted_sum, prog_bar=True)
                        recorder.variable(f"training/param_norm", param_norm, prog_bar=True)

        self.optimizer.step()

        if len(step_flags) > 0:
            assert len(step_flags) == 1, "You should set smaller steps_per_epoch_num! "
            learning_rate = self.step_scheduler(self.scheduler)
            recorder.variable(f"training/lr", learning_rate, prog_bar=False)
            recorder.variables(self.param_tracker(), namespace='training/', prog_bar=False)

    def evaluate(self, dataloader, name="evaluating"):
        recorder = mlogging.MainProcessLogger()

        self.network.eval()
        self.dis_nn.eval()
        with torch.inference_mode():
            self.metric.reset()

            for i, batch_input in enumerate(mlogging.progress_bar(dataloader, desc=name)):
                nn_output = self.network(batch_input['audio'])
                gen_weighted_sum, gen_loss_dict = self.dis_nn(fake=nn_output['generated_audio'],
                                                              real=batch_input['audio'],
                                                              get_gen_loss=True)
                nn_output['gen_loss'] = gen_weighted_sum
                weighted_sum, loss_dict = self.loss_nn(nn_output, batch_input)

                recorder.variables(loss_dict, namespace=f'{name}/', log=False, )
                recorder.variable(f"{name}/loss_sum", weighted_sum.item(), log=False)
                self.metric.update(nn_output, batch_input, )
                if i == 0:
                    mlogging.log_sample(nn_output, batch_input, audio_sample_rate=self.mc.sample_rate, log_name=name)

            mlogging.MainProcessLogger().variables(recorder.get_results(), prog_bar=True)
            metric_results = self.metric.log_results(RS.tlog if ACC.is_main_process else None, namespace=name)
            return metric_results

    def train_epoch(self):
        recorder = mlogging.MainProcessLogger()

        accumulate_num = self.mc.gradient_accumulation_steps

        for batch_i, accumulated_batch in enumerate(utils.iter.batched(self.train_progress, accumulate_num)):

            if batch_i % self.mc.train_dis_every_n_batches == 0:
                self.step_train_discriminator(accumulated_batch, accumulate_num, recorder)

            self.step_train_generator(accumulated_batch, accumulate_num, recorder)

        return recorder.get_results()

    def train_epoch_without_discriminator(self):
        recorder = mlogging.MainProcessLogger()

        accumulate_num = self.mc.gradient_accumulation_steps

        orig_dis_nn_forward = self.dis_nn.forward
        self.dis_nn.forward = lambda *_, **__: (0., dict())

        for batch_i, accumulated_batch in enumerate(utils.iter.batched(self.train_progress, accumulate_num)):
            self.step_train_generator(accumulated_batch, accumulate_num, recorder)

        self.dis_nn.forward = orig_dis_nn_forward
        return recorder.get_results()
