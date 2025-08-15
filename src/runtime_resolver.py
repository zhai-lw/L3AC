import os
import random
import string
import sys
import time
from functools import cached_property
from pathlib import Path

import torch
from pydantic import Field, field_validator, ValidationInfo, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

import xtract
import utils
from utils.file import PROJECT_PATH, SOURCE_PATH


def generate_version():
    time_version = time.strftime("%y%m%d%H%M%S", time.localtime())[1:]
    random_version = ''.join(random.sample(string.ascii_uppercase + string.digits, 4))
    return time_version + '_' + random_version


class Settings(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True, env_prefix='RT_')

    config: str = Field('debug', )
    version: str = Field(default_factory=generate_version)

    data_path: Path = Field((PROJECT_PATH / 'data').resolve())
    etc_path: Path = Field((PROJECT_PATH / 'etc').resolve())
    output_dir: Path = Field((PROJECT_PATH / 'output').resolve())
    random_seed: int = Field(None)

    cpu_num: int = Field(default_factory=lambda: min(os.cpu_count() // 4, 8))
    enough_memory: bool = Field(default=False)
    waiting_cuda: bool = Field(default=False)
    gpu_num: int = Field(default=0)
    cuda_devices: list[int] = Field(default=None, validate_default=True)

    data_precision: torch.dtype | str = Field(torch.float32)

    source_release: bool = Field(default=False)
    log_record: bool = Field(default=False)

    @field_validator('random_seed', mode='before')
    @classmethod
    def set_random_seed(cls, random_seed, info: ValidationInfo) -> int:
        if random_seed is None:
            random_seed = 42 * utils.iter.consume((info.data['output_dir'] / 'log').iterdir())
        return random_seed

    @field_validator('cuda_devices', mode='before')
    @classmethod
    def detect_cuda(cls, cuda_devices: list[int] | None, info: ValidationInfo) -> list[int]:
        if cuda_devices is None:
            cuda_devices = xtract.gpu.auto_choose_cuda(info.data['gpu_num'], waiting=info.data['waiting_cuda'])
        return cuda_devices

    @field_validator('data_precision', mode='before')
    @classmethod
    def convert_precision(cls, data_precision: torch.dtype | str) -> torch.dtype:
        if isinstance(data_precision, str):
            return xtract.nn.utils.get_torch_precision(data_precision)
        else:
            return data_precision

    @computed_field
    @cached_property
    def config_path(self) -> str:
        return self.config.split("-")[0]

    @computed_field
    @cached_property
    def config_name(self) -> str:
        return self.config.replace('/', '-')

    @computed_field
    @cached_property
    def runtime_name(self) -> str:
        current_running = Path(sys.argv[0])
        return current_running.parent.name + '.' + current_running.stem + '.' + self.config_name

    @cached_property
    def output_path(self) -> Path:
        (output_path := self.output_dir / self.runtime_name).mkdir(exist_ok=True)
        return output_path

    @cached_property
    def log_path(self) -> Path:
        return self.output_dir / f"log/{self.runtime_name}.{self.version}"

    @cached_property
    def tlog(self) -> xtract.tensor_log.Writer:
        if self.log_record is False:
            raise RuntimeError("log record is disabled")
        return xtract.tensor_log.Writer(log_dir=self.log_path)

    def init_runtime(self):
        xtract.nn.seed_everything(self.random_seed)
        xtract.nn.EPS = xtract.nn.get_eps(self.data_precision)
        xtract.tensor_log.init_tensor_rich_repr()
        if self.log_record:
            self.log_path.mkdir(exist_ok=False)
            utils.log.LOG_FILE = self.log_path / 'log.txt'
            if self.source_release:
                utils.log.release_source(self.log_path, SOURCE_PATH)

            log = utils.log.get_logger()
            utils.output.dictionary(self.model_dump(), out_fun=log.info)
            self.tlog.hyper_parameters(self.model_dump(), tag="runtime_settings")
            xtract.gpu.output_stat(cuda2use=self.cuda_devices, output_func=log.info)
            xtract.gpu.check_stat(cuda2use=self.cuda_devices, output_func=log.warning)


RUNTIME_SETTING: Settings


def init_setting(env_file=PROJECT_PATH / 'runtime.env', **kwargs) -> Settings:
    global RUNTIME_SETTING
    RUNTIME_SETTING = Settings(_env_file=env_file, **kwargs)
    return RUNTIME_SETTING


def init_runtime(env_file=PROJECT_PATH / 'runtime.env', log_status=True, **kwargs) -> Settings:
    init_setting(env_file=env_file, **kwargs)
    RUNTIME_SETTING.init_runtime()
    return RUNTIME_SETTING
