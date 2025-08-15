import os
from pathlib import Path

import accelerate
import torch

import utils
import runtime_resolver


def get_gpu_nums() -> int:
    # import torch
    # return torch.distributed.get_world_size()
    return int(os.environ.get('LOCAL_WORLD_SIZE', 1))


def set_available_gpus(device_indexes):
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, sorted(device_indexes)))


def get_acc_precision(data_precision: torch.dtype) -> str:
    return {
        torch.bfloat16: accelerate.utils.PrecisionType.BF16,
        torch.float16: accelerate.utils.PrecisionType.FP16,
    }.get(data_precision, accelerate.utils.PrecisionType.NO)


RS = runtime_resolver.init_setting(
    env_file=utils.file.PROJECT_PATH / 'runtime_gpu.env',
    gpu_num=get_gpu_nums(),
)

set_available_gpus(RS.cuda_devices)

RS = accelerate.utils.broadcast_object_list([RS, ], from_process=0)[0]

ACC = accelerate.Accelerator(
    mixed_precision=get_acc_precision(RS.data_precision),
    step_scheduler_with_optimizer=False,
)

if ACC.is_main_process:
    RS.log_record = True
else:
    RS.log_record = False
RS.init_runtime()
accelerate.utils.set_seed(RS.random_seed)

CONFIG_DIR = Path(__file__).parent / "configs"
