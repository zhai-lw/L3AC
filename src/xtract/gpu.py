import collections
import os
import time
from dataclasses import dataclass

# requirements: pip install pynvml
from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, \
    nvmlDeviceGetMemoryInfo, nvmlShutdown, nvmlDeviceGetName

MEM_UNIT = 1024 * 1024 * 1024  # 1GB

if "CUDA_VISIBLE_DEVICES" in os.environ:
    GPU_ID_A_CUDA = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
else:
    nvmlInit()
    GPU_ID_A_CUDA = list(range(nvmlDeviceGetCount()))
    nvmlShutdown()


@dataclass
class GPUInfo:
    index: int
    cuda_index: int
    cuda_visible: bool
    name: str
    total: float
    free: float

    @property
    def free_percent(self) -> int:
        return round(self.free / self.total * 100)

    @property
    def used_percent(self) -> int:
        return 100 - self.free_percent

    @property
    def priority(self) -> float:
        return self.free_percent // 5 + self.cuda_index * 0.01

    @property
    def busy(self):
        return self.used_percent > 10

    def __str__(self):
        return f"gpu:{self.index}-cuda:{self.cuda_index}({self.name}):\t" \
               f"total: {self.total:4.1f}GB used: {self.used_percent:02d}%"


def get_stat():
    nvmlInit()
    gpu_stats = []
    for i in range(nvmlDeviceGetCount()):
        handle = nvmlDeviceGetHandleByIndex(i)
        meminfo = nvmlDeviceGetMemoryInfo(handle)
        name = nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        name = "_".join(name.split(" ")[-2:])
        if i in GPU_ID_A_CUDA:
            cuda_index = GPU_ID_A_CUDA.index(i)
            cuda_visible = True
        else:
            cuda_index = -1
            cuda_visible = False
        gpu_stats.append(
            GPUInfo(
                index=i,
                cuda_index=cuda_index,
                cuda_visible=cuda_visible,
                name=name,
                total=meminfo.total / MEM_UNIT,
                free=meminfo.free / MEM_UNIT,
            )
        )
    nvmlShutdown()
    return gpu_stats


def output_stat(gpu2use=None, cuda2use=None, output_func=print):
    if cuda2use is not None:
        gpu2use = [GPU_ID_A_CUDA[i] for i in cuda2use]
    if gpu2use is not None:
        output = lambda x: output_func(f"{x}" + ("\t<- using" if x.index in gpu2use else "\t\t"))
    else:
        output = output_func
    collections.deque(map(output, get_stat()), maxlen=0)


def check_stat(gpu2use=None, cuda2use=None, output_func=print):
    stats = get_stat()
    if cuda2use is not None:
        gpu2use = [GPU_ID_A_CUDA[i] for i in cuda2use]
    check_pass = True
    for i in gpu2use:
        if stats[i].busy:
            output_func(f"gpu:{stats[i].index} is busy, "
                        f"total memory:{stats[i].total:.1f}GB, used {stats[i].used_percent:.1f}%")
            check_pass = False
    return check_pass


def auto_choose_cuda(gpu_num=1, ignore_invisible=True, waiting=False):
    while True:
        gpu_stats = get_stat()
        if ignore_invisible:
            gpu_stats = [gpu for gpu in gpu_stats if gpu.cuda_visible]
        if waiting:
            free_gpus = sum((not gpu.busy) for gpu in gpu_stats)
            if free_gpus < gpu_num:
                print('.', end='', flush=True)
                time.sleep(5.0)
                continue
        break

    choices = sorted(gpu_stats, key=lambda gpu: gpu.priority, reverse=True)
    assert len(choices) >= gpu_num, f"Not enough free GPUs, only {len(choices)} available"
    return sorted(gpu.cuda_index for gpu in choices[:gpu_num])
