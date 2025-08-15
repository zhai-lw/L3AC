import contextlib
import pathlib
from collections import OrderedDict
from typing import Union, Iterable, Sequence

import numpy

# requirements: pip install rich tensorboard
import rich.progress
import torch.utils.tensorboard

import utils

log = utils.log.get_logger()


class Writer(torch.utils.tensorboard.SummaryWriter):
    def __init__(
            self,
            log_dir: pathlib.Path,
            comment="",
            purge_step=None,
            max_queue=10,
            flush_secs=120,
            filename_suffix="",
            progress: rich.progress.Progress = None
    ):
        if log_dir is None:
            self.log_dir = utils.file.DATA_PATH.joinpath("output/tensor.log")
        else:
            self.log_dir = log_dir
        super().__init__(log_dir=self.log_dir, comment=comment, purge_step=purge_step, max_queue=max_queue,
                         flush_secs=flush_secs, filename_suffix=filename_suffix)

        if progress is None:
            self.progress = utils.log.default_rich_progress.progress
        else:
            self.progress = progress
        self.task_variable = OrderedDict()
        self.step_dict = Steps()
        # tensorboard_run_cmd = f"tensorboard --logdir {self.log_dir.parent} --port 6006"
        # log.info(f"Run '{tensorboard_run_cmd}' to start tensorboard")
        # log.info("Then visit http://localhost:6006/ to view the log")

    def progressbar(self, sequence: Union[Iterable, Sequence],
                    length=None, desc: str = 'working'):
        with self.new_task(desc) as task_id:
            progress_sequence = self.progress.track(sequence, task_id=task_id, total=length)
            for item in progress_sequence:
                yield item
            self.flush()

    @contextlib.contextmanager
    def new_task(self, description: str):
        task_id = self.progress.add_task(description)
        desc_variable = {}
        self.task_variable[task_id] = desc_variable

        yield task_id

        self.progress.remove_task(task_id=task_id)
        del self.task_variable[task_id]

    @staticmethod
    def auto_step(func):
        def wrapper(self, name, *args, step=None, **kwargs):
            if step is None:
                step = self.step_dict.get_and_update(func.__name__, name)
            func(self, name, *args, **kwargs, step=step)
            return step

        return wrapper

    @auto_step
    def variable(self, name, value, step, prog_bar=False):
        self.add_scalar(name, value, global_step=step, new_style=True)
        if prog_bar and len(self.task_variable) > 0:
            task_id, desc_variable = next(reversed(self.task_variable.items()))
            desc_variable[name] = f"{value:.4g}"
            self.progress.update(task_id, suffix=" ".join(f"{k}:{v}" for k, v in desc_variable.items()))

    @auto_step
    def variables(self, name: str, tag_variables: dict, step):
        self.add_scalars(main_tag=name, tag_scalar_dict=tag_variables, global_step=step)

    @auto_step
    def embeddings(self, name: str, embeddings: torch.Tensor, labels: list[str] | None, step):
        self.add_embedding(mat=embeddings, metadata=labels, tag=name, global_step=step)

    @auto_step
    def distribution(self, name, data, step, bins="tensorflow"):
        if isinstance(data, (list, tuple)):
            data = numpy.array(data)
        self.add_histogram(name, data, global_step=step, bins=bins)

    @auto_step
    def image(self, name, image, step):
        self.add_image(tag=name, img_tensor=image, global_step=step)

    @auto_step
    def figure(self, name, figure, step):
        self.add_figure(tag=name, figure=figure, global_step=step, close=True)

    @auto_step
    def audio(self, name, audio, step, tag='audio', sample_rate=16000, add_spectrogram=False):
        assert "\t" not in tag, "tag cannot contain any special char like \\t, \\n, etc."
        tag = f"{name}/{step:04d}|{tag}"
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu()
        else:
            audio = torch.from_numpy(audio)
        self.add_audio(tag=tag, snd_tensor=audio[None, :], sample_rate=sample_rate, global_step=step)
        if add_spectrogram:
            spectrogram_fig = utils.plot.audio_spectrogram(audio.numpy(), title=tag, show=False, out_path=None, )
            self.figure(name=name, figure=spectrogram_fig, step=step)

    def metrics(self, hparams: dict, metrics: dict):
        hp = {}
        for k, v in hparams.items():
            if isinstance(v, (int, float, str, bool)):
                hp[k] = v
            elif isinstance(v, torch.Tensor):
                hp[k] = v
            else:
                hp[k] = str(v)
        self.add_hparams(hp, metrics)

    def hyper_parameters(self, hparams, tag="hyper_parameters"):
        hp = ''
        for k, v in hparams.items():
            hp += f"{k}:\t {v}\n"
        self.add_text(tag, hp)

    def start_tensorboard(self, port="6006"):
        # log.warning("It may get stuck if TensorBoard is initiated during training.")
        log.info(f"tensorboard --logdir {str(self.log_dir.parent)} --port {port}")
        import subprocess
        subprocess.Popen(["tensorboard", "--logdir", str(self.log_dir.parent), f"--port {port}", ],
                         stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
                         )
        log.info(f"tensorboard started, visit http://localhost:{port}/ to view the log")


class Steps:
    def __init__(self):
        self.step_dict = {}

    def get_and_update(self, func_name, name):
        if func_name not in self.step_dict:
            self.step_dict[func_name] = {}
        func_dict = self.step_dict[func_name]
        step = func_dict.get(name, 0)
        func_dict[name] = step + 1
        return step


def init_tensor_rich_repr():
    from torch import Tensor

    @utils.module.add_method(Tensor)
    def __rich_repr__(obj):
        yield "shape", obj.shape
        yield "dtype", obj.dtype
        yield "device", obj.device
        yield "requires_grad", obj.requires_grad
        flatten_obj = obj.reshape(-1)
        if flatten_obj.numel() > 5:
            yield "value", (f"[{', '.join(f'{t.item():.4g}' for t in flatten_obj[:3])}, "
                            f"..., {flatten_obj[-1].item():.4g}]")
        else:
            yield "value", f"[{', '.join([f'{t.item():.4g}' for t in flatten_obj])}]"
