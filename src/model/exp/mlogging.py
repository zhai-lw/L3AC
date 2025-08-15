from collections import defaultdict

import numpy
import torch

import utils.iter

import tools.audio.extend

from .acc_runtime import ACC, RS


def progress_bar(iterable: utils.iter.SizedIterable, flag_frequency: int = None, **kwargs):
    iter_len = len(iterable)
    if flag_frequency is not None:
        iterable = zip(iterable, utils.iter.UpSampler(([True, ] * flag_frequency),
                                                      num=len(iterable), fill_value=False))
    if ACC.is_main_process:
        iterable = RS.tlog.progressbar(iterable, length=iter_len, **kwargs)
    return iterable


class MainProcessLogger:

    def __init__(self):
        self.recorder = defaultdict(list)

    @ACC.on_main_process
    def variable(self, name, value, log=True, prog_bar=False):
        if isinstance(value, torch.Tensor) or isinstance(value, numpy.ndarray):
            value = value.item()
        self.recorder[name].append(value)
        if log:
            RS.tlog.variable(name, value, prog_bar=prog_bar)

    @ACC.on_main_process
    def variables(self, value_dict, namespace='', **kwargs):
        for name, value in value_dict.items():
            self.variable(namespace + name, value, **kwargs)

    def get_results(self):
        return {name: numpy.mean(values).item() for name, values in self.recorder.items()}


def get_captions(gen_audio: numpy.ndarray, ref_audio: numpy.ndarray, sample_rate: int):
    predict_pesq = tools.audio.extend.pesq(gen_audio, sample_rate=sample_rate,
                                           ref_audio=ref_audio)
    predict_stoi = tools.audio.extend.stoi(gen_audio, sample_rate=sample_rate,
                                           ref_audio=ref_audio)
    return f"pesq: {predict_pesq:.2f}, stoi: {predict_stoi:.2f}"


@ACC.on_main_process
def log_sample(nn_output, ref_input, audio_sample_rate: int, log_name: str = "result"):
    input_audio = ref_input['audio'][0].detach().cpu().numpy()
    gen_audio = nn_output['generated_audio'][0].detach().cpu().numpy()

    current_step = RS.tlog.audio(name=f"{log_name}", audio=input_audio, tag="input_audio",
                                 sample_rate=audio_sample_rate, add_spectrogram=True)

    captions = get_captions(gen_audio, ref_audio=input_audio, sample_rate=audio_sample_rate)

    RS.tlog.audio(name=f"{log_name}", audio=gen_audio, tag=f"gen_audio-{captions}", step=current_step,
                  sample_rate=audio_sample_rate, add_spectrogram=True)
