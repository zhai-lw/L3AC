import torch

import utils
from xtract.nn import t2n
import tools.audio.extend

from . import base


class T2NAutoSigner(utils.args.AutoSigner):
    @staticmethod
    def format_args(arg_name: str, arg_value):
        if isinstance(arg_value, torch.Tensor):
            arg_value = t2n(arg_value)
        return arg_value


class PESQ(base.ScoreMetric):
    desired_sample_rate = 0

    def __init__(self, input_sample_rate, cpu_num=0):
        super().__init__()
        self.input_sample_rate = input_sample_rate
        self.cpu_num = cpu_num

    @T2NAutoSigner()
    def update(self, generated_audio, audio):
        new_scores = tools.audio.extend.pesq_batch(generated_audio, sample_rate=self.input_sample_rate,
                                                   ref_audio=audio, cpu_num=self.cpu_num)
        self.scores += [s for s in new_scores if isinstance(s, float)]


class STOI(base.ScoreMetric):
    desired_sample_rate = 0

    def __init__(self, input_sample_rate):
        super().__init__()
        self.input_sample_rate = input_sample_rate

    @T2NAutoSigner()
    def update(self, generated_audio, audio):
        new_scores = tools.audio.extend.stoi_batch(generated_audio, sample_rate=self.input_sample_rate,
                                                   ref_audio=audio, )
        self.scores += new_scores
