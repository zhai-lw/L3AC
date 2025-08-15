import torch

from utils.args import AutoSigner

from . import base


class Accuracy(base.ScoreMetric):

    @AutoSigner()
    def update(self, prediction: torch.Tensor, label: torch.Tensor):
        predict_label = prediction.argmax(dim=1, keepdim=False)  # get the index of the max log-probability
        self.scores += predict_label.eq(label).tolist()
