from torch import nn

from utils.args import AutoSigner


class LossWrapper(nn.Module):
    def __init__(self, loss_nn):
        super().__init__()
        self.loss_nn = loss_nn

    @AutoSigner()
    def forward(self, prediction, label):
        return self.loss_nn(prediction, label)


class NetworkLoss(nn.Module):
    def __init__(self, loss_name):
        super().__init__()
        self.loss_name = loss_name

    def forward(self, nn_output, _):
        return nn_output[self.loss_name]
