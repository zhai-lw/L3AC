import torch
import torch.nn.functional as F

import xtract.nn as xnn
from . import dac_discriminator


class Discriminator(xnn.Module):
    def __init__(self, sample_rate=16000, fft_size=(2048, 1024, 512)):
        super().__init__()
        self.dis_nn = dac_discriminator.Discriminator(sample_rate=sample_rate, fft_sizes=fft_size)

    @property
    def trainable_modules(self) -> dict[str, torch.nn.Module]:
        return {
            "dis_nn": self.dis_nn,
        }

    def forward(self, fake, real, get_dis_loss=False, get_gen_loss=False):
        if get_dis_loss:
            return self.discriminator_loss(fake, real)
        if get_gen_loss:
            return self.generator_loss(fake, real)
        raise ValueError(f"Please specific the loss type")

    def discriminator_loss(self, fake, real):
        d_fake = self.dis_nn(fake.clone().detach())
        d_real = self.dis_nn(real)

        loss_d = 0
        for x_fake, x_real in zip(d_fake, d_real):
            loss_d += torch.mean(x_fake[-1] ** 2)
            loss_d += torch.mean((1 - x_real[-1]) ** 2)
        return loss_d, {
            "dis": loss_d,
        }

    def generator_loss(self, fake, real):
        d_fake = self.dis_nn(fake)
        d_real = self.dis_nn(real)

        loss_g = 0
        for x_fake in d_fake:
            loss_g += torch.mean((1 - x_fake[-1]) ** 2)

        loss_feature = 0

        for i in range(len(d_fake)):
            for j in range(len(d_fake[i]) - 1):
                loss_feature += F.l1_loss(d_fake[i][j], d_real[i][j].detach())
        return loss_g + loss_feature * 2, {
            "gen": loss_g,
            "feat": loss_feature,
        }
