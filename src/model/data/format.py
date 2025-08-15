from functools import partial

import librosa
import numpy
import torch

from prepare.data_process import DN
from xtract.data import XDataset


class DataCollector(torch.utils.data.Dataset):
    def __init__(self, dataset: XDataset, ds_name: str,
                 sample_rate: int, data_normalize=False, ):
        self.xds = dataset
        data_id_max_len = len(str(len(self.xds)))
        self.data_id_formatter = f'{ds_name}.{{0:0{data_id_max_len}d}}'

        self.sample_rate = sample_rate
        self.data_normalize = data_normalize
        orig_sr = self.xds.x_features[DN.audio].frame_rate
        if orig_sr == self.sample_rate:
            self.resampler = lambda x: x
        else:
            self.resampler = partial(librosa.resample, orig_sr=orig_sr, target_sr=self.sample_rate,
                                     res_type="soxr_vhq")

    def __len__(self):
        return len(self.xds)

    def format_audio(self, audio_data):
        resampled_data = self.resampler(audio_data)
        if self.data_normalize:
            resampled_data = normalize_(resampled_data[None, :])[0]
        return resampled_data

    def __getitem__(self, key):
        data_item = self.xds[key]
        data_item[DN.id] = self.data_id_formatter.format(key)
        data_item[DN.audio] = self.format_audio(data_item[DN.audio])
        return data_item

    def __getitems__(self, keys):
        data_items = self.xds[keys]
        data_items[DN.id] = [self.data_id_formatter.format(key) for key in keys]
        data_items[DN.audio] = [self.format_audio(data_audio) for data_audio in data_items[DN.audio]]
        return data_items


def normalize_(audio_data, normalize_rate=0.75):
    top001length = int(audio_data.shape[1] * 0.001)
    if isinstance(audio_data, numpy.ndarray):
        ta = numpy.partition(numpy.abs(audio_data), -top001length, axis=1)[:, -top001length:]
        audio_data /= (ta.mean(axis=1, keepdims=True) / normalize_rate + 1e-6)
    elif isinstance(audio_data, torch.Tensor):
        ta = torch.topk(torch.abs(audio_data), top001length, dim=1, largest=True, sorted=False)[0]
        audio_data /= (ta.mean(dim=1, keepdim=True) / normalize_rate + 1e-6).detach()

    audio_data[audio_data > 1] = 1
    audio_data[audio_data < -1] = -1
    return audio_data
