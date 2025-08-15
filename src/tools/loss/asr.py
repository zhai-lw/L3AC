import pathlib

import numpy
import torch
import torchaudio
import whisper

import utils
import xtract.nn as xnn

log = utils.log.get_logger()


class PerceptualLoss(torch.nn.Module):

    def __init__(self, model_path: pathlib.Path, sample_rate=16000, reduction_func=torch.nn.MSELoss(reduction='none')):
        super().__init__()
        self.resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        if model_path.exists():
            self.model = whisper.load_model(str(model_path), "cpu")
        else:
            log.warning("No model found, downloading...")
            self.model = whisper.load_model('tiny.en', "cpu", download_root=str(model_path.parent))
        xnn.freeze(self.model)
        self.options = whisper.DecodingOptions(language="en", without_timestamps=True)
        log.info(
            f"Model is {'multilingual' if self.model.is_multilingual else 'English-only'} "
            f"and has {sum(numpy.prod(p.shape) for p in self.model.parameters()):,} parameters."
        )
        log.info(f"finished loading model {model_path}")
        self.reduction_func = reduction_func

    def forward(self, predictions, targets):
        predict_features = self.get_feature(predictions)
        target_features = self.get_feature(targets)
        return self.reduction_func(predict_features, target_features).sum(2).mean()

    def get_feature(self, audios: torch.Tensor):
        audios = self.resampler(audios)
        pad_audios = whisper.pad_or_trim(audios)
        tensor_mel = whisper.log_mel_spectrogram(pad_audios)
        features = self.model.encoder(tensor_mel)
        return features

    def get_results_from_features(self, features):
        results = self.model.decode(features, options=self.options)
        return [r.text for r in results]
