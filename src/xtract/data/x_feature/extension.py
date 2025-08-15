import io
from functools import partial

import numpy

import utils
from .base import XFeature


class XCompress(XFeature):
    def __init__(self, skip_encode=False):
        self.skip_encode = skip_encode
        if self.skip_encode:
            self.encode = self.identity
            self.check = self.check_decode

    @staticmethod
    def identity(data):
        return data if data is not None else b''

    def check_decode(self, x_data):
        if x_data is not None:
            assert isinstance(self.decode(x_data), numpy.ndarray)


if utils.module.installed("soundfile"):
    import soundfile
    from .audio_coder import FfmpegAudioCoder


    class XWave(XCompress):
        def __init__(self, compress_fmt='WAV', skip_encode=False, frame_rate=16000, dtype=numpy.float32):
            super().__init__(skip_encode=skip_encode)
            self.compress_fmt = compress_fmt
            self.frame_rate = frame_rate
            self.dtype = dtype
            if self.compress_fmt == 'MP3':
                self.encode_data = self.get_ffmpeg_encoder()
                self.is_equal = self.is_equal_mp3
            elif self.compress_fmt.startswith('MP3'):
                self.encode_data = self.get_ffmpeg_encoder(quality=self.compress_fmt.split('-')[1])
                self.is_equal = self.is_equal_mp3
            else:
                assert self.compress_fmt in soundfile.available_formats()

        def encode_data(self, data: numpy.ndarray) -> bytes:
            buf = io.BytesIO()
            soundfile.write(buf, data, samplerate=self.frame_rate, format=self.compress_fmt)
            return buf.getvalue()

        def get_ffmpeg_encoder(self, fmt="mp3", quality='high'):
            assert fmt in ('mp3',)
            return partial(FfmpegAudioCoder(quality=quality).encode, sample_rate=self.frame_rate)

        def decode_data(self, x_data: bytes) -> numpy.ndarray:
            buf = io.BytesIO(x_data)
            return soundfile.read(buf, dtype=self.dtype)[0]

        @staticmethod
        def is_equal(decoded_data, original_data) -> bool:
            return numpy.abs(decoded_data - original_data).mean() < 0.02

        @staticmethod
        def is_equal_mp3(decoded_data, original_data) -> bool:
            data_len = min(len(original_data), len(decoded_data))
            return numpy.abs(decoded_data[:data_len] - original_data[:data_len]).mean() < 0.2

if utils.module.installed("PIL"):
    from PIL import Image


    class XImg(XCompress):
        def __init__(self, compress_fmt: str = 'png', skip_encode=False):  # 'png', 'jpeg'
            super().__init__(skip_encode=skip_encode)
            self.compress_fmt = compress_fmt

        def encode_data(self, data: numpy.ndarray) -> bytes:
            buf = io.BytesIO()
            img = Image.fromarray(data)
            img.save(buf, format=self.compress_fmt)
            return buf.getvalue()

        def decode_data(self, x_data: bytes) -> numpy.ndarray:
            buf = io.BytesIO(x_data)
            img = Image.open(buf, formats=(self.compress_fmt,))
            return numpy.array(img)

        @staticmethod
        def is_equal(decoded_data, original_data) -> bool:
            return numpy.array_equal(decoded_data, original_data)
