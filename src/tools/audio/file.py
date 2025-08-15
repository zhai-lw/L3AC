import io
import pathlib
import subprocess
import tempfile

import librosa
import soundfile
import numpy

import utils

"""
Suggest format: normal audio (100s*16000)


|            | compress_rate | recompress_rate | loss_mean1k | loss_max1k | encode_ms  | decode_ms |
| ---------- | ------------- | --------------- | ----------- | ---------- | ---------- | --------- |
| MP3        | 0.047045      | 0.998624        | 7.260235    | 126.335412 | 152.930377 | 9.940090  |
| mp3 -q:a 4 | 0.065216      | 1.000073        | 5.205944    | 100.687072 | 112.502859 | 11.366169 | (avg 165 kbps)
| OGG        | 0.073566      | 0.999370        | 3.956596    | 81.144884  | 134.530599 | 15.468786 | (within 100s)
| mp3 -q:a 2 | 0.078369      | 0.999865        | 3.917494    | 104.247212 | 116.230586 | 11.938436 | (avg 190 kbps)
| mp3 -q:a 0 | 0.100013      | 0.999911        | 2.483851    | 60.079247  | 118.986747 | 12.820218 | (avg 245 kbps)
| WVE        | 0.250011      | 0.969509        | 0.636336    | 15.836895  | 4.110870   | 1.578004  |
| FLAC       | 0.327815      | 1.000082        | 0.007618    | 0.015259   | 38.935962  | 18.636148 |
| WAV        | 0.500015      | 0.969233        | 0.015254    | 0.030517   | 4.124341   | 1.719475  |


"""

default_f_type = numpy.float32
default_i_type = numpy.int16


def f2i(audio_data: numpy.ndarray[float], i_type=default_i_type) -> numpy.ndarray[int]:
    data = audio_data * numpy.iinfo(i_type).max
    return data.astype(i_type)


def i2f(data: numpy.ndarray[int], f_type=default_f_type) -> numpy.ndarray[float]:
    return data.astype(f_type) / numpy.iinfo(data.dtype).max


if utils.module.installed("pydub"):
    import pydub
    from pydub import AudioSegment, playback


    def ffmpeg_decode(x_data: bytes) -> numpy.ndarray[float]:
        conversion_command = ['ffmpeg', '-y', '-read_ahead_limit', '-1',
                              '-i', 'cache:pipe:0',
                              '-acodec', 'pcm_s16le', '-vn', '-f', 'wav', '-']
        p = subprocess.Popen(conversion_command, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p_out, p_err = p.communicate(x_data)
        p_out = bytearray(p_out)
        pydub.audio_segment.fix_wav_headers(p_out)
        return decode(bytes(p_out))


    def f2a(audio_data: numpy.ndarray[float], frame_rate=16000, sample_width=2) -> AudioSegment:
        channels = 1 if len(audio_data.shape) == 1 else audio_data.shape[1]
        i_type = numpy.dtype(f'int{sample_width * 8}').type
        data = f2i(audio_data, i_type=i_type)
        data = data[:data.size // sample_width * sample_width]
        audio_segment = AudioSegment(data=data.tobytes(), channels=channels, frame_rate=frame_rate,
                                     sample_width=sample_width)
        return audio_segment


    def a2f(audio_segment: AudioSegment, channels=1, frame_rate=16000, f_type=default_f_type) -> numpy.ndarray[float]:
        audio_segment = audio_segment.set_channels(channels)
        audio_segment = audio_segment.set_frame_rate(frame_rate)
        data = numpy.array(audio_segment.get_array_of_samples())
        if audio_segment.channels != 1:
            data = data.reshape((-1, audio_segment.channels))
        audio_data = i2f(data, f_type=f_type)
        return audio_data


    def read(audio_path: pathlib.Path):
        return AudioSegment.from_file(audio_path)


    def read_and_format(audio_path: pathlib.Path, channels=1, frame_rate=16000, check=False) -> numpy.ndarray[float]:
        audio_segment = read(audio_path)
        if check:
            assert channels == audio_segment.channels
            assert frame_rate == audio_segment.frame_rate
        return a2f(audio_segment, channels=channels, frame_rate=frame_rate)


    def play(audio_data: numpy.ndarray[float], frame_rate=16000, sample_width=2):
        audio_segment = f2a(audio_data, frame_rate=frame_rate, sample_width=sample_width)
        playback.play(audio_segment)


def load(audio_path, channels=1, frame_rate=16000, f_type=default_f_type) -> numpy.ndarray[float]:
    try:
        audio_data, original_frame_rate = soundfile.read(audio_path, dtype=f_type, always_2d=False)
    except soundfile.LibsndfileError as e:
        raise BrokenPipeError(f'Error when loading audio file {audio_path}: {e}')
    if channels == 1 and audio_data.ndim == 2:
        audio_data = librosa.to_mono(audio_data.T)
    if original_frame_rate != frame_rate:
        audio_data = librosa.resample(audio_data, orig_sr=original_frame_rate, target_sr=frame_rate,
                                      res_type="soxr_vhq")
    return audio_data


def example() -> numpy.ndarray[float]:
    example_file = librosa.example('libri1')
    return load(example_file)


def export(audio_data: numpy.ndarray[float], out_f, fmt='wav', frame_rate=16000):
    """
    out_f can be a file path or a file-like object(io.BytesIO())
    """
    soundfile.write(out_f, audio_data, samplerate=frame_rate, format=fmt)


def export_to_bytes_buf(*args, **kwargs) -> io.BytesIO:
    buf = io.BytesIO()
    export(*args, out_f=buf, **kwargs)
    return buf


def encode(audio_data: numpy.ndarray[float], encode_fmt='wav') -> bytes:
    return export_to_bytes_buf(audio_data, fmt=encode_fmt).getvalue()


def decode(x_data: bytes, dtype=default_f_type) -> numpy.ndarray[float]:
    buf = io.BytesIO(x_data)
    return soundfile.read(buf, dtype=dtype)[0]


class FfmpegAudioCoder:
    codec_fmt = "mp3"

    def __init__(self, quality='high'):
        self.target_file = tempfile.NamedTemporaryFile()
        self.wav_file = tempfile.NamedTemporaryFile()
        if quality == 'high':
            self.quality_args = ['-q:a', '0']  # avg 245 kbps
        elif quality == 'medium':
            self.quality_args = ['-q:a', '2']  # avg 190 kbps
        elif quality == 'low':
            self.quality_args = ['-q:a', '4']  # avg 165 kbps

    def encode(self, audio_data: numpy.ndarray[float], sample_rate=16000) -> bytes:
        # self.target_file.seek(0)
        soundfile.write(self.wav_file.name, audio_data, sample_rate, format="wav")
        conversion_command = ['ffmpeg', '-y',
                              '-f', 'wav', '-i', self.wav_file.name,
                              *self.quality_args,
                              '-f', self.codec_fmt, self.target_file.name]
        subprocess.run(conversion_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.target_file.seek(0)
        x_data = self.target_file.read()
        return x_data

    def __getstate__(self):
        return self.codec_fmt, self.quality_args

    def __setstate__(self, state):
        self.codec_fmt, self.quality_args = state
        self.target_file = tempfile.NamedTemporaryFile()
        self.wav_file = tempfile.NamedTemporaryFile()
