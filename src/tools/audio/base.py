import functools

import librosa
import numpy


def resample(audio_data: numpy.ndarray[float], original_frame_rate: int, target_frame_rate: int):
    # audio_data: (audio_len) or (num, audio_len)
    return librosa.resample(audio_data, orig_sr=original_frame_rate, target_sr=target_frame_rate, res_type="soxr_vhq")


def get_spl(audio: numpy.ndarray[float]):
    pa = numpy.sqrt(numpy.sum(numpy.power(audio, 2)) / len(audio))
    spl = 20 * numpy.log10(pa / 2) + 100
    return spl


def get_spl_per_second(audio: numpy.ndarray[float], frame_rate=16000):
    """

    @param audio: numpy.ndarray[float]
    @param frame_rate: int
    @return: list
    """
    sql = [get_spl(audio[i * frame_rate: (i + 1) * frame_rate]) for i in range((len(audio)) // frame_rate)]
    return sql


def adjust_volume(audio: numpy.ndarray[float], volume, m=0.026):
    """

    @param audio: numpy.ndarray[float]
    @param volume: db
    @param m: max
    """
    audio2change = audio.copy()
    # first change
    pa = numpy.sqrt(numpy.sum(audio ** 2) / len(audio))
    change_rate = 10 ** (volume / 20 - 5) * 2 / pa
    max_value = m * change_rate if m * change_rate < 1. else 1.
    audio2change[audio2change > max_value] = max_value
    audio2change[audio2change < -max_value] = -max_value
    # second change
    pa = numpy.sqrt(numpy.sum(numpy.power(audio2change, 2)) / len(audio2change))
    change_rate = 10 ** (volume / 20 - 5) * 2 / pa
    audio2change *= change_rate
    return audio2change.astype(audio.dtype)


def snr(clean_signal: numpy.ndarray[float], noise: numpy.ndarray[float]):
    return 10 * numpy.log10(numpy.mean(clean_signal ** 2) / numpy.mean(noise ** 2))


@functools.lru_cache
def max_sample(sample_width: int = 2):
    return int((1 << (sample_width * 8 - 1)) - 1)
