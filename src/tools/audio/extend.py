from multiprocessing import cpu_count

import pesq as pesq_lib
import pystoi

from . import base

DEFAULT_CPU_NUM = min(cpu_count() // 2, 8)


def resample_audio(desired_sample_rate: int, other_audio_args: tuple[str] = ()):
    def decorator(func):
        def wrap(audio, sample_rate, **kwargs):
            audio = base.resample(audio, original_frame_rate=sample_rate, target_frame_rate=desired_sample_rate)
            for other_arg in other_audio_args & kwargs.keys():
                kwargs[other_arg] = base.resample(kwargs[other_arg], original_frame_rate=sample_rate,
                                                  target_frame_rate=desired_sample_rate)
            return func(audio, sample_rate=desired_sample_rate, **kwargs)

        return wrap

    return decorator


@resample_audio(desired_sample_rate=16000, other_audio_args=('ref_audio',))
def pesq(audio, sample_rate, ref_audio, ):
    return pesq_lib.pesq(fs=16000, mode='wb',
                         ref=ref_audio, deg=audio,
                         on_error=pesq_lib.PesqError.RETURN_VALUES)


@resample_audio(desired_sample_rate=16000, other_audio_args=('ref_audio',))
def pesq_batch(audio, sample_rate, ref_audio, cpu_num=DEFAULT_CPU_NUM):
    pesq_scores = pesq_lib.pesq_batch(fs=16000, mode='wb',
                                      ref=ref_audio, deg=audio,
                                      n_processor=cpu_num,
                                      on_error=pesq_lib.PesqError.RETURN_VALUES)  # return -1 when error
    return pesq_scores


def stoi(audio, sample_rate, ref_audio, ):
    return pystoi.stoi(ref_audio, audio, fs_sig=sample_rate, extended=False)


def stoi_batch(audio, sample_rate, ref_audio, ):
    return [stoi(a, sample_rate, ref_a) for a, ref_a in zip(audio, ref_audio)]
