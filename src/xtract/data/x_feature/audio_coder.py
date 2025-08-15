import subprocess
import tempfile

import numpy
import soundfile


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
