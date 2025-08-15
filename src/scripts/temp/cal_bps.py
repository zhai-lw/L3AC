import dataclasses
import math


@dataclasses.dataclass
class CodecConfig:
    codebook_size: list[int]
    compress_rate: list[int]
    sample_rate: int = 16000

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / math.prod(self.compress_rate)

    @property
    def bpf(self) -> float:
        return math.log2(math.prod(self.codebook_size))

    @property
    def bps(self) -> float:
        return self.bpf * self.frame_rate
        # return math.ceil(self.bpf) * self.frame_rate


def main():
    configs = {
        # '413 bps': CodecConfig(codebook_size=[5, 5, 5, 5, 5, 5], compress_rate=[5, 4, 3, 3, 3]),
        '0.46kbps': CodecConfig(codebook_size=[7, 7, 7, 5, 5, 5], compress_rate=[9, 5, 4, 3]),
        # '500 bps': CodecConfig(codebook_size=[7, 7, 7, 7, 7, 7], compress_rate=[5, 4, 3, 3, 3]),
        '0.75kbps': CodecConfig(codebook_size=[7, 7, 7, 7, 7, 7], compress_rate=[6, 5, 4, 3]),
        '1.0 kbps': CodecConfig(codebook_size=[7, 7, 7, 7, 7, 7], compress_rate=[6, 5, 3, 3]),
        '1.5 kbps': CodecConfig(codebook_size=[7, 7, 7, 7, 7, 7], compress_rate=[6, 5, 3, 2]),  # v1
        # '1490bps': CodecConfig(codebook_size=[7, 7, 5, 5, 5, 5], compress_rate=[8, 5, 4]),  # v2
        # '2k  bps': CodecConfig(codebook_size=[7, 7, 5, 5, 5, 5], compress_rate=[6, 5, 4]),
        # '3.0 kbps': CodecConfig(codebook_size=[7, 7, 7, 7, 7, 7], compress_rate=[6, 5, 3]),
        '3.0 kbps': CodecConfig(codebook_size=[9, 9, 9, 7, 7, 7], compress_rate=[6, 4, 4]),
        '!4k  bps': CodecConfig(codebook_size=[7, 7, 7, 7, 7, 7], compress_rate=[6, 4, 3]),
    }
    for name, config in configs.items():
        print(f"{name}:\t bps: {config.bps:.2f},\t frame_rate: {config.frame_rate:.2f} ")
        print(config)


if __name__ == '__main__':
    main()
