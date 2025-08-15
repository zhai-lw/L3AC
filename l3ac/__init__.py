import logging
import math
from pathlib import Path

import torch
from pydantic import computed_field, field_validator, ValidationInfo

from .en_codec import ModelConfig
from .en_codec import EnCodec
from l3ac.xtract.config import FileConfig

CONFIG_DIR = Path(__file__).parent / "configs"

log = logging.getLogger("L3AC")


def list_models() -> list[str]:
    return [config_path.relative_to(CONFIG_DIR).stem for config_path in CONFIG_DIR.rglob("*.toml")]


def get_model(config_name):
    codec_config = L3ACConfig(config_file=CONFIG_DIR / f"{config_name}.toml")
    l3ac_codec = L3AC(codec_config)
    l3ac_codec.load_pretrained()
    return l3ac_codec


def get_model_info(model: EnCodec, eval_flops_seconds=10, sample_rate: int = 16000):
    mc = model.mc

    from ptflops import get_model_complexity_info

    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(
            model, input_res=(eval_flops_seconds * sample_rate,),
            as_strings=True, print_per_layer_stat=False)

    compress_rate = math.prod(mc.compress_rates) * mc.en_coder_compress_rate
    codebook_size = math.prod(mc.vq_config['levels'])
    frame_rate = sample_rate / compress_rate
    bps = frame_rate * math.log2(codebook_size)
    receptive_field = mc.en_coder_window_size / frame_rate

    return {
        'macs': macs,
        'params': params,
        'codebook_size': codebook_size,
        'frame_rate': frame_rate,
        'bps': bps,
        'receptive_field': receptive_field,
    }


class L3ACConfig(FileConfig):
    config_file: Path

    model_name: str = "debug"
    sample_rate: int = 16000
    model_version: str = "v0.0"
    model_dir: Path = Path.home() / ".cache" / "l3ac"
    weight_url: str = None

    network_config: ModelConfig = None

    @computed_field
    @property
    def model_tag(self) -> str:
        return f"{self.model_name}.{self.model_version}"

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_tag

    @field_validator('weight_url', mode='before')
    @classmethod
    def init_weight_url(cls, weight_url: str, info: ValidationInfo):
        if weight_url is None:
            weight_url = (f"https://huggingface.co/zhai-lw/L3AC/resolve/main/weights/"
                          f"{info.data['model_name']}.{info.data['model_version']}/"
                          "{}.pt")
        return weight_url


class L3AC:

    def __init__(self, config: L3ACConfig):
        self.config = config
        self.network = EnCodec(config.network_config)

    def download_weights(self):
        import requests
        self.config.model_path.mkdir(parents=True, exist_ok=True)
        for module_name in self.network.trainable_modules:
            weight_url = self.config.weight_url.format(module_name)
            weight_path = self.config.model_path / f"{module_name}.pt"
            if weight_path.exists():
                log.info(f"{module_name}({weight_path}) already exists, skip download")
            else:
                log.warning(f"Downloading {module_name}({weight_url}) to {weight_path}")
                response = requests.get(weight_url)
                response.raise_for_status()
                weight_path.write_bytes(response.content)

    def load_pretrained(self):
        self.download_weights()
        self.network.load_model(model_path=self.config.model_path)

    def encode_audio(self, audio_data: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        audio_data, audio_length = self.network.preprocess(audio_data)
        feature = self.network.encoder(audio_data.unsqueeze(1))
        trans_feature = self.network.en_encoder(feature)

        q_trans_feature, indices, _ = self.network.quantizer(trans_feature)
        return q_trans_feature, indices

    def decode_audio(self, audio_feature: torch.Tensor = None, indices: torch.Tensor = None) -> torch.Tensor:
        if audio_feature is None:
            audio_feature = self.network.quantizer.to_features(indices)
        q_feature = self.network.en_decoder(audio_feature)
        audio_data = self.network.decoder(q_feature).squeeze(1)
        return audio_data
