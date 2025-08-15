import librosa
import torch

import l3ac


def example():
    print(l3ac.list_models())
    codec = l3ac.get_model("1kbps")
    print(f"loaded codec and codec sample rate: {codec.config.sample_rate}")
    print("model info: ", l3ac.get_model_info(codec.network))

    sample_audio, sample_rate = librosa.load(librosa.example("libri1"))
    sample_audio = sample_audio[None, :]
    print(f"loaded sample audio and audio sample_rate :{sample_rate}")

    sample_audio = librosa.resample(sample_audio, orig_sr=sample_rate, target_sr=codec.config.sample_rate)

    codec.network.to(device="cuda")
    codec.network.eval()
    with torch.inference_mode():
        audio_in = torch.tensor(sample_audio, dtype=torch.float32, device="cuda")
        _, audio_length = audio_in.shape
        print(f"{audio_in.shape=}")
        q_feature, indices = codec.encode_audio(audio_in)
        audio_out = codec.decode_audio(q_feature)  # or
        # audio_out = codec.decode_audio(indices=indices['indices'])
        generated_audio = audio_out[:, :audio_length].detach().cpu().numpy()

    print(f"MSE: {((sample_audio - generated_audio) ** 2).mean()}")


if __name__ == '__main__':
    example()
