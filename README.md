# L3AC

This repository contains the implementation of L3AC, a lightweight audio codec based on a single quantizer,
introduced in the paper titled "L3AC: Towards a Lightweight and Lossless Audio Codec".

[Paper](https://arxiv.org/abs/2504.04949)

[Model Weights](https://huggingface.co/zhai-lw/L3AC)


<figure class="image">
  <img src="./bubble_chart.svg" alt="Comparison of various audio codec">
  <figcaption>Comparison of various audio codec</figcaption>
</figure>

## install

```
pip install l3ac
```

### demo

Firstly, make sure you have installed the librosa package to load the example audio file. You can install it using pip:

```
pip install librosa
```

Then, you can use the following code to load a sample audio file, encode it using the L3AC model, and decode it back
to audio. The code also calculates the mean squared error (MSE) between the original and generated audio.

```python
import librosa
import torch
import l3ac

all_models = l3ac.list_models()
print(f"Available models: {all_models}")

MODEL_USED = '1kbps'
codec = l3ac.get_model(MODEL_USED)
print(f"loaded codec({MODEL_USED}) and codec sample rate: {codec.config.sample_rate}")

sample_audio, sample_rate = librosa.load(librosa.example("libri1"))
sample_audio = sample_audio[None, :]
print(f"loaded sample audio and audio sample_rate :{sample_rate}")

sample_audio = librosa.resample(sample_audio, orig_sr=sample_rate, target_sr=codec.config.sample_rate)

codec.network.cuda()
codec.network.eval()
with torch.inference_mode():
    audio_in = torch.tensor(sample_audio, dtype=torch.float32, device='cuda')
    _, audio_length = audio_in.shape
    print(f"{audio_in.shape=}")
    q_feature, indices = codec.encode_audio(audio_in)
    audio_out = codec.decode_audio(q_feature)  # or
    # audio_out = codec.decode_audio(indices=indices['indices'])
    generated_audio = audio_out[:, :audio_length].detach().cpu().numpy()

mse = ((sample_audio - generated_audio) ** 2).mean().item()
print(f"codec({MODEL_USED}) mse: {mse}")
```

### available models

| config_name | Sample rate(Hz) | tokens/s | Codebook size | Bitrate(bps) |
|-------------|-----------------|----------|---------------|--------------|
| 0k75bps     | 16,000          | 44.44    | 117,649       | 748.6        |
| 1kbps       | 16,000          | 59.26    | 117,649       | 998.2        |
| 1k5bps      | 16,000          | 88.89    | 117,649       | 1497.3       |
| 3kbps       | 16,000          | 166.67   | 250,047       | 2988.6       |
