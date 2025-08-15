## env

```shell
conda create -n l3ac cuda=12.6 python=3.13 -c nvidia
conda activate l3ac

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install accelerate datasets einops pynvml tensorboard
pip install pydantic-settings lz4 bidict
pip install scipy seaborn rich

pip install local-attention

pip install soundfile librosa
pip install openai-whisper pesq pystoi jiwer ptflops

# dac related
pip install git+https://github.com/carlthome/audiotools.git@upgrade-dependencies
pip install descript-audio-codec

```

## data prepare

see scripts in ./src/prepare/data_process

## training model

```shell
accelerate launch --num_processes=1 $(pwd)/src/main.py --config 1kbps

```