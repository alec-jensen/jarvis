# jarvis
ai voice assistant made with gemma 3 and piper

## setup

```bash
sudo dnf install portaudio-devel

uv sync

mkdir piper_voices
cd piper_voices
python -m piper.download_voices en_GB-northern_english_male-medium

# install torch
uv pip install torch torchvision torchaudio
# cuda support for NVIDIA GPUs (optional)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```