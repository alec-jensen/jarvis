import torch
import os

# --- Model Configuration ---
MODEL_PATH = "./models/gemma-3n-e2b-it"
DEVICE = "cuda"
DTYPE = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float16 if DEVICE == "cuda" else torch.float32

# --- Audio Settings ---
SAMPLE_RATE = 16000
CHANNELS = 1
BUFFER_DURATION = 3
RECORDING_BLOCK_SIZE = int(SAMPLE_RATE * 0.1)
SPEAKING_THRESHOLD = 0.005
SILENCE_TIMEOUT_SECONDS = 1.5
INITIAL_LISTENING_TIMEOUT = 5

# --- Piper TTS Configuration ---
PIPER_VOICE_DIR = "piper_voices"
PIPER_MODEL_NAME = "en_GB-northern_english_male-medium.onnx"
PIPER_MODEL_PATH = os.path.join(PIPER_VOICE_DIR, PIPER_MODEL_NAME)
PIPER_CONFIG_PATH = os.path.join(PIPER_VOICE_DIR, PIPER_MODEL_NAME + ".json")
