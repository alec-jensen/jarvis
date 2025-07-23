import os
import json
import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice
from config import PIPER_MODEL_PATH, PIPER_CONFIG_PATH

class PiperTTS:
    def __init__(self):
        if not os.path.exists(PIPER_MODEL_PATH) or not os.path.exists(PIPER_CONFIG_PATH):
            raise FileNotFoundError(f"Piper model files not found: {PIPER_MODEL_PATH}, {PIPER_CONFIG_PATH}")
        self.voice = PiperVoice.load(PIPER_MODEL_PATH, PIPER_CONFIG_PATH)
        with open(PIPER_CONFIG_PATH, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        self.output_stream = sd.OutputStream(
            samplerate=self.config['audio']['sample_rate'],
            channels=1,
            dtype='int16'
        )
        self.output_stream.start()

    def speak(self, text):
        print(f"Jarvis: {text}")
        audio_bytes = b''.join(chunk.audio_int16_bytes for chunk in self.voice.synthesize(text))
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        self.output_stream.write(audio_array)

if __name__ == "__main__":
    tts = PiperTTS()
    tts.speak("This is a test of the Piper TTS system. If you hear this, TTS is working!")
