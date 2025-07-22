import numpy as np
import sounddevice as sd
from config import SAMPLE_RATE, CHANNELS, RECORDING_BLOCK_SIZE, SPEAKING_THRESHOLD, SILENCE_TIMEOUT_SECONDS, INITIAL_LISTENING_TIMEOUT

class AudioInput:
    def __init__(self):
        self.stream = None

    def listen(self, on_speech_start=None, on_speech_end=None):
        print("AudioInput: Ready. Speak to activate.")
        audio_buffer = []
        listening_start_time = None
        is_active_listening = False
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32', blocksize=RECORDING_BLOCK_SIZE) as stream:
            while True:
                chunk, overflowed = stream.read(RECORDING_BLOCK_SIZE)
                rms = np.sqrt(np.mean(chunk**2))
                if not is_active_listening:
                    if rms > SPEAKING_THRESHOLD:
                        print("Speech detected.")
                        if on_speech_start:
                            on_speech_start()
                        is_active_listening = True
                        listening_start_time = stream.time  # always a float
                        audio_buffer = []
                if is_active_listening:
                    audio_buffer.append(chunk)
                    if rms > SPEAKING_THRESHOLD:
                        listening_start_time = stream.time
                    current_time = stream.time
                    if listening_start_time is not None and (
                        (current_time - listening_start_time > SILENCE_TIMEOUT_SECONDS) or
                        (current_time - listening_start_time > INITIAL_LISTENING_TIMEOUT and len(audio_buffer) > 0)
                    ):
                        full_audio_data = np.concatenate(audio_buffer)
                        if on_speech_end:
                            on_speech_end(full_audio_data)
                        audio_buffer = []
                        is_active_listening = False
                        listening_start_time = None

if __name__ == "__main__":
    def _on_start():
        print("[TEST] Speech started!")
    def _on_end(audio):
        print(f"[TEST] Speech ended. Audio shape: {audio.shape}")
    ai = AudioInput()
    ai.listen(_on_start, _on_end)
