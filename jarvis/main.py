
from jarvis.asr import AudioInput
from jarvis.gemma_llm import GemmaLLM
from jarvis.tts import PiperTTS

# --- Main Orchestration ---
def main():
    print("AI Jarvis: Ready! Say 'Hey Byte!' to begin.")
    print("Press Ctrl+C to stop.")
    tts = PiperTTS()
    llm = GemmaLLM()
    audio_input = AudioInput()

    def on_speech_start():
        tts.speak("Hello! What would you like to know about the Computer Science Club?")

    def on_speech_end(audio_data):
        print(f"Processing {len(audio_data)/16000:.2f} seconds of audio...")
        response = llm.process_audio(audio_data)
        if response:
            tts.speak(response)
        else:
            tts.speak("I'm sorry, I couldn't generate a clear response. Can I help with something else?")
        print("\nAI Jarvis: Ready! Say 'Hey Byte!' to begin.")

    audio_input.listen(on_speech_start, on_speech_end)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting Jarvis. Goodbye!")