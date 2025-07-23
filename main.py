from asr import AudioInput
from gemma_llm import GemmaLLM
from tts import PiperTTS

# --- Main Orchestration ---

def main():
    print("Loading models")
    tts = PiperTTS()
    llm = GemmaLLM()
    audio_input = AudioInput()
    print("Ready")

    # memory: last 10 turns
    chat_history = []

    def on_speech_start():
        pass

    def on_speech_end(audio_data):
        print(f"Processing {len(audio_data)/16000:.2f} seconds of audio...")
        chat_history.append({"role": "user", "content": "[audio input]"})
        chat_history[:] = chat_history[-10:]
        response = llm.process_chat_history(chat_history, max_tokens=100)
        print("Generating audio")
        if response:
            tts.speak(response)
            chat_history.append({"role": "assistant", "content": response})
            chat_history[:] = chat_history[-10:]
        else:
            pass
            # tts.speak("I'm sorry, I couldn't generate a clear response. Can I help with something else?")
        print("\nListening now...")

    audio_input.listen(on_speech_start, on_speech_end)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutting down...")