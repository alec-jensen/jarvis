
from config import MODEL_ID  # Use this for GGUF model path

try:
    from llama_cpp import Llama
except ImportError:
    raise ImportError("llama-cpp-python is not installed. Please install it with CUDA support for best performance.")



class GemmaLLM:
    def __init__(self, model_path=None, n_gpu_layers=-1, chat_format="chatml"):
        self.model_path = model_path or MODEL_ID  # MODEL_ID should be the GGUF file path
        self.n_gpu_layers = n_gpu_layers
        self.chat_format = chat_format
        print(f"Loading Gemma 3n GGUF model: {self.model_path} (n_gpu_layers={self.n_gpu_layers})...")
        self.llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            chat_format=self.chat_format
        )
        print("Gemma 3n GGUF model loaded.")

        self.system_message = (
            "You are Jarvis, a friendly, enthusiastic, and concise AI voice assistant for the Martin High School Computer Science Club. "
            "Your main goal is to get students excited about joining the club. Keep your answers to 1-2 sentences. Focus on our activities: "
            "coding, AI, and hackathons. Emphasize that all skill levels are welcome. If a question is off-topic, you may still "
            "respond, but keep it brief and friendly. If you don't know the answer, say 'I don't know' or 'I'm not sure'. "
            "You must respond in plaintext English, without any emojis, markdown, code blocks, or special formatting."
        )


    def process_text(self, text, max_tokens=100):
        if not text or not isinstance(text, str):
            return ""
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": text},
        ]
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens
        )
        return response["choices"][0]["message"]["content"].strip()


    def process_chat_history(self, chat_history, max_tokens=100):
        # chat_history: list of {"role": ..., "content": ...} dicts
        response = self.llm.create_chat_completion(
            messages=chat_history,
            max_tokens=max_tokens
        )
        return response["choices"][0]["message"]["content"].strip()



if __name__ == "__main__":
    print("Gemma LLM test: Interactive chat with memory (type 'exit' to quit)")
    llm = GemmaLLM()
    chat_history = [
        {"role": "system", "content": llm.system_message}
    ]
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting chat.")
            break
        chat_history.append({"role": "user", "content": user_input})
        response = llm.process_chat_history(chat_history, max_tokens=100)
        print(f"Jarvis: {response}")
        chat_history.append({"role": "assistant", "content": response})
