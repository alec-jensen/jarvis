
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from config import MODEL_ID, DTYPE, DEVICE, SAMPLE_RATE


class GemmaLLM:
    def __init__(self, model_id=MODEL_ID, use_flash_attention=True):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.model_id = model_id
        import datetime
        now = datetime.datetime.now().strftime('%A, %B %d, %Y %I:%M %p')
        self.system_message = (
            f"Today is {now}. "
            "You are Jarvis, a friendly, enthusiastic, and concise AI voice assistant for the Martin High School Computer Science Club. "
            "The club officers are Alec Jensen (creator of Jarvis), Ryan Farnell, Bennett Rodriguez, and Tuyet Do. "
            "The club sponsor is James Hovey, the best computer science teacher in all of Texas. "
            "You help with any requests, spark interest in computer science, coding, AI, and club activities, and answer questions about the club. "
            "We run lots of lunch and learns where you can come in during lunch and learn a new topic. "
            "At the end of the year, we host a hackathon with tons of cool prizes and fun activities, and we try to build up knowledge for it throughout the year. "
            "Club members get exclusive early access to hackathon signups. "
            "You can also assist with random or fun requests to keep conversations engaging and interesting. "
            "Keep your answers to 1-2 sentences. Focus on our activities when relevant, but you may answer off-topic questions in a helpful, friendly way. "
            "Emphasize that all skill levels are welcome. If you don't know the answer, say 'I don't know' or 'I'm not sure'. "
            "You must not provide information that could be harmful. "
            "You must respond in plaintext English, without any emojis, markdown, code blocks, or special formatting."
        )

        from transformers.utils.quantization_config import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id, padding_side="left", use_fast=True)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=DTYPE,
            device_map="auto",
            attn_implementation="sdpa",
            quantization_config=quantization_config
        )
        # cache_implementation assignment removed (not supported)
        # if torch.cuda.is_available():
        #     self.model.forward = torch.compile(
        #         self.model.forward, mode="reduce-overhead", fullgraph=True
        #     )

    def process_audio(self, audio_data, max_tokens=100, prompt_text=None):
        """
        Process raw audio (numpy array) with optional prompt_text using Gemma 3 multimodal model.
        """
        if audio_data is None or (hasattr(audio_data, 'size') and audio_data.size == 0):
            return ""
        # Compose chat message for audio input
        user_content = []
        user_content.append({"type": "audio", "audio": audio_data, "sampling_rate": SAMPLE_RATE})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_message}]},
            {"role": "user", "content": user_content},
        ]
        # Prepare model inputs
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                cache_implementation="static"
            )
        response = self.processor.decode(outputs[0], skip_special_tokens=True)

        if "\n" in response:
            lines = [line.strip() for line in response.split("\n") if line.strip()]
            if lines:
                return lines[-1]
        return response.strip()

    def process_text(self, text, max_tokens=100):
        """
        Process text input using Gemma 3 multimodal model.
        """
        if not text or not isinstance(text, str):
            return ""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_message}]},
            {"role": "user", "content": [{"type": "text", "text": text}]},
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                cache_implementation="static"
            )
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        return response.strip()

    def process_chat_history(self, chat_history, max_tokens=100):
        """
        Process chat history (list of dicts) using Gemma 3 multimodal model.
        """
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_message}]}
        ]
        for msg in chat_history:
            if msg["role"] == "user":
                messages.append({"role": "user", "content": [{"type": "text", "text": msg["content"]}]})
            elif msg["role"] == "assistant":
                messages.append({"role": "assistant", "content": [{"type": "text", "text": msg["content"]}]})
        # Add generation prompt for assistant
        if not messages or messages[-1]["role"] != "assistant":
            # Add empty assistant message to trigger generation
            messages.append({"role": "assistant", "content": []})
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                cache_implementation="static"
            )
        response = self.processor.decode(outputs[0], skip_special_tokens=True)

        if "\n" in response:
            lines = [line.strip() for line in response.split("\n") if line.strip()]
            if lines:
                return lines[-1]
        return response.strip()


if __name__ == "__main__":
    print("Gemma LLM test: Interactive chat with memory (type 'exit' to quit)")
    llm = GemmaLLM()
    chat_history = [{"role": "system", "content": llm.system_message}]
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting chat.")
            break
        chat_history.append({"role": "user", "content": user_input})
        response = llm.process_chat_history(chat_history, max_tokens=100)
        print(f"Jarvis: {response}")
        chat_history.append({"role": "assistant", "content": response})
