# modules/llm_service.py
import torch
from transformers import pipeline, BitsAndBytesConfig
from config import LLM_MODEL_ID, MAX_NEW_TOKENS
import os
import asyncio


class LLMService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LLMService, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        print(f"[LLMService] Loading model: {LLM_MODEL_ID}...")

        hf_token = os.getenv("HF_TOKEN")

        try:
            # 4-bit quantization config
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            model_kwargs = {
                "torch_dtype": torch.bfloat16,
                "quantization_config": quantization_config,
                "device_map": "auto",
            }

            self.pipe = pipeline(
                "text-generation",
                model=LLM_MODEL_ID,
                model_kwargs=model_kwargs,
                token=hf_token,
            )

            print(f"[LLMService] ✅ Model loaded (4-bit quantized) on {self.pipe.device}")
            self._initialized = True

        except Exception as e:
            print(f"[LLMService] ⚠️ 4-bit loading failed: {e}")
            try:
                print("[LLMService] Retrying without quantization...")
                self.pipe = pipeline(
                    "text-generation",
                    model=LLM_MODEL_ID,
                    model_kwargs={"device_map": "auto"},
                    token=hf_token,
                )
                print(f"[LLMService] ✅ Model loaded (no quantization) on {self.pipe.device}")
                self._initialized = True
            except Exception as fallback_error:
                print(f"[LLMService] ❌ Fallback also failed: {fallback_error}")
                raise

    def _clean_messages(self, messages: list[dict]) -> list[dict]:
        """
        Clean messages for HuggingFace models:
        - Remove leading assistant messages (model rejects them)
        - Ensure non-empty message list
        """
        # Skip leading assistant messages
        start_index = 0
        while start_index < len(messages) and messages[start_index].get("role") == "assistant":
            start_index += 1

        clean = messages[start_index:]
        return clean if clean else [{"role": "user", "content": "Xin chào"}]

    def generate(self, messages: list[dict], generation_kwargs: dict = None) -> str:
        if generation_kwargs is None:
            generation_kwargs = {}

        if not self._initialized:
            return "Lỗi: Dịch vụ LLM chưa được khởi tạo."

        try:
            clean_messages = self._clean_messages(messages)

            prompt_text = self.pipe.tokenizer.apply_chat_template(
                clean_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Build stop token IDs
            eos_token_id = self.pipe.tokenizer.eos_token_id
            stop_token_ids = [eos_token_id]
            for special in ("<|eot_id|>", "<|im_end|>"):
                token_id = self.pipe.tokenizer.convert_tokens_to_ids(special)
                if token_id != self.pipe.tokenizer.unk_token_id:
                    stop_token_ids.append(token_id)

            # Merge default params with user overrides
            default_kwargs = {
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.95,
                "return_full_text": False,
                "eos_token_id": stop_token_ids,
            }
            default_kwargs.update(generation_kwargs)

            outputs = self.pipe(prompt_text, **default_kwargs)

            if isinstance(outputs, list) and len(outputs) > 0:
                first = outputs[0]
                if isinstance(first, dict) and "generated_text" in first:
                    text = first["generated_text"].strip()

                    # Clean stop tokens from output
                    for token_id in stop_token_ids:
                        token_str = self.pipe.tokenizer.convert_ids_to_tokens([token_id])
                        if token_str and token_str[0]:
                            text = text.replace(token_str[0], "")

                    return text if text else "Tôi không thể tạo phản hồi. Vui lòng thử lại."

            return "Xin lỗi, không có phản hồi được tạo ra."

        except Exception as e:
            print(f"[LLMService] ❌ Generation error: {e}")
            import traceback
            traceback.print_exc()
            return f"Xin lỗi, tôi gặp lỗi khi tạo câu trả lời: {str(e)}"

    async def agenerate(self, messages: list[dict], generation_kwargs: dict = None) -> str:
        if generation_kwargs is None:
            generation_kwargs = {}
        return await asyncio.to_thread(self.generate, messages, generation_kwargs)


# Singleton instance
llm_service = LLMService()
