# modules/embedding_service.py
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_ID


class EmbeddingService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(EmbeddingService, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = EMBEDDING_MODEL_ID):
        if self._initialized:
            return
        print(f"[EmbeddingService] Loading model: {model_name}...")
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            print("[EmbeddingService] ✅ Model loaded successfully.")
            self._initialized = True
        except Exception as e:
            print(f"[EmbeddingService] ❌ Error loading model: {e}")
            raise

    def get_model(self):
        return self.embedding_model


# Singleton: export the model directly for LangChain compatibility
embedding_service = EmbeddingService().get_model()
