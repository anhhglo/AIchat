# config.py - Centralized configuration
import os

# === Server Config ===
API_SERVER_HOST = os.getenv("API_SERVER_HOST", "127.0.0.1")
API_SERVER_PORT = int(os.getenv("API_SERVER_PORT", "8000"))
MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8001"))
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", f"http://{MCP_SERVER_HOST}:{MCP_SERVER_PORT}/tools/call")

# === LLM Config ===
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
MAX_NEW_TOKENS = 1024

# === OCR Config ===
OCR_MODEL_ID = os.getenv("OCR_MODEL_ID", "stepfun-ai/GOT-OCR-2.0-hf")

# === Embedding Config ===
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2 outputs 384-dim vectors

# === RAG / Ingestion Config ===
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# === Pinecone Config ===
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "aichat-knowledge")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# === MongoDB Config ===
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "aichat")
MONGO_CHAT_COLLECTION = "chat_history"
MONGO_USERS_COLLECTION = "users"
MONGO_MAX_HISTORY = int(os.getenv("MONGO_MAX_HISTORY", "50"))  # Max messages per thread to load

# === Redis Config ===
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_CACHE_TTL = int(os.getenv("REDIS_CACHE_TTL", "3600"))  # 1 hour default


# === Tavily Config ===
TAVILY_MAX_RESULTS = 5

# === Upload Config ===
UPLOAD_DIR = "data_storage/uploaded_files"
