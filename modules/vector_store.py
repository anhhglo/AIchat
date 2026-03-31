# modules/vector_store.py
"""
Pinecone Vector Store - replaces ChromaDB.
Stores document embeddings for RAG (Knowledge Base).
Supports filtering by user_id and file source.
"""

from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from modules.embedding_service import embedding_service
from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
    EMBEDDING_DIMENSION,
)
import hashlib
import time


class VectorStore:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(VectorStore, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        print(f"[VectorStore] Initializing Pinecone index: {PINECONE_INDEX_NAME}...")
        try:
            self.pc = Pinecone(api_key=PINECONE_API_KEY)
            self.index_name = PINECONE_INDEX_NAME
            self.dimension = EMBEDDING_DIMENSION

            # Create index if it doesn't exist
            existing = [idx.name for idx in self.pc.list_indexes()]
            if self.index_name not in existing:
                print(f"[VectorStore] Creating new index '{self.index_name}'...")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
                )
                # Wait for index to be ready
                while not self.pc.describe_index(self.index_name).status["ready"]:
                    print("[VectorStore] Waiting for index to be ready...")
                    time.sleep(2)

            self.index = self.pc.Index(self.index_name)
            stats = self.index.describe_index_stats()
            print(
                f"[VectorStore] ✅ Pinecone connected: index='{self.index_name}' "
                f"vectors={stats.total_vector_count}"
            )
            self._initialized = True

        except Exception as e:
            print(f"[VectorStore] ❌ Error initializing Pinecone: {e}")
            raise

    def _generate_id(self, text: str, metadata: dict) -> str:
        """Generate a deterministic ID for a document chunk."""
        raw = f"{metadata.get('source', '')}_{metadata.get('user_id', '')}_{metadata.get('chunk_number', 0)}_{text[:100]}"
        return hashlib.md5(raw.encode()).hexdigest()

    def add_documents(self, documents: list[Document]) -> None:
        """Add LangChain Documents to Pinecone."""
        if not documents:
            return

        print(f"[VectorStore] Adding {len(documents)} documents to Pinecone...")
        try:
            # Embed all texts at once
            texts = [doc.page_content for doc in documents]
            embeddings = embedding_service.embed_documents(texts)

            # Build upsert vectors
            vectors = []
            for doc, emb in zip(documents, embeddings):
                vec_id = self._generate_id(doc.page_content, doc.metadata)
                metadata = {
                    "text": doc.page_content[:900],  # Pinecone metadata limit
                    "source": doc.metadata.get("source", ""),
                    "user_id": doc.metadata.get("user_id", ""),
                    "chunk_number": doc.metadata.get("chunk_number", 0),
                }
                vectors.append({"id": vec_id, "values": emb, "metadata": metadata})

            # Upsert in batches of 100
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i : i + batch_size]
                self.index.upsert(vectors=batch)

            print(f"[VectorStore] ✅ {len(vectors)} vectors upserted.")

        except Exception as e:
            print(f"[VectorStore] ❌ Error adding documents: {e}")
            raise

    def similarity_search(self, query: str, k: int = 3, filter_dict: dict = None) -> list[dict]:
        """
        Core search method: embed query → Pinecone query → return results.
        Returns list of dicts: [{"page_content": ..., "metadata": {...}}, ...]
        """
        try:
            query_embedding = embedding_service.embed_query(query)

            kwargs = {
                "vector": query_embedding,
                "top_k": k,
                "include_metadata": True,
            }
            if filter_dict:
                kwargs["filter"] = filter_dict

            results = self.index.query(**kwargs)

            docs = []
            for match in results.get("matches", []):
                meta = match.get("metadata", {})
                docs.append({
                    "page_content": meta.get("text", ""),
                    "metadata": {
                        "source": meta.get("source", ""),
                        "user_id": meta.get("user_id", ""),
                        "chunk_number": meta.get("chunk_number", 0),
                        "score": match.get("score", 0),
                    },
                })
            return docs

        except Exception as e:
            print(f"[VectorStore] ❌ Search error: {e}")
            return []

    def search_by_file(self, file_source: str, query: str, k: int = 3) -> list[dict]:
        """Search filtered by file source name."""
        print(f"[VectorStore] Searching by file: {file_source}")
        return self.similarity_search(query, k=k, filter_dict={"source": {"$eq": file_source}})

    def search_by_user(self, thread_id: str, query: str, k: int = 3) -> list[dict]:
        """Search filtered by user/thread ID."""
        print(f"[VectorStore] Searching by user: {thread_id}")
        return self.similarity_search(query, k=k, filter_dict={"user_id": {"$eq": thread_id}})

    def delete_by_user(self, thread_id: str) -> None:
        """Delete all vectors for a specific user."""
        try:
            # Pinecone serverless: delete by metadata filter
            self.index.delete(filter={"user_id": {"$eq": thread_id}})
            print(f"[VectorStore] ✅ Deleted vectors for user: {thread_id}")
        except Exception as e:
            print(f"[VectorStore] ❌ Delete error: {e}")


# Singleton instance
vector_store = VectorStore()
