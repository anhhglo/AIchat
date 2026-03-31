# modules/chat_history.py
"""
MongoDB Chat History Service.
Provides persistent "short-term memory" for conversations.
Each thread_id has its own message history stored in MongoDB.
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
from datetime import datetime, timezone
from typing import Optional
from config import MONGO_URI, MONGO_DB_NAME, MONGO_CHAT_COLLECTION, MONGO_MAX_HISTORY


class ChatHistoryService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ChatHistoryService, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        print(f"[ChatHistory] Connecting to MongoDB: {MONGO_URI}...")
        try:
            self.client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command("ping")

            self.db = self.client[MONGO_DB_NAME]
            self.collection = self.db[MONGO_CHAT_COLLECTION]

            # Create indexes for fast queries
            self.collection.create_index([("thread_id", ASCENDING), ("timestamp", ASCENDING)])
            self.collection.create_index([("thread_id", ASCENDING)], name="thread_lookup")

            print(f"[ChatHistory] ✅ MongoDB connected: db='{MONGO_DB_NAME}' collection='{MONGO_CHAT_COLLECTION}'")
            self._initialized = True

        except Exception as e:
            print(f"[ChatHistory] ❌ MongoDB connection failed: {e}")
            print("[ChatHistory] ⚠️ Falling back to in-memory mode")
            self.client = None
            self.db = None
            self.collection = None
            self._initialized = True  # Still mark initialized to avoid retry loops

    @property
    def is_connected(self) -> bool:
        return self.collection is not None

    def add_message(self, thread_id: str, role: str, content: str, metadata: dict = None) -> None:
        """Save a single message to MongoDB."""
        if not self.is_connected:
            return

        doc = {
            "thread_id": thread_id,
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc),
            "metadata": metadata or {},
        }

        try:
            self.collection.insert_one(doc)
        except Exception as e:
            print(f"[ChatHistory] ❌ Error saving message: {e}")

    def add_messages(self, thread_id: str, messages: list[dict]) -> None:
        """Save multiple messages to MongoDB at once."""
        if not self.is_connected or not messages:
            return

        docs = [
            {
                "thread_id": thread_id,
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
                "timestamp": datetime.now(timezone.utc),
                "metadata": msg.get("metadata", {}),
            }
            for msg in messages
        ]

        try:
            self.collection.insert_many(docs)
        except Exception as e:
            print(f"[ChatHistory] ❌ Error saving messages: {e}")

    def get_history(self, thread_id: str, limit: int = None) -> list[dict]:
        """
        Load chat history for a thread, sorted by timestamp (oldest first).
        Returns: [{"role": "user", "content": "..."}, ...]
        """
        if not self.is_connected:
            return []

        max_limit = limit or MONGO_MAX_HISTORY

        try:
            cursor = (
                self.collection
                .find({"thread_id": thread_id}, {"_id": 0, "role": 1, "content": 1, "timestamp": 1})
                .sort("timestamp", ASCENDING)
                .limit(max_limit)
            )
            messages = [{"role": doc["role"], "content": doc["content"]} for doc in cursor]
            print(f"[ChatHistory] Loaded {len(messages)} messages for thread: {thread_id}")
            return messages

        except Exception as e:
            print(f"[ChatHistory] ❌ Error loading history: {e}")
            return []

    def get_recent_messages(self, thread_id: str, n: int = 10) -> list[dict]:
        """Get the N most recent messages (useful for context window)."""
        if not self.is_connected:
            return []

        try:
            # Get last N, then reverse to chronological order
            cursor = (
                self.collection
                .find({"thread_id": thread_id}, {"_id": 0, "role": 1, "content": 1})
                .sort("timestamp", DESCENDING)
                .limit(n)
            )
            messages = [{"role": doc["role"], "content": doc["content"]} for doc in cursor]
            messages.reverse()  # Chronological order
            return messages

        except Exception as e:
            print(f"[ChatHistory] ❌ Error loading recent messages: {e}")
            return []

    def clear_history(self, thread_id: str) -> int:
        """Delete all messages for a thread. Returns count deleted."""
        if not self.is_connected:
            return 0

        try:
            result = self.collection.delete_many({"thread_id": thread_id})
            count = result.deleted_count
            print(f"[ChatHistory] ✅ Cleared {count} messages for thread: {thread_id}")
            return count

        except Exception as e:
            print(f"[ChatHistory] ❌ Error clearing history: {e}")
            return 0

    def list_threads(self, limit: int = 50) -> list[dict]:
        """List all active threads with their message count and last activity."""
        if not self.is_connected:
            return []

        try:
            pipeline = [
                {"$group": {
                    "_id": "$thread_id",
                    "message_count": {"$sum": 1},
                    "last_activity": {"$max": "$timestamp"},
                    "first_message": {"$min": "$timestamp"},
                }},
                {"$sort": {"last_activity": -1}},
                {"$limit": limit},
            ]
            results = list(self.collection.aggregate(pipeline))
            return [
                {
                    "thread_id": r["_id"],
                    "message_count": r["message_count"],
                    "last_activity": r["last_activity"].isoformat() if r["last_activity"] else None,
                }
                for r in results
            ]

        except Exception as e:
            print(f"[ChatHistory] ❌ Error listing threads: {e}")
            return []

    def get_thread_summary(self, thread_id: str) -> Optional[dict]:
        """Get summary info for a specific thread."""
        if not self.is_connected:
            return None

        try:
            count = self.collection.count_documents({"thread_id": thread_id})
            if count == 0:
                return None

            first = self.collection.find_one({"thread_id": thread_id}, sort=[("timestamp", ASCENDING)])
            last = self.collection.find_one({"thread_id": thread_id}, sort=[("timestamp", DESCENDING)])

            return {
                "thread_id": thread_id,
                "message_count": count,
                "first_message": first["timestamp"].isoformat() if first else None,
                "last_activity": last["timestamp"].isoformat() if last else None,
            }

        except Exception as e:
            print(f"[ChatHistory] ❌ Error getting thread summary: {e}")
            return None


# Singleton instance
chat_history_service = ChatHistoryService()
