# mcp_server.py
# MCP Tool Server - Port 8001
# Provides tools via JSON-RPC-style endpoints
# Now with: Pinecone (vector), MongoDB (chat history + users), Redis (cache)

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

from modules.llm_service import llm_service
from modules.ocr_service import ocr_service
from modules.embedding_service import embedding_service
from modules.vector_store import vector_store
from modules.web_search_service import web_search_service
from modules.ingestion_pipeline import ingestion_pipeline
from modules.chat_history import chat_history_service
from modules.cache_service import cache_service

from config import MCP_SERVER_HOST, MCP_SERVER_PORT

app = FastAPI(
    title="AIchat MCP Tool Server",
    description="AI Tools via MCP — Pinecone + MongoDB + Redis",
    version="5.0",
)


class ToolCallRequest(BaseModel):
    tool_name: str
    tool_args: Dict[str, Any] = {}


# ============================================================
# TOOL REGISTRY
# ============================================================
TOOL_REGISTRY = {
    # LLM
    "llm_generate": {
        "description": "Generate text using LLM",
        "args": {"messages": "list[dict]", "generation_kwargs": "dict (optional)"},
    },
    # Web
    "web_search": {"description": "Search the web using Tavily", "args": {"query": "str"}},
    "web_crawl": {"description": "Crawl a URL and extract content", "args": {"url": "str"}},
    # Vector Store (Pinecone)
    "vector_search_by_file": {
        "description": "Search Pinecone filtered by file source",
        "args": {"file_source": "str", "query": "str", "k": "int (optional)"},
    },
    "vector_search_by_user": {
        "description": "Search Pinecone filtered by user/thread ID",
        "args": {"thread_id": "str", "query": "str", "k": "int (optional)"},
    },
    "ingest_file": {
        "description": "Ingest file into Pinecone vector store",
        "args": {"file_path": "str", "file_source": "str", "thread_id": "str"},
    },
    # Chat History (MongoDB)
    "save_message": {
        "description": "Save a chat message to MongoDB",
        "args": {"thread_id": "str", "role": "str", "content": "str"},
    },
    "save_messages": {
        "description": "Save multiple chat messages to MongoDB at once",
        "args": {"thread_id": "str", "messages": "list[dict]"},
    },
    "get_chat_history": {
        "description": "Load chat history from MongoDB",
        "args": {"thread_id": "str", "limit": "int (optional)"},
    },
    "get_recent_messages": {
        "description": "Get N most recent messages for a thread",
        "args": {"thread_id": "str", "n": "int (optional, default=10)"},
    },
    "clear_chat_history": {
        "description": "Clear chat history for a thread",
        "args": {"thread_id": "str"},
    },
    "list_threads": {
        "description": "List all active chat threads",
        "args": {"limit": "int (optional)"},
    },
    # Cache (Redis)
    "cache_get": {
        "description": "Get cached response from Redis",
        "args": {"query": "str", "context_key": "str (optional)"},
    },
    "cache_set": {
        "description": "Cache a response in Redis",
        "args": {"query": "str", "response": "str", "context_key": "str (optional)"},
    },
    "cache_clear": {"description": "Clear all cached responses", "args": {}},
    "cache_stats": {"description": "Get Redis cache statistics", "args": {}},

}


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    return {"service": "AIchat MCP Tool Server", "version": "5.0", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check with all service statuses."""
    return {
        "status": "healthy",
        "service": "MCP Tool Server",
        "tools_available": len(TOOL_REGISTRY),
        "services": {
            "pinecone": vector_store._initialized,
            "mongodb": chat_history_service.is_connected,
            "redis": cache_service.is_connected,
        },
    }


@app.get("/tools/list")
async def list_tools():
    return {"tools": TOOL_REGISTRY}


@app.post("/tools/call")
async def handle_tool_call(request: ToolCallRequest):
    """Main MCP endpoint — dispatch tool calls."""
    print(f"[MCP Server] tool='{request.tool_name}' args={list(request.tool_args.keys())}")

    try:
        match request.tool_name:
            # ==================== LLM ====================
            case "llm_generate":
                messages = request.tool_args.get("messages", [])
                gen_kwargs = request.tool_args.get("generation_kwargs", {})

                # Check Redis cache first
                query_for_cache = messages[-1].get("content", "") if messages else ""
                cached = cache_service.get_cached_response(query_for_cache, context_key="llm")
                if cached:
                    return {"success": True, "result": cached, "cached": True}

                result = await llm_service.agenerate(messages=messages, generation_kwargs=gen_kwargs)

                # Cache the response
                cache_service.set_cached_response(query_for_cache, result, context_key="llm")
                return {"success": True, "result": result, "cached": False}

            # ==================== WEB SEARCH ====================
            case "web_search":
                result = await web_search_service.asearch_tavily(**request.tool_args)
                return {"success": True, "result": result}

            case "web_crawl":
                result = await web_search_service.acrawl_tavily(**request.tool_args)
                return {"success": True, "result": result}

            # ==================== PINECONE VECTOR STORE ====================
            case "vector_search_by_file":
                file_source = request.tool_args.get("file_source")
                query = request.tool_args.get("query", "")
                k = request.tool_args.get("k", 3)
                if not file_source:
                    return {"success": False, "error": "Thiếu tham số 'file_source'"}
                docs = vector_store.search_by_file(file_source=file_source, query=query, k=k)
                return {"success": True, "result": docs}

            # Keep old name for backward compatibility
            case "get_filtered_retriever":
                file_source = request.tool_args.get("file_source")
                query = request.tool_args.get("query", "")
                if not file_source:
                    return {"success": False, "error": "Thiếu tham số 'file_source'"}
                docs = vector_store.search_by_file(file_source=file_source, query=query)
                return {"success": True, "result": docs}

            case "vector_search_by_user":
                thread_id = request.tool_args.get("thread_id")
                query = request.tool_args.get("query", "")
                k = request.tool_args.get("k", 3)
                docs = vector_store.search_by_user(thread_id=thread_id, query=query, k=k)
                return {"success": True, "result": docs}

            # Keep old name for backward compatibility
            case "get_retriever_for_user":
                thread_id = request.tool_args.get("thread_id")
                query = request.tool_args.get("query", "")
                docs = vector_store.search_by_user(thread_id=thread_id, query=query)
                return {"success": True, "result": docs}

            case "ingest_file":
                success = ingestion_pipeline.ingest_file(**request.tool_args)
                return {"success": success}

            # ==================== MONGODB CHAT HISTORY ====================
            case "save_message":
                thread_id = request.tool_args.get("thread_id")
                role = request.tool_args.get("role")
                content = request.tool_args.get("content")
                metadata = request.tool_args.get("metadata", {})
                chat_history_service.add_message(thread_id, role, content, metadata)
                return {"success": True}

            case "save_messages":
                thread_id = request.tool_args.get("thread_id")
                messages = request.tool_args.get("messages", [])
                chat_history_service.add_messages(thread_id, messages)
                return {"success": True}

            case "get_chat_history":
                thread_id = request.tool_args.get("thread_id")
                limit = request.tool_args.get("limit")
                history = chat_history_service.get_history(thread_id, limit)
                return {"success": True, "result": history}

            case "get_recent_messages":
                thread_id = request.tool_args.get("thread_id")
                n = request.tool_args.get("n", 10)
                messages = chat_history_service.get_recent_messages(thread_id, n)
                return {"success": True, "result": messages}

            case "clear_chat_history":
                thread_id = request.tool_args.get("thread_id")
                count = chat_history_service.clear_history(thread_id)
                return {"success": True, "deleted_count": count}

            case "list_threads":
                limit = request.tool_args.get("limit", 50)
                threads = chat_history_service.list_threads(limit)
                return {"success": True, "result": threads}

            # ==================== REDIS CACHE ====================
            case "cache_get":
                query = request.tool_args.get("query", "")
                context_key = request.tool_args.get("context_key", "")
                cached = cache_service.get_cached_response(query, context_key)
                return {"success": True, "result": cached, "hit": cached is not None}

            case "cache_set":
                query = request.tool_args.get("query", "")
                response = request.tool_args.get("response", "")
                context_key = request.tool_args.get("context_key", "")
                ok = cache_service.set_cached_response(query, response, context_key)
                return {"success": ok}

            case "cache_clear":
                count = cache_service.clear_all()
                return {"success": True, "cleared_count": count}

            case "cache_stats":
                stats = cache_service.get_stats()
                return {"success": True, "result": stats}

            case _:
                raise HTTPException(status_code=400, detail=f"Tool '{request.tool_name}' không tồn tại.")

    except HTTPException:
        raise
    except Exception as e:
        print(f"[MCP Server] ❌ Error: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host=MCP_SERVER_HOST, port=MCP_SERVER_PORT)
