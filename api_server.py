# api_server.py
# API Server (Client) - Port 8000
# Receives requests from Frontend, delegates to LangGraph orchestrator + MCP Server.

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, Field
from typing import Optional, List
import os
import shutil
import traceback
import httpx

from config import (
    MCP_SERVER_URL, UPLOAD_DIR, API_SERVER_HOST, API_SERVER_PORT,
    MCP_SERVER_HOST, MCP_SERVER_PORT
)

# Create upload directory
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Try to import compiled graph
try:
    from agents.orchestrator import compiled_graph
except ImportError as e:
    print(f"[API Server] ❌ Import error: {e}")
    compiled_graph = None


# HTTP client for MCP Server
_mcp_client: httpx.AsyncClient = None


# ============================================================
# MCP HELPER
# ============================================================

async def _call_mcp(tool_name: str, tool_args: dict = None) -> dict:
    """Helper to call MCP Server tools."""
    try:
        resp = await _mcp_client.post(
            MCP_SERVER_URL,
            json={"tool_name": tool_name, "tool_args": tool_args or {}},
        )
        return resp.json()
    except Exception as e:
        print(f"[API] ⚠️ MCP call '{tool_name}' failed: {e}")
        return {"success": False, "error": str(e)}


async def _save_message_to_mongo(thread_id: str, role: str, content: str):
    """Helper: save a single message to MongoDB via MCP."""
    await _call_mcp("save_message", {
        "thread_id": thread_id,
        "role": role,
        "content": content,
    })


# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage httpx client lifecycle."""
    global _mcp_client
    _mcp_client = httpx.AsyncClient(timeout=300.0)
    print("[API Server] ✅ HTTP client initialized")
    yield
    await _mcp_client.aclose()
    print("[API Server] 🔒 HTTP client closed")


app = FastAPI(
    title="AIchat API Server",
    version="6.0",
    description="API Server — LangGraph + Pinecone + MongoDB + Redis",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to the specific frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SlowAPI Rate Limiter setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ============================================================
# REQUEST MODELS
# ============================================================

class ChatRequest(BaseModel):
    query: str
    use_search: bool = False
    use_ocr: bool = False
    target_url: Optional[str] = None
    history: List[dict] = Field(default_factory=list)
    thread_id: Optional[str] = None


# ============================================================
# HELPERS
# ============================================================

def build_initial_state(
    messages: list,
    thread_id: str,
    use_search: bool = False,
    use_ocr: bool = False,
    target_url: Optional[str] = None,
    uploaded_file_path: Optional[str] = None,
) -> dict:
    """Build initial state dict for LangGraph."""
    return {
        "messages": messages,
        "thread_id": thread_id,
        "use_search": use_search,
        "use_ocr": use_ocr,
        "target_url": target_url,
        "uploaded_file_path": uploaded_file_path,
        "intermediate_context": None,
        "error": None,
        "retry_count": 0,
        "next_node": None,
    }


def extract_response(result: dict) -> str:
    """Extract last assistant message from graph result."""
    final_messages = result.get("messages", [])
    if final_messages and len(final_messages) > 0:
        last = final_messages[-1]
        content = last.get("content", "")
        if content and content.strip():
            return content
    return "Xin chào! Tôi là trợ lý AI. Tôi có thể giúp gì cho bạn?"


# ============================================================
# PUBLIC ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    return {"message": "AIchat API is running", "version": "6.0", "auth": "not required"}


@app.get("/health")
async def health_check():
    graph_status = "ready" if compiled_graph is not None else "error"

    mcp_data = {}
    try:
        resp = await _mcp_client.get(f"http://{MCP_SERVER_HOST}:{MCP_SERVER_PORT}/health", timeout=5.0)
        if resp.status_code == 200:
            mcp_data = resp.json()
    except Exception:
        mcp_data = {"status": "unreachable"}

    return {
        "status": "healthy",
        "service": "AIchat API Server",
        "graph_status": graph_status,
        "mcp_server": mcp_data,
    }

# ============================================================
# PROTECTED ENDPOINTS — CHAT
# ============================================================

@app.post("/chat")
@limiter.limit("10/minute")
async def chat_endpoint(request: Request, chat_data: ChatRequest):
    """Main chat endpoint"""
    # Sử dụng chat_data thay cho request cũ để lấy thông tin từ Body JSON
    thread_id = chat_data.thread_id or f"thread_{os.urandom(4).hex()}"

    print(f"[API] Chat: '{chat_data.query[:50]}...' (search={chat_data.use_search})")

    if compiled_graph is None:
        return JSONResponse(
            status_code=500,
            content={"error": "LangGraph chưa được khởi tạo", "success": False},
        )

    try:
        # 1. Load history từ MongoDB thông qua MCP
        if not chat_data.history:
            history_resp = await _call_mcp("get_recent_messages", {
                "thread_id": thread_id, "n": 20
            })
            stored_history = history_resp.get("result", []) if history_resp.get("success") else []
        else:
            stored_history = chat_data.history

        messages = stored_history + [{"role": "user", "content": chat_data.query}]

        # 2. Kiểm tra Redis cache thông qua MCP
        cache_resp = await _call_mcp("cache_get", {
            "query": chat_data.query,
            "context_key": f"chat_{thread_id}",
        })
        
        if cache_resp.get("success") and cache_resp.get("hit"):
            cached_response = cache_resp["result"]
            await _save_message_to_mongo(thread_id, "user", chat_data.query)
            await _save_message_to_mongo(thread_id, "assistant", cached_response)
            return {
                "response": cached_response,
                "thread_id": thread_id,
                "success": True,
                "cached": True,
            }

        # 3. Chạy LangGraph Orchestrator
        config = {"configurable": {"thread_id": thread_id}}
        state = build_initial_state(
            messages=messages,
            thread_id=thread_id,
            use_search=chat_data.use_search,
            use_ocr=chat_data.use_ocr,
            target_url=chat_data.target_url,
        )

        result = await compiled_graph.ainvoke(state, config)
        response_content = extract_response(result)

        # 4. Lưu hội thoại vào MongoDB
        await _save_message_to_mongo(thread_id, "user", chat_data.query)
        await _save_message_to_mongo(thread_id, "assistant", response_content)

        # 5. Lưu vào Redis cache cho lần sau
        await _call_mcp("cache_set", {
            "query": chat_data.query,
            "response": response_content,
            "context_key": f"chat_{thread_id}",
        })

        return {
            "response": response_content,
            "thread_id": thread_id,
            "success": True,
            "cached": False,
        }

    except Exception as e:
        print(f"[API] ❌ Chat error: {e}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Internal server error: {str(e)}",
                "success": False,
                "fallback_response": "Xin lỗi, tôi gặp sự cố kỹ thuật. Vui lòng thử lại sau.",
            },
        )

@app.post("/chat_file")
@limiter.limit("10/minute")
async def chat_file_endpoint(
    request: Request,
    file: UploadFile = File(...),
    query: str = Form(...),
    thread_id: Optional[str] = Form(None),
    use_search: bool = Form(False),
):
    """File upload + chat"""
    # Reject large files (> 5MB)
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    MAX_FILE_SIZE = 5 * 1024 * 1024 # 5MB
    if file_size > MAX_FILE_SIZE:
        return JSONResponse(status_code=413, content={"error": "File quá lớn (Tối đa 5MB)", "success": False})
    thread_id = thread_id or f"thread_{os.urandom(4).hex()}"

    print(f"[API] File upload: file={file.filename}")

    if compiled_graph is None:
        return JSONResponse(
            status_code=500,
            content={"error": "LangGraph chưa được khởi tạo", "success": False},
        )

    saved_file_path = None
    try:
        safe_filename = os.path.basename(file.filename or "uploaded_file")
        saved_file_path = os.path.join(UPLOAD_DIR, safe_filename)
        with open(saved_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Ingest file via MCP → Pinecone
        ingest_resp = await _call_mcp("ingest_file", {
            "file_path": saved_file_path,
            "file_source": safe_filename,
            "thread_id": thread_id,
        })
        if not ingest_resp.get("success"):
            print(f"[API] ⚠️ Ingestion failed: {ingest_resp.get('error')}")

        # Run LangGraph
        messages = [{"role": "user", "content": query}]
        config = {"configurable": {"thread_id": thread_id}}

        state = build_initial_state(
            messages=messages,
            thread_id=thread_id,
            use_search=use_search,
            use_ocr=True,
            uploaded_file_path=saved_file_path,
        )

        result = await compiled_graph.ainvoke(state, config)
        response_content = extract_response(result)

        # Save to MongoDB
        await _save_message_to_mongo(thread_id, "user", f"[File: {safe_filename}] {query}")
        await _save_message_to_mongo(thread_id, "assistant", response_content)

        return {"response": response_content, "thread_id": thread_id, "success": True}

    except Exception as e:
        print(f"[API] ❌ File chat error: {e}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}", "success": False},
        )

@app.get("/history/threads")
async def get_all_threads(limit: int = 50):
    """List all recent threads"""
    resp = await _call_mcp("list_threads", {"limit": limit})
    if resp.get("success"):
        return {"threads": resp["result"], "success": True}
    return JSONResponse(status_code=500, content={"error": resp.get("error"), "success": False})

@app.get("/history/{thread_id}")
async def get_my_history(thread_id: str, limit: int = 50):
    """Load chat history"""
    resp = await _call_mcp("get_chat_history", {"thread_id": thread_id, "limit": limit})
    if resp.get("success"):
        return {"thread_id": thread_id, "messages": resp["result"], "success": True}
    return JSONResponse(status_code=500, content={"error": resp.get("error"), "success": False})


@app.delete("/history/{thread_id}")
async def delete_my_history(thread_id: str):
    """Clear chat history"""
    resp = await _call_mcp("clear_chat_history", {"thread_id": thread_id})
    return {"success": resp.get("success"), "deleted_count": resp.get("deleted_count", 0)}


# ============================================================
# PROTECTED ENDPOINTS — CACHE
# ============================================================

@app.get("/cache/stats")
async def cache_stats():
    """Get Redis cache statistics."""
    resp = await _call_mcp("cache_stats", {})
    if resp.get("success"):
        return {"stats": resp["result"], "success": True}
    return JSONResponse(status_code=500, content={"error": resp.get("error"), "success": False})


@app.delete("/cache")
async def clear_cache():
    """Clear all cached responses."""
    resp = await _call_mcp("cache_clear", {})
    return {"success": resp.get("success"), "cleared_count": resp.get("cleared_count", 0)}
