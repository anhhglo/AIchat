# agents/orchestrator.py
"""
LangGraph Orchestrator (MCP Client version).
Routes queries through MCP Server tools using an LLM-powered Router Agent.
"""

from langgraph.graph import StateGraph, END, START
from agents.state import AgentState
import asyncio
import os
import re
import datetime
import httpx

from config import MCP_SERVER_URL

# Shared HTTP client (with proper timeout)
_client = httpx.AsyncClient(timeout=300.0)


# ============================================================
# 1. HELPER FUNCTIONS
# ============================================================

def get_last_user_query(messages: list) -> str:
    """Find the most recent user message to avoid grabbing assistant text."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return messages[-1].get("content", "") if messages else ""


def create_master_prompt(query: str, context: str) -> str:
    """
    Universal Master Prompt v10.0
    Handles: OCR cleaning → Intent classification → Execution → Formatting.
    """
    current_date = datetime.datetime.now().strftime("%d/%m/%Y")

    return f"""Bạn là một Trợ lý AI Chuyên gia Đa năng (Expert AI Assistant).
Ngày hôm nay: {current_date}

### 1. DỮ LIỆU ĐẦU VÀO (RAW DATA SOURCE):
⚠️ Dữ liệu dưới đây được trích xuất tự động (OCR/Web) nên có thể chứa lỗi.
--------------------------------------------------
{context}
--------------------------------------------------

### 2. YÊU CẦU CỦA NGƯỜI DÙNG:
"{query}"

### 3. QUY TRÌNH XỬ LÝ:

**BƯỚC 1: SÀNG LỌC & LÀM SẠCH**
- Bỏ qua ký tự rác, chuỗi vô nghĩa, đoạn lặp lại.
- Sửa lỗi chính tả OCR (VD: "Xä höi" → "Xã hội").
- Chỉ giữ câu văn có ý nghĩa.

**BƯỚC 2: THỰC THI NHIỆM VỤ**
- BẢNG: Quét từng dòng, tìm chính xác dữ liệu.
- DỊCH: Chỉ dịch câu có nghĩa sang Tiếng Việt. KHÔNG dịch rác.
- TÓM TẮT: Tổng hợp ý chính từ dữ liệu sạch.
- THÔNG TIN CỤ THỂ: Trích xuất câu trả lời trực tiếp.

**BƯỚC 3: ĐỊNH DẠNG**
- Trả lời bằng TIẾNG VIỆT (trừ khi yêu cầu khác).
- KHÔNG in quá trình suy luận.
- Luôn ghi [Nguồn: ...] ở cuối.

**CÂU TRẢ LỜI CỦA BẠN (NGẮN GỌN, CHÍNH XÁC):**
"""


async def call_mcp_tool(tool_name: str, tool_args: dict = None) -> dict:
    """Call MCP Server with retry logic."""
    if tool_args is None:
        tool_args = {}

    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            response = await _client.post(
                MCP_SERVER_URL,
                json={"tool_name": tool_name, "tool_args": tool_args},
            )
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError:
            if attempt < max_retries:
                wait = (attempt + 1) * 2
                print(f"[MCP Client] ⚠️ Connection failed, retrying in {wait}s... ({attempt+1}/{max_retries})")
                await asyncio.sleep(wait)
            else:
                print(f"[MCP Client] ❌ MCP Server unreachable after {max_retries} retries")
                return {"success": False, "error": "MCP Server không phản hồi"}
        except httpx.HTTPStatusError as e:
            print(f"[MCP Client] ❌ HTTP error calling {tool_name}: {e.response.status_code}")
            return {"success": False, "error": f"HTTP Error: {e.response.status_code}"}
        except Exception as e:
            print(f"[MCP Client] ❌ Error calling {tool_name}: {e}")
            return {"success": False, "error": str(e)}


# ============================================================
# 2. CACHE HELPER
# ============================================================

async def check_cache(query: str, context_key: str) -> str | None:
    """Check Redis cache via MCP. Returns cached response or None."""
    resp = await call_mcp_tool("cache_get", {"query": query, "context_key": context_key})
    if resp.get("success") and resp.get("hit"):
        print(f"[Cache] ✅ Orchestrator-level HIT ({context_key}): '{query[:40]}...'")
        return resp.get("result")
    return None


async def save_cache(query: str, response: str, context_key: str) -> None:
    """Save response to Redis cache via MCP."""
    await call_mcp_tool("cache_set", {
        "query": query, "response": response, "context_key": context_key,
    })


# ============================================================
# 3. TOOL NODES
# ============================================================

async def node_simple_chat(state: AgentState) -> dict:
    """Simple chat: direct LLM generation without context."""
    print("--- Node: Simple Chat ---")
    messages = state["messages"]
    query = get_last_user_query(messages)

    # Check cache first
    cached = await check_cache(query, "simple_chat")
    if cached:
        return {"messages": messages + [{"role": "assistant", "content": cached}]}

    result = await call_mcp_tool("llm_generate", {"messages": messages})

    if result.get("success"):
        reply = result["result"]
        # Cache the response
        await save_cache(query, reply, "simple_chat")
    else:
        reply = f"Xin lỗi, tôi gặp lỗi: {result.get('error', 'Unknown')}"

    return {"messages": messages + [{"role": "assistant", "content": reply}]}


async def node_local_rag(state: AgentState) -> dict:
    """Local RAG: search uploaded file + optional web search, then generate."""
    print("--- Node: Local RAG ---")
    messages = state["messages"]
    query = get_last_user_query(messages)
    file_path = state.get("uploaded_file_path")
    use_search = state.get("use_search", False)

    if not file_path:
        return {"messages": messages + [{"role": "assistant", "content": "Lỗi: Không tìm thấy file đã upload."}]}

    file_name = os.path.basename(file_path)

    # 1. Search in uploaded file
    print(f"[Local RAG] Querying vector store for file: {file_name}")
    rag_resp = await call_mcp_tool("get_filtered_retriever", {
        "file_source": file_name, "query": query
    })

    if rag_resp.get("success"):
        docs = rag_resp.get("result", [])
        file_context = (
            "\n\n".join(f"Trích đoạn [File: {file_name}]:\n{d.get('page_content')}" for d in docs)
            if docs else f"(Không tìm thấy thông tin trong file {file_name})"
        )
    else:
        file_context = f"(Lỗi đọc file: {rag_resp.get('error')})"

    # 2. Optional web search
    web_context = ""
    if use_search:
        print(f"[Local RAG] Web search fallback: {query}")
        search_resp = await call_mcp_tool("web_search", {"query": query})
        if search_resp.get("success"):
            web_context = "\n\n".join(
                f"[Web: {r.get('url')}]:\n{r.get('content')}"
                for r in search_resp.get("result", []) if r.get("content")
            )

    # 3. Build full context and generate
    full_context = f"""=== THÔNG TIN TỪ FILE ===
{file_context}

=== THÔNG TIN TỪ INTERNET ===
{web_context or "(Không sử dụng tìm kiếm web)"}"""

    prompt = create_master_prompt(query, full_context)
    rag_messages = messages[:-1] + [{"role": "user", "content": prompt}]

    # Check cache for this file + query combo
    cache_key_ctx = f"local_rag:{file_name}"
    cached = await check_cache(query, cache_key_ctx)
    if cached:
        return {"messages": messages + [{"role": "assistant", "content": cached}]}

    llm_resp = await call_mcp_tool("llm_generate", {
        "messages": rag_messages,
        "generation_kwargs": {"temperature": 0.1, "repetition_penalty": 1.1},
    })
    reply = llm_resp.get("result", "Lỗi sinh văn bản RAG.")

    # Cache the response
    await save_cache(query, reply, cache_key_ctx)

    return {"messages": messages + [{"role": "assistant", "content": reply}]}


async def node_global_rag(state: AgentState) -> dict:
    """Global KB RAG: search all user documents, fallback to web if empty."""
    print("--- Node: Global RAG ---")
    messages = state["messages"]
    query = get_last_user_query(messages)
    thread_id = state["thread_id"]

    print(f"[Global RAG] Querying for user: {thread_id}")
    resp = await call_mcp_tool("get_retriever_for_user", {
        "thread_id": thread_id, "query": query
    })

    if not resp.get("success"):
        return {"messages": messages + [{"role": "assistant", "content": f"Lỗi RAG: {resp.get('error')}"}]}

    docs = resp.get("result", [])

    # No results → fallback to web search
    if not docs:
        print("[Global RAG] No docs found. Falling back to Web Search.")
        return {"messages": messages, "use_search": True}

    context = "\n\n---\n\n".join(
        f"Trích đoạn từ [file: {d.get('metadata', {}).get('source', 'unknown')}]:\n{d.get('page_content')}"
        for d in docs
    )

    # Check cache
    cached = await check_cache(query, "global_rag")
    if cached:
        return {"messages": messages + [{"role": "assistant", "content": cached}]}

    prompt = create_master_prompt(query, context)
    rag_messages = messages[:-1] + [{"role": "user", "content": prompt}]

    llm_resp = await call_mcp_tool("llm_generate", {
        "messages": rag_messages,
        "generation_kwargs": {"temperature": 0.1, "repetition_penalty": 1.1},
    })
    reply = llm_resp.get("result", "Lỗi sinh văn bản RAG.")

    # Cache the response
    await save_cache(query, reply, "global_rag")

    return {"messages": messages + [{"role": "assistant", "content": reply}]}


async def node_web_search(state: AgentState) -> dict:
    """Web RAG: smart URL detection → crawl deep or search, then generate."""
    print("--- Node: Web Search ---")
    messages = state["messages"]
    query = get_last_user_query(messages)

    # Detect URLs in query
    url_pattern = re.compile(r"(https?://\S+)")
    found_urls = url_pattern.findall(query)

    context_parts = []

    # Case 1: URL found → deep crawl
    if found_urls:
        target_url = found_urls[0].strip('",\'')
        print(f"[Web Search] URL detected: {target_url}. Deep crawling...")
        crawl = await call_mcp_tool("web_crawl", {"url": target_url})
        if crawl.get("success"):
            data = crawl.get("result", {})
            content = data.get("clean_text", "") if isinstance(data, dict) else str(data)
            if content:
                context_parts.append(f"=== NỘI DUNG TỪ URL: {target_url} ===\n{content}")

    # Case 2: No URL content → general search
    if not context_parts:
        print(f"[Web Search] Searching: {query}")
        search = await call_mcp_tool("web_search", {"query": query})
        if search.get("success"):
            for r in search.get("result", []):
                if r.get("content"):
                    context_parts.append(f"[Nguồn: {r.get('url', '')}]\n{r.get('content')}")

    if not context_parts:
        return {"messages": messages + [{"role": "assistant", "content": "Xin lỗi, không tìm thấy thông tin liên quan."}]}

    # Check cache
    cached = await check_cache(query, "web_search")
    if cached:
        return {"messages": messages + [{"role": "assistant", "content": cached}]}

    prompt = create_master_prompt(query, "\n\n".join(context_parts))
    rag_messages = messages[:-1] + [{"role": "user", "content": prompt}]

    llm_resp = await call_mcp_tool("llm_generate", {
        "messages": rag_messages,
        "generation_kwargs": {"temperature": 0.1, "repetition_penalty": 1.1},
    })
    reply = llm_resp.get("result", "Lỗi sinh văn bản Web RAG.")

    # Cache the response
    await save_cache(query, reply, "web_search")

    return {"messages": messages + [{"role": "assistant", "content": reply}]}


async def node_hybrid_crawl(state: AgentState) -> dict:
    """Hybrid: crawl target URL + search related content simultaneously."""
    print("--- Node: Hybrid Crawl ---")
    messages = state["messages"]
    query = get_last_user_query(messages)
    url = state.get("target_url", "")

    async def do_crawl():
        resp = await call_mcp_tool("web_crawl", {"url": url})
        if resp.get("success"):
            data = resp.get("result", {})
            text = data.get("clean_text", "") if isinstance(data, dict) else str(data)
            return f"[Nội dung từ URL ({url})]:\n{text}"
        return ""

    async def do_search():
        resp = await call_mcp_tool("web_search", {"query": query})
        if resp.get("success"):
            return "\n\n".join(
                f"[Nguồn: {r.get('url', '')}]:\n{r.get('content', '')}"
                for r in resp.get("result", [])
            )
        return ""

    crawl_result, search_result = await asyncio.gather(do_crawl(), do_search())

    full_context = f"""=== DỮ LIỆU CRAWL ===
{crawl_result}

=== DỮ LIỆU TÌM KIẾM ===
{search_result}"""

    # Check cache
    cache_key_ctx = f"hybrid:{url}"
    cached = await check_cache(query, cache_key_ctx)
    if cached:
        return {"messages": messages + [{"role": "assistant", "content": cached}]}

    prompt = create_master_prompt(query, full_context)
    rag_messages = messages[:-1] + [{"role": "user", "content": prompt}]

    llm_resp = await call_mcp_tool("llm_generate", {
        "messages": rag_messages,
        "generation_kwargs": {"temperature": 0.1, "repetition_penalty": 1.1},
    })
    reply = llm_resp.get("result", "Lỗi sinh văn bản Hybrid RAG.")

    # Cache the response
    await save_cache(query, reply, cache_key_ctx)

    return {"messages": messages + [{"role": "assistant", "content": reply}]}


# ============================================================
# 4. ROUTER AGENT
# ============================================================

async def node_router_agent(state: AgentState) -> dict:
    """LLM-powered Router: classify query and decide which tool node to use."""
    print("--- Router Agent: Analyzing query... ---")
    messages = state["messages"]
    query = get_last_user_query(messages)
    use_search = state.get("use_search", False)
    use_ocr = state.get("use_ocr", False)
    target_url = state.get("target_url")

    # Priority 1: File just uploaded
    if use_ocr:
        print("[Router] → local_rag (file upload detected)")
        return {"next_node": "to_local_rag"}

    # Priority 2: URL + search
    if target_url and use_search:
        print("[Router] → hybrid_crawl (URL + search)")
        return {"next_node": "to_hybrid_crawl"}

    # Priority 3: Ask LLM to classify
    router_prompt = f"""Bạn là tác tử định tuyến. Phân loại câu hỏi vào MỘT danh mục.
Chỉ trả lời MỘT từ: 'web_search', 'global_rag', hoặc 'simple_chat'.

1. 'web_search': Nếu use_search=True, hoặc cần thông tin thời sự/thực tế.
2. 'global_rag': Nếu hỏi về tài liệu đã tải lên, kiến thức nội bộ.
3. 'simple_chat': Chào hỏi, trò chuyện thông thường.

Câu hỏi: "{query}"
use_search: {use_search}

Từ khóa trả lời:"""

    resp = await call_mcp_tool("llm_generate", {
        "messages": [{"role": "user", "content": router_prompt}]
    })

    if not resp.get("success"):
        print("[Router] LLM failed, fallback → simple_chat")
        return {"next_node": "to_simple_chat"}

    decision = resp.get("result", "simple_chat").strip().lower().replace("'", "").replace('"', "")
    print(f"[Router] LLM decision: {decision}")

    if "web_search" in decision:
        return {"next_node": "to_web_search"}
    if "global_rag" in decision:
        return {"next_node": "to_global_rag"}

    return {"next_node": "to_simple_chat"}


# ============================================================
# 5. BUILD & COMPILE LANGGRAPH
# ============================================================

print("[Orchestrator] Building LangGraph workflow...")

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("router_agent", node_router_agent)
workflow.add_node("simple_chat", node_simple_chat)
workflow.add_node("local_rag", node_local_rag)
workflow.add_node("global_rag", node_global_rag)
workflow.add_node("web_search", node_web_search)
workflow.add_node("hybrid_crawl", node_hybrid_crawl)

# Entry point
workflow.add_edge(START, "router_agent")

# Conditional routing from router
workflow.add_conditional_edges(
    "router_agent",
    lambda state: state["next_node"],
    {
        "to_local_rag": "local_rag",
        "to_global_rag": "global_rag",
        "to_web_search": "web_search",
        "to_hybrid_crawl": "hybrid_crawl",
        "to_simple_chat": "simple_chat",
    },
)

# Global RAG can fallback to web search if no results
workflow.add_conditional_edges(
    "global_rag",
    lambda state: "to_web_search" if state.get("use_search") else END,
    {
        "to_web_search": "web_search",
        "END": END,
    },
)

# Terminal edges
workflow.add_edge("simple_chat", END)
workflow.add_edge("local_rag", END)
workflow.add_edge("web_search", END)
workflow.add_edge("hybrid_crawl", END)

# Compile
compiled_graph = workflow.compile()
print("[Orchestrator] ✅ LangGraph workflow compiled successfully")
