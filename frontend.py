# frontend.py
# Streamlit Frontend for AIchat
# Streamlit frontend without Authentication

import streamlit as st
import requests
import uuid
import time

# ================= CONFIG =================
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://aichat-api:8000")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8001")
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"
UPLOAD_ENDPOINT = f"{API_BASE_URL}/chat_file"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
HISTORY_ENDPOINT = f"{API_BASE_URL}/history"
THREADS_ENDPOINT = f"{API_BASE_URL}/history/threads"
CACHE_STATS_ENDPOINT = f"{API_BASE_URL}/cache/stats"

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AIchat - Trợ lý AI Đa Nhiệm",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
    /* Popover button style */
    div[data-testid="stPopover"] > button {
        border: none;
        background-color: transparent;
        font-size: 24px;
        padding: 0px;
    }
    div[data-testid="stPopover"] > button:hover {
        color: #ff4b4b;
        background-color: transparent;
    }

    /* Custom header gradient */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2em;
        font-weight: 800;
        margin-bottom: 0;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: 600;
    }
    .status-ok { background: #d4edda; color: #155724; }
    .status-err { background: #f8d7da; color: #721c24; }

    /* Response time */
    .response-time {
        color: #888;
        font-size: 0.75em;
        text-align: right;
    }

    /* Cache hit badge */
    .cache-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 8px;
        font-size: 0.7em;
        font-weight: 600;
        background: #fff3cd;
        color: #856404;
        margin-left: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ================= HELPER FUNCTIONS =================

def check_service_health(url: str, timeout: float = 3.0) -> dict:
    """Check if a service is responding."""
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            return {"status": "healthy", "data": resp.json()}
        return {"status": "error", "data": {"message": f"HTTP {resp.status_code}"}}
    except requests.ConnectionError:
        return {"status": "offline", "data": {"message": "Không thể kết nối"}}
    except Exception as e:
        return {"status": "error", "data": {"message": str(e)}}


def render_status_badge(label: str, status: str) -> str:
    """Render a colored status badge."""
    css_class = "status-ok" if status in ("healthy", "ready", True) else "status-err"
    icon = "🟢" if status in ("healthy", "ready", True) else "🔴"
    display_status = status if isinstance(status, str) else ("connected" if status else "disconnected")
    return f'{icon} <span class="status-badge {css_class}">{label}: {display_status}</span>'


def load_history_from_mongo(thread_id: str) -> list:
    """Load chat history from MongoDB via API Server."""
    try:
        resp = requests.get(f"{HISTORY_ENDPOINT}/{thread_id}", timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("success"):
                messages = data.get("messages", [])
                if messages:
                    print(f"[Frontend] Loaded {len(messages)} messages from MongoDB for {thread_id}")
                    return messages
    except Exception as e:
        print(f"[Frontend] Failed to load history from MongoDB: {e}")
    return []


def clear_history_from_mongo(thread_id: str) -> int:
    """Clear chat history in MongoDB via API Server."""
    try:
        resp = requests.delete(f"{HISTORY_ENDPOINT}/{thread_id}", timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("deleted_count", 0)
    except Exception as e:
        print(f"[Frontend] Failed to clear MongoDB history: {e}")
    return 0

def load_threads_from_api() -> list:
    """Load list of all threads from API Server."""
    try:
        resp = requests.get(THREADS_ENDPOINT, timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("success"):
                return data.get("threads", [])
    except Exception as e:
        print(f"[Frontend] Failed to load threads from API: {e}")
    return []

# ================= SESSION STATE =================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"user_{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! 👋 Tôi là trợ lý AI đa nhiệm. Nhấn ➕ để gửi file hoặc nhập câu hỏi trực tiếp."}
    ]

# Handle switching threads safely
if "switch_to_thread" in st.session_state:
    new_thread_id = st.session_state.switch_to_thread
    st.session_state.thread_id = new_thread_id
    mongo_history = load_history_from_mongo(new_thread_id)
    if mongo_history:
        st.session_state.messages = mongo_history
    else:
        st.session_state.messages = [
            {"role": "assistant", "content": "Xin chào! 👋 Bắt đầu cuộc trò chuyện mới nào."}
        ]
    del st.session_state.switch_to_thread
    st.session_state.history_loaded = True
    
# Load history from MongoDB on first visit for this thread
elif "history_loaded" not in st.session_state:
    mongo_history = load_history_from_mongo(st.session_state.thread_id)
    if mongo_history:
        st.session_state.messages = mongo_history
    st.session_state.history_loaded = True

# Fetch threads list
available_threads = load_threads_from_api()

# ================= SIDEBAR =================

with st.sidebar:
    st.markdown("### 💬 Cuộc trò chuyện")
    
    # New chat button
    if st.button("➕ Tạo Chat Mới", use_container_width=True, type="primary"):
        st.session_state.thread_id = f"user_{uuid.uuid4().hex[:8]}"
        st.session_state.messages = [
            {"role": "assistant", "content": "Phiên mới đã được tạo! 🎉 Tôi sẵn sàng giúp bạn."}
        ]
        st.session_state.history_loaded = True
        st.rerun()

    st.markdown("#### Lịch sử Chat")
    
    if available_threads:
        for t in available_threads:
            tid = t.get('thread_id')
            msg_count = t.get('message_count', 0)
            
            # Format display label (Use partial thread ID or just ID)
            is_active = (tid == st.session_state.thread_id)
            label = f"{'🟢 ' if is_active else '📄 '}{tid} ({msg_count} msgs)"
            
            if st.button(label, key=f"btn_{tid}", use_container_width=True, disabled=is_active):
                st.session_state.switch_to_thread = tid
                st.rerun()
    else:
        st.info("Chưa có lịch sử.")

    st.divider()

    st.markdown("### ⚙️ Bảng điều khiển")

    # Session info
    st.text_input("🔗 Session ID hiện tại", value=st.session_state.thread_id, disabled=True)

    if st.button("🗑️ Xóa chat hiện tại", use_container_width=True):
        # Clear MongoDB history for this thread
        deleted = clear_history_from_mongo(st.session_state.thread_id)
        st.session_state.messages = [
            {"role": "assistant", "content": f"Cuộc trò chuyện đã được xóa! 🧹 ({deleted} tin nhắn đã xóa từ DB)"}
        ]
        st.rerun()

    st.divider()

    # Health check
    st.markdown("### 📡 Trạng thái hệ thống")
    if st.button("🔍 Kiểm tra kết nối", use_container_width=True):
        with st.spinner("Đang kiểm tra..."):
            api_health = check_service_health(f"{API_BASE_URL}/health")
            mcp_health = check_service_health(f"{MCP_SERVER_URL}/health")

            st.markdown(
                render_status_badge("API Server", api_health["status"]),
                unsafe_allow_html=True,
            )
            st.markdown(
                render_status_badge("MCP Server", mcp_health["status"]),
                unsafe_allow_html=True,
            )

            # Show infrastructure services from MCP health
            if mcp_health["status"] == "healthy":
                services = mcp_health.get("data", {}).get("services", {})
                for name, status in services.items():
                    label = {"pinecone": "Pinecone", "mongodb": "MongoDB", "redis": "Redis"}.get(name, name)
                    st.markdown(
                        render_status_badge(label, status),
                        unsafe_allow_html=True,
                    )

            # Show graph info
            if api_health["status"] == "healthy":
                data = api_health.get("data", {})
                graph_status = data.get("graph_status", "unknown")
                st.markdown(
                    render_status_badge("LangGraph", graph_status),
                    unsafe_allow_html=True,
                )

    st.divider()

    # Cache stats
    st.markdown("### 📊 Thống kê")
    total_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.metric("Tin nhắn đã gửi", total_msgs)

    try:
        cache_resp = requests.get(CACHE_STATS_ENDPOINT, timeout=3.0)
        if cache_resp.status_code == 200:
            stats = cache_resp.json().get("stats", {})
            if stats.get("status") == "connected":
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Cache entries", stats.get("cached_responses", 0))
                with col_b:
                    st.metric("Cache hits", stats.get("total_hits", 0))
    except Exception:
        pass  # Cache stats are optional


# ================= MAIN UI =================

st.markdown('<p class="main-header">🤖 AIchat</p>', unsafe_allow_html=True)
st.caption("Trợ lý AI Đa Nhiệm — Chat • RAG • Web Search • OCR • MongoDB • Redis")

# 1. DISPLAY CHAT HISTORY
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Show response time if available
            if msg.get("response_time"):
                time_html = f'<p class="response-time">⏱️ {msg["response_time"]:.1f}s'
                if msg.get("cached"):
                    time_html += ' <span class="cache-badge">⚡ CACHED</span>'
                time_html += '</p>'
                st.markdown(time_html, unsafe_allow_html=True)


# 2. INPUT AREA
# Popover for attachments and toggles
with st.popover("➕", help="Đính kèm tài liệu hoặc cài đặt"):
    st.markdown("### 📎 Đính kèm tài liệu")
    uploaded_file = st.file_uploader(
        "Chọn file (PDF, Ảnh)...",
        type=["pdf", "png", "jpg", "jpeg"],
        key="file_uploader_widget",
    )
    st.divider()
    st.markdown("### 🔧 Tùy chọn")
    use_search = st.toggle("🌐 Bật tìm kiếm Web", value=False)
    use_ocr = st.toggle("📷 Bắt buộc OCR", value=False)

# Chat input
prompt = st.chat_input("Nhập câu hỏi tại đây...")


# 3. PROCESS USER INPUT
if prompt:
    start_time = time.time()

    # --- CASE 1: FILE ATTACHMENT ---
    if uploaded_file:
        with chat_container:
            with st.chat_message("user"):
                st.markdown(f"📎 **[{uploaded_file.name}]**\n\n{prompt}")

        st.session_state.messages.append({
            "role": "user",
            "content": f"📎 **[File: {uploaded_file.name}]**\n\n{prompt}",
        })

        with st.chat_message("assistant"):
            with st.spinner("📄 Đang tải file và phân tích..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    form_data = {
                        "query": prompt,
                        "thread_id": st.session_state.thread_id,
                        "use_search": use_search,
                    }
                    response = requests.post(UPLOAD_ENDPOINT, files=files, data=form_data, timeout=180)

                    elapsed = time.time() - start_time

                    if response.status_code == 200:
                        ans = response.json().get("response", "Đã xong.")
                        st.markdown(ans)
                        st.markdown(
                            f'<p class="response-time">⏱️ {elapsed:.1f}s</p>',
                            unsafe_allow_html=True,
                        )
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": ans,
                            "response_time": elapsed,
                        })
                        st.rerun()  # Clear file uploader
                    else:
                        error_detail = response.text[:200]
                        st.error(f"❌ Lỗi Server ({response.status_code}): {error_detail}")
                except requests.Timeout:
                    st.error("⏰ Quá thời gian chờ! Server xử lý quá lâu.")
                    st.button("🔄 Thử lại", key="retry_file")
                except requests.ConnectionError:
                    st.error("🔌 Không thể kết nối tới API Server! Hãy kiểm tra server.")
                except Exception as e:
                    st.error(f"❌ Lỗi: {e}")

    # --- CASE 2: TEXT CHAT ---
    else:
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("⏳ _Đang suy nghĩ..._")

            try:
                payload = {
                    "query": prompt,
                    "use_search": use_search,
                    "use_ocr": use_ocr,
                    "thread_id": st.session_state.thread_id,
                    # Don't send history — API server loads from MongoDB
                    "history": [],
                }
                response = requests.post(CHAT_ENDPOINT, json=payload, timeout=120)

                elapsed = time.time() - start_time

                if response.status_code == 200:
                    resp_json = response.json()
                    ans = resp_json.get("response", "")
                    is_cached = resp_json.get("cached", False)

                    message_placeholder.markdown(ans)

                    time_html = f'<p class="response-time">⏱️ {elapsed:.1f}s'
                    if is_cached:
                        time_html += ' <span class="cache-badge">⚡ CACHED</span>'
                    time_html += '</p>'
                    st.markdown(time_html, unsafe_allow_html=True)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": ans,
                        "response_time": elapsed,
                        "cached": is_cached,
                    })
                else:
                    error_detail = response.text[:200]
                    message_placeholder.error(f"❌ Lỗi Server ({response.status_code}): {error_detail}")

            except requests.Timeout:
                message_placeholder.error("⏰ Quá thời gian chờ! Server xử lý quá lâu.")
            except requests.ConnectionError:
                message_placeholder.error("🔌 Không thể kết nối tới API Server!")
            except Exception as e:
                message_placeholder.error(f"❌ Lỗi: {e}")
