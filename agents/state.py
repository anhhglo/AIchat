# agents/state.py
from typing import TypedDict, List, Optional


class AgentState(TypedDict):
    """
    Trạng thái hệ thống, được truyền giữa các node trong LangGraph.
    """
    # Lịch sử hội thoại
    messages: List[dict]  # [{"role": "user", "content": "..."},...]

    # ID phiên hội thoại (để lọc file theo user)
    thread_id: str


    # Cờ (flags) từ UI
    use_search: bool
    use_ocr: bool

    # Dữ liệu đầu vào
    uploaded_file_path: Optional[str]
    target_url: Optional[str]

    # Dữ liệu trung gian (context từ RAG/search)
    intermediate_context: Optional[str]

    # Quyết định định tuyến của Router Agent
    next_node: Optional[str]

    # Trạng thái lỗi
    error: Optional[str]
    retry_count: int
