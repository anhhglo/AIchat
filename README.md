BETA BETA !!!

# AIchat - Trợ lý AI Đa Nhiệm

Hệ thống AI Agent 3 tầng sử dụng **MCP Protocol**, **LangGraph**, và **Streamlit**.
Đã được cấu hình để sẵn sàng triển khai **Production** bằng **Docker Compose** và **NGINX**.

## Kiến trúc Hệ thống

```
┌─────────────────────────────────┐
│        NGINX (Port 80)          │ ◀─── Public Traffic
└─────┬───────────────────┬───────┘
      │                   │
┌─────▼──────────┐ ┌──────▼───────────┐     ┌──────────────────┐
│   Frontend     │ │   API Server     │────▶│  MCP Tool Server │
│  (Streamlit)   │ │  (FastAPI +      │     │    (FastAPI)     │
│  Port: 8501    │─▶│  Gunicorn)      │     │    Port: 8001    │
└────────────────┘ └──────┬───────────┘     └────────┬─────────┘
                          │                          │
           ┌──────────────┼──────────────────────────┤
           │              │                          │
    ┌──────▼───────┐ ┌────▼─────────┐         ┌──────▼───────┐
    │   chroma     │ │   MongoDB    │         │    Redis     │
    │(Vector Store)│ │  • Lịch sử   │         │  • Cache     │
    └──────────────┘ └──────────────┘         └──────────────┘
```

## Tính Năng & Bảo Mật (Production)
- **Rate Limit Khắc Nghiệt**: Backend được cấu trúc để chống spam API, giới hạn 10 tin nhắn/phút theo IP và tự dội ngược file > 5MB.
- **Docker Networks**: Mongo, Redis, MCP Server, và API Server chạy trong một mạng Network nội bộ ảo mà người dùng không thể can thiệp. NGINX là kênh Public duy nhất tiếp nhận Frontend.
- **Gunicorn Scalability**: FastAPI chạy Backend bằng Gunicorn với Uvicorn Workers để phục vụ đa luồng đồng thời 1 cách mượt mà.

## Yêu cầu
- Máy chú cài đặt **Docker** và **Docker Compose**.

## Triển khai (Production)

Thay vì chạy tay từng Service bằng Python, từ giờ bạn chỉ cần gõ 1 lệnh duy nhất:

```bash
# Xây dựng và Chạy hệ thống ẩn dưới nền mạng cục bộ
docker compose up -d --build
```

Để dừng ứng dụng:
```bash
docker compose down
```
# Dùng cloudflared tunnel từ Ubuntu để truy cập từ bên ngoài 
```bash
cloudflared tunnel --url http://localhost:80
```

## Cài đặt (Development Local)

Nếu bạn không muốn chạy Docker mà muốn Test trực tiếp:

```bash
pip install -r requirements.txt
# Terminal 1: MCP Tool Server
python mcp_server.py
# Terminal 2: API Server
python main.py
# Terminal 3: Frontend
streamlit run frontend.py
```
