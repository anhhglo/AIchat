# main.py
# Entry point for the API Server
import uvicorn
from dotenv import load_dotenv
from config import API_SERVER_HOST, API_SERVER_PORT


def start_server():
    """Load env vars and start FastAPI server."""
    print("[Main] Loading environment variables...")
    load_dotenv()

    print(f"[Main] Starting API Server on {API_SERVER_HOST}:{API_SERVER_PORT}")
    uvicorn.run(
        "api_server:app",
        host=API_SERVER_HOST,
        port=API_SERVER_PORT,
        reload=True,  # Disable in production
    )


if __name__ == "__main__":
    start_server()
