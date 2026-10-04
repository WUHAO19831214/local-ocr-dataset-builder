from __future__ import annotations

import socket
import threading
import time
import urllib.request

import uvicorn
import webview

from backend.app.main import app
from backend.app.process_registry import stop_active_processes


APP_TITLE = "Local OCR Dataset Builder"
HOST = "127.0.0.1"


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])


def wait_until_ready(url: str, timeout_seconds: float = 20.0) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/api/health", timeout=1.0) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last_error = exc
            time.sleep(0.2)
    raise RuntimeError(f"FastAPI 服务启动超时：{last_error}")


def main() -> None:
    port = find_free_port()
    url = f"http://{HOST}:{port}"
    config = uvicorn.Config(app, host=HOST, port=port, log_level="info", access_log=False)
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    wait_until_ready(url)
    webview.create_window(APP_TITLE, url, width=1180, height=820, min_size=(760, 620))

    try:
        webview.start()
    finally:
        stop_active_processes()
        server.should_exit = True
        server_thread.join(timeout=5)


if __name__ == "__main__":
    main()
