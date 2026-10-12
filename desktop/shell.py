"""学屿 macOS 桌面壳：FastAPI 服务 + pywebview 原生窗口。

- 端口动态分配（127.0.0.1），避免冲突
- 原生窗口加载本机服务地址；数据全部落在本机数据目录
"""
from __future__ import annotations

import logging
import socket
import threading
import time
import urllib.request

log = logging.getLogger("xueyu.shell")

APP_TITLE = "学屿 · AI 学习助手"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server(port: int) -> None:
    import uvicorn

    from app.main import app

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)


def _wait_ready(port: int, timeout: float = 30.0) -> bool:
    url = f"http://127.0.0.1:{port}/api/overview"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.3)
    return False




def main() -> None:
    import webview

    port = _free_port()
    t = threading.Thread(target=_start_server, args=(port,), daemon=True, name="xueyu-server")
    t.start()
    try:
        with open("/tmp/xueyu-ready", "w") as f:
            f.write(str(port))  # 就绪标记：用于启动诊断
    except Exception:
        pass
    if not _wait_ready(port):
        log.error("后端服务启动超时")
        import traceback

        with open("/tmp/xueyu-crash.log", "a", encoding="utf-8") as f:
            f.write("server did not become ready\n")
            if not t.is_alive():
                f.write("server thread died:\n" + "".join(traceback.format_stack()) + "\n")
        raise SystemExit(1)

    webview.create_window(
        APP_TITLE,
        f"http://127.0.0.1:{port}/",
        width=1400,
        height=900,
        min_size=(1080, 700),
        background_color="#faf4e6",
    )
    webview.start()


if __name__ == "__main__":
    main()
