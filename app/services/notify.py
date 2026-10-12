"""系统通知 / 提醒：macOS osascript 原生通知；开发环境无 GUI 时静默降级。"""
from __future__ import annotations

import logging
import subprocess
import sys

log = logging.getLogger("xueyu.notify")


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def notify(title: str, body: str, *, task_id: str = "") -> bool:
    """发送 macOS 系统通知。返回是否成功。"""
    if sys.platform != "darwin":
        log.info("[notify] %s: %s", title, body)
        return False
    script = f'display notification "{_escape(body)}" with title "{_escape(title)}" sound name "Glass"'
    try:
        subprocess.run(["osascript", "-e", script], check=False, timeout=5,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("notify failed: %s", e)
        return False
