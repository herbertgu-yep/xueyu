"""MCP（Model Context Protocol）stdio 客户端 —— 零依赖实现。

- 通过子进程 stdio 与 MCP 服务器通信（newline-delimited JSON-RPC 2.0）
- 支持握手 initialize → tools/list → tools/call
- 用于深度研究：把外部 MCP 工具并入 ReAct 工具循环
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Optional

PROTOCOL_VERSION = "2024-11-05"


class McpStdioClient:
    def __init__(self, name: str, command: str, args: Optional[list[str]] = None,
                 env: Optional[dict[str, str]] = None, timeout: float = 10.0):
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.timeout = timeout
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.tools: list[dict] = []
        self._pending: dict[str, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self.started = False

    async def _send(self, payload: dict) -> None:
        assert self.proc and self.proc.stdin
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        self.proc.stdin.write(line.encode())
        await self.proc.stdin.drain()

    async def _request(self, method: str, params: Optional[dict] = None, timeout: Optional[float] = None) -> Any:
        rid = uuid.uuid4().hex[:8]
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        await self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        try:
            return await asyncio.wait_for(fut, timeout or self.timeout)
        finally:
            self._pending.pop(rid, None)

    async def _read_loop(self) -> None:
        assert self.proc and self.proc.stdout
        while True:
            line = await self.proc.stdout.readline()
            if not line:
                break
            try:
                msg = json.loads(line)
            except Exception:
                continue
            rid = msg.get("id")
            if rid and rid in self._pending:
                fut = self._pending.pop(rid)
                if not fut.done():
                    if "error" in msg:
                        fut.set_exception(RuntimeError(msg["error"].get("message", "MCP error")))
                    else:
                        fut.set_result(msg.get("result"))
            # notification（如 tools/list_changed）暂忽略

    async def start(self) -> None:
        import os

        env = {**os.environ, **self.env}
        self.proc = await asyncio.create_subprocess_exec(
            self.command, *self.args,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, env=env,
        )
        self._reader_task = asyncio.get_running_loop().create_task(self._read_loop())
        await self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "xueyu", "version": "0.1.0"},
        })
        await self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        result = await self._request("tools/list", {})
        self.tools = [
            {"name": t.get("name", ""), "description": (t.get("description") or "")[:300],
             "inputSchema": t.get("inputSchema") or {"type": "object", "properties": {}}}
            for t in (result or {}).get("tools", [])
        ]
        self.started = True

    async def call(self, tool_name: str, arguments: dict) -> str:
        result = await self._request("tools/call", {"name": tool_name, "arguments": arguments}, timeout=60)
        parts = []
        for item in (result or {}).get("content", []):
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts)[:8000] or "（工具无输出）"

    async def stop(self) -> None:
        if self.proc:
            try:
                self.proc.kill()
            except Exception:
                pass
            self.proc = None
        self.started = False


async def connect_all(servers: list[dict]) -> tuple[list[McpStdioClient], list[str]]:
    """连接所有启用的 MCP 服务器；返回 (clients, 错误列表)。"""
    clients, errors = [], []
    for s in servers:
        if not s.get("enabled", True):
            continue
        client = McpStdioClient(s.get("name", "mcp"), s["command"], s.get("args"), s.get("env"))
        try:
            await client.start()
            clients.append(client)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{s.get('name')}: {e}")
    return clients, errors
