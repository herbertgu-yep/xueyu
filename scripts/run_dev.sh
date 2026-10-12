#!/bin/bash
# 学屿 · 开发模式启动（浏览器访问，不需要 pywebview 窗口）
# 用法：./scripts/run_dev.sh [port]
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${1:-8765}"
if [ ! -d .venv ]; then
  echo "首次运行：创建虚拟环境并安装依赖…"
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -e '.[rag,ocr]'
  else
    python3 -m venv .venv
    .venv/bin/pip install -e '.[rag,ocr]'
  fi
fi
echo "学屿开发服务：http://127.0.0.1:${PORT}/  （数据目录：./data）"
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT"
