"""学屿 · 全局配置与数据目录。

所有数据（SQLite / 笔记 md / 向量索引 / 研究报告）都落在数据目录内，不出本机。
优先级：环境变量 XUEYU_DATA > PyInstaller 打包后的用户目录 > 开发仓库 ./data
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def default_data_dir() -> Path:
    env = os.environ.get("XUEYU_DATA")
    if env:
        return Path(env).expanduser().resolve()
    if getattr(sys, "frozen", False):  # PyInstaller 打包的 .app
        return Path.home() / "Library" / "Application Support" / "学屿"
    return Path(__file__).resolve().parent.parent / "data"


APP_NAME = "学屿"
APP_NAME_EN = "XueYu"
APP_VERSION = "0.1.0"

DATA_DIR = default_data_dir()
DB_PATH = DATA_DIR / "xueyu.db"
NOTES_DIR = DATA_DIR / "notes"            # 笔记以 .md + frontmatter 落盘（Obsidian 兼容）
DOCS_DIR = DATA_DIR / "docs"              # 上传的原始文档
TEXT_DIR = DATA_DIR / "docs_text"         # 解析出的纯文本
REPORTS_DIR = DATA_DIR / "reports"        # 研究报告（task.json + report.md）
CHROMA_DIR = DATA_DIR / "chroma"          # 本地向量索引持久化目录

# 艾宾浩斯固定复习点（天）；第 0 天为「学习日」（新学）
EBBINGHAUS_INTERVALS = [1, 2, 4, 7, 15, 30]
# 思考程度 → 研究深度参数
THINKING_PROFILES = {
    "quick": {"subquestions": 2, "results_per_search": 4, "fetch_pages": 2, "max_sources": 12, "report_words": "3000-5000"},
    "standard": {"subquestions": 4, "results_per_search": 6, "fetch_pages": 3, "max_sources": 24, "report_words": "6000-9000"},
    "deep": {"subquestions": 6, "results_per_search": 8, "fetch_pages": 4, "max_sources": 40, "report_words": "10000-15000"},
}


def ensure_dirs() -> None:
    for p in (DATA_DIR, NOTES_DIR, DOCS_DIR, TEXT_DIR, REPORTS_DIR, CHROMA_DIR):
        p.mkdir(parents=True, exist_ok=True)


ensure_dirs()
