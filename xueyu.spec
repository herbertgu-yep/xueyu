# -*- mode: python ; coding: utf-8 -*-
"""学屿 macOS 客户端 PyInstaller 配置。构建：pyinstaller xueyu.spec"""
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

datas = [("ui", "ui")]
binaries = []
hiddenimports = []

# 重型动态导入包整体收集
for pkg in [
    "litellm", "llama_index", "llama_index.core", "llama_index.embeddings.huggingface",
    "chromadb", "tokenizers", "tiktoken",
    "pywebview", "uvicorn", "ddgs", "rank_bm25",
]:
    if os.environ.get("XUEYU_FULL_EMBED"):  # 需要本地 embedding 的构建：XUEYU_FULL_EMBED=1
        pass
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as e:
        print(f"collect_all({pkg}) 跳过: {e}")

# uvicorn 的动态加载模块
hiddenimports += [
    "uvicorn.logging", "uvicorn.loops.auto", "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.auto", "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.auto", "uvicorn.protocols.websockets.wsproto_impl",
    "uvicorn.lifespan.on", "uvicorn.lifespan.off",
    "app.main", "app.agent.mcp_client", "app.services.rag_service",
]

a = Analysis(
    ["main.py"],
    pathex=[os.path.abspath(".")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # 本地 embedding 栈（torch/transformers）体积过大——打包版 embedding 走 API 服务
    excludes=[
        "tkinter", "matplotlib", "PyQt5", "PyQt6", "PySide2", "PySide6", "IPython",
        "torch", "torchvision", "torchaudio", "torchgen", "torchfun",
        "transformers", "sentence_transformers", "safetensors",
        "onnxruntime", "sklearn", "scipy", "sympy", "networkx", "joblib",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="学屿",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,           # GUI 应用，无终端窗口
    icon="assets/icon.icns",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="学屿",
)
app = BUNDLE(
    coll,
    name="学屿.app",
    icon="assets/icon.icns",
    bundle_identifier="com.xueyu.app",
    info_plist={
        "CFBundleDisplayName": "学屿",
        "CFBundleName": "学屿",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "11.0",
        "NSHumanReadableCopyright": "© 2026 XueYu Contributors · MIT",
    },
)
