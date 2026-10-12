"""文档解析：把上传文档统一解析为纯文本。

- PDF：PyMuPDF 逐页抽取；文本过少的「扫描件」页走 RapidOCR（可选依赖，缺失时跳过并记录）
- Word：python-docx（段落 + 表格）
- PPT：python-pptx（文本框 + 备注）
- Markdown / TXT：直接读
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pymupdf

MAX_TEXT_CHARS = 600_000  # 超长文档截断保护（向量索引与分析用）


def parse_pdf(path: Path, *, ocr: Optional[Callable[[bytes], str]] = None) -> tuple[str, int, bool]:
    """返回 (文本, 页数, 是否用过 OCR)。"""
    doc = pymupdf.open(path)
    pages: list[str] = []
    ocr_used = False
    for page in doc:
        text = page.get_text("text").strip()
        if len(text) < 24 and ocr is not None:  # 疑似扫描页
            try:
                pix = page.get_pixmap(dpi=200)
                ocr_text = ocr(pix.tobytes("png")).strip()
                if ocr_text:
                    text = ocr_text
                    ocr_used = True
            except Exception:
                pass  # OCR 失败时保留原样（可能真没字）
        pages.append(f"【第 {page.number + 1} 页】\n{text}")
    n = doc.page_count
    doc.close()
    return "\n\n".join(pages)[:MAX_TEXT_CHARS], n, ocr_used


def parse_docx(path: Path) -> tuple[str, Optional[int]]:
    from docx import Document as DocxDocument

    d = DocxDocument(str(path))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for table in d.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)[:MAX_TEXT_CHARS], None


def parse_pptx(path: Path) -> tuple[str, Optional[int]]:
    from pptx import Presentation

    prs = Presentation(str(path))
    parts = []
    for i, slide in enumerate(prs.slides, 1):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                t = shape.text_frame.text.strip()
                if t:
                    texts.append(t)
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
            nt = slide.notes_slide.notes_text_frame.text.strip()
            if nt:
                texts.append(f"备注：{nt}")
        if texts:
            parts.append(f"【第 {i} 页】\n" + "\n".join(texts))
    return "\n\n".join(parts)[:MAX_TEXT_CHARS], len(parts)


def parse_plain(path: Path) -> tuple[str, Optional[int]]:
    return path.read_text(encoding="utf-8", errors="ignore")[:MAX_TEXT_CHARS], None


def parse_any(path: Path, fmt: Optional[str] = None) -> dict:
    """统一入口。返回 {text, pages, ocr_used, fmt}。"""
    fmt = (fmt or path.suffix.lstrip(".").lower()).lower()
    if fmt in ("md", "markdown", "txt"):
        text, pages = parse_plain(path)
    elif fmt == "pdf":
        text, pages, _ = parse_pdf(path, ocr=get_ocr())
    elif fmt == "docx":
        text, pages = parse_docx(path)
    elif fmt == "pptx":
        text, pages = parse_pptx(path)
    elif fmt == "doc":
        raise ValueError("旧版 .doc 请先在 Word/WPS 中另存为 .docx")
    elif fmt == "ppt":
        raise ValueError("旧版 .ppt 请先在 PowerPoint/WPS 中另存为 .pptx")
    else:
        raise ValueError(f"暂不支持的格式：.{fmt}")
    return {"text": text, "pages": pages, "ocr_used": False, "fmt": "md" if fmt == "markdown" else fmt}


_OCR_FN: Optional[Callable[[bytes], str]] = None
_OCR_TRIED = False


def get_ocr() -> Optional[Callable[[bytes], str]]:
    """惰性加载 RapidOCR；未安装或初始化失败返回 None（跳过 OCR 路径）。"""
    global _OCR_FN, _OCR_TRIED
    if _OCR_TRIED:
        return _OCR_FN
    _OCR_TRIED = True
    try:
        from rapidocr_onnxruntime import RapidOCR

        engine = RapidOCR()

        def run(png: bytes) -> str:
            result, _ = engine(png)
            if not result:
                return ""
            return "\n".join(line[1] for line in result)

        _OCR_FN = run
    except Exception:
        _OCR_FN = None
    return _OCR_FN
