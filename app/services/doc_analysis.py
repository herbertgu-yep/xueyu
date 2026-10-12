"""文档分析：AI 摘要 / 思维导图 / 考点自测题 / 单文档问答（RAG）。

产物直接落在 Document 行内；分析是真实 LLM 调用，未配置模型时报错提示。
"""
from __future__ import annotations

import json
from typing import Optional

from sqlmodel import Session

from app.storage.db import engine
from app.storage.models import Document
from app.services import llm

SUMMARY_SYSTEM = """你是严谨的学习助手。基于给定文档内容输出「划重点」式摘要：
- 用 Markdown 无序列表，每条一行核心要点（8-16 条，视内容密度）
- 保留关键术语、数字、结论；不要客套话，不要总起段
- 使用中文（文档本身是英文时，术语可保留英文）"""

MINDMAP_SYSTEM = """你是学习助手。把文档整理成思维导图大纲，输出 Markdown：
- 第一行是 `# 中心主题`
- 每个一级分支用 `## 分支名`
- 分支下的要点用 `- 要点` 列出，重点要点可再嵌套一层子要点
- 共 2-4 层、15-40 个节点，每个节点不超过 20 字，覆盖文档主要结构与关键细节
只输出 Markdown 大纲本身，不要任何解释或代码栏标记。"""

QUIZ_SYSTEM = """你是出题老师。基于文档内容出考点自测题 JSON：
[{"question": "题目（可以是问答/填空/简答）", "answer": "参考答案，2-4 句"}]
- 6-10 题，覆盖不同章节的高频考点，避免同义重复；只输出 JSON"""


def _doc_text(doc: Document, cap: int = 24000) -> str:
    text = doc.text_content or ""
    if len(text) > cap:
        head = text[: int(cap * 0.7)]
        tail = text[-int(cap * 0.3):]
        return head + "\n…（中间内容略）…\n" + tail
    return text


def analyze(doc_id: int, kind: str) -> dict:
    """kind ∈ summary | mindmap | quiz | all。"""
    with Session(engine) as s:
        doc = s.get(Document, doc_id)
        if not doc:
            raise KeyError("文档不存在")
        text = _doc_text(doc)
        if not text.strip():
            raise ValueError("该文档没有可分析的文本（可能为扫描件，需启用 OCR）")
        error = None
        try:
            if kind in ("summary", "all"):
                doc.summary_md = _summarize(text)
            if kind in ("mindmap", "all"):
                doc.mindmap_json = json.dumps({"markdown": _mindmap(text)}, ensure_ascii=False)
            if kind in ("quiz", "all"):
                doc.quiz_json = json.dumps(_quiz(text), ensure_ascii=False)
            doc.analyzed_at = doc.analyzed_at or _now()
            doc.analysis_error = None
        except (llm.LLMConfigError, llm.LLMCallError) as e:
            error = str(e)
            doc.analysis_error = error
        s.add(doc)
        s.commit()
        s.refresh(doc)
    from app.services.shelf_service import serialize

    return {"doc": serialize(doc), "error": error}


def _now() -> str:
    from datetime import datetime

    return datetime.now().astimezone().isoformat(timespec="seconds")


def _summarize(text: str) -> str:
    return llm.acompletion([
        {"role": "system", "content": SUMMARY_SYSTEM},
        {"role": "user", "content": f"文档内容：\n\n{text}"},
    ], temperature=0.3, fn="doc")


def _mindmap(text: str) -> str:
    md = llm.acompletion([
        {"role": "system", "content": MINDMAP_SYSTEM},
        {"role": "user", "content": f"文档内容：\n\n{text}"},
    ], temperature=0.2, fn="doc")
    if "#" not in md:
        raise llm.LLMCallError("思维导图大纲未包含任何层级结构｜模型原始输出片段：" + md[:150])
    return md


def _quiz(text: str) -> list[dict]:
    data = llm.ask_json_sync(QUIZ_SYSTEM, f"文档内容：\n\n{text}", temperature=0.4, fn="doc")
    if not isinstance(data, list):
        raise llm.LLMCallError("自测题 JSON 结构不符合约定")
    return [{"question": str(x.get("question", "")).strip(),
             "answer": str(x.get("answer", "")).strip()}
            for x in data if x.get("question")]


def ask(doc_id: int, question: str, history: Optional[list[dict]] = None) -> dict:
    """单文档问答：基于文档完整原文回答（不依赖向量索引）。"""
    from app.services import llm as llm_mod

    cached = llm_mod.semantic_cache_get(f"doc{doc_id}:{question}")
    if cached:
        return {"answer": cached, "sources": [], "cached": True}
    with Session(engine) as s:
        doc = s.get(Document, doc_id)
        if not doc:
            raise KeyError("文档不存在")
    context = doc.text_content or ""
    if len(context) > 60000:
        context = context[:40000] + "\n…（中间内容略）…\n" + context[-15000:]
    if not context.strip():
        return {"answer": "该文档没有可分析的文本（可能为扫描件，需先启用 OCR 解析）。", "sources": [], "cached": False}
    msgs = [{"role": "system", "content":
             "你是学习助手，基于提供的完整文档回答问题。回答用中文，紧扣文档内容；"
             "文档中没有的信息要明确说明文档未涉及。"}]
    for h in (history or [])[-6:]:
        msgs.append({"role": h.get("role", "user"), "content": str(h.get("content", ""))[:2000]})
    msgs.append({"role": "user", "content": f"文档《{doc.title}》全文：\n\n{context}\n\n问题：{question}"})
    answer = llm.acompletion(msgs, temperature=0.3, fn="doc")
    llm_mod.semantic_cache_put(f"doc{doc_id}:{question}", answer)
    return {"answer": answer, "sources": [{"doc_id": doc_id, "title": doc.title}], "cached": False}
