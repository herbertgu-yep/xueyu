"""生成片段级检索评测题：每篇论文抽 4 个分块，让模型出「术语型 + 口语型」各一题。

判定依据存的是答案原文片段：检索结果的前 3 个分块里，只要有任意一个
包含这段原文，才算命中。比「正确文档进 top-3」严格得多。

题目由脚本生成：分块采样用固定种子（可复现），题目文本由模型生成，重跑会有措辞差异。
答案必须能在原文中逐字找到，题目不能直接包含答案，答案不能是全语料烂大街的词。

用法：
  XUEYU_DATA=~/.xueyu_eval_clean python -m scripts.gen_eval_questions
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import random
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

OUT_PATH = pathlib.Path(__file__).resolve().parent.parent / "tests" / "data" / "eval_questions.json"
SEED = 20261024
PER_DOC = 4   # 每篇最多抽 4 个分块当候选，取前 2 个合格的，每块出术语/口语各一题

NEGATIVES = [
    "量子纠错里的表面码如何检测比特翻转？",
    "CRISPR-Cas9 的脱靶效应怎么评估？",
    "Kubernetes Ingress 的 TLS 证书如何自动续期？",
    "明朝一条鞭法的主要内容是什么？",
    "飞机机翼的层流控制有哪些方法？",
    "咖啡萃取中粉水比对风味有什么影响？",
    "5G NR 的时分双工帧结构是怎样的？",
    "线性规划单纯形法遇到退化情况怎么处理？",
    "交响乐中赋格的主题如何展开？",
    "蛋白质冷冻电镜的三维重构流程是什么？",
]

PROMPT = """你是检索评测的出题人。下面是一篇论文的一个片段。请只依据这个片段出 2 道题：

1. 一道「术语型」：使用片段里的专业术语提问，答案必须能在这个片段里找到。
2. 一道「口语型」：用普通人的大白话提问，不要照抄专业术语，但答案同样必须在这个片段里。

要求：
- 每题给出 answer：从片段中逐字复制的一段原文（5-60 个字符），它必须能直接回答问题；
- 不要在问题里直接写出 answer；
- 问题要具体，不要问「这段讲了什么」这种开放题；
- 只输出 JSON，不要解释。

输出格式：
{"questions": [
  {"q": "术语型问题", "type": "term", "answer": "逐字原文"},
  {"q": "口语型问题", "type": "colloquial", "answer": "逐字原文"}
]}

片段：
{chunk}
"""

_WS = re.compile(r"[\s\u3000]+")
_KEEP = re.compile(r"[^0-9a-z\u4e00-\u9fff]+")


def norm(text: str) -> str:
    """去空白、标点、大小写，只留字母数字和汉字；用于「原文是否包含答案」的判断。"""
    return _KEEP.sub("", _WS.sub("", text).lower())


def _eligible(chunks: list[str]) -> list[int]:
    idx = []
    for i, c in enumerate(chunks):
        if not (600 <= len(c) <= 2600):
            continue
        digits = sum(ch.isdigit() for ch in c)
        if digits / max(1, len(c)) > 0.12:   # 参考文献/表格页
            continue
        if c.lstrip().startswith(("References", "REFERENCES", "Appendix")):
            continue
        idx.append(i)
    return idx


def _pick(chunks: list[str], doc_id: int) -> list[int]:
    cand = _eligible(chunks)
    if len(cand) <= PER_DOC:
        return cand
    rng = random.Random(SEED + doc_id)
    # 均匀分段后每段随机取一个，避免 4 个块全挤在论文同一处
    step = len(cand) / PER_DOC
    picked = []
    for i in range(PER_DOC):
        lo, hi = int(i * step), max(int(i * step) + 1, int((i + 1) * step))
        picked.append(rng.choice(cand[lo:hi]))
    return sorted(set(picked))


async def _ask(chunk: str, sem: asyncio.Semaphore) -> list[dict]:
    from app.services import llm as llm_mod

    async with sem:
        try:
            data = await llm_mod.ask_json(
                "你是检索评测的出题人，只输出 JSON。",
                PROMPT.replace("{chunk}", chunk[:2200]), temperature=0.3, fn="research")
            return data.get("questions", [])[:2]
        except Exception as e:  # noqa: BLE001
            print(f"  出题失败：{e}")
            return []


def _valid(item: dict, chunk: str, answer_doc_freq: int) -> bool:
    q = str(item.get("q") or "").strip()
    ans = str(item.get("answer") or "").strip()
    if not (8 <= len(q) <= 80) or not (5 <= len(ans) <= 80):
        return False
    if norm(ans) not in norm(chunk):          # 答案必须是原文
        return False
    if len(norm(ans)) >= 6 and norm(ans) in norm(q):   # 题目里不能直接有答案
        return False
    if answer_doc_freq > 3:                   # 烂大街的短语不能当答案
        return False
    return True


async def generate(data_dir: str) -> dict:
    import os

    os.environ["XUEYU_DATA"] = data_dir

    from sqlmodel import Session, select

    from app.services.rag_service import split_chunks
    from app.storage.db import engine
    from app.storage.models import Document

    with Session(engine) as s:
        docs = s.exec(select(Document)).all()
    if len(docs) < 10:
        raise SystemExit(f"评测库里只有 {len(docs)} 篇文档，先跑 fetch_eval_corpus --ingest")

    all_chunks = [(d, split_chunks(d.text_content)) for d in docs]
    normed = [(d.title, [norm(c) for c in chunks]) for d, chunks in all_chunks]

    sem = asyncio.Semaphore(5)
    tasks, meta = [], []
    for d, chunks in all_chunks:
        for ci in _pick(chunks, d.id):
            tasks.append(_ask(chunks[ci], sem))
            meta.append((d, chunks[ci], ci))
    results = await asyncio.gather(*tasks)

    def doc_freq(answer: str) -> int:
        na = norm(answer)
        return sum(1 for _, chunks in normed if any(na in c for c in chunks))

    per_doc: dict[str, list[tuple[int, list[dict]]]] = {}
    dropped = 0
    for (d, chunk, ci), items in zip(meta, results):
        picked: dict[str, dict] = {}
        for item in items:
            kind = item.get("type")
            if kind not in ("term", "colloquial") or kind in picked:
                continue
            picked[kind] = item
        good = []
        for item in picked.values():
            if not _valid(item, chunk, doc_freq(item["answer"])):
                dropped += 1
                continue
            good.append({"q": str(item["q"]).strip(), "type": item["type"],
                         "doc": d.title, "chunk_index": ci,
                         "answer": str(item["answer"]).strip()})
        if len(good) == 2:                  # 术语、口语各一题都合格，才采用这个分块
            per_doc.setdefault(d.title, []).append((ci, good))

    questions = []
    for entries in per_doc.values():
        for _ci, good in sorted(entries)[:2]:   # 每篇取 2 个分块 × 2 题 = 4 题
            questions.extend(good)

    # 同一道题只留一条
    seen_q, unique = set(), []
    for q in questions:
        key = norm(q["q"])
        if key in seen_q:
            continue
        seen_q.add(key)
        unique.append(q)

    return {
        "note": "片段级评测题：前 3 个检索结果中，任一命中的分块包含 answer 原文即算命中。"
                "题目由 scripts/gen_eval_questions.py 生成，固定采样种子，可复现。",
        "questions": unique,
        "negatives": [{"q": q, "type": "negative"} for q in NEGATIVES],
        "_dropped": dropped,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(pathlib.Path.home() / ".xueyu_eval_clean"))
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    data = asyncio.run(generate(str(pathlib.Path(args.data).expanduser())))
    path = pathlib.Path(args.out)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    by_doc: dict[str, int] = {}
    for q in data["questions"]:
        by_doc[q["doc"]] = by_doc.get(q["doc"], 0) + 1
    print(f"生成 {len(data['questions'])} 题，覆盖 {len(by_doc)} 篇文档，丢弃 {data['_dropped']} 条不合格")
    print("每篇题数：", by_doc)
    print("负样本：", len(data["negatives"]))
    print(f"写入 {path}")


if __name__ == "__main__":
    main()
