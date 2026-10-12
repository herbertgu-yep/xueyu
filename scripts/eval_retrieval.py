"""检索质量测评：同一问题集跑四种检索模式，top-3 命中率对比。

判定规则（先于实验确定）：标准答案文档的任一分块进入检索结果前 3 名记为命中。

模式：
  vector_only  向量语义单路（query_shelf）
  bm25_only    BM25 关键词单路
  hybrid_rrf   BM25 + 向量双路 RRF 融合（产品默认）
  hybrid_full  hybrid + Multi-Query 扩展 + LLM 重排（需配置可用模型）

用法：XUEYU_DATA=<数据目录> python -m scripts.eval_retrieval
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

QUESTIONS_PATH = pathlib.Path(__file__).resolve().parent.parent / "tests" / "data" / "eval_questions.json"
OUT_PATH = pathlib.Path(__file__).resolve().parent.parent / "docs" / "evaluation-data.json"


def _title_of(chunk: dict) -> str:
    return str(chunk.get("title") or (chunk.get("metadata") or {}).get("title") or "")


def hit_top3(chunks: list[dict], doc: str) -> bool:
    """标准答案文档的分块是否出现在前 3。"""
    return any(doc in _title_of(c) for c in chunks[:3])


def run_mode(name: str, qs: list[dict], runner) -> dict:
    hits = 0
    per_type: dict[str, list[int]] = {"term": [], "colloquial": []}
    latency: list[float] = []
    rows = []
    for q in qs:
        t0 = time.perf_counter()
        chunks = runner(q["q"])
        latency.append((time.perf_counter() - t0) * 1000)
        h = hit_top3(chunks, q["doc"])
        hits += h
        per_type[q["type"]].append(int(h))
        rows.append({"q": q["q"], "type": q["type"], "expected": q["doc"], "hit": h,
                     "top1": _title_of(chunks[0]) if chunks else None})
    return {
        "mode": name,
        "hit_rate": round(hits / len(qs), 3),
        "term_hit_rate": round(sum(per_type["term"]) / len(per_type["term"]), 3),
        "colloquial_hit_rate": round(sum(per_type["colloquial"]) / len(per_type["colloquial"]), 3),
        "p50_ms": round(sorted(latency)[len(latency) // 2], 1),
        "rows": rows,
    }


def main() -> None:
    data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    qs = data["questions"]
    print(f"问题集：{len(qs)} 题（术语 {sum(1 for q in qs if q['type'] == 'term')} / "
          f"口语 {sum(1 for q in qs if q['type'] == 'colloquial')}）\n")

    import asyncio

    from app.services import rag_service

    results = []

    # 1) 向量单路：query_shelf 即纯向量检索
    results.append(run_mode("vector_only", qs, lambda q: rag_service.query_shelf(q, k=3)))

    # 2) BM25 单路：调内部 BM25 检索，按分数取 top-3
    from app.services.rag_service import _bm25_for_docs, _tokenize_zh

    def bm25_run(query: str) -> list[dict]:
        bm25, chunks = _bm25_for_docs(None)
        scores = bm25.get_scores(_tokenize_zh(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
        return [chunks[i] for i in order]

    results.append(run_mode("bm25_only", qs, bm25_run))

    # 3) 混合默认（RRF，无多路无重排）
    results.append(run_mode("hybrid_rrf", qs, lambda q: asyncio.run(rag_service.query_shelf_hybrid(q, k=3))[0]))

    # 4) 混合全开（Multi-Query + LLM 重排）
    results.append(run_mode("hybrid_full", qs, lambda q: asyncio.run(
        rag_service.query_shelf_hybrid(q, k=3, multi_query=True, rerank=True))[0]))

    OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'模式':14s}{'命中率':>8s}{'术语':>8s}{'口语':>8s}{'p50(ms)':>10s}")
    for r in results:
        print(f"{r['mode']:14s}{r['hit_rate']:>8.1%}{r['term_hit_rate']:>8.1%}"
              f"{r['colloquial_hit_rate']:>8.1%}{r['p50_ms']:>10.1f}")


if __name__ == "__main__":
    main()
