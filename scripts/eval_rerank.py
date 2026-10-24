"""重排器对比：固定同一批候选，分别用 cross-encoder 与 LLM listwise 排序。

完整链路（含 Multi-Query 改写）每次运行都会因为改写结果不同而波动，
无法判断差异来自改写还是重排。这个脚本把改写结果缓存下来，
两种重排器吃完全相同的候选池，只比较排序能力。

用法：
  XUEYU_DATA=<评测数据目录> python -m scripts.eval_rerank
  XUEYU_DATA=<评测数据目录> python -m scripts.eval_rerank --refresh-rewrites
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

ROOT = pathlib.Path(__file__).resolve().parent.parent
QUESTIONS_PATH = ROOT / "tests" / "data" / "eval_questions.json"
OUT_PATH = ROOT / "docs" / "rerank-comparison.json"
RECALL_K = 12


def _rewrite(query: str) -> list[str]:
    """与产品里一致的多查询改写。"""
    from app.services import llm as llm_mod

    try:
        data = asyncio.run(llm_mod.ask_json(
            "你是检索查询改写器。把查询改写为 2-3 个不同视角的检索式 JSON："
            '{"variants": ["…"]}（含原查询的关键概念、同义词、上下位词）；只输出 JSON。',
            f"原查询：{query}", temperature=0.2, fn="research"))
        variants = [str(v).strip() for v in data.get("variants", []) if str(v).strip()][:3]
        return [query] + variants
    except Exception as e:  # noqa: BLE001
        print(f"  改写失败，退回原查询：{e}")
        return [query]


def _candidates(queries: list[str]) -> list[dict]:
    """按产品口径召回：每题多路向量 + 多路 BM25 → RRF。"""
    from app.services.rag_service import _bm25_for_docs, _rrf_fuse, _tokenize_zh, query_shelf

    vector_hits: dict[str, dict] = {}
    bm25_hits: dict[str, dict] = {}
    bm25, corpus = _bm25_for_docs(None)
    for q in queries:
        for h in query_shelf(q, k=RECALL_K):
            key = h.get("chunk_id") or f"{h['doc_id']}:{h['text'][:80]}"
            if key not in vector_hits or (h.get("score") or 0) > (vector_hits[key].get("score") or 0):
                vector_hits[key] = h
        if bm25 and corpus:
            scores = bm25.get_scores(_tokenize_zh(q))
            ranked = sorted(zip(corpus, scores), key=lambda x: -x[1])[:RECALL_K]
            for c, sc in ranked:
                if sc > 0:
                    key = c.get("chunk_id") or f"{c['doc_id']}:{c['text'][:80]}"
                    bm25_hits.setdefault(key, {**c, "score": round(float(sc), 3)})
    return _rrf_fuse(list(vector_hits.values()), list(bm25_hits.values()))


def _hit(chunks: list[dict], expected: str) -> bool:
    return any(expected in str(c.get("title") or "") for c in chunks[:3])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-rewrites", action="store_true")
    ap.add_argument("--cache", default=str(pathlib.Path.home() / ".xueyu_eval_clean" / "rewrites.json"))
    ap.add_argument("--ce-limits", default="", help="逗号分隔的 CE 候选上限，如 12,24,40；为空则不限")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    qs = data["questions"]

    cache_path = pathlib.Path(args.cache).expanduser()
    rewrites: dict[str, list[str]] = {}
    if cache_path.exists() and not args.refresh_rewrites:
        rewrites = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"复用改写缓存：{cache_path}")
    for q in qs:
        if q["q"] not in rewrites:
            rewrites[q["q"]] = _rewrite(q["q"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(rewrites, ensure_ascii=False, indent=2), encoding="utf-8")

    from app.services import rerank as rerank_mod
    from app.services.rag_service import _llm_listwise_rerank

    rows, ce_hits, llm_hits = [], 0, 0
    ce_time = llm_time = 0.0
    ce_limits = [int(x) for x in args.ce_limits.split(",") if x.strip()] or [None]
    ce_stats = {lim: {"hits": 0, "ms": 0.0} for lim in ce_limits}
    for q in qs:
        cand = _candidates(rewrites[q["q"]])
        for lim in ce_limits:
            t0 = time.perf_counter()
            ce_ranked = rerank_mod.rerank(q["q"], cand, limit=lim) or cand
            ce_stats[lim]["ms"] += time.perf_counter() - t0
            ce_stats[lim]["hits"] += _hit(ce_ranked, q["doc"])
            if lim == ce_limits[0]:
                ce_ranked_first = ce_ranked

        if args.no_llm:
            llm_ranked = cand
            llm_ok = False
            llm_time_this = 0.0
        else:
            t1 = time.perf_counter()
            try:
                llm_ranked = asyncio.run(_llm_listwise_rerank(q["q"], cand))
            except Exception as e:  # noqa: BLE001
                print(f"  LLM 重排失败：{e}")
                llm_ranked = cand
            llm_time_this = time.perf_counter() - t1
            llm_time += llm_time_this
            llm_ok = _hit(llm_ranked, q["doc"])

        ce_ok = _hit(ce_ranked_first, q["doc"])
        ce_hits += ce_ok
        llm_hits += llm_ok
        rows.append({"q": q["q"], "type": q["type"], "expected": q["doc"],
                     "candidates": len(cand),
                     "ce_hit": ce_ok, "ce_top1": ce_ranked_first[0]["title"] if ce_ranked_first else None,
                     "llm_hit": llm_ok, "llm_top1": llm_ranked[0]["title"] if llm_ranked else None})

    n = len(qs)
    for lim in ce_limits:
        st = ce_stats[lim]
        label = "不限" if lim is None else f"top{lim}"
        print(f"CE {label:<6} 命中 {st['hits']}/{n}  单题 {st['ms'] / n * 1000:.0f}ms")
    if not args.no_llm:
        for key in ("term", "colloquial"):
            sub = [r for r in rows if r["type"] == key]
            print(f"{key:<12} CE {sum(r['ce_hit'] for r in sub)}/{len(sub)}  "
                  f"LLM {sum(r['llm_hit'] for r in sub)}/{len(sub)}")
        print(f"{'总计':<12} CE {ce_hits}/{n}  LLM {llm_hits}/{n}  "
              f"LLM 单题 {llm_time / n * 1000:.0f}ms")

    OUT_PATH.write_text(json.dumps({
        "recall_k": RECALL_K,
        "ce_limits": {("all" if lim is None else lim): {
            "hit_rate": round(st["hits"] / n, 3), "ms_per_query": round(st["ms"] / n * 1000, 1)}
            for lim, st in ce_stats.items()},
        "llm_hit_rate": None if args.no_llm else round(llm_hits / n, 3),
        "llm_ms_per_query": None if args.no_llm else round(llm_time / n * 1000, 1),
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"逐题结果：{OUT_PATH}")


if __name__ == "__main__":
    main()
