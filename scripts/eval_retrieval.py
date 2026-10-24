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
NEG_OUT_PATH = pathlib.Path(__file__).resolve().parent.parent / "docs" / "negative-check.json"


def _title_of(chunk: dict) -> str:
    return str(chunk.get("title") or (chunk.get("metadata") or {}).get("title") or "")


def _norm(text: str) -> str:
    """去空白、标点、大小写，只留字母数字和汉字。"""
    import re

    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", re.sub(r"[\s\u3000]+", "", str(text).lower()))


def hit_top3(chunks: list[dict], answer: str) -> bool:
    """前 3 个分块里，任一块包含答案原文才算命中（片段级判定）。"""
    want = _norm(answer)
    if not want:
        return False
    return any(want in _norm(c.get("text") or "") for c in chunks[:3])


def run_mode(name: str, qs: list[dict], runner) -> dict:
    hits = 0
    per_type: dict[str, list[int]] = {"term": [], "colloquial": []}
    latency: list[float] = []
    rows = []
    for q in qs:
        t0 = time.perf_counter()
        chunks = runner(q["q"])
        latency.append((time.perf_counter() - t0) * 1000)
        h = hit_top3(chunks, q["answer"])
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
    # 这一档依赖真实模型调用；如果没配 Key 或调用失败，必须让评测直接失败，
    # 否则会得到一份"看起来正常、实际没跑 LLM"的假数据。
    from app.services import settings_store

    if not settings_store.is_configured("research"):
        raise SystemExit("未配置模型，hybrid_full 无法评测；请先配置 Key 或只跑前三档")

    full_meta: list[dict] = []
    pos_scores: list[float] = []

    def _full(q: str):
        chunks, meta = asyncio.run(
            rag_service.query_shelf_hybrid(q, k=3, multi_query=True, rerank=True))
        full_meta.append(meta)
        score = chunks[0].get("rerank_score") if chunks else None
        if score is not None:
            pos_scores.append(float(score))
        return chunks

    results.append(run_mode("hybrid_full", qs, _full))

    failed = [m for m in full_meta if not m.get("reranked")]
    if failed:
        raise SystemExit(
            f"hybrid_full 有 {len(failed)} 题没有真正完成重排，数据不可用；"
            f"首个错误：{failed[0].get('rerank_failed') or failed[0].get('rewrite_failed')}")
    rewritten = sum(1 for m in full_meta if len(m.get("queries", [])) > 1)
    kinds = {m.get("reranked") for m in full_meta}
    models = {m.get("rerank_model") for m in full_meta}
    print(f"\n自检：{rewritten}/{len(full_meta)} 题完成了多查询改写；"
          f"重排方式 {sorted(kinds)}，模型 {sorted(x for x in models if x)}")

    # 负样本：语料里没有答案的问题。只衡量 cross-encoder 的最高分能否区分「有答案 / 没答案」，
    # 不改产品行为——现在检索层不拒答，这里只给一个阈值参考。
    negatives = data.get("negatives", [])
    if negatives and kinds == {"cross_encoder"}:
        neg_rows = []
        for n in negatives:
            chunks, _meta = asyncio.run(
                rag_service.query_shelf_hybrid(n["q"], k=3, multi_query=False, rerank=True))
            score = chunks[0].get("rerank_score") if chunks else None
            neg_rows.append({"q": n["q"], "top1": _title_of(chunks[0]) if chunks else None,
                             "score": float(score) if score is not None else None})
        neg_scores = [r["score"] for r in neg_rows if r["score"] is not None]

        def auc(pos: list[float], neg: list[float]) -> float:
            if not pos or not neg:
                return 0.0
            wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
            return wins / (len(pos) * len(neg))

        cuts, best = sorted(set(pos_scores + neg_scores)), None
        for t in cuts:
            tpr = sum(s >= t for s in pos_scores) / max(1, len(pos_scores))
            tnr = sum(s < t for s in neg_scores) / max(1, len(neg_scores))
            bal = (tpr + tnr) / 2
            if best is None or bal > best["balanced_acc"]:
                best = {"threshold": round(t, 4), "balanced_acc": round(bal, 3),
                        "positive_recall": round(tpr, 3), "negative_reject": round(tnr, 3)}
        NEG_OUT_PATH.write_text(json.dumps({
            "n_positive": len(pos_scores), "n_negative": len(neg_scores),
            "auc": round(auc(pos_scores, neg_scores), 3),
            "positive_median": round(sorted(pos_scores)[len(pos_scores) // 2], 4) if pos_scores else None,
            "negative_median": round(sorted(neg_scores)[len(neg_scores) // 2], 4) if neg_scores else None,
            "best_threshold": best, "negatives": neg_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"负样本检查：AUC {auc(pos_scores, neg_scores):.3f}；"
              f"最佳阈值 {best['threshold'] if best else None}，"
              f"有答案召回 {best['positive_recall'] if best else None} / "
              f"无答案拒绝 {best['negative_reject'] if best else None}")
    elif negatives:
        print("负样本检查跳过：当前重排不是 cross-encoder，没有可比较的分数")

    OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'模式':14s}{'命中率':>8s}{'术语':>8s}{'口语':>8s}{'p50(ms)':>10s}")
    for r in results:
        print(f"{r['mode']:14s}{r['hit_rate']:>8.1%}{r['term_hit_rate']:>8.1%}"
              f"{r['colloquial_hit_rate']:>8.1%}{r['p50_ms']:>10.1f}")


if __name__ == "__main__":
    main()
