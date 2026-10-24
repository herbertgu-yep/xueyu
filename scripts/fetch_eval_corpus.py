"""下载并校验检索评测语料（20 篇 arXiv 论文）。

每个文件都带一个「必须出现在正文前两页」的关键词：下载完立刻核对，
防止把 A 论文存成 B 的名字——那种错配会让评测结果失去意义。

用法：
  python -m scripts.fetch_eval_corpus --out ~/.xueyu_eval_docs
  python -m scripts.fetch_eval_corpus --out ~/.xueyu_eval_docs --ingest --data ~/.xueyu_eval_clean
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# (文件名, arXiv id, 前两页必须出现的关键词)
CORPUS = [
    ("attention-is-all-you-need", "1706.03762", "attention is all you need"),
    ("rag-lewis-2020", "2005.11401", "retrieval-augmented generation"),
    ("react-yao-2022", "2210.03629", "react"),
    ("chain-of-thought-wei", "2201.11903", "chain-of-thought"),
    ("tree-of-thoughts", "2305.10601", "tree of thoughts"),
    ("gpt4-report", "2303.08774", "gpt-4"),
    ("graph-attention-review", "1710.10903", "graph attention networks"),
    ("ddpm-ho", "2006.11239", "denoising diffusion"),
    ("bert", "1810.04805", "bert"),
    ("xlnet", "1906.08237", "xlnet"),
    ("fedavg-survey", "1602.05629", "learning of deep networks"),
    ("gnn-robustness-at-scale", "2110.14038", "robustness of graph neural networks"),
    ("vit", "2010.11929", "image is worth"),
    ("emergent-abilities", "2206.07682", "emergent abilities"),
    ("palm-scaling", "2204.02311", "palm"),
    ("toolformer", "2302.04761", "toolformer"),
    ("reflexion", "2303.11366", "reflexion"),
    ("hugginggpt", "2303.17580", "hugginggpt"),
    ("self-rag", "2310.11511", "self-rag"),
    ("llm-planning-survey", "2402.02716", "planning of llm agents"),
]


def _download(url: str, dest: pathlib.Path) -> None:
    if dest.exists() and dest.stat().st_size > 10_000:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        dest.write_bytes(r.read())


def _verify(path: pathlib.Path, keyword: str) -> bool:
    import pymupdf

    doc = pymupdf.open(path)
    text = " ".join(doc[i].get_text() for i in range(min(2, doc.page_count))).lower()
    doc.close()
    # PDF 里常见连字（ﬃ / ﬁ），比较前统一展开，避免误判
    text = text.replace("ﬃ", "ffi").replace("ﬁ", "fi").replace("ﬂ", "fl")
    return keyword.lower() in text


def fetch(out: pathlib.Path) -> list[pathlib.Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths, bad = [], []
    for name, arxiv_id, keyword in CORPUS:
        dest = out / f"{name}.pdf"
        _download(f"https://arxiv.org/pdf/{arxiv_id}", dest)
        ok = _verify(dest, keyword)
        if not ok:
            bad.append(f"{name}（应为 {arxiv_id}，关键词「{keyword}」未出现）")
        paths.append(dest)
        print(f"  {name:<28} {dest.stat().st_size / 1e6:6.1f}MB  {'ok' if ok else 'FAIL'}")
    if bad:
        raise SystemExit("语料校验失败，以下文件内容与文件名不符：\n  " + "\n  ".join(bad))
    print(f"语料校验通过：{len(paths)} 篇")
    return paths


def ingest(paths: list[pathlib.Path], data_dir: pathlib.Path) -> None:
    import os

    os.environ["XUEYU_DATA"] = str(data_dir)

    from app.storage.db import init_db
    from app.services import shelf_service

    init_db()
    for p in paths:
        doc = shelf_service.save_upload(p.name, p.read_bytes(), title=p.stem)
        print(f"  入库 {doc['title']:<28} chunks={doc.get('chunk_count')} indexed={doc.get('indexed')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(pathlib.Path.home() / ".xueyu_eval_docs"))
    ap.add_argument("--ingest", action="store_true", help="下载后导入指定数据目录并建索引")
    ap.add_argument("--data", default=str(pathlib.Path.home() / ".xueyu_eval_clean"))
    args = ap.parse_args()

    paths = fetch(pathlib.Path(args.out).expanduser())
    if args.ingest:
        ingest(paths, pathlib.Path(args.data).expanduser())


if __name__ == "__main__":
    main()
