"""CLI: build BM25 index over SciFact and emit a TREC run file."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.baselines.bm25 import BM25Config, BM25Retriever  # noqa: E402
from src.data.load_scifact import load_scifact  # noqa: E402
from src.eval.metrics import DEFAULT_METRICS, evaluate_run, save_eval, write_run  # noqa: E402
from src.utils import Timer, ensure_dir, get_logger  # noqa: E402

log = get_logger("run_bm25")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BM25 baseline retrieval.")
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--split", default="test")
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out", default="results/runs/bm25.trec")
    p.add_argument("--metrics-out", default="results/metrics/bm25.json")
    p.add_argument("--k1", type=float, default=1.5)
    p.add_argument("--b", type=float, default=0.75)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.dataset != "scifact":
        raise NotImplementedError(f"Only scifact supported (got {args.dataset!r})")

    corpus = load_scifact(split=args.split, raw_dir=args.raw_dir)

    retriever = BM25Retriever(config=BM25Config(k1=args.k1, b=args.b))
    with Timer("bm25 index"):
        retriever.index(corpus.docs)

    with Timer("bm25 search"):
        rankings = retriever.search_all(corpus.queries, topk=args.topk)

    ensure_dir(Path(args.out).parent)
    write_run(args.out, rankings, tag="bm25")

    result = evaluate_run(
        qrels_map=corpus.qrels_map(),
        run=rankings,
        metrics=DEFAULT_METRICS,
        run_name="bm25",
    )
    save_eval(result, args.metrics_out)

    log.info("BM25 metrics: %s", result.metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
