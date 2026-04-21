"""CLI: sweep ColBERT nbits in {1, 2, 4} on SciFact."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.ablation.nbits_sweep import run_sweep  # noqa: E402
from src.data.load_scifact import load_scifact  # noqa: E402
from src.utils import get_logger  # noqa: E402

log = get_logger("run_ablation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ColBERT nbits ablation sweep.")
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--split", default="test")
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--nbits", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--checkpoint", default="colbert-ir/colbertv2.0")
    p.add_argument("--max-doc-len", type=int, default=256)
    p.add_argument("--out-dir", default="results/runs")
    p.add_argument("--metrics-dir", default="results/metrics")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.dataset != "scifact":
        raise NotImplementedError(f"Only scifact supported (got {args.dataset!r})")

    corpus = load_scifact(split=args.split, raw_dir=args.raw_dir)

    records = run_sweep(
        corpus=corpus,
        nbits_values=tuple(args.nbits),
        topk=args.topk,
        checkpoint=args.checkpoint,
        runs_dir=args.out_dir,
        metrics_dir=args.metrics_dir,
        max_document_length=args.max_doc_len,
    )
    log.info("Ablation complete: %d variants succeeded.", len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
