"""CLI: evaluate a TREC run file against SciFact qrels."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.load_scifact import load_scifact  # noqa: E402
from src.eval.metrics import DEFAULT_METRICS, evaluate_run, load_run, save_eval  # noqa: E402
from src.utils import get_logger  # noqa: E402

log = get_logger("run_eval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a TREC run file.")
    p.add_argument("--run", required=True, help="Path to TREC-format run file")
    p.add_argument("--out", required=True, help="Output JSON metrics path")
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--split", default="test")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.dataset != "scifact":
        raise NotImplementedError(f"Only scifact supported (got {args.dataset!r})")

    corpus = load_scifact(split=args.split, raw_dir=args.raw_dir)
    rankings = load_run(args.run)

    run_name = Path(args.run).stem
    result = evaluate_run(
        qrels_map=corpus.qrels_map(),
        run=rankings,
        metrics=args.metrics,
        run_name=run_name,
    )
    save_eval(result, args.out)

    log.info("%s metrics: %s", run_name, result.metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
