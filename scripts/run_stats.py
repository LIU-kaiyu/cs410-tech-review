"""CLI: pairwise paired t-tests across a set of evaluation JSON files.

Given two or more evaluation JSON files (produced by run_eval.py / run_bm25.py /
run_colbert.py / run_ablation.py), compute pairwise paired Student's t-tests on the
shared per-query metric vectors, apply Holm-Bonferroni correction per metric family,
and write a summary CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.eval.stats import holm_bonferroni, paired_ttest  # noqa: E402
from src.utils import ensure_dir, get_logger  # noqa: E402

log = get_logger("run_stats")


def _load_per_query(path: Path) -> tuple[str, Dict[str, Dict[str, float]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    name = data.get("run_name", path.stem)
    per_query = data.get("per_query", {})
    if not per_query:
        raise ValueError(f"{path}: no per_query block found.")
    return name, per_query


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pairwise paired t-tests across run evals.")
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Evaluation JSON files (from save_eval).")
    p.add_argument("--metrics", nargs="+", default=["ndcg@10", "mrr@10"])
    p.add_argument("--out", default="results/metrics/pairwise_ttests.csv")
    p.add_argument("--alpha", type=float, default=0.05)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    loaded = [_load_per_query(Path(p)) for p in args.inputs]

    rows: List[dict] = []
    for metric in args.metrics:
        metric_rows: List[dict] = []
        p_values: List[float] = []
        for (name_a, a), (name_b, b) in combinations(loaded, 2):
            res = paired_ttest(a, b, metric=metric, system_a=name_a, system_b=name_b)
            metric_rows.append({
                "metric": res.metric,
                "system_a": res.system_a,
                "system_b": res.system_b,
                "n": res.n,
                "mean_a": res.mean_a,
                "mean_b": res.mean_b,
                "mean_delta": res.mean_delta,
                "t_statistic": res.t_statistic,
                "p_value": res.p_value,
                "ci_low": res.ci_low,
                "ci_high": res.ci_high,
            })
            p_values.append(res.p_value)

        decisions = holm_bonferroni(p_values, alpha=args.alpha)
        for row, sig in zip(metric_rows, decisions):
            row["holm_significant"] = sig
        rows.extend(metric_rows)

    out = Path(args.out)
    ensure_dir(out.parent)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %d rows to %s", len(rows), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
