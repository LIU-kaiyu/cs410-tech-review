"""Summarize completed ColBERT nbits runs without rebuilding indexes."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils import ensure_dir, get_logger  # noqa: E402

log = get_logger("summarize_ablation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize existing ColBERT nbits results.")
    p.add_argument("--nbits", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--metrics-dir", default="results/metrics")
    return p.parse_args()


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    ensure_dir(metrics_dir)

    records: list[dict] = []
    for nbits in args.nbits:
        metrics_path = metrics_dir / f"colbert_nbits{nbits}.json"
        latency_path = metrics_dir / f"colbert_nbits{nbits}_latency.json"
        index_path = metrics_dir / f"index_scifact_nbits{nbits}.json"

        missing = [p for p in (metrics_path, latency_path, index_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing files for nbits={nbits}: {missing}")

        metrics = _read_json(metrics_path)
        latency = _read_json(latency_path)
        index = _read_json(index_path)

        records.append(
            {
                "nbits": nbits,
                "effective_nbits": index["effective_nbits"],
                "n_docs": index["n_docs"],
                "n_queries": latency["n_queries"],
                "build_time_s": index["build_time_s"],
                "index_mib": index["index_mib"],
                "search_total_s": latency["total_time_s"],
                "latency_mean_ms": latency["mean_ms"],
                "latency_p50_ms": latency["p50_ms"],
                "latency_p95_ms": latency["p95_ms"],
                "metrics": metrics["metrics"],
            }
        )

    json_path = metrics_dir / "ablation_nbits.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    csv_path = metrics_dir / "ablation_nbits.csv"
    headers = [
        "nbits",
        "effective_nbits",
        "n_docs",
        "n_queries",
        "build_time_s",
        "index_mib",
        "search_total_s",
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "metric_ndcg@10",
        "metric_mrr@10",
        "metric_map",
        "metric_recall@100",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        for record in records:
            row = {k: record[k] for k in headers if k in record}
            for metric, value in record["metrics"].items():
                row[f"metric_{metric}"] = value
            writer.writerow(row)

    log.info("Wrote %s and %s", json_path, csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
