"""CLI: generate analysis figures from results/metrics/*.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils import ensure_dir, get_logger  # noqa: E402

log = get_logger("make_figures")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate analysis figures.")
    p.add_argument("--metrics-dir", default="results/metrics")
    p.add_argument("--out-dir", default="results/figures")
    return p.parse_args()


def _load_metric_jsons(metrics_dir: Path) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for path in sorted(metrics_dir.glob("*.json")):
        # Skip index-stats / latency-only files.
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "metrics" in data and "run_name" in data:
            out[data["run_name"]] = data["metrics"]
    return out


def _plot_metric_bars(metric_map: Dict[str, Dict[str, float]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if not metric_map:
        log.warning("No metrics to plot.")
        return
    systems = list(metric_map.keys())
    metrics = ["ndcg@10", "mrr@10", "map", "recall@100"]

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / len(metrics)
    x = range(len(systems))
    for i, m in enumerate(metrics):
        values = [metric_map[s].get(m, 0.0) for s in systems]
        ax.bar([xi + i * width for xi in x], values, width=width, label=m)
    ax.set_xticks([xi + width * (len(metrics) - 1) / 2 for xi in x])
    ax.set_xticklabels(systems, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Retrieval metrics across systems")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    out_path = out_dir / "metrics_bar.png"
    fig.savefig(out_path, dpi=150)
    log.info("Wrote %s", out_path)
    plt.close(fig)


def _plot_nbits_quality_vs_size(metrics_dir: Path, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    records_path = metrics_dir / "ablation_nbits.json"
    if not records_path.exists():
        log.info("No ablation_nbits.json — skipping nbits plot.")
        return
    with open(records_path, "r", encoding="utf-8") as f:
        records: List[dict] = json.load(f)
    if not records:
        return
    records.sort(key=lambda r: r["nbits"])

    sizes = [r["index_mib"] for r in records]
    ndcgs = [r["metrics"].get("ndcg@10", 0.0) for r in records]
    labels = [f"nbits={r['nbits']}" for r in records]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sizes, ndcgs, marker="o", linewidth=2)
    for x, y, lab in zip(sizes, ndcgs, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("Index size (MiB)")
    ax.set_ylabel("NDCG@10")
    ax.set_title("ColBERT residual compression: size vs. quality")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "nbits_size_vs_ndcg.png"
    fig.savefig(out_path, dpi=150)
    log.info("Wrote %s", out_path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    out_dir = ensure_dir(args.out_dir)

    metric_map = _load_metric_jsons(metrics_dir)
    _plot_metric_bars(metric_map, out_dir)
    _plot_nbits_quality_vs_size(metrics_dir, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
