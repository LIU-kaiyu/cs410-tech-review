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


def _plot_bm25_colbert_deltas(metrics_dir: Path, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    bm25_path = metrics_dir / "bm25.json"
    colbert_path = metrics_dir / "colbert_nbits2.json"
    if not bm25_path.exists() or not colbert_path.exists():
        log.info("Missing BM25 or ColBERT nbits=2 metrics — skipping delta plot.")
        return

    with open(bm25_path, "r", encoding="utf-8") as f:
        bm25 = json.load(f)["metrics"]
    with open(colbert_path, "r", encoding="utf-8") as f:
        colbert = json.load(f)["metrics"]

    metrics = ["ndcg@10", "mrr@10", "map", "recall@100"]
    labels = ["NDCG@10", "MRR@10", "MAP", "Recall@100"]
    deltas = [colbert[m] - bm25[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, deltas, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"])
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("ColBERT nbits=2 minus BM25")
    ax.set_title("Where ColBERT improves over BM25")
    ax.grid(axis="y", alpha=0.25)
    for bar, delta in zip(bars, deltas):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            delta,
            f"+{delta:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    out_path = out_dir / "bm25_colbert_delta.png"
    fig.savefig(out_path, dpi=150)
    log.info("Wrote %s", out_path)
    plt.close(fig)


def _plot_nbits_latency(metrics_dir: Path, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    records_path = metrics_dir / "ablation_nbits.json"
    if not records_path.exists():
        log.info("No ablation_nbits.json — skipping latency plot.")
        return
    with open(records_path, "r", encoding="utf-8") as f:
        records: List[dict] = json.load(f)
    if not records:
        return
    records.sort(key=lambda r: r["nbits"])

    nbits = [r["nbits"] for r in records]
    p50 = [r["latency_p50_ms"] for r in records]
    p95 = [r["latency_p95_ms"] for r in records]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(nbits, p50, marker="o", linewidth=2, label="p50")
    ax.plot(nbits, p95, marker="o", linewidth=2, label="p95")
    ax.set_xticks(nbits)
    ax.set_xlabel("Residual-compression nbits")
    ax.set_ylabel("Latency (ms/query)")
    ax.set_title("ColBERT query latency across nbits")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / "nbits_latency.png"
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
    _plot_bm25_colbert_deltas(metrics_dir, out_dir)
    _plot_nbits_latency(metrics_dir, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
