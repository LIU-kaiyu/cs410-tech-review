"""Evaluation metrics via `ranx`.

Handles TREC-format run-file I/O and wraps `ranx` for NDCG@10, MRR@10, MAP, and
Recall@100. Persists both aggregated and per-query metrics so downstream significance
tests (see `stats.py`) have what they need.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from ..utils import ensure_dir, get_logger

log = get_logger(__name__)

DEFAULT_METRICS = ("ndcg@10", "mrr@10", "map", "recall@100")


# ---------- TREC run-file I/O ------------------------------------------------

def write_run(
    path: str | Path,
    rankings: Dict[str, Sequence[Tuple[str, float]]],
    tag: str = "run",
) -> int:
    """Write a TREC-format run file: ``qid Q0 doc_id rank score tag``."""
    p = Path(path)
    ensure_dir(p.parent)
    lines = 0
    with open(p, "w", encoding="utf-8") as f:
        for qid, hits in rankings.items():
            for rank, (doc_id, score) in enumerate(hits, start=1):
                f.write(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score:.6f}\t{tag}\n")
                lines += 1
    log.info("Wrote %d lines to %s", lines, p)
    return lines


def load_run(path: str | Path) -> Dict[str, List[Tuple[str, float]]]:
    """Load a TREC run file into {qid: [(doc_id, score), ...]}."""
    p = Path(path)
    out: Dict[str, List[Tuple[str, float]]] = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _q0, doc_id, _rank, score, _tag = parts[:6]
            out.setdefault(qid, []).append((doc_id, float(score)))
    # Preserve descending score order within each qid.
    for qid, hits in out.items():
        hits.sort(key=lambda x: -x[1])
    return out


# ---------- Evaluation -------------------------------------------------------

@dataclass
class EvalResult:
    run_name: str
    metrics: Dict[str, float]
    per_query: Dict[str, Dict[str, float]]

    def to_dict(self) -> dict:
        return {
            "run_name": self.run_name,
            "metrics": self.metrics,
            "per_query": self.per_query,
        }


def _to_ranx(
    qrels_map: Dict[str, Dict[str, int]],
    run: Dict[str, Sequence[Tuple[str, float]]],
):
    """Convert to ranx Qrels/Run objects. Imported lazily to keep test startup cheap."""
    from ranx import Qrels, Run

    qrels_ranx = Qrels(qrels_map)
    run_dict: Dict[str, Dict[str, float]] = {
        qid: {doc_id: float(score) for doc_id, score in hits} for qid, hits in run.items()
    }
    run_ranx = Run(run_dict)
    return qrels_ranx, run_ranx


def evaluate_run(
    qrels_map: Dict[str, Dict[str, int]],
    run: Dict[str, Sequence[Tuple[str, float]]],
    metrics: Iterable[str] = DEFAULT_METRICS,
    run_name: str = "run",
) -> EvalResult:
    """Compute aggregate and per-query metrics via ranx."""
    from ranx import evaluate

    metrics = list(metrics)
    qrels_ranx, run_ranx = _to_ranx(qrels_map, run)

    agg = evaluate(qrels_ranx, run_ranx, metrics=metrics, return_mean=True)
    if isinstance(agg, float):
        agg = {metrics[0]: agg}

    per_q_raw = evaluate(qrels_ranx, run_ranx, metrics=metrics, return_mean=False)
    # ranx returns {metric: [per-query values]} aligned to qrels qid order.
    qids = list(qrels_ranx.qrels.keys())
    per_query: Dict[str, Dict[str, float]] = {qid: {} for qid in qids}
    if isinstance(per_q_raw, dict):
        for metric, values in per_q_raw.items():
            for qid, v in zip(qids, values):
                per_query[qid][metric] = float(v)
    else:  # single-metric edge case
        for qid, v in zip(qids, per_q_raw):
            per_query[qid][metrics[0]] = float(v)

    return EvalResult(
        run_name=run_name,
        metrics={k: float(v) for k, v in agg.items()},
        per_query=per_query,
    )


def save_eval(result: EvalResult, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    log.info("Saved metrics to %s", p)
