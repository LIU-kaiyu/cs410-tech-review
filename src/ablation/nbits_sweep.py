"""Ablation sweep over ColBERT residual-compression `nbits`.

For each `nbits` in the sweep: rebuild the index, run all queries, evaluate with
ranx, and record index size, build time, query latency, NDCG@10, MRR@10, MAP.
Results are emitted as CSV + per-variant JSON so downstream plotting is trivial.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

from ..colbert.index import ColBERTIndexConfig, ColBERTIndexer, save_index_stats
from ..colbert.search import ColBERTSearcher
from ..data.schema import Corpus
from ..eval.metrics import DEFAULT_METRICS, evaluate_run, save_eval, write_run
from ..utils import ensure_dir, get_logger

log = get_logger(__name__)


@dataclass
class AblationRecord:
    nbits: int
    n_docs: int
    n_queries: int
    build_time_s: float
    index_mib: float
    search_total_s: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    metrics: Dict[str, float] = field(default_factory=dict)

    def flat(self) -> Dict[str, float | int | str]:
        out: Dict[str, float | int | str] = {
            "nbits": self.nbits,
            "n_docs": self.n_docs,
            "n_queries": self.n_queries,
            "build_time_s": self.build_time_s,
            "index_mib": self.index_mib,
            "search_total_s": self.search_total_s,
            "latency_mean_ms": self.latency_mean_ms,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
        }
        for k, v in self.metrics.items():
            out[f"metric_{k}"] = v
        return out


def run_sweep(
    corpus: Corpus,
    nbits_values: Sequence[int] = (1, 2, 4),
    topk: int = 100,
    checkpoint: str = "colbert-ir/colbertv2.0",
    index_name_prefix: str = "scifact",
    runs_dir: str | Path = "results/runs",
    metrics_dir: str | Path = "results/metrics",
    max_document_length: int = 256,
) -> List[AblationRecord]:
    """Run the full nbits sweep. Returns one record per successful variant."""
    ensure_dir(runs_dir)
    ensure_dir(metrics_dir)
    qrels_map = corpus.qrels_map()

    records: List[AblationRecord] = []
    for nbits in nbits_values:
        variant = f"nbits{nbits}"
        try:
            log.info("=== Ablation variant: %s ===", variant)
            index_cfg = ColBERTIndexConfig(
                checkpoint=checkpoint,
                index_name=f"{index_name_prefix}_{variant}",
                nbits=nbits,
                max_document_length=max_document_length,
            )
            indexer = ColBERTIndexer(config=index_cfg)
            index_stats = indexer.build_index(corpus.docs, overwrite=True)
            save_index_stats(
                index_stats,
                Path(metrics_dir) / f"index_{variant}.json",
            )

            searcher = ColBERTSearcher(model=indexer._model)
            rankings, latency = searcher.search(corpus.queries, topk=topk)

            run_path = Path(runs_dir) / f"colbert_{variant}.trec"
            write_run(run_path, rankings, tag=f"colbert_{variant}")

            eval_result = evaluate_run(
                qrels_map=qrels_map,
                run=rankings,
                metrics=DEFAULT_METRICS,
                run_name=f"colbert_{variant}",
            )
            save_eval(eval_result, Path(metrics_dir) / f"colbert_{variant}.json")

            rec = AblationRecord(
                nbits=nbits,
                n_docs=index_stats.n_docs,
                n_queries=latency.n_queries,
                build_time_s=index_stats.build_time_s,
                index_mib=index_stats.index_bytes / (1024 * 1024),
                search_total_s=latency.total_time_s,
                latency_mean_ms=latency.mean_ms,
                latency_p50_ms=latency.p50_ms,
                latency_p95_ms=latency.p95_ms,
                metrics=eval_result.metrics,
            )
            records.append(rec)
        except Exception as exc:  # keep sweep alive on per-variant failure
            log.exception("Variant %s failed: %s", variant, exc)

    summary_path = Path(metrics_dir) / "ablation_nbits.csv"
    _write_summary_csv(records, summary_path)
    _write_summary_json(records, Path(metrics_dir) / "ablation_nbits.json")
    return records


def _write_summary_csv(records: Sequence[AblationRecord], path: Path) -> None:
    ensure_dir(path.parent)
    if not records:
        log.warning("No ablation records to write.")
        return
    headers = list(records[0].flat().keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in records:
            writer.writerow(r.flat())
    log.info("Wrote ablation CSV: %s", path)


def _write_summary_json(records: Sequence[AblationRecord], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, indent=2)
    log.info("Wrote ablation JSON: %s", path)
