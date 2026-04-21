"""ColBERT retrieval via RAGatouille.

Runs batched late-interaction search over a prebuilt index and emits a ranking dict
compatible with the project's TREC run-file writer and ranx evaluator. Records
per-query latency so we can report p50/p95 alongside quality metrics.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..data.schema import Query
from ..utils import get_logger

log = get_logger(__name__)


@dataclass
class LatencyStats:
    n_queries: int
    total_time_s: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


@dataclass
class ColBERTSearcher:
    model: object  # loaded RAGPretrainedModel

    def search(
        self,
        queries: Sequence[Query],
        topk: int = 100,
    ) -> Tuple[Dict[str, List[Tuple[str, float]]], LatencyStats]:
        """Search the prebuilt index. Returns rankings + latency stats."""
        if not queries:
            return {}, LatencyStats(0, 0.0, 0.0, 0.0, 0.0, 0.0)

        rankings: Dict[str, List[Tuple[str, float]]] = {}
        latencies_ms: List[float] = []

        t_total = time.perf_counter()
        for q in queries:
            t0 = time.perf_counter()
            hits = self.model.search(query=q.text, k=topk)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            ranked: List[Tuple[str, float]] = []
            for h in hits:
                # RAGatouille returns dicts with 'document_id' and 'score'.
                doc_id = str(h.get("document_id") or h.get("doc_id") or h.get("id"))
                score = float(h.get("score", 0.0))
                ranked.append((doc_id, score))
            rankings[q.qid] = ranked

        total = time.perf_counter() - t_total
        arr = np.asarray(latencies_ms, dtype=float)
        stats = LatencyStats(
            n_queries=len(queries),
            total_time_s=total,
            mean_ms=float(arr.mean()),
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
        )
        log.info(
            "ColBERT search: %d queries, total=%.1fs, p50=%.1fms, p95=%.1fms",
            stats.n_queries, stats.total_time_s, stats.p50_ms, stats.p95_ms,
        )
        return rankings, stats
