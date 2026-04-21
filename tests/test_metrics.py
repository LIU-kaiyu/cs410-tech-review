"""Unit tests for TREC run I/O and the ranx-backed evaluator."""
from __future__ import annotations

from pathlib import Path

import pytest

ranx = pytest.importorskip("ranx")

from src.eval.metrics import evaluate_run, load_run, write_run  # noqa: E402


def test_write_then_load_run_roundtrip(tmp_path: Path):
    rankings = {
        "q1": [("d1", 3.2), ("d2", 1.4)],
        "q2": [("d3", 5.0), ("d1", 2.7), ("d2", 0.1)],
    }
    path = tmp_path / "run.trec"
    written = write_run(path, rankings, tag="unit")
    assert written == 5

    loaded = load_run(path)
    # Ordering should be score-descending per qid.
    assert [d for d, _ in loaded["q1"]] == ["d1", "d2"]
    assert [d for d, _ in loaded["q2"]] == ["d3", "d1", "d2"]


def test_perfect_ranking_yields_top_scores():
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    rankings = {
        "q1": [("d1", 1.0), ("d2", 0.1), ("d3", 0.05)],
        "q2": [("d2", 1.0), ("d1", 0.1), ("d3", 0.05)],
    }
    result = evaluate_run(qrels, rankings, metrics=("ndcg@10", "mrr@10"))
    assert result.metrics["ndcg@10"] == pytest.approx(1.0)
    assert result.metrics["mrr@10"] == pytest.approx(1.0)


def test_reversed_ranking_lowers_mrr():
    qrels = {"q1": {"d1": 1}}
    good = {"q1": [("d1", 1.0), ("d2", 0.5), ("d3", 0.1)]}
    bad = {"q1": [("d3", 1.0), ("d2", 0.5), ("d1", 0.1)]}
    g = evaluate_run(qrels, good, metrics=("mrr@10",))
    b = evaluate_run(qrels, bad, metrics=("mrr@10",))
    assert g.metrics["mrr@10"] > b.metrics["mrr@10"]
    # Rank 3 → 1/3
    assert b.metrics["mrr@10"] == pytest.approx(1.0 / 3.0)


def test_per_query_metrics_present_for_each_qid():
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    rankings = {
        "q1": [("d1", 1.0), ("d2", 0.5)],
        "q2": [("d2", 1.0), ("d1", 0.5)],
    }
    result = evaluate_run(qrels, rankings, metrics=("ndcg@10",))
    assert set(result.per_query.keys()) == {"q1", "q2"}
    for qid in result.per_query:
        assert "ndcg@10" in result.per_query[qid]
