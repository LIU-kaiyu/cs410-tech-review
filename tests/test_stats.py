"""Unit tests for significance stats: paired t-test, bootstrap CI, Holm-Bonferroni."""
from __future__ import annotations

import numpy as np
import pytest

scipy = pytest.importorskip("scipy")

from src.eval.stats import bootstrap_ci, holm_bonferroni, paired_ttest  # noqa: E402


def _per_query(values: dict[str, float], metric: str = "ndcg@10"):
    return {qid: {metric: v} for qid, v in values.items()}


def test_constant_positive_delta_flags_significance():
    """If B is uniformly 0.1 above A on 30 queries, t-test should be wildly significant."""
    a = {f"q{i}": 0.5 for i in range(30)}
    b = {f"q{i}": 0.6 for i in range(30)}
    res = paired_ttest(_per_query(a), _per_query(b), metric="ndcg@10",
                       system_a="A", system_b="B")
    assert res.n == 30
    assert res.mean_delta == pytest.approx(0.1)
    assert res.p_value < 1e-6  # deterministic delta => huge t-stat
    assert res.t_statistic > 0   # B > A


def test_identical_vectors_yield_null_result():
    shared = {f"q{i}": 0.3 for i in range(20)}
    res = paired_ttest(_per_query(shared), _per_query(shared), metric="ndcg@10")
    assert res.p_value == pytest.approx(1.0)
    assert res.t_statistic == pytest.approx(0.0)
    assert res.mean_delta == pytest.approx(0.0)


def test_negative_delta_gives_negative_t():
    a = {f"q{i}": 0.7 for i in range(15)}
    b = {f"q{i}": 0.5 for i in range(15)}
    res = paired_ttest(_per_query(a), _per_query(b), metric="ndcg@10")
    assert res.t_statistic < 0
    assert res.mean_delta == pytest.approx(-0.2)


def test_mismatched_qids_raise():
    a = _per_query({"q1": 0.5, "q2": 0.6})
    b = _per_query({"q1": 0.4, "q3": 0.7})
    with pytest.raises(ValueError, match="mismatched qids"):
        paired_ttest(a, b, metric="ndcg@10")


def test_qid_alignment_is_order_insensitive():
    """t-stat must not depend on dict insertion order — qids align by key."""
    a_ordered = _per_query({"q1": 0.1, "q2": 0.5, "q3": 0.9})
    b_shuffled = _per_query({"q3": 1.0, "q1": 0.2, "q2": 0.6})
    res = paired_ttest(a_ordered, b_shuffled, metric="ndcg@10")
    # deltas are [0.1, 0.1, 0.1] regardless of dict order
    assert res.mean_delta == pytest.approx(0.1)


def test_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(42)
    values = rng.normal(loc=0.3, scale=0.05, size=200)
    lo, hi = bootstrap_ci(values, n_bootstrap=500, seed=0)
    assert lo < float(values.mean()) < hi


def test_holm_bonferroni_orders_correctly():
    # 3 hypotheses at alpha=0.05: thresholds are 0.0167, 0.025, 0.05 (sorted)
    p = [0.01, 0.02, 0.04]
    decisions = holm_bonferroni(p, alpha=0.05)
    # 0.01 <= 0.0167 → True; 0.02 <= 0.025 → True; 0.04 <= 0.05 → True
    assert decisions == [True, True, True]


def test_holm_bonferroni_stops_on_first_failure():
    # Sorted: 0.01, 0.03, 0.04 — thresholds 0.0167, 0.025, 0.05.
    # 0.01 passes; 0.03 > 0.025 fails; 0.04 forbidden even though 0.04 <= 0.05.
    p = [0.01, 0.03, 0.04]
    decisions = holm_bonferroni(p, alpha=0.05)
    assert decisions == [True, False, False]


def test_holm_bonferroni_preserves_input_order():
    p = [0.5, 0.001, 0.2]  # sorted order is index 1, 2, 0
    decisions = holm_bonferroni(p, alpha=0.05)
    # Only p=0.001 should pass; it sits at index 1.
    assert decisions[1] is True
    assert decisions[0] is False
    assert decisions[2] is False
