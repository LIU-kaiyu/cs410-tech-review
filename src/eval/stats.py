"""Statistical significance helpers.

Paired Student's t-test on per-query metric vectors, bootstrap 95% CIs on the mean
delta, and Holm-Bonferroni correction for multiple-comparison adjustments. Used to
compare BM25 vs. ColBERT and pairwise across ablation variants.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    from ..utils import get_logger
except ImportError:
    import logging
    def get_logger(name: str):  # type: ignore[misc]
        return logging.getLogger(name)

log = get_logger(__name__)


@dataclass
class PairedTestResult:
    metric: str
    system_a: str
    system_b: str
    n: int
    mean_a: float
    mean_b: float
    mean_delta: float
    t_statistic: float
    p_value: float
    ci_low: float
    ci_high: float


def _align_per_query(
    a: Dict[str, Dict[str, float]],
    b: Dict[str, Dict[str, float]],
    metric: str,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Align two per-query metric maps by qid. Raises if the qid sets diverge."""
    qa = set(a.keys())
    qb = set(b.keys())
    if qa != qb:
        missing_in_b = sorted(qa - qb)[:3]
        missing_in_a = sorted(qb - qa)[:3]
        raise ValueError(
            "Per-query metric maps have mismatched qids. "
            f"Missing from B: {missing_in_b}... Missing from A: {missing_in_a}..."
        )
    qids = sorted(qa)
    va = np.array([a[q][metric] for q in qids], dtype=float)
    vb = np.array([b[q][metric] for q in qids], dtype=float)
    return qids, va, vb


def paired_ttest(
    a: Dict[str, Dict[str, float]],
    b: Dict[str, Dict[str, float]],
    metric: str,
    system_a: str = "A",
    system_b: str = "B",
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> PairedTestResult:
    """Two-sided paired Student's t-test on per-query metric vectors.

    Delta is defined as B - A, so positive means B improves over A.
    """
    from scipy import stats as sp_stats

    qids, va, vb = _align_per_query(a, b, metric)
    deltas = vb - va
    n = len(deltas)

    if np.allclose(deltas, 0.0):
        t_stat, p_val = 0.0, 1.0
    else:
        t_stat, p_val = sp_stats.ttest_rel(vb, va)

    ci_low, ci_high = bootstrap_ci(deltas, n_bootstrap=n_bootstrap, seed=seed)

    return PairedTestResult(
        metric=metric,
        system_a=system_a,
        system_b=system_b,
        n=n,
        mean_a=float(va.mean()),
        mean_b=float(vb.mean()),
        mean_delta=float(deltas.mean()),
        t_statistic=float(t_stat),
        p_value=float(p_val),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
    )


def bootstrap_ci(
    values: Sequence[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap CI on the mean of `values`."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"))
    draws = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = arr[draws].mean(axis=1)
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return lo, hi


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> List[bool]:
    """Return per-hypothesis significance decisions under Holm-Bonferroni at level alpha.

    Input order is preserved in the output.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    decisions = [False] * m
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if p[idx] <= threshold:
            decisions[idx] = True
        else:
            # Once one fails, all subsequent (larger p) also fail.
            break
    return decisions
