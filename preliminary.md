# Preliminary Report — ColBERTv2 vs. BM25 on BEIR SciFact

**Course:** CS410 Technical Review
**Author:** Kyle B.
**Date:** 2026-05-03
**Status:** Work-in-progress; final report pending.

## 1. Summary

This project benchmarks ColBERTv2 late-interaction retrieval (via the
RAGatouille wrapper) against a BM25 baseline on the BEIR SciFact dataset
(5,183 docs / 300 test queries) and ablates ColBERTv2's residual compression
parameter `nbits ∈ {1, 2, 4}`. Headline numbers from runs completed to date:
ColBERT (`nbits=2`) improves over BM25 by **+2.6 NDCG@10**, **+3.3 MRR@10**,
**+3.3 MAP**, and **+4.0 Recall@100** percentage points on SciFact test.
Significance testing, the `nbits=1` ablation point, and an investigation of
the surprisingly identical `nbits=2` vs. `nbits=4` results are still
outstanding.

## 2. Methods

- **Dataset:** BEIR SciFact (test split: 300 queries, 5,183 documents,
  3 graded relevance levels). Loaded via the BEIR loader and normalized to
  JSONL through [src/data/load_scifact.py](src/data/load_scifact.py).
- **BM25 baseline:** `rank_bm25` with `k1=1.5, b=0.75, ε=0.25`, lowercased,
  English stopwords removed, `min_token_len=2`. Implemented in
  [src/baselines/bm25.py](src/baselines/bm25.py).
- **ColBERTv2:** Checkpoint `colbert-ir/colbertv2.0` (110M-param BERT-base
  encoder producing 128-dim per-token vectors), `query_maxlen=32`,
  `doc_maxlen=256`. Indexed and queried via RAGatouille
  ([src/colbert/index.py](src/colbert/index.py),
  [src/colbert/search.py](src/colbert/search.py)).
- **Ablation:** [src/ablation/nbits_sweep.py](src/ablation/nbits_sweep.py)
  builds one index per `nbits` value, runs all 300 test queries, and emits
  per-variant TREC run files plus a summary CSV/JSON.
- **Evaluation:** All metrics computed by **ranx**
  ([src/eval/metrics.py](src/eval/metrics.py)) — NDCG@10, MRR@10, MAP,
  Recall@100. Per-query metric vectors are persisted to JSON so paired
  significance tests do not require re-running retrieval.
- **Significance testing:** Paired Student's t-test on per-query metric
  vectors (`scipy.stats.ttest_rel`) with Holm-Bonferroni correction across
  the family of comparisons, plus 2,000-resample bootstrap 95% CIs.
  Implementation: [src/eval/stats.py](src/eval/stats.py)
  (`paired_ttest`, `bootstrap_ci`, `holm_bonferroni`). Validated by 9 unit
  tests in [tests/test_stats.py](tests/test_stats.py) covering null deltas,
  qid alignment/order invariance, bootstrap CI bounds, and the
  Holm-Bonferroni step-down algorithm; all 9 pass under the project's
  Python 3.10 / scipy 1.11 stack. The infrastructure has not yet been
  *applied* to the BM25-vs-ColBERT runs (that step lives in
  [notebooks/04_analysis_and_figures.ipynb](notebooks/04_analysis_and_figures.ipynb)
  and is the §5.1 next action).
- **Hardware:** GPU run (CUDA). Full ColBERT indexing of SciFact at one
  `nbits` setting builds in ~19 s; querying all 300 test queries finishes
  in 100–140 s.

## 3. Current Results

### 3.1 Aggregate retrieval metrics — SciFact test (n=300)

| System              | NDCG@10 | MRR@10 | MAP   | Recall@100 |
| ------------------- | ------: | -----: | ----: | ---------: |
| BM25                |  0.6641 | 0.6317 | 0.6251 |     0.8759 |
| ColBERTv2, nbits=2  |  **0.6898** | **0.6646** | **0.6577** |     **0.9160** |
| ColBERTv2, nbits=4  |  0.6898 | 0.6646 | 0.6577 |     0.9160 |
| ColBERTv2, nbits=1  |  *(not run)* | — | — | — |

ColBERTv2 outperforms BM25 on every metric. The gain is largest on
Recall@100 (+4.0 pp), consistent with late interaction's known strength on
queries whose relevant documents use paraphrased or semantically related
vocabulary rather than exact term overlap — exactly the pattern SciFact's
biomedical claim-evidence pairs exhibit.

### 3.2 Index footprint and query latency

| nbits | Build time | Index size | Search total | Latency p50 | Latency p95 |
| :---: | ---------: | ---------: | -----------: | ----------: | ----------: |
|   2   |     18.58 s |    84.95 MiB |       140.05 s |       407.6 ms |       700.5 ms |
|   4   |     19.75 s |    84.95 MiB |       102.81 s |       301.0 ms |       596.5 ms |

Both indexes were built independently (different on-disk paths, different
build times), yet produced byte-identical sizes (`index_bytes = 89,076,819`).
See §4 for why this is interesting.

### 3.3 Figures generated

Located in [results/figures/](results/figures/):

- `bm25_ndcg_hist.png` — per-query NDCG@10 distribution under BM25.
- `ablation_size_vs_ndcg.png` — index size vs. NDCG@10 across `nbits` values.
- `ablation_metrics_bar.png` — multi-metric bar chart (BM25 vs. each
  ColBERTv2 variant).
- `ablation_latency.png` — query latency p50/p95 vs. `nbits`.

## 4. Open Question — `nbits=2` ≡ `nbits=4`?

The `nbits=2` and `nbits=4` results are **identical to four decimal places
on every reported metric**, despite being produced from independently built
indexes that observably differ in latency (140 s vs. 103 s search total).
Index sizes are also identical to the byte. Three plausible explanations:

1. **Real null result.** At SciFact's scale (5K docs), per-token residual
   quantization noise is small relative to the late-interaction signal, so
   the candidate generation and final MaxSim re-ranking agree exactly on the
   top-100 ordering for every query. The latency delta would then come from
   faster decompression at higher `nbits` (less computation in the residual
   decoder despite the same on-disk footprint).
2. **RAGatouille / colbert-ai parameter passthrough issue.** The `nbits`
   value may not be propagated correctly to the underlying residual codec,
   so both indexes end up encoded with the same effective compression.
3. **ColBERTv2 internal cap.** The ColBERTv2 residual codec might cap
   precision below 4 bits regardless of the requested `nbits`.

This needs to be resolved before the final report makes any claim about the
size/quality trade-off. The investigation plan: read
`colbert.indexing.codecs.residual.ResidualCodec`, inspect on-disk centroid
and residual files for both indexes, and either confirm a real null result
(which is itself a publishable observation) or file a dependency issue.

## 5. What Remains

In priority order:

1. **Run notebook 04 to completion.** Both the test harness and the four
   planned report figures are already implemented; this step *applies* the
   already-validated `paired_ttest` and `holm_bonferroni` functions from
   [src/eval/stats.py](src/eval/stats.py) to the actual BM25 and ColBERTv2
   per-query metric vectors (n=300 paired observations) and renders the
   figures from the resulting numbers. Output: corrected p-values and
   bootstrap CIs for the §3.1 deltas.
2. **Resolve §4** (the `nbits=2`/`nbits=4` equivalence) by inspection.
3. **Add `nbits=1`** to the ablation. Most aggressive compression point and
   the most informative single missing measurement.
4. **Run notebook 05** ([notebooks/05_colbert_maxsim_viz.ipynb](notebooks/05_colbert_maxsim_viz.ipynb)),
   which loads the same checkpoint outside RAGatouille, encodes
   query 13 (*"5% of perinatal mortality is due to low birth weight"*)
   against gold doc 1606628, and produces a MaxSim heatmap. Sanity check on
   un-quantized FP32 produced a final ColBERT score of **14.78**, matching
   the score recorded in `results/runs/colbert_smoke50.trec` for that pair.
   Intended use: explanatory figure in the report's Background section, not
   a result.
5. **Write the report.** Stand up `report/` with a LaTeX skeleton and
   populate Methods / Results from notebooks 04 and 05.

## 6. Reproducibility Notes

All commands are wired through `make`; canonical entrypoints live in
`scripts/`. To reproduce results to date from a fresh clone:

```bash
make setup          # pip install -r requirements.txt
make data           # downloads SciFact via BEIR
make bm25           # results/runs/bm25.trec
make ablation       # results/runs/colbert_nbits{1,2,4}.trec
make eval EXP=bm25
make eval EXP=colbert_nbits2
# (notebook 04 takes over from here for significance + figures)
```

ColBERTv2 indexing requires CUDA in practice. On Windows, WSL2 is required
because ColBERT's compiled CUDA kernels are unstable on native Windows.
BM25 and the eval harness run fine on CPU.

The full plumbing (data loading, retrievers, evaluators, statistics, tests)
is in place. The remaining work is ablation completeness, significance
testing, the `nbits` investigation, and writing.
