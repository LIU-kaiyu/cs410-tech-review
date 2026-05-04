# Final Progress Report — ColBERTv2 vs. BM25 on BEIR SciFact

**Course:** CS410 Technical Review
**Authors:** Kaiyu Liu and Yuyang Zeng
**Date:** 2026-05-04
**Status:** Experiments, metrics, figures, and report draft are complete. Final
PDF compilation remains, because this local machine does not currently have a
TeX compiler installed.

## 1. Summary

This project benchmarks ColBERTv2 late-interaction retrieval through the
RAGatouille wrapper against a BM25 baseline on the BEIR SciFact dataset
(5,183 documents / 300 test queries). It also ablates ColBERTv2 residual
compression with `nbits ∈ {1, 2, 4}`.

The final experiment shows that ColBERTv2 improves over BM25 on all measured
retrieval metrics, though the paired tests do not reach Holm-corrected
statistical significance on this 300-query SciFact test set. The `nbits`
variants produce identical rankings and identical measured index sizes in this
small-corpus setup after verifying that RAGatouille received the requested
`effective_nbits` values.

## 2. Methods

- **Dataset:** BEIR SciFact test split, loaded through the BEIR loader and
  normalized to JSONL with [src/data/load_scifact.py](src/data/load_scifact.py).
- **BM25 baseline:** `rank_bm25` with `k1=1.5`, `b=0.75`, `epsilon=0.25`,
  lowercasing, English stopword removal, and `min_token_len=2`. Implementation:
  [src/baselines/bm25.py](src/baselines/bm25.py).
- **ColBERTv2:** Checkpoint `colbert-ir/colbertv2.0`, `query_maxlen=32`,
  `doc_maxlen=256`, indexed and queried through RAGatouille. Implementation:
  [src/colbert/index.py](src/colbert/index.py) and
  [src/colbert/search.py](src/colbert/search.py).
- **Ablation:** `nbits ∈ {1, 2, 4}` with one index and one TREC run per value.
  The index-stat JSON records both requested `nbits` and observed
  `effective_nbits`.
- **Evaluation:** `ranx` computes NDCG@10, MRR@10, MAP, and Recall@100 in
  [src/eval/metrics.py](src/eval/metrics.py). Per-query metric vectors are
  saved for paired significance tests.
- **Significance testing:** Paired Student's t-test over aligned per-query
  metric vectors, with Holm-Bonferroni correction and bootstrap confidence
  intervals through [src/eval/stats.py](src/eval/stats.py).
- **Environment:** Python 3.10 conda environment `colbert-review`. This run
  used CPU/MPS fallback rather than CUDA, so indexing was substantially slower
  than a GPU run.

## 3. Final Results

### 3.1 Aggregate Retrieval Metrics — SciFact Test (n=300)

| System             | NDCG@10 | MRR@10 | MAP    | Recall@100 |
| ------------------ | ------: | -----: | -----: | ---------: |
| BM25               |  0.6641 | 0.6317 | 0.6251 |     0.8759 |
| ColBERTv2 nbits=1  |  0.6924 | 0.6693 | 0.6623 |     0.9120 |
| ColBERTv2 nbits=2  |  0.6924 | 0.6693 | 0.6623 |     0.9120 |
| ColBERTv2 nbits=4  |  0.6924 | 0.6693 | 0.6623 |     0.9120 |

ColBERTv2 improves over BM25 by about +0.0283 NDCG@10, +0.0376 MRR@10,
+0.0371 MAP, and +0.0361 Recall@100.

### 3.2 Index Footprint and Query Latency

| nbits | Build time | Index size | Search total | Latency p50 | Latency p95 |
| :---: | ---------: | ---------: | -----------: | ----------: | ----------: |
|   1   |   1092.6 s |   84.95 MiB |       22.4 s |     63.1 ms |    188.0 ms |
|   2   |   1284.5 s |   84.95 MiB |       22.6 s |     69.2 ms |     76.5 ms |
|   4   |   1261.4 s |   84.95 MiB |       22.7 s |     66.4 ms |     77.8 ms |

The equal index sizes suggest that, for this small SciFact corpus, fixed
metadata and other index files dominate the measured on-disk footprint. The
identical retrieval metrics indicate that this dataset is not large or
sensitive enough for the selected `nbits` values to change the final top-100
rankings.

### 3.3 Significance Testing

The BM25-to-ColBERT deltas are positive but not Holm-significant:

| Comparison | Metric | Delta | Raw p-value | Holm significant |
| ---------- | ------ | ----: | ----------: | :--------------: |
| BM25 → ColBERT nbits=2 | NDCG@10 | +0.0283 | 0.1148 | No |
| BM25 → ColBERT nbits=2 | MRR@10  | +0.0376 | 0.0540 | No |

Because the three ColBERT variants produced identical per-query scores, their
pairwise deltas are zero.

## 4. Implementation Notes

The main implementation issue was that RAGatouille did not reliably expose
`nbits` as an `index()` keyword argument. The project now sets
`rag_model.model.config.nbits` before indexing and records `effective_nbits` in
the index-stat output. A regression test in
[tests/test_colbert_index.py](tests/test_colbert_index.py) protects this
behavior.

The repo also now includes a lightweight summary script,
[scripts/summarize_ablation.py](scripts/summarize_ablation.py), so existing
ColBERT outputs can be summarized without rebuilding indexes.

## 5. Generated Artifacts

- Metrics JSON/CSV: [results/metrics/](results/metrics/)
- Figures: [results/figures/metrics_bar.png](results/figures/metrics_bar.png)
  and [results/figures/nbits_size_vs_ndcg.png](results/figures/nbits_size_vs_ndcg.png)
- Report source: [report/report.tex](report/report.tex)
- Ignored large local artifacts: SciFact data, RAGatouille indexes, and TREC
  run files.

## 6. What Remains

1. Compile [report/report.tex](report/report.tex) to PDF in Overleaf or after
   installing a local TeX distribution such as MacTeX.
2. Read through the final report prose and adjust any course-specific wording
   or page-length requirements.
3. Optionally add one qualitative example or MaxSim visualization if the final
   review needs more interpretability discussion.
4. Commit and push the finished repo when ready.

## 7. Reproducibility Commands

```bash
conda env create -f environment.yml
conda activate colbert-review
make data
make test
make bm25
make ablation
make summarize
make stats
make figures
```

If the indexes already exist and only the tables need to be regenerated, run:

```bash
make summarize stats figures
```
