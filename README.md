# ColBERT Tech Review: RAGatouille on SciFact

CS410 Technical Review — an empirical study of ColBERTv2 late-interaction retrieval via the
RAGatouille wrapper, benchmarked against a BM25 sparse-retrieval baseline on the BEIR SciFact
benchmark, with a residual-compression ablation over `nbits ∈ {1, 2, 4}`.

## Current Status

The BM25 baseline and the ColBERTv2/RAGatouille runs are complete on the BEIR
SciFact test split. ColBERTv2 improves over BM25 on every aggregate metric:
NDCG@10 rises from `0.6641` to `0.6924`, MRR@10 from `0.6317` to `0.6693`,
MAP from `0.6251` to `0.6623`, and Recall@100 from `0.8759` to `0.9120`.

The residual-compression ablation over `nbits ∈ {1, 2, 4}` also completed. The
saved index stats record both requested `nbits` and observed `effective_nbits`;
all three runs matched the requested value. On this small 5,183-document corpus,
all three bit widths produced identical retrieval metrics and identical measured
index sizes.

The final LaTeX source is in `report/report.tex`. This machine does not have
`pdflatex`/`latexmk`, so compile it on Overleaf or install a local TeX
distribution before running `make report`.

## Project Goals

1. Establish a reproducible BM25 baseline on SciFact test (NDCG@10, MRR@10, MAP, Recall@100).
2. Index and query SciFact with ColBERTv2 via RAGatouille end-to-end.
3. Ablate ColBERT's residual compression (`nbits`) and quantify the memory-vs-quality trade-off.
4. Report significance via paired Student's t-tests with Holm-Bonferroni correction.
5. Produce a written technical review (PDF) with tables, figures, and discussion.

## Repository Layout

```
tech review/
├── configs/          # YAML configs for bm25 and colbert runs
├── data/
│   ├── raw/          # BEIR download cache (git-ignored)
│   └── processed/    # Normalized corpus/queries/qrels JSONL
├── src/
│   ├── data/         # SciFact loader + dataclass schema
│   ├── baselines/    # BM25 (rank_bm25) retriever
│   ├── colbert/      # RAGatouille index + search wrappers
│   ├── eval/         # ranx metrics + paired t-test stats
│   ├── ablation/     # nbits sweep orchestration
│   └── utils/        # io, logging, timing helpers
├── scripts/          # CLI entrypoints (run_bm25.py, run_colbert.py, ...)
├── notebooks/        # Exploratory + analysis notebooks
├── results/
│   ├── runs/         # TREC-format run files
│   ├── metrics/      # JSON/CSV metric reports
│   └── figures/      # Report-ready figures
├── tests/            # pytest unit tests
└── report/           # LaTeX source + compiled PDF
```

## Setup

### Prerequisites

- Python 3.10
- CUDA-capable GPU recommended for ColBERT (CPU works but is slow)
- On Windows: **WSL2 strongly recommended** — ColBERT's compiled CUDA kernels are
  flaky on native Windows. BM25 and the eval harness run fine natively.

### Install

```bash
# Option A: plain pip
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Option B: conda
conda env create -f environment.yml
conda activate colbert-review
```

Pin exact installed versions after first successful install:

```bash
pip freeze > requirements.lock.txt
```

## Usage

All common workflows are wired through `make`:

```bash
make setup       # install dependencies
make data        # download + normalize SciFact
make test        # run unit tests
make bm25        # BM25 baseline -> results/runs/bm25.trec
make colbert     # ColBERT nbits=2 -> results/runs/colbert_nbits2.trec
make ablation    # Sweep nbits in {1, 2, 4}
make eval EXP=bm25
make stats       # Paired t-tests + Holm-Bonferroni correction
make figures
make report
```

Or use the Python entrypoints directly:

```bash
python scripts/run_bm25.py    --dataset scifact --split test --topk 100 --out results/runs/bm25.trec
python scripts/run_colbert.py --dataset scifact --split test --topk 100 --nbits 2 --out results/runs/colbert_nbits2.trec
python scripts/run_eval.py    --run results/runs/bm25.trec --out results/metrics/bm25.json
```

## Datasets

SciFact is fetched via the [BEIR](https://github.com/beir-cellar/beir) loader on first run.
Expected counts:

| Split | Queries | Corpus |
|-------|---------|--------|
| train | 809     | 5,183  |
| dev   | 339     | 5,183  |
| test  | 300     | 5,183  |

(Note: BEIR SciFact uses 300 test queries; 1,109 is the total across splits.)

## Metrics

All metrics are computed with [ranx](https://github.com/AmenRa/ranx) for reproducibility:
`NDCG@10`, `MRR@10`, `MAP`, `Recall@100`.

Significance: paired Student's t-test on per-query metric vectors
(`scipy.stats.ttest_rel`), with Holm-Bonferroni correction for multiple comparisons.
Bootstrap 95% CIs are reported alongside point estimates.

## References

- Khattab & Zaharia, 2020. *ColBERT: Efficient and Effective Passage Search via
  Contextualized Late Interaction over BERT.* SIGIR 2020.
  → `reference materials/2004.12832v2.pdf`
- Wadden et al., 2020. *Fact or Fiction: Verifying Scientific Claims.* EMNLP 2020.
  → `reference materials/2020.emnlp-main.609.pdf`
- RAG Survey (2023). → `reference materials/2312.10997v5.pdf`
- Clavié, 2024. *RAGatouille* — `pip install ragatouille`.

## License

Course project — internal use. Third-party references retain their original licenses.
