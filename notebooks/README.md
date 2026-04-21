# Notebooks

Analysis notebooks to be filled in after the corresponding scripts run.
They are placeholders here so the repo layout is complete; real content is driven by
the CLI scripts and metric JSON files they emit.

| Notebook | Purpose |
|---|---|
| `01_bm25_baseline.ipynb` | Walk through BM25 retrieval, inspect hit qualitatively |
| `02_colbert_index_search.ipynb` | Smoke-test ColBERT indexing on a 50-doc slice |
| `03_ablation_nbits.ipynb` | Visualize size vs. quality trade-off across `nbits` |
| `04_analysis_and_figures.ipynb` | Final significance tests + report figures |

To create: `jupyter notebook notebooks/`
