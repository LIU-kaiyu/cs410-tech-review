"""Generate the 4 project notebooks as valid .ipynb files."""
import json, os

def code(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}
def md(src):   return {"cell_type":"markdown","metadata":{},"source":src}

META = {
    "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
    "language_info": {"name":"python","version":"3.10.0"}
}

SETUP_CONSTS = """\
import os, sys

# ── Auto-detect environment ───────────────────────────────────────────────────
try:
    import google.colab; _IN_COLAB = True
except ImportError:
    _IN_COLAB = False

_IN_KAGGLE = os.path.exists("/kaggle/working")

if _IN_COLAB or _IN_KAGGLE:
    # Remote: repo will be cloned in the next cell — update REPO_URL first.
    REPO_URL  = "https://github.com/YOUR_ORG/cs410-tech-review.git"  # ← REPLACE
    REPO_ROOT = "/content/cs410-tech-review" if _IN_COLAB else "/kaggle/working/cs410-tech-review"
else:
    # Local: locate the repo root relative to this notebook.
    _here = os.path.abspath(".")
    if os.path.basename(_here) == "notebooks" and os.path.isdir(os.path.join(_here, "..", "src")):
        REPO_ROOT = os.path.abspath(os.path.join(_here, ".."))
    elif os.path.isdir(os.path.join(_here, "src")):
        REPO_ROOT = _here
    else:
        REPO_ROOT = _here  # fallback: set manually if this prints the wrong path
    REPO_URL = None
    print(f"Local mode — REPO_ROOT: {REPO_ROOT}")\
"""

def setup_clone():
    return code("""\
# Clone repo (Colab / Kaggle only — skipped automatically in local mode).
if REPO_URL and not os.path.isdir(REPO_ROOT):
    !git clone {REPO_URL} {REPO_ROOT}
elif REPO_URL:
    print(f"Repo already present at {REPO_ROOT}")
else:
    print("Local mode — skipping clone.")\
""")

def setup_pip():
    # Remote (Colab / Kaggle) installs a fully-resolved set that actually works
    # for ColBERT + RAGatouille on Python 3.10. Local runs use the preconfigured
    # `colbert-review` conda env, so pip is skipped — otherwise pip's resolver
    # emits spurious "ResolutionImpossible" errors against already-installed
    # packages (ragatouille/colbert-ai version chains don't re-resolve cleanly).
    return code("""\
if _IN_COLAB or _IN_KAGGLE:
    %pip install -q \\
        "torch>=2.1.0,<2.4.0" \\
        "transformers==4.44.2" \\
        "tokenizers<0.20" \\
        "faiss-cpu>=1.7.4" \\
        "ragatouille==0.0.9.post2" \\
        "colbert-ai>=0.2.19" \\
        "langchain==0.1.20" \\
        "langchain-core==0.1.53" \\
        "rank_bm25>=0.2.2" \\
        "beir>=2.0.0" \\
        "ranx>=0.3.16" \\
        "scipy>=1.11.0" \\
        "numpy>=1.24.0,<2.0.0" \\
        "pandas>=2.0.0" \\
        "pydantic>=2.0.0" \\
        "matplotlib>=3.7.0" \\
        "seaborn>=0.12.0" \\
        "tqdm>=4.65.0" \\
        "pyyaml>=6.0" \\
        "ninja"
    print("Dependencies installed.")
else:
    # Local: colbert-review conda env already provides everything.
    # Inject the env's bin dir at the head of PATH so subprocesses the kernel
    # spawns (notably ColBERT's JIT C++ extension build via ninja) resolve the
    # right ninja/g++/nvcc, regardless of how the kernel was launched.
    import sys
    _env_bin = os.path.dirname(sys.executable)
    _path = os.environ.get("PATH", "")
    if _env_bin not in _path.split(os.pathsep):
        os.environ["PATH"] = _env_bin + os.pathsep + _path
    print(f"Local mode — using existing env at {_env_bin}")\
""")

def setup_paths(with_index_root=False):
    extra = "\nINDEX_ROOT  = Path(REPO_ROOT) / \".ragatouille\"\n" if with_index_root else ""
    dirs  = "RUNS_DIR, METRICS_DIR, FIGURES_DIR, DATA_DIR" + (", INDEX_ROOT" if with_index_root else "")
    return code(f"""\
import sys, os
from pathlib import Path

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

RUNS_DIR    = Path(REPO_ROOT) / "results" / "runs"
METRICS_DIR = Path(REPO_ROOT) / "results" / "metrics"
FIGURES_DIR = Path(REPO_ROOT) / "results" / "figures"
DATA_DIR    = Path(REPO_ROOT) / "data" / "raw"{extra}
for d in [{dirs}]:
    os.makedirs(d, exist_ok=True)

print("REPO_ROOT  :", REPO_ROOT)
print("RUNS_DIR   :", RUNS_DIR)
print("METRICS_DIR:", METRICS_DIR)\
""")

def setup_gpu(colbert=False):
    warn = (
        'print("WARNING: No GPU. ColBERT indexing will be very slow.")'
        if colbert else
        'print("No GPU — CPU mode. BM25 is unaffected.")'
    )
    return code(f"""\
import torch
if torch.cuda.is_available():
    device = "cuda"
    print(f"GPU: {{torch.cuda.get_device_name(0)}}")
else:
    device = "cpu"
    {warn}\
""")

# ═══════════════════════════════════════════════════════════════════════════════
# Notebook 01 — BM25 baseline
# ═══════════════════════════════════════════════════════════════════════════════

nb01_cells = [
md("""\
# 01 — BM25 Baseline on BEIR SciFact

End-to-end walkthrough: data loading → BM25 indexing → retrieval → TREC metric
evaluation → qualitative inspection.

Run this notebook first. The run file and metrics JSON it writes are consumed
by `04_analysis_and_figures.ipynb`.\
"""),
code(SETUP_CONSTS),
setup_clone(),
setup_pip(),
setup_paths(),
setup_gpu(),

md("## 1. Load SciFact"),
code("""\
from src.data.load_scifact import load_scifact

# Downloads ~17 MB on first run; BEIR caches locally afterwards.
corpus = load_scifact(split="test", raw_dir=str(DATA_DIR))

print(f"docs    : {len(corpus.docs):,}")
print(f"queries : {len(corpus.queries):,}")
print(f"qrels   : {len(corpus.qrels):,}")\
"""),

md("## 2. Corpus inspection"),
code("""\
import pandas as pd

doc_df = pd.DataFrame([
    {"doc_id": d.doc_id, "title": d.title[:80], "text_words": len(d.text.split())}
    for d in corpus.docs[:5]
])
print("=== Sample documents ===")
display(doc_df)

q_df = pd.DataFrame([{"qid": q.qid, "text": q.text} for q in corpus.queries[:5]])
print("\\n=== Sample queries ===")
display(q_df)\
"""),

md("## 3. BM25 indexing"),
code("""\
import time
from src.baselines.bm25 import BM25Config, BM25Retriever

cfg = BM25Config(k1=1.5, b=0.75, epsilon=0.25, lowercase=True,
                 remove_stopwords=True, min_token_len=2)
retriever = BM25Retriever(config=cfg)

t0 = time.perf_counter()
retriever.index(corpus.docs)
print(f"Index built in {time.perf_counter()-t0:.2f}s over {len(corpus.docs):,} docs.")\
"""),

md("## 4. Retrieve top-100 for all test queries"),
code("""\
t0 = time.perf_counter()
bm25_rankings = retriever.search_all(corpus.queries, topk=100)
elapsed = time.perf_counter() - t0

total_hits = sum(len(v) for v in bm25_rankings.values())
print(f"{len(bm25_rankings):,} queries | {total_hits:,} hits | {elapsed:.2f}s")\
"""),

md("## 5. Qualitative hit inspection"),
code("""\
doc_map = corpus.doc_map()
q_map   = corpus.query_map()

sample_qid = corpus.queries[0].qid
print(f"Query [{sample_qid}]: {q_map[sample_qid].text}\\n")

for rank, (doc_id, score) in enumerate(bm25_rankings[sample_qid][:5], 1):
    doc = doc_map[doc_id]
    print(f"  Rank {rank} | doc_id={doc_id} | score={score:.4f}")
    print(f"    Title  : {doc.title[:90]}")
    print(f"    Snippet: {doc.text[:180]}...")
    print()\
"""),

md("## 6. Write TREC run file"),
code("""\
from src.eval.metrics import write_run

run_path = RUNS_DIR / "bm25.trec"
n_lines = write_run(run_path, bm25_rankings, tag="bm25")
print(f"Wrote {n_lines:,} lines -> {run_path}")\
"""),

md("## 7. Evaluate — NDCG@10, MRR@10, MAP, Recall@100"),
code("""\
from src.eval.metrics import evaluate_run, save_eval, DEFAULT_METRICS

qrels_map = corpus.qrels_map()

eval_result = evaluate_run(
    qrels_map=qrels_map,
    run=bm25_rankings,
    metrics=DEFAULT_METRICS,
    run_name="bm25",
)

print("=== BM25 aggregate metrics ===")
for metric, value in sorted(eval_result.metrics.items()):
    print(f"  {metric:<15}: {value:.4f}")

metrics_path = METRICS_DIR / "bm25.json"
save_eval(eval_result, metrics_path)
print(f"\\nSaved -> {metrics_path}")\
"""),

md("## 8. Per-query NDCG@10 distribution"),
code("""\
import matplotlib.pyplot as plt
import numpy as np

ndcg_vals = [v.get("ndcg@10", 0.0) for v in eval_result.per_query.values()]

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(ndcg_vals, bins=20, color="#4C72B0", edgecolor="white")
ax.axvline(np.mean(ndcg_vals), color="crimson", linestyle="--",
           label=f"mean={np.mean(ndcg_vals):.3f}")
ax.set_xlabel("NDCG@10")
ax.set_ylabel("Number of queries")
ax.set_title("BM25 per-query NDCG@10 — SciFact test")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "bm25_ndcg_hist.png", dpi=150, bbox_inches="tight")
plt.show()\
"""),

md("## 9. Worst-performing queries"),
code("""\
per_q_sorted = sorted(eval_result.per_query.items(),
                      key=lambda kv: kv[1].get("ndcg@10", 0.0))

print("=== 3 worst queries (NDCG@10) ===\\n")
for qid, m in per_q_sorted[:3]:
    rel_docs  = set(qrels_map.get(qid, {}).keys())
    top10_ids = [d for d, _ in bm25_rankings.get(qid, [])[:10]]
    hit       = any(d in rel_docs for d in top10_ids)
    print(f"  [{qid}] ndcg@10={m.get('ndcg@10', 0):.3f}  relevant_in_top10={hit}")
    print(f"    Query: {q_map[qid].text}")
    print()\
"""),

md("""\
## Summary

`results/runs/bm25.trec` and `results/metrics/bm25.json` are written.
Notebook 04 loads these for the final significance tests and report figures.\
"""),
]

# ═══════════════════════════════════════════════════════════════════════════════
# Notebook 02 — ColBERT smoke test (50 docs)
# ═══════════════════════════════════════════════════════════════════════════════

nb02_cells = [
md("""\
# 02 — ColBERT Index + Search (50-doc smoke test)

Verify RAGatouille can build a ColBERT index and run late-interaction search.
We slice 50 documents so the index builds in a few minutes even on CPU.

Full-corpus indexing and the nbits ablation are in `03_ablation_nbits.ipynb`.\
"""),
code(SETUP_CONSTS),
setup_clone(),
setup_pip(),
setup_paths(with_index_root=True),
setup_gpu(colbert=True),

md("## 1. Load SciFact and slice 50 docs"),
code("""\
from src.data.load_scifact import load_scifact
from src.data.schema import Corpus

full_corpus = load_scifact(split="test", raw_dir=str(DATA_DIR))

# Prioritise docs that have at least one qrel for meaningful recall numbers.
qrel_doc_ids  = {qr.doc_id for qr in full_corpus.qrels}
relevant_docs = [d for d in full_corpus.docs if d.doc_id in qrel_doc_ids]
filler_docs   = [d for d in full_corpus.docs if d.doc_id not in qrel_doc_ids]
slice_docs    = (relevant_docs + filler_docs)[:50]
slice_doc_ids = {d.doc_id for d in slice_docs}

slice_qrels   = [qr for qr in full_corpus.qrels  if qr.doc_id in slice_doc_ids]
slice_qids    = {qr.qid for qr in slice_qrels}
slice_queries = [q  for q  in full_corpus.queries if q.qid   in slice_qids]

mini_corpus = Corpus(docs=slice_docs, queries=slice_queries, qrels=slice_qrels)
mini_corpus.validate()

print(f"Mini corpus: {len(mini_corpus.docs)} docs, "
      f"{len(mini_corpus.queries)} queries, "
      f"{len(mini_corpus.qrels)} qrels")\
"""),

md("## 2. Build ColBERT index (nbits=2)"),
code("""\
import time
from src.colbert.index import ColBERTIndexConfig, ColBERTIndexer

index_cfg = ColBERTIndexConfig(
    checkpoint="colbert-ir/colbertv2.0",
    index_name="scifact_smoke50",
    nbits=2,
    max_document_length=256,
    index_root=str(INDEX_ROOT),
)
indexer = ColBERTIndexer(config=index_cfg)

print("Building index (downloads checkpoint on first run)...")
t0 = time.perf_counter()
index_stats = indexer.build_index(mini_corpus.docs, overwrite=True)
elapsed = time.perf_counter() - t0

print(f"\\nDone in {elapsed:.1f}s")
print(f"  n_docs     : {index_stats.n_docs}")
print(f"  nbits      : {index_stats.nbits}")
print(f"  index size : {index_stats.index_bytes / 1024:.1f} KiB")
print(f"  index path : {index_stats.index_path}")\
"""),

md("## 3. Late-interaction search"),
code("""\
from src.colbert.search import ColBERTSearcher

searcher = ColBERTSearcher(model=indexer._model)

print(f"Searching {len(mini_corpus.queries)} queries (topk=10)...")
rankings, latency = searcher.search(mini_corpus.queries, topk=10)

print(f"\\nLatency stats:")
print(f"  n_queries : {latency.n_queries}")
print(f"  total_s   : {latency.total_time_s:.2f}s")
print(f"  mean_ms   : {latency.mean_ms:.1f}ms")
print(f"  p50_ms    : {latency.p50_ms:.1f}ms")
print(f"  p95_ms    : {latency.p95_ms:.1f}ms")\
"""),

md("## 4. Qualitative inspection"),
code("""\
import pandas as pd

doc_map        = mini_corpus.doc_map()
q_map          = mini_corpus.query_map()
qrels_map_mini = mini_corpus.qrels_map()

sample_qid = mini_corpus.queries[0].qid
print(f"Query [{sample_qid}]: {q_map[sample_qid].text}\\n")
print(f"Relevant docs: {list(qrels_map_mini.get(sample_qid, {}).keys())}\\n")

rows = []
for rank, (doc_id, score) in enumerate(rankings.get(sample_qid, []), 1):
    doc = doc_map.get(doc_id)
    rows.append({
        "rank": rank, "doc_id": doc_id, "score": round(score, 4),
        "relevant": "YES" if doc_id in qrels_map_mini.get(sample_qid, {}) else "-",
        "title": (doc.title[:70] if doc else "N/A"),
    })
display(pd.DataFrame(rows))\
"""),

md("## 5. Write TREC run and evaluate"),
code("""\
from src.eval.metrics import write_run, evaluate_run, save_eval, DEFAULT_METRICS

run_path = RUNS_DIR / "colbert_smoke50.trec"
write_run(run_path, rankings, tag="colbert_smoke50")

qrels_map_mini = mini_corpus.qrels_map()
eval_result = evaluate_run(
    qrels_map=qrels_map_mini,
    run=rankings,
    metrics=DEFAULT_METRICS,
    run_name="colbert_smoke50",
)

print("=== Smoke-test metrics (50 docs, topk=10) ===")
for metric, value in sorted(eval_result.metrics.items()):
    print(f"  {metric:<15}: {value:.4f}")

save_eval(eval_result, METRICS_DIR / "colbert_smoke50.json")\
"""),

md("## 6. Index-size file breakdown"),
code("""\
from pathlib import Path

index_path = Path(index_stats.index_path)
if index_path.exists():
    files     = sorted(index_path.rglob("*"))
    rows      = [{"file": str(f.relative_to(index_path)), "size_kb": f.stat().st_size / 1024}
                 for f in files if f.is_file()]
    total_kib = sum(r["size_kb"] for r in rows)
    display(pd.DataFrame(rows).sort_values("size_kb", ascending=False).head(15))
    print(f"\\nTotal: {total_kib:.1f} KiB across {len(rows)} files")
else:
    print(f"Index path not found: {index_path}")\
"""),

md("""\
## Summary

- ColBERT checkpoint `colbert-ir/colbertv2.0` indexed 50 SciFact docs.
- Late-interaction search confirmed working; latency numbers above.
- Full-corpus indexing and the nbits ablation are in notebook 03.\
"""),
]

# ═══════════════════════════════════════════════════════════════════════════════
# Notebook 03 — nbits ablation sweep
# ═══════════════════════════════════════════════════════════════════════════════

nb03_cells = [
md("""\
# 03 — Ablation: Residual Compression `nbits` ∈ {1, 2, 4}

Runs the full SciFact corpus through the nbits sweep and visualises the
index-size vs. quality trade-off.

**Prerequisites:** Notebook 01 must have run (`results/metrics/bm25.json`).

**Runtime:** ~30–60 min on GPU; 3–9 h on CPU. Plan accordingly.\
"""),
code(SETUP_CONSTS),
setup_clone(),
setup_pip(),

code("""\
import sys, os, json
from pathlib import Path

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

RUNS_DIR    = Path(REPO_ROOT) / "results" / "runs"
METRICS_DIR = Path(REPO_ROOT) / "results" / "metrics"
FIGURES_DIR = Path(REPO_ROOT) / "results" / "figures"
DATA_DIR    = Path(REPO_ROOT) / "data" / "raw"
INDEX_ROOT  = Path(REPO_ROOT) / ".ragatouille"

for d in [RUNS_DIR, METRICS_DIR, FIGURES_DIR, DATA_DIR, INDEX_ROOT]:
    os.makedirs(d, exist_ok=True)
print("Paths ready.")\
"""),

setup_gpu(colbert=True),

md("## 1. Load full SciFact corpus"),
code("""\
from src.data.load_scifact import load_scifact

corpus = load_scifact(split="test", raw_dir=str(DATA_DIR))
print(f"Corpus: {len(corpus.docs):,} docs | "
      f"{len(corpus.queries):,} queries | "
      f"{len(corpus.qrels):,} qrels")\
"""),

md("""\
## 2. Run nbits sweep

`run_sweep` builds one index per nbits value, runs all queries, evaluates,
and writes per-variant TREC files, JSON metric files, and a summary CSV.\
"""),
code("""\
from src.ablation.nbits_sweep import run_sweep

print("Starting nbits sweep...")
records = run_sweep(
    corpus=corpus,
    nbits_values=(1, 2, 4),
    topk=100,
    checkpoint="colbert-ir/colbertv2.0",
    index_name_prefix="scifact",
    runs_dir=str(RUNS_DIR),
    metrics_dir=str(METRICS_DIR),
    max_document_length=256,
)

print(f"\\nSweep complete — {len(records)} variants:")
for r in records:
    print(f"  nbits={r.nbits}: {r.index_mib:.1f} MiB | "
          f"ndcg@10={r.metrics.get('ndcg@10', float('nan')):.4f} | "
          f"p50={r.latency_p50_ms:.1f}ms")\
"""),

md("## 3. Load results into a DataFrame"),
code("""\
import pandas as pd

with open(METRICS_DIR / "ablation_nbits.json") as f:
    raw = json.load(f)  # list of AblationRecord dicts

# Flatten nested "metrics" sub-dict into top-level columns.
rows = []
for r in raw:
    row = {k: v for k, v in r.items() if k != "metrics"}
    row.update(r.get("metrics", {}))
    rows.append(row)
df = pd.DataFrame(rows)

display(df[["nbits","n_docs","build_time_s","index_mib",
            "latency_p50_ms","latency_p95_ms",
            "ndcg@10","mrr@10","map","recall@100"]].to_string(index=False))\
"""),

md("## 4. Load BM25 reference metrics"),
code("""\
bm25_json = METRICS_DIR / "bm25.json"
bm25_metrics = {}
if not bm25_json.exists():
    print(f"WARNING: {bm25_json} not found — run notebook 01 first.")
else:
    with open(bm25_json) as f:
        bm25_metrics = json.load(f).get("metrics", {})
    print("BM25 metrics:")
    for k, v in sorted(bm25_metrics.items()):
        print(f"  {k}: {v:.4f}")\
"""),

md("## 5. Index size vs. NDCG@10"),
code("""\
import matplotlib.pyplot as plt

NB_COLORS = {1: "#E8505B", 2: "#F9A825", 4: "#43A047"}

fig, ax = plt.subplots(figsize=(7, 5))
for _, row in df.iterrows():
    nb = int(row["nbits"])
    ax.scatter(row["index_mib"], row["ndcg@10"],
               s=140, color=NB_COLORS.get(nb, "grey"), zorder=3)
    ax.annotate(f"nbits={nb}",
                xy=(row["index_mib"], row["ndcg@10"]),
                xytext=(6, 3), textcoords="offset points", fontsize=10)

if "ndcg@10" in bm25_metrics:
    ax.axhline(bm25_metrics["ndcg@10"], color="steelblue", linestyle="--",
               linewidth=1.5, label=f"BM25 ({bm25_metrics['ndcg@10']:.3f})")

ax.set_xlabel("Index size (MiB)")
ax.set_ylabel("NDCG@10")
ax.set_title("ColBERT: index size vs. NDCG@10 — SciFact test")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "ablation_size_vs_ndcg.png", dpi=150, bbox_inches="tight")
plt.show()\
"""),

md("## 6. Multi-metric bar chart"),
code("""\
import numpy as np

MPLOT = ["ndcg@10", "mrr@10", "map", "recall@100"]
x, width = np.arange(len(MPLOT)), 0.22

fig, ax = plt.subplots(figsize=(9, 5))
for i, (_, row) in enumerate(df.sort_values("nbits").iterrows()):
    nb   = int(row["nbits"])
    vals = [row.get(m, float("nan")) for m in MPLOT]
    ax.bar(x + (i - 1) * width, vals, width,
           label=f"ColBERT nbits={nb}", color=NB_COLORS.get(nb, "grey"), alpha=0.85)

if bm25_metrics:
    bm25_vals = [bm25_metrics.get(m, float("nan")) for m in MPLOT]
    ax.bar(x + 1 * width + 0.02, bm25_vals, width,
           label="BM25", color="steelblue", alpha=0.70)

ax.set_xticks(x)
ax.set_xticklabels(MPLOT, fontsize=11)
ax.set_ylabel("Score")
ax.set_title("ColBERT (nbits ablation) vs. BM25 — SciFact test")
ax.legend(fontsize=9)
ax.set_ylim(0, 1.0)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "ablation_metrics_bar.png", dpi=150, bbox_inches="tight")
plt.show()\
"""),

md("## 7. Query latency vs. nbits"),
code("""\
df_s = df.sort_values("nbits")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(df_s["nbits"], df_s["latency_p50_ms"], marker="o", label="p50", color="#1E88E5")
ax.plot(df_s["nbits"], df_s["latency_p95_ms"], marker="s", linestyle="--",
        label="p95", color="#FB8C00")
ax.set_xticks([1, 2, 4])
ax.set_xlabel("nbits")
ax.set_ylabel("Query latency (ms)")
ax.set_title("ColBERT query latency vs. nbits")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "ablation_latency.png", dpi=150, bbox_inches="tight")
plt.show()\
"""),

md("## 8. Summary table"),
code("""\
SCOLS = ["nbits","index_mib","build_time_s",
         "latency_p50_ms","latency_p95_ms",
         "ndcg@10","mrr@10","map","recall@100"]

display(df[SCOLS].sort_values("nbits")
        .style.format({c: "{:.3f}" for c in SCOLS if c != "nbits"})
        .set_caption("ColBERT nbits ablation — SciFact test"))\
"""),
]

# ═══════════════════════════════════════════════════════════════════════════════
# Notebook 04 — analysis and report figures
# ═══════════════════════════════════════════════════════════════════════════════

nb04_cells = [
md("""\
# 04 — Analysis, Significance Tests, and Report Figures

Loads all pre-computed metrics, runs paired t-tests with Holm-Bonferroni
correction, and produces four final report figures.

**Prerequisites** (run notebooks 01 and 03 first):
- `results/metrics/bm25.json`
- `results/metrics/colbert_nbits1.json`
- `results/metrics/colbert_nbits2.json`
- `results/metrics/colbert_nbits4.json`
- `results/metrics/ablation_nbits.csv`\
"""),
code(SETUP_CONSTS),
setup_clone(),
setup_pip(),

code("""\
import sys, os, json
from pathlib import Path

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

RUNS_DIR    = Path(REPO_ROOT) / "results" / "runs"
METRICS_DIR = Path(REPO_ROOT) / "results" / "metrics"
FIGURES_DIR = Path(REPO_ROOT) / "results" / "figures"

for d in [RUNS_DIR, METRICS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)
print("Paths ready.")\
"""),

md("## 0. Verify prerequisites"),
code("""\
REQUIRED = [
    METRICS_DIR / "bm25.json",
    METRICS_DIR / "colbert_nbits1.json",
    METRICS_DIR / "colbert_nbits2.json",
    METRICS_DIR / "colbert_nbits4.json",
    METRICS_DIR / "ablation_nbits.csv",
]
missing = [str(p) for p in REQUIRED if not p.exists()]
if missing:
    raise FileNotFoundError(
        "Missing prerequisite files — run notebooks 01 and 03 first:\\n"
        + "\\n".join(missing)
    )
print("All prerequisite files present.")\
"""),

md("## 1. Load all evaluation results"),
code("""\
import pandas as pd

def _load(path):
    with open(path) as f:
        return json.load(f)

SYSTEMS = ["bm25", "colbert_nbits1", "colbert_nbits2", "colbert_nbits4"]
LABELS  = ["BM25", "ColBERT nbits=1", "ColBERT nbits=2", "ColBERT nbits=4"]

all_evals = {
    "bm25":           _load(METRICS_DIR / "bm25.json"),
    "colbert_nbits1": _load(METRICS_DIR / "colbert_nbits1.json"),
    "colbert_nbits2": _load(METRICS_DIR / "colbert_nbits2.json"),
    "colbert_nbits4": _load(METRICS_DIR / "colbert_nbits4.json"),
}

agg_rows = [{"system": name, **data["metrics"]} for name, data in all_evals.items()]
agg_df   = pd.DataFrame(agg_rows).set_index("system")
print("=== Aggregate metrics ===")
display(agg_df.style.format("{:.4f}").highlight_max(axis=0, color="#d4edda"))\
"""),

md("""\
## 2. Paired t-tests — BM25 vs. each ColBERT variant

Delta = ColBERT − BM25 (positive means ColBERT wins).\
"""),
code("""\
from src.eval.stats import paired_ttest, holm_bonferroni

COMPARISONS    = [("colbert_nbits1","ColBERT nbits=1"),
                  ("colbert_nbits2","ColBERT nbits=2"),
                  ("colbert_nbits4","ColBERT nbits=4")]
TARGET_METRICS = ["ndcg@10", "map"]

results = []
for sys_key, sys_label in COMPARISONS:
    for metric in TARGET_METRICS:
        results.append(paired_ttest(
            a=all_evals["bm25"]["per_query"],
            b=all_evals[sys_key]["per_query"],
            metric=metric, system_a="BM25", system_b=sys_label,
            n_bootstrap=2000, seed=42,
        ))

print(f"{'System B':<22} {'Metric':<12} {'mean_A':>8} {'mean_B':>8} "
      f"{'delta':>8} {'p-value':>9} {'95% CI':>22}")
print("-" * 95)
for r in results:
    ci = f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}]"
    print(f"{r.system_b:<22} {r.metric:<12} {r.mean_a:>8.4f} {r.mean_b:>8.4f} "
          f"{r.mean_delta:>+8.4f} {r.p_value:>9.4f} {ci:>22}")\
"""),

md("## 3. Holm-Bonferroni correction"),
code("""\
decisions = holm_bonferroni([r.p_value for r in results], alpha=0.05)

print("=== Holm-Bonferroni corrected (alpha=0.05) ===\\n")
print(f"{'System B':<22} {'Metric':<12} {'p-value':>9} {'sig?':>6}")
print("-" * 55)
for r, sig in zip(results, decisions):
    print(f"{r.system_b:<22} {r.metric:<12} {r.p_value:>9.4f} "
          f"{'YES *' if sig else 'no':>6}")\
"""),

md("## 4. Pairwise ablation comparisons within ColBERT"),
code("""\
ABLATION_PAIRS = [
    ("colbert_nbits1","colbert_nbits2","nbits1 vs nbits2"),
    ("colbert_nbits2","colbert_nbits4","nbits2 vs nbits4"),
    ("colbert_nbits1","colbert_nbits4","nbits1 vs nbits4"),
]

abl_results  = [
    paired_ttest(
        a=all_evals[ka]["per_query"], b=all_evals[kb]["per_query"],
        metric="ndcg@10", system_a=ka, system_b=kb, n_bootstrap=2000, seed=42,
    )
    for ka, kb, _ in ABLATION_PAIRS
]
abl_decisions = holm_bonferroni([r.p_value for r in abl_results], alpha=0.05)

print("=== Pairwise ablation — NDCG@10 (Holm-Bonferroni) ===\\n")
print(f"{'Comparison':<25} {'delta':>8} {'p-value':>9} {'sig?':>6} {'95% CI':>22}")
print("-" * 75)
for (_, _, label), r, sig in zip(ABLATION_PAIRS, abl_results, abl_decisions):
    ci = f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}]"
    print(f"{label:<25} {r.mean_delta:>+8.4f} {r.p_value:>9.4f} "
          f"{'YES *' if sig else 'no':>6} {ci:>22}")\
"""),

md("## 5. Figure 1 — System comparison bar chart"),
code("""\
import matplotlib.pyplot as plt
import numpy as np

PALETTE = ["#5C85D6","#E8505B","#F9A825","#43A047"]
MPLOT   = ["ndcg@10","mrr@10","map","recall@100"]
x, width = np.arange(len(MPLOT)), 0.18

fig, ax = plt.subplots(figsize=(10, 5))
for i, (sk, lb, col) in enumerate(zip(SYSTEMS, LABELS, PALETTE)):
    vals = [all_evals[sk]["metrics"].get(m, 0.0) for m in MPLOT]
    ax.bar(x + (i - 1.5) * width, vals, width,
           label=lb, color=col, alpha=0.88, edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels(MPLOT, fontsize=11)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.0)
ax.set_title("BM25 vs. ColBERT (nbits ablation) — SciFact test")
ax.legend(fontsize=9, ncol=4)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "figure1_system_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: figure1_system_comparison.png")\
"""),

md("## 6. Figure 2 — Ablation size-quality Pareto"),
code("""\
abl_df = pd.read_csv(METRICS_DIR / "ablation_nbits.csv")
abl_df = abl_df.rename(columns=lambda c: c.replace("metric_","") if c.startswith("metric_") else c)

NB_COLORS = {1:"#E8505B", 2:"#F9A825", 4:"#43A047"}

fig, ax = plt.subplots(figsize=(7, 5))
for _, row in abl_df.iterrows():
    nb = int(row["nbits"])
    ax.scatter(row["index_mib"], row["ndcg@10"],
               s=160, color=NB_COLORS.get(nb,"grey"), zorder=4)
    ax.annotate(f"  nbits={nb} ({row['ndcg@10']:.3f})",
                xy=(row["index_mib"], row["ndcg@10"]), fontsize=9, va="center")

bm25_ndcg = all_evals["bm25"]["metrics"].get("ndcg@10")
if bm25_ndcg:
    ax.axhline(bm25_ndcg, color="steelblue", linestyle="--", linewidth=1.5,
               label=f"BM25 NDCG@10 = {bm25_ndcg:.3f}")

ax.set_xlabel("Index size (MiB)")
ax.set_ylabel("NDCG@10")
ax.set_title("ColBERT residual compression: size vs. quality")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "figure2_ablation_pareto.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: figure2_ablation_pareto.png")\
"""),

md("## 7. Figure 3 — Significance heatmap"),
code("""\
import seaborn as sns

SIG_METRICS = ["ndcg@10", "map"]
SYS_KEYS    = ["colbert_nbits1","colbert_nbits2","colbert_nbits4"]
SYS_LBLS    = ["ColBERT nbits=1","ColBERT nbits=2","ColBERT nbits=4"]

delta_mat  = np.zeros((len(SYS_KEYS), len(SIG_METRICS)))
annot_mat  = np.empty_like(delta_mat, dtype=object)

for i, sk in enumerate(SYS_KEYS):
    for j, m in enumerate(SIG_METRICS):
        r = paired_ttest(
            a=all_evals["bm25"]["per_query"],
            b=all_evals[sk]["per_query"],
            metric=m, system_a="BM25", system_b=sk, n_bootstrap=2000, seed=42,
        )
        delta_mat[i, j] = r.mean_delta
        annot_mat[i, j] = f"{r.mean_delta:+.3f}" + (" *" if r.p_value < 0.05 else "")

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(delta_mat, annot=annot_mat, fmt="",
            xticklabels=SIG_METRICS, yticklabels=SYS_LBLS,
            center=0, cmap="RdYlGn", linewidths=0.5, linecolor="white", ax=ax,
            cbar_kws={"label": "delta (ColBERT - BM25)"})
ax.set_title("Mean delta vs. BM25 (* p<0.05, uncorrected)")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "figure3_significance_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: figure3_significance_heatmap.png")\
"""),

md("## 8. Figure 4 — Per-query NDCG@10 box plots"),
code("""\
PER_Q_DATA = {
    "BM25":              [v.get("ndcg@10",0.0) for v in all_evals["bm25"]["per_query"].values()],
    "ColBERT\\nnbits=1": [v.get("ndcg@10",0.0) for v in all_evals["colbert_nbits1"]["per_query"].values()],
    "ColBERT\\nnbits=2": [v.get("ndcg@10",0.0) for v in all_evals["colbert_nbits2"]["per_query"].values()],
    "ColBERT\\nnbits=4": [v.get("ndcg@10",0.0) for v in all_evals["colbert_nbits4"]["per_query"].values()],
}
BOX_COLORS = ["#5C85D6","#E8505B","#F9A825","#43A047"]

fig, ax = plt.subplots(figsize=(8, 5))
bp = ax.boxplot(list(PER_Q_DATA.values()), labels=list(PER_Q_DATA.keys()),
                patch_artist=True, notch=True,
                medianprops=dict(color="black", linewidth=2))
for patch, c in zip(bp["boxes"], BOX_COLORS):
    patch.set_facecolor(c)
    patch.set_alpha(0.75)

ax.set_ylabel("NDCG@10")
ax.set_title("Per-query NDCG@10 distribution — SciFact test")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "figure4_perquery_boxplot.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: figure4_perquery_boxplot.png")\
"""),

md("## 9. Final metrics table"),
code("""\
summary = [
    {"System": lb,
     **{m: f"{all_evals[sk]['metrics'].get(m, float('nan')):.4f}"
        for m in ["ndcg@10","mrr@10","map","recall@100"]}}
    for sk, lb in zip(SYSTEMS, LABELS)
]
summary_df = pd.DataFrame(summary).set_index("System")
display(summary_df)

table_path = FIGURES_DIR / "final_metrics_table.csv"
summary_df.to_csv(table_path)
print(f"Saved: {table_path}")\
"""),

md("""\
## Findings template

Fill this in before submitting:

**RQ1 — Does ColBERT outperform BM25 on SciFact?**
- ColBERT nbits=2 NDCG@10: ___ | BM25 NDCG@10: ___
- p-value: ___ | Holm-Bonferroni corrected: ___

**RQ2 — Trade-off across nbits ∈ {1, 2, 4}?**
- Index size range: ___ — ___ MiB
- NDCG@10 range: ___ — ___
- Pairwise significance: ___\
"""),
]

# ═══════════════════════════════════════════════════════════════════════════════
# Write .ipynb files
# ═══════════════════════════════════════════════════════════════════════════════

def write_nb(path, cells):
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": META, "cells": cells}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    size = os.path.getsize(path)
    print(f"Wrote {path}  ({size:,} bytes)")

out_dir = os.path.dirname(os.path.abspath(__file__))
write_nb(os.path.join(out_dir, "01_bm25_baseline.ipynb"),        nb01_cells)
write_nb(os.path.join(out_dir, "02_colbert_index_search.ipynb"), nb02_cells)
write_nb(os.path.join(out_dir, "03_ablation_nbits.ipynb"),       nb03_cells)
write_nb(os.path.join(out_dir, "04_analysis_and_figures.ipynb"), nb04_cells)
