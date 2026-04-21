.PHONY: help setup data test bm25 colbert ablation eval figures report clean all

PYTHON ?= python
DATASET ?= scifact
SPLIT ?= test
TOPK ?= 100

help:
	@echo "ColBERT Tech Review — available targets:"
	@echo "  setup      Install Python dependencies"
	@echo "  data       Download and normalize SciFact"
	@echo "  test       Run unit tests"
	@echo "  bm25       Build BM25 index and produce run file"
	@echo "  colbert    Build ColBERT index (nbits=2) and produce run file"
	@echo "  ablation   Sweep nbits in {1,2,4}"
	@echo "  eval       Evaluate a run file (EXP=<name>)"
	@echo "  figures    Generate figures from results/metrics"
	@echo "  report     Compile LaTeX report"
	@echo "  clean      Remove indices and generated artifacts"
	@echo "  all        Full pipeline: data -> bm25 -> colbert -> ablation -> eval -> figures"

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) -m src.data.load_scifact --out data/processed

test:
	$(PYTHON) -m pytest tests/ -v

bm25:
	$(PYTHON) scripts/run_bm25.py --dataset $(DATASET) --split $(SPLIT) --topk $(TOPK) \
	  --out results/runs/bm25.trec

colbert:
	$(PYTHON) scripts/run_colbert.py --dataset $(DATASET) --split $(SPLIT) --topk $(TOPK) \
	  --nbits 2 --out results/runs/colbert_nbits2.trec

ablation:
	$(PYTHON) scripts/run_ablation.py --dataset $(DATASET) --split $(SPLIT) --topk $(TOPK) \
	  --nbits 1 2 4 --out-dir results/runs

eval:
	$(PYTHON) scripts/run_eval.py --run results/runs/$(EXP).trec --out results/metrics/$(EXP).json

figures:
	$(PYTHON) scripts/make_figures.py --metrics-dir results/metrics --out-dir results/figures

report:
	cd report && pdflatex report.tex && pdflatex report.tex

clean:
	rm -rf .ragatouille indexes __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

all: data test bm25 colbert ablation figures
