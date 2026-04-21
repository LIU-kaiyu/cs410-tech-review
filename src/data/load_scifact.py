"""SciFact loader (BEIR).

Downloads and parses the BEIR SciFact dataset, normalizes it into the project's
`Corpus` schema, and optionally persists to JSONL for reproducible downstream runs.

CLI:

    python -m src.data.load_scifact --out data/processed --split test
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

from ..utils import ensure_dir, get_logger, write_jsonl
from .schema import Corpus, Doc, Qrel, Query

log = get_logger(__name__)

BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"


def _download_and_load(dataset: str, split: str, raw_dir: Path) -> Tuple[dict, dict, dict]:
    """Wrap BEIR's loader so the heavy import is lazy."""
    from beir import util as beir_util
    from beir.datasets.data_loader import GenericDataLoader

    ensure_dir(raw_dir)
    dataset_dir = raw_dir / dataset
    if not dataset_dir.exists():
        url = f"{BEIR_BASE_URL}/{dataset}.zip"
        log.info("Downloading %s from %s", dataset, url)
        beir_util.download_and_unzip(url, str(raw_dir))

    log.info("Loading %s split=%s from %s", dataset, split, dataset_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=str(dataset_dir)).load(split=split)
    return corpus, queries, qrels


def load_scifact(split: str = "test", raw_dir: str | Path = "data/raw") -> Corpus:
    """Load SciFact and normalize to `Corpus`.

    BEIR returns:
      corpus:  {doc_id: {"title": str, "text": str}}
      queries: {qid: str}
      qrels:   {qid: {doc_id: int}}
    """
    raw_dir = Path(raw_dir)
    beir_corpus, beir_queries, beir_qrels = _download_and_load("scifact", split, raw_dir)

    docs = [
        Doc(doc_id=str(doc_id), title=entry.get("title", "") or "", text=entry.get("text", "") or "")
        for doc_id, entry in beir_corpus.items()
    ]
    queries = [Query(qid=str(qid), text=text) for qid, text in beir_queries.items()]
    qrels = [
        Qrel(qid=str(qid), doc_id=str(doc_id), relevance=int(rel))
        for qid, rels in beir_qrels.items()
        for doc_id, rel in rels.items()
    ]

    corpus = Corpus(docs=docs, queries=queries, qrels=qrels)
    corpus.validate()
    log.info(
        "Loaded SciFact split=%s: %d docs, %d queries, %d qrels",
        split, len(docs), len(queries), len(qrels),
    )
    return corpus


def save_corpus(corpus: Corpus, out_dir: str | Path, split: str) -> None:
    out_dir = ensure_dir(out_dir)
    docs_path = out_dir / "scifact_corpus.jsonl"
    queries_path = out_dir / f"scifact_queries_{split}.jsonl"
    qrels_path = out_dir / f"scifact_qrels_{split}.jsonl"

    write_jsonl(docs_path, ({"doc_id": d.doc_id, "title": d.title, "text": d.text} for d in corpus.docs))
    write_jsonl(queries_path, ({"qid": q.qid, "text": q.text} for q in corpus.queries))
    write_jsonl(
        qrels_path,
        ({"qid": q.qid, "doc_id": q.doc_id, "relevance": q.relevance} for q in corpus.qrels),
    )
    log.info("Wrote %s, %s, %s", docs_path, queries_path, qrels_path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and normalize BEIR SciFact.")
    p.add_argument("--split", default="test", choices=["train", "dev", "test"])
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out", default="data/processed")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    corpus = load_scifact(split=args.split, raw_dir=args.raw_dir)
    save_corpus(corpus, out_dir=args.out, split=args.split)


if __name__ == "__main__":
    main()
