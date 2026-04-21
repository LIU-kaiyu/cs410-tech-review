"""Schema-level tests for the SciFact loader.

The full BEIR download is a network operation and is not exercised in unit tests.
These tests validate the JSONL persistence layer against a synthetic corpus so the
loader contract stays stable.
"""
from __future__ import annotations

from pathlib import Path

from src.data.load_scifact import save_corpus
from src.data.schema import Corpus, Doc, Qrel, Query
from src.utils import read_jsonl


def _tiny_corpus() -> Corpus:
    docs = [Doc("d1", "T1", "body1"), Doc("d2", "T2", "body2")]
    queries = [Query("q1", "hello"), Query("q2", "world")]
    qrels = [Qrel("q1", "d1", 1), Qrel("q2", "d2", 2)]
    return Corpus(docs=docs, queries=queries, qrels=qrels)


def test_save_corpus_writes_three_jsonl_files(tmp_path: Path):
    corpus = _tiny_corpus()
    save_corpus(corpus, out_dir=tmp_path, split="test")

    docs_path = tmp_path / "scifact_corpus.jsonl"
    queries_path = tmp_path / "scifact_queries_test.jsonl"
    qrels_path = tmp_path / "scifact_qrels_test.jsonl"

    assert docs_path.exists()
    assert queries_path.exists()
    assert qrels_path.exists()

    doc_rows = list(read_jsonl(docs_path))
    assert {r["doc_id"] for r in doc_rows} == {"d1", "d2"}

    query_rows = list(read_jsonl(queries_path))
    assert {r["qid"] for r in query_rows} == {"q1", "q2"}

    qrel_rows = list(read_jsonl(qrels_path))
    assert any(r["qid"] == "q1" and r["doc_id"] == "d1" for r in qrel_rows)
    assert all(isinstance(r["relevance"], int) for r in qrel_rows)
