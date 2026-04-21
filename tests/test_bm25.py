"""Unit tests for the BM25 baseline."""
from __future__ import annotations

import pytest

from src.baselines.bm25 import BM25Config, BM25Retriever, tokenize
from src.data.schema import Doc, Query


def _toy_docs():
    return [
        Doc(doc_id="d1", title="Cats", text="cats chase mice in the garden"),
        Doc(doc_id="d2", title="Dogs", text="dogs chase cats but not mice"),
        Doc(doc_id="d3", title="Birds", text="birds fly over the garden at dawn"),
        Doc(doc_id="d4", title="", text="the quick brown fox jumps over a lazy dog"),
    ]


def test_tokenize_lowercases_and_filters_stopwords():
    cfg = BM25Config()
    tokens = tokenize("The Cats Chase Mice!", cfg)
    assert "the" not in tokens
    assert "cats" in tokens
    assert "chase" in tokens
    assert "mice" in tokens


def test_index_requires_docs():
    with pytest.raises(ValueError):
        BM25Retriever().index([])


def test_search_without_index_raises():
    with pytest.raises(RuntimeError):
        BM25Retriever().search(Query("q1", "anything"))


def test_top_ranked_doc_matches_query_topic():
    retriever = BM25Retriever().index(_toy_docs())
    hits = retriever.search(Query("q1", "cats chase mice"), topk=3)
    assert hits, "expected at least one hit"
    top_doc_id = hits[0][0]
    assert top_doc_id in {"d1", "d2"}, f"expected a cats/mice doc, got {top_doc_id}"
    # scores must be descending
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_topk_is_respected():
    retriever = BM25Retriever().index(_toy_docs())
    hits = retriever.search(Query("q1", "garden"), topk=2)
    assert len(hits) <= 2


def test_empty_query_returns_empty_hits():
    retriever = BM25Retriever().index(_toy_docs())
    # A query made entirely of stopwords tokenizes to [] and must return [].
    hits = retriever.search(Query("q1", "the a an of"), topk=5)
    assert hits == []


def test_search_all_covers_every_query():
    retriever = BM25Retriever().index(_toy_docs())
    queries = [Query("q1", "cats"), Query("q2", "birds garden")]
    out = retriever.search_all(queries, topk=2)
    assert set(out.keys()) == {"q1", "q2"}
