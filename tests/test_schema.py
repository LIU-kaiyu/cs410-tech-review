"""Unit tests for the corpus schema and its cross-reference validator."""
from __future__ import annotations

import pytest

from src.data.schema import Corpus, Doc, Qrel, Query


def _mini_corpus() -> Corpus:
    docs = [
        Doc(doc_id="d1", title="T1", text="alpha"),
        Doc(doc_id="d2", title="T2", text="beta"),
    ]
    queries = [Query(qid="q1", text="alpha beta")]
    qrels = [Qrel(qid="q1", doc_id="d1", relevance=1)]
    return Corpus(docs=docs, queries=queries, qrels=qrels)


def test_full_text_combines_title_and_text():
    assert Doc("d", "Title", "body").full_text == "Title. body"
    assert Doc("d", "", "body").full_text == "body"


def test_validate_happy_path():
    _mini_corpus().validate()


def test_qrels_map_roundtrip():
    c = _mini_corpus()
    qm = c.qrels_map()
    assert qm == {"q1": {"d1": 1}}


def test_validate_rejects_orphan_qrel_doc():
    c = _mini_corpus()
    c.qrels.append(Qrel(qid="q1", doc_id="ghost", relevance=1))
    with pytest.raises(ValueError, match="qrel doc_ids"):
        c.validate()


def test_validate_rejects_orphan_qrel_qid():
    c = _mini_corpus()
    c.qrels.append(Qrel(qid="qGhost", doc_id="d1", relevance=1))
    with pytest.raises(ValueError, match="qrel qids"):
        c.validate()


def test_validate_rejects_empty_query_text():
    c = _mini_corpus()
    c.queries.append(Query(qid="q2", text=""))
    with pytest.raises(ValueError, match="empty text"):
        c.validate()


def test_validate_rejects_doc_with_no_title_or_text():
    c = _mini_corpus()
    c.docs.append(Doc(doc_id="d3", title="", text=""))
    with pytest.raises(ValueError, match="empty title and text"):
        c.validate()
