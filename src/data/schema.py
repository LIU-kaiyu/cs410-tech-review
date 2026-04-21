"""Typed schema for corpus, queries, and qrels.

A single canonical representation shared by the BM25 baseline, the ColBERT/RAGatouille
wrappers, and the evaluation harness. Keeping this narrow and typed prevents downstream
code from coupling to BEIR's raw dict shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List


@dataclass(frozen=True)
class Doc:
    doc_id: str
    title: str
    text: str

    @property
    def full_text(self) -> str:
        if self.title:
            return f"{self.title}. {self.text}"
        return self.text


@dataclass(frozen=True)
class Query:
    qid: str
    text: str


@dataclass(frozen=True)
class Qrel:
    qid: str
    doc_id: str
    relevance: int


@dataclass
class Corpus:
    docs: List[Doc] = field(default_factory=list)
    queries: List[Query] = field(default_factory=list)
    qrels: List[Qrel] = field(default_factory=list)

    @property
    def doc_ids(self) -> List[str]:
        return [d.doc_id for d in self.docs]

    @property
    def query_ids(self) -> List[str]:
        return [q.qid for q in self.queries]

    def doc_map(self) -> Dict[str, Doc]:
        return {d.doc_id: d for d in self.docs}

    def query_map(self) -> Dict[str, Query]:
        return {q.qid: q for q in self.queries}

    def qrels_map(self) -> Dict[str, Dict[str, int]]:
        """Return ranx-compatible {qid: {doc_id: rel}}."""
        out: Dict[str, Dict[str, int]] = {}
        for qrel in self.qrels:
            out.setdefault(qrel.qid, {})[qrel.doc_id] = qrel.relevance
        return out

    def __iter_docs__(self) -> Iterator[Doc]:
        return iter(self.docs)

    def validate(self) -> None:
        """Sanity-check cross-references. Raises ValueError on problems."""
        doc_ids = set(self.doc_ids)
        query_ids = set(self.query_ids)
        if not doc_ids:
            raise ValueError("Corpus has no documents.")
        if not query_ids:
            raise ValueError("Corpus has no queries.")

        orphan_qrel_docs = {q.doc_id for q in self.qrels} - doc_ids
        if orphan_qrel_docs:
            raise ValueError(
                f"{len(orphan_qrel_docs)} qrel doc_ids not found in corpus; "
                f"examples: {list(orphan_qrel_docs)[:3]}"
            )

        orphan_qrel_qids = {q.qid for q in self.qrels} - query_ids
        if orphan_qrel_qids:
            raise ValueError(
                f"{len(orphan_qrel_qids)} qrel qids not found in queries; "
                f"examples: {list(orphan_qrel_qids)[:3]}"
            )

        for d in self.docs:
            if not d.doc_id:
                raise ValueError("Document with empty doc_id.")
            if not d.text and not d.title:
                raise ValueError(f"Document {d.doc_id} has empty title and text.")

        for q in self.queries:
            if not q.qid:
                raise ValueError("Query with empty qid.")
            if not q.text:
                raise ValueError(f"Query {q.qid} has empty text.")
