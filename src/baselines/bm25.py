"""BM25 sparse-retrieval baseline (rank_bm25).

A deliberately simple, pure-Python baseline to establish the sparse-retrieval floor on
SciFact. Uses Okapi BM25 via `rank_bm25` with a minimal English tokenizer.

Run file output is TREC-style:

    qid Q0 doc_id rank score tag
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from ..data.schema import Doc, Query
from ..utils import get_logger

log = get_logger(__name__)

_DEFAULT_STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "has", "have", "he", "in", "is", "it", "its", "of", "on", "or",
    "she", "that", "the", "their", "there", "they", "this", "to", "was", "we",
    "were", "which", "will", "with", "you", "your",
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass
class BM25Config:
    k1: float = 1.5
    b: float = 0.75
    epsilon: float = 0.25
    lowercase: bool = True
    remove_stopwords: bool = True
    min_token_len: int = 2


def tokenize(text: str, cfg: BM25Config) -> List[str]:
    if cfg.lowercase:
        text = text.lower()
    tokens = _TOKEN_RE.findall(text)
    if cfg.min_token_len > 1:
        tokens = [t for t in tokens if len(t) >= cfg.min_token_len]
    if cfg.remove_stopwords:
        tokens = [t for t in tokens if t not in _DEFAULT_STOPWORDS]
    return tokens


@dataclass
class BM25Retriever:
    config: BM25Config = field(default_factory=BM25Config)
    _bm25: object = field(default=None, init=False, repr=False)
    _doc_ids: List[str] = field(default_factory=list, init=False, repr=False)

    def index(self, docs: Sequence[Doc]) -> "BM25Retriever":
        """Tokenize and build the BM25 index. Preserves doc order for ranking output."""
        from rank_bm25 import BM25Okapi

        if not docs:
            raise ValueError("Cannot index an empty document list.")

        log.info("Tokenizing %d documents for BM25 indexing.", len(docs))
        corpus_tokens = [tokenize(d.full_text, self.config) for d in docs]
        self._doc_ids = [d.doc_id for d in docs]
        self._bm25 = BM25Okapi(
            corpus_tokens,
            k1=self.config.k1,
            b=self.config.b,
            epsilon=self.config.epsilon,
        )
        log.info("BM25 index built.")
        return self

    def search(self, query: Query, topk: int = 100) -> List[Tuple[str, float]]:
        if self._bm25 is None:
            raise RuntimeError("Index not built. Call .index() first.")
        tokens = tokenize(query.text, self.config)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        # Order by score desc; stable tie-break by original doc order.
        ranked = sorted(
            zip(self._doc_ids, scores),
            key=lambda x: (-x[1], x[0]),
        )
        return [(doc_id, float(score)) for doc_id, score in ranked[:topk]]

    def search_all(
        self, queries: Iterable[Query], topk: int = 100
    ) -> Dict[str, List[Tuple[str, float]]]:
        out: Dict[str, List[Tuple[str, float]]] = {}
        for q in queries:
            out[q.qid] = self.search(q, topk=topk)
        return out
