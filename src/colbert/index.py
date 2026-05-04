"""ColBERT indexing via RAGatouille.

Wraps `RAGPretrainedModel` to build a late-interaction index over a document
collection, exposing `nbits` and `max_document_length` as first-class knobs. Records
build-time and on-disk index size for the ablation.
"""
from __future__ import annotations

import json
import inspect
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from ..data.schema import Doc
from ..utils import ensure_dir, get_logger

log = get_logger(__name__)


@dataclass
class ColBERTIndexConfig:
    checkpoint: str = "colbert-ir/colbertv2.0"
    index_name: str = "scifact"
    nbits: int = 2
    max_document_length: int = 256
    index_root: str = ".ragatouille"


@dataclass
class IndexStats:
    index_name: str
    nbits: int
    effective_nbits: int
    n_docs: int
    build_time_s: float
    index_bytes: int = 0
    index_path: str = ""


def _dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def _set_effective_nbits(rag_model: object, nbits: int) -> int:
    """Set the ColBERT residual-compression bit width under RAGatouille.

    RAGatouille's high-level `RAGPretrainedModel.index()` API does not expose
    `nbits`, but the wrapped ColBERT model keeps the live ColBERTConfig at
    `rag_model.model.config`. The PLAID indexer consumes that config when
    building the index.
    """
    if nbits <= 0:
        raise ValueError(f"nbits must be positive, got {nbits}.")

    colbert_model = getattr(rag_model, "model", None)
    colbert_config = getattr(colbert_model, "config", None)
    if colbert_config is None:
        raise RuntimeError(
            "Could not find RAGatouille's wrapped ColBERT config; "
            "cannot set nbits for the ablation."
        )

    setattr(colbert_config, "nbits", int(nbits))

    # Keep the inference checkpoint config aligned when RAGatouille has already
    # materialized it. Indexing uses colbert_model.config, but this avoids later
    # surprises when inspecting the loaded model.
    inference_ckpt = getattr(colbert_model, "inference_ckpt", None)
    inference_config = getattr(inference_ckpt, "colbert_config", None)
    if inference_config is not None:
        setattr(inference_config, "nbits", int(nbits))

    effective = int(getattr(colbert_config, "nbits"))
    if effective != int(nbits):
        raise RuntimeError(f"Requested nbits={nbits}, but effective nbits={effective}.")
    return effective


def _supports_kwarg(fn: object, name: str) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return name in signature.parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in signature.parameters.values()
    )


@dataclass
class ColBERTIndexer:
    config: ColBERTIndexConfig = field(default_factory=ColBERTIndexConfig)
    _model: object = field(default=None, init=False, repr=False)

    def _load_model(self):
        if self._model is not None:
            return self._model
        from ragatouille import RAGPretrainedModel  # lazy import

        log.info("Loading ColBERT checkpoint: %s", self.config.checkpoint)
        self._model = RAGPretrainedModel.from_pretrained(
            self.config.checkpoint,
            index_root=self.config.index_root,
        )
        return self._model

    def build_index(
        self,
        docs: Sequence[Doc],
        index_name: Optional[str] = None,
        overwrite: bool = True,
    ) -> IndexStats:
        """Build a ColBERT index. Returns build stats including on-disk size."""
        name = index_name or self.config.index_name
        model = self._load_model()
        effective_nbits = _set_effective_nbits(model, self.config.nbits)

        collection = [d.full_text for d in docs]
        document_ids = [d.doc_id for d in docs]

        log.info(
            "Building ColBERT index '%s' (nbits=%d, n_docs=%d, max_doc_len=%d)",
            name, effective_nbits, len(docs), self.config.max_document_length,
        )
        t0 = time.perf_counter()
        index_kwargs = {
            "collection": collection,
            "document_ids": document_ids,
            "index_name": name,
            "max_document_length": self.config.max_document_length,
            "split_documents": False,
            "overwrite_index": overwrite,
        }
        if _supports_kwarg(model.index, "bsize"):
            index_kwargs["bsize"] = 32
        model.index(**index_kwargs)
        build_time = time.perf_counter() - t0

        index_path = Path(self.config.index_root) / "colbert" / "indexes" / name
        stats = IndexStats(
            index_name=name,
            nbits=self.config.nbits,
            effective_nbits=effective_nbits,
            n_docs=len(docs),
            build_time_s=build_time,
            index_bytes=_dir_size_bytes(index_path),
            index_path=str(index_path),
        )
        log.info(
            "Index '%s' built in %.1fs, %.1f MiB on disk.",
            name, build_time, stats.index_bytes / (1024 * 1024),
        )
        return stats

    def load_existing(self, index_name: Optional[str] = None):
        """Load a previously built index for search."""
        from ragatouille import RAGPretrainedModel  # lazy import

        name = index_name or self.config.index_name
        index_path = Path(self.config.index_root) / "colbert" / "indexes" / name
        log.info("Loading existing ColBERT index: %s", index_path)
        self._model = RAGPretrainedModel.from_index(str(index_path))
        return self._model


def save_index_stats(stats: IndexStats, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    payload = {
        "index_name": stats.index_name,
        "nbits": stats.nbits,
        "effective_nbits": stats.effective_nbits,
        "n_docs": stats.n_docs,
        "build_time_s": stats.build_time_s,
        "index_bytes": stats.index_bytes,
        "index_mib": stats.index_bytes / (1024 * 1024),
        "index_path": stats.index_path,
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log.info("Wrote index stats to %s", p)
