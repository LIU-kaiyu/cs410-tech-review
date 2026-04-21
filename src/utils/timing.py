"""Simple context-manager timer."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Timer:
    label: str = "block"
    start: float = field(default=0.0, init=False)
    elapsed: float = field(default=0.0, init=False)
    _logger: Optional[object] = None

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed = time.perf_counter() - self.start
        if self._logger is not None:
            self._logger.info("%s took %.3fs", self.label, self.elapsed)

    def __float__(self) -> float:
        return float(self.elapsed)
