"""Public API for the metrics package."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .stats import ResidualStats
from .analyzer.array import ArrayAnalyzer

__all__ = ["ResidualStats", "analyze", "main", "ArrayAnalyzer"]


def analyze(
    y_pred: Iterable[float],
    y_true: Iterable[float] | float | None = 0.0,
    metrics: Iterable[str] | None = None,
) -> ResidualStats:
    """Return residual statistics for ``y_pred`` and ``y_true``."""

    analyzer = ArrayAnalyzer(np.asarray(list(y_pred), dtype=float), y_true)
    return analyzer.summary(metrics)


def main() -> None:
    """Entry point for ``python -m metrics``."""

    print("Hello from metrics!")
