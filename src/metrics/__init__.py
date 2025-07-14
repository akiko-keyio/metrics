"""Public API for the metrics package."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .stats import ResidualStats
from .analyzer.array import ArrayAnalyzer
from .analyzer.dataframe import DataFrameAnalyzer
import pandas as pd

__all__ = ["ResidualStats", "analyze", "main", "ArrayAnalyzer", "DataFrameAnalyzer"]


def analyze(
    y_pred: Iterable[float] | pd.DataFrame,
    y_true: Iterable[float] | float | None = 0.0,
    *,
    pred_col: str | None = None,
    true_col: str | None = None,
    group: str | list[str] | None = "total",
    metrics: Iterable[str] | None = None,
) -> pd.DataFrame | ResidualStats:
    """Return residual statistics for the given inputs."""

    if isinstance(y_pred, pd.DataFrame) and pred_col and true_col:
        return DataFrameAnalyzer(y_pred, pred_col, true_col).summary(
            group=group, metrics=metrics
        )

    return ArrayAnalyzer(np.asarray(list(y_pred), dtype=float), y_true).summary(
        metrics
    )


def main() -> None:
    """Entry point for ``python -m metrics``."""

    print("Hello from metrics!")
