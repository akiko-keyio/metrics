"""High level entry points for the residual analysis package.

The package exposes a single :func:`analyze` convenience function which will
dispatch the input data to the appropriate analyzer implementation.  It
supports numpy arrays, pandas ``DataFrame`` objects and plain iterables.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .stats import ResidualStats
from .analyzer.array import ArrayAnalyzer
from .analyzer.dataframe import DataFrameAnalyzer
from .analyzer.matrix import MatrixAnalyzer
import pandas as pd

__all__ = [
    "ResidualStats",
    "analyze",
    "main",
    "ArrayAnalyzer",
    "DataFrameAnalyzer",
    "MatrixAnalyzer",
]


def analyze(
    y_pred: Iterable[float] | pd.DataFrame,
    y_true: Iterable[float] | float | None = 0.0,
    *,
    pred_col: str | None = None,
    true_col: str | None = None,
    group: str | list[str] | None = "total",
    metrics: Iterable[str] | None = None,
) -> pd.DataFrame | ResidualStats:
    """Compute residual statistics for various input types.

    Parameters
    ----------
    y_pred:
        Predicted values or a :class:`pandas.DataFrame` containing them.
    y_true:
        True values matching ``y_pred`` or a scalar baseline.
    pred_col, true_col:
        When ``y_pred`` is a DataFrame these specify the column names.
    group:
        Optional grouping rule(s) for aggregating the statistics.  ``"total"``
        returns overall metrics while ``"time:<rule>"`` or a list provides
        time-based or multi column grouping.
    metrics:
        Additional metric names to compute besides the defaults.

    Returns
    -------
    pandas.DataFrame | ResidualStats
        Either a table of grouped statistics or a single ``ResidualStats``
        instance depending on the ``group`` argument.
    """

    if isinstance(y_pred, pd.DataFrame):
        if pred_col and true_col:
            return DataFrameAnalyzer(y_pred, pred_col, true_col).summary(
                group=group, metrics=metrics
            )
        return MatrixAnalyzer(y_pred).summary(group=group, metrics=metrics)

    return ArrayAnalyzer(np.asarray(list(y_pred), dtype=float), y_true).summary(metrics)


def main() -> None:
    """Entry point for ``python -m metrics`` used by the console script."""

    print("Hello from metrics!")
