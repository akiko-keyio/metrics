"""Public API for the metrics package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .dataframe import DataFrameAnalyzer
from .metrics_factory import METRICS, register_metric


@dataclass
class Stats:
    """Container for metric results."""

    rms: float
    bias: float
    std: float


def analyze(pred: Sequence[float], true: Sequence[float], *, ddof: int = 1) -> Stats:
    """Compute basic residual statistics for two sequences."""
    arr_pred = np.asarray(pred, dtype=float)
    arr_true = np.asarray(true, dtype=float)
    res = arr_pred - arr_true
    rms = METRICS["rms"](res=res)
    bias = METRICS["bias"](res=res)
    std = METRICS["std"](res=res, ddof=ddof)
    return Stats(rms=rms, bias=bias, std=std)


__all__ = [
    "DataFrameAnalyzer",
    "analyze",
    "Stats",
    "register_metric",
    "METRICS",
]
