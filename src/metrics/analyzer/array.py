"""Analyzer for 1-D arrays."""

from __future__ import annotations

import numpy as np

from .base import BaseAnalyzer


class ArrayAnalyzer(BaseAnalyzer):
    """Analyze vector residuals."""

    def __init__(self, y_pred: np.ndarray, y_true: np.ndarray | float = 0.0) -> None:
        """Compute residuals from ``y_pred`` and ``y_true`` and initialise base."""
        y_pred = np.asarray(y_pred, dtype=float)
        if np.isscalar(y_true):
            y_true_arr = np.full_like(y_pred, float(y_true))
        else:
            y_true_arr = np.asarray(y_true, dtype=float)
        residuals = y_pred - y_true_arr
        super().__init__(residuals, y_true_arr, y_pred)
