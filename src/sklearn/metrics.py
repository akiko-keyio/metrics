from __future__ import annotations

import numpy as np

__all__ = ["r2_score"]


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the coefficient of determination (R^2)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.nansum((y_true - y_pred) ** 2)
    ss_tot = np.nansum((y_true - np.nanmean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1.0 - ss_res / ss_tot
