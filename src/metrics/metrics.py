from __future__ import annotations

from collections.abc import Callable

import numpy as np

METRICS: dict[str, Callable[..., float]] = {}


def register_metric(
    name: str,
) -> Callable[[Callable[..., float]], Callable[..., float]]:
    """Register *name* as a metric."""

    def decorator(fn: Callable[..., float]) -> Callable[..., float]:
        METRICS[name] = fn
        return fn

    return decorator


@register_metric("rms")
def rms(res: np.ndarray) -> float:
    return np.sqrt(np.nanmean(res**2))


@register_metric("bias")
def bias(res: np.ndarray) -> float:
    return np.nanmean(res)


@register_metric("std")
def std(res: np.ndarray) -> float:
    return np.nanstd(res, ddof=0)


@register_metric("r2")
def r2(res: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the coefficient of determination using scikit-learn."""

    from sklearn.metrics import r2_score

    return r2_score(y_true, y_pred)
