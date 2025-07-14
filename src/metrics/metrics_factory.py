# metrics/metrics_factory.py (修正后)

"""Collection of built-in metric functions used by the analyzers."""

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
    """Root mean squared error of the residuals."""
    return np.sqrt(np.nanmean(res**2))


@register_metric("bias")
def bias(res: np.ndarray) -> float:
    """Mean bias of the residuals."""
    return np.nanmean(res)


@register_metric("std")
def std(res: np.ndarray, *, ddof: int = 0) -> float:
    """
    Standard deviation of the residuals.

    Parameters
    ----------
    res : np.ndarray
        Array of residuals.
    ddof : int, default 0
        Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N is the number of elements. ddof=1 provides an unbiased estimate.
    """
    # np.nanstd 内部已经处理了 n <= ddof 的情况，会返回 nan 并有 RuntimeWarning
    # 我们这里无需额外检查
    return np.nanstd(res, ddof=ddof)


@register_metric("r2")
def r2(res: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination without external dependencies."""
    ss_res = np.nansum(res**2)
    ss_tot = np.nansum((y_true - np.nanmean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1.0 - ss_res / ss_tot
