"""Base classes used to compute residual statistics.

An analyzer prepares residual arrays and dispatches metric functions declared in
``metrics.METRICS``.  Concrete subclasses provide domain specific logic for
building the residuals from various input types.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable

import numpy as np

from ..metrics import METRICS
from ..stats import ResidualStats


class BaseAnalyzer:
    """Prepare residuals and compute metric summaries."""

    def __init__(
        self,
        residuals: np.ndarray,
        y_true: np.ndarray | None = None,
        y_pred: np.ndarray | None = None,
    ) -> None:
        """Store residuals and optional truth/prediction arrays."""
        self.res = np.asarray(residuals, dtype=float)
        self.y_true = np.asarray(y_true, dtype=float) if y_true is not None else None
        self.y_pred = np.asarray(y_pred, dtype=float) if y_pred is not None else None

    def _run_metric(self, name: str) -> float:
        """Execute the metric *name* with available data."""
        fn = METRICS[name]
        sig = inspect.signature(fn)
        kwargs = {}
        if "res" in sig.parameters:
            kwargs["res"] = self.res
        if "y_true" in sig.parameters and self.y_true is not None:
            kwargs["y_true"] = self.y_true
        if "y_pred" in sig.parameters and self.y_pred is not None:
            kwargs["y_pred"] = self.y_pred
        return fn(**kwargs)

    def summary(self, metrics: Iterable[str] | None = None) -> ResidualStats:
        """Return a :class:`ResidualStats` populated with the requested metrics."""
        base = {"rms", "bias", "std"}
        wanted = base | set(metrics or ())
        results = {m: self._run_metric(m) for m in wanted}
        return ResidualStats(
            rms=results["rms"],
            bias=results["bias"],
            std=results["std"],
            r2=results.get("r2"),
        )
