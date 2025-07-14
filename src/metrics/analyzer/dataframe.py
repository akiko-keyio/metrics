"""Analyzer for pandas DataFrame residuals."""
# mypy: ignore-errors

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import pandas as pd

from .. import grouping
from ..metrics import METRICS
from .base import BaseAnalyzer


class DataFrameAnalyzer(BaseAnalyzer):
    """Analyze residuals stored in a :class:`pandas.DataFrame`."""

    def __init__(self, df: pd.DataFrame, pred_col: str, true_col: str) -> None:
        """Create an analyzer from ``df`` using prediction and truth columns."""
        self.df = df
        self.pred_col = pred_col
        self.true_col = true_col
        residuals = df[pred_col] - df[true_col]
        super().__init__(
            residuals.to_numpy(), df[true_col].to_numpy(), df[pred_col].to_numpy()
        )
        self.residuals = residuals

    def summary(
        self,
        group: str | list[str] | None = "total",
        metrics: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Return grouped statistics for the DataFrame residuals."""
        # type: ignore[override]
        if group in (None, "total"):
            stats = super().summary(metrics)
            return pd.DataFrame([stats.as_dict()])

        if isinstance(group, str):
            group_list = [group]
        else:
            group_list = list(cast(Iterable[str], group))

        df = self.df.copy()
        df["res"] = self.residuals
        group_keys: list[pd.Series] = []
        for g in group_list:
            if g.startswith("time:"):
                rule = g.split(":", 1)[1]
                group_keys.append(grouping.make_time_grouper(df.index, rule))
            else:
                group_keys.append(df[g])

        base = ["rms", "bias", "std"]
        wanted = base + list(metrics or [])
        funcs = [METRICS[name] for name in wanted]
        result = df.groupby(group_keys)["res"].agg(funcs)
        result.columns = wanted
        return result.reset_index()
