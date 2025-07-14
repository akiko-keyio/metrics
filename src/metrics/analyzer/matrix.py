"""Analyzer for 2-D matrices with time index and site columns."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import pandas as pd

from .. import grouping
from ..metrics import METRICS
from .base import BaseAnalyzer


class MatrixAnalyzer(BaseAnalyzer):
    """Analyze residual matrices (time rows, site columns)."""

    def __init__(self, mat: pd.DataFrame) -> None:
        """Create an analyzer from a matrix-like ``DataFrame``."""
        self.mat = mat
        super().__init__(mat.to_numpy().ravel())

    def summary(  # type: ignore[override]
        self,
        group: str | list[str] | None = "total",
        metrics: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Return residual statistics aggregated over the requested groups."""
        if group in (None, "total"):
            stats = super().summary(metrics)
            return pd.DataFrame([stats.as_dict()])

        if isinstance(group, str):
            group_list = [group]
        else:
            group_list = list(cast(Iterable[str], group))

        long = (
            self.mat.stack().rename_axis(["time", "site"]).rename("res").reset_index()
        )
        group_keys: list[pd.Series] = []
        for g in group_list:
            if g == "site":
                group_keys.append(long["site"])
            elif g.startswith("time:"):
                rule = g.split(":", 1)[1]
                group_keys.append(grouping.make_time_grouper(long["time"], rule))
            else:
                raise KeyError(f"unsupported group {g}")

        base = {"rms", "bias", "std"}
        wanted = base | set(metrics or ())
        agg_funcs = {name: METRICS[name] for name in wanted}
        result = long.groupby(group_keys)["res"].agg(agg_funcs)
        return result.reset_index()
