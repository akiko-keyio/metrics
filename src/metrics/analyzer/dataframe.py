from __future__ import annotations
from collections.abc import Iterable
from typing import cast, List

import pandas as pd

from ..metrics import METRICS
from .base import BaseAnalyzer


class DataFrameAnalyzer(BaseAnalyzer):
    """Analyzer that operates directly on a ``pandas.DataFrame``.

    Parameters
    ----------
    df : pd.DataFrame
        The original dataframe containing the prediction, truth, and optional
        grouping columns. Must have a ``time`` column if grouping by temporal
        attributes (``year``, ``season``, ``month``, ``hour``).
    pred_col : str | list[str]
        One or multiple column names holding model predictions.
    true_col : str
        Ground-truth column.
    """

    def __init__(self, df: pd.DataFrame, pred_col: str | list[str], true_col: str) -> None:
        self.df: pd.DataFrame = df
        self.true_col: str = true_col

        # Normalize to list for internal use
        self.pred_cols: List[str] = [pred_col] if isinstance(pred_col, str) else list(pred_col)
        self.pred_col = pred_col  # keep original reference, backwards compat

        # Initialize BaseAnalyzer with the first predictor to satisfy parent API
        residuals = df[self.pred_cols[0]] - df[true_col]
        super().__init__(
            residuals.to_numpy(), df[true_col].to_numpy(), df[self.pred_cols[0]].to_numpy()
        )

    def _prepare_long(self, group_list: list[str]) -> pd.DataFrame:
        """Return a *long* DataFrame with columns [*group*..., ``var``, ``res``]."""
        df = self.df.copy()

        # If any temporal grouping keys are requested, extract them from the `time` column
        temporal_keys = {"year", "season", "month", "hour"}
        requested_time = [g for g in group_list if g in temporal_keys]
        if requested_time:
            if "time" not in df.columns:
                raise KeyError("A 'time' column is required for temporal grouping.")
            df = df.set_index("time")
            for key in requested_time:
                if key == "year":
                    df[key] = df.index.year
                elif key == "season":
                    def _month_to_season(m: int) -> str:
                        return (
                            "Winter", "Winter", "Winter",  # Jan-Mar
                            "Spring", "Spring", "Spring",  # Apr-Jun
                            "Summer", "Summer", "Summer",  # Jul-Sep
                            "Autumn", "Autumn", "Autumn"   # Oct-Dec
                        )[m - 1]
                    df[key] = df.index.month.map(_month_to_season)
                elif key == "month":
                    df[key] = df.index.month
                elif key == "hour":
                    df[key] = df.index.hour

        # Melt predictions into long form
        id_vars = [self.true_col] + group_list
        df_long = df[id_vars + self.pred_cols].melt(
            id_vars=id_vars,
            value_vars=self.pred_cols,
            var_name="var",
            value_name="pred",
        )
        df_long["res"] = df_long["pred"] - df_long[self.true_col]
        return df_long

    def summary(
        self,
        group: str | list[str] | None = "total",
        metrics: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Compute error metrics.

        Parameters
        ----------
        group : str | list[str] | None
            Grouping key(s). Can be any column in the DataFrame, including
            temporal attributes ``year``, ``season``, ``month``, or ``hour``
            extracted from the ``time`` column. ``None`` or ``"total"`` yields
            a single, global result.
        metrics : Iterable[str] | None
            Iterable of additional metric names found in ``METRICS``. Built-in
            metrics (``rms``, ``bias``, ``std``) are always included.

        Returns
        -------
        pd.DataFrame
            A tidy DataFrame with grouping columns, a ``var`` column indicating
            the prediction variable, and one column per requested metric.
        """
        # Determine grouping columns
        if group in (None, "total"):
            group_list: list[str] = []
        else:
            group_list = [group] if isinstance(group, str) else list(cast(Iterable[str], group))

        df_long = self._prepare_long(group_list)

        # Prepare metrics
        base_metrics = {"rms", "bias", "std"}
        wanted = list(base_metrics.union(metrics or []))
        func_list = [METRICS[name] for name in wanted]

        # Group and aggregate
        group_keys = group_list + ["var"] if group_list else ["var"]
        result = df_long.groupby(group_keys)["res"].agg(func_list)

        # Rename columns deterministically
        name_map = {func.__name__: want for func, want in zip(func_list, wanted)}
        result = result.rename(columns=name_map).reset_index()

        # Reorder: grouping keys then metrics
        result = result[group_keys + wanted]
        return result
