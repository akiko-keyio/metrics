"""Helper utilities for time-based grouping."""

from __future__ import annotations

import pandas as pd


def make_time_grouper(index: pd.DatetimeIndex, rule: str) -> pd.Series:
    """Return a Series representing *index* grouped by *rule*."""

    rule = rule.upper()
    if rule in {"Q", "QS", "Q-DEC"}:
        return index.to_period("Q")
    return index.floor(rule)
