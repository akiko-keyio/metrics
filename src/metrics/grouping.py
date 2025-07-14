"""Helper utilities for time-based grouping.

These helpers convert :class:`~pandas.DatetimeIndex` objects into labels that
can be fed directly into ``pandas`` aggregation functions.  Only a very small
subset of ``pandas`` grouping functionality is implemented as required by the
package.
"""

from __future__ import annotations

import pandas as pd


def make_time_grouper(index: pd.Index | pd.Series, rule: str) -> pd.Series:
    """Return labels for ``index`` grouped according to ``rule``.

    Parameters
    ----------
    index:
        Index to convert into grouping labels.
    rule:
        Pandas offset alias such as ``"H"`` or ``"3D"``.

    Returns
    -------
    pandas.Series
        A series of group labels used for aggregation.
    """

    rule = rule.upper()
    dt = pd.to_datetime(index)
    if isinstance(dt, pd.Series):
        if rule in {"Q", "QS", "Q-DEC"}:
            return dt.dt.to_period("Q")
        return dt.dt.floor(rule)
    if rule in {"Q", "QS", "Q-DEC"}:
        return dt.to_period("Q")
    return dt.floor(rule)
