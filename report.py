"""Generate residual analysis report from ``data/train.parquet``."""

from __future__ import annotations

import pandas as pd

from metrics import DataFrameAnalyzer


def main() -> None:
    """Read ``train.parquet`` and write a CSV report of residual statistics."""
    df = pd.read_parquet("data/train.parquet").set_index("time")
    df["ztd_nwm_pred"] = df["ztd_nwm"] - df["pred"]

    reports = []
    for pred in ["ztd_nwm", "ztd_nwm_pred"]:
        analyzer = DataFrameAnalyzer(df, pred, "ztd_gnss")
        rpt = analyzer.summary(group=["site", "time:D"], metrics=("rms", "bias", "std"))
        rpt.insert(0, "variable", pred)
        reports.append(rpt)

    final = pd.concat(reports, ignore_index=True)
    final.to_csv("data/report.csv", index=False)


if __name__ == "__main__":
    main()
