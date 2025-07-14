"""Generate residual analysis report from ``data/train.parquet``."""

from __future__ import annotations

import pandas as pd

from metrics import DataFrameAnalyzer


def main() -> None:
    """Read ``train.parquet`` and write a CSV report of residual statistics."""
    df = pd.read_parquet("data/train.parquet")
    df["ztd_nwm_pred"] = df["ztd_nwm"] - df["pred"]
    print(df)
# 3.8772036062255126
    analyzer = DataFrameAnalyzer(df, ["ztd_nwm", "ztd_nwm_pred"], "ztd_gnss")
    rpt = analyzer.summary(group=["month","site"], metrics=("rms", "bias", "std"))
    rpt.to_csv("data/report.csv",index=False)


if __name__ == "__main__":
    main()
