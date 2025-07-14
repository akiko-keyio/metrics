"""Generate residual analysis report from ``data/train.parquet``."""

from __future__ import annotations

import pandas as pd

from metrics.dataframe import DataFrameAnalyzer


def main() -> None:
    """Read ``train.parquet`` and write a CSV report of residual statistics."""
    df = pd.read_parquet("data/train.parquet")
    df=pd.concat([df]*10)
    print(len(df))
    df["ztd_nwm_pred"] = df["ztd_nwm"] - df["pred"]

# 3.8772036062255126
    import time
    t0=time.perf_counter()
    analyzer = DataFrameAnalyzer(df, ["ztd_nwm", "ztd_nwm_pred"], "ztd_gnss")
    rpt = analyzer.summary(group=["season","site"], metrics=("rms", "bias", "std"))

    print(time.perf_counter() - t0)
    rpt.round(10).to_csv("data/report_gen.csv",index=False)



if __name__ == "__main__":
    main()
