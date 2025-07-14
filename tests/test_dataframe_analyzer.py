import numpy as np
import pandas as pd
from pytest import approx

from metrics import DataFrameAnalyzer


def test_summary_total_numpy():
    df = pd.DataFrame(
        {
            "pred1": [1.0, 2.0, 3.0],
            "pred2": [1.0, 2.1, 2.9],
            "true": [1.1, 1.9, 3.05],
        }
    )
    analyzer = DataFrameAnalyzer(df, ["pred1", "pred2"], "true")
    out = analyzer.summary(group=None, engine="numpy", ddof=0)

    res1 = df["pred1"].to_numpy() - df["true"].to_numpy()
    res2 = df["pred2"].to_numpy() - df["true"].to_numpy()

    expected = {
        "pred1": {
            "rms": np.sqrt(np.nanmean(res1**2)),
            "bias": np.nanmean(res1),
            "std": np.nanstd(res1, ddof=0),
        },
        "pred2": {
            "rms": np.sqrt(np.nanmean(res2**2)),
            "bias": np.nanmean(res2),
            "std": np.nanstd(res2, ddof=0),
        },
    }

    assert list(out["var"]) == ["pred1", "pred2"]
    for idx, var in enumerate(["pred1", "pred2"]):
        assert out.loc[idx, "rms"] == approx(expected[var]["rms"])
        assert out.loc[idx, "bias"] == approx(expected[var]["bias"])
        assert out.loc[idx, "std"] == approx(expected[var]["std"])
