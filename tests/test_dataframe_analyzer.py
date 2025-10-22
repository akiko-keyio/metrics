import numpy as np
import pandas as pd
from pytest import approx

from metrics import DataFrameAnalyzer, register_metric


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


def test_summary_with_marginals_numpy():
    df = pd.DataFrame(
        {
            "site": ["A", "A", "B", "B"],
            "season": ["Winter", "Summer", "Winter", "Summer"],
            "pred": [1.4, 2.5, 1.7, 4.2],
            "true": [1.0, 2.0, 1.5, 4.5],
            "time": pd.to_datetime(
                [
                    "2024-01-15",
                    "2024-06-15",
                    "2024-01-20",
                    "2024-06-20",
                ]
            ),
        }
    )
    analyzer = DataFrameAnalyzer(df, "pred", "true")
    out = analyzer.summary(
        group=["site", "season"],
        engine="numpy",
        ddof=0,
        include_marginals=True,
    )

    total_label = DataFrameAnalyzer._TOTAL_LABEL
    expected_groups = {
        ("A", "Winter"),
        ("A", "Summer"),
        ("B", "Winter"),
        ("B", "Summer"),
        (total_label, "Winter"),
        (total_label, "Summer"),
        ("A", total_label),
        ("B", total_label),
        (total_label, total_label),
    }

    observed_groups = {(row["site"], row["season"]) for _, row in out.iterrows()}
    assert observed_groups == expected_groups

    def expected_metrics(mask):
        residuals = df.loc[mask, "pred"] - df.loc[mask, "true"]
        return {
            "bias": residuals.mean(),
            "rms": np.sqrt((residuals**2).mean()),
            "std": residuals.std(ddof=0),
        }

    checks = {
        (total_label, total_label): expected_metrics(df.index),
        (total_label, "Winter"): expected_metrics(df["season"] == "Winter"),
        ("A", total_label): expected_metrics(df["site"] == "A"),
    }

    for (site, season), metrics in checks.items():
        row = out[(out["site"] == site) & (out["season"] == season)].iloc[0]
        assert row["bias"] == approx(metrics["bias"])
        assert row["rms"] == approx(metrics["rms"])
        assert row["std"] == approx(metrics["std"])


def test_summary_total_includes_var_with_generic_engine():
    @register_metric("p95")
    def _p95(res: np.ndarray) -> float:
        return float(np.nanpercentile(res, 95))

    df = pd.DataFrame(
        {
            "pred": [1.1, 2.5, 3.4, 4.2],
            "true": [1.0, 2.0, 3.0, 4.0],
        }
    )

    analyzer = DataFrameAnalyzer(df, "pred", "true")
    out = analyzer.summary(group=None, metrics=("rms", "bias", "std", "p95"))

    assert list(out["var"]) == ["pred"]
    assert out.loc[0, "p95"] == approx(0.485)
