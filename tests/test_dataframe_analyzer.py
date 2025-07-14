import pandas as pd

from metrics.analyzer.dataframe import DataFrameAnalyzer


def test_dataframe_summary_group() -> None:
    df = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=4, freq="H"),
            "site": ["a", "a", "b", "b"],
            "true": [1.0, 2.0, 3.0, 4.0],
            "pred": [1.0, 1.5, 2.5, 4.5],
        }
    ).set_index("time")
    analyzer = DataFrameAnalyzer(df, "pred", "true")
    out = analyzer.summary(group=["site", "time:H"], metrics=("rms",))
    assert set(out.columns).issuperset({"site", "rms"})
