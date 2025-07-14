import pandas as pd

from metrics.analyzer.matrix import MatrixAnalyzer


def make_matrix() -> pd.DataFrame:
    data = {
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [1.5, 2.5, 3.5, 4.5],
    }
    idx = pd.date_range("2024-01-01", periods=4, freq="H")
    return pd.DataFrame(data, index=idx)


def test_matrix_summary_total() -> None:
    mat = make_matrix()
    analyzer = MatrixAnalyzer(mat)
    out = analyzer.summary()
    assert set(out.columns) >= {"rms", "bias", "std"}


def test_matrix_summary_group() -> None:
    mat = make_matrix()
    analyzer = MatrixAnalyzer(mat)
    out = analyzer.summary(group=["site", "time:H"], metrics=("rms",))
    assert "site" in out.columns
    assert "rms" in out.columns
