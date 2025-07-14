import numpy as np

from metrics import analyze


def test_analyze_basic() -> None:
    residuals = np.array([1.0, -1.0, 2.0, -2.0])
    stats = analyze(residuals)

    np.testing.assert_allclose(stats.rms, np.sqrt(np.mean(residuals**2)))
    np.testing.assert_allclose(stats.bias, np.mean(residuals))
    np.testing.assert_allclose(stats.std, np.std(residuals, ddof=0))


def test_analyze_with_true() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 1.5, 3.5])
    stats = analyze(y_pred, y_true, metrics=("rms", "bias", "std", "r2"))

    residuals = y_pred - y_true
    np.testing.assert_allclose(stats.rms, np.sqrt(np.mean(residuals**2)))
    np.testing.assert_allclose(stats.bias, np.mean(residuals))
    np.testing.assert_allclose(stats.std, np.std(residuals, ddof=0))
    from sklearn.metrics import r2_score

    np.testing.assert_allclose(stats.r2, r2_score(y_true, y_pred))
