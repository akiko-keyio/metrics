from metrics import analyze, Stats


def test_analyze_basic():
    stats = analyze([1, 2, 3], [1, 2, 3])
    assert isinstance(stats, Stats)
    assert stats.rms == 0
    assert stats.bias == 0
    assert stats.std == 0
