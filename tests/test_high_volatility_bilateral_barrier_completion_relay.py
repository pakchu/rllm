import numpy as np

from training.build_high_volatility_bilateral_barrier_completion_relay_support import bilateral_passages


def test_bilateral_passages_identifies_upper_second() -> None:
    closes = np.ones(96)
    closes[10:30] = 0.98
    closes[30:] = 1.02
    assert bilateral_passages(closes, reference=1.0, threshold=0.01) == (30, 10, 1)


def test_bilateral_passages_identifies_lower_second() -> None:
    closes = np.ones(96)
    closes[8:40] = 1.02
    closes[40:] = 0.98
    assert bilateral_passages(closes, reference=1.0, threshold=0.01) == (8, 40, -1)


def test_bilateral_passages_requires_both_distinct_hits() -> None:
    closes = np.ones(96)
    closes[5:] = 1.02
    assert bilateral_passages(closes, reference=1.0, threshold=0.01) == (5, -1, 0)
