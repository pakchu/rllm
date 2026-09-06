import pandas as pd

from training import evaluate_high_volatility_cross_alt_synchronized_flip_relay_gross9_novelty as novelty


def _clock(times, sides):
    return pd.DataFrame(
        {
            "entry_time": pd.to_datetime(times, utc=True),
            "exit_time": pd.to_datetime(times, utc=True) + pd.Timedelta("8h"),
            "side": sides,
        }
    )


def test_pair_passes_disjoint_balanced_clocks():
    candidate = _clock(["2024-01-01T00:00:00Z", "2024-02-01T00:00:00Z"], [1, -1])
    comparator = _clock(["2024-01-15T00:00:00Z", "2024-02-15T00:00:00Z"], [-1, 1])
    result = novelty.pair(candidate, comparator)
    assert result["passed"] is True
    assert all(result["checks"].values())


def test_pair_rejects_identical_clock():
    candidate = _clock(["2024-01-01T00:00:00Z", "2024-02-01T00:00:00Z"], [1, -1])
    result = novelty.pair(candidate, candidate.copy())
    assert result["passed"] is False
    assert result["checks"]["exact_entry_jaccard"] is False
