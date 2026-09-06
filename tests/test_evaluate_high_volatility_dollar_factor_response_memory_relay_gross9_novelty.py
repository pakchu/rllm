import pandas as pd

from training import evaluate_high_volatility_dollar_factor_response_memory_relay_gross9_novelty as novelty


def frame(entries, sides, hold_hours=12):
    entry = pd.to_datetime(entries, utc=True)
    return pd.DataFrame({"entry_time": entry, "exit_time": entry + pd.Timedelta(hours=hold_hours), "side": sides})


def test_pair_passes_for_disjoint_clocks_and_exposure():
    candidate = frame(["2024-01-01 21:05", "2024-01-03 21:05"], [1, -1])
    comparator = frame(["2024-01-02 09:05", "2024-01-04 09:05"], [-1, 1])
    result = novelty.pair(candidate, comparator)
    assert result["passed"]
    assert all(result["checks"].values())


def test_pair_rejects_identical_clock():
    candidate = frame(["2024-01-01 21:05", "2024-01-03 21:05"], [1, -1])
    result = novelty.pair(candidate, candidate.copy())
    assert not result["passed"]
    assert not result["checks"]["exact_entry_jaccard"]
