import pandas as pd

from training import evaluate_confirmation_ladder_settlement_impact_efficiency_relay_gross9_novelty as novelty


def _clock(times, sides):
    entries = pd.to_datetime(times, utc=True)
    return pd.DataFrame({"entry_time": entries, "exit_time": entries + pd.Timedelta("6h"), "side": sides})


def test_pair_passes_disjoint_asynchronous_clocks():
    candidate = _clock(["2024-01-01T00:00:00Z", "2024-02-01T00:00:00Z"], [1, -1])
    comparator = _clock(["2024-01-15T00:00:00Z", "2024-02-15T00:00:00Z"], [-1, 1])
    assert novelty.evaluate_pair(candidate, comparator)["passed"] is True


def test_pair_rejects_identical_clock():
    candidate = _clock(["2024-01-01T00:00:00Z", "2024-02-01T00:00:00Z"], [1, -1])
    result = novelty.evaluate_pair(candidate, candidate.copy())
    assert result["passed"] is False
    assert result["checks"]["exact_entry_jaccard"] is False
