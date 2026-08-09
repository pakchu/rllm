import pandas as pd

from training import evaluate_range_body_efficiency_fracture_reversal_gross9_novelty as novelty


def _clock(entries, hold_hours=72, sides=None):
    entries = pd.to_datetime(entries, utc=True)
    if sides is None:
        sides = [1] * len(entries)
    return pd.DataFrame(
        {
            "entry_time": entries,
            "exit_time": entries + pd.Timedelta(hours=hold_hours),
            "side": sides,
        }
    )


def test_disjoint_pair_passes_all_limits():
    candidate = _clock(["2024-01-01T00:05:00Z", "2024-01-11T00:05:00Z"], sides=[1, -1])
    comparator = _clock(["2024-01-06T12:00:00Z", "2024-01-16T12:00:00Z"])
    result = novelty.evaluate_pair(candidate, comparator)
    assert result["passed"] is True
    assert all(result["checks"].values())


def test_exact_overlap_fails():
    candidate = _clock(["2024-01-01T00:05:00Z", "2024-01-11T00:05:00Z"])
    comparator = candidate.copy()
    result = novelty.evaluate_pair(candidate, comparator)
    assert result["passed"] is False
    assert result["checks"]["exact_entry_jaccard"] is False
