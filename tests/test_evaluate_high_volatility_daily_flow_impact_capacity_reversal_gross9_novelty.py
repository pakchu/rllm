import pandas as pd

from training import evaluate_high_volatility_daily_flow_impact_capacity_reversal_gross9_novelty as e


def clock(times, sides):
    entries = pd.to_datetime(times, utc=True)
    return pd.DataFrame(
        {
            "entry_time": entries,
            "exit_time": entries + pd.Timedelta(hours=12),
            "side": sides,
        }
    )


def test_pair_metrics_apply_all_frozen_limits():
    candidate = clock(["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"], [1, -1])
    distant = clock(["2024-02-01T00:00:00Z"], [1])
    result = e.evaluate_pair(candidate, distant)
    assert result["passed"] is True
    assert all(result["checks"].values())
    assert set(result["checks"]) == set(e.LIMITS)


def test_exact_overlap_fails_frozen_limits():
    candidate = clock(["2024-01-01T00:00:00Z"], [1])
    result = e.evaluate_pair(candidate, candidate.copy())
    assert result["passed"] is False
    assert result["checks"]["exact_entry_jaccard"] is False
