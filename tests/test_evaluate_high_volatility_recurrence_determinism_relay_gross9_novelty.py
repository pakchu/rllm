import pandas as pd

from training import evaluate_high_volatility_recurrence_determinism_relay_gross9_novelty as novelty


def _clock(entries: list[str], sides: list[int]) -> pd.DataFrame:
    entry = pd.to_datetime(entries, utc=True)
    return pd.DataFrame(
        {
            "entry_time": entry,
            "exit_time": entry + pd.Timedelta(hours=24),
            "side": sides,
        }
    )


def test_pair_applies_all_frozen_limits() -> None:
    candidate = _clock(
        ["2023-07-01T00:05:00Z", "2023-07-08T12:05:00Z"],
        [1, -1],
    )
    comparator = _clock(
        ["2023-07-03T00:05:00Z", "2023-07-11T12:05:00Z"],
        [-1, 1],
    )
    result = novelty.pair(candidate, comparator)
    assert set(result["checks"]) == set(novelty.LIMITS)
    assert result["passed"] is True


def test_exact_clock_collision_fails_novelty() -> None:
    candidate = _clock(["2023-07-01T00:05:00Z"], [1])
    comparator = _clock(["2023-07-01T00:05:00Z"], [1])
    result = novelty.pair(candidate, comparator)
    assert result["checks"]["exact_entry_jaccard"] is False
    assert result["passed"] is False


def test_preregistered_limit_names_are_bound_without_changing_metric_names() -> None:
    registration = novelty.load(novelty.PREREG)
    assert registration["novelty_gates"]["occupied_5m_jaccard_max"] == 0.25
    assert novelty.LIMITS["occupied_5m_bar_jaccard"] == 0.25
