from __future__ import annotations

import pandas as pd

from training import (
    evaluate_high_volatility_multi_condition_pairwise_concordance_gross9_novelty as novelty,
)


def _clock(entries: list[str], sides: list[int]) -> pd.DataFrame:
    entry = pd.to_datetime(entries, utc=True)
    return pd.DataFrame(
        {
            "entry_time": entry,
            "exit_time": entry + pd.Timedelta(hours=8),
            "side": sides,
        }
    )


def test_pair_applies_every_frozen_metric_and_limit() -> None:
    candidate = _clock(
        ["2023-07-01T00:05:00Z", "2023-07-08T12:05:00Z"], [1, -1]
    )
    comparator = _clock(
        ["2023-07-03T00:05:00Z", "2023-07-11T12:05:00Z"], [-1, 1]
    )
    result = novelty.evaluate_pair(candidate, comparator)
    assert set(result["checks"]) == set(novelty.LIMITS)
    assert result["passed"] is True
    assert {
        "exact_entry_jaccard",
        "one_to_one_6h_max_matched_share",
        "occupied_5m_bar_jaccard",
        "absolute_signed_exposure_pearson",
    } <= set(result["metrics"])


def test_exact_entry_collision_fails_novelty() -> None:
    candidate = _clock(["2023-07-01T00:05:00Z"], [1])
    result = novelty.evaluate_pair(candidate, candidate.copy())
    assert result["checks"]["exact_entry_jaccard"] is False
    assert result["passed"] is False


def test_frozen_controls_bind_only_the_source_supported_pair() -> None:
    registration, support, manifest = novelty.load_frozen_controls()
    assert registration["policy_id"] == novelty.POLICY
    assert support["eligible_pairs_for_combination_gross9"] == [novelty.PAIR]
    assert support["pairs"][novelty.PAIR]["support_passed"] is True
    assert support["pairs"][novelty.PAIR]["clock"] == {
        "path": novelty.PAIR_CLOCK.as_posix(),
        "sha256": novelty.PAIR_CLOCK_SHA,
        "rows": novelty.PAIR_CLOCK_ROWS,
    }
    assert set(manifest["clocks"]) == set(novelty.gross9.EXPECTED_WEIGHTS)


def test_run_is_deterministic_and_keeps_economics_sealed(tmp_path) -> None:
    output = tmp_path / "gross9.json"
    first = novelty.run(output)
    first_bytes = output.read_bytes()
    second = novelty.run(output)
    assert output.read_bytes() == first_bytes
    assert second == first
    assert first["source_support_passed"] is True
    assert tuple(first["gross9_sleeves"]) == tuple(novelty.gross9.EXPECTED_WEIGHTS)
    assert first["advance_to_economic_outcomes"] == (
        first["source_support_passed"] and first["every_gross9_sleeve_passed"]
    )
    assert first["evidence_boundary"] == {
        "pair_clock_rows_opened": novelty.PAIR_CLOCK_ROWS,
        "gross9_structural_clock_rows_opened": sum(
            sum(counts.values()) for counts in novelty.gross9.EXPECTED_COUNTS.values()
        ),
        "btc_execution_rows_opened": 0,
        "btc_price_or_return_rows_opened": 0,
        "funding_rows_opened": 0,
        "economic_outcome_rows_opened": 0,
        "portfolio_return_or_pnl_metrics_computed": False,
        "outcomes_opened": False,
    }
    assert first["manifest_hash"] == novelty.canonical_hash(
        {key: value for key, value in first.items() if key != "manifest_hash"}
    )
