from __future__ import annotations

import pandas as pd

from training import (
    evaluate_high_volatility_state_ordered_filter_gross9_novelty as novelty,
)
from training import preregister_high_volatility_state_ordered_filter as prereg


def _clock(entries: list[str], sides: list[int]) -> pd.DataFrame:
    entry = pd.to_datetime(entries, utc=True)
    return pd.DataFrame(
        {
            "entry_time": entry,
            "exit_time": entry + pd.Timedelta(hours=8),
            "side": sides,
        }
    )


def test_reuses_every_hvmcpac_structural_metric_and_limit() -> None:
    assert novelty.LIMITS is novelty.hvmcpac.LIMITS
    assert novelty.evaluate_pair is novelty.hvmcpac.evaluate_pair
    candidate = _clock(
        ["2023-07-01T00:05:00Z", "2023-07-08T12:05:00Z"], [1, -1]
    )
    comparator = _clock(
        ["2023-07-03T00:05:00Z", "2023-07-11T12:05:00Z"], [-1, 1]
    )
    result = novelty.evaluate_pair(candidate, comparator)
    assert set(result["checks"]) == set(novelty.LIMITS)
    assert result["passed"] is True
    assert set(novelty.LIMITS) == {
        "exact_entry_jaccard",
        "one_to_one_6h_max_matched_share",
        "occupied_5m_bar_jaccard",
        "absolute_signed_exposure_pearson",
    }
    assert novelty.LIMITS == {
        "exact_entry_jaccard": 0.10,
        "one_to_one_6h_max_matched_share": 0.35,
        "occupied_5m_bar_jaccard": 0.25,
        "absolute_signed_exposure_pearson": 0.35,
    }


def test_frozen_controls_verify_three_eligible_clocks_and_exclude_rejects() -> None:
    registration, support, manifest = novelty.load_frozen_controls()
    assert registration["policy_id"] == novelty.POLICY
    assert tuple(support["eligible_candidates_for_combination_gross9"]) == (
        novelty.ELIGIBLE_CANDIDATES
    )
    assert set(novelty.ELIGIBLE_CANDIDATES).isdisjoint(novelty.REJECTED_CANDIDATES)
    assert set(novelty.ELIGIBLE_CANDIDATES + novelty.REJECTED_CANDIDATES) == set(
        prereg.CANDIDATE_FAMILY
    )
    for candidate in novelty.ELIGIBLE_CANDIDATES:
        assert support["candidates"][candidate]["support_passed"] is True
        assert support["candidates"][candidate]["clock"] == novelty.CLOCKS[candidate]
        assert novelty.sha(novelty.CLOCKS[candidate]["path"]) == novelty.CLOCKS[
            candidate
        ]["sha256"]
    for candidate in novelty.REJECTED_CANDIDATES:
        assert support["candidates"][candidate]["support_passed"] is False
    assert set(manifest["clocks"]) == set(novelty.gross9.EXPECTED_WEIGHTS)


def test_exact_entry_collision_fails_candidate_sleeve() -> None:
    candidate = _clock(["2023-07-01T00:05:00Z"], [1])
    result = novelty.evaluate_pair(candidate, candidate.copy())
    assert result["checks"]["exact_entry_jaccard"] is False
    assert result["passed"] is False


def test_run_is_deterministic_separate_and_economics_blind(tmp_path) -> None:
    output = tmp_path / "gross9.json"
    first = novelty.run(output)
    first_bytes = output.read_bytes()
    second = novelty.run(output)
    assert output.read_bytes() == first_bytes
    assert second == first
    assert tuple(first["candidate_results"]) == novelty.ELIGIBLE_CANDIDATES
    assert tuple(first["source_rejected_candidates"]) == novelty.REJECTED_CANDIDATES
    assert first["eligible_candidates_for_economics"] == list(
        novelty.ELIGIBLE_CANDIDATES
    )
    assert first["all_source_eligible_candidates_passed"] is True
    for candidate, result in first["candidate_results"].items():
        assert tuple(result["gross9_sleeves"]) == tuple(
            novelty.gross9.EXPECTED_WEIGHTS
        )
        assert result["gross9_novelty_status"] == "passed"
        assert all(
            sleeve["metrics"][metric] <= limit
            for sleeve in result["gross9_sleeves"].values()
            for metric, limit in novelty.LIMITS.items()
        )
        assert result["advance_to_economic_outcomes"] == all(
            sleeve["passed"] for sleeve in result["gross9_sleeves"].values()
        )
        assert (candidate in first["eligible_candidates_for_economics"]) == result[
            "advance_to_economic_outcomes"
        ]
    assert all(
        row == {
            "source_support_passed": False,
            "gross9_evaluated": False,
            "advance_to_economic_outcomes": False,
            "decision": "terminal_source_support_reject",
        }
        for row in first["source_rejected_candidates"].values()
    )
    assert first["evidence_boundary"] == {
        "eligible_candidate_clock_rows_opened": sum(
            record["rows"] for record in novelty.CLOCKS.values()
        ),
        "rejected_candidate_clock_rows_opened": 0,
        "gross9_structural_clock_rows_opened": sum(
            sum(counts.values()) for counts in novelty.gross9.EXPECTED_COUNTS.values()
        ),
        "price_or_return_rows_opened": 0,
        "funding_rows_opened": 0,
        "economic_outcome_rows_opened": 0,
        "portfolio_return_or_pnl_metrics_computed": False,
        "outcomes_opened": False,
    }
    assert first["manifest_hash"] == novelty.canonical_hash(
        {key: value for key, value in first.items() if key != "manifest_hash"}
    )
