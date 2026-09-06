from __future__ import annotations

import pandas as pd

from training import (
    evaluate_high_volatility_cross_structure_priority_router_gross9_novelty as novelty,
)
from training import (
    preregister_high_volatility_cross_structure_priority_router as prereg,
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


def test_reuses_every_hvsof_authority_metric_and_limit() -> None:
    assert novelty.gross9 is novelty.hvsof.gross9
    assert novelty.metric is novelty.hvsof.metric
    assert novelty.LIMITS is novelty.hvsof.LIMITS
    assert novelty.evaluate_pair is novelty.hvsof.evaluate_pair
    candidate = _clock(["2023-07-01T00:05:00Z", "2023-07-08T12:05:00Z"], [1, -1])
    comparator = _clock(["2023-07-03T00:05:00Z", "2023-07-11T12:05:00Z"], [-1, 1])
    result = novelty.evaluate_pair(candidate, comparator)
    assert set(result["checks"]) == set(novelty.LIMITS)
    assert result["passed"] is True
    assert novelty.LIMITS == {
        "exact_entry_jaccard": 0.10,
        "one_to_one_6h_max_matched_share": 0.35,
        "occupied_5m_bar_jaccard": 0.25,
        "absolute_signed_exposure_pearson": 0.35,
    }


def test_frozen_controls_verify_prereg_support_both_router_and_gross9_hashes() -> None:
    assert novelty.sha(novelty.PREREG) == novelty.PREREG_SHA
    assert novelty.sha(novelty.SUPPORT) == novelty.SUPPORT_SHA
    registration, support, manifest = novelty.load_frozen_controls()
    assert registration["policy_id"] == novelty.POLICY
    assert tuple(registration["candidate_family"]) == prereg.CANDIDATE_FAMILY
    assert tuple(support["eligible_routers_for_combination_gross9"]) == (
        novelty.ELIGIBLE_ROUTERS
    )
    assert len(novelty.ELIGIBLE_ROUTERS) == 2
    assert all(
        router.endswith("__ELIGIBLE_BY__HVTCCR-8")
        for router in novelty.ELIGIBLE_ROUTERS
    )
    for router in novelty.ELIGIBLE_ROUTERS:
        assert support["candidates"][router]["support_passed"] is True
        assert support["candidates"][router]["clock"] == novelty.CLOCKS[router]
        assert (
            novelty.sha(novelty.CLOCKS[router]["path"])
            == novelty.CLOCKS[router]["sha256"]
        )
    assert set(manifest["clocks"]) == set(novelty.gross9.EXPECTED_WEIGHTS)
    for sleeve, record in manifest["clocks"].items():
        assert novelty.sha(record["path"]) == record["sha256"], sleeve


def test_exact_entry_collision_fails_router_sleeve() -> None:
    router = _clock(["2023-07-01T00:05:00Z"], [1])
    result = novelty.evaluate_pair(router, router.copy())
    assert result["checks"]["exact_entry_jaccard"] is False
    assert result["passed"] is False


def test_run_twice_is_deterministic_separate_and_economics_blind(tmp_path) -> None:
    output = tmp_path / "gross9.json"
    first = novelty.run(output)
    first_bytes = output.read_bytes()
    second = novelty.run(output)
    assert output.read_bytes() == first_bytes
    assert second == first
    assert tuple(first["router_results"]) == novelty.ELIGIBLE_ROUTERS
    assert first["eligible_routers_for_economics"] == list(novelty.ELIGIBLE_ROUTERS)
    assert first["all_source_supported_routers_passed"] is True
    for router, result in first["router_results"].items():
        assert tuple(result["gross9_sleeves"]) == tuple(novelty.gross9.EXPECTED_WEIGHTS)
        assert result["gross9_novelty_status"] == "passed"
        assert all(
            sleeve["metrics"][metric] <= limit
            for sleeve in result["gross9_sleeves"].values()
            for metric, limit in novelty.LIMITS.items()
        )
        assert result["advance_to_economic_outcomes"] == all(
            sleeve["passed"] for sleeve in result["gross9_sleeves"].values()
        )
        assert (router in first["eligible_routers_for_economics"]) == result[
            "advance_to_economic_outcomes"
        ]
    assert first["evidence_boundary"] == {
        "eligible_router_clock_rows_opened": sum(
            record["rows"] for record in novelty.CLOCKS.values()
        ),
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
