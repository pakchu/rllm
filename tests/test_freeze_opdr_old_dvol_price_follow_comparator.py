from __future__ import annotations

from dataclasses import asdict

from training import freeze_opdr_old_dvol_price_follow_comparator as freeze


def test_frozen_candidate_is_the_declared_legacy_comparator() -> None:
    assert freeze.EXPECTED_CANDIDATE.name == (
        "dvol_rich_move_follow_v80_p80_h48"
    )
    assert asdict(freeze.EXPECTED_CANDIDATE) == {
        "family": "dvol_rich_move_follow",
        "vol_tail_quantile": 0.8,
        "price_tail_quantile": 0.8,
        "hold_hours": 48,
    }


def test_build_clock_reproduces_frozen_canonical_hash() -> None:
    clock = freeze.build_clock()
    assert len(clock) == freeze.EXPECTED_ROWS
    assert (
        freeze.legacy.canonical_hash(clock.to_dict(orient="records"))
        == freeze.EXPECTED_CANONICAL_CLOCK_HASH
    )
    assert set(clock["side"]) == {-1, 1}


def test_freeze_report_keeps_opdr_outcomes_sealed() -> None:
    report = freeze.run()
    assert report["opdr_outcomes_opened"] is False
    assert report["opdr_outcome_sources_opened"] == []
    assert report["btc_execution_rows_loaded_for_opdr"] == 0
    assert report["funding_rows_loaded_for_opdr"] == 0
    assert report["clock"]["rows"] == freeze.EXPECTED_ROWS
