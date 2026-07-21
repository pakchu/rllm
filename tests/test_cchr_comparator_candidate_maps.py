from __future__ import annotations

from training import cchr_comparator_clock_common as common
from training import export_cchr_dtv_pure_clocks as dtv
from training import export_cchr_far_pure_clocks as far
from training import export_cchr_live_portfolio_pure_clocks as live
from training import export_cchr_pdlh_pure_clocks as pdlh
from training import preregister_cross_collateral_cohort_handoff_relay as prereg


def test_pure_clock_candidate_maps_match_frozen_cchr_preregistration() -> None:
    frozen = prereg.comparator_candidate_map()
    generated = {
        "pdlh": pdlh.pdlh_candidate_map(),
        "dtv": dtv.comparator_candidate_map(),
        "far": far.far_candidate_map(),
        "live": live.candidate_map(),
    }
    expected_counts = {"pdlh": 16, "dtv": 24, "far": 12, "live": 3}

    for family, candidate_map in generated.items():
        expected = {
            candidate_id: definition
            for candidate_id, definition in frozen.items()
            if definition["family"] == family
        }
        assert len(candidate_map) == expected_counts[family]
        assert candidate_map == expected
        assert common.candidate_map_hash(candidate_map) == common.candidate_map_hash(
            expected
        )

    combined = {
        candidate_id: definition
        for candidate_map in generated.values()
        for candidate_id, definition in candidate_map.items()
    }
    frozen_subset = {
        candidate_id: definition
        for candidate_id, definition in frozen.items()
        if definition["family"] in generated
    }
    assert dict(sorted(combined.items())) == frozen_subset
