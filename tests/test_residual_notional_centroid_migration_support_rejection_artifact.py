from __future__ import annotations

import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/residual_notional_centroid_migration_support_2026-07-20.json"
)
SOURCE = Path(
    "training/preregister_residual_notional_centroid_migration.py"
)
DOCUMENT = Path(
    "docs/residual-notional-centroid-migration-preregistration-2026-07-20.md"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rncm_support_rejection_is_outcome_blind_and_hash_bound() -> None:
    assert _sha256(RESULT) == (
        "887c532eb3163cfac47eb9fc2956326f02491b2890e4c0231e084807978577dc"
    )
    assert _sha256(SOURCE) == (
        "733ef4c3aaa823f19c8fe9303d3405def0c86f593c35bb2556a69edc3f67ad6f"
    )
    assert _sha256(DOCUMENT) == (
        "fb2ed44cb0eb561c1d436c02c73d4028680b7ba292f67d8bce86ddf3ed23a11f"
    )
    result = json.loads(RESULT.read_text())
    protocol = result["protocol"]
    assert protocol["outcomes_opened"] is False
    assert protocol["external_ohlc_funding_return_or_equity_loaded"] is False
    assert protocol["selection_end_exclusive"] == "2024-01-01 00:00:00"
    assert result["selected_quantile"] is None
    assert result["all_support_gates_pass"] is False
    assert result["rejection_reason"] == "no frozen quantile passed incidence"
    assert not Path(
        "results/residual_notional_centroid_migration_event_clock_2026-07-20.json"
    ).exists()


def test_rncm_synthetic_nulls_and_incidence_match_the_frozen_result() -> None:
    result = json.loads(RESULT.read_text())
    synthetic = result["synthetic_control"]
    assert synthetic["passes"] is True
    assert set(synthetic["scenarios"]) == {
        "smooth_symmetric",
        "tick_rounded_anchor",
        "stepped_asymmetric",
        "missing_rows",
        "discrete_asymmetric_ladder",
    }
    for scenario in synthetic["scenarios"].values():
        for trial in scenario.values():
            assert trial == {"raw_events": 0, "nonoverlap_events": 0}

    observed = [
        (
            trial["quantile"],
            trial["raw_event_count"],
            trial["support"]["nonoverlap_total"],
            trial["support"]["by_quarter"],
        )
        for trial in result["threshold_trials"]
    ]
    assert observed == [
        (0.995, 5, 5, {"q1": 1, "q2": 2, "q3": 0, "q4": 2}),
        (0.99, 16, 16, {"q1": 4, "q2": 3, "q3": 4, "q4": 5}),
        (0.985, 32, 31, {"q1": 8, "q2": 6, "q3": 6, "q4": 11}),
        (0.975, 45, 39, {"q1": 10, "q2": 7, "q3": 8, "q4": 14}),
    ]
    assert all(
        trial["support"]["passes_incidence"] is False
        for trial in result["threshold_trials"]
    )
