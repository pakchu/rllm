from __future__ import annotations

import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/residual_quote_curvature_impulse_support_2026-07-20.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rqci_rejection_is_outcome_blind_and_hash_bound() -> None:
    assert _sha256(RESULT) == (
        "aef0f375245cd340a29da121551dbf2ebf80ef3e08a4123ad7ef9c5e0c414f68"
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
        "results/residual_quote_curvature_impulse_event_clock_2026-07-20.json"
    ).exists()
    artifacts = result["frozen_artifacts"]
    for key in (
        "preregistration_source",
        "preregistration_document",
        "source_decision_document",
        "shared_causal_utility_source",
    ):
        assert _sha256(Path(artifacts[key])) == artifacts[f"{key}_sha256"]


def test_rqci_nulls_and_frozen_incidence_match() -> None:
    result = json.loads(RESULT.read_text())
    synthetic = result["synthetic_control"]
    assert synthetic["passes"] is True
    for scenario in synthetic["scenarios"].values():
        for trial in scenario.values():
            assert trial == {"raw_events": 0, "nonoverlap_events": 0}

    observed = [
        (
            trial["quantile"],
            trial["raw_event_count"],
            trial["support"]["nonoverlap_total"],
            trial["support"]["by_quarter"],
            trial["support"]["h1"],
            trial["support"]["h2"],
        )
        for trial in result["threshold_trials"]
    ]
    assert observed == [
        (0.995, 3, 3, {"q1": 1, "q2": 0, "q3": 0, "q4": 2}, 1, 2),
        (0.99, 10, 8, {"q1": 1, "q2": 1, "q3": 1, "q4": 5}, 2, 6),
        (0.985, 29, 23, {"q1": 3, "q2": 4, "q3": 5, "q4": 11}, 7, 16),
        (0.975, 66, 50, {"q1": 8, "q2": 8, "q3": 17, "q4": 17}, 16, 34),
        (0.95, 192, 148, {"q1": 14, "q2": 36, "q3": 45, "q4": 53}, 50, 98),
    ]
    assert all(
        trial["support"]["passes_incidence"] is False
        for trial in result["threshold_trials"]
    )
