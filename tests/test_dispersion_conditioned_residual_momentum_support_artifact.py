from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training.preregister_dispersion_conditioned_residual_momentum import canonical_hash


RESULT = Path("results/dispersion_conditioned_residual_momentum_support_2026-07-17.json")
CLOCK = Path("data/dispersion_conditioned_residual_momentum_support_clock_2023_2024.csv.gz")
EXPECTED_RESULT_SHA256 = "7093fc93a05bc509c55a0480033fe6fd82262d6207d6992e6c14486cc90dfd37"
EXPECTED_CLOCK_SHA256 = "d7cb7b5066692b8dccc6dbc2051d01c9522acc1e0769e63b7a6135bbffeae992"


def test_support_artifacts_are_locked_before_outcomes() -> None:
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED_RESULT_SHA256
    assert hashlib.sha256(CLOCK.read_bytes()).hexdigest() == EXPECTED_CLOCK_SHA256
    payload = json.loads(RESULT.read_text())
    body = {
        key: value
        for key, value in payload.items()
        if key not in {"manifest_hash", "created_at"}
    }
    assert canonical_hash(body) == payload["manifest_hash"]
    assert payload["post_entry_returns_or_pnl_calculated"] is False
    assert payload["2023_selection_outcomes_opened"] is False
    assert payload["2024_test_outcomes_opened"] is False
    assert payload["support"]["passes_support"] is True


def test_support_is_broad_and_reports_reduced_gross_separately() -> None:
    payload = json.loads(RESULT.read_text())
    support = payload["support"]
    assert support["events"] == 92
    assert support["year_counts"] == {"2023": 39, "2024": 53}
    assert support["half_counts"] == {
        "2023H1": 13,
        "2023H2": 26,
        "2024H1": 26,
        "2024H2": 27,
    }
    assert support["gross_scale_counts"] == {"0.25": 29, "1.0": 63}
    assert support["unique_ordered_pairs"] == 25
    assert support["maximum_ordered_pair_share"] <= 0.20
    assert support["maximum_month_share"] <= 0.15
    assert len(support["long_symbols"]) == 6
    assert len(support["short_symbols"]) == 6


def test_clock_overlap_is_outcome_blind_and_not_overclaimed() -> None:
    payload = json.loads(RESULT.read_text())
    overlap = payload["outcome_blind_clock_overlap"]["LORE_2023_2024"]
    assert overlap["post_entry_returns_or_pnl_read"] is False
    assert overlap["exact_entry_jaccard"] < 0.02
    assert overlap["position_time_jaccard_5m"] > 0.20
