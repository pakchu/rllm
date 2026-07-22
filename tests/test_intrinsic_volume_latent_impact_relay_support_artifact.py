from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from training import build_intrinsic_volume_latent_impact_relay_support as s


REPORT = Path("results/intrinsic_volume_latent_impact_relay_support_2026-07-23.json")
CLOCK = Path("data/intrinsic_volume_latent_impact_relay_clocks_2020_2023.csv.gz")


def test_support_artifact_is_terminal_outcome_blind_rejection() -> None:
    payload = json.loads(REPORT.read_text())
    assert payload["policy_id"] == "IVLIR-72"
    assert payload["support_passed"] is False
    assert payload["authorized_next_stage"] is None
    assert payload["failed_checks"] == ["maximum_same_side_run"]
    assert payload["outcomes_opened"] is False
    assert payload["post_entry_return_computed"] is False
    assert payload["funding_loaded"] is False
    assert payload["leakage_guard"]["no_post_entry_price_access"] is True
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"report_manifest_hash", "created_at"}
    }
    assert payload["report_manifest_hash"] == s.canonical_hash(core)


def test_primary_support_counts_and_failed_run_are_frozen() -> None:
    payload = json.loads(REPORT.read_text())
    assert payload["windows"]["train_2020_2022"]["events"] == 180
    assert payload["windows"]["selection_2023"]["events"] == 53
    assert payload["windows"]["selection_2023_h1"]["events"] == 23
    assert payload["windows"]["selection_2023_h2"]["events"] == 30
    assert payload["windows"]["all"]["long"] == 82
    assert payload["windows"]["all"]["short"] == 151
    assert payload["windows"]["all"]["maximum_same_side_run"] == 26
    assert payload["support_checks"]["maximum_same_side_run"] is False
    assert all(
        passed
        for name, passed in payload["support_checks"].items()
        if name != "maximum_same_side_run"
    )


def test_clock_is_hash_bound_and_contains_no_outcome_values() -> None:
    payload = json.loads(REPORT.read_text())
    assert payload["clock"]["sha256"] == s.sha256_file(CLOCK)
    assert payload["clock"]["rows"] == 2_038
    clock = pd.read_csv(CLOCK)
    assert list(clock.columns) == s.CLOCK_COLUMNS
    assert set(clock["side"]) == {"LONG", "SHORT"}
    assert not any(
        token in column.lower()
        for column in clock.columns
        for token in ("open", "high", "low", "close", "return", "pnl", "mdd")
    )
