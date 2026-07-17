from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import build_venue_ticket_migration_shock_support as support
from training import preregister_venue_ticket_migration_shock as prereg


def test_prior_quantile_excludes_current_and_quarantined_values() -> None:
    values = pd.Series([1.0, 2.0, 100.0, 4.0, 5.0])
    clean = pd.Series([True, True, False, True, True])
    result = support.prior_clean_quantile(
        values, clean, quantile=0.5, window=4, min_periods=2
    )
    assert np.isnan(result.iloc[0])
    assert result.iloc[2] == pytest.approx(1.5)
    assert result.iloc[3] == pytest.approx(1.5)
    assert result.iloc[4] == pytest.approx(2.0)


def test_quarantine_carries_forward_exactly_configured_bars() -> None:
    available = pd.Series([True, False, True, True, True, True])
    result = support.quarantine_mask(available, post_gap_bars=2)
    assert result.tolist() == [False, True, True, True, False, False]


def test_policy_mutation_is_rejected_before_source_loading() -> None:
    changed = replace(prereg.Policy(), hold_bars=144)
    with pytest.raises(ValueError, match="policy is frozen"):
        support.run_support(changed)


def test_clock_hash_is_column_order_stable() -> None:
    row = {column: 0 for column in support.CLOCK_COLUMNS}
    row.update(
        {
            "origin_date": "2020-01-01 00:00:00",
            "signal_date": "2020-01-01 00:00:00",
            "entry_date": "2020-01-01 00:10:00",
            "exit_date": "2020-01-02 00:10:00",
            "side": 1,
            "branch": "primary_spot_dominant",
            "delay_bars": 2,
            "hold_bars": 288,
        }
    )
    frame = pd.DataFrame([row])
    assert support._clock_sha256(frame) == support._clock_sha256(
        frame[list(reversed(frame.columns))]
    )


def test_support_loader_contract_has_no_outcome_columns() -> None:
    assert "open" not in support.SPOT_COLUMNS
    assert "high" not in support.SPOT_COLUMNS
    assert "low" not in support.SPOT_COLUMNS
    assert "close" not in support.SPOT_COLUMNS
    assert "open" not in support.PERP_COLUMNS
    assert set(support.POLICY_NAMES) == {
        "primary",
        "direction_flip",
        "no_ticket_level",
        "no_ticket_shock",
        "no_coherence",
        "no_price_acceptance",
        "other_venue_side",
        "one_hour_signal_delay",
        "one_day_shifted_clock",
        "random_side",
    }


def test_frozen_spot_audit_is_passed_with_fail_closed_quarantine() -> None:
    registration = support.verify_preregistration()
    audit_path = registration["source_contract"]["spot_audit"]
    audit = json.loads(Path(audit_path).read_text())
    assert audit["outcomes_opened"] is False
    assert audit["decision"] == "pass_with_fail_closed_quarantine"


def test_frozen_support_artifact_is_outcome_blind_and_passing() -> None:
    artifact = Path("results/venue_ticket_migration_shock_support_2026-07-17.json")
    clock = Path("results/venue_ticket_migration_shock_clock_2026-07-17.csv")
    payload = json.loads(artifact.read_text())
    assert payload["protocol"]["outcomes_opened"] is False
    assert payload["protocol"]["price_or_return_loaded"] is False
    assert payload["source"]["market_columns_loaded"] == ["date"]
    assert payload["source"]["price_or_outcome_columns_loaded"] == []
    assert payload["support_decision"] == "pass"
    assert payload["support"]["train_2020_2022"] == 336
    assert payload["support"]["selection_2023"] == 101
    assert (
        payload["primary_clock_sha256"]
        == hashlib.sha256(clock.read_bytes()).hexdigest()
    )
    assert payload["primary_clock_sha256"] == (
        "7baf6f7de33e66417061dbea6f51efc6ea4993b2b5f2b9e0c09627a68adc57e2"
    )
