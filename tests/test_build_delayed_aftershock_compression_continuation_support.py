from __future__ import annotations

import hashlib
import json
import numpy as np
import pandas as pd

from training import build_delayed_aftershock_compression_continuation_support as support


def test_prior_quantile_excludes_current_observation() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 100.0])
    result = support.prior_quantile(
        values,
        quantile=1.0,
        window=3,
        min_periods=1,
    )
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == 1.0
    assert result.iloc[3] == 3.0


def test_nonoverlap_keeps_first_event_after_previous_exit() -> None:
    rows = [
        {"entry_position": 10, "exit_position": 20, "anchor_position": 1},
        {"entry_position": 20, "exit_position": 30, "anchor_position": 2},
        {"entry_position": 21, "exit_position": 31, "anchor_position": 3},
    ]
    selected = support._nonoverlap(rows)
    assert [row["entry_position"] for row in selected] == [10, 21]


def test_clock_contract_forbids_same_bar_entry_and_outcome_columns() -> None:
    dates = pd.date_range("2023-01-01", periods=60, freq="5min")
    row = {
        "policy": "primary",
        "anchor_position": 0,
        "signal_position": 9,
        "entry_position": 10,
        "exit_position": 58,
        "anchor_date": dates[0],
        "signal_date": dates[9],
        "entry_date": dates[10],
        "exit_date": dates[58],
        "side": 1,
        "hold_bars": 48,
    }
    frame = pd.DataFrame([row], columns=support.CLOCK_COLUMNS)
    support.assert_clock_contract(frame)
    assert set(frame.columns).isdisjoint({"open", "close", "return", "pnl", "funding"})


def test_orthogonality_metrics_detect_exact_and_near_entries() -> None:
    clock = pd.DataFrame(
        [
            {
                "policy": "primary",
                "anchor_position": 0,
                "signal_position": 9,
                "entry_position": 10,
                "exit_position": 58,
                "anchor_date": "2023-01-01",
                "signal_date": "2023-01-01 00:45",
                "entry_date": "2023-01-01 00:50",
                "exit_date": "2023-01-01 04:50",
                "side": 1,
                "hold_bars": 48,
            },
            {
                "policy": "primary",
                "anchor_position": 90,
                "signal_position": 99,
                "entry_position": 100,
                "exit_position": 148,
                "anchor_date": "2023-01-01 07:30",
                "signal_date": "2023-01-01 08:15",
                "entry_date": "2023-01-01 08:20",
                "exit_date": "2023-01-01 12:20",
                "side": -1,
                "hold_bars": 48,
            },
        ],
        columns=support.CLOCK_COLUMNS,
    )
    baselines = {
        "baseline": {
            "entry_positions": np.array([10, 170]),
            "hold_bars": 48,
        }
    }
    result = support.orthogonality_metrics(clock, baselines, length=220)
    assert result["maximum_exact_entry_jaccard"] == 1.0 / 3.0
    assert result["candidate_entries_within_six_hours_of_any_baseline_share"] == 1.0


def test_default_preregistration_identity_is_frozen() -> None:
    payload = support.verify_preregistration()
    assert payload["manifest_hash"] == support.PREREGISTRATION_MANIFEST_HASH
    assert payload["outcomes_opened"] is False


def test_frozen_support_rejects_before_any_outcome() -> None:
    payload = json.loads(open(support.DEFAULT_OUTPUT).read())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert support.prereg.canonical_hash(core) == payload["manifest_hash"]
    assert payload["outcomes_opened"] is False
    assert payload["post_entry_returns_or_pnl_calculated"] is False
    assert payload["funding_loaded"] is False
    assert payload["support"]["passes_support"] is False
    assert payload["support"]["checks"]["exact_baseline_clock_binding"] is False
    assert payload["orthogonality"]["admission_eligible"] is False
    assert payload["support"]["primary_total"] == 1
    assert payload["support"]["counts"]["2023"] == 0
    clock_bytes = open(payload["clock"]["path"], "rb").read()
    assert hashlib.sha256(clock_bytes).hexdigest() == payload["clock"]["sha256"]
    clock = pd.read_csv(payload["clock"]["path"])
    assert set(clock.columns) == set(support.CLOCK_COLUMNS)
    assert set(clock.columns).isdisjoint(
        {"open", "high", "low", "close", "return", "pnl", "funding"}
    )
    assert payload["sealed_windows"] == [
        "all_post_entry_outcomes",
        "2024",
        "2025",
        "2026_ytd",
    ]
