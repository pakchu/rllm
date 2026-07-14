from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from training import preregister_cross_collateral_liquidity_hysteresis as cclh


def _cfg() -> cclh.Config:
    return replace(
        cclh.Config(),
        robust_baseline_bars=4,
        robust_min_periods=2,
        confirmation_bars=3,
        exit_confirmation_bars=3,
        hold_bars=2,
        minimum_nonoverlap_total=1,
        minimum_nonoverlap_per_half=0,
        minimum_nonoverlap_per_quarter=0,
        minimum_side_share=0.0,
        maximum_quarter_share=1.0,
    )


def test_full_depth_geometry_orients_bullish_and_bearish_states() -> None:
    frame = pd.DataFrame(index=range(2))
    for distance in range(1, 6):
        frame[f"um_depth_m{distance}"] = float(distance)
        frame[f"um_depth_p{distance}"] = float(distance)
        frame.loc[0, f"cm_depth_m{distance}"] = 10.0 * distance
        frame.loc[0, f"cm_depth_p{distance}"] = float(distance**2)
        frame.loc[1, f"cm_depth_m{distance}"] = float(distance**2)
        frame.loc[1, f"cm_depth_p{distance}"] = 10.0 * distance
    geometry = cclh.cross_collateral_geometry(frame)
    assert geometry.loc[0, "cross_pressure"] > 0.0
    assert geometry.loc[0, "cross_elasticity"] > 0.0
    assert geometry.loc[1, "cross_pressure"] < 0.0
    assert geometry.loc[1, "cross_elasticity"] < 0.0


def test_hysteresis_emits_only_confirmed_entry_and_flip() -> None:
    pressure = pd.Series([1.0] * 8 + [0.0] * 3 + [-1.0] * 4)
    elasticity = pressure.copy()
    state = cclh.hysteresis_state_machine(
        pressure,
        elasticity,
        pd.Series([True] * len(pressure)),
        _cfg(),
    )
    assert state.index[state["event_side"].ne(0)].tolist() == [2, 13]
    assert state.loc[2, "event_side"] == 1
    assert state.loc[7, "active_state"] == 1
    assert state.loc[10, "active_state"] == 0
    assert state.loc[13, "event_side"] == -1


def test_missing_source_requires_confirmed_exit_without_future_cancellation() -> None:
    pressure = pd.Series([1.0] * 4 + [np.nan] * 3)
    elasticity = pressure.copy()
    clean = pd.Series([True] * 4 + [False] * 3)
    state = cclh.hysteresis_state_machine(
        pressure,
        elasticity,
        clean,
        _cfg(),
    )
    assert state.loc[2, "event_side"] == 1
    assert state.loc[4, "active_state"] == 1
    assert state.loc[6, "active_state"] == 0
    assert state.loc[4:, "event_side"].eq(0).all()


def test_state_machine_is_prefix_invariant() -> None:
    pressure = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
    clean = pd.Series([True] * 5)
    baseline = cclh.hysteresis_state_machine(
        pressure,
        pressure,
        clean,
        _cfg(),
    )
    changed = pressure.copy()
    changed.loc[4] = -1_000_000.0
    replay = cclh.hysteresis_state_machine(
        changed,
        changed,
        clean,
        _cfg(),
    )
    pd.testing.assert_frame_equal(baseline.loc[:3], replay.loc[:3])


def test_tolerant_event_jaccard_uses_one_to_one_matches() -> None:
    result = cclh.tolerant_event_jaccard(
        [10, 20, 100],
        [11, 12, 24, 200],
        tolerance_bars=5,
    )
    assert result == {
        "first_event_count": 3,
        "second_event_count": 4,
        "matched_event_count": 2,
        "tolerance_bars": 5,
        "jaccard": 0.4,
    }


def test_frozen_cclh_support_passes_without_opening_outcomes() -> None:
    result = json.loads(
        Path(
            "results/cross_collateral_liquidity_hysteresis_support_2026-07-14.json"
        ).read_text()
    )
    assert result["protocol"]["outcomes_opened_for_cclh"] is False
    assert result["protocol"]["support_parameters_searched"] is False
    assert result["protocol"]["support_rejected"] is False
    assert result["all_support_gates_pass"] is True
    assert result["support"]["nonoverlap_total"] == 167
    assert result["support"]["by_quarter"] == {
        "q1": 33,
        "q2": 38,
        "q3": 48,
        "q4": 48,
    }
    assert result["support"]["h1"] == 71
    assert result["support"]["h2"] == 96
    assert result["scheduled_side_counts"] == {"long": 88, "short": 79}
    assert result["clv_event_overlap"]["matched_event_count"] == 34
    assert result["clv_event_overlap"]["passes_independence"] is True
    for key in (
        "preregistration_source",
        "preregistration_document",
        "feature_source",
        "clv_source",
        "clv_support",
        "scheduler_source",
    ):
        path = Path(result["frozen_artifacts"][key])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == result["frozen_artifacts"][f"{key}_sha256"]
