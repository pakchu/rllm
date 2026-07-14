from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import preregister_radial_liquidity_wavefront_cascade as rlwc


def _wave_inputs(rows: int = 12) -> dict[str, pd.Series]:
    outer = pd.Series(np.zeros(rows))
    middle = pd.Series(np.zeros(rows))
    inner = pd.Series(np.zeros(rows))
    outer.iloc[2] = 1.50
    middle.iloc[3] = 1.10
    inner.iloc[5] = 1.50
    efficiency = pd.Series(np.full(rows, 0.50))
    valid = pd.Series(np.ones(rows, dtype=bool))
    clean = pd.Series(np.ones(rows, dtype=bool))
    return {
        "outer": outer,
        "middle": middle,
        "inner": inner,
        "outer_efficiency": efficiency.copy(),
        "middle_efficiency": efficiency.copy(),
        "inner_efficiency": efficiency.copy(),
        "outer_raw_valid": valid.copy(),
        "middle_raw_valid": valid.copy(),
        "inner_raw_valid": valid.copy(),
        "clean": clean,
    }


def _detect(inputs: dict[str, pd.Series]) -> pd.Series:
    return rlwc.detect_wavefront(**inputs, cfg=rlwc.Config())


def test_wavefront_requires_ordered_outer_middle_inner_arrival() -> None:
    wave = _detect(_wave_inputs())
    assert np.flatnonzero(wave).tolist() == [5]


@pytest.mark.parametrize(
    ("field", "position", "value"),
    [
        ("inner", 1, 1.00),
        ("outer_raw_valid", 2, False),
        ("middle_raw_valid", 3, False),
        ("inner_raw_valid", 5, False),
        ("middle_efficiency", 3, 0.34),
        ("outer", 5, 1.00),
        ("clean", 1, False),
    ],
)
def test_wavefront_fails_closed_on_invalid_stage(
    field: str,
    position: int,
    value: float | bool,
) -> None:
    inputs = _wave_inputs()
    inputs[field].iloc[position] = value
    assert not _detect(inputs).any()


def test_lagged_normalization_never_revises_earlier_scores() -> None:
    cfg = replace(
        rlwc.Config(),
        robust_baseline_bars=4,
        robust_min_periods=3,
    )
    values = pd.Series([1.0, 2.0, 4.0, 2.0, 1.0, 3.0, 2.0, 5.0])
    clean = pd.Series([True] * len(values))
    original = rlwc._lagged_z(values, clean, cfg)
    changed = values.copy()
    changed.iloc[-1] = 500.0
    replay = rlwc._lagged_z(changed, clean, cfg)
    pd.testing.assert_series_equal(original.iloc[:-1], replay.iloc[:-1])


@pytest.mark.parametrize(("kind", "sign"), [("add", 1.0), ("withdraw", -1.0)])
def test_venue_wave_uses_all_five_shells_with_matching_raw_sign(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    sign: float,
) -> None:
    rows = 12
    frame = pd.DataFrame({"source_complete": [True] * rows})
    for shell in range(1, 6):
        frame[f"um_shell_flow_net_m{shell}"] = np.zeros(rows)
        frame[f"um_shell_flow_efficiency_m{shell}"] = np.full(rows, 0.50)
    frame.loc[2, ["um_shell_flow_net_m4", "um_shell_flow_net_m5"]] = (
        sign * 1.50
    )
    frame.loc[3, "um_shell_flow_net_m3"] = sign * 1.10
    frame.loc[5, ["um_shell_flow_net_m1", "um_shell_flow_net_m2"]] = (
        sign * 1.50
    )
    monkeypatch.setattr(
        rlwc,
        "_lagged_z",
        lambda values, clean, cfg: values.where(clean).astype(float),
    )
    wave = rlwc._wave_for(frame, "um", "m", kind, rlwc.Config())
    assert np.flatnonzero(wave).tolist() == [5]


def test_cross_venue_tolerance_is_current_or_immediately_prior_bar() -> None:
    wave = pd.Series([False, True, False, False])
    assert rlwc._recent(wave, 2).tolist() == [False, True, True, False]
    with pytest.raises(ValueError, match="two-bar"):
        rlwc._recent(wave, 3)


def test_direction_rule_is_an_exact_bid_ask_symmetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=8, freq="5min"),
            "source_complete": True,
        }
    )
    events = {
        ("um", "p", "withdraw"): 2,
        ("cm", "p", "withdraw"): 2,
        ("um", "m", "add"): 2,
        ("um", "m", "withdraw"): 5,
        ("cm", "m", "withdraw"): 5,
        ("cm", "p", "add"): 5,
    }

    def fake_wave(
        frame: pd.DataFrame,
        venue: str,
        side: str,
        kind: str,
        cfg: rlwc.Config,
    ) -> pd.Series:
        del cfg
        result = pd.Series(False, index=frame.index)
        position = events.get((venue, side, kind))
        if position is not None:
            result.iloc[position] = True
        return result

    monkeypatch.setattr(rlwc, "_wave_for", fake_wave)
    signal, _ = rlwc.build_signal(frame, rlwc.Config())
    assert signal.loc[2, "side"] == 1
    assert signal.loc[2, "branch"] == "ask_withdrawal_wave_bid_addition"
    assert signal.loc[5, "side"] == -1
    assert signal.loc[5, "branch"] == "bid_withdrawal_wave_ask_addition"

    frame.loc[2, "source_complete"] = False
    replay, _ = rlwc.build_signal(frame, rlwc.Config())
    assert replay.loc[2, "side"] == 0


def test_rlwc_clock_enters_next_open_and_exits_after_144_bars() -> None:
    rows = 160
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows, freq="5min"),
            "quarantined": False,
        }
    )
    signal = pd.DataFrame(
        {
            "side": np.zeros(rows, dtype=np.int8),
            "hold_bars": np.zeros(rows, dtype=np.int16),
            "branch": "none",
        }
    )
    signal.loc[5, ["side", "hold_bars", "branch"]] = [
        1,
        144,
        "ask_withdrawal_wave_bid_addition",
    ]
    schedule = rlwc.pdf._quarterly_schedule(signal, frame)
    assert len(schedule) == 1
    assert schedule.loc[0, "signal_position"] == 5
    assert schedule.loc[0, "entry_position"] == 6
    assert schedule.loc[0, "exit_position"] == 150


def test_rlwc_v1_parameters_are_not_a_search_grid() -> None:
    cfg = rlwc.Config()
    assert cfg.wave_lookback_bars == 6
    assert cfg.outer_z == 1.25
    assert cfg.middle_z == 1.00
    assert cfg.inner_z == 1.25
    assert cfg.minimum_stage_efficiency == 0.35
    assert cfg.recent_wave_bars == 2
    assert cfg.hold_bars == 144
    assert cfg.minimum_nonoverlap_total == 120
    assert cfg.minimum_nonoverlap_per_half == 45
    assert cfg.minimum_nonoverlap_per_quarter == 20
    assert cfg.minimum_side_share == 0.35
    assert cfg.maximum_quarter_share == 0.40
    assert cfg.maximum_prior_event_jaccard == 0.35
    rlwc._validate_frozen_config(cfg)
    with pytest.raises(ValueError, match="config is frozen"):
        rlwc._validate_frozen_config(replace(cfg, outer_z=1.00))


def test_support_schema_contains_no_price_or_outcome_column() -> None:
    required = rlwc._required_columns()
    assert len(required) == 140
    assert not any(
        token in column.split("_")
        for column in required
        for token in ("open", "high", "low", "close", "return", "pnl")
    )


def test_frozen_rlwc_support_rejects_before_outcomes_open() -> None:
    path = Path(
        "results/radial_liquidity_wavefront_cascade_support_2026-07-14.json"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "59e162056a443ee78e6f51a99da446dd8871e55463fd78018a52cea4196bc1fa"
    )
    result = json.loads(path.read_text())
    assert result["protocol"]["outcomes_opened_for_rlwc"] is False
    assert result["protocol"]["price_or_return_loaded"] is False
    assert result["protocol"]["support_rejected"] is True
    assert result["feature"]["raw_candidate_count"] == 0
    assert result["support"]["nonoverlap_total"] == 0
    assert result["all_support_gates_pass"] is False
    assert result["support_calibration"] == {
        "outcomes_opened_for_rlwc": False,
        "parameters_searched": False,
        "all_parameters_fixed": True,
        "further_support_repairs_allowed": False,
    }
