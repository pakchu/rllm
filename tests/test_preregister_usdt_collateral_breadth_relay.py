from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from training import preregister_usdt_collateral_breadth_relay as ucbr


def _source_frame(rows: int = 674) -> pd.DataFrame:
    dates = pd.date_range("2023-08-01", periods=rows, freq="1h", tz="UTC")
    phase = np.arange(rows) % 4
    base = np.choose(phase, [-0.002, -0.001, 0.001, 0.002]).astype(float)
    data: dict[str, object] = {
        "date": dates,
        "source_available_at": dates + pd.Timedelta("1h"),
    }
    for index, (log_column, valid_column) in enumerate(
        zip(ucbr.LOG_COLUMNS, ucbr.VALID_COLUMNS, strict=True)
    ):
        data[log_column] = base * (1.0 + index * 0.05)
        data[valid_column] = True
    data["valid_breadth"] = 4
    data["source_complete"] = True
    return pd.DataFrame(data)


def test_robust_z_excludes_current_and_masks_invalid_member() -> None:
    values = pd.Series(np.tile([-2.0, -1.0, 1.0, 2.0], 169)[:673])
    valid = pd.Series(True, index=values.index)
    baseline = ucbr.prior_robust_z(values, valid, ucbr.FROZEN_CONFIG)
    changed = values.copy()
    changed.iloc[-1] = 100.0
    shocked = ucbr.prior_robust_z(changed, valid, ucbr.FROZEN_CONFIG)
    assert shocked.iloc[-1]["median"] == baseline.iloc[-1]["median"]
    assert shocked.iloc[-1]["scale"] == baseline.iloc[-1]["scale"]
    assert shocked.iloc[-1]["z"] != baseline.iloc[-1]["z"]
    invalid = valid.copy()
    invalid.iloc[-1] = False
    assert np.isnan(ucbr.prior_robust_z(changed, invalid, ucbr.FROZEN_CONFIG).iloc[-1]["z"])


def test_primary_requires_three_strong_issuers_and_maps_usdt_weakness_short() -> None:
    frame = _source_frame()
    for column in ucbr.LOG_COLUMNS[:3]:
        frame.loc[672, column] = 0.02
    frame.loc[672, ucbr.LOG_COLUMNS[3]] = -0.02
    for column in ucbr.LOG_COLUMNS:
        frame.loc[673, column] = 0.0
    states = ucbr.signal_states(frame)
    assert bool(states.loc[672, "primary_active"])
    assert states.loc[672, "primary_source_sign"] == 1
    assert states.loc[672, "primary_side"] == -1
    assert states.loc[672, "primary_agreeing_breadth"] == 3
    assert not bool(states.loc[672, "all_four_active"])
    assert bool(states.loc[672, "leave_out_fdusd_active"])
    assert bool(states.loc[673, "stale_1h_active"])
    assert states.loc[673, "stale_1h_side"] == -1

    weakened = frame.copy()
    weakened.loc[672, ucbr.VALID_COLUMNS[0]] = False
    weakened.loc[672, "valid_breadth"] = 3
    assert not bool(ucbr.signal_states(weakened).loc[672, "primary_active"])


def test_median_control_requires_three_finite_zscores_not_only_current_books() -> None:
    frame = _source_frame()
    for column in ucbr.LOG_COLUMNS:
        frame.loc[672, column] = 0.02
    frame.loc[:671, ucbr.VALID_COLUMNS[2:]] = False
    states = ucbr.signal_states(frame)
    assert states.loc[672, "valid_breadth"] == 4
    assert not bool(states.loc[672, "median_only_active"])


def test_schedule_delays_entry_and_preserves_stale_feature_provenance() -> None:
    dates = pd.date_range("2023-09-01", periods=3, freq="1h", tz="UTC")
    frame = pd.DataFrame(
        {
            "date": dates,
            "source_available_at": dates + pd.Timedelta("1h"),
        }
    )
    states = pd.DataFrame(
        {
            "z_usdcusdt": [2.0, 0.0, 0.0],
            "z_tusdusdt": [2.0, 0.0, 0.0],
            "z_usdpusdt": [2.0, 0.0, 0.0],
            "z_fdusdusdt": [0.0, 0.0, 0.0],
            "valid_breadth": [4, 4, 4],
            "median_z": [2.0, 0.0, 0.0],
            "primary_active": [True, False, False],
            "primary_source_sign": [1, 0, 0],
            "primary_side": [-1, 0, 0],
            "primary_agreeing_breadth": [3, 0, 0],
            "primary_consensus_strength": [2.0, np.nan, np.nan],
            "stale_1h_active": [False, True, False],
            "stale_1h_source_sign": [0, 1, 0],
            "stale_1h_side": [0, -1, 0],
            "stale_1h_agreeing_breadth": [0, 3, 0],
            "stale_1h_consensus_strength": [np.nan, 2.0, np.nan],
        }
    )
    primary = ucbr.schedule(frame, states, control="primary")
    stale = ucbr.schedule(frame, states, control="stale_1h")
    assert primary.iloc[0]["entry_time"] == dates[0] + pd.Timedelta("65min")
    assert primary.iloc[0]["exit_time"] == dates[0] + pd.Timedelta("785min")
    assert stale.iloc[0]["source_hour_start"] == dates[0]
    assert stale.iloc[0]["feature_available_time"] == dates[0] + pd.Timedelta("1h")
    assert stale.iloc[0]["decision_time"] == dates[1] + pd.Timedelta("1h")
    assert stale.iloc[0]["entry_time"] == primary.iloc[0]["entry_time"] + pd.Timedelta("1h")


def test_frozen_config_rejects_threshold_repair() -> None:
    with pytest.raises(ValueError, match="configuration is frozen"):
        ucbr.signal_states(
            _source_frame(), replace(ucbr.FROZEN_CONFIG, z_threshold=1.0)
        )


def test_real_source_loader_is_hash_bound_and_outcome_blind() -> None:
    frame, audit = ucbr.load_source()
    assert tuple(frame.columns) == ucbr.SOURCE_COLUMNS
    assert len(frame) == audit["rows"] == 3_672
    assert audit["minimum_valid_breadth"] == 3
    assert frame["source_complete"].all()


def test_preregistration_has_no_incidence_or_realized_outcome(tmp_path) -> None:
    payload = ucbr.preregistration_payload(tmp_path / "freeze.json")
    assert payload["outcomes_opened"] is False
    assert payload["real_event_incidence_opened"] is False
    assert payload["post_2023_source_rows_opened"] is False
    serialized = str(payload).lower()
    for forbidden in (
        "observed_trade_count",
        "realized_absolute_return",
        "realized_cagr",
        "observed_event_count",
    ):
        assert forbidden not in serialized
