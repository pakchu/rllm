from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training import preregister_coinm_roll_migration_alpha as roll


def test_robust_z_is_prefix_independent_and_resets_by_pair() -> None:
    values = pd.Series(np.linspace(1.0, 100.0, 100))
    pairs = pd.Series(["a"] * 60 + ["b"] * 40)
    left = roll.causal_robust_z(values, pairs, window=20, min_periods=10)
    changed = values.copy()
    changed.iloc[80:] *= 1000.0
    right = roll.causal_robust_z(changed, pairs, window=20, min_periods=10)
    assert np.allclose(left.iloc[:80], right.iloc[:80], equal_nan=True)
    assert left.iloc[60:70].isna().all()
    assert np.isfinite(left.iloc[70])


def test_feature_valid_parser_does_not_treat_false_strings_as_true() -> None:
    parsed = roll.parse_feature_valid(pd.Series(["True", "False", "1", "0"]))
    assert parsed.tolist() == [True, False, True, False]
    with pytest.raises(ValueError, match="invalid tokens"):
        roll.parse_feature_valid(pd.Series(["yes"]))


def test_source_seal_rejects_hash_mismatch_before_loading(tmp_path) -> None:
    source = tmp_path / "source.csv.gz"
    manifest = tmp_path / "manifest.json"
    source.write_bytes(b"mutated")
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="source SHA mismatch"):
        roll.verify_source_seal(source, manifest)


def _signal_frame(rows: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=rows, freq="5min")
    return pd.DataFrame(
        {
            "signal_bar_open_utc": dates,
            "feature_available_time_utc": dates + pd.Timedelta("5min"),
            "trade_earliest_time_utc": dates + pd.Timedelta("5min"),
            "front_symbol": ["BTCUSD_230331"] * rows,
            "next_symbol": ["BTCUSD_230630"] * rows,
            "front_open": np.full(rows, 100.0),
            "front_close": np.full(rows, 100.0),
            "front_volume": np.full(rows, 90.0),
            "front_taker_buy_volume": np.full(rows, 45.0),
            "next_open": np.full(rows, 100.0),
            "next_close": np.full(rows, 100.0),
            "next_volume": np.full(rows, 10.0),
            "next_taker_buy_volume": np.full(rows, 5.0),
            "feature_valid": np.ones(rows, dtype=bool),
            "feature_invalid_reason": ["ok"] * rows,
            "front_hours_to_delivery": np.full(rows, 30 * 24.0),
            "next_hours_to_delivery": np.full(rows, 120 * 24.0),
        }
    )


def test_signal_state_uses_raw_contract_share_not_log_share(monkeypatch) -> None:
    frame = _signal_frame()
    monkeypatch.setattr(roll, "ROBUST_WINDOW_BARS", 20)
    monkeypatch.setattr(roll, "ROBUST_MIN_PERIODS", 10)
    state = roll.build_signal_state(frame)
    assert np.allclose(state["front_share"], 0.9)
    assert np.allclose(state["next_share"], 0.1)


def test_next_led_clock_uses_completed_bar_acceptance_and_fixed_side() -> None:
    frame = _signal_frame(3)
    state = pd.DataFrame(
        {
            "source_valid": [True, True, True],
            "total_volume": [100.0, 100.0, 100.0],
            "prior_q25_total_volume": [50.0, 50.0, 50.0],
            "front_share": [0.7] * 3,
            "next_share": [0.3] * 3,
            "z_front_share": [0.0] * 3,
            "z_next_share": [1.2, 1.2, 1.2],
            "z_abs_front_pressure": [0.0] * 3,
            "z_abs_next_pressure": [2.2, 2.2, 2.2],
            "front_direction": [1, 1, 1],
            "next_direction": [1, -1, 1],
            "front_bar_return": [-0.0001, 0.00005, -0.0002],
            "next_bar_return": [0.0005, -0.0006, 0.0004],
        }
    )
    active, side = roll.candidate_clock(frame, state, roll.CANDIDATES[0])
    assert active.tolist() == [True, True, False]
    assert side.tolist() == [1, -1, 0]


def test_front_rejection_fades_pressure_and_requires_quiet_next() -> None:
    frame = _signal_frame(3)
    state = pd.DataFrame(
        {
            "source_valid": [True, True, True],
            "total_volume": [100.0, 100.0, 100.0],
            "prior_q25_total_volume": [50.0, 50.0, 50.0],
            "front_share": [0.8] * 3,
            "next_share": [0.2] * 3,
            "z_front_share": [0.9] * 3,
            "z_next_share": [0.0] * 3,
            "z_abs_front_pressure": [1.5] * 3,
            "z_abs_next_pressure": [0.7, 0.8, 0.7],
            "front_direction": [1, 1, -1],
            "next_direction": [1, 1, -1],
            "front_bar_return": [-0.0003, -0.0003, 0.0003],
            "next_bar_return": [-0.0001, -0.0001, 0.0001],
        }
    )
    active, side = roll.candidate_clock(frame, state, roll.CANDIDATES[1])
    assert active.tolist() == [True, False, True]
    assert side.tolist() == [-1, 0, 1]


def test_delivery_buffer_blocks_signal_whose_exit_is_too_close() -> None:
    frame = _signal_frame(1)
    frame.loc[0, "front_hours_to_delivery"] = 12.5
    state = pd.DataFrame(
        {
            "source_valid": [True],
            "total_volume": [100.0],
            "prior_q25_total_volume": [50.0],
            "front_share": [0.7],
            "next_share": [0.3],
            "z_front_share": [0.0],
            "z_next_share": [2.0],
            "z_abs_front_pressure": [0.0],
            "z_abs_next_pressure": [3.0],
            "front_direction": [1],
            "next_direction": [1],
            "front_bar_return": [0.0],
            "next_bar_return": [0.001],
        }
    )
    active, _ = roll.candidate_clock(frame, state, roll.CANDIDATES[0])
    assert not active[0]


def test_schedule_enters_after_completed_bar_and_never_overlaps() -> None:
    frame = _signal_frame(30)
    active = np.zeros(len(frame), dtype=bool)
    side = np.zeros(len(frame), dtype=np.int8)
    active[[0, 6, 13]] = True
    side[[0, 6, 13]] = [1, -1, -1]
    schedule = roll.nonoverlapping_schedule(
        frame,
        active,
        side,
        roll.CANDIDATES[0],
        start=pd.Timestamp("2023-01-01"),
        end=pd.Timestamp("2023-01-02"),
    )
    assert len(schedule) == 2
    assert pd.Timestamp(schedule.iloc[0]["entry_time"]) == pd.Timestamp(
        schedule.iloc[0]["signal_bar_open"]
    ) + pd.Timedelta("5min")
    assert pd.Timestamp(schedule.iloc[1]["entry_time"]) >= pd.Timestamp(
        schedule.iloc[0]["exit_time"]
    )
    assert schedule["traded_leg"].eq("next").all()


def test_support_gate_requires_both_sides_and_time_dispersion() -> None:
    summary = {}
    for name in roll.SUPPORT_WINDOWS:
        summary[name] = {
            "total": 500,
            "longs": 250,
            "shorts": 250,
            "max_month_fraction": 0.10,
            "max_symbol_fraction": 0.10,
        }
    assert all(roll.support_gates(summary).values())
    summary["select_2023"] = {**summary["select_2023"], "longs": 500, "shorts": 0}
    assert not all(roll.support_gates(summary).values())


def test_exclusive_writer_refuses_to_replace_freeze(tmp_path) -> None:
    output = tmp_path / "freeze.json"
    roll.write_exclusive(output, {"value": 1})
    with pytest.raises(FileExistsError):
        roll.write_exclusive(output, {"value": 2})


def test_implementation_path_is_not_cwd_sensitive(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert roll.implementation_path() == "training/preregister_coinm_roll_migration_alpha.py"


def test_real_source_is_physically_sealed_before_2024() -> None:
    seal = roll.verify_source_seal(roll.Config.input_csv, roll.Config.manifest_json)
    assert seal["output_sha256"] == roll.EXPECTED_SOURCE_SHA256
    source = roll.load_source(roll.Config.input_csv)
    assert source["signal_bar_open_utc"].min() == pd.Timestamp("2020-07-01")
    assert source["trade_earliest_time_utc"].max() == pd.Timestamp(
        "2023-12-31 23:55"
    )
    assert source["trade_earliest_time_utc"].max() < pd.Timestamp("2024-01-01")
