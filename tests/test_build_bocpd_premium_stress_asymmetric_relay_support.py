import numpy as np
import pandas as pd

from training import build_bocpd_premium_stress_asymmetric_relay_support as support
from training import preregister_bocpd_premium_stress_asymmetric_relay as prereg


def _panel(dates: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.DataFrame({"date": dates})
    for name in support.FEATURES:
        frame[name] = 0.0
    frame["bocpd_state"] = 1
    return frame


def test_frozen_signals_require_hourly_grid_and_bocpd_only_for_long():
    dates = pd.DatetimeIndex(
        ["2023-07-01T00:50:00Z", "2023-07-01T00:55:00Z", "2023-07-01T01:55:00Z"]
    )
    frame = _panel(dates)
    frame["funding_rate"] = -0.001
    frame["trend_96"] = 0.02
    frame.loc[2, "bocpd_state"] = 5
    frame["htf_3d_range_pos"] = -1.0
    frame["premium_index_zscore"] = -2.0

    longs, shorts = support.state_signals(frame)
    assert longs.tolist() == [False, True, False]
    assert shorts.tolist() == [False, True, True]

    no_gate_longs, _ = support.state_signals(frame, "no_bocpd_gate")
    assert no_gate_longs.tolist() == [False, True, True]


def test_clock_skips_conflict_reserves_globally_and_uses_asymmetric_holds(monkeypatch):
    frame = _panel(pd.date_range("2023-07-01T00:55:00Z", periods=5, freq="h"))
    monkeypatch.setattr(
        support,
        "state_signals",
        lambda _frame, _control="primary": (
            np.array([True, True, False, False, False]),
            np.array([True, False, True, False, True]),
        ),
    )
    clock = support.build_clock(frame)
    assert len(clock) == 1
    assert clock.iloc[0]["side"] == 1
    assert clock.iloc[0]["entry_time"] == pd.Timestamp("2023-07-01T02:00:00Z")
    assert clock.iloc[0]["exit_time"] == pd.Timestamp("2023-07-03T02:00:00Z")


def test_short_clock_has_24_hour_hold_and_direction_flip_changes_only_side(monkeypatch):
    frame = _panel(pd.DatetimeIndex(["2024-02-01T00:55:00Z"]))
    monkeypatch.setattr(
        support,
        "state_signals",
        lambda _frame, _control="primary": (np.array([False]), np.array([True])),
    )
    primary = support.build_clock(frame)
    flipped = support.build_clock(frame, "direction_flip")
    assert primary.iloc[0]["exit_time"] - primary.iloc[0]["entry_time"] == pd.Timedelta(
        hours=24
    )
    flipped_without_side = flipped.drop(columns="side").assign(control="primary")
    assert primary.drop(columns="side").equals(flipped_without_side)
    assert primary["side"].tolist() == [-1]
    assert flipped["side"].tolist() == [1]


def test_bocpd_filter_is_prefix_invariant():
    index = pd.date_range("2020-01-01T01:00:00", periods=36, freq="h")
    values = np.linspace(-0.02, 0.03, len(index))
    hourly = pd.DataFrame(
        {"ret1": values, "flow24": np.sin(np.arange(len(index)) / 4.0) / 100.0},
        index=index,
    )
    contract = {
        "hourly_inputs": ["ret1", "flow24"],
        "standardization_mean": [0.0, 0.0],
        "standardization_std": [1.0, 1.0],
        "hazard_lambda_hours": 336,
        "max_run_length": 1000,
        "prior_kappa": 0.1,
        "prior_alpha": 2.0,
        "prior_beta": 1.0,
        "short_run_horizon_hours": 6,
    }
    prefix = support.bocpd_hourly_output(hourly.iloc[:24], contract)
    full = support.bocpd_hourly_output(hourly, contract).iloc[:24]
    pd.testing.assert_frame_equal(prefix.reset_index(drop=True), full.reset_index(drop=True))


def test_exact_support_gates_and_terminal_rejection_without_repair():
    support.validate_frozen_contract(prereg.build())
    rows = []
    split_specs = {
        "train": ("2023-07-01", 8),
        "test": ("2024-01-01", 12),
        "eval": ("2025-01-01", 12),
        "final": ("2026-01-01", 8),
    }
    for split, (start, count) in split_specs.items():
        for position in range(count):
            rows.append(
                {
                    "split": split,
                    "side": 1 if position % 2 == 0 else -1,
                    "entry_time": pd.Timestamp(start, tz="UTC")
                    + pd.DateOffset(months=position % 3),
                }
            )
    clock = pd.DataFrame(rows)
    _, checks, passed = support.support_verdict(clock)
    assert passed
    assert checks == {
        f"{split}_{gate}": True
        for split in support.SPLITS
        for gate in ("minimum_events", "side_balance", "month_concentration")
    }

    rejected = clock[~((clock["split"] == "final") & (clock.index == clock.index[-1]))]
    _, rejected_checks, rejected_passed = support.support_verdict(rejected)
    assert not rejected_passed
    assert rejected_checks["final_minimum_events"] is False
    decision = "pass_to_novelty" if rejected_passed else "terminal_source_support_reject"
    assert decision == "terminal_source_support_reject"
    assert support.ECONOMIC_OUTCOMES_AUTHORIZED is False
