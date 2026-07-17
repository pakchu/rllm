from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import build_aggregate_fill_compression_sweep_support as support
from training import preregister_aggregate_fill_compression_sweep as prereg


def _policy(**changes: object) -> prereg.Policy:
    return replace(
        prereg.Policy(),
        baseline_bars=2,
        baseline_min_periods=2,
        compression_quantile=0.5,
        coherence_quantile=0.5,
        response_quantile=0.5,
        activity_quantile=0.5,
        minimum_agg_trade_count=1,
        execution_delay_bars=2,
        hold_bars=2,
        **changes,
    )


def _frame(rows: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=rows, freq="5min"),
            "agg_trade_count": np.full(rows, 100.0),
            "underlying_trades_per_agg_event": np.ones(rows),
            "quote_notional": np.ones(rows),
            "signed_quote_notional": np.ones(rows),
            "flow_coherence": np.ones(rows),
            "signed_price_response": np.ones(rows),
            "source_available": np.ones(rows, dtype=bool),
            "quarantined": np.zeros(rows, dtype=bool),
        }
    )


def test_prior_quantile_is_strictly_prior_and_skips_quarantine() -> None:
    values = pd.Series([1.0, 2.0, 100.0, 4.0, 5.0])
    clean = pd.Series([True, True, False, True, True])
    result = support.prior_clean_quantile(
        values, clean, quantile=0.5, window=2, min_periods=2
    )
    assert result.iloc[3] == 1.5
    assert result.iloc[4] == 3.0


def test_primary_requires_compression_coherence_response_and_activity() -> None:
    frame = _frame()
    frame.loc[2, "underlying_trades_per_agg_event"] = 3.0
    frame.loc[2, "quote_notional"] = 3.0
    frame.loc[2, "flow_coherence"] = 3.0
    frame.loc[2, "signed_price_response"] = 3.0
    frame.loc[2, "signed_quote_notional"] = -2.0
    signals, diagnostics = support.classify_signals(frame, _policy())
    assert diagnostics["primary"].iloc[2]
    assert signals["primary"].loc[2, "side"] == -1
    changed = frame.copy()
    changed.loc[2, "underlying_trades_per_agg_event"] = 0.5
    _, changed_diagnostics = support.classify_signals(changed, _policy())
    assert not changed_diagnostics["primary"].iloc[2]
    assert changed_diagnostics["no_compression"].iloc[2]


def test_signal_prefix_is_invariant_to_future_mutation() -> None:
    frame = _frame(24)
    frame.loc[2, [
        "underlying_trades_per_agg_event",
        "quote_notional",
        "flow_coherence",
        "signed_price_response",
    ]] = 3.0
    baseline, _ = support.classify_signals(frame, _policy())
    changed = frame.copy()
    changed.loc[12:, "underlying_trades_per_agg_event"] = 1e12
    changed.loc[12:, "signed_quote_notional"] = -1e12
    replay, _ = support.classify_signals(changed, _policy())
    pd.testing.assert_frame_equal(baseline["primary"].loc[:11], replay["primary"].loc[:11])


def test_schedule_reserves_compute_bar_and_full_hold() -> None:
    frame = _frame(12)
    signal = pd.DataFrame(
        {
            "origin_position": [-1, -1, 2, *([-1] * 9)],
            "side": [0, 0, 1, *([0] * 9)],
            "branch": ["none", "none", "afcs_144", *(["none"] * 9)],
            "delay_bars": [0, 0, 2, *([0] * 9)],
            "hold_bars": [0, 0, 2, *([0] * 9)],
        }
    )
    segments = [(frame.loc[0, "date"], frame.loc[11, "date"] + pd.Timedelta(minutes=5))]
    schedule = support.nonoverlapping_schedule(signal, frame, segments=segments)
    assert schedule.loc[0, "signal_position"] == 2
    assert schedule.loc[0, "entry_position"] == 4
    assert schedule.loc[0, "exit_position"] == 6
    frame.loc[5, "quarantined"] = True
    assert support.nonoverlapping_schedule(signal, frame, segments=segments).empty


def test_shift_and_random_side_controls_are_clock_causal() -> None:
    frame = _frame(400)
    frame.loc[2, [
        "underlying_trades_per_agg_event",
        "quote_notional",
        "flow_coherence",
        "signed_price_response",
    ]] = 3.0
    signals, _ = support.classify_signals(frame, _policy())
    primary_pos = int(np.flatnonzero(signals["primary"]["side"].ne(0))[0])
    delayed_pos = int(np.flatnonzero(signals["one_hour_signal_delay"]["side"].ne(0))[0])
    shifted_pos = int(np.flatnonzero(signals["one_day_shifted_clock"]["side"].ne(0))[0])
    assert delayed_pos == primary_pos + 12
    assert shifted_pos == primary_pos + 288
    assert signals["random_side"].loc[primary_pos, "origin_position"] == primary_pos


def test_quarantine_extends_forward_only() -> None:
    available = pd.Series([True, True, False, True, True, True])
    gap = pd.Series([False, False, False, False, False, True])
    result = support.quarantine_mask(available, gap, post_gap_bars=2)
    assert result.tolist() == [False, False, True, True, True, True]


def test_frozen_policy_mutation_is_rejected() -> None:
    with pytest.raises(ValueError, match="policy is frozen"):
        support.run_support(replace(prereg.Policy(), hold_bars=145))


def test_frozen_support_artifact_is_outcome_blind_and_passing() -> None:
    path = Path("results/aggregate_fill_compression_sweep_support_2026-07-17.json")
    payload = json.loads(path.read_text())
    assert payload["protocol"]["outcomes_opened"] is False
    assert payload["source"]["market_columns_loaded"] == ["date"]
    assert payload["source"]["price_or_outcome_columns_loaded"] == []
    assert payload["support_decision"] == "pass"
    assert payload["support"]["train_2020_2022"] == 421
    assert payload["support"]["selection_2023"] == 152
    assert payload["primary_clock_sha256"] == (
        "bf1611554604c1930ba2212e674ea434f7c9793377b3f33ef531b3b4e0381688"
    )
