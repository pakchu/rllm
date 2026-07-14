from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from training import preregister_causal_adaptive_relational_tokens as carta


def _frame(rows: int = 320) -> pd.DataFrame:
    close = np.full(rows, 100.0)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows, freq="5min"),
            "open": close.copy(),
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "signed_quote_notional": np.ones(rows),
            "signed_event_imbalance": np.full(rows, 0.10),
            "micro_log_return": np.zeros(rows),
            "flow_coherence": np.full(rows, 0.10),
            "buy_sell_event_size_log_ratio": np.full(rows, 0.10),
            "interarrival_burstiness": np.full(rows, 0.10),
            "event_notional_hhi": np.full(rows, 0.01),
            "underlying_trades_per_agg_event": np.full(rows, 1.5),
            "normalized_effective_event_count": np.full(rows, 10.0),
            "sign_flip_rate": np.full(rows, 0.50),
            "signed_price_response": np.zeros(rows),
            "agg_trade_count": np.full(rows, 100),
            "quarantined": np.zeros(rows, dtype=bool),
        }
    )
    frame.loc[300, [
        "signed_quote_notional",
        "signed_event_imbalance",
        "micro_log_return",
        "flow_coherence",
        "buy_sell_event_size_log_ratio",
        "interarrival_burstiness",
        "event_notional_hhi",
        "close",
    ]] = [10.0, -0.50, -0.01, 0.60, 1.20, 0.90, 0.20, 99.0]
    frame.loc[300, ["open", "high", "low"]] = [100.0, 100.1, 98.9]
    frame.loc[301, ["signed_quote_notional", "signed_event_imbalance", "close"]] = [
        5.0,
        0.20,
        99.5,
    ]
    frame.loc[302, ["signed_quote_notional", "signed_event_imbalance", "close"]] = [
        -2.0,
        -0.20,
        100.5,
    ]
    frame.loc[301:302, "open"] = frame.loc[301:302, "close"]
    frame.loc[301:302, "high"] = frame.loc[301:302, "close"] + 0.1
    frame.loc[301:302, "low"] = frame.loc[301:302, "close"] - 0.1
    return frame


def _cfg() -> carta.Config:
    return replace(
        carta.Config(),
        setup_tension_quantile=0.50,
        structure_quantile=0.50,
        baseline_bars=10,
        baseline_min_periods=3,
        confirmation_bars=2,
        hold_bars=2,
        minimum_agg_trade_count=1,
    )


def test_candidate_is_observed_only_after_fixed_transition() -> None:
    state = carta.compute_carta_state(_frame(), _cfg())
    assert state.loc[300, "setup"]
    assert not state.loc[300, "candidate"]
    assert state.loc[302, "candidate"]
    assert state.loc[302, "reference_direction"] == 1
    assert state.loc[302, "hold_bars"] == 2


def test_token_prefix_is_invariant_to_future_changes() -> None:
    frame = _frame()
    baseline = carta.compute_carta_state(frame, _cfg())
    changed = frame.copy()
    changed.loc[303:, "signed_quote_notional"] = -1_000_000.0
    changed.loc[303:, "signed_event_imbalance"] = -1.0
    changed.loc[303:, ["open", "high", "low", "close"]] = 1.0
    replay = carta.compute_carta_state(changed, _cfg())
    pd.testing.assert_frame_equal(
        baseline.loc[:302, list(carta.TOKEN_COLUMNS)],
        replay.loc[:302, list(carta.TOKEN_COLUMNS)],
    )


def test_prompt_tokens_are_symbolic_and_exclude_timestamp() -> None:
    state = carta.compute_carta_state(_frame(), _cfg())
    tokens = carta.relational_tokens(state.loc[302])
    assert "date" not in tokens
    assert set(tokens) == set(carta.TOKEN_COLUMNS)
    assert all(isinstance(value, str) for value in tokens.values())
    assert all("2023-" not in value for value in tokens.values())


def test_schedule_requires_event_origin_inside_split() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-06-30 23:45", periods=8, freq="5min"),
            "quarantined": np.zeros(8, dtype=bool),
        }
    )
    state = pd.DataFrame(
        {
            "side": [0, 0, 0, 1, 0, -1, 0, 0],
            "hold_bars": [0, 0, 0, 1, 0, 1, 0, 0],
            "branch": ["none", "none", "none", "carta_candidate", "none", "carta_candidate", "none", "none"],
            "origin_position": [-1, -1, -1, 1, -1, 3, -1, -1],
        }
    )
    schedule = carta.nonoverlapping_carta_schedule(
        state,
        frame,
        start="2023-07-01",
        end="2024-01-01",
    )
    assert schedule["signal_position"].tolist() == [5]
    assert schedule["origin_position"].tolist() == [3]


def test_support_stopping_rule_selects_strictest_passing_quantile() -> None:
    trials = [
        {"setup_tension_quantile": 0.96, "passes_support": True},
        {"setup_tension_quantile": 0.975, "passes_support": True},
        {"setup_tension_quantile": 0.98, "passes_support": False},
    ]
    assert carta._selected_support_quantile(trials) == 0.975


def test_frozen_support_artifact_has_no_unavailable_model_tokens() -> None:
    result = json.loads(
        Path(
            "results/causal_adaptive_relational_tokens_support_2026-07-14.json"
        ).read_text()
    )
    assert result["protocol"]["outcomes_opened_for_carta"] is False
    assert result["all_support_gates_pass"] is True
    assert result["support_calibration"]["selected_setup_tension_quantile"] == 0.975
    assert result["support"]["nonoverlap_total"] == 559
    assert result["support"]["by_year"] == {
        "2020": 205,
        "2021": 60,
        "2022": 58,
        "2023": 236,
    }
    assert result["signature_support"]["select_unseen_signature_count"] == 236
    for counts in result["observed_token_vocabulary"].values():
        assert "-1" not in counts
        assert "UNAVAILABLE" not in counts
