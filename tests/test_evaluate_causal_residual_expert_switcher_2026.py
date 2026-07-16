from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import training.evaluate_causal_residual_expert_switcher_2026 as evaluator
from training.select_leave_one_out_residual_exhaustion_pre2025 import MarketBundle


def seed_frame(rows: int = 80) -> pd.DataFrame:
    signal = pd.date_range("2025-01-01", periods=rows, freq="13h")
    frame = pd.DataFrame(
        {
            "signal_time": signal,
            "entry_time": signal + pd.Timedelta(minutes=5),
            "exit_time": signal + pd.Timedelta(hours=12, minutes=5),
            "edge": np.linspace(-0.02, 0.02, rows),
            "range_risk": 0.01,
        }
    )
    for number, feature in enumerate(evaluator.CURRENT_FEATURES, start=1):
        frame[feature] = np.linspace(-1.0, 1.0, rows) ** number
    frame["continuation_net_log_return"] = frame["edge"] / 2.0
    frame["reversion_net_log_return"] = -frame["edge"] / 2.0
    return frame


def base_clock(rows: int = 2) -> pd.DataFrame:
    signal = pd.date_range("2026-01-02", periods=rows, freq="13h")
    return pd.DataFrame(
        {
            "policy_id": "CRES01_BASE",
            "signal_time": signal,
            "feature_available_time": signal,
            "entry_time": signal + pd.Timedelta(minutes=5),
            "exit_time": signal + pd.Timedelta(hours=12, minutes=5),
            "residual_horizon_hours": 12,
            "hold_hours": 12,
            "continuation_long_symbol": "ETHUSDT",
            "continuation_short_symbol": "ADAUSDT",
            "continuation_long_weight_gross1": 0.4,
            "continuation_short_weight_abs_gross1": 0.6,
            "continuation_long_beta": 1.5,
            "continuation_short_beta": 1.0,
            "loser_residual_z": -2.0,
            "winner_residual_z": 2.0,
            "loser_flow_z": 0.0,
            "winner_flow_z": 0.0,
            "setup_score": 2.0,
            "range_risk": 0.01,
        }
    )


def empty_bundle() -> MarketBundle:
    return MarketBundle(pd.DatetimeIndex([]), {}, {}, {})


def test_walk_materializes_each_decision_before_opening_its_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluator, "verify_evaluation_freeze", lambda: {})
    order: list[str] = []
    original_decide = evaluator.decide_event

    def decide(history, current):
        order.append(f"decide:{current['signal_time']}")
        return original_decide(history, current)

    def outcome(bundle, row):
        order.append(f"outcome:{row['signal_time']}")
        return {
            "continuation_net_log_return": 0.01,
            "reversion_net_log_return": -0.01,
            "edge": 0.02,
        }

    monkeypatch.setattr(evaluator, "decide_event", decide)
    decisions, history = evaluator.walk_forward_decisions(
        empty_bundle(), base_clock(), seed_frame(), outcome_provider=outcome
    )
    assert len(decisions) == 2
    assert len(history) == 82
    assert order[0].startswith("decide:")
    assert order[1].startswith("outcome:")
    assert order[2].startswith("decide:")
    assert order[3].startswith("outcome:")
    assert decisions["decision_materialized_before_outcome"].all()


def test_decision_excludes_outcome_not_published_by_signal() -> None:
    history = evaluator.enrich_lag_features(seed_frame())
    current = evaluator._current_event(base_clock(1).iloc[0], history)
    available = evaluator.decide_event(history, current)
    history.loc[history.index[-1], "exit_time"] = current["signal_time"]
    unavailable = evaluator.decide_event(history, current)
    assert unavailable["training_rows"] == available["training_rows"] - 1


def test_selected_clock_scales_reversion_and_preserves_beta_neutrality() -> None:
    decisions = base_clock(1)
    decisions["choice"] = "reversion"
    decisions["gross_scale"] = 0.5
    decisions["predicted_edge"] = -0.01
    decisions["confidence_threshold"] = 0.005
    selected = evaluator.selected_clock(decisions)
    assert selected.loc[0, "long_symbol"] == "ADAUSDT"
    assert selected.loc[0, "short_symbol"] == "ETHUSDT"
    assert selected.loc[0, "long_weight"] == pytest.approx(0.3)
    assert selected.loc[0, "short_weight_abs"] == pytest.approx(0.2)
    exposure = selected.loc[0, "long_weight"] * selected.loc[0, "long_beta"]
    exposure -= selected.loc[0, "short_weight_abs"] * selected.loc[0, "short_beta"]
    assert exposure == pytest.approx(0.0)


def test_selected_clock_keeps_schema_when_every_event_is_flat() -> None:
    decisions = base_clock(1)
    decisions["choice"] = "flat"
    decisions["gross_scale"] = 1.0
    decisions["predicted_edge"] = 0.0
    decisions["confidence_threshold"] = 0.01
    selected = evaluator.selected_clock(decisions)
    assert selected.empty
    assert {"entry_time", "exit_time", "long_symbol", "short_symbol"}.issubset(
        selected.columns
    )


def test_zero_prediction_is_flat() -> None:
    history = evaluator.enrich_lag_features(seed_frame())
    history["edge"] = 0.01
    current = evaluator._current_event(base_clock(1).iloc[0], history)
    decision = evaluator.decide_event(history, current)
    assert decision["choice"] == "flat"
    assert decision["predicted_edge"] == pytest.approx(0.0)


def test_verify_freeze_rejects_missing_file(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluator, "EVALUATION_FREEZE", tmp_path / "missing.json")
    with pytest.raises(ValueError, match="freeze is missing"):
        evaluator.verify_evaluation_freeze()


def test_freeze_validation_rejects_outcome_opened(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "freeze.json"
    core = {
        "outcomes_opened": True,
        "evaluation_source": str(evaluator.EVALUATION_SOURCE),
    }
    path.write_text(json.dumps({**core, "manifest_hash": evaluator.canonical_hash(core)}))
    monkeypatch.setattr(evaluator, "EVALUATION_FREEZE", path)
    with pytest.raises(ValueError, match="already opened outcomes"):
        evaluator.verify_evaluation_freeze()


def test_run_stops_at_support_before_market_outcomes(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(
        evaluator,
        "verify_support_and_clock",
        lambda: (_ for _ in ()).throw(ValueError("bad support")),
    )
    monkeypatch.setattr(
        evaluator,
        "load_bundle",
        lambda: pytest.fail("market outcome boundary must not be reached"),
    )
    with pytest.raises(ValueError, match="bad support"):
        evaluator.run(tmp_path / "result.json", tmp_path / "result.md")


def test_result_hash_roundtrip() -> None:
    payload = evaluator._seal({"a": 1, "b": [2, 3]})
    assert payload["manifest_hash"] == evaluator.canonical_hash({"a": 1, "b": [2, 3]})
