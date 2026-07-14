from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import evaluate_cash_sponsored_perp_rejection as evaluator
from training import preregister_cash_sponsored_perp_rejection as cspr


def _frame(rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows, freq="5min"),
            "quarantined": False,
            "perp_quarantined": False,
            "spot_quarantined": False,
            "spot_signed_quote_notional": 10.0,
            "spot_micro_log_return": 0.01,
            "spot_flow_coherence": 0.8,
            "spot_buyer_execution_centroid": 99.0,
            "spot_seller_execution_centroid": 100.0,
            "spot_close": 101.0,
            "signed_quote_notional": -10.0,
            "signed_event_imbalance": -0.5,
            "micro_log_return": 0.01,
            "flow_coherence": 0.8,
            "agg_trade_count": 100,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
        }
    )


def _primary_schedule(frame: pd.DataFrame) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    cfg = cspr.Config(baseline_bars=4, baseline_min_periods=2, hold_bars=2)
    signal, controls = cspr.classify_events(frame, cfg, quantile=0.5)
    return controls, cspr.nonoverlapping_schedule(signal, frame)


def _metrics(
    *,
    absolute_return: float = 5.0,
    ratio: float = 4.0,
    mdd: float = 1.0,
    trades: int = 100,
    p_value: float = 0.01,
) -> dict[str, object]:
    return {
        "absolute_return_pct": absolute_return,
        "cagr_to_strict_mdd": ratio,
        "strict_mdd_pct": mdd,
        "trade_count": trades,
        "weekly_cluster_sign_flip": {"p_value_one_sided": p_value},
    }


def test_evaluation_config_is_frozen() -> None:
    evaluator._validate_evaluation_config(evaluator.EvaluationConfig())
    with pytest.raises(ValueError, match="evaluation config is frozen"):
        evaluator._validate_evaluation_config(
            evaluator.EvaluationConfig(leverage=1.0)
        )


def test_windows_do_not_open_2024_or_later() -> None:
    assert max(end for _, end in evaluator.WINDOWS.values()) == "2024-01-01"
    assert set(evaluator.WINDOWS) == {
        "train",
        "select2023",
        "select2023_h1",
        "select2023_h2",
    }


def test_direction_flip_uses_exact_primary_clock() -> None:
    frame = _frame()
    controls, primary = _primary_schedule(frame)
    cfg = cspr.Config(baseline_bars=4, baseline_min_periods=2, hold_bars=2)
    schedules = evaluator.build_control_schedules(frame, controls, primary, cfg)
    flipped = schedules["direction_flip"]
    assert flipped[["signal_position", "entry_position", "exit_position"]].equals(
        primary[["signal_position", "entry_position", "exit_position"]]
    )
    assert np.array_equal(flipped["side"].to_numpy(), -primary["side"].to_numpy())


def test_single_source_controls_do_not_depend_on_opposite_quarantine() -> None:
    frame = _frame()
    controls, primary = _primary_schedule(frame)
    cfg = cspr.Config(baseline_bars=4, baseline_min_periods=2, hold_bars=2)
    baseline = evaluator.build_control_schedules(frame, controls, primary, cfg)

    altered = frame.copy()
    altered["perp_quarantined"] = True
    spot_only = evaluator._schedule_from_control(
        altered,
        cfg,
        mask=controls["spot_only"],
        side=cspr._directions(altered)["side"],
        quarantine=altered["spot_quarantined"],
        branch="spot_only",
    )
    assert len(spot_only) == len(baseline["spot_only"])

    altered = frame.copy()
    altered["spot_quarantined"] = True
    perp_side = np.sign(cspr._directions(altered)["perp_return"])
    perp_only = evaluator._schedule_from_control(
        altered,
        cfg,
        mask=controls["perp_only"],
        side=perp_side,
        quarantine=altered["perp_quarantined"],
        branch="perp_only",
    )
    assert len(perp_only) == len(baseline["perp_only"])


def test_stale_controls_do_not_use_current_spot_quarantine() -> None:
    frame = _frame()
    controls, primary = _primary_schedule(frame)
    cfg = cspr.Config(baseline_bars=4, baseline_min_periods=2, hold_bars=2)
    baseline = evaluator.build_control_schedules(frame, controls, primary, cfg)
    altered = frame.copy()
    altered["spot_quarantined"] = True
    rebuilt = evaluator.build_control_schedules(altered, controls, primary, cfg)
    assert evaluator._canonical_clock_records(rebuilt["spot_lag_1h"]) == (
        evaluator._canonical_clock_records(baseline["spot_lag_1h"])
    )


def test_role_swap_trades_perp_price_direction() -> None:
    frame = _frame()
    frame["spot_micro_log_return"] = -0.01
    frame["micro_log_return"] = -0.01
    frame["signed_quote_notional"] = 10.0
    frame["signed_event_imbalance"] = 0.5
    cfg = cspr.Config(baseline_bars=4, baseline_min_periods=2, hold_bars=2)
    signal, controls = cspr.classify_events(frame, cfg, quantile=0.5)
    primary = cspr.nonoverlapping_schedule(signal, frame)
    schedules = evaluator.build_control_schedules(frame, controls, primary, cfg)
    assert not schedules["role_swap"].empty
    assert schedules["role_swap"]["side"].eq(-1).all()


def test_slice_schedule_does_not_reschedule_boundary_events() -> None:
    schedule = pd.DataFrame(
        {
            "signal_date": ["2022-12-31 23:55:00", "2023-01-01 00:05:00"],
            "entry_date": ["2023-01-01 00:00:00", "2023-01-01 00:10:00"],
            "exit_date": ["2023-01-01 00:10:00", "2023-01-01 00:20:00"],
        }
    )
    selected = evaluator.slice_schedule(
        schedule,
        start="2023-01-01",
        end="2024-01-01",
    )
    assert len(selected) == 1
    assert selected.iloc[0]["signal_date"] == "2023-01-01 00:05:00"


def test_qualification_requires_primary_to_beat_every_control() -> None:
    windows: dict[str, dict[str, dict[str, object]]] = {}
    for window in evaluator.WINDOWS:
        windows[window] = {
            policy: _metrics(trades=40 if "h" in window else 100)
            for policy in evaluator.POLICY_NAMES
        }
    for control in evaluator.QUALIFICATION_CONTROLS:
        windows["train"][control] = _metrics(ratio=2.0)
        windows["select2023"][control] = _metrics(ratio=2.0)
    assert evaluator._qualification(windows)["qualifies"]

    windows["train"]["spot_only"] = _metrics(ratio=5.0)
    windows["select2023"]["spot_only"] = _metrics(ratio=5.0)
    result = evaluator._qualification(windows)
    assert not result["qualifies"]
    assert any("does not beat spot_only" in item for item in result["failures"])


def test_freeze_manifest_rejects_opened_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"outcomes_opened_for_cspr12": True}))
    monkeypatch.setattr(evaluator, "EVALUATION_FREEZE", freeze)
    with pytest.raises(ValueError, match="not frozen before outcomes"):
        evaluator.verify_evaluation_freeze()
