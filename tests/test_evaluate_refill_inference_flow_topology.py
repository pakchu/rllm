from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import evaluate_refill_inference_flow_topology as evaluator
from training import preregister_refill_inference_flow_topology as rift


def _frame(rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows, freq="5min"),
            "quarantined": False,
            "perp_quarantined": False,
            "spot_quarantined": False,
            "spot_close_vs_centroid_mid_bp": 10.0,
            "spot_micro_log_return": 0.01,
            "micro_log_return": 0.01,
            "spot_signed_quote_notional": 100.0,
            "signed_quote_notional": 100.0,
            "signed_event_imbalance": 0.5,
            "spot_minute_price_path_efficiency": 0.9,
            "spot_minute_flow_path_efficiency": 0.9,
            "spot_minute_flow_price_alignment": 1.0,
            "spot_minute_flow_sign_flip_rate": 0.0,
            "event_notional_hhi": 0.1,
            "interarrival_burstiness": 0.8,
            "agg_trade_count": 100,
        }
    )


def _clocks() -> tuple[pd.DataFrame, rift.Config, dict[str, pd.Series], pd.DataFrame]:
    frame = _frame()
    cfg = rift.Config(
        baseline_bars=4,
        baseline_min_periods=2,
        hold_bars=2,
        minimum_nonoverlap_total=1,
        minimum_nonoverlap_per_year=0,
        minimum_nonoverlap_per_2023_half=0,
    )
    signal, controls, _ = rift.classify_sequences(frame, cfg, quantile=0.5)
    primary = rift.nonoverlapping_schedule(signal, frame)
    return frame, cfg, controls, primary


def _metrics(
    *,
    absolute_return: float = 5.0,
    ratio: float = 4.0,
    mdd: float = 1.0,
    trades: int = 100,
    p_value: float = 0.01,
    gross_bp: float = 20.0,
) -> dict[str, object]:
    return {
        "absolute_return_pct": absolute_return,
        "cagr_to_strict_mdd": ratio,
        "strict_mdd_pct": mdd,
        "trade_count": trades,
        "mean_gross_underlying_move_bp": gross_bp,
        "weekly_cluster_sign_flip": {"p_value_one_sided": p_value},
    }


def test_evaluation_config_is_frozen() -> None:
    evaluator._validate_evaluation_config(evaluator.EvaluationConfig())
    with pytest.raises(ValueError, match="evaluation config is frozen"):
        evaluator._validate_evaluation_config(
            evaluator.EvaluationConfig(minimum_mean_gross_underlying_bp=0.0)
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
    frame, cfg, controls, primary = _clocks()
    schedules = evaluator.build_control_schedules(frame, controls, primary, cfg)
    flipped = schedules["direction_flip"]
    assert flipped[["signal_position", "entry_position", "exit_position"]].equals(
        primary[["signal_position", "entry_position", "exit_position"]]
    )
    assert np.array_equal(flipped["side"].to_numpy(), -primary["side"].to_numpy())


def test_spot_only_schedule_does_not_use_perp_quarantine() -> None:
    frame, cfg, controls, primary = _clocks()
    baseline = evaluator.build_control_schedules(frame, controls, primary, cfg)
    altered = frame.copy()
    altered["perp_quarantined"] = True
    spot_only = evaluator._schedule_from_control(
        altered,
        cfg,
        mask=controls["spot_only"],
        action=1,
        quarantine=altered["spot_quarantined"],
        branch="spot_only",
    )
    assert evaluator._canonical_clock_records(spot_only) == (
        evaluator._canonical_clock_records(baseline["spot_only"])
    )


def test_gross_underlying_move_inverts_exact_cost_multiplier() -> None:
    cfg = evaluator.EvaluationConfig()
    raw = 0.002
    cost = (cfg.fee_rate + cfg.slippage_rate) * cfg.leverage
    net = (1.0 - cost) * (1.0 + cfg.leverage * raw) * (1.0 - cost) - 1.0
    recovered = evaluator.mean_gross_underlying_move_bp(net * 100.0, cfg)
    assert np.isclose(recovered, 20.0)


def test_slice_schedule_never_rebuilds_boundary_clock() -> None:
    schedule = pd.DataFrame(
        {
            "signal_date": ["2022-12-31 23:55:00", "2023-01-01 00:05:00"],
            "entry_date": ["2023-01-01 00:00:00", "2023-01-01 00:10:00"],
            "exit_date": ["2023-01-01 00:10:00", "2023-01-01 08:10:00"],
        }
    )
    selected = evaluator.slice_schedule(
        schedule, start="2023-01-01", end="2024-01-01"
    )
    assert len(selected) == 1
    assert selected.iloc[0]["signal_date"] == "2023-01-01 00:05:00"


def test_qualification_enforces_gross_hurdle_and_all_controls() -> None:
    cfg = evaluator.EvaluationConfig()
    windows: dict[str, dict[str, dict[str, object]]] = {}
    for window in evaluator.WINDOWS:
        windows[window] = {
            policy: _metrics(trades=40 if "h" in window else 100)
            for policy in evaluator.POLICY_NAMES
        }
    for control in evaluator.QUALIFICATION_CONTROLS:
        windows["train"][control] = _metrics(ratio=2.0)
        windows["select2023"][control] = _metrics(ratio=2.0)
    assert evaluator._qualification(windows, cfg)["qualifies"]

    windows["select2023"]["primary"] = _metrics(gross_bp=12.0)
    result = evaluator._qualification(windows, cfg)
    assert not result["qualifies"]
    assert any("not above 12 bp" in item for item in result["failures"])


def test_freeze_manifest_rejects_opened_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"outcomes_opened_for_rift96": True}))
    monkeypatch.setattr(evaluator, "EVALUATION_FREEZE", freeze)
    with pytest.raises(ValueError, match="not frozen before outcomes"):
        evaluator.verify_evaluation_freeze()


def test_run_stops_at_freeze_guard_before_loading_price_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing-freeze.json"
    monkeypatch.setattr(evaluator, "EVALUATION_FREEZE", missing)

    def forbidden_load(_: object) -> tuple[pd.DataFrame, dict[str, object]]:
        raise AssertionError("price frame must not load before evaluator freeze")

    monkeypatch.setattr(rift, "load_causal_frame", forbidden_load)
    with pytest.raises(ValueError, match="freeze manifest is missing"):
        evaluator.run_evaluation(evaluator.EvaluationConfig())


def test_freeze_manifest_rejects_price_loaded_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "outcomes_opened_for_rift96": False,
                "evaluation_source": str(evaluator.EVALUATION_SOURCE),
                "evaluation_source_sha256": evaluator._sha256(
                    evaluator.EVALUATION_SOURCE
                ),
                "evaluation_source_commit": "0" * 40,
                "preregistration_commit": evaluator.PREREGISTRATION_COMMIT,
                "support_commit": evaluator.SUPPORT_COMMIT,
                "support_result_sha256": evaluator.PREREGISTRATION_RESULT_SHA256,
                "event_clock_sha256": evaluator.EVENT_CLOCK_SHA256,
                "opened_windows": [],
                "returns_or_prices_loaded_during_freeze": True,
                "mutable_parameters": [],
                "sealed_windows": [
                    *evaluator.WINDOWS,
                    "test2024",
                    "eval2025",
                    "ytd2026",
                ],
            }
        )
    )
    monkeypatch.setattr(evaluator, "EVALUATION_FREEZE", freeze)
    with pytest.raises(ValueError, match="loaded returns or prices"):
        evaluator.verify_evaluation_freeze()
