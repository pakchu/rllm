from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import evaluate_cash_auction_transfer_catchup_handoff as evaluator
from training import preregister_cash_auction_transfer_catchup_handoff as catch


def _frame(rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=rows, freq="5min"),
            "quarantined": False,
        }
    )


def _primary_schedule(frame: pd.DataFrame, cfg: catch.Config) -> pd.DataFrame:
    active = pd.Series(False, index=frame.index)
    active.iloc[[0, 13, 26]] = True
    signal = pd.DataFrame(
        {
            "side": np.where(active, [1 if i != 13 else -1 for i in frame.index], 0),
            "branch": np.where(active, "catch12", "none"),
            "hold_bars": np.where(active, cfg.hold_bars, 0),
        }
    )
    return catch.nonoverlapping_schedule(signal, frame)


def _control_inputs(
    frame: pd.DataFrame,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    controls: dict[str, pd.Series] = {}
    sides: dict[str, pd.Series] = {}
    for offset, name in enumerate(evaluator.POLICY_NAMES):
        active = pd.Series(False, index=frame.index)
        active.iloc[offset % 10 :: 13] = True
        controls[name] = active
        sides[name] = pd.Series(
            np.where(np.arange(len(frame)) % 2 == 0, 1.0, -1.0),
            index=frame.index,
        )
    return controls, sides


def _metrics(
    *,
    absolute_return: float = 5.0,
    ratio: float = 4.0,
    mdd: float = 1.0,
    trades: int = 300,
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
            evaluator.EvaluationConfig(minimum_2023_half_trades=199)
        )


def test_windows_do_not_open_2024_or_later() -> None:
    assert max(end for _, end in evaluator.WINDOWS.values()) == "2024-01-01"
    assert set(evaluator.WINDOWS) == {
        "train",
        "select2023",
        "select2023_h1",
        "select2023_h2",
    }


def test_score_bearing_and_falsification_sets_are_frozen() -> None:
    assert set(evaluator.SCORE_BEARING_CONTROLS).isdisjoint(
        evaluator.FALSIFICATION_CONTROLS
    )
    assert set(evaluator.SCORE_BEARING_CONTROLS) | set(
        evaluator.FALSIFICATION_CONTROLS
    ) == set(evaluator.POLICY_NAMES) - {"primary"}


def test_direction_flip_uses_exact_primary_clock() -> None:
    frame = _frame()
    cfg = catch.Config(hold_bars=12)
    controls, sides = _control_inputs(frame)
    primary = _primary_schedule(frame, cfg)
    schedules = evaluator.build_control_schedules(frame, controls, sides, primary, cfg)
    flipped = schedules["direction_flip"]
    assert flipped[["signal_position", "entry_position", "exit_position"]].equals(
        primary[["signal_position", "entry_position", "exit_position"]]
    )
    assert np.array_equal(flipped["side"].to_numpy(), -primary["side"].to_numpy())
    assert tuple(schedules) == evaluator.POLICY_NAMES


def test_control_schedule_rejects_active_zero_side() -> None:
    frame = _frame()
    cfg = catch.Config(hold_bars=12)
    mask = pd.Series(False, index=frame.index)
    mask.iloc[0] = True
    with pytest.raises(ValueError, match="invalid active side"):
        evaluator._schedule_from_control(
            frame,
            cfg,
            mask=mask,
            side=pd.Series(0.0, index=frame.index),
            branch="broken",
        )


def test_gross_underlying_move_inverts_exact_cost_multiplier() -> None:
    cfg = evaluator.EvaluationConfig()
    raw = 0.002
    cost = (cfg.fee_rate + cfg.slippage_rate) * cfg.leverage
    net = (1.0 - cost) * (1.0 + cfg.leverage * raw) * (1.0 - cost) - 1.0
    recovered = evaluator.mean_gross_underlying_move_bp(net * 100.0, cfg)
    assert np.isclose(recovered, 20.0)


def test_strict_mdd_uses_favorable_first_and_excludes_exit_bar() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=3, freq="5min"),
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 110.0, 1_000.0],
            "low": [100.0, 90.0, 1.0],
        }
    )
    schedule = pd.DataFrame(
        [
            {
                "signal_position": 0,
                "entry_position": 1,
                "exit_position": 2,
                "signal_date": "2023-01-01 00:00:00",
                "entry_date": "2023-01-01 00:05:00",
                "exit_date": "2023-01-01 00:10:00",
                "side": 1,
                "branch": "catch12",
                "hold_bars": 1,
            }
        ]
    )
    cfg = evaluator.EvaluationConfig(cluster_permutations=1)
    result = evaluator.simulate_schedule(
        frame, schedule, start="2023-01-01", end="2023-01-02", cfg=cfg
    )
    assert np.isclose(result["strict_mdd_pct"], (1.0 - 0.95 / 1.05) * 100.0)
    assert result["strict_mdd_pct"] < 10.0


def test_slice_schedule_never_rebuilds_boundary_clock() -> None:
    schedule = pd.DataFrame(
        {
            "signal_date": ["2022-12-31 23:55:00", "2023-01-01 00:05:00"],
            "entry_date": ["2023-01-01 00:00:00", "2023-01-01 00:10:00"],
            "exit_date": ["2023-01-01 01:00:00", "2023-01-01 01:10:00"],
        }
    )
    selected = evaluator.slice_schedule(schedule, start="2023-01-01", end="2024-01-01")
    assert len(selected) == 1
    assert selected.iloc[0]["signal_date"] == "2023-01-01 00:05:00"


def test_qualification_enforces_half_count_gross_hurdle_and_score_controls() -> None:
    cfg = evaluator.EvaluationConfig()
    windows: dict[str, dict[str, dict[str, object]]] = {}
    for window in evaluator.WINDOWS:
        windows[window] = {policy: _metrics() for policy in evaluator.POLICY_NAMES}
    for control in evaluator.SCORE_BEARING_CONTROLS:
        windows["train"][control] = _metrics(ratio=2.0)
        windows["select2023"][control] = _metrics(ratio=2.0)
    windows["train"]["direction_flip"] = _metrics(ratio=99.0)
    windows["select2023"]["direction_flip"] = _metrics(ratio=99.0)
    assert evaluator._qualification(windows, cfg)["qualifies"]

    windows["select2023_h2"]["primary"] = _metrics(trades=199)
    result = evaluator._qualification(windows, cfg)
    assert not result["qualifies"]
    assert any("fewer than 200 trades" in item for item in result["failures"])

    windows["select2023_h2"]["primary"] = _metrics()
    windows["select2023"]["primary"] = _metrics(gross_bp=12.0)
    result = evaluator._qualification(windows, cfg)
    assert not result["qualifies"]
    assert any("not above 12 bp" in item for item in result["failures"])


def test_load_execution_market_requires_exact_clock_and_ohlc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    market_path = tmp_path / "market.csv.gz"
    market = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=3, freq="5min"),
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
        }
    )
    market.to_csv(market_path, index=False, compression="gzip")
    digest = hashlib.sha256(market_path.read_bytes()).hexdigest()
    monkeypatch.setattr(evaluator, "MARKET_DATA", market_path)
    monkeypatch.setattr(evaluator, "MARKET_DATA_SHA256", digest)
    signal = pd.DataFrame({"date": market["date"]})
    frame, source = evaluator.load_execution_market(signal, {"rows": 3})
    assert frame[["open", "high", "low", "close"]].equals(
        market[["open", "high", "low", "close"]]
    )
    assert source["market_rows"] == 3

    broken = signal.copy()
    broken.loc[2, "date"] += pd.Timedelta("5min")
    with pytest.raises(ValueError, match="does not align"):
        evaluator.load_execution_market(broken, {"rows": 3})


def test_freeze_manifest_rejects_opened_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"outcomes_opened_for_catch12": True}))
    monkeypatch.setattr(evaluator, "EVALUATION_FREEZE", freeze)
    with pytest.raises(ValueError, match="not frozen before outcomes"):
        evaluator.verify_evaluation_freeze()


def test_run_stops_at_freeze_guard_before_loading_any_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing-freeze.json"
    monkeypatch.setattr(evaluator, "EVALUATION_FREEZE", missing)

    def forbidden_signal_load(_: object) -> tuple[pd.DataFrame, dict[str, object]]:
        raise AssertionError("signal frame must not load before evaluator freeze")

    def forbidden_market_load(
        _: pd.DataFrame, __: dict[str, object]
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        raise AssertionError("market frame must not load before evaluator freeze")

    monkeypatch.setattr(catch, "load_causal_frame", forbidden_signal_load)
    monkeypatch.setattr(evaluator, "load_execution_market", forbidden_market_load)
    with pytest.raises(ValueError, match="freeze manifest is missing"):
        evaluator.run_evaluation(evaluator.EvaluationConfig())


def test_run_reserves_all_control_clocks_before_loading_market(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    events: list[str] = []
    monkeypatch.setattr(evaluator, "verify_evaluation_freeze", lambda: {})
    monkeypatch.setattr(evaluator, "verify_preregistration", lambda: ({}, {}))
    monkeypatch.setattr(catch, "load_causal_frame", lambda _cfg: (frame, {}))
    monkeypatch.setattr(
        evaluator,
        "verify_signal_replay",
        lambda *_args: ({}, {}, pd.DataFrame()),
    )

    def reserve_controls(*_args: object) -> dict[str, pd.DataFrame]:
        events.append("controls_reserved")
        return {}

    def open_market(*_args: object) -> tuple[pd.DataFrame, dict[str, object]]:
        assert events == ["controls_reserved"]
        events.append("market_opened")
        raise RuntimeError("stop after sequencing check")

    monkeypatch.setattr(evaluator, "build_control_schedules", reserve_controls)
    monkeypatch.setattr(evaluator, "load_execution_market", open_market)
    with pytest.raises(RuntimeError, match="sequencing check"):
        evaluator.run_evaluation(evaluator.EvaluationConfig())
    assert events == ["controls_reserved", "market_opened"]


def test_freeze_manifest_rejects_price_loaded_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "outcomes_opened_for_catch12": False,
                "evaluation_source": str(evaluator.EVALUATION_SOURCE),
                "evaluation_source_sha256": evaluator._sha256(
                    evaluator.EVALUATION_SOURCE
                ),
                "evaluation_source_commit": "0" * 40,
                "preregistration_commit": evaluator.PREREGISTRATION_COMMIT,
                "support_commit": evaluator.SUPPORT_COMMIT,
                "clock_commit": evaluator.CLOCK_COMMIT,
                "support_result_sha256": evaluator.SUPPORT_RESULT_SHA256,
                "event_clock_sha256": evaluator.EVENT_CLOCK_SHA256,
                "market_data_sha256": evaluator.MARKET_DATA_SHA256,
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
