from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import evaluate_packet_churn_persistence_pre2024 as evaluator
from training.search_inventory_purge_reclaim_alpha import ExecutionEngine, Trade


def _clock() -> pd.DataFrame:
    setup = pd.Timestamp("2023-01-01 00:00:00")
    confirmation = setup + pd.Timedelta("30min")
    available = confirmation + pd.Timedelta("5min")
    entry = available + pd.Timedelta("5min")
    exit_time = entry + pd.Timedelta(minutes=5 * evaluator.HOLD_BARS)
    return pd.DataFrame(
        [
            {
                "setup_position": 0,
                "confirmation_end_position": 6,
                "entry_position": 8,
                "exit_position": 104,
                "side": 1,
                "setup_bar_date": str(setup),
                "confirmation_end_bar_date": str(confirmation),
                "signal_available_at": str(available),
                "entry_date": str(entry),
                "exit_date": str(exit_time),
            }
        ]
    )


def _market(rows: int = 130) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=rows, freq="5min")
    close = np.linspace(100.0, 110.0, rows)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
        }
    )


def _funding() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "funding_rate": pd.Series(dtype=float),
        }
    )


def test_latency_controls_preserve_signal_and_hold() -> None:
    clock = _clock()
    immediate = evaluator.shift_execution_clock(clock, -1)
    extra = evaluator.shift_execution_clock(clock, 1)
    assert immediate.loc[0, "entry_position"] == 7
    assert extra.loc[0, "entry_position"] == 9
    assert immediate.loc[0, "exit_position"] - immediate.loc[0, "entry_position"] == 96
    assert extra.loc[0, "exit_position"] - extra.loc[0, "entry_position"] == 96
    assert pd.Timestamp(immediate.loc[0, "entry_date"]) == pd.Timestamp(
        clock.loc[0, "signal_available_at"]
    )
    assert (
        immediate.loc[0, "signal_available_at"] == clock.loc[0, "signal_available_at"]
    )


def test_window_clock_requires_setup_signal_entry_and_exit_containment() -> None:
    clock = _clock()
    crossing = clock.copy()
    crossing.loc[0, "setup_bar_date"] = "2022-12-31 23:55:00"
    combined = pd.concat([crossing, clock], ignore_index=True)
    selected = evaluator._window_clock(combined, "selection_2023")
    assert len(selected) == 1
    assert selected.iloc[0]["setup_bar_date"] == "2023-01-01 00:00:00"


def test_clock_dates_must_match_feature_grid_positions(tmp_path: Path) -> None:
    feature = tmp_path / "feature.csv"
    pd.DataFrame(
        {"date": pd.date_range("2023-01-01", periods=130, freq="5min")}
    ).to_csv(feature, index=False)
    cfg = replace(evaluator.EvaluationConfig(), feature_csv=str(feature))
    evaluator._verify_clock_against_feature_grid(cfg, _clock())
    broken = _clock()
    broken.loc[0, "entry_date"] = "2023-01-01 00:45:00"
    with pytest.raises(ValueError, match="entry_date"):
        evaluator._verify_clock_against_feature_grid(cfg, broken)


def test_build_trades_uses_frozen_entry_and_exit() -> None:
    cfg = evaluator.EvaluationConfig()
    engine = ExecutionEngine(_market(), _funding(), evaluator._engine_config(cfg))
    trades = evaluator._build_trades(engine, _clock())
    assert len(trades) == 1
    assert trades[0].entry_position == 8
    assert trades[0].exit_position == 104
    assert trades[0].side == 1
    flipped = evaluator._build_trades(engine, _clock(), flip=True)
    assert flipped[0].side == -1


def test_exact_boundary_funding_keeps_debit_and_drops_credit() -> None:
    cfg = evaluator.EvaluationConfig()
    clock = _clock()
    entry = pd.Timestamp(clock.loc[0, "entry_date"])
    exit_time = pd.Timestamp(clock.loc[0, "exit_date"])
    funding = pd.DataFrame(
        {
            "date": [entry, entry + pd.Timedelta("4h"), exit_time],
            "funding_rate": [0.01, -0.01, -0.02],
        }
    )
    engine = ExecutionEngine(_market(), funding, evaluator._engine_config(cfg))
    trade = evaluator._build_trades(engine, clock)[0]
    expected = (1.0 - cfg.leverage * 0.01) * (1.0 + cfg.leverage * 0.01)
    assert trade.funding_factor == pytest.approx(expected)
    assert trade.funding_debit_factor == pytest.approx(1.0 - cfg.leverage * 0.01)


def test_strict_mdd_applies_virtual_exit_cost_at_adverse_mark() -> None:
    cfg = replace(
        evaluator.EvaluationConfig(),
        leverage=0.5,
        fee_rate=0.01,
        slippage_rate=0.0,
    )
    trade = Trade(
        signal_position=0,
        entry_position=1,
        exit_position=2,
        side=1,
        gross_return=0.10,
        price_factor=1.05,
        funding_factor=1.0,
        funding_debit_factor=1.0,
        favorable_price_factor=1.05,
        adverse_price_factor=1.0,
        entry_date="2020-01-01 00:05:00",
    )
    stats = evaluator.strict_equity_stats(
        [trade], start="2020-01-01", end="2021-01-01", cfg=cfg
    )
    execution = 1.0 - cfg.leverage * cfg.fee_rate
    intratrade_peak = execution * trade.favorable_price_factor
    adverse_liquidation = execution * execution
    expected_mdd = (1.0 - adverse_liquidation / intratrade_peak) * 100.0
    assert stats["strict_mdd_pct"] == pytest.approx(expected_mdd)


def test_train_gate_requires_return_ratio_mdd_count_stress_and_cluster() -> None:
    base = {
        "absolute_return_pct": 1.0,
        "cagr_to_strict_mdd": 1.5,
        "strict_mdd_pct": 15.0,
        "trades": 100,
        "weekly_cluster_sign_flip": {"p_value_one_sided": 0.09},
    }
    stress = {"absolute_return_pct": 0.01}
    assert all(evaluator.train_gates(base, stress).values())
    assert not evaluator.train_gates(dict(base, strict_mdd_pct=15.01), stress)[
        "strict_mdd_at_most_15"
    ]
    assert not evaluator.train_gates(base, {"absolute_return_pct": 0.0})[
        "ten_bp_per_side_stress_positive"
    ]


def test_selection_gate_requires_both_halves_stress_and_cluster() -> None:
    full = {
        "absolute_return_pct": 1.0,
        "cagr_to_strict_mdd": 3.0,
        "strict_mdd_pct": 15.0,
        "weekly_cluster_sign_flip": {"p_value_one_sided": 0.09},
    }
    half = {"absolute_return_pct": 0.01}
    stress = {"absolute_return_pct": 0.01}
    assert all(evaluator.selection_gates(full, half, half, stress).values())
    assert not evaluator.selection_gates(
        full, half, {"absolute_return_pct": 0.0}, stress
    )["each_half_absolute_return_positive"]


def test_market_and_funding_prefix_loaders_do_not_parse_later_rows(
    tmp_path: Path,
) -> None:
    market_path = tmp_path / "market.csv"
    market_path.write_text(
        "date,open,high,low,close\n"
        "2022-12-31 23:50:00,100,101,99,100\n"
        "2022-12-31 23:55:00,100,101,99,100\n"
        "2023-01-01 00:00:00,not-opened,not-opened,not-opened,not-opened\n"
    )
    funding_path = tmp_path / "funding.csv"
    funding_path.write_text(
        "funding_time_utc,funding_rate\n"
        "2022-12-31 08:00:00+00:00,0.0001\n"
        "2022-12-31 16:00:00+00:00,0.0001\n"
        "2023-01-01 00:00:00+00:00,not-opened\n"
    )
    cfg = replace(
        evaluator.EvaluationConfig(),
        market_csv=str(market_path),
        funding_csv=str(funding_path),
    )
    market = evaluator._load_market_prefix(cfg, rows=2)
    funding = evaluator._load_funding_prefix(cfg, rows=2)
    assert len(market) == 2
    assert len(funding) == 2
    assert market["date"].max() < pd.Timestamp("2023-01-01")
    assert funding["date"].max() < pd.Timestamp("2023-01-01")


def test_outcome_boundary_scan_reads_timestamps_without_outcome_values(
    tmp_path: Path,
) -> None:
    dates = pd.date_range("2022-12-31 23:55:00", periods=3, freq="5min")
    feature = tmp_path / "feature.csv"
    market = tmp_path / "market.csv"
    funding = tmp_path / "funding.csv"
    pd.DataFrame({"date": dates, "secret": ["x", "y", "z"]}).to_csv(
        feature, index=False
    )
    pd.DataFrame({"date": dates, "open": ["x", "y", "z"]}).to_csv(market, index=False)
    pd.DataFrame(
        {
            "funding_time_utc": [
                "2022-12-31 16:00:00+00:00",
                "2023-01-01 00:00:00+00:00",
            ],
            "funding_rate": ["x", "y"],
        }
    ).to_csv(funding, index=False)
    cfg = replace(
        evaluator.EvaluationConfig(),
        feature_csv=str(feature),
        market_csv=str(market),
        funding_csv=str(funding),
    )
    boundaries = evaluator._outcome_boundaries(cfg)
    assert boundaries["market"]["rows_pre2023"] == 1
    assert boundaries["funding"]["rows_pre2023"] == 1


def test_selection_loader_refuses_failed_train_artifact(tmp_path: Path) -> None:
    freeze = tmp_path / "freeze.json"
    freeze.write_text("{}\n")
    train_output = tmp_path / "train.json"
    cfg = replace(
        evaluator.EvaluationConfig(),
        freeze_output=str(freeze),
        train_output=str(train_output),
    )
    freeze_data = {
        "outcome_boundaries": {
            "market": {"rows_pre2023": 2},
            "funding": {"rows_pre2023": 2},
        },
        "frozen_clocks": {"primary": {"hash": "clock"}},
    }
    payload = {
        "schema_version": 1,
        "created_at": "fixed",
        "config": asdict(cfg),
        "freeze_sha256": evaluator.sha256_file(freeze),
        "primary": {
            "name": evaluator.SELECTED_NAME,
            "clock_hash": "clock",
            "gates": {"absolute_return_positive": False},
            "passes": False,
        },
        "decision": "reject_before_selection",
        "protocol": {
            "opened_windows": ["train_2020_2022"],
            "selection_2023_opened": False,
            "market_value_rows_parsed": 2,
            "funding_value_rows_parsed": 2,
        },
    }
    payload["result_hash"] = evaluator._result_hash(payload)
    train_output.write_text(json.dumps(payload) + "\n")
    with pytest.raises(PermissionError, match="remains sealed"):
        evaluator._verify_passing_train_result(cfg, freeze_data)


def test_self_consistent_handwritten_train_cannot_unlock_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text("{}\n")
    train_output = tmp_path / "train.json"
    cfg = replace(
        evaluator.EvaluationConfig(),
        freeze_output=str(freeze_path),
        train_output=str(train_output),
    )
    freeze = {
        "outcome_boundaries": {
            "market": {"rows_pre2023": 2},
            "funding": {"rows_pre2023": 2},
        },
        "frozen_clocks": {"primary": {"hash": "clock"}},
    }
    payload = {
        "schema_version": 1,
        "created_at": "fixed",
        "config": asdict(cfg),
        "freeze_sha256": evaluator.sha256_file(freeze_path),
        "primary": {
            "name": evaluator.SELECTED_NAME,
            "clock_hash": "clock",
            "gates": {"all": True},
            "passes": True,
        },
        "decision": "open_selection_2023",
        "protocol": {
            "opened_windows": ["train_2020_2022"],
            "selection_2023_opened": False,
            "market_value_rows_parsed": 2,
            "funding_value_rows_parsed": 2,
        },
    }
    payload["result_hash"] = evaluator._result_hash(payload)
    train_output.write_text(json.dumps(payload) + "\n")
    monkeypatch.setattr(
        evaluator,
        "_compute_train_report",
        lambda *_args, **_kwargs: {"created_at": "fixed", "replayed": True},
    )
    with pytest.raises(ValueError, match="does not exactly replay"):
        evaluator._verify_passing_train_result(cfg, freeze)


def test_selection_checks_train_before_parsing_outcome_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    monkeypatch.setattr(evaluator, "verify_evaluator_freeze", lambda _cfg: {})

    def deny(
        _cfg: evaluator.EvaluationConfig, _freeze: dict[str, object]
    ) -> dict[str, object]:
        raise PermissionError("sealed")

    def forbidden_loader(
        _cfg: evaluator.EvaluationConfig, *, rows: int
    ) -> pd.DataFrame:
        nonlocal called
        called = True
        return pd.DataFrame({"rows": [rows]})

    monkeypatch.setattr(evaluator, "_verify_passing_train_result", deny)
    monkeypatch.setattr(evaluator, "_load_market_prefix", forbidden_loader)
    with pytest.raises(PermissionError, match="sealed"):
        evaluator.evaluate_selection(evaluator.EvaluationConfig())
    assert called is False


def test_frozen_support_clock_passes_execution_contract() -> None:
    evaluator._verify_static_dependencies()
    clock = evaluator._load_clock()
    assert len(clock) == 192
    assert evaluator.execution_clock_hash(clock) == evaluator.execution_clock_hash(
        evaluator._load_clock()
    )
