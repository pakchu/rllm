from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from training import backtest_pposm_sft_rlvr_gate as gate
from training.search_inventory_purge_reclaim_alpha import Trade
from training.search_pullback_premium_overheat_state_machine_alpha import (
    Config as StrategyConfig,
)


def _trade(signal: int, date: str, *, factor: float = 1.02) -> Trade:
    return Trade(
        signal_position=signal,
        entry_position=signal + 1,
        exit_position=signal + 2,
        side=1,
        gross_return=factor - 1.0,
        price_factor=factor,
        funding_factor=1.0,
        funding_debit_factor=1.0,
        favorable_price_factor=max(1.0, factor),
        adverse_price_factor=min(1.0, factor),
        entry_date=str(pd.Timestamp(date)),
    )


def _schedules() -> dict[str, list[Trade]]:
    schedules = {window: [] for _, window, _, _ in gate.builder.SPLIT_WINDOWS}
    schedules["test_2024"] = [_trade(10, "2024-01-08", factor=1.03)]
    schedules["eval_2025"] = [_trade(20, "2025-01-08", factor=0.99)]
    schedules["holdout_2026"] = [_trade(30, "2026-01-08", factor=1.02)]
    return schedules


def _oos_rows(schedules: dict[str, list[Trade]]) -> list[dict]:
    return [
        {"metadata": {"identity": gate.builder.trade_identity(window, trade)}}
        for window, _, _, _ in gate.REPORT_WINDOWS
        for trade in schedules[window]
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_predictions_require_exact_positional_length(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    _write_jsonl(path, [{"prediction": "TRADE"}])
    with pytest.raises(ValueError, match="positional prediction length mismatch"):
        gate.load_positional_predictions(path, expected_length=2)


def test_baseline_identity_lock_rejects_duplicates_and_reordering() -> None:
    schedules = _schedules()
    rows = _oos_rows(schedules)
    baseline, by_window = gate.freeze_oos_baseline(schedules, rows)
    assert baseline == tuple(
        trade
        for window, _, _, _ in gate.REPORT_WINDOWS
        for trade in schedules[window]
    )
    assert by_window["test_2024"] == tuple(schedules["test_2024"])

    duplicate = [rows[0], rows[0], rows[2]]
    with pytest.raises(ValueError, match="not unique"):
        gate.freeze_oos_baseline(schedules, duplicate)
    with pytest.raises(ValueError, match="positionally match"):
        gate.freeze_oos_baseline(schedules, list(reversed(rows)))


def test_apply_veto_only_returns_frozen_subset() -> None:
    baseline = tuple(_schedules()[window][0] for window, _, _, _ in gate.REPORT_WINDOWS)
    selected = gate.apply_veto_only(baseline, ["TRADE", "NO_TRADE", "TRADE"])
    assert selected == (baseline[0], baseline[2])
    assert all(any(trade is original for original in baseline) for trade in selected)


def test_backtest_reports_windows_costs_retention_and_deterministic_signflip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schedules = _schedules()
    rows_path = tmp_path / "oos.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "report.json"
    _write_jsonl(rows_path, _oos_rows(schedules))
    _write_jsonl(
        predictions_path,
        [
            {"prediction": "TRADE"},
            {"prediction": "NO_TRADE"},
            {"prediction": "TRADE"},
        ],
    )
    strategy_cfg = replace(
        StrategyConfig(), leverage=0.5, fee_rate=0.0005, slippage_rate=0.0001
    )
    monkeypatch.setattr(
        gate.builder,
        "load_frozen_manifest",
        lambda _: ({"freeze_hash": "frozen"}, strategy_cfg),
    )
    monkeypatch.setattr(
        gate.builder,
        "replay_frozen_schedules",
        lambda manifest, cfg: (pd.DataFrame(), pd.DataFrame(), schedules),
    )
    cfg = gate.Config(
        manifest=tmp_path / "manifest.json",
        oos_data=rows_path,
        predictions=predictions_path,
        output=output_path,
        signflip_permutations=100,
        signflip_seed=7,
    )

    first = gate.backtest(cfg)
    second = gate.backtest(cfg)

    assert first == second == json.loads(output_path.read_text(encoding="utf-8"))
    assert set(first["windows"]) == {
        "test_2024",
        "eval_2025",
        "holdout_2026",
        "combined_2024_2026_06_02",
    }
    combined = first["windows"]["combined_2024_2026_06_02"]
    assert combined["baseline_trade_count"] == 3
    assert combined["selected_trade_count"] == 2
    assert combined["vetoed_trade_count"] == 1
    assert set(combined["costs"]) == {"base_6bp", "stress_10bp"}
    for result in combined["costs"].values():
        assert result["baseline"]["equity_stats"]["trades"] == 3
        assert result["gated"]["equity_stats"]["trades"] == 2
        assert "strict_mdd_pct" in result["gated"]["equity_stats"]
        assert "cagr_to_strict_mdd" in result["gated"]["equity_stats"]
        assert result["gated"]["return_retention_vs_baseline"] is not None
        signflip = result["gated"]["one_sided_utc_week_sign_flip"]
        assert signflip["permutations"] == 100
        assert signflip["seed"] == 7
    assert first["invariants"]["replacement_allowed"] is False
