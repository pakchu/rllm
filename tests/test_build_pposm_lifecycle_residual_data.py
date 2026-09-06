from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from training import build_pposm_lifecycle_residual_data as lifecycle
from training.search_inventory_purge_reclaim_alpha import Trade


@dataclass(frozen=True)
class _Cfg:
    leverage: float = 1.0
    fee_rate: float = 0.0
    slippage_rate: float = 0.0


class _Engine:
    def __init__(self, market: pd.DataFrame, trades: dict[tuple[int, int], Trade | None]):
        self.market = market
        self._trades = trades

    def trade_at(self, signal: int, side: int, hold: int, tp_bps: int, sl_bps: int):
        return self._trades.get((int(signal), int(tp_bps)))


def _trade(signal: int, entry: int, exit_: int, factor: float) -> Trade:
    return Trade(
        signal_position=signal,
        entry_position=entry,
        exit_position=exit_,
        side=1,
        gross_return=factor - 1.0,
        price_factor=factor,
        funding_factor=1.0,
        funding_debit_factor=1.0,
        favorable_price_factor=max(1.0, factor),
        adverse_price_factor=min(1.0, factor),
        entry_date=str(pd.Timestamp("2021-01-01") + pd.Timedelta(minutes=5 * entry)),
    )


def _manifest() -> dict:
    return {
        "freeze_hash": "freeze",
        "state_thresholds": {
            "htf_1w_return_1_q50": 0.0,
            "rex_576_range_width_pct_q50": 0.0,
            "quote_vol_z_1d_q20": 0.0,
            "premium_index_change_q67": 10_000.0,
            "rex_576_range_pos_q67": 10_000.0,
        },
        "spec": {
            "side": 1,
            "hold_bars": 10,
            "stop_bps": 200,
            "capitulation_take_bps": 400,
            "normal_take_bps": 1200,
        },
    }


def _state(rows: int) -> pd.DataFrame:
    frame = pd.DataFrame(
        {name: np.arange(rows, dtype=float) + i for i, name in enumerate(lifecycle.pposm.FEATURE_QUANTILES)}
    )
    frame["capitulation"] = False
    frame["overheat"] = False
    return frame


def test_builder_emits_only_always_tp4_reference_anchors() -> None:
    market = pd.DataFrame({"date": pd.date_range("2021-01-01", periods=8, freq="5min")})
    active = np.zeros(len(market), dtype=bool)
    active[[0, 1]] = True
    engine = _Engine(
        market,
        {
            (0, 400): _trade(0, 1, 3, 1.10),
            (0, 1200): _trade(0, 1, 3, 1.20),
            (1, 400): _trade(1, 2, 4, 0.50),
            (1, 1200): _trade(1, 2, 4, 2.00),
        },
    )

    rows = lifecycle.rows_from_train_context(
        market, _state(len(market)), active, _manifest(), _Cfg(), engine  # type: ignore[arg-type]
    )
    assert len(rows) == 2
    assert {row["metadata"]["signal_position"] for row in rows} == {0}
    assert all(row["metadata"]["reference_anchor"] is True for row in rows)
    assert all("|pre_2024|0" in row["metadata"]["base_identity"] for row in rows)


def test_lifecycle_utility_uses_log_equity_ratio_and_stress_mdd_gate() -> None:
    baseline = {
        cost: {"absolute_return_pct": 10.0}
        for cost in ("base_6bp", "stress_10bp")
    }
    replacement = {
        cost: {"absolute_return_pct": 11.0}
        for cost in ("base_6bp", "stress_10bp")
    }
    deltas = {
        "base_6bp": {
            "cagr_to_strict_mdd": 0.2,
            "strict_mdd_pct": 0.0,
        },
        "stress_10bp": {
            "cagr_to_strict_mdd": 0.1,
            "strict_mdd_pct": 0.0,
        },
    }
    utility, gates, components = lifecycle.switch_utility_from_deltas(
        deltas, replacement, baseline
    )
    assert utility == pytest.approx(0.15)
    assert all(gates.values())
    assert components["delta_log_equity_stress"] > 0.0

    deltas["stress_10bp"]["strict_mdd_pct"] = 0.02
    utility, gates, _ = lifecycle.switch_utility_from_deltas(
        deltas, replacement, baseline
    )
    assert utility < 0.0
    assert gates["stress_mdd_delta_le_0_01pp"] is False


def test_prompt_is_signal_time_only_and_identity_is_stable() -> None:
    prompt = lifecycle.lifecycle_prompt(
        {"a": 1.0}, {"capitulation": False}, candidate="SKIP"
    )
    assert "candidate_action: SKIP" in prompt
    assert "signal_time_state" in prompt
    assert "residual_util" not in prompt
    assert "return_pct" not in prompt
    assert lifecycle.lifecycle_identity("pre_2024", 7, "TP12") == (
        "pposm-lifecycle-residual|TP12|pre_2024|7"
    )


def test_build_is_train_only_and_writes_no_oos(monkeypatch, tmp_path) -> None:
    market = pd.DataFrame({"date": pd.date_range("2021-01-01", periods=5, freq="5min")})
    active = np.zeros(len(market), dtype=bool)
    active[0] = True
    engine = _Engine(
        market,
        {
            (0, 400): _trade(0, 1, 2, 1.01),
            (0, 1200): _trade(0, 1, 2, 1.02),
        },
    )
    manifest = _manifest()
    monkeypatch.setattr(lifecycle.frozen, "load_frozen_manifest", lambda path: (manifest, _Cfg()))
    monkeypatch.setattr(
        lifecycle,
        "load_train_context",
        lambda manifest, cfg: (market, pd.DataFrame(), _state(len(market)), active, engine),
    )

    summary = lifecycle.build(
        lifecycle.Config(
            manifest=tmp_path / "manifest.json",
            train_output=tmp_path / "train.jsonl",
            summary_output=tmp_path / "summary.json",
        )
    )

    assert summary["rows"]["oos"] == 0
    assert summary["causality"]["default_mode_train_only"] is True
    assert not (tmp_path / "oos.jsonl").exists()
    rows = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert {row["split"] for row in rows} == {"train"}
