from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from training import build_pposm_sft_rlvr_data as builder
from training.search_inventory_purge_reclaim_alpha import Trade
from training.search_pullback_premium_overheat_state_machine_alpha import SPEC, Config


def _trade(signal: int = 1, *, factor: float = 1.02) -> Trade:
    return Trade(
        signal_position=signal,
        entry_position=signal + 1,
        exit_position=signal + 2,
        side=1,
        gross_return=0.04,
        price_factor=factor,
        funding_factor=0.999,
        funding_debit_factor=0.999,
        favorable_price_factor=1.02,
        adverse_price_factor=0.99,
        entry_date=str(pd.Timestamp("2023-01-01") + pd.Timedelta(minutes=5 * (signal + 1))),
    )


def _manifest() -> dict:
    thresholds = {
        "fit_active_events": 408,
        "htf_1w_return_1_q50": 0.1,
        "rex_576_range_width_pct_q50": 0.2,
        "quote_vol_z_1d_q20": -0.3,
        "premium_index_change_q67": 0.4,
        "rex_576_range_pos_q67": 0.5,
    }
    return {"spec": SPEC, "state_thresholds": thresholds}


def _frames(rows: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    market = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=rows, freq="5min")})
    state = pd.DataFrame(
        {
            name: [float(index + 1)] * rows
            for index, name in enumerate(builder.pposm.FEATURE_QUANTILES)
        }
    )
    state["capitulation"] = False
    state["overheat"] = False
    return market, state


def test_prompt_is_signal_time_only_and_oos_utility_is_label_only() -> None:
    market, state = _frames()
    trade = _trade()
    cfg = Config()
    row = builder.build_row(
        split="oos",
        window="test_2024",
        trade=trade,
        market=market,
        state=state,
        manifest=_manifest(),
        strategy_cfg=cfg,
    )

    assert "signal_features:" in row["prompt"]
    assert "frozen_state: normal" in row["prompt"]
    assert "frozen_take_profit_bps: 1200" in row["prompt"]
    assert "frozen_hold_bars: 576" in row["prompt"]
    assert "net_return" not in row["prompt"]
    assert "exit_time" not in row["prompt"]
    assert "net_return" in row["metadata"]
    assert row["metadata"]["offline_label_only"] is True
    assert row["metadata"]["utility_available_for_training"] is False
    assert row["metadata"]["clock"]["entry_position"] == trade.signal_position + 1


def test_exact_base_cost_factor_controls_target_and_train_reward() -> None:
    market, state = _frames()
    cfg = replace(Config(), leverage=0.5, fee_rate=0.0005, slippage_rate=0.0001)
    trade = _trade(factor=1.002)
    expected = (1.0 - 0.5 * 0.0006) ** 2 * 1.002 * 0.999
    assert builder.exact_base_cost_net_factor(trade, cfg) == pytest.approx(expected)

    row = builder.build_row(
        split="train",
        window="pre_2024",
        trade=trade,
        market=market,
        state=state,
        manifest=_manifest(),
        strategy_cfg=cfg,
    )
    assert row["target"] == ("TRADE" if expected > 1.0 else "NO_TRADE")
    assert row["metadata"]["net_return"] == pytest.approx(expected - 1.0)


def test_no_replacement_identity_rejects_cross_window_overlap() -> None:
    schedules = {window: [] for _, window, _, _ in builder.SPLIT_WINDOWS}
    schedules["pre_2024"] = [_trade(1)]
    schedules["test_2024"] = [_trade(5)]
    builder.assert_no_replacement_identity(schedules)

    schedules["test_2024"] = [_trade(2)]
    with pytest.raises(RuntimeError, match="overlap"):
        builder.assert_no_replacement_identity(schedules)


def test_rows_are_byte_deterministic_and_preserve_schedule_identity() -> None:
    market, state = _frames()
    schedules = {window: [] for _, window, _, _ in builder.SPLIT_WINDOWS}
    schedules["pre_2024"] = [_trade(1)]
    schedules["test_2024"] = [_trade(5)]
    cfg = Config()

    first = builder.rows_from_schedules(market, state, schedules, _manifest(), cfg)
    second = builder.rows_from_schedules(market, state, schedules, _manifest(), cfg)
    assert builder._jsonl_bytes(first[0] + first[1]) == builder._jsonl_bytes(second[0] + second[1])
    identities = [row["metadata"]["identity"] for rows in first for row in rows]
    assert identities == [
        builder.trade_identity("pre_2024", schedules["pre_2024"][0]),
        builder.trade_identity("test_2024", schedules["test_2024"][0]),
    ]
