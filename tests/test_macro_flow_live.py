from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from execution import macro_flow
from execution.approved_portfolio_signals import build_macro_targets
from execution.portfolio_live import (
    _score_sleeve,
    _validate_portfolio_mode,
    _validate_signed_net_approval,
)
from execution.signed_net_hedge import parse_policy
from execution.wave_execution import WaveExecutionConfig


def _market(periods: int = 18_024) -> pd.DataFrame:
    dates = pd.date_range("2026-06-01", periods=periods, freq="5min", tz="UTC")
    t = np.arange(periods, dtype=float)
    close = 100_000.0 * np.exp(0.00001 * t + 0.002 * np.sin(t / 200.0))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "date": dates.tz_convert(None),
            "open": open_,
            "high": np.maximum(open_, close) * 1.001,
            "low": np.minimum(open_, close) * 0.999,
            "close": close,
            "volume": 1.0,
            "quote_asset_volume": 1_000.0,
            "taker_buy_quote": np.where((t // 12) % 24 < 12, 530.0, 470.0),
            "dxy": 100.0 * np.exp(-0.00001 * t),
            "dxy_available": 1.0,
        }
    )


def test_macro_runtime_contract_is_hash_bound_to_approval_and_source():
    path = Path("configs/live/macro_flow_runtime_2026-09-07.json")
    config = json.loads(path.read_text())
    macro_flow.validate_macro_flow_runtime_config(config, configured_side="AUTO")
    changed = json.loads(path.read_text())
    changed["dollar_component_weight"] = 0.7
    with pytest.raises(macro_flow.MacroFlowContractError, match="component_weight"):
        macro_flow.validate_macro_flow_runtime_config(changed, configured_side="AUTO")


def test_portfolio_scorer_emits_exact_approved_hourly_macro_target():
    market = _market()
    decision = pd.Timestamp(market.iloc[-1]["date"], tz="UTC")
    # Trim to an hourly execution slot: decision HH:00 -> execution HH:05.
    decision = decision.floor("1h")
    market = market.loc[pd.to_datetime(market["date"], utc=True) <= decision].copy()
    features = pd.DataFrame(index=market.index)
    sleeve = {
        "name": "macro_flow",
        "source": "configs/live/macro_flow_runtime_2026-09-07.json",
        "side": "AUTO",
        "weight": 1.0,
    }
    score = _score_sleeve(
        sleeve=sleeve,
        enriched=market,
        features=features,
        exec_cfg=WaveExecutionConfig(interval_minutes=5),
        asof=decision + pd.Timedelta(minutes=4),
    )
    execution = decision + pd.Timedelta(minutes=5)
    expected = build_macro_targets(market, asof=execution).loc[execution]
    assert score["kind"] == "target"
    assert score["emit"] is True
    assert score["ready"] is True
    assert score["target_fraction"] == pytest.approx(float(expected))
    assert score["signal_id"] == f"macro_flow:{execution.isoformat()}"


def test_macro_scorer_catches_up_latest_hourly_target_after_missed_slot():
    market = _market()
    decision = pd.Timestamp(market.iloc[-1]["date"], tz="UTC").floor("1h")
    market = market.loc[pd.to_datetime(market["date"], utc=True) <= decision].copy()
    next_bar = market.iloc[[-1]].copy()
    next_bar["date"] = decision.tz_localize(None) + pd.Timedelta(minutes=5)
    shifted = pd.concat([market, next_bar], ignore_index=True)
    score = _score_sleeve(
        sleeve={
            "name": "macro_flow",
            "source": "configs/live/macro_flow_runtime_2026-09-07.json",
            "side": "AUTO",
            "weight": 1.0,
        },
        enriched=shifted,
        features=pd.DataFrame(index=shifted.index),
        exec_cfg=WaveExecutionConfig(interval_minutes=5),
        asof=decision + pd.Timedelta(minutes=9),
    )
    source_execution = decision + pd.Timedelta(minutes=5)
    current_execution = decision + pd.Timedelta(minutes=10)
    expected = build_macro_targets(shifted, asof=current_execution).loc[
        source_execution
    ]
    assert score["kind"] == "target"
    assert score["emit"] is True
    assert score["ready"] is True
    assert score["target_fraction"] == pytest.approx(float(expected))
    assert score["signal_id"] == f"macro_flow:{source_execution.isoformat()}"
    assert score["execution_time"] == current_execution.isoformat()
    assert score["policy_metadata"]["source_execution_time"] == (
        source_execution.isoformat()
    )
    assert "macro_flow=target_catch_up" in score["reasons"]


def test_full_runtime_portfolio_matches_approved_nonzero_weights():
    portfolio = json.loads(
        Path(
            "configs/live/portfolio_g9_macro1_dollar_short05_signed_net_runtime_ready_2026-09-07.json"
        ).read_text()
    )
    policy = parse_policy(portfolio)
    assert policy is not None
    _validate_signed_net_approval(portfolio, policy=policy, live=False)
    assert sum(policy.weights.values()) == 6
    changed = json.loads(json.dumps(portfolio))
    changed["base_sleeves"][0]["weight"] = 1.1
    changed_policy = parse_policy(changed)
    with pytest.raises(RuntimeError, match="weights differ"):
        _validate_signed_net_approval(changed, policy=changed_policy, live=False)


def test_superseded_unrefined_portfolio_is_rejected_for_live_orders():
    portfolio = json.loads(
        Path(
            "configs/live/portfolio_g9_macro1_dollar_short05_signed_net_mainnet_live_2026-09-07.json"
        ).read_text()
    )
    policy = parse_policy(portfolio)
    assert policy is not None
    with pytest.raises(RuntimeError, match="not authorized for live orders"):
        _validate_portfolio_mode(portfolio, live=True)


def test_refined_g9_live_portfolio_excludes_retired_unrefined_alphas():
    portfolio = json.loads(
        Path(
            "configs/live/portfolio_g9_refined_only_signed_net_mainnet_live_2026-09-09.json"
        ).read_text()
    )
    policy = parse_policy(portfolio)
    assert policy is not None
    _validate_portfolio_mode(portfolio, live=True)
    _validate_signed_net_approval(portfolio, policy=policy, live=True)
    assert dict(policy.weights) == {
        "fresh_kimchi_fx": Decimal("1.0"),
        "frozen_annual_rank7": Decimal("1.5"),
        "rex_taker_low_range_position": Decimal("0.2"),
        "cand_rex_veto_7": Decimal("0.8"),
        "markov_transition_long": Decimal("1.0"),
    }
    assert sum(policy.weights.values()) == Decimal("4.5")
    assert "macro_flow" not in policy.weights
    assert "dollar_rally_short" not in policy.weights
