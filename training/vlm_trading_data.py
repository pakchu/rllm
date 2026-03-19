"""VLM GRPO trading dataset builders and reward helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd
from PIL import Image

from models.option_b_vlm import (
    TradingPromptState,
    build_trading_prompt,
    get_action_labels,
    get_default_action_label,
    make_action_system_prompt,
    parse_action_label,
)
from preprocessing.chart_generator import ChartGenerator, ChartGeneratorConfig
from preprocessing.market_features import build_market_feature_frame
from preprocessing.scalars import compute_range_volatility_pct
from preprocessing.timeframe import make_window


@dataclass(frozen=True)
class VLMTrainingSample:
    """One training sample for VLM GRPO."""

    prompt: str
    image: Image.Image
    target_action: str
    next_return: float
    date: str
    action_utility_buy: float = 0.0
    action_utility_hold: float = 0.0
    action_utility_sell: float = 0.0
    dynamic_risk_weight: float = 0.0


def action_from_next_return(next_return: float, hold_band: float = 0.0005) -> str:
    """
    Convert next-step return into label.

    - return > +hold_band: BUY
    - return < -hold_band: SELL
    - otherwise: HOLD
    """
    if next_return > hold_band:
        return "BUY"
    if next_return < -hold_band:
        return "SELL"
    return "HOLD"


def action_from_utilities(
    utility_buy: float,
    utility_hold: float,
    utility_sell: float,
    hold_margin: float = 0.0,
) -> str:
    """Choose label from per-action utilities with optional hold dead-zone."""
    table = {
        "BUY": float(utility_buy),
        "HOLD": float(utility_hold),
        "SELL": float(utility_sell),
    }
    best_label = max(table, key=lambda k: table[k])
    best_value = float(table[best_label])
    hold_value = float(table["HOLD"])
    if best_label != "HOLD" and (best_value - hold_value) <= float(hold_margin):
        return "HOLD"
    return best_label


def action_from_trade_gate_next_return(
    next_return: float,
    hold_band: float = 0.0005,
) -> str:
    """Binary trade gate label from absolute horizon return."""
    return "TRADE" if abs(float(next_return)) > float(hold_band) else "NO_TRADE"


def action_from_trade_gate_utilities(
    utility_buy: float,
    utility_hold: float,
    utility_sell: float,
    hold_margin: float = 0.0,
) -> str:
    """Binary trade gate label from directional-vs-hold utility gap."""
    directional_best = max(float(utility_buy), float(utility_sell))
    hold_value = float(utility_hold)
    if (directional_best - hold_value) > float(hold_margin):
        return "TRADE"
    return "NO_TRADE"


def _window_drawdown_pct(window: pd.DataFrame) -> float:
    """Max drawdown over close series in [0, 1]."""
    if len(window) == 0:
        return 0.0
    closes = np.asarray(window["close"], dtype=np.float64)
    peak = np.maximum.accumulate(closes)
    peak = np.maximum(peak, 1e-12)
    drawdowns = np.maximum(0.0, 1.0 - closes / peak)
    return float(np.max(drawdowns))


def compute_dynamic_risk_weight(
    *,
    range_volatility_pct: float,
    trend_pct: float,
    drawdown_pct: float,
    base_risk_weight: float = 0.0,
    regime_weight_volatility: float = 0.0,
    regime_weight_downtrend: float = 0.0,
    regime_weight_drawdown: float = 0.0,
    min_risk_weight: float = 0.0,
    max_risk_weight: float = 1.0,
) -> float:
    """
    Dynamic risk weighting inspired by qr-dqn:
    increase risk aversion under high volatility, downtrend, and drawdown.
    """
    rw = float(base_risk_weight)
    vol_score = min(1.0, max(0.0, float(range_volatility_pct) / 0.03))
    downtrend_score = min(1.0, max(0.0, -float(trend_pct)) / 0.02)
    drawdown_score = min(1.0, max(0.0, float(drawdown_pct)) / 0.10)
    rw += float(regime_weight_volatility) * vol_score
    rw += float(regime_weight_downtrend) * downtrend_score
    rw += float(regime_weight_drawdown) * drawdown_score
    lo = float(min(min_risk_weight, max_risk_weight))
    hi = float(max(min_risk_weight, max_risk_weight))
    return float(np.clip(rw, lo, hi))


def _clip_trade_return(
    raw_return: float,
    *,
    stop_loss: float | None = None,
    take_profit: float | None = None,
) -> float:
    """Apply optional stop-loss / take-profit clipping to raw trade return."""
    ret = float(raw_return)
    if stop_loss is not None and stop_loss > 0.0:
        ret = max(ret, -float(stop_loss))
    if take_profit is not None and take_profit > 0.0:
        ret = min(ret, float(take_profit))
    return ret


def compute_action_utilities(
    *,
    open_t: float,
    open_th: float,
    horizon_min_low: float,
    horizon_max_high: float,
    fee_rate: float = 0.0005,
    slippage_rate: float = 0.0001,
    leverage: float = 1.0,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    use_log_return: bool = True,
    dynamic_risk_weight: float = 0.0,
    hold_reward_bias: float = 0.0,
) -> dict[str, float]:
    """
    Compute utility for BUY/HOLD/SELL with round-trip costs and downside risk.

    BUY/SELL assume independent flat->position->flat horizon trades.
    """
    if open_t <= 0.0:
        return {"BUY": float("-inf"), "HOLD": float(hold_reward_bias), "SELL": float("-inf")}

    # Horizon return from open_t to open_{t+h}
    raw_long = (float(open_th) - float(open_t)) / float(open_t)
    raw_short = -raw_long

    raw_long = _clip_trade_return(raw_long, stop_loss=stop_loss, take_profit=take_profit)
    raw_short = _clip_trade_return(raw_short, stop_loss=stop_loss, take_profit=take_profit)

    # Adverse excursion proxy over trade holding period.
    long_adverse = max(0.0, (float(open_t) - float(horizon_min_low)) / float(open_t))
    short_adverse = max(0.0, (float(horizon_max_high) - float(open_t)) / float(open_t))

    round_trip_cost = 2.0 * (float(fee_rate) + float(slippage_rate))
    lev = max(0.0, float(leverage))
    risk_w = max(0.0, float(dynamic_risk_weight))

    net_long = lev * raw_long - lev * round_trip_cost - risk_w * long_adverse
    net_short = lev * raw_short - lev * round_trip_cost - risk_w * short_adverse
    net_hold = float(hold_reward_bias)

    if use_log_return:
        eps = -0.999999
        u_buy = float(math.log1p(max(eps, net_long)))
        u_sell = float(math.log1p(max(eps, net_short)))
        u_hold = float(net_hold)
    else:
        u_buy = float(net_long)
        u_sell = float(net_short)
        u_hold = float(net_hold)

    return {"BUY": u_buy, "HOLD": u_hold, "SELL": u_sell}


def completion_text_to_label(completion) -> str:
    """Extract BUY/HOLD/SELL label from GRPO completion payload."""
    return completion_text_to_label_for_schema(completion, action_schema="buy_hold_sell")


def completion_text_to_label_for_schema(
    completion,
    *,
    action_schema: str = "buy_hold_sell",
) -> str:
    """Extract action label from GRPO completion payload for the chosen schema."""
    labels = get_action_labels(action_schema)
    default = get_default_action_label(action_schema)

    def _extract_texts(content):
        if isinstance(content, str):
            return [content]
        if isinstance(content, list):
            texts = []
            for c in content:
                if isinstance(c, dict):
                    if c.get("type") == "text" and c.get("text") is not None:
                        texts.append(str(c["text"]))
                    elif c.get("text") is not None:
                        texts.append(str(c["text"]))
            return texts
        return []

    if isinstance(completion, str):
        return parse_action_label(completion, default=default, labels=labels)
    if isinstance(completion, list) and completion:
        # conversational completion format
        assistant_texts = []
        fallback_texts = []
        for item in completion:
            if isinstance(item, dict):
                texts = _extract_texts(item.get("content"))
                if not texts:
                    continue
                if str(item.get("role", "")).lower() == "assistant":
                    assistant_texts.extend(texts)
                else:
                    fallback_texts.extend(texts)

        if assistant_texts:
            return parse_action_label(" ".join(assistant_texts), default=default, labels=labels)
        if fallback_texts:
            return parse_action_label(" ".join(fallback_texts), default=default, labels=labels)
    return default


def reward_from_action(
    predicted_action: str,
    target_action: str,
    next_return: float,
    hold_band: float = 0.0005,
    buy_reward_weight: float = 1.0,
    hold_reward_weight: float = 1.0,
    sell_reward_weight: float = 1.0,
    reward_mode: str = "classification",
    action_utility_buy: float | None = None,
    action_utility_hold: float | None = None,
    action_utility_sell: float | None = None,
    utility_reward_scale: float = 400.0,
    utility_gap_scale: float = 400.0,
    action_schema: str = "buy_hold_sell",
) -> float:
    """
    Reward action quality.

    Base:
      +1.0 for exact target match, -1.0 otherwise.
    Magnitude boost:
      larger |return| gives slightly larger absolute reward.
    """
    scale = 1.0 + min(abs(next_return) / max(hold_band, 1e-9), 5.0) * 0.1
    pred = str(predicted_action).upper()
    tgt = str(target_action).upper()
    schema = str(action_schema).strip().lower()

    class_weight_map = {
        "BUY": float(buy_reward_weight),
        "HOLD": float(hold_reward_weight),
        "SELL": float(sell_reward_weight),
    }
    if schema == "trade_gate":
        class_weight_map = {
            "TRADE": 0.5 * (float(buy_reward_weight) + float(sell_reward_weight)),
            "NO_TRADE": float(hold_reward_weight),
        }
    class_weight = class_weight_map.get(tgt, 1.0)

    mode = str(reward_mode).lower().strip()
    if mode not in {"classification", "utility"}:
        raise ValueError(
            "reward_mode must be one of {'classification','utility'}, "
            f"got {reward_mode}"
        )
    if mode == "utility":
        if (
            action_utility_buy is not None
            and action_utility_hold is not None
            and action_utility_sell is not None
        ):
            utility_table = {
                "BUY": float(action_utility_buy),
                "HOLD": float(action_utility_hold),
                "SELL": float(action_utility_sell),
            }
        else:
            # fallback: infer a simple utility proxy from next_return only
            utility_table = {
                "BUY": float(next_return),
                "HOLD": 0.0,
                "SELL": float(-next_return),
            }
        if schema == "trade_gate":
            utility_table = {
                "TRADE": float(max(utility_table["BUY"], utility_table["SELL"])),
                "NO_TRADE": float(utility_table["HOLD"]),
            }
        hold_key = "NO_TRADE" if schema == "trade_gate" else "HOLD"
        pred_u = float(utility_table.get(pred, utility_table[hold_key]))
        hold_u = float(utility_table[hold_key])
        best_u = float(max(utility_table.values()))
        utility_reward = float(utility_reward_scale) * (pred_u - hold_u)
        utility_regret = float(utility_gap_scale) * (best_u - pred_u)
        reward = utility_reward - utility_regret
        # give a fixed bonus when selecting utility-optimal action
        if abs(best_u - pred_u) <= 1e-12:
            reward += 1.0
        return float(reward * class_weight)

    if schema == "trade_gate":
        if pred == tgt:
            return float(1.0 * scale * class_weight)
        return float(-1.0 * scale * class_weight)

    if pred == tgt:
        return float(1.0 * scale * class_weight)

    # Asymmetric penalties to preserve learning signal when all samples are
    # "wrong" but with different severity:
    # - opposite direction should be worse than neutral HOLD miss
    # - for HOLD target, directional mistakes are equally penalized
    if tgt == "BUY":
        if pred == "SELL":
            return float(-1.5 * scale * class_weight)  # opposite direction
        return float(-0.5 * scale * class_weight)  # HOLD/unknown
    if tgt == "SELL":
        if pred == "BUY":
            return float(-1.5 * scale * class_weight)  # opposite direction
        return float(-0.5 * scale * class_weight)  # HOLD/unknown
    # tgt == HOLD
    return float(-1.0 * scale * class_weight)


def make_grpo_reward_func(
    hold_band: float = 0.0005,
    buy_reward_weight: float = 1.0,
    hold_reward_weight: float = 1.0,
    sell_reward_weight: float = 1.0,
    reward_mode: str = "classification",
    utility_reward_scale: float = 400.0,
    utility_gap_scale: float = 400.0,
    action_schema: str = "buy_hold_sell",
):
    """Build reward function compatible with TRL GRPOTrainer."""

    def _reward_func(
        completions,
        target_action,
        next_return,
        action_utility_buy=None,
        action_utility_hold=None,
        action_utility_sell=None,
        **kwargs,
    ):
        del kwargs
        rewards = []
        for i, (comp, tgt, ret) in enumerate(zip(completions, target_action, next_return)):
            pred = completion_text_to_label_for_schema(comp, action_schema=action_schema)
            buy_u = (
                None
                if action_utility_buy is None
                else float(action_utility_buy[i])
            )
            hold_u = (
                None
                if action_utility_hold is None
                else float(action_utility_hold[i])
            )
            sell_u = (
                None
                if action_utility_sell is None
                else float(action_utility_sell[i])
            )
            rewards.append(
                reward_from_action(
                    predicted_action=pred,
                    target_action=str(tgt),
                    next_return=float(ret),
                    hold_band=hold_band,
                    buy_reward_weight=buy_reward_weight,
                    hold_reward_weight=hold_reward_weight,
                    sell_reward_weight=sell_reward_weight,
                    reward_mode=reward_mode,
                    action_utility_buy=buy_u,
                    action_utility_hold=hold_u,
                    action_utility_sell=sell_u,
                    utility_reward_scale=utility_reward_scale,
                    utility_gap_scale=utility_gap_scale,
                    action_schema=action_schema,
                )
            )
        return rewards

    return _reward_func


def _chw_to_pil(chw: np.ndarray) -> Image.Image:
    hwc = np.clip(chw.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(hwc)


def _symbolic_market_labels(window: pd.DataFrame) -> tuple[str, str, str, str, float]:
    """Derive symbolic regime labels from a historical window."""
    if len(window) == 0:
        return "RANGE", "MID", "NEUTRAL", "WEAK", 0.0

    close_0 = float(window["close"].iloc[0])
    close_t = float(window["close"].iloc[-1])
    trend = 0.0 if close_0 == 0 else (close_t - close_0) / close_0
    abs_trend = abs(trend)

    if trend > 0.004:
        regime = "UPTREND"
    elif trend < -0.004:
        regime = "DOWNTREND"
    else:
        regime = "RANGE"

    if abs_trend >= 0.01:
        trend_strength = "STRONG"
    elif abs_trend >= 0.004:
        trend_strength = "MEDIUM"
    else:
        trend_strength = "WEAK"

    lookback = 12
    idx_ref = max(0, len(window) - 1 - lookback)
    close_ref = float(window["close"].iloc[idx_ref])
    momentum_ret = 0.0 if close_ref == 0 else (close_t - close_ref) / close_ref
    if momentum_ret > 0.003:
        momentum = "BULLISH"
    elif momentum_ret < -0.003:
        momentum = "BEARISH"
    else:
        momentum = "NEUTRAL"

    vol = float(compute_range_volatility_pct(window))
    if vol < 0.01:
        vol_level = "LOW"
    elif vol < 0.03:
        vol_level = "MID"
    else:
        vol_level = "HIGH"

    return regime, vol_level, momentum, trend_strength, vol


def _bucket_label(
    value: float,
    *,
    thresholds: tuple[float, ...],
    labels: tuple[str, ...],
) -> str:
    if len(labels) != len(thresholds) + 1:
        raise ValueError("labels length must equal len(thresholds) + 1")
    x = float(value)
    for threshold, label in zip(thresholds, labels):
        if x < float(threshold):
            return label
    return labels[-1]


def _trend_alignment_label(feature_row: pd.Series) -> str:
    sma12 = float(feature_row.get("sma12_ratio", 0.0))
    sma24 = float(feature_row.get("sma24_ratio", 0.0))
    sma48 = float(feature_row.get("sma48_ratio", 0.0))
    if sma12 > 0.0 and sma24 > 0.0 and sma48 > 0.0:
        return "BULL_STACK"
    if sma12 < 0.0 and sma24 < 0.0 and sma48 < 0.0:
        return "BEAR_STACK"
    if sma12 > 0.0 and sma48 < 0.0:
        return "BULL_REVERSAL"
    if sma12 < 0.0 and sma48 > 0.0:
        return "BEAR_REVERSAL"
    return "MIXED"


def _location_label(feature_row: pd.Series) -> str:
    bb_z = float(feature_row.get("bb_z", 0.0))
    range_pos = float(feature_row.get("range_pos", 0.0))
    score = 0.65 * bb_z + 0.75 * range_pos
    return _bucket_label(
        score,
        thresholds=(-1.6, -0.6, 0.6, 1.6),
        labels=(
            "EXTREME_DISCOUNT",
            "DISCOUNT",
            "NEAR_FAIR",
            "PREMIUM",
            "EXTREME_PREMIUM",
        ),
    )


def _oscillator_label(feature_row: pd.Series) -> str:
    rsi_norm = float(feature_row.get("rsi_norm", 0.0))
    mfi_norm = float(feature_row.get("mfi_norm", 0.0))
    score = 0.6 * rsi_norm + 0.4 * mfi_norm
    return _bucket_label(
        score,
        thresholds=(-0.65, -0.2, 0.2, 0.65),
        labels=("WASHOUT", "OVERSOLD", "NEUTRAL", "OVERBOUGHT", "BLOWOFF"),
    )


def _volume_state_label(feature_row: pd.Series) -> str:
    volume_z = float(feature_row.get("volume_zscore", 0.0))
    return _bucket_label(
        volume_z,
        thresholds=(-1.0, 0.75, 2.0),
        labels=("QUIET", "NORMAL", "SURGE", "CLIMAX"),
    )


def _candle_pattern_label(feature_row: pd.Series) -> str:
    body_to_range = float(feature_row.get("body_to_range", 0.0))
    body_ratio = float(feature_row.get("body_ratio", 0.0))
    upper_shadow = float(feature_row.get("upper_shadow", 0.0))
    lower_shadow = float(feature_row.get("lower_shadow", 0.0))
    if body_to_range < 0.15:
        return "DOJI"
    if lower_shadow > upper_shadow * 1.5 and body_ratio >= 0.0:
        return "BULL_REJECTION"
    if upper_shadow > lower_shadow * 1.5 and body_ratio <= 0.0:
        return "BEAR_REJECTION"
    if body_to_range >= 0.55 and body_ratio > 0.0:
        return "BULL_IMPULSE"
    if body_to_range >= 0.55 and body_ratio < 0.0:
        return "BEAR_IMPULSE"
    return "BALANCED"


def _order_flow_label(feature_row: pd.Series, *, has_taker_flow: bool) -> str:
    if not has_taker_flow:
        return "UNKNOWN"
    taker_imbalance = float(feature_row.get("taker_imbalance", 0.0))
    return _bucket_label(
        taker_imbalance,
        thresholds=(-0.2, 0.2),
        labels=("SELLER_DOMINANT", "BALANCED", "BUYER_DOMINANT"),
    )


def _risk_state_label(feature_row: pd.Series) -> str:
    score = (
        18.0 * float(feature_row.get("range_vol", 0.0))
        + 5.0 * float(feature_row.get("window_drawdown", 0.0))
        + 0.75 * abs(float(feature_row.get("return_zscore_48", 0.0)))
    )
    return _bucket_label(
        score,
        thresholds=(0.45, 0.95, 1.5),
        labels=("CALM", "NORMAL", "ELEVATED", "STRESS"),
    )


def _engineered_prompt_features(
    window: pd.DataFrame,
    feature_row: pd.Series,
) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, str], ...], tuple[str, ...]]:
    numeric_features: list[tuple[str, float]] = [
        ("Close Z-Score (48)", float(feature_row.get("close_zscore_48", 0.0))),
        ("Return Z-Score (48)", float(feature_row.get("return_zscore_48", 0.0))),
        ("Momentum 1h (%)", float(feature_row.get("trend_12", 0.0))),
        ("Momentum 2h (%)", float(feature_row.get("trend_24", 0.0))),
        ("Momentum 8h (%)", float(feature_row.get("trend_96", 0.0))),
        ("BB Z-Score", float(feature_row.get("bb_z", 0.0))),
        ("Range Position", float(feature_row.get("range_pos", 0.0))),
        ("RSI14", float((float(feature_row.get("rsi_norm", 0.0)) + 1.0) * 50.0)),
        ("MFI14", float((float(feature_row.get("mfi_norm", 0.0)) + 1.0) * 50.0)),
        ("Volume Z-Score", float(feature_row.get("volume_zscore", 0.0))),
        ("Body/Range", float(feature_row.get("body_to_range", 0.0))),
        ("Window Drawdown", float(feature_row.get("window_drawdown", 0.0))),
    ]
    if "taker_buy_base" in window.columns:
        numeric_features.append(
            ("Taker Buy Ratio", float(feature_row.get("taker_buy_ratio", 0.5)))
        )
    if "number_of_trades" in window.columns:
        numeric_features.append(
            ("Trades Ratio", float(feature_row.get("trades_ratio", 0.0)))
        )
    if "funding_rate" in window.columns:
        numeric_features.append(
            ("Funding Rate", float(feature_row.get("funding_rate", 0.0)))
        )
    if "open_interest" in window.columns:
        numeric_features.append(
            ("Open Interest Z-Score", float(feature_row.get("oi_zscore", 0.0)))
        )

    trend_alignment = _trend_alignment_label(feature_row)
    location = _location_label(feature_row)
    oscillator = _oscillator_label(feature_row)
    volume_state = _volume_state_label(feature_row)
    candle_pattern = _candle_pattern_label(feature_row)
    order_flow = _order_flow_label(
        feature_row, has_taker_flow="taker_buy_base" in window.columns
    )
    risk_state = _risk_state_label(feature_row)

    symbolic_features = (
        ("Trend Alignment", trend_alignment),
        ("Location", location),
        ("Oscillator", oscillator),
        ("Volume State", volume_state),
        ("Candle Pattern", candle_pattern),
        ("Order Flow", order_flow),
        ("Risk State", risk_state),
    )
    tags = tuple(
        tag
        for tag in (
            trend_alignment,
            location,
            oscillator,
            volume_state,
            order_flow,
            risk_state,
        )
        if tag != "UNKNOWN"
    )
    return tuple(numeric_features), symbolic_features, tags


def build_vlm_training_samples(
    market_df: pd.DataFrame,
    timeframe: str = "1m",
    window_size: int = 96,
    resolution: int = 320,
    cache_dir: str | None = None,
    hold_band: float = 0.0005,
    target_horizon: int = 1,
    label_mode: str = "next_return",
    utility_hold_margin: float = 0.0,
    utility_fee_rate: float = 0.0005,
    utility_slippage_rate: float = 0.0001,
    utility_leverage: float = 1.0,
    utility_stop_loss: float | None = None,
    utility_take_profit: float | None = None,
    utility_use_log_return: bool = True,
    utility_base_risk_weight: float = 0.0,
    utility_regime_weight_volatility: float = 0.0,
    utility_regime_weight_downtrend: float = 0.0,
    utility_regime_weight_drawdown: float = 0.0,
    utility_min_risk_weight: float = 0.0,
    utility_max_risk_weight: float = 1.0,
    utility_hold_reward_bias: float = 0.0,
    max_samples: int | None = None,
    sample_mode: str = "sequential",
    sample_seed: int = 42,
    prompt_style: str = "numeric",
    prompt_feature_mode: str = "basic_v0",
    action_schema: str = "buy_hold_sell",
) -> List[VLMTrainingSample]:
    """
    Build trading-derived VLM samples with image + prompt + target action.

    Uses only historical window `[t-w+1, t]` for prompt/image.
    Target action is derived by:
      - label_mode='next_return': horizon return sign
      - label_mode='utility': cost/risk-aware action utility argmax
    where horizon return uses `(open_{t+h} - open_t)/open_t`.
    """
    renderer = ChartGenerator(
        ChartGeneratorConfig(
            resolution=resolution,
            cache_dir=cache_dir,
            show_indicators=True,
            show_oscillators=True,
        )
    )

    horizon = max(1, int(target_horizon))
    start_t = window_size - 1
    end_t = len(market_df) - horizon  # because we access t+h
    if end_t <= start_t:
        return []

    sample_mode_key = str(sample_mode).lower().strip()
    if sample_mode_key not in {"sequential", "random", "balanced", "uniform"}:
        raise ValueError(
            "sample_mode must be one of {'sequential','random','balanced','uniform'}, "
            f"got {sample_mode}"
        )
    label_mode_key = str(label_mode).lower().strip()
    if label_mode_key not in {"next_return", "utility"}:
        raise ValueError(
            "label_mode must be one of {'next_return','utility'}, "
            f"got {label_mode}"
        )
    prompt_feature_mode_key = str(prompt_feature_mode).lower().strip()
    if prompt_feature_mode_key not in {"basic_v0", "engineered_v1"}:
        raise ValueError(
            "prompt_feature_mode must be one of {'basic_v0','engineered_v1'}, "
            f"got {prompt_feature_mode}"
        )
    action_schema_key = str(action_schema).lower().strip()
    get_action_labels(action_schema_key)

    feature_frame = build_market_feature_frame(market_df, window_size=window_size)

    # First pass: compute candidate metadata without rendering images.
    candidates = []
    for t in range(start_t, end_t):
        open_t = float(market_df.loc[t, "open"])
        open_th = float(market_df.loc[t + horizon, "open"])
        next_return = 0.0 if open_t == 0 else (open_th - open_t) / open_t
        window = make_window(market_df, t=t, w=window_size)
        feature_row = feature_frame.iloc[t]
        trend_pct = float(feature_row.get("trend_96", 0.0))
        vol_pct = float(feature_row.get("range_vol", compute_range_volatility_pct(window)))
        drawdown_pct = float(feature_row.get("window_drawdown", _window_drawdown_pct(window)))
        risk_weight = compute_dynamic_risk_weight(
            range_volatility_pct=vol_pct,
            trend_pct=trend_pct,
            drawdown_pct=drawdown_pct,
            base_risk_weight=utility_base_risk_weight,
            regime_weight_volatility=utility_regime_weight_volatility,
            regime_weight_downtrend=utility_regime_weight_downtrend,
            regime_weight_drawdown=utility_regime_weight_drawdown,
            min_risk_weight=utility_min_risk_weight,
            max_risk_weight=utility_max_risk_weight,
        )
        horizon_slice = market_df.iloc[t + 1 : t + horizon + 1]
        if len(horizon_slice) > 0:
            horizon_min_low = float(horizon_slice["low"].min())
            horizon_max_high = float(horizon_slice["high"].max())
        else:
            horizon_min_low = open_t
            horizon_max_high = open_t
        utilities = compute_action_utilities(
            open_t=open_t,
            open_th=open_th,
            horizon_min_low=horizon_min_low,
            horizon_max_high=horizon_max_high,
            fee_rate=utility_fee_rate,
            slippage_rate=utility_slippage_rate,
            leverage=utility_leverage,
            stop_loss=utility_stop_loss,
            take_profit=utility_take_profit,
            use_log_return=utility_use_log_return,
            dynamic_risk_weight=risk_weight,
            hold_reward_bias=utility_hold_reward_bias,
        )
        if action_schema_key == "trade_gate":
            if label_mode_key == "utility":
                target_action = action_from_trade_gate_utilities(
                    utility_buy=utilities["BUY"],
                    utility_hold=utilities["HOLD"],
                    utility_sell=utilities["SELL"],
                    hold_margin=utility_hold_margin,
                )
            else:
                target_action = action_from_trade_gate_next_return(
                    next_return,
                    hold_band=hold_band,
                )
        else:
            if label_mode_key == "utility":
                target_action = action_from_utilities(
                    utility_buy=utilities["BUY"],
                    utility_hold=utilities["HOLD"],
                    utility_sell=utilities["SELL"],
                    hold_margin=utility_hold_margin,
                )
            else:
                target_action = action_from_next_return(next_return, hold_band=hold_band)
        candidates.append(
            (
                t,
                target_action,
                float(next_return),
                float(utilities["BUY"]),
                float(utilities["HOLD"]),
                float(utilities["SELL"]),
                float(risk_weight),
            )
        )

    rng = np.random.default_rng(sample_seed)
    positions = np.arange(len(candidates))
    if max_samples is None or max_samples >= len(candidates):
        selected_pos = positions
    elif sample_mode_key == "sequential":
        selected_pos = positions[: max_samples]
    elif sample_mode_key == "uniform":
        # Evenly cover the full candidate period, including both endpoints,
        # instead of taking only the earliest chunk.
        n = int(len(candidates))
        m = int(max_samples)
        selected_pos = np.rint(np.linspace(0, n - 1, num=m)).astype(np.int64)
        selected_pos = np.clip(selected_pos, 0, n - 1)
    elif sample_mode_key == "random":
        selected_pos = np.sort(rng.choice(positions, size=max_samples, replace=False))
    else:  # balanced
        labels = np.array([candidates[i][1] for i in positions], dtype=object)
        selected_chunks = []
        schema_labels = get_action_labels(action_schema_key)
        per_class = max(1, max_samples // max(1, len(schema_labels)))
        for label in schema_labels:
            idx = positions[labels == label]
            if len(idx) == 0:
                continue
            take = min(per_class, len(idx))
            chosen = rng.choice(idx, size=take, replace=False)
            selected_chunks.append(chosen)
        selected_pos = (
            np.concatenate(selected_chunks)
            if selected_chunks
            else np.asarray([], dtype=np.int64)
        )
        if len(selected_pos) < max_samples:
            remaining_needed = max_samples - len(selected_pos)
            pool = np.setdiff1d(positions, selected_pos, assume_unique=False)
            if len(pool) > 0:
                extra_take = min(remaining_needed, len(pool))
                extra = rng.choice(pool, size=extra_take, replace=False)
                selected_pos = np.concatenate([selected_pos, extra])
        selected_pos = np.sort(selected_pos)

    samples: List[VLMTrainingSample] = []
    for pos in selected_pos:
        (
            t,
            target_action,
            next_return,
            action_utility_buy,
            action_utility_hold,
            action_utility_sell,
            dynamic_risk_weight,
        ) = candidates[int(pos)]
        window = make_window(market_df, t=t, w=window_size)
        image_chw = renderer.render_window(window)
        image = _chw_to_pil(image_chw)

        regime, vol_level, momentum, trend_strength, window_vol = _symbolic_market_labels(window)
        feature_row = feature_frame.iloc[t]
        extra_numeric_features: tuple[tuple[str, float], ...] = ()
        extra_symbolic_features: tuple[tuple[str, str], ...] = ()
        context_tags: tuple[str, ...] = ()
        if prompt_feature_mode_key == "engineered_v1":
            (
                extra_numeric_features,
                extra_symbolic_features,
                context_tags,
            ) = _engineered_prompt_features(window, feature_row)
        state = TradingPromptState(
            timeframe=timeframe,
            position_size_pct=0.0,
            last_entry_price=0.0,
            range_volatility_pct=window_vol,
            regime_label=regime,
            volatility_label=vol_level,
            momentum_label=momentum,
            trend_strength_label=trend_strength,
            extra_numeric_features=extra_numeric_features,
            extra_symbolic_features=extra_symbolic_features,
            context_tags=context_tags,
        )
        prompt = build_trading_prompt(
            state,
            prompt_style=prompt_style,
            action_schema=action_schema_key,
        )

        samples.append(
            VLMTrainingSample(
                prompt=prompt,
                image=image,
                target_action=target_action,
                next_return=float(next_return),
                date=str(pd.to_datetime(market_df.loc[t, "date"])),
                action_utility_buy=float(action_utility_buy),
                action_utility_hold=float(action_utility_hold),
                action_utility_sell=float(action_utility_sell),
                dynamic_risk_weight=float(dynamic_risk_weight),
            )
        )

    return samples


def samples_to_hf_records(
    samples: Iterable[VLMTrainingSample],
    *,
    action_schema: str = "buy_hold_sell",
) -> list[dict]:
    """Convert samples into Hugging Face Dataset records."""
    system_prompt = make_action_system_prompt(action_schema)
    records = []
    for s in samples:
        records.append(
            {
                # TRL multimodal GRPO expects conversational prompts (not pre-templated strings).
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": s.prompt},
                ],
                "image": s.image,
                "target_action": s.target_action,
                "next_return": float(s.next_return),
                "action_utility_buy": float(s.action_utility_buy),
                "action_utility_hold": float(s.action_utility_hold),
                "action_utility_sell": float(s.action_utility_sell),
                "dynamic_risk_weight": float(s.dynamic_risk_weight),
                "date": s.date,
            }
        )
    return records
