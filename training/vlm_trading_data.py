"""VLM GRPO trading dataset builders and reward helpers."""

from __future__ import annotations

import math
import json
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
from training.path_outcome_dataset import PathOutcomeConfig, compute_trade_path_outcome


@dataclass(frozen=True)
class VLMTrainingSample:
    """One training sample for VLM/text GRPO."""

    prompt: str
    image: Image.Image | None
    target_action: str
    next_return: float
    date: str
    action_utility_buy: float = 0.0
    action_utility_hold: float = 0.0
    action_utility_sell: float = 0.0
    action_utility_map: dict[str, float] | None = None
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


def action_from_trade_side_next_return(
    next_return: float,
    hold_band: float = 0.0005,
) -> str | None:
    """Directional trade-side label; returns None when no trade should be taken."""
    ret = float(next_return)
    if abs(ret) <= float(hold_band):
        return None
    return "LONG" if ret > 0.0 else "SHORT"


def action_from_trade_side_utilities(
    utility_buy: float,
    utility_hold: float,
    utility_sell: float,
    hold_margin: float = 0.0,
) -> str | None:
    """Directional trade-side label; returns None when hold dominates directional utility."""
    if action_from_trade_gate_utilities(
        utility_buy=utility_buy,
        utility_hold=utility_hold,
        utility_sell=utility_sell,
        hold_margin=hold_margin,
    ) != "TRADE":
        return None
    return "LONG" if float(utility_buy) >= float(utility_sell) else "SHORT"


def parse_multi_horizon_bars(value: str | Iterable[int] | None) -> tuple[int, ...]:
    """Parse comma/space separated hold-bar horizons for multi-horizon actions."""
    if value is None:
        return (36, 72, 144)
    if isinstance(value, str):
        raw_items = [x.strip() for part in value.split(",") for x in part.split()]
        vals = [int(x) for x in raw_items if x]
    else:
        vals = [int(x) for x in value]
    vals = sorted({int(x) for x in vals if int(x) > 0})
    if not vals:
        raise ValueError("multi_horizon_bars must contain at least one positive integer")
    allowed = {36, 72, 144}
    unsupported = set(vals).difference(allowed)
    if unsupported:
        raise ValueError(
            "multi_horizon_bars currently supports only schema labels "
            f"{sorted(allowed)}, got unsupported {sorted(unsupported)}"
        )
    return tuple(vals)


def action_from_multi_horizon_path_outcomes(
    market_df: pd.DataFrame,
    signal_pos: int,
    *,
    hold_bars_list: tuple[int, ...] = (36, 72, 144),
    path_entry_delay_bars: int = 1,
    utility_fee_rate: float = 0.0005,
    utility_slippage_rate: float = 0.0001,
    utility_leverage: float = 1.0,
    path_mae_penalty: float = 1.0,
    path_mfe_bonus: float = 0.0,
    utility_hold_margin: float = 0.0,
    path_min_net_return: float = 0.0,
    path_max_mae: float = 1.0,
) -> tuple[str, dict[str, float], float]:
    """Choose NO_TRADE or best side+horizon from executable path outcomes."""
    best_label = "NO_TRADE"
    best_utility = float(utility_hold_margin)
    best_net_return = 0.0
    utilities: dict[str, float] = {"NO_TRADE": 0.0}
    for hold_bars in hold_bars_list:
        cfg = PathOutcomeConfig(
            hold_bars=int(hold_bars),
            entry_delay_bars=path_entry_delay_bars,
            fee_rate=utility_fee_rate,
            slippage_rate=utility_slippage_rate,
            leverage=utility_leverage,
            mae_penalty=path_mae_penalty,
            mfe_bonus=path_mfe_bonus,
            hold_margin=utility_hold_margin,
            min_net_return=path_min_net_return,
            max_mae=path_max_mae,
        )
        for side in ("LONG", "SHORT"):
            outcome = compute_trade_path_outcome(market_df, signal_pos, side, cfg)
            if outcome is None:
                continue
            label = f"{side}_{int(hold_bars)}"
            utilities[label] = float(outcome.utility)
            trade_ok = (
                float(outcome.utility) > float(utility_hold_margin)
                and float(outcome.net_return) > float(path_min_net_return)
                and float(outcome.mae) <= float(path_max_mae)
            )
            if trade_ok and float(outcome.utility) > best_utility:
                best_label = label
                best_utility = float(outcome.utility)
                best_net_return = float(outcome.net_return)
    return best_label, utilities, best_net_return


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
    action_utility_map: dict[str, float] | None = None,
    utility_reward_scale: float = 400.0,
    utility_gap_scale: float = 400.0,
    wrong_trade_penalty: float = 0.0,
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
    elif schema == "trade_side":
        class_weight_map = {
            "LONG": float(buy_reward_weight),
            "SHORT": float(sell_reward_weight),
        }
    elif schema == "multi_horizon_side":
        class_weight_map = {"NO_TRADE": float(hold_reward_weight)}
        for label in get_action_labels(schema):
            if label.startswith("LONG_"):
                class_weight_map[label] = float(buy_reward_weight)
            elif label.startswith("SHORT_"):
                class_weight_map[label] = float(sell_reward_weight)
    class_weight = class_weight_map.get(tgt, 1.0)

    mode = str(reward_mode).lower().strip()
    if mode not in {"classification", "utility"}:
        raise ValueError(
            "reward_mode must be one of {'classification','utility'}, "
            f"got {reward_mode}"
        )
    if mode == "utility":
        if schema == "multi_horizon_side" and action_utility_map is not None:
            utility_table = {
                str(k).upper(): float(v) for k, v in action_utility_map.items()
            }
            hold_u = float(utility_table.get("NO_TRADE", 0.0))
            pred_u = float(utility_table.get(pred, hold_u))
            best_u = float(max(utility_table.values())) if utility_table else hold_u
            utility_reward = float(utility_reward_scale) * (pred_u - hold_u)
            utility_regret = float(utility_gap_scale) * (best_u - pred_u)
            reward = utility_reward - utility_regret
            if pred == tgt:
                reward += 1.0
            return float(reward * class_weight)
        if schema == "multi_horizon_side":
            if pred == tgt:
                return float(1.0 * scale * class_weight)
            return float(-1.0 * scale * class_weight)
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
        if schema == "trade_side":
            hold_u = float(action_utility_hold if action_utility_hold is not None else 0.0)
            utility_table = {
                "LONG": float(action_utility_buy if action_utility_buy is not None else next_return),
                "SHORT": float(action_utility_sell if action_utility_sell is not None else -next_return),
            }
            pred_u = float(utility_table.get(pred, hold_u))
        else:
            hold_key = "NO_TRADE" if schema == "trade_gate" else "HOLD"
            pred_u = float(utility_table.get(pred, utility_table[hold_key]))
            hold_u = float(utility_table[hold_key])
        best_u = float(max(utility_table.values()))
        utility_reward = float(utility_reward_scale) * (pred_u - hold_u)
        utility_regret = float(utility_gap_scale) * (best_u - pred_u)
        reward = utility_reward - utility_regret
        if float(wrong_trade_penalty) > 0.0:
            if schema == "buy_hold_sell":
                pred_is_trade = pred in {"BUY", "SELL"}
                target_is_trade = tgt in {"BUY", "SELL"}
                if pred_is_trade and tgt == "HOLD":
                    reward -= float(wrong_trade_penalty)
                elif pred_is_trade and target_is_trade and pred != tgt:
                    reward -= float(wrong_trade_penalty)
            elif schema == "trade_gate" and pred == "TRADE" and tgt == "NO_TRADE":
                reward -= float(wrong_trade_penalty)
        # give a fixed bonus when selecting utility-optimal action
        if abs(best_u - pred_u) <= 1e-12:
            reward += 1.0
        return float(reward * class_weight)

    if schema == "trade_gate":
        if pred == tgt:
            return float(1.0 * scale * class_weight)
        return float(-1.0 * scale * class_weight)
    if schema == "trade_side":
        if pred == tgt:
            return float(1.0 * scale * class_weight)
        return float(-1.0 * scale * class_weight)
    if schema == "multi_horizon_side":
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
    wrong_trade_penalty: float = 0.0,
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
        action_utility_map=None,
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
            utility_map = None
            if action_utility_map is not None:
                raw_map = action_utility_map[i]
                if isinstance(raw_map, str):
                    try:
                        utility_map = json.loads(raw_map)
                    except json.JSONDecodeError:
                        utility_map = None
                elif isinstance(raw_map, dict):
                    utility_map = raw_map
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
                    action_utility_map=utility_map,
                    utility_reward_scale=utility_reward_scale,
                    utility_gap_scale=utility_gap_scale,
                    wrong_trade_penalty=wrong_trade_penalty,
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


def _macro_state_label(feature_row: pd.Series) -> str:
    dxy_z = float(feature_row.get("dxy_zscore", 0.0))
    dxy_mom = float(feature_row.get("dxy_momentum", 0.0))
    usdkrw_z = float(feature_row.get("usdkrw_zscore", 0.0))
    if dxy_z > 1.0 or dxy_mom > 0.01 or usdkrw_z > 1.25:
        return "DOLLAR_STRENGTH"
    if dxy_z < -1.0 or dxy_mom < -0.01 or usdkrw_z < -1.25:
        return "DOLLAR_WEAKNESS"
    return "MACRO_NEUTRAL"


def _korea_premium_label(feature_row: pd.Series) -> str:
    premium = float(feature_row.get("kimchi_premium", 0.0))
    premium_z = float(feature_row.get("kimchi_premium_zscore", 0.0))
    if premium > 0.03 or premium_z > 1.25:
        return "KIMCHI_PREMIUM_HIGH"
    if premium < -0.005 or premium_z < -1.25:
        return "KIMCHI_DISCOUNT"
    return "KIMCHI_NEUTRAL"


def _return_pct_from_window(window: pd.DataFrame, bars: int) -> float:
    """Past-only close return over at most ``bars`` rows ending at the signal row."""
    if len(window) < 2:
        return 0.0
    lookback = max(1, int(bars))
    ref_idx = max(0, len(window) - 1 - lookback)
    ref = float(window["close"].iloc[ref_idx])
    now = float(window["close"].iloc[-1])
    if ref == 0.0:
        return 0.0
    return float((now - ref) / ref)


def _rolling_path_stats(window: pd.DataFrame, bars: int) -> tuple[float, float, float]:
    """Past-only path return, max drawdown, and max runup over recent bars."""
    if len(window) < 2:
        return 0.0, 0.0, 0.0
    recent = window.iloc[max(0, len(window) - int(bars)) :]
    closes = np.asarray(recent["close"], dtype=np.float64)
    if len(closes) < 2 or closes[0] == 0.0:
        return 0.0, 0.0, 0.0
    path_return = float((closes[-1] - closes[0]) / closes[0])
    running_peak = np.maximum.accumulate(np.maximum(closes, 1e-12))
    running_trough = np.minimum.accumulate(np.maximum(closes, 1e-12))
    max_drawdown = float(np.max(np.maximum(0.0, 1.0 - closes / running_peak)))
    max_runup = float(np.max(np.maximum(0.0, closes / running_trough - 1.0)))
    return path_return, max_drawdown, max_runup


def _realized_vol_pct(window: pd.DataFrame, bars: int) -> float:
    if len(window) < 3:
        return 0.0
    recent = window.iloc[max(0, len(window) - int(bars) - 1) :]
    close = recent["close"].astype(float)
    ret = np.log(close / close.shift(1).replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
    if len(ret) == 0:
        return 0.0
    return float(ret.std(ddof=0) * math.sqrt(max(1, int(bars))))


def _sign_label(value: float, *, flat: float = 0.0015) -> str:
    if value > flat:
        return "UP"
    if value < -flat:
        return "DOWN"
    return "FLAT"


def _pressure_label(value: float, *, low: float = -0.35, high: float = 0.35) -> str:
    if value <= low:
        return "SHORT_PRESSURE"
    if value >= high:
        return "LONG_PRESSURE"
    return "MIXED_PRESSURE"


def _regime_playbook_label(
    *,
    short_ret: float,
    medium_ret: float,
    long_ret: float,
    range_pos: float,
    vol_short: float,
    vol_long: float,
    risk_state: str,
) -> str:
    """LLM-readable playbook token from past-only trend/range/volatility state."""
    aligned_up = short_ret > 0.0015 and medium_ret > 0.0025 and long_ret > 0.004
    aligned_down = short_ret < -0.0015 and medium_ret < -0.0025 and long_ret < -0.004
    vol_expanding = vol_short > max(0.0001, vol_long * 1.25)
    near_high = range_pos > 0.65
    near_low = range_pos < -0.65
    stressed = risk_state in {"ELEVATED", "STRESS"}

    if aligned_up and vol_expanding and not near_high:
        return "TREND_FOLLOW_LONG"
    if aligned_down and vol_expanding and not near_low:
        return "TREND_FOLLOW_SHORT"
    if near_low and short_ret > -0.004 and not stressed:
        return "MEAN_REVERT_LONG"
    if near_high and short_ret < 0.004 and not stressed:
        return "MEAN_REVERT_SHORT"
    if vol_short < max(0.0001, vol_long * 0.70) and abs(medium_ret) < 0.004:
        return "SQUEEZE_WAIT"
    if abs(medium_ret) < 0.003 and abs(range_pos) < 0.45:
        return "CHOP_WAIT"
    return "REGIME_MIXED"


def _cross_market_pressure_label(feature_row: pd.Series) -> str:
    """Combine DXY, USDKRW, and kimchi into a crypto risk pressure token."""
    dxy_z = float(feature_row.get("dxy_zscore", 0.0))
    dxy_mom = float(feature_row.get("dxy_momentum", 0.0))
    usdkrw_z = float(feature_row.get("usdkrw_zscore", 0.0))
    usdkrw_mom = float(feature_row.get("usdkrw_momentum", 0.0))
    kimchi_z = float(feature_row.get("kimchi_premium_zscore", 0.0))
    kimchi_chg = float(feature_row.get("kimchi_premium_change", 0.0))
    # Dollar strength and rising USDKRW usually pressure BTC risk assets.
    # Rising kimchi can be local risk appetite, but extreme/high and falling
    # premium can warn of local demand exhaustion.
    score = (
        -0.45 * dxy_z
        - 16.0 * dxy_mom
        - 0.35 * usdkrw_z
        - 12.0 * usdkrw_mom
        + 0.25 * kimchi_z
        + 10.0 * kimchi_chg
    )
    return _pressure_label(score)


def _edge_state_prompt_features(
    window: pd.DataFrame,
    feature_row: pd.Series,
) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, str], ...], tuple[str, ...]]:
    """
    LLM-oriented, past-only feature summary.

    Unlike engineered_v1's raw indicator dump, this mode exposes the trading
    structure the policy must reason over: multi-horizon path shape, regime
    playbook, cross-market pressure, and strict-drawdown risk context.
    """
    ret_1h = _return_pct_from_window(window, 12)
    ret_2h = _return_pct_from_window(window, 24)
    ret_8h = _return_pct_from_window(window, 96)
    path_ret_6h, path_dd_6h, path_runup_6h = _rolling_path_stats(window, 72)
    path_ret_12h, path_dd_12h, path_runup_12h = _rolling_path_stats(window, 144)
    vol_1h = _realized_vol_pct(window, 12)
    vol_8h = _realized_vol_pct(window, 96)

    risk_state = _risk_state_label(feature_row)
    range_pos = float(feature_row.get("range_pos", 0.0))
    playbook = _regime_playbook_label(
        short_ret=ret_1h,
        medium_ret=ret_2h,
        long_ret=ret_8h,
        range_pos=range_pos,
        vol_short=vol_1h,
        vol_long=vol_8h,
        risk_state=risk_state,
    )
    cross_pressure = _cross_market_pressure_label(feature_row)
    trend_alignment = _trend_alignment_label(feature_row)
    location = _location_label(feature_row)
    oscillator = _oscillator_label(feature_row)
    candle_pattern = _candle_pattern_label(feature_row)
    order_flow = _order_flow_label(
        feature_row, has_taker_flow="taker_buy_base" in window.columns
    )

    numeric_features: list[tuple[str, float]] = [
        ("Past Return 1h", ret_1h),
        ("Past Return 2h", ret_2h),
        ("Past Return 8h", ret_8h),
        ("Past Path Return 6h", path_ret_6h),
        ("Past Path Drawdown 6h", path_dd_6h),
        ("Past Path Runup 6h", path_runup_6h),
        ("Past Path Return 12h", path_ret_12h),
        ("Past Path Drawdown 12h", path_dd_12h),
        ("Past Path Runup 12h", path_runup_12h),
        ("Realized Vol 1h", vol_1h),
        ("Realized Vol 8h", vol_8h),
        ("Range Position", range_pos),
        ("Order Flow Imbalance", float(feature_row.get("taker_imbalance", 0.0))),
    ]
    if "dxy" in window.columns or "usdkrw" in window.columns:
        numeric_features.extend(
            [
                ("DXY Z", float(feature_row.get("dxy_zscore", 0.0))),
                ("DXY Momentum", float(feature_row.get("dxy_momentum", 0.0))),
                ("USDKRW Z", float(feature_row.get("usdkrw_zscore", 0.0))),
                ("USDKRW Momentum", float(feature_row.get("usdkrw_momentum", 0.0))),
            ]
        )
    if "kimchi_premium" in window.columns:
        numeric_features.extend(
            [
                ("Kimchi Z", float(feature_row.get("kimchi_premium_zscore", 0.0))),
                ("Kimchi Change", float(feature_row.get("kimchi_premium_change", 0.0))),
            ]
        )

    symbolic_features = (
        ("Playbook", playbook),
        ("Short Horizon Direction", _sign_label(ret_1h)),
        ("Medium Horizon Direction", _sign_label(ret_2h)),
        ("Long Horizon Direction", _sign_label(ret_8h)),
        ("Trend Alignment", trend_alignment),
        ("Location", location),
        ("Oscillator", oscillator),
        ("Candle Pattern", candle_pattern),
        ("Order Flow", order_flow),
        ("Risk State", risk_state),
        ("Cross Market Pressure", cross_pressure),
    )
    tags = tuple(
        tag
        for tag in (
            playbook,
            _sign_label(ret_1h),
            _sign_label(ret_2h),
            _sign_label(ret_8h),
            trend_alignment,
            location,
            order_flow,
            risk_state,
            cross_pressure,
        )
        if tag != "UNKNOWN"
    )
    return tuple(numeric_features), symbolic_features, tags


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
    if "dxy" in window.columns:
        numeric_features.extend(
            [
                ("Dollar Index", float(feature_row.get("dxy", 0.0))),
                ("Dollar Index Z-Score", float(feature_row.get("dxy_zscore", 0.0))),
                ("Dollar Index Momentum", float(feature_row.get("dxy_momentum", 0.0))),
            ]
        )
    if "kimchi_premium" in window.columns:
        numeric_features.extend(
            [
                ("Kimchi Premium", float(feature_row.get("kimchi_premium", 0.0))),
                ("Kimchi Premium Z-Score", float(feature_row.get("kimchi_premium_zscore", 0.0))),
                ("Kimchi Premium Change", float(feature_row.get("kimchi_premium_change", 0.0))),
            ]
        )
    if "usdkrw" in window.columns:
        numeric_features.extend(
            [
                ("USDKRW Z-Score", float(feature_row.get("usdkrw_zscore", 0.0))),
                ("USDKRW Momentum", float(feature_row.get("usdkrw_momentum", 0.0))),
            ]
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
    macro_state = _macro_state_label(feature_row) if "dxy" in window.columns or "usdkrw" in window.columns else "MACRO_UNKNOWN"
    korea_premium = _korea_premium_label(feature_row) if "kimchi_premium" in window.columns else "KIMCHI_UNKNOWN"

    symbolic_items = [
        ("Trend Alignment", trend_alignment),
        ("Location", location),
        ("Oscillator", oscillator),
        ("Volume State", volume_state),
        ("Candle Pattern", candle_pattern),
        ("Order Flow", order_flow),
        ("Risk State", risk_state),
    ]
    if macro_state != "MACRO_UNKNOWN":
        symbolic_items.append(("Macro Dollar State", macro_state))
    if korea_premium != "KIMCHI_UNKNOWN":
        symbolic_items.append(("Korea Premium State", korea_premium))
    symbolic_features = tuple(symbolic_items)
    tags = tuple(
        tag
        for tag in (
            trend_alignment,
            location,
            oscillator,
            volume_state,
            order_flow,
            risk_state,
            macro_state,
            korea_premium,
        )
        if tag not in {"UNKNOWN", "MACRO_UNKNOWN", "KIMCHI_UNKNOWN"}
    )
    return tuple(numeric_features), symbolic_features, tags



def _edge_state_v3_prompt_features(
    window: pd.DataFrame,
    feature_row: pd.Series,
) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, str], ...], tuple[str, ...]]:
    """Past-only LLM decision card with explicit side/step reasoning cues.

    V3 keeps the v2 edge-state facts, then adds compressed categorical
    decisions that a small text LLM can learn more easily than raw decimals:
    trade readiness, side thesis, no-trade cause, and a past-only step focus.
    These are derived only from the input window and backward-asof external
    features; no target or future path fields are exposed.
    """
    numeric_v2, symbolic_v2, tags_v2 = _edge_state_prompt_features(window, feature_row)
    numeric_dict = {k: float(v) for k, v in numeric_v2}

    ret_1h = float(numeric_dict.get("Past Return 1h", 0.0))
    ret_2h = float(numeric_dict.get("Past Return 2h", 0.0))
    ret_8h = float(numeric_dict.get("Past Return 8h", 0.0))
    path_ret_6h = float(numeric_dict.get("Past Path Return 6h", 0.0))
    path_dd_6h = float(numeric_dict.get("Past Path Drawdown 6h", 0.0))
    path_runup_6h = float(numeric_dict.get("Past Path Runup 6h", 0.0))
    vol_1h = float(numeric_dict.get("Realized Vol 1h", 0.0))
    vol_8h = float(numeric_dict.get("Realized Vol 8h", 0.0))
    range_pos = float(numeric_dict.get("Range Position", 0.0))
    flow = float(numeric_dict.get("Order Flow Imbalance", 0.0))

    sym = {str(k): str(v) for k, v in symbolic_v2}
    playbook = sym.get("Playbook", "REGIME_MIXED")
    risk_state = sym.get("Risk State", "NORMAL")
    cross_pressure = sym.get("Cross Market Pressure", "MIXED_PRESSURE")
    trend_alignment = sym.get("Trend Alignment", "MIXED")
    location = sym.get("Location", "NEAR_FAIR")
    oscillator = sym.get("Oscillator", "NEUTRAL")

    vol_ratio = vol_1h / max(vol_8h, 1e-6)
    path_efficiency = abs(path_ret_6h) / max(path_dd_6h + path_runup_6h, 1e-6)
    side_pressure = (1.5 * ret_1h) + ret_2h + (0.5 * ret_8h) + (0.002 * flow)
    risk_penalty = 1.0 if risk_state == "STRESS" else 0.5 if risk_state == "ELEVATED" else 0.0
    tradeability_score = (abs(side_pressure) * 120.0) + min(vol_ratio, 3.0) * 0.25 + min(path_efficiency, 2.0) * 0.20 - risk_penalty

    if risk_state == "STRESS":
        step_focus = "WAIT_RISK"
    elif vol_ratio >= 1.45 or abs(ret_1h) >= 0.004:
        step_focus = "FAST_36"
    elif abs(ret_2h) >= 0.004 or playbook in {"MEAN_REVERT_LONG", "MEAN_REVERT_SHORT"}:
        step_focus = "MID_72"
    elif abs(ret_8h) >= 0.006 and risk_state in {"CALM", "NORMAL"}:
        step_focus = "SLOW_144"
    else:
        step_focus = "WAIT_CHOP"

    long_votes = 0
    short_votes = 0
    if ret_1h > 0.0015: long_votes += 1
    if ret_2h > 0.0025: long_votes += 1
    if ret_8h > 0.0040: long_votes += 1
    if trend_alignment in {"BULL_STACK", "BULL_REVERSAL"}: long_votes += 1
    if location in {"DISCOUNT", "EXTREME_DISCOUNT"} or oscillator in {"WASHOUT", "OVERSOLD"}: long_votes += 1
    if cross_pressure == "LONG_PRESSURE": long_votes += 1
    if ret_1h < -0.0015: short_votes += 1
    if ret_2h < -0.0025: short_votes += 1
    if ret_8h < -0.0040: short_votes += 1
    if trend_alignment in {"BEAR_STACK", "BEAR_REVERSAL"}: short_votes += 1
    if location in {"PREMIUM", "EXTREME_PREMIUM"} or oscillator in {"OVERBOUGHT", "BLOWOFF"}: short_votes += 1
    if cross_pressure == "SHORT_PRESSURE": short_votes += 1

    long_setup = "LONG_ALIGNED" if long_votes >= 4 and long_votes > short_votes else "LONG_POSSIBLE" if long_votes >= 3 else "LONG_WEAK"
    short_setup = "SHORT_ALIGNED" if short_votes >= 4 and short_votes > long_votes else "SHORT_POSSIBLE" if short_votes >= 3 else "SHORT_WEAK"
    if risk_state == "STRESS":
        trade_readiness = "AVOID_RISK"
        no_trade_cause = "RISK_STRESS"
    elif playbook in {"SQUEEZE_WAIT", "CHOP_WAIT"} and max(long_votes, short_votes) < 4:
        trade_readiness = "WAIT_FOR_EDGE"
        no_trade_cause = "CHOP_OR_SQUEEZE"
    elif long_setup == "LONG_ALIGNED" or short_setup == "SHORT_ALIGNED":
        trade_readiness = "SETUP_READY"
        no_trade_cause = "NONE"
    elif cross_pressure == "MIXED_PRESSURE" and max(long_votes, short_votes) < 3:
        trade_readiness = "WAIT_FOR_EDGE"
        no_trade_cause = "MACRO_MIXED"
    else:
        trade_readiness = "WATCHLIST"
        no_trade_cause = "EDGE_WEAK"

    numeric_extra = (
        ("Vol Expansion Ratio", float(vol_ratio)),
        ("Path Efficiency 6h", float(path_efficiency)),
        ("Side Pressure Score", float(side_pressure)),
        ("Tradeability Score", float(tradeability_score)),
        ("Long Evidence Votes", float(long_votes)),
        ("Short Evidence Votes", float(short_votes)),
    )
    symbolic_extra = (
        ("Step Focus", step_focus),
        ("Trade Readiness", trade_readiness),
        ("Long Thesis", long_setup),
        ("Short Thesis", short_setup),
        ("No Trade Cause", no_trade_cause),
    )
    tags = tuple(dict.fromkeys((*tags_v2, step_focus, trade_readiness, long_setup, short_setup, no_trade_cause)))
    return tuple(numeric_v2) + numeric_extra, tuple(symbolic_v2) + symbolic_extra, tags


def _wide_location_bucket(label: str) -> str:
    if label in {"EXTREME_DISCOUNT", "DISCOUNT"}:
        return "DISCOUNT_ZONE"
    if label in {"EXTREME_PREMIUM", "PREMIUM"}:
        return "PREMIUM_ZONE"
    return "FAIR_ZONE"


def _regime_memory_prompt_features(
    window: pd.DataFrame,
    current_symbolic: tuple[tuple[str, str], ...],
    *,
    local_window_size: int = 96,
    forward_bars: int = 36,
    stride: int = 6,
) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, str], ...], tuple[str, ...]]:
    """Summarize past-only outcomes of similar regimes inside ``window``.

    For a signal at t, this function only uses anchors whose forward outcome
    is fully contained in the input window, i.e. anchor + forward_bars <= t.
    That is historical regime memory, not target leakage.
    """
    if len(window) < max(48, int(forward_bars) + 8):
        return (
            (("Similar Regime Count", 0.0), ("Similar Regime Forward 36", 0.0), ("Similar Regime Reversal Rate", 0.0)),
            (("Regime Memory", "MEMORY_THIN"), ("Regime Trap Risk", "TRAP_UNKNOWN")),
            ("MEMORY_THIN",),
        )
    cur = {str(k): str(v) for k, v in current_symbolic}
    cur_trend = cur.get("Trend Alignment", "MIXED")
    cur_loc = _wide_location_bucket(cur.get("Location", "NEAR_FAIR"))
    cur_osc = cur.get("Oscillator", "NEUTRAL")
    cur_pressure = cur.get("Cross Market Pressure", "MIXED_PRESSURE")
    cur_step = cur.get("Step Focus", "WAIT_CHOP")
    cur_long = cur.get("Long Thesis", "LONG_WEAK")
    cur_short = cur.get("Short Thesis", "SHORT_WEAK")
    expected_sign = 0
    if cur_long == "LONG_ALIGNED" and cur_short != "SHORT_ALIGNED":
        expected_sign = 1
    elif cur_short == "SHORT_ALIGNED" and cur_long != "LONG_ALIGNED":
        expected_sign = -1

    local_df = window.reset_index(drop=True).copy()
    feature_frame = build_market_feature_frame(local_df, window_size=min(int(local_window_size), len(local_df)))
    last_anchor = len(local_df) - 1 - int(forward_bars)
    first_anchor = max(32, int(local_window_size) // 2)
    matched_returns: list[float] = []
    matched_expected: list[float] = []
    for anchor in range(first_anchor, max(first_anchor, last_anchor) + 1, max(1, int(stride))):
        local_win = local_df.iloc[max(0, anchor - int(local_window_size) + 1) : anchor + 1]
        if len(local_win) < 24:
            continue
        _, sym2, _ = _edge_state_prompt_features(local_win, feature_frame.iloc[anchor])
        _, sym3, _ = _edge_state_v3_prompt_features(local_win, feature_frame.iloc[anchor])
        sym = {str(k): str(v) for k, v in (*sym2, *sym3)}
        score = 0
        score += sym.get("Trend Alignment") == cur_trend
        score += _wide_location_bucket(sym.get("Location", "NEAR_FAIR")) == cur_loc
        score += sym.get("Oscillator") == cur_osc
        score += sym.get("Cross Market Pressure") == cur_pressure
        score += sym.get("Step Focus") == cur_step
        if score < 3:
            continue
        now = float(local_df.loc[anchor, "close"])
        later = float(local_df.loc[anchor + int(forward_bars), "close"])
        if now <= 0.0:
            continue
        fwd = (later - now) / now
        matched_returns.append(float(fwd))
        if expected_sign != 0:
            matched_expected.append(float(expected_sign * fwd))

    n = len(matched_returns)
    if n == 0:
        return (
            (("Similar Regime Count", 0.0), ("Similar Regime Forward 36", 0.0), ("Similar Regime Reversal Rate", 0.0)),
            (("Regime Memory", "MEMORY_THIN"), ("Regime Trap Risk", "TRAP_UNKNOWN")),
            ("MEMORY_THIN",),
        )
    arr = np.asarray(matched_returns, dtype=np.float64)
    mean_fwd = float(np.mean(arr))
    if expected_sign == 0 or not matched_expected:
        reversal_rate = 0.0
        trap = "TRAP_UNKNOWN"
        memory = "MEMORY_MIXED" if abs(mean_fwd) < 0.0015 else ("MEMORY_LONG_DRIFT" if mean_fwd > 0 else "MEMORY_SHORT_DRIFT")
    else:
        exp_arr = np.asarray(matched_expected, dtype=np.float64)
        reversal_rate = float(np.mean(exp_arr < 0.0))
        if reversal_rate >= 0.60 and n >= 3:
            trap = "TRAP_HIGH"
            memory = "RECENT_REVERSAL_RISK"
        elif float(np.mean(exp_arr > 0.0)) >= 0.60 and n >= 3:
            trap = "TRAP_LOW"
            memory = "RECENT_CONTINUATION"
        else:
            trap = "TRAP_MIXED"
            memory = "MEMORY_MIXED"
    numeric = (
        ("Similar Regime Count", float(n)),
        ("Similar Regime Forward 36", mean_fwd),
        ("Similar Regime Reversal Rate", float(reversal_rate)),
    )
    symbolic = (("Regime Memory", memory), ("Regime Trap Risk", trap))
    return numeric, symbolic, (memory, trap)


def _edge_state_v4_prompt_features(
    window: pd.DataFrame,
    feature_row: pd.Series,
) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, str], ...], tuple[str, ...]]:
    """Edge-state v3 plus past-only similar-regime memory."""
    numeric_v3, symbolic_v3, tags_v3 = _edge_state_v3_prompt_features(window, feature_row)
    mem_numeric, mem_symbolic, mem_tags = _regime_memory_prompt_features(window, symbolic_v3)
    return tuple(numeric_v3) + mem_numeric, tuple(symbolic_v3) + mem_symbolic, tuple(dict.fromkeys((*tags_v3, *mem_tags)))


def _bucket_signed(value: float, *, pos: float, neg: float, high_label: str, low_label: str, mid_label: str) -> str:
    if value >= pos:
        return high_label
    if value <= neg:
        return low_label
    return mid_label


def _edge_state_v5_prompt_features(
    window: pd.DataFrame,
    feature_row: pd.Series,
) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, str], ...], tuple[str, ...]]:
    """V4 plus regime-activation descriptors from Kimchi-flow alpha audits.

    The added fields are still past-only.  They are not a direct trading rule;
    they describe whether the audited 2025-like Kimchi/liquidity regime is
    present and whether long/short entry context resembles historical winners.
    """
    numeric_v4, symbolic_v4, tags_v4 = _edge_state_v4_prompt_features(window, feature_row)
    kimchi_change = float(feature_row.get("kimchi_premium_change", 0.0))
    kimchi_z = float(feature_row.get("kimchi_premium_zscore", 0.0))
    trades_ratio = float(feature_row.get("trades_ratio", 0.0))
    taker_imbalance = float(feature_row.get("taker_imbalance", 0.0))
    range_pos = float(feature_row.get("range_pos", 0.0))
    bb_z = float(feature_row.get("bb_z", 0.0))
    rsi_norm = float(feature_row.get("rsi_norm", 0.0))
    close_z = float(feature_row.get("close_zscore_48", 0.0))
    sma48 = float(feature_row.get("sma48_ratio", 0.0))
    window_dd = float(feature_row.get("window_drawdown", 0.0))
    volume_ratio = float(feature_row.get("volume_ratio", 0.0))
    usdkrw_mom = float(feature_row.get("usdkrw_momentum", 0.0))

    # Thresholds are deliberately rounded from audits, not copied as a hidden
    # optimizer.  The LLM receives regime/context cues and can abstain.
    kimchi_flow_regime = (kimchi_change <= -0.0007) and (trades_ratio >= 0.78 or trades_ratio <= 0.47)
    if kimchi_flow_regime and trades_ratio >= 0.78:
        flow_activation = "KIMCHI_FLOW_LONG_ACTIVE"
    elif kimchi_flow_regime and trades_ratio <= 0.47:
        flow_activation = "KIMCHI_FLOW_SHORT_ACTIVE"
    elif kimchi_change <= -0.0007:
        flow_activation = "KIMCHI_FLOW_WATCH"
    else:
        flow_activation = "KIMCHI_FLOW_INACTIVE"

    long_context_score = 0
    if taker_imbalance >= 0.03:
        long_context_score += 1
    if bb_z >= 0.5 or close_z >= 0.5:
        long_context_score += 1
    if rsi_norm >= 0.08:
        long_context_score += 1
    if range_pos >= 0.25:
        long_context_score += 1
    if long_context_score >= 3:
        long_context = "LONG_CONTEXT_WINLIKE"
    elif long_context_score >= 2:
        long_context = "LONG_CONTEXT_MIXED"
    else:
        long_context = "LONG_CONTEXT_WEAK"

    short_context_score = 0
    if close_z >= 0.5 or sma48 >= 0.002:
        short_context_score += 1
    if window_dd <= 0.006:
        short_context_score += 1
    if volume_ratio >= 0.35:
        short_context_score += 1
    if taker_imbalance <= -0.03:
        short_context_score += 1
    if usdkrw_mom <= 0.0005:
        short_context_score += 1
    if short_context_score >= 3:
        short_context = "SHORT_CONTEXT_WINLIKE"
    elif short_context_score >= 2:
        short_context = "SHORT_CONTEXT_MIXED"
    else:
        short_context = "SHORT_CONTEXT_WEAK"

    failure_cue_score = 0
    if window_dd >= 0.010:
        failure_cue_score += 1
    if usdkrw_mom >= 0.0012:
        failure_cue_score += 1
    if abs(kimchi_z) < 0.2 and abs(taker_imbalance) < 0.03:
        failure_cue_score += 1
    failure_cue = "ABSTAIN_FAILURE_REGIME" if failure_cue_score >= 2 else "NO_FAILURE_CUE"

    numeric_extra = (
        ("Kimchi Flow Change", kimchi_change),
        ("Kimchi Z", kimchi_z),
        ("Trades Participation", trades_ratio),
        ("Taker Imbalance", taker_imbalance),
        ("LLM Long Context Score", float(long_context_score)),
        ("LLM Short Context Score", float(short_context_score)),
        ("LLM Failure Cue Score", float(failure_cue_score)),
    )
    symbolic_extra = (
        ("Kimchi Flow Regime", flow_activation),
        ("Long Entry Context", long_context),
        ("Short Entry Context", short_context),
        ("Regime Failure Cue", failure_cue),
    )
    tags = tuple(dict.fromkeys((*tags_v4, flow_activation, long_context, short_context, failure_cue)))
    return tuple(numeric_v4) + numeric_extra, tuple(symbolic_v4) + symbolic_extra, tags


def _edge_state_v6_prompt_features(
    window: pd.DataFrame,
    feature_row: pd.Series,
) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, str], ...], tuple[str, ...]]:
    """V5 plus completed-weekly regime features.

    Weekly features come from the previous completed weekly candle only.  They
    are intended to help the LLM decide whether Kimchi-flow should be filtered
    selectively or kept broadly active in higher-timeframe trend regimes.
    """
    numeric_v5, symbolic_v5, tags_v5 = _edge_state_v5_prompt_features(window, feature_row)
    weekly_return_1w = float(feature_row.get("weekly_return_1w", 0.0))
    weekly_return_4w = float(feature_row.get("weekly_return_4w", 0.0))
    weekly_range_1w = float(feature_row.get("weekly_range_1w", 0.0))
    weekly_range_pos = float(feature_row.get("weekly_range_pos", 0.0))
    weekly_drawdown_4w = float(feature_row.get("weekly_drawdown_4w", 0.0))

    if weekly_return_4w >= 0.08 and weekly_drawdown_4w <= 0.08:
        weekly_regime = "WEEKLY_BROAD_RISK_ON"
    elif weekly_return_4w <= -0.08 or weekly_drawdown_4w >= 0.15:
        weekly_regime = "WEEKLY_DEFENSIVE_FILTER"
    elif abs(weekly_return_4w) <= 0.03 and weekly_range_1w >= 0.08:
        weekly_regime = "WEEKLY_CHOP_FILTER"
    else:
        weekly_regime = "WEEKLY_MIXED"

    if weekly_range_pos >= 0.50:
        weekly_location = "WEEKLY_UPPER_RANGE"
    elif weekly_range_pos <= -0.50:
        weekly_location = "WEEKLY_LOWER_RANGE"
    else:
        weekly_location = "WEEKLY_MID_RANGE"

    weekly_filter_score = 0
    if weekly_regime in {"WEEKLY_DEFENSIVE_FILTER", "WEEKLY_CHOP_FILTER"}:
        weekly_filter_score += 1
    if weekly_range_pos <= -0.50 and weekly_return_1w < 0.0:
        weekly_filter_score += 1
    if weekly_drawdown_4w >= 0.10:
        weekly_filter_score += 1

    numeric_extra = (
        ("Weekly Return 1w", weekly_return_1w),
        ("Weekly Return 4w", weekly_return_4w),
        ("Weekly Range 1w", weekly_range_1w),
        ("Weekly Range Position", weekly_range_pos),
        ("Weekly Drawdown 4w", weekly_drawdown_4w),
        ("Weekly Filter Score", float(weekly_filter_score)),
    )
    symbolic_extra = (
        ("Weekly Regime", weekly_regime),
        ("Weekly Location", weekly_location),
    )
    tags = tuple(dict.fromkeys((*tags_v5, weekly_regime, weekly_location)))
    return tuple(numeric_v5) + numeric_extra, tuple(symbolic_v5) + symbolic_extra, tags


def _higher_timeframe_regime(
    label: str,
    *,
    return_1: float,
    return_4: float,
    range_1: float,
    range_pos: float,
    drawdown_4: float,
) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, str], ...], tuple[str, ...]]:
    key = label.upper()
    if return_4 >= 0.08 and drawdown_4 <= 0.08:
        regime = f"{key}_BROAD_RISK_ON"
    elif return_4 <= -0.08 or drawdown_4 >= 0.15:
        regime = f"{key}_STRESS"
    elif abs(return_4) <= 0.03 and range_1 >= 0.08:
        regime = f"{key}_CHOP"
    else:
        regime = f"{key}_MIXED"

    if range_pos >= 0.50:
        location = f"{key}_UPPER_RANGE"
    elif range_pos <= -0.50:
        location = f"{key}_LOWER_RANGE"
    else:
        location = f"{key}_MID_RANGE"

    stress_score = 0
    if regime in {f"{key}_STRESS", f"{key}_CHOP"}:
        stress_score += 1
    if range_pos <= -0.50 and return_1 < 0.0:
        stress_score += 1
    if drawdown_4 >= 0.10:
        stress_score += 1

    numeric = (
        (f"{label} Return 1", return_1),
        (f"{label} Return 4", return_4),
        (f"{label} Range 1", range_1),
        (f"{label} Range Position", range_pos),
        (f"{label} Drawdown 4", drawdown_4),
        (f"{label} Stress Score", float(stress_score)),
    )
    symbolic = ((f"{label} Regime", regime), (f"{label} Location", location))
    return numeric, symbolic, (regime, location)


def _edge_state_v7_prompt_features(
    window: pd.DataFrame,
    feature_row: pd.Series,
) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, str], ...], tuple[str, ...]]:
    """V5 plus completed 4h/1d/3d/1w regime features.

    This generalizes v6 beyond weekly context.  Each higher timeframe excludes
    the current incomplete candle via shifted completed-bar feature alignment.
    """
    numeric_v5, symbolic_v5, tags_v5 = _edge_state_v5_prompt_features(window, feature_row)
    specs = (
        ("4H", "htf_4h"),
        ("1D", "htf_1d"),
        ("3D", "htf_3d"),
        ("1W", "htf_1w"),
    )
    numeric_extra: list[tuple[str, float]] = []
    symbolic_extra: list[tuple[str, str]] = []
    tag_extra: list[str] = []
    stress_total = 0.0
    for label, prefix in specs:
        numeric, symbolic, tags = _higher_timeframe_regime(
            label,
            return_1=float(feature_row.get(f"{prefix}_return_1", 0.0)),
            return_4=float(feature_row.get(f"{prefix}_return_4", 0.0)),
            range_1=float(feature_row.get(f"{prefix}_range_1", 0.0)),
            range_pos=float(feature_row.get(f"{prefix}_range_pos", 0.0)),
            drawdown_4=float(feature_row.get(f"{prefix}_drawdown_4", 0.0)),
        )
        numeric_extra.extend(numeric)
        symbolic_extra.extend(symbolic)
        tag_extra.extend(tags)
        for name, value in numeric:
            if name.endswith("Stress Score"):
                stress_total += float(value)

    if stress_total >= 5.0:
        mtf_mode = "MTF_STRESS_BROAD_OR_REVERSAL"
    elif stress_total <= 1.0:
        mtf_mode = "MTF_LOW_STRESS_SELECTIVE"
    else:
        mtf_mode = "MTF_MIXED_FILTER"
    numeric_extra.append(("MTF Stress Total", stress_total))
    symbolic_extra.append(("MTF Activation Mode", mtf_mode))
    tag_extra.append(mtf_mode)
    return tuple(numeric_v5) + tuple(numeric_extra), tuple(symbolic_v5) + tuple(symbolic_extra), tuple(dict.fromkeys((*tags_v5, *tag_extra)))

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
    path_entry_delay_bars: int = 1,
    path_mae_penalty: float = 1.0,
    path_mfe_bonus: float = 0.0,
    path_min_net_return: float = 0.0,
    path_max_mae: float = 1.0,
    multi_horizon_bars: str | Iterable[int] | None = None,
    max_samples: int | None = None,
    sample_mode: str = "sequential",
    sample_seed: int = 42,
    prompt_style: str = "numeric",
    prompt_feature_mode: str = "basic_v0",
    action_schema: str = "buy_hold_sell",
    trade_side_sample_policy: str = "trade_only",
    modality: str = "multimodal",
    sample_dates: Iterable[str] | None = None,
) -> List[VLMTrainingSample]:
    """
    Build trading-derived VLM samples with image + prompt + target action.

    Uses only historical window `[t-w+1, t]` for prompt/image.
    Target action is derived by:
      - label_mode='next_return': horizon return sign
      - label_mode='utility': cost/risk-aware action utility argmax
      - label_mode='path_outcome': delayed-entry executable path utility
    where horizon return uses `(open_{t+h} - open_t)/open_t`.
    """
    modality_key = str(modality).lower().strip()
    if modality_key not in {"multimodal", "text_only"}:
        raise ValueError(
            "modality must be one of {'multimodal','text_only'}, "
            f"got {modality}"
        )
    renderer = None
    if modality_key == "multimodal":
        renderer = ChartGenerator(
            ChartGeneratorConfig(
                resolution=resolution,
                cache_dir=cache_dir,
                show_indicators=True,
                show_oscillators=True,
            )
        )

    horizon = max(1, int(target_horizon))
    action_schema_key = str(action_schema).lower().strip()
    get_action_labels(action_schema_key)
    multi_horizons = parse_multi_horizon_bars(multi_horizon_bars)
    max_required_horizon = (
        max(horizon, max(multi_horizons))
        if action_schema_key == "multi_horizon_side"
        else horizon
    )
    start_t = window_size - 1
    end_t = len(market_df) - max_required_horizon  # because we access t+h or multi-horizon paths
    if end_t <= start_t:
        return []

    sample_mode_key = str(sample_mode).lower().strip()
    if sample_mode_key not in {"sequential", "random", "balanced", "uniform"}:
        raise ValueError(
            "sample_mode must be one of {'sequential','random','balanced','uniform'}, "
            f"got {sample_mode}"
        )
    label_mode_key = str(label_mode).lower().strip()
    if label_mode_key not in {"next_return", "utility", "path_outcome"}:
        raise ValueError(
            "label_mode must be one of {'next_return','utility','path_outcome'}, "
            f"got {label_mode}"
        )
    prompt_feature_mode_key = str(prompt_feature_mode).lower().strip()
    if prompt_feature_mode_key not in {"basic_v0", "engineered_v1", "edge_state_v2", "edge_state_v3", "edge_state_v4", "edge_state_v5", "edge_state_v6", "edge_state_v7"}:
        raise ValueError(
            "prompt_feature_mode must be one of "
            "{'basic_v0','engineered_v1','edge_state_v2','edge_state_v3','edge_state_v4','edge_state_v5','edge_state_v6','edge_state_v7'}, "
            f"got {prompt_feature_mode}"
        )
    trade_side_sample_policy_key = str(trade_side_sample_policy).lower().strip()
    if trade_side_sample_policy_key not in {"trade_only", "directional_all"}:
        raise ValueError(
            "trade_side_sample_policy must be one of "
            "{'trade_only','directional_all'}, "
            f"got {trade_side_sample_policy}"
        )

    feature_frame = build_market_feature_frame(market_df, window_size=window_size)
    sample_date_set = None
    if sample_dates is not None:
        sample_date_set = {str(pd.to_datetime(x)) for x in sample_dates}

    # First pass: compute candidate metadata without rendering images.
    candidates = []
    for t in range(start_t, end_t):
        row_date = str(pd.to_datetime(market_df.loc[t, "date"]))
        if sample_date_set is not None and row_date not in sample_date_set:
            continue
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
        action_utility_map = {
            "BUY": 0.0,
            "HOLD": 0.0,
            "SELL": 0.0,
        }
        if action_schema_key == "multi_horizon_side" and label_mode_key == "path_outcome":
            target_action, multi_utilities, next_return = action_from_multi_horizon_path_outcomes(
                market_df,
                t,
                hold_bars_list=multi_horizons,
                path_entry_delay_bars=path_entry_delay_bars,
                utility_fee_rate=utility_fee_rate,
                utility_slippage_rate=utility_slippage_rate,
                utility_leverage=utility_leverage,
                path_mae_penalty=path_mae_penalty,
                path_mfe_bonus=path_mfe_bonus,
                utility_hold_margin=utility_hold_margin,
                path_min_net_return=path_min_net_return,
                path_max_mae=path_max_mae,
            )
            long_utils = [v for k, v in multi_utilities.items() if k.startswith("LONG_")]
            short_utils = [v for k, v in multi_utilities.items() if k.startswith("SHORT_")]
            utilities = {
                "BUY": float(max(long_utils, default=0.0)),
                "HOLD": float(multi_utilities.get("NO_TRADE", 0.0)),
                "SELL": float(max(short_utils, default=0.0)),
            }
            action_utility_map = {
                str(label).upper(): float(multi_utilities.get(label, 0.0))
                for label in get_action_labels(action_schema_key)
            }
            best_outcome = None
            path_trade_ok = target_action != "NO_TRADE"
        elif label_mode_key == "path_outcome":
            path_cfg = PathOutcomeConfig(
                hold_bars=horizon,
                entry_delay_bars=path_entry_delay_bars,
                fee_rate=utility_fee_rate,
                slippage_rate=utility_slippage_rate,
                leverage=utility_leverage,
                mae_penalty=path_mae_penalty,
                mfe_bonus=path_mfe_bonus,
                hold_margin=utility_hold_margin,
                min_net_return=path_min_net_return,
                max_mae=path_max_mae,
            )
            long_outcome = compute_trade_path_outcome(market_df, t, "LONG", path_cfg)
            short_outcome = compute_trade_path_outcome(market_df, t, "SHORT", path_cfg)
            if long_outcome is None or short_outcome is None:
                continue
            best_outcome = long_outcome if long_outcome.utility >= short_outcome.utility else short_outcome
            path_trade_ok = (
                float(best_outcome.utility) > float(utility_hold_margin)
                and float(best_outcome.net_return) > float(path_min_net_return)
                and float(best_outcome.mae) <= float(path_max_mae)
            )
            utilities = {
                "BUY": float(long_outcome.utility),
                "HOLD": float(utility_hold_reward_bias),
                "SELL": float(short_outcome.utility),
            }
            action_utility_map = dict(utilities)
            next_return = float(best_outcome.net_return)
        else:
            best_outcome = None
            path_trade_ok = False
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
            action_utility_map = dict(utilities)
        if action_schema_key == "multi_horizon_side":
            if label_mode_key != "path_outcome":
                if float(utilities["BUY"]) >= max(float(utilities["SELL"]), float(utilities["HOLD"])):
                    target_action = f"LONG_{max(multi_horizons)}"
                elif float(utilities["SELL"]) >= max(float(utilities["BUY"]), float(utilities["HOLD"])):
                    target_action = f"SHORT_{max(multi_horizons)}"
                else:
                    target_action = "NO_TRADE"
        elif action_schema_key == "trade_gate":
            if label_mode_key == "path_outcome":
                target_action = "TRADE" if path_trade_ok else "NO_TRADE"
            elif label_mode_key == "utility":
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
        elif action_schema_key == "trade_side":
            if trade_side_sample_policy_key == "directional_all":
                if label_mode_key == "path_outcome":
                    target_action = (
                        "LONG"
                        if best_outcome is not None and best_outcome.side == "LONG"
                        else "SHORT"
                    )
                elif label_mode_key == "utility":
                    target_action = (
                        "LONG"
                        if float(utilities["BUY"]) >= float(utilities["SELL"])
                        else "SHORT"
                    )
                else:
                    target_action = "LONG" if next_return >= 0.0 else "SHORT"
            else:
                if label_mode_key == "path_outcome":
                    target_action = (
                        None
                        if not path_trade_ok
                        else (
                            "LONG"
                            if best_outcome is not None and best_outcome.side == "LONG"
                            else "SHORT"
                        )
                    )
                elif label_mode_key == "utility":
                    target_action = action_from_trade_side_utilities(
                        utility_buy=utilities["BUY"],
                        utility_hold=utilities["HOLD"],
                        utility_sell=utilities["SELL"],
                        hold_margin=utility_hold_margin,
                    )
                else:
                    target_action = action_from_trade_side_next_return(
                        next_return,
                        hold_band=hold_band,
                    )
        else:
            if label_mode_key == "path_outcome":
                if not path_trade_ok:
                    target_action = "HOLD"
                else:
                    target_action = (
                        "BUY"
                        if best_outcome is not None and best_outcome.side == "LONG"
                        else "SELL"
                    )
            elif label_mode_key == "utility":
                target_action = action_from_utilities(
                    utility_buy=utilities["BUY"],
                    utility_hold=utilities["HOLD"],
                    utility_sell=utilities["SELL"],
                    hold_margin=utility_hold_margin,
                )
            else:
                target_action = action_from_next_return(next_return, hold_band=hold_band)
        if target_action is None:
            continue
        candidates.append(
            (
                t,
                target_action,
                float(next_return),
                float(utilities["BUY"]),
                float(utilities["HOLD"]),
                float(utilities["SELL"]),
                float(risk_weight),
                action_utility_map,
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
            action_utility_map,
        ) = candidates[int(pos)]
        window = make_window(market_df, t=t, w=window_size)
        image = None
        if renderer is not None:
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
        elif prompt_feature_mode_key == "edge_state_v2":
            (
                extra_numeric_features,
                extra_symbolic_features,
                context_tags,
            ) = _edge_state_prompt_features(window, feature_row)
        elif prompt_feature_mode_key == "edge_state_v3":
            (
                extra_numeric_features,
                extra_symbolic_features,
                context_tags,
            ) = _edge_state_v3_prompt_features(window, feature_row)
        elif prompt_feature_mode_key == "edge_state_v4":
            (
                extra_numeric_features,
                extra_symbolic_features,
                context_tags,
            ) = _edge_state_v4_prompt_features(window, feature_row)
        elif prompt_feature_mode_key == "edge_state_v5":
            (
                extra_numeric_features,
                extra_symbolic_features,
                context_tags,
            ) = _edge_state_v5_prompt_features(window, feature_row)
        elif prompt_feature_mode_key == "edge_state_v6":
            (
                extra_numeric_features,
                extra_symbolic_features,
                context_tags,
            ) = _edge_state_v6_prompt_features(window, feature_row)
        elif prompt_feature_mode_key == "edge_state_v7":
            (
                extra_numeric_features,
                extra_symbolic_features,
                context_tags,
            ) = _edge_state_v7_prompt_features(window, feature_row)
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
            modality=modality_key,
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
                action_utility_map={
                    str(k).upper(): float(v) for k, v in dict(action_utility_map).items()
                },
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
        record = {
            # TRL conversational prompt format.
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": s.prompt},
            ],
            "target_action": s.target_action,
                "next_return": float(s.next_return),
                "action_utility_buy": float(s.action_utility_buy),
                "action_utility_hold": float(s.action_utility_hold),
                "action_utility_sell": float(s.action_utility_sell),
            "action_utility_map": json.dumps(
                {
                    str(k).upper(): float(v)
                    for k, v in dict(s.action_utility_map or {}).items()
                },
                sort_keys=True,
            ),
            "dynamic_risk_weight": float(s.dynamic_risk_weight),
            "date": s.date,
        }
        if s.image is not None:
            record["image"] = s.image
        records.append(record)
    return records
