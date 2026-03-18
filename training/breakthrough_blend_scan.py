"""Blend-model scoreband scan for breakthrough search.

Large-change axis:
- model blend (exp6 / exp4 / monthly-finetuned)
- decision method (static blend vs trend-adaptive blend)
- data slice (recent 3 months first, then 1-year follow-up)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from envs.trading_env import TradingEnv, TradingEnvConfig
from evaluation.backtest import (
    build_underlying_curve,
    extract_policy_action_probs,
    infer_expected_image_shape,
    normalize_action_probs,
    periods_per_year_from_timeframe,
)
from evaluation.metrics import summarize_metrics
from training.data_sources import load_market_data


EXP6 = "checkpoints/ppo_option_a_real_w384_t786k_exp6_balanced.zip"
EXP4 = "checkpoints/ppo_option_a_real_exp4_nosym.zip"
FT4096_FEB = "checkpoints/rolling_monthly_finetune_5m_ft4096/rolling_finetune_5m_2026-02_ft4096.zip"


@dataclass
class BlendConfig:
    name: str
    model_a: str
    model_b: str
    mode: str  # static | trend
    weight_a: float = 1.0
    weight_up: float = 1.0
    weight_down: float = 1.0
    trend_threshold: float = 0.002


def candidate_configs() -> list[BlendConfig]:
    return [
        BlendConfig("exp6_only_reference", EXP6, EXP4, mode="static", weight_a=1.0),
        BlendConfig("blend_exp6_exp4_w085", EXP6, EXP4, mode="static", weight_a=0.85),
        BlendConfig("blend_exp6_exp4_w075", EXP6, EXP4, mode="static", weight_a=0.75),
        BlendConfig("blend_exp6_exp4_w065", EXP6, EXP4, mode="static", weight_a=0.65),
        BlendConfig(
            "blend_exp6_exp4_trend_adapt",
            EXP6,
            EXP4,
            mode="trend",
            weight_up=0.90,
            weight_down=0.35,
            trend_threshold=0.002,
        ),
        BlendConfig("blend_exp6_ft4096_w080", EXP6, FT4096_FEB, mode="static", weight_a=0.80),
        BlendConfig("blend_exp6_ft4096_w060", EXP6, FT4096_FEB, mode="static", weight_a=0.60),
        BlendConfig(
            "blend_exp6_ft4096_trend_adapt",
            EXP6,
            FT4096_FEB,
            mode="trend",
            weight_up=0.85,
            weight_down=0.45,
            trend_threshold=0.002,
        ),
    ]


def _blend_weight(cfg: BlendConfig, trend: float) -> float:
    if cfg.mode == "static":
        return float(np.clip(cfg.weight_a, 0.0, 1.0))
    if trend >= float(cfg.trend_threshold):
        return float(np.clip(cfg.weight_up, 0.0, 1.0))
    if trend <= -float(cfg.trend_threshold):
        return float(np.clip(cfg.weight_down, 0.0, 1.0))
    return 0.5 * float(np.clip(cfg.weight_up, 0.0, 1.0)) + 0.5 * float(
        np.clip(cfg.weight_down, 0.0, 1.0)
    )


def run_blend_scoreband_backtest(
    *,
    cfg: BlendConfig,
    start_date: str,
    end_date: str,
    symbol: str = "BTCUSDT",
    timeframe: str = "5m",
    market_type: str = "futures",
    window_size: int = 384,
    leverage: float = 1.0,
    initial_equity: float = 1000.0,
    fee_rate: float = 0.0005,
    slippage_rate: float = 0.0001,
    score_center_alpha: float = 0.02,
    score_entry_threshold: float = 0.005,
    score_flip_threshold: float = 0.02,
    score_neutral_band: float = 0.001,
    debiased_action: str = "mirror_scalar",
) -> dict[str, Any]:
    model_a = PPO.load(cfg.model_a)
    model_b = PPO.load(cfg.model_b)

    shape_a = infer_expected_image_shape(model_a)
    shape_b = infer_expected_image_shape(model_b)
    if shape_a != shape_b:
        raise ValueError(f"image shape mismatch: a={shape_a}, b={shape_b}")

    market_df = load_market_data(
        source="binance",
        timeframe=timeframe,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        market_type=market_type,
    )

    env = TradingEnv(
        market_df=market_df,
        images=None,
        config=TradingEnvConfig(
            window_size=window_size,
            leverage=leverage,
            initial_equity=initial_equity,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            hold_action_mode="flat",
            image_shape=shape_a,
        ),
    )

    obs, info = env.reset()
    score_ema = 0.0
    score_ema_init = False
    equity_curve = [float(info["equity"])]

    action_counts = {0: 0, 1: 0, 2: 0}
    trade_rebalance_steps = 0
    trade_turnover_legs = 0
    trade_entries = 0
    trade_exits = 0
    trade_direct_flips = 0
    trade_long_entries = 0
    trade_short_entries = 0
    trade_long_exits = 0
    trade_short_exits = 0

    done = False
    while not done:
        side_now = int(info.get("position_side", 0))
        trend = float(obs["scalars"][3])
        w_a = _blend_weight(cfg, trend)

        p_a = extract_policy_action_probs(model_a, obs, debiased_action=debiased_action)
        p_b = extract_policy_action_probs(model_b, obs, debiased_action=debiased_action)
        p = normalize_action_probs(w_a * p_a + (1.0 - w_a) * p_b)

        score_raw = float(p[0] - p[2])
        if not score_ema_init:
            score_ema = score_raw
            score_ema_init = True
        else:
            a = float(score_center_alpha)
            score_ema = (1.0 - a) * score_ema + a * score_raw
        score = score_raw - score_ema

        if side_now == 0:
            if score >= float(score_entry_threshold):
                action_i = 0
            elif score <= -float(score_entry_threshold):
                action_i = 2
            elif abs(score) <= float(score_neutral_band):
                action_i = 1
            else:
                action_i = 1
        elif side_now > 0:
            action_i = 2 if score <= -float(score_flip_threshold) else 0
        else:
            action_i = 0 if score >= float(score_flip_threshold) else 2

        action_counts[action_i] += 1
        prev_side = int(info.get("position_side", 0))
        obs, step_reward, terminated, truncated, info = env.step(action_i)
        del step_reward
        curr_side = int(info.get("position_side", 0))

        if curr_side != prev_side:
            trade_rebalance_steps += 1
            trade_turnover_legs += abs(curr_side - prev_side)
            if prev_side == 0 and curr_side != 0:
                trade_entries += 1
                if curr_side > 0:
                    trade_long_entries += 1
                else:
                    trade_short_entries += 1
            elif prev_side != 0 and curr_side == 0:
                trade_exits += 1
                if prev_side > 0:
                    trade_long_exits += 1
                else:
                    trade_short_exits += 1
            else:
                trade_direct_flips += 1
                trade_entries += 1
                trade_exits += 1
                if prev_side > 0:
                    trade_long_exits += 1
                    trade_short_entries += 1
                else:
                    trade_short_exits += 1
                    trade_long_entries += 1

        equity_curve.append(float(info["equity"]))
        done = bool(terminated or truncated)

    start_idx = window_size - 1
    end_idx = start_idx + len(equity_curve)
    open_prices = market_df["open"].iloc[start_idx:end_idx].to_numpy(dtype=np.float64)
    benchmark_curve = build_underlying_curve(open_prices, initial_equity=initial_equity)

    report = summarize_metrics(
        equity=equity_curve,
        underlying=benchmark_curve,
        periods_per_year=periods_per_year_from_timeframe(timeframe),
    )
    steps = int(len(equity_curve) - 1)
    report["num_steps"] = float(steps)
    report["action_counts"] = {
        "buy": float(action_counts[0]),
        "hold": float(action_counts[1]),
        "sell": float(action_counts[2]),
    }
    report["action_ratio"] = {
        "buy": action_counts[0] / max(1, steps),
        "hold": action_counts[1] / max(1, steps),
        "sell": action_counts[2] / max(1, steps),
    }
    report["trade_counts"] = {
        "rebalance_steps": int(trade_rebalance_steps),
        "turnover_legs": int(trade_turnover_legs),
        "entries_total": int(trade_entries),
        "exits_total": int(trade_exits),
        "direct_flips": int(trade_direct_flips),
        "long_entries": int(trade_long_entries),
        "short_entries": int(trade_short_entries),
        "long_exits": int(trade_long_exits),
        "short_exits": int(trade_short_exits),
    }
    total_fees = float(info.get("total_fees", 0.0))
    total_slippage = float(info.get("total_slippage", 0.0))
    report["execution_cost"] = {
        "total_fees": total_fees,
        "total_slippage": total_slippage,
        "total_cost": float(total_fees + total_slippage),
    }
    report["blend"] = {
        "mode": cfg.mode,
        "model_a": str(Path(cfg.model_a).resolve()),
        "model_b": str(Path(cfg.model_b).resolve()),
        "weight_a": float(cfg.weight_a),
        "weight_up": float(cfg.weight_up),
        "weight_down": float(cfg.weight_down),
        "trend_threshold": float(cfg.trend_threshold),
        "debiased_action": str(debiased_action),
    }
    return report


def run_scan(*, output: str, topn: int = 3) -> dict[str, Any]:
    recent_period = ("2025-12-01", "2026-02-28")
    followup_period = ("2025-03-01", "2026-02-28")

    cfgs = candidate_configs()
    out: dict[str, Any] = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "recent_period": {"start": recent_period[0], "end": recent_period[1]},
        "followup_period": {"start": followup_period[0], "end": followup_period[1]},
        "results": {},
    }

    for i, cfg in enumerate(cfgs, start=1):
        print(f"[recent {i}/{len(cfgs)}] {cfg.name}", flush=True)
        recent = run_blend_scoreband_backtest(
            cfg=cfg,
            start_date=recent_period[0],
            end_date=recent_period[1],
        )
        out["results"][cfg.name] = {
            "config": {
                "model_a": cfg.model_a,
                "model_b": cfg.model_b,
                "mode": cfg.mode,
                "weight_a": cfg.weight_a,
                "weight_up": cfg.weight_up,
                "weight_down": cfg.weight_down,
                "trend_threshold": cfg.trend_threshold,
            },
            "recent_3m": recent,
        }

        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(out, indent=2))

    ranking_recent = sorted(
        [
            {
                "name": k,
                "return_pct": float(v["recent_3m"]["cumulative_return_pct"]),
                "mdd_pct": float(v["recent_3m"]["max_drawdown_pct"]),
                "sharpe": float(v["recent_3m"]["sharpe_ratio"]),
                "recent_score": float(v["recent_3m"]["cumulative_return_pct"])
                - 0.05 * float(v["recent_3m"]["max_drawdown_pct"]),
            }
            for k, v in out["results"].items()
        ],
        key=lambda x: (x["recent_score"], x["return_pct"]),
        reverse=True,
    )
    out["ranking_recent"] = ranking_recent

    follow_names = [x["name"] for x in ranking_recent[: max(1, int(topn))]]
    out["followup_last1y"] = {}
    for name in follow_names:
        cfg_data = out["results"][name]["config"]
        cfg = BlendConfig(
            name=name,
            model_a=str(cfg_data["model_a"]),
            model_b=str(cfg_data["model_b"]),
            mode=str(cfg_data["mode"]),
            weight_a=float(cfg_data["weight_a"]),
            weight_up=float(cfg_data["weight_up"]),
            weight_down=float(cfg_data["weight_down"]),
            trend_threshold=float(cfg_data["trend_threshold"]),
        )
        print(f"[followup 1y] {name}", flush=True)
        rep = run_blend_scoreband_backtest(
            cfg=cfg,
            start_date=followup_period[0],
            end_date=followup_period[1],
        )
        out["followup_last1y"][name] = rep
        out["results"][name]["last_1y"] = rep
        Path(output).write_text(json.dumps(out, indent=2))

    if out["followup_last1y"]:
        out["ranking_followup_last1y"] = sorted(
            [
                {
                    "name": k,
                    "return_pct": float(v["cumulative_return_pct"]),
                    "mdd_pct": float(v["max_drawdown_pct"]),
                    "sharpe": float(v["sharpe_ratio"]),
                    "score": float(v["cumulative_return_pct"]) - 0.05 * float(v["max_drawdown_pct"]),
                }
                for k, v in out["followup_last1y"].items()
            ],
            key=lambda x: (x["score"], x["return_pct"]),
            reverse=True,
        )
    else:
        out["ranking_followup_last1y"] = []

    Path(output).write_text(json.dumps(out, indent=2))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Blend scoreband breakthrough scan")
    p.add_argument("--topn", type=int, default=3)
    p.add_argument(
        "--output",
        type=str,
        default="results/breakthrough_blend_scan_v1.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = run_scan(output=args.output, topn=args.topn)
    print("[saved]", args.output)
    print("top recent:", out.get("ranking_recent", [])[:3])
    print("top last1y:", out.get("ranking_followup_last1y", [])[:3])


if __name__ == "__main__":
    main()
