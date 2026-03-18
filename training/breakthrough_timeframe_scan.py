"""Broad breakthrough scan across timeframes and execution configs.

This script evaluates multiple execution configurations over fixed validation
periods, ranks by a robust score, then re-evaluates top candidates on blind
periods.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from evaluation.backtest import run_backtest


VAL_PERIODS: dict[str, tuple[str, str]] = {
    "2025-04": ("2025-04-01", "2025-04-30"),
    "2025-05": ("2025-05-01", "2025-05-31"),
    "2025-08": ("2025-08-01", "2025-08-31"),
    "2025-11": ("2025-11-01", "2025-11-30"),
}

BLIND_PERIODS: dict[str, tuple[str, str]] = {
    "2025-06": ("2025-06-01", "2025-06-30"),
    "2025-07": ("2025-07-01", "2025-07-31"),
    "2025-09": ("2025-09-01", "2025-09-30"),
    "2025-10": ("2025-10-01", "2025-10-31"),
}


EXP1 = "checkpoints/ppo_option_a_real_exp1.zip"
EXP3 = "checkpoints/ppo_option_a_real_exp3.zip"
EXP4 = "checkpoints/ppo_option_a_real_exp4_nosym.zip"
EXP5 = "checkpoints/ppo_option_a_real_w384_t262k_exp5.zip"
EXP6 = "checkpoints/ppo_option_a_real_w384_t786k_exp6_balanced.zip"


@dataclass
class Config:
    name: str
    params: dict[str, Any]


def default_configs() -> list[Config]:
    return [
        Config(
            "exp6_policy_det",
            {
                "model_path": EXP6,
                "window_size": 384,
                "decision_mode": "policy",
            },
        ),
        Config(
            "exp4_policy_det",
            {
                "model_path": EXP4,
                "window_size": 96,
                "decision_mode": "policy",
            },
        ),
        Config(
            "exp5_policy_det",
            {
                "model_path": EXP5,
                "window_size": 384,
                "decision_mode": "policy",
            },
        ),
        Config(
            "exp6_scoreband_mid",
            {
                "model_path": EXP6,
                "window_size": 384,
                "decision_mode": "score_band",
                "debiased_action": "mirror_scalar",
                "flat_start_policy": "as_is",
                "score_centering": "ema",
                "score_center_alpha": 0.02,
                "score_entry_threshold": 0.005,
                "score_flip_threshold": 0.02,
                "score_neutral_band": 0.001,
            },
        ),
        Config(
            "exp6_scoreband_safe",
            {
                "model_path": EXP6,
                "window_size": 384,
                "decision_mode": "score_band",
                "debiased_action": "mirror_scalar",
                "flat_start_policy": "as_is",
                "score_centering": "ema",
                "score_center_alpha": 0.02,
                "score_entry_threshold": 0.01,
                "score_flip_threshold": 0.03,
                "score_neutral_band": 0.002,
                "directional_tie_hold_eps": 0.005,
            },
        ),
        Config(
            "regime_raw_t002_c30_none",
            {
                "model_path": EXP6,
                "window_size": 384,
                "decision_mode": "regime_switch",
                "regime_model_up": EXP6,
                "regime_model_down": EXP4,
                "regime_neutral_policy": "hold",
                "regime_transition_mode": "none",
                "regime_score_mode": "raw",
                "regime_ret_short": 240,
                "regime_ret_long": 4320,
                "regime_ema_fast": 720,
                "regime_ema_slow": 4320,
                "regime_vol_lookback": 240,
                "regime_weight_ret_long": 1.0,
                "regime_weight_ema_slope": 1.2,
                "regime_weight_ret_short": 0.4,
                "regime_weight_vol": 0.0,
                "regime_weight_drawdown": 0.0,
                "regime_enter_threshold": 0.02,
                "regime_confirm_bars": 30,
            },
        ),
        Config(
            "regime_raw_t002_c12_none",
            {
                "model_path": EXP6,
                "window_size": 384,
                "decision_mode": "regime_switch",
                "regime_model_up": EXP6,
                "regime_model_down": EXP4,
                "regime_neutral_policy": "hold",
                "regime_transition_mode": "none",
                "regime_score_mode": "raw",
                "regime_ret_short": 240,
                "regime_ret_long": 4320,
                "regime_ema_fast": 720,
                "regime_ema_slow": 4320,
                "regime_vol_lookback": 240,
                "regime_weight_ret_long": 1.0,
                "regime_weight_ema_slope": 1.2,
                "regime_weight_ret_short": 0.4,
                "regime_weight_vol": 0.0,
                "regime_weight_drawdown": 0.0,
                "regime_enter_threshold": 0.02,
                "regime_confirm_bars": 12,
            },
        ),
        Config(
            "regime_raw_t003_c30_none",
            {
                "model_path": EXP6,
                "window_size": 384,
                "decision_mode": "regime_switch",
                "regime_model_up": EXP6,
                "regime_model_down": EXP4,
                "regime_neutral_policy": "hold",
                "regime_transition_mode": "none",
                "regime_score_mode": "raw",
                "regime_ret_short": 240,
                "regime_ret_long": 4320,
                "regime_ema_fast": 720,
                "regime_ema_slow": 4320,
                "regime_vol_lookback": 240,
                "regime_weight_ret_long": 1.0,
                "regime_weight_ema_slope": 1.2,
                "regime_weight_ret_short": 0.4,
                "regime_weight_vol": 0.0,
                "regime_weight_drawdown": 0.0,
                "regime_enter_threshold": 0.03,
                "regime_confirm_bars": 30,
            },
        ),
        Config(
            "regime_raw_t002_c30_switch",
            {
                "model_path": EXP6,
                "window_size": 384,
                "decision_mode": "regime_switch",
                "regime_model_up": EXP6,
                "regime_model_down": EXP4,
                "regime_neutral_policy": "hold",
                "regime_transition_mode": "force_on_switch",
                "regime_score_mode": "raw",
                "regime_ret_short": 240,
                "regime_ret_long": 4320,
                "regime_ema_fast": 720,
                "regime_ema_slow": 4320,
                "regime_vol_lookback": 240,
                "regime_weight_ret_long": 1.0,
                "regime_weight_ema_slope": 1.2,
                "regime_weight_ret_short": 0.4,
                "regime_weight_vol": 0.0,
                "regime_weight_drawdown": 0.0,
                "regime_enter_threshold": 0.02,
                "regime_confirm_bars": 30,
            },
        ),
        Config(
            "regime_raw_t002_c30_neutral_up",
            {
                "model_path": EXP6,
                "window_size": 384,
                "decision_mode": "regime_switch",
                "regime_model_up": EXP6,
                "regime_model_down": EXP4,
                "regime_neutral_policy": "up",
                "regime_transition_mode": "none",
                "regime_score_mode": "raw",
                "regime_ret_short": 240,
                "regime_ret_long": 4320,
                "regime_ema_fast": 720,
                "regime_ema_slow": 4320,
                "regime_vol_lookback": 240,
                "regime_weight_ret_long": 1.0,
                "regime_weight_ema_slope": 1.2,
                "regime_weight_ret_short": 0.4,
                "regime_weight_vol": 0.0,
                "regime_weight_drawdown": 0.0,
                "regime_enter_threshold": 0.02,
                "regime_confirm_bars": 30,
            },
        ),
        Config(
            "regime_raw_t002_c30_neutral_down",
            {
                "model_path": EXP6,
                "window_size": 384,
                "decision_mode": "regime_switch",
                "regime_model_up": EXP6,
                "regime_model_down": EXP4,
                "regime_neutral_policy": "down",
                "regime_transition_mode": "none",
                "regime_score_mode": "raw",
                "regime_ret_short": 240,
                "regime_ret_long": 4320,
                "regime_ema_fast": 720,
                "regime_ema_slow": 4320,
                "regime_vol_lookback": 240,
                "regime_weight_ret_long": 1.0,
                "regime_weight_ema_slope": 1.2,
                "regime_weight_ret_short": 0.4,
                "regime_weight_vol": 0.0,
                "regime_weight_drawdown": 0.0,
                "regime_enter_threshold": 0.02,
                "regime_confirm_bars": 30,
            },
        ),
        Config(
            "regime_raw_t002_c30_down_exp3",
            {
                "model_path": EXP6,
                "window_size": 384,
                "decision_mode": "regime_switch",
                "regime_model_up": EXP6,
                "regime_model_down": EXP3,
                "regime_neutral_policy": "hold",
                "regime_transition_mode": "none",
                "regime_score_mode": "raw",
                "regime_ret_short": 240,
                "regime_ret_long": 4320,
                "regime_ema_fast": 720,
                "regime_ema_slow": 4320,
                "regime_vol_lookback": 240,
                "regime_weight_ret_long": 1.0,
                "regime_weight_ema_slope": 1.2,
                "regime_weight_ret_short": 0.4,
                "regime_weight_vol": 0.0,
                "regime_weight_drawdown": 0.0,
                "regime_enter_threshold": 0.02,
                "regime_confirm_bars": 30,
            },
        ),
        Config(
            "regime_raw_t002_c30_up_exp5_down_exp4",
            {
                "model_path": EXP5,
                "window_size": 384,
                "decision_mode": "regime_switch",
                "regime_model_up": EXP5,
                "regime_model_down": EXP4,
                "regime_neutral_policy": "hold",
                "regime_transition_mode": "none",
                "regime_score_mode": "raw",
                "regime_ret_short": 240,
                "regime_ret_long": 4320,
                "regime_ema_fast": 720,
                "regime_ema_slow": 4320,
                "regime_vol_lookback": 240,
                "regime_weight_ret_long": 1.0,
                "regime_weight_ema_slope": 1.2,
                "regime_weight_ret_short": 0.4,
                "regime_weight_vol": 0.0,
                "regime_weight_drawdown": 0.0,
                "regime_enter_threshold": 0.02,
                "regime_confirm_bars": 30,
            },
        ),
        Config(
            "regime_z_t09_none",
            {
                "model_path": EXP6,
                "window_size": 384,
                "decision_mode": "regime_switch",
                "regime_model_up": EXP6,
                "regime_model_down": EXP4,
                "regime_neutral_policy": "hold",
                "regime_transition_mode": "none",
                "regime_score_mode": "zscore",
                "regime_enter_threshold": 0.9,
                "regime_confirm_bars": 3,
            },
        ),
    ]


def robust_summary(reports: dict[str, dict[str, Any]]) -> dict[str, float]:
    rets = [float(v["cumulative_return_pct"]) for v in reports.values()]
    mdds = [float(v["max_drawdown_pct"]) for v in reports.values()]
    mins = [float(v["min_sharpe"]) for v in reports.values()]

    m = mean(rets) if rets else 0.0
    s = pstdev(rets) if len(rets) > 1 else 0.0
    worst = min(rets) if rets else 0.0
    best = max(rets) if rets else 0.0
    mean_mdd = mean(mdds) if mdds else 0.0
    mean_min = mean(mins) if mins else 0.0

    robust = m + 0.2 * worst - 0.4 * s - 0.03 * mean_mdd
    return {
        "mean_return_pct": float(m),
        "std_return_pct": float(s),
        "worst_return_pct": float(worst),
        "best_return_pct": float(best),
        "mean_mdd_pct": float(mean_mdd),
        "mean_min_sharpe": float(mean_min),
        "robust_score": float(robust),
    }


def run_scan(
    *,
    symbol: str,
    market_type: str,
    timeframes: list[str],
    topn: int,
    output: str,
) -> dict[str, Any]:
    configs = default_configs()

    common_base = {
        "source": "binance",
        "symbol": symbol,
        "market_type": market_type,
        "deterministic": True,
        "hold_action_mode": None,
        "use_images": None,
        "image_cache_dir": "data/image_cache_backtest",
    }

    out: dict[str, Any] = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "validation_periods": VAL_PERIODS,
        "blind_periods": BLIND_PERIODS,
        "timeframes": timeframes,
        "results": {},
    }

    total_val_runs = len(timeframes) * len(configs) * len(VAL_PERIODS)
    idx = 0

    for tf in timeframes:
        tf_payload: dict[str, Any] = {"results": {}}
        for cfg in configs:
            val_reports: dict[str, dict[str, Any]] = {}
            for period_name, (start, end) in VAL_PERIODS.items():
                idx += 1
                print(
                    f"[VAL {idx}/{total_val_runs}] tf={tf} cfg={cfg.name} "
                    f"period={period_name} {start}..{end}",
                    flush=True,
                )
                kwargs = dict(common_base)
                kwargs.update(cfg.params)
                kwargs.update(
                    {
                        "timeframe": tf,
                        "start_date": start,
                        "end_date": end,
                    }
                )
                rep = run_backtest(**kwargs)
                val_reports[period_name] = rep

            tf_payload["results"][cfg.name] = {
                "params": cfg.params,
                "validation": val_reports,
                "validation_summary": robust_summary(val_reports),
            }

        ranking = sorted(
            [
                {
                    "name": name,
                    **payload["validation_summary"],
                }
                for name, payload in tf_payload["results"].items()
            ],
            key=lambda x: (float(x["robust_score"]), float(x["mean_return_pct"])),
            reverse=True,
        )
        tf_payload["ranking_validation"] = ranking

        top_candidates = [x["name"] for x in ranking[:topn]]
        tf_payload["topn_blind_eval"] = {}

        for cfg_name in top_candidates:
            blind_reports: dict[str, dict[str, Any]] = {}
            cfg = tf_payload["results"][cfg_name]
            params = cfg["params"]
            for period_name, (start, end) in BLIND_PERIODS.items():
                print(
                    f"[BLIND] tf={tf} cfg={cfg_name} period={period_name} {start}..{end}",
                    flush=True,
                )
                kwargs = dict(common_base)
                kwargs.update(params)
                kwargs.update(
                    {
                        "timeframe": tf,
                        "start_date": start,
                        "end_date": end,
                    }
                )
                rep = run_backtest(**kwargs)
                blind_reports[period_name] = rep

            blind_summary = robust_summary(blind_reports)
            tf_payload["topn_blind_eval"][cfg_name] = {
                "blind": blind_reports,
                "blind_summary": blind_summary,
            }

        out["results"][tf] = tf_payload

        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(out, indent=2))

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Breakthrough broad scan across timeframes")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--market-type", type=str, default="futures")
    parser.add_argument("--timeframes", type=str, default="5m,15m")
    parser.add_argument("--topn", type=int, default=3)
    parser.add_argument(
        "--output",
        type=str,
        default="results/breakthrough_timeframe_scan_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tfs = [x.strip() for x in str(args.timeframes).split(",") if x.strip()]
    report = run_scan(
        symbol=args.symbol,
        market_type=args.market_type,
        timeframes=tfs,
        topn=args.topn,
        output=args.output,
    )
    print("[done]", args.output)
    for tf in tfs:
        ranking = report["results"][tf]["ranking_validation"]
        print(tf, "top1", ranking[0]["name"], ranking[0]["robust_score"])


if __name__ == "__main__":
    main()
