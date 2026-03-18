"""Leakage-safe target search for 3Y CAGR/MDD objective.

Objective:
  - evaluation period >= 3 years (includes latest 1 year)
  - target CAGR >= 50
  - strict MDD <= 15
  - no data leakage (train end < eval start)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation.backtest import run_backtest
from training.train_sb3 import train_option_a_from_source


@dataclass(frozen=True)
class TrainPreset:
    name: str
    mode: str  # existing | train
    model_path: str = ""
    timesteps: int = 131_072
    drawdown_penalty: float = 0.0
    seed: int = 1401


@dataclass(frozen=True)
class ExecPreset:
    name: str
    kwargs: dict[str, Any]


def _years_between(start: str, end: str) -> float:
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    days = max(1.0, float((e - s).days))
    return days / 365.25


def _cagr_from_return(cumulative_return_pct: float, years: float) -> float:
    gross = 1.0 + float(cumulative_return_pct) / 100.0
    if gross <= 0.0:
        return -100.0
    return float((gross ** (1.0 / years) - 1.0) * 100.0)


def _score(cagr_pct: float, mdd_pct: float, target_cagr: float, target_mdd: float) -> float:
    # Prioritize meeting MDD hard constraint first.
    mdd_penalty = max(0.0, float(mdd_pct) - float(target_mdd)) * 3.0
    cagr_gap = max(0.0, float(target_cagr) - float(cagr_pct))
    return float(cagr_pct - mdd_penalty - 1.5 * cagr_gap)


def _default_train_presets() -> list[TrainPreset]:
    return [
        TrainPreset(
            name="pre2023_exp8_existing",
            mode="existing",
            model_path="checkpoints/ppo_option_a_pre2023_15m_t131k_exp8.zip",
        ),
        TrainPreset(
            name="pre2023_bench8k_existing",
            mode="existing",
            model_path="checkpoints/_bench_pre2023_15m_t8k.zip",
        ),
        TrainPreset(
            name="pre2023_t131k_ddp0015_exp9",
            mode="train",
            timesteps=131_072,
            drawdown_penalty=0.0015,
            seed=1491,
        ),
        TrainPreset(
            name="pre2023_t131k_ddp0030_exp10",
            mode="train",
            timesteps=131_072,
            drawdown_penalty=0.0030,
            seed=1492,
        ),
    ]


def _default_exec_presets() -> list[ExecPreset]:
    return [
        ExecPreset("policy_det", {"decision_mode": "policy"}),
        ExecPreset(
            "scoreband_mid_exit003",
            {
                "decision_mode": "score_band",
                "debiased_action": "mirror_scalar",
                "flat_start_policy": "as_is",
                "score_centering": "ema",
                "score_center_alpha": 0.02,
                "score_entry_threshold": 0.005,
                "score_flip_threshold": 0.02,
                "score_neutral_band": 0.001,
                "score_exit_threshold": 0.003,
            },
        ),
        ExecPreset(
            "scoreband_safe_exit006",
            {
                "decision_mode": "score_band",
                "debiased_action": "mirror_scalar",
                "flat_start_policy": "as_is",
                "score_centering": "ema",
                "score_center_alpha": 0.02,
                "score_entry_threshold": 0.010,
                "score_flip_threshold": 0.03,
                "score_neutral_band": 0.002,
                "score_exit_threshold": 0.006,
                "directional_tie_hold_eps": 0.004,
            },
        ),
        ExecPreset(
            "scoreband_safe_exit006_vol09",
            {
                "decision_mode": "score_band",
                "debiased_action": "mirror_scalar",
                "flat_start_policy": "as_is",
                "score_centering": "ema",
                "score_center_alpha": 0.02,
                "score_entry_threshold": 0.010,
                "score_flip_threshold": 0.03,
                "score_neutral_band": 0.002,
                "score_exit_threshold": 0.006,
                "directional_tie_hold_eps": 0.004,
                "volatility_gate": "hard",
                "volatility_threshold": 0.09,
            },
        ),
        ExecPreset(
            "scoreband_safe_exit006_vol12",
            {
                "decision_mode": "score_band",
                "debiased_action": "mirror_scalar",
                "flat_start_policy": "as_is",
                "score_centering": "ema",
                "score_center_alpha": 0.02,
                "score_entry_threshold": 0.010,
                "score_flip_threshold": 0.03,
                "score_neutral_band": 0.002,
                "score_exit_threshold": 0.006,
                "directional_tie_hold_eps": 0.004,
                "volatility_gate": "hard",
                "volatility_threshold": 0.12,
            },
        ),
    ]


def run_search(
    *,
    train_start: str,
    train_end: str,
    eval_start: str,
    eval_end: str,
    timeframe: str,
    symbol: str,
    market_type: str,
    window_size: int,
    target_cagr: float,
    target_mdd: float,
    ckpt_dir: str,
    output: str,
    topn_last1y: int = 6,
) -> dict[str, Any]:
    if pd.to_datetime(train_end) >= pd.to_datetime(eval_start):
        raise ValueError(
            f"Leakage risk: train_end ({train_end}) must be strictly earlier than eval_start ({eval_start})."
        )

    years = _years_between(eval_start, eval_end)
    if years < 3.0:
        raise ValueError(f"Evaluation period must be >=3 years, got {years:.3f} years.")

    last1y_start = (
        pd.to_datetime(eval_end) - pd.DateOffset(years=1) + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    ckpt_root = Path(ckpt_dir)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    train_presets = _default_train_presets()
    exec_presets = _default_exec_presets()

    out: dict[str, Any] = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "objective": {
            "target_cagr_pct": float(target_cagr),
            "target_mdd_pct": float(target_mdd),
            "strict_no_leakage": True,
        },
        "periods": {
            "train": {"start": train_start, "end": train_end},
            "eval_3y": {"start": eval_start, "end": eval_end, "years": years},
            "eval_last1y": {"start": last1y_start, "end": eval_end},
        },
        "setup": {
            "timeframe": timeframe,
            "symbol": symbol,
            "market_type": market_type,
            "window_size": window_size,
        },
        "models": {},
        "results": [],
        "ranking": [],
        "top_last1y": [],
    }

    common_train = {
        "source": "binance",
        "symbol": symbol,
        "start_date": train_start,
        "end_date": train_end,
        "timeframe": timeframe,
        "market_type": market_type,
        "window_size": window_size,
        "leverage": 1.0,
        "use_images": False,
        "image_resolution": 64,
        "image_cache_dir": "data/image_cache_option_a_real",
        "flat_hold_penalty": 0.001,
        "open_position_bonus": 0.0002,
        "directional_bias_penalty": 0.001,
        "directional_symmetry_prob": 0.5,
        "action_balance_penalty": 0.0015,
        "random_reset_start": True,
        "hold_action_mode": "flat",
        "learning_rate": 1.5e-4,
        "ent_coef": 0.02,
        "n_steps": 2048,
        "batch_size": 512,
        "n_epochs": 10,
        "target_kl": None,
        "mirror_augmentation": True,
        "n_envs": 1,
        "vec_env": "dummy",
    }
    common_bt = {
        "source": "binance",
        "symbol": symbol,
        "timeframe": timeframe,
        "market_type": market_type,
        "start_date": eval_start,
        "end_date": eval_end,
        "window_size": window_size,
        "deterministic": True,
        "hold_action_mode": None,
        "use_images": None,
        "image_cache_dir": "data/image_cache_backtest",
    }

    for tp in train_presets:
        if tp.mode == "existing":
            model_path = str(Path(tp.model_path).resolve())
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Missing existing model: {model_path}")
            out["models"][tp.name] = {
                "mode": "existing",
                "model_path": model_path,
            }
            print(f"[MODEL] use existing: {tp.name} -> {model_path}", flush=True)
            continue

        model_path = ckpt_root / f"{tp.name}.zip"
        if model_path.exists():
            out["models"][tp.name] = {
                "mode": "train_cached",
                "model_path": str(model_path.resolve()),
                "timesteps": int(tp.timesteps),
                "drawdown_penalty": float(tp.drawdown_penalty),
                "seed": int(tp.seed),
            }
            print(f"[MODEL] reuse cached: {tp.name}", flush=True)
            continue

        train_kwargs = dict(common_train)
        train_kwargs.update(
            {
                "total_timesteps": int(tp.timesteps),
                "drawdown_penalty": float(tp.drawdown_penalty),
                "seed": int(tp.seed),
                "save_path": str(model_path),
            }
        )
        print(
            f"[TRAIN] {tp.name} timesteps={tp.timesteps} ddp={tp.drawdown_penalty}",
            flush=True,
        )
        saved = train_option_a_from_source(**train_kwargs)
        out["models"][tp.name] = {
            "mode": "train_new",
            "model_path": str(Path(saved).resolve()),
            "timesteps": int(tp.timesteps),
            "drawdown_penalty": float(tp.drawdown_penalty),
            "seed": int(tp.seed),
        }
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(out, indent=2))

    for model_name, model_meta in out["models"].items():
        model_path = str(model_meta["model_path"])
        for ep in exec_presets:
            bt_kwargs = dict(common_bt)
            bt_kwargs.update(ep.kwargs)
            bt_kwargs["model_path"] = model_path
            print(f"[EVAL] {model_name} + {ep.name}", flush=True)
            rep = run_backtest(**bt_kwargs)
            cagr = _cagr_from_return(float(rep["cumulative_return_pct"]), years=years)
            mdd = float(rep["strict_mdd_pct"])
            row = {
                "model": model_name,
                "execution": ep.name,
                "model_path": model_path,
                "report": rep,
                "cagr_pct": cagr,
                "target_hit": bool(cagr >= target_cagr and mdd <= target_mdd),
                "score": _score(
                    cagr_pct=cagr,
                    mdd_pct=mdd,
                    target_cagr=target_cagr,
                    target_mdd=target_mdd,
                ),
            }
            out["results"].append(row)
            Path(output).write_text(json.dumps(out, indent=2))

    ranked = sorted(out["results"], key=lambda x: float(x["score"]), reverse=True)
    out["ranking"] = [
        {
            "model": x["model"],
            "execution": x["execution"],
            "cagr_pct": float(x["cagr_pct"]),
            "mdd_pct": float(x["report"]["strict_mdd_pct"]),
            "return_pct": float(x["report"]["cumulative_return_pct"]),
            "sharpe": float(x["report"]["sharpe_ratio"]),
            "target_hit": bool(x["target_hit"]),
            "score": float(x["score"]),
        }
        for x in ranked
    ]

    for x in ranked[: max(1, int(topn_last1y))]:
        bt_kwargs = dict(common_bt)
        bt_kwargs.update(
            {
                "model_path": x["model_path"],
                "start_date": last1y_start,
                "end_date": eval_end,
            }
        )
        if x["execution"] != "policy_det":
            for ep in exec_presets:
                if ep.name == x["execution"]:
                    bt_kwargs.update(ep.kwargs)
                    break
        rep_1y = run_backtest(**bt_kwargs)
        years_1y = _years_between(last1y_start, eval_end)
        out["top_last1y"].append(
            {
                "model": x["model"],
                "execution": x["execution"],
                "period": {"start": last1y_start, "end": eval_end},
                "return_pct": float(rep_1y["cumulative_return_pct"]),
                "cagr_pct": _cagr_from_return(float(rep_1y["cumulative_return_pct"]), years_1y),
                "mdd_pct": float(rep_1y["strict_mdd_pct"]),
                "sharpe": float(rep_1y["sharpe_ratio"]),
                "trades": rep_1y.get("trade_counts", {}),
            }
        )

    Path(output).write_text(json.dumps(out, indent=2))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leakage-safe 3Y target search (CAGR/MDD)")
    p.add_argument("--train-start", type=str, default="2020-01-01")
    p.add_argument("--train-end", type=str, default="2022-12-31")
    p.add_argument("--eval-start", type=str, default="2023-01-01")
    p.add_argument("--eval-end", type=str, default="2026-02-28")
    p.add_argument("--timeframe", type=str, default="15m")
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--market-type", type=str, default="futures")
    p.add_argument("--window-size", type=int, default=384)
    p.add_argument("--target-cagr", type=float, default=50.0)
    p.add_argument("--target-mdd", type=float, default=15.0)
    p.add_argument("--topn-last1y", type=int, default=6)
    p.add_argument(
        "--ckpt-dir",
        type=str,
        default="checkpoints/target_3y_objective_search_v1",
    )
    p.add_argument(
        "--output",
        type=str,
        default="results/target_3y_objective_search_v1.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = run_search(
        train_start=args.train_start,
        train_end=args.train_end,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        timeframe=args.timeframe,
        symbol=args.symbol,
        market_type=args.market_type,
        window_size=args.window_size,
        target_cagr=args.target_cagr,
        target_mdd=args.target_mdd,
        ckpt_dir=args.ckpt_dir,
        output=args.output,
        topn_last1y=args.topn_last1y,
    )
    print("[done]", args.output)
    if report.get("ranking"):
        print("[top-1]", json.dumps(report["ranking"][0], indent=2))


if __name__ == "__main__":
    main()
