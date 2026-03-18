"""Expanding-window monthly retrain evaluation for Option A on real market data.

For each test month, retrain a fresh model on all prior data (expanding window),
then evaluate on that month. Compare against fixed baseline checkpoint.
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
from training.train_sb3 import train_option_a_from_source


FIXED_MODEL = "checkpoints/ppo_option_a_real_w384_t786k_exp6_balanced.zip"


@dataclass
class MonthRange:
    label: str
    start: str
    end: str


def default_months() -> list[MonthRange]:
    return [
        MonthRange("2025-03", "2025-03-01", "2025-03-31"),
        MonthRange("2025-04", "2025-04-01", "2025-04-30"),
        MonthRange("2025-05", "2025-05-01", "2025-05-31"),
        MonthRange("2025-06", "2025-06-01", "2025-06-30"),
        MonthRange("2025-07", "2025-07-01", "2025-07-31"),
        MonthRange("2025-08", "2025-08-01", "2025-08-31"),
        MonthRange("2025-09", "2025-09-01", "2025-09-30"),
        MonthRange("2025-10", "2025-10-01", "2025-10-31"),
        MonthRange("2025-11", "2025-11-01", "2025-11-30"),
        MonthRange("2025-12", "2025-12-01", "2025-12-31"),
        MonthRange("2026-01", "2026-01-01", "2026-01-31"),
        MonthRange("2026-02", "2026-02-01", "2026-02-28"),
    ]


def _scoreband_mid_params() -> dict[str, Any]:
    return {
        "decision_mode": "score_band",
        "debiased_action": "mirror_scalar",
        "flat_start_policy": "as_is",
        "score_centering": "ema",
        "score_center_alpha": 0.02,
        "score_entry_threshold": 0.005,
        "score_flip_threshold": 0.02,
        "score_neutral_band": 0.001,
    }


def _summary_from_returns(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "mean_return_pct": 0.0,
            "std_return_pct": 0.0,
            "worst_return_pct": 0.0,
            "best_return_pct": 0.0,
            "mean_mdd_pct": 0.0,
            "negative_months": 0,
            "months": 0,
            "compound_return_pct": 0.0,
            "robust_score": 0.0,
        }

    rets = [float(x["cumulative_return_pct"]) for x in rows]
    mdds = [float(x["max_drawdown_pct"]) for x in rows]
    m = mean(rets)
    s = pstdev(rets) if len(rets) > 1 else 0.0
    worst = min(rets)
    best = max(rets)
    mmdd = mean(mdds)
    eq = 1.0
    for r in rets:
        eq *= 1.0 + r / 100.0
    comp = (eq - 1.0) * 100.0
    robust = m + 0.2 * worst - 0.4 * s - 0.03 * mmdd
    return {
        "mean_return_pct": float(m),
        "std_return_pct": float(s),
        "worst_return_pct": float(worst),
        "best_return_pct": float(best),
        "mean_mdd_pct": float(mmdd),
        "negative_months": int(sum(1 for x in rets if x < 0.0)),
        "months": int(len(rows)),
        "compound_return_pct": float(comp),
        "robust_score": float(robust),
    }


def run_eval(
    *,
    train_start: str,
    timesteps: int,
    output: str,
    ckpt_dir: str,
    timeframe: str = "5m",
    symbol: str = "BTCUSDT",
    market_type: str = "futures",
    seed_base: int = 1000,
) -> dict[str, Any]:
    months = default_months()

    # Need previous month for training, so test starts at the second entry.
    tests = months[1:]

    out: dict[str, Any] = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "setup": {
            "train_start": train_start,
            "timesteps": timesteps,
            "timeframe": timeframe,
            "symbol": symbol,
            "market_type": market_type,
            "fixed_model": FIXED_MODEL,
            "decision_params": _scoreband_mid_params(),
        },
        "folds": [],
    }

    ckpt_root = Path(ckpt_dir)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    common_bt = {
        "source": "binance",
        "symbol": symbol,
        "timeframe": timeframe,
        "market_type": market_type,
        "window_size": 384,
        "deterministic": True,
        "hold_action_mode": None,
        "use_images": None,
        "image_cache_dir": "data/image_cache_backtest",
    }
    common_bt.update(_scoreband_mid_params())

    for idx, test_month in enumerate(tests, start=1):
        prev_month = months[idx - 1]
        train_end = prev_month.end

        model_path = ckpt_root / f"rolling_retrain_{timeframe}_{test_month.label}_t{timesteps}.zip"
        seed = seed_base + idx

        print(
            f"[TRAIN {idx}/{len(tests)}] train={train_start}..{train_end} "
            f"-> test={test_month.label}",
            flush=True,
        )

        saved = train_option_a_from_source(
            source="binance",
            symbol=symbol,
            start_date=train_start,
            end_date=train_end,
            timeframe=timeframe,
            market_type=market_type,
            total_timesteps=timesteps,
            window_size=384,
            leverage=1.0,
            use_images=False,
            image_resolution=64,
            image_cache_dir="data/image_cache_option_a_real",
            flat_hold_penalty=0.001,
            open_position_bonus=0.0002,
            directional_bias_penalty=0.001,
            directional_symmetry_prob=0.5,
            action_balance_penalty=0.0015,
            random_reset_start=True,
            hold_action_mode="flat",
            learning_rate=1.5e-4,
            ent_coef=0.02,
            n_steps=2048,
            batch_size=512,
            n_epochs=10,
            target_kl=None,
            mirror_augmentation=True,
            n_envs=1,
            vec_env="dummy",
            seed=seed,
            save_path=str(model_path),
        )

        fold_row: dict[str, Any] = {
            "test_month": test_month.label,
            "train_range": {"start": train_start, "end": train_end},
            "test_range": {"start": test_month.start, "end": test_month.end},
            "trained_model": saved,
            "seed": seed,
        }

        r_kwargs = dict(common_bt)
        r_kwargs.update(
            {
                "model_path": str(model_path),
                "start_date": test_month.start,
                "end_date": test_month.end,
            }
        )
        retrain_rep = run_backtest(**r_kwargs)
        fold_row["retrain_test"] = retrain_rep

        f_kwargs = dict(common_bt)
        f_kwargs.update(
            {
                "model_path": FIXED_MODEL,
                "start_date": test_month.start,
                "end_date": test_month.end,
            }
        )
        fixed_rep = run_backtest(**f_kwargs)
        fold_row["fixed_test"] = fixed_rep

        fold_row["delta_return_pct"] = float(
            retrain_rep["cumulative_return_pct"] - fixed_rep["cumulative_return_pct"]
        )
        out["folds"].append(fold_row)

        print(
            f"[DONE {test_month.label}] retrain={retrain_rep['cumulative_return_pct']:.3f}% "
            f"fixed={fixed_rep['cumulative_return_pct']:.3f}% "
            f"delta={fold_row['delta_return_pct']:.3f}%p",
            flush=True,
        )

        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(out, indent=2))

    retrain_rows = [x["retrain_test"] for x in out["folds"]]
    fixed_rows = [x["fixed_test"] for x in out["folds"]]
    deltas = [float(x["delta_return_pct"]) for x in out["folds"]]

    retrain_summary = _summary_from_returns(retrain_rows)
    fixed_summary = _summary_from_returns(fixed_rows)

    out["summary"] = {
        "retrain": retrain_summary,
        "fixed": fixed_summary,
        "delta": {
            "mean_return_delta_pct": float(mean(deltas) if deltas else 0.0),
            "median_return_delta_pct": float(sorted(deltas)[len(deltas) // 2] if deltas else 0.0),
            "wins": int(sum(1 for x in deltas if x > 0.0)),
            "losses": int(sum(1 for x in deltas if x < 0.0)),
            "months": int(len(deltas)),
        },
    }

    Path(output).write_text(json.dumps(out, indent=2))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rolling monthly retrain vs fixed evaluation")
    p.add_argument("--train-start", type=str, default="2024-10-01")
    p.add_argument("--timesteps", type=int, default=65536)
    p.add_argument("--timeframe", type=str, default="5m")
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--market-type", type=str, default="futures")
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--ckpt-dir", type=str, default="checkpoints/rolling_monthly_retrain_5m")
    p.add_argument(
        "--output",
        type=str,
        default="results/rolling_monthly_retrain_5m_t65536.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = run_eval(
        train_start=args.train_start,
        timesteps=args.timesteps,
        output=args.output,
        ckpt_dir=args.ckpt_dir,
        timeframe=args.timeframe,
        symbol=args.symbol,
        market_type=args.market_type,
        seed_base=args.seed_base,
    )
    print("[done]", args.output)
    print(json.dumps(report.get("summary", {}), indent=2))


if __name__ == "__main__":
    main()
