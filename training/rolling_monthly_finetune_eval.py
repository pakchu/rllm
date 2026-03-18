"""Rolling monthly warm-start fine-tune evaluation for Option A (5m focus).

Unlike scratch retrain, this keeps prior weights and incrementally fine-tunes each month.
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
from training.data_sources import load_market_data
from training.train_sb3 import _augment_with_mirror_market, _build_vec_env


BASE_MODEL = "checkpoints/ppo_option_a_real_w384_t786k_exp6_balanced.zip"


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


def _make_train_env(train_start: str, train_end: str, seed: int):
    market_df = load_market_data(
        source="binance",
        input_csv=None,
        timeframe="5m",
        symbol="BTCUSDT",
        start_date=train_start,
        end_date=train_end,
        market_type="futures",
        seed=seed,
    )
    train_df = _augment_with_mirror_market(market_df)
    env = _build_vec_env(
        market_df=train_df,
        window_size=384,
        leverage=1.0,
        images=None,
        image_resolution=64,
        flat_hold_penalty=0.001,
        open_position_bonus=0.0002,
        directional_bias_penalty=0.001,
        directional_symmetry_prob=0.5,
        action_balance_penalty=0.0015,
        random_reset_start=True,
        hold_action_mode="flat",
        n_envs=1,
        vec_env="dummy",
    )
    return env


def run_eval(
    *,
    train_start: str,
    finetune_timesteps: int,
    output: str,
    ckpt_dir: str,
    seed_base: int = 2000,
) -> dict[str, Any]:
    from stable_baselines3 import PPO

    months = default_months()
    tests = months[1:]

    out: dict[str, Any] = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "setup": {
            "train_start": train_start,
            "finetune_timesteps": finetune_timesteps,
            "base_model": BASE_MODEL,
            "decision_params": _scoreband_mid_params(),
        },
        "folds": [],
    }

    ckpt_root = Path(ckpt_dir)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    current_model_path = BASE_MODEL

    common_bt = {
        "source": "binance",
        "symbol": "BTCUSDT",
        "timeframe": "5m",
        "market_type": "futures",
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

        seed = seed_base + idx
        ft_path = ckpt_root / f"rolling_finetune_5m_{test_month.label}_ft{finetune_timesteps}.zip"

        print(
            f"[FINETUNE {idx}/{len(tests)}] base={Path(current_model_path).name} "
            f"train={train_start}..{train_end} -> test={test_month.label}",
            flush=True,
        )

        env = _make_train_env(train_start, train_end, seed)

        model = PPO.load(current_model_path, env=env, seed=seed)
        model.learn(total_timesteps=finetune_timesteps, reset_num_timesteps=False, progress_bar=False)
        model.save(str(ft_path))
        current_model_path = str(ft_path)

        fold_row: dict[str, Any] = {
            "test_month": test_month.label,
            "train_range": {"start": train_start, "end": train_end},
            "test_range": {"start": test_month.start, "end": test_month.end},
            "finetuned_from": str(ft_path),
            "seed": seed,
        }

        f_kwargs = dict(common_bt)
        f_kwargs.update(
            {
                "model_path": BASE_MODEL,
                "start_date": test_month.start,
                "end_date": test_month.end,
            }
        )
        fixed_rep = run_backtest(**f_kwargs)

        t_kwargs = dict(common_bt)
        t_kwargs.update(
            {
                "model_path": str(ft_path),
                "start_date": test_month.start,
                "end_date": test_month.end,
            }
        )
        ft_rep = run_backtest(**t_kwargs)

        fold_row["fixed_test"] = fixed_rep
        fold_row["finetune_test"] = ft_rep
        fold_row["delta_return_pct"] = float(
            ft_rep["cumulative_return_pct"] - fixed_rep["cumulative_return_pct"]
        )
        out["folds"].append(fold_row)

        print(
            f"[DONE {test_month.label}] ft={ft_rep['cumulative_return_pct']:.3f}% "
            f"fixed={fixed_rep['cumulative_return_pct']:.3f}% "
            f"delta={fold_row['delta_return_pct']:.3f}%p",
            flush=True,
        )

        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(out, indent=2))

    ft_rows = [x["finetune_test"] for x in out["folds"]]
    fixed_rows = [x["fixed_test"] for x in out["folds"]]
    deltas = [float(x["delta_return_pct"]) for x in out["folds"]]

    out["summary"] = {
        "finetune": _summary_from_returns(ft_rows),
        "fixed": _summary_from_returns(fixed_rows),
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
    p = argparse.ArgumentParser(description="Rolling monthly warm-start fine-tune eval")
    p.add_argument("--train-start", type=str, default="2024-10-01")
    p.add_argument("--finetune-timesteps", type=int, default=16384)
    p.add_argument("--seed-base", type=int, default=2000)
    p.add_argument(
        "--ckpt-dir",
        type=str,
        default="checkpoints/rolling_monthly_finetune_5m_ft16384",
    )
    p.add_argument(
        "--output",
        type=str,
        default="results/rolling_monthly_finetune_5m_ft16384.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = run_eval(
        train_start=args.train_start,
        finetune_timesteps=args.finetune_timesteps,
        output=args.output,
        ckpt_dir=args.ckpt_dir,
        seed_base=args.seed_base,
    )
    print("[done]", args.output)
    print(json.dumps(report.get("summary", {}), indent=2))


if __name__ == "__main__":
    main()
