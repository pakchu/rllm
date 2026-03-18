"""Quick robustness sweep for Option A PPO settings."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

from evaluation.backtest import run_backtest_multi_seed
from training.train_sb3 import train_option_a_from_source


def _candidate_grid():
    learning_rates = [2.0e-4, 1.0e-4]
    ent_coefs = [0.02, 0.05]
    target_kls = [None, 0.03]
    for lr, ent, tkl in product(learning_rates, ent_coefs, target_kls):
        yield {
            "learning_rate": lr,
            "ent_coef": ent,
            "target_kl": tkl,
            "open_position_bonus": 0.001,
            "flat_hold_penalty": 0.001,
            "directional_bias_penalty": 0.0005,
            "directional_symmetry_prob": 0.5,
            "action_balance_penalty": 0.001,
            "random_reset_start": True,
            "hold_action_mode": "flat",
            "n_steps": 512,
            "batch_size": 128,
            "n_epochs": 5,
        }


def _score(result: dict) -> float:
    summary = result["summary"]
    mean_ret = float(summary["mean_cumulative_return_pct"])
    std_ret = float(summary["std_cumulative_return_pct"])
    max_ratio = float(result["mean_action_max_ratio"])
    collapse_penalty = max(0.0, max_ratio - 0.98) * 50.0
    return mean_ret - 0.5 * std_ret - collapse_penalty


def run_sweep(
    output_json: str = "results/robustness_sweep.json",
    train_rows: int = 3000,
    eval_rows: int = 2000,
    timesteps: int = 4096,
    train_seed: int = 42,
    eval_seeds: list[int] | None = None,
    window_size: int = 32,
    synthetic_drift: float = 0.0005,
    synthetic_regime_amplitude: float = 0.0004,
    synthetic_regime_period: int = 720,
    max_candidates: int = 8,
) -> dict:
    eval_seeds = eval_seeds or [101, 102, 103]
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_dir = out.parent / "robustness_models"
    run_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, cfg in enumerate(_candidate_grid(), start=1):
        if idx > max_candidates:
            break

        model_path = run_dir / f"candidate_{idx}.zip"
        train_option_a_from_source(
            source="synthetic",
            num_rows=train_rows,
            synthetic_drift=synthetic_drift,
            synthetic_regime_amplitude=synthetic_regime_amplitude,
            synthetic_regime_period=synthetic_regime_period,
            total_timesteps=timesteps,
            window_size=window_size,
            save_path=str(model_path),
            seed=train_seed,
            **cfg,
        )
        bt = run_backtest_multi_seed(
            model_path=str(model_path),
            seeds=eval_seeds,
            source="synthetic",
            num_rows=eval_rows,
            synthetic_drift=synthetic_drift,
            synthetic_regime_amplitude=synthetic_regime_amplitude,
            synthetic_regime_period=synthetic_regime_period,
            window_size=window_size,
            flat_hold_penalty=cfg["flat_hold_penalty"],
            deterministic=True,
            flat_start_policy="as_is",
        )
        action_maxes = [
            max(
                float(rep["action_ratio"]["buy"]),
                float(rep["action_ratio"]["hold"]),
                float(rep["action_ratio"]["sell"]),
            )
            for rep in bt["reports"]
        ]
        enriched = {
            "candidate_index": idx,
            "config": cfg,
            "model_path": str(model_path.resolve()),
            "summary": bt["summary"],
            "reports": bt["reports"],
            "mean_action_max_ratio": float(sum(action_maxes) / len(action_maxes)),
        }
        enriched["score"] = _score(enriched)
        results.append(enriched)

    best = max(results, key=lambda x: float(x["score"])) if results else None
    payload = {
        "train_seed": train_seed,
        "eval_seeds": eval_seeds,
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "timesteps": timesteps,
        "window_size": window_size,
        "synthetic_drift": synthetic_drift,
        "synthetic_regime_amplitude": synthetic_regime_amplitude,
        "synthetic_regime_period": float(synthetic_regime_period),
        "num_candidates": len(results),
        "best_candidate_index": (best["candidate_index"] if best else None),
        "best_score": (best["score"] if best else None),
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Option A robustness sweep.")
    parser.add_argument("--output-json", type=str, default="results/robustness_sweep.json")
    parser.add_argument("--train-rows", type=int, default=3000)
    parser.add_argument("--eval-rows", type=int, default=2000)
    parser.add_argument("--timesteps", type=int, default=4096)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--eval-seeds", type=str, default="101,102,103")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--synthetic-drift", type=float, default=0.0005)
    parser.add_argument("--synthetic-regime-amplitude", type=float, default=0.0004)
    parser.add_argument("--synthetic-regime-period", type=int, default=720)
    parser.add_argument("--max-candidates", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_seeds = [int(x.strip()) for x in args.eval_seeds.split(",") if x.strip()]
    report = run_sweep(
        output_json=args.output_json,
        train_rows=args.train_rows,
        eval_rows=args.eval_rows,
        timesteps=args.timesteps,
        train_seed=args.train_seed,
        eval_seeds=eval_seeds,
        window_size=args.window_size,
        synthetic_drift=args.synthetic_drift,
        synthetic_regime_amplitude=args.synthetic_regime_amplitude,
        synthetic_regime_period=args.synthetic_regime_period,
        max_candidates=args.max_candidates,
    )
    print(report)


if __name__ == "__main__":
    main()
