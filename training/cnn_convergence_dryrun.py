"""Quick convergence check for Option A CNN+PPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.trading_env import TradingEnv, TradingEnvConfig
from models.option_a import build_policy_kwargs
from preprocessing.chart_generator import build_images
from training.data_sources import make_synthetic_market_df


def eval_one_episode(model: PPO, eval_df, eval_images, cfg: TradingEnvConfig):
    env = TradingEnv(market_df=eval_df, images=eval_images, config=cfg)
    obs, info = env.reset()
    start_equity = float(info["equity"])
    total_reward = 0.0
    steps = 0
    action_counts = {0: 0, 1: 0, 2: 0}
    while True:
        action, _ = model.predict(obs, deterministic=True)
        action_i = int(action)
        if action_i in action_counts:
            action_counts[action_i] += 1
        obs, reward, terminated, truncated, info = env.step(action_i)
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break
    final_equity = float(info["equity"])
    ret_pct = (final_equity / start_equity - 1.0) * 100.0
    denom = max(1, steps)
    return {
        "steps": steps,
        "episode_reward": total_reward,
        "final_equity": final_equity,
        "cumulative_return_pct": ret_pct,
        "action_counts": {
            "buy": float(action_counts[0]),
            "hold": float(action_counts[1]),
            "sell": float(action_counts[2]),
        },
        "action_ratio": {
            "buy": action_counts[0] / denom,
            "hold": action_counts[1] / denom,
            "sell": action_counts[2] / denom,
        },
    }


def run_dryrun(
    output_json: str = "results/cnn_convergence_dryrun.json",
    train_rows: int = 5000,
    eval_rows: int = 3000,
    drift: float = 0.0,
    regime_amplitude: float = 0.0008,
    regime_period: int = 480,
    window_size: int = 32,
    image_resolution: int = 64,
    flat_hold_penalty: float = 0.001,
    directional_bias_penalty: float = 0.0,
    hold_action_mode: str = "flat",
    chunks: int = 8,
    chunk_steps: int = 4096,
) -> dict:
    train_df = make_synthetic_market_df(
        num_rows=train_rows,
        seed=42,
        drift=drift,
        regime_amplitude=regime_amplitude,
        regime_period=regime_period,
    )
    eval_df = make_synthetic_market_df(
        num_rows=eval_rows,
        seed=43,
        drift=drift,
        regime_amplitude=regime_amplitude,
        regime_period=regime_period,
    )

    train_images = build_images(
        train_df,
        window_size=window_size,
        resolution=image_resolution,
        cache_dir="data/image_cache_cnn_dry_train",
    )
    eval_images = build_images(
        eval_df,
        window_size=window_size,
        resolution=image_resolution,
        cache_dir="data/image_cache_cnn_dry_eval",
    )

    cfg = TradingEnvConfig(
        window_size=window_size,
        leverage=1.0,
        initial_equity=1000.0,
        image_shape=(3, image_resolution, image_resolution),
        flat_hold_penalty=flat_hold_penalty,
        directional_bias_penalty=directional_bias_penalty,
        hold_action_mode=hold_action_mode,
    )

    def _make_train_env():
        return Monitor(TradingEnv(market_df=train_df, images=train_images, config=cfg))

    train_env = DummyVecEnv([_make_train_env])
    model = PPO(
        "MultiInputPolicy",
        train_env,
        policy_kwargs=build_policy_kwargs(),
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.2,
        learning_rate=2.5e-4,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=42,
        verbose=0,
    )

    baseline = eval_one_episode(model, eval_df, eval_images, cfg)
    history = []
    for i in range(1, chunks + 1):
        model.learn(total_timesteps=chunk_steps, reset_num_timesteps=False, progress_bar=False)
        metrics = eval_one_episode(model, eval_df, eval_images, cfg)
        history.append({"chunk": i, **metrics})

    returns = [h["cumulative_return_pct"] for h in history]
    report = {
        "baseline": baseline,
        "history": history,
        "improved_vs_baseline": (
            returns[-1] > baseline["cumulative_return_pct"] if returns else False
        ),
        "improved_last_vs_first": returns[-1] > returns[0] if len(returns) >= 2 else False,
        "best_return": max(returns) if returns else baseline["cumulative_return_pct"],
        "best_chunk": (returns.index(max(returns)) + 1) if returns else 0,
        "hold_action_mode": hold_action_mode,
        "directional_bias_penalty": float(directional_bias_penalty),
        "synthetic_drift": float(drift),
        "synthetic_regime_amplitude": float(regime_amplitude),
        "synthetic_regime_period": float(regime_period),
    }

    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Option A CNN convergence dry-run.")
    parser.add_argument("--output-json", type=str, default="results/cnn_convergence_dryrun.json")
    parser.add_argument("--train-rows", type=int, default=5000)
    parser.add_argument("--eval-rows", type=int, default=3000)
    parser.add_argument("--drift", type=float, default=0.0)
    parser.add_argument("--regime-amplitude", type=float, default=0.0008)
    parser.add_argument("--regime-period", type=int, default=480)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--image-resolution", type=int, default=64)
    parser.add_argument("--flat-hold-penalty", type=float, default=0.001)
    parser.add_argument("--directional-bias-penalty", type=float, default=0.0)
    parser.add_argument(
        "--hold-action-mode",
        type=str,
        default="flat",
        choices=["flat", "maintain"],
    )
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--chunk-steps", type=int, default=4096)
    return parser.parse_args()


def main():
    args = parse_args()
    report = run_dryrun(
        output_json=args.output_json,
        train_rows=args.train_rows,
        eval_rows=args.eval_rows,
        drift=args.drift,
        regime_amplitude=args.regime_amplitude,
        regime_period=args.regime_period,
        window_size=args.window_size,
        image_resolution=args.image_resolution,
        flat_hold_penalty=args.flat_hold_penalty,
        directional_bias_penalty=args.directional_bias_penalty,
        hold_action_mode=args.hold_action_mode,
        chunks=args.chunks,
        chunk_steps=args.chunk_steps,
    )
    print(report)


if __name__ == "__main__":
    main()
