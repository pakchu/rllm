"""RLLM CLI entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.backtest import run_backtest, run_backtest_multi_seed
from training.data_sources import load_market_data
from training.prepare_dataset import prepare_observation_dataset, save_dataset_npz
from training.train_sb3 import train_option_a_from_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RLLM pipeline CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare-dataset", help="Generate image+scalar observation dataset")
    prep.add_argument(
        "--source",
        type=str,
        default="synthetic",
        choices=["synthetic", "csv", "binance"],
    )
    prep.add_argument("--input-csv", type=str, default="")
    prep.add_argument("--timeframe", type=str, default="1m")
    prep.add_argument("--symbol", type=str, default="BTCUSDT")
    prep.add_argument("--start-date", type=str, default="")
    prep.add_argument("--end-date", type=str, default="")
    prep.add_argument("--market-type", type=str, default="futures")
    prep.add_argument("--num-rows", type=int, default=8000)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--window-size", type=int, default=96)
    prep.add_argument("--resolution", type=int, default=320)
    prep.add_argument("--cache-dir", type=str, default="data/image_cache")
    prep.add_argument(
        "--image-render-backend",
        type=str,
        default="matplotlib",
        choices=["matplotlib", "fast"],
    )
    prep.add_argument("--output", type=str, default="data/observations.npz")

    train = sub.add_parser("train-smoke", help="Run Option A PPO smoke training")
    train.add_argument("--source", type=str, default="synthetic", choices=["synthetic", "csv"])
    train.add_argument("--input-csv", type=str, default="")
    train.add_argument("--timeframe", type=str, default="1m")
    train.add_argument("--num-rows", type=int, default=8000)
    train.add_argument("--synthetic-drift", type=float, default=0.0)
    train.add_argument("--synthetic-regime-amplitude", type=float, default=0.0004)
    train.add_argument("--synthetic-regime-period", type=int, default=720)
    train.add_argument("--timesteps", type=int, default=2048)
    train.add_argument("--window-size", type=int, default=96)
    train.add_argument("--leverage", type=float, default=1.0)
    train.add_argument("--use-images", action="store_true", default=False)
    train.add_argument("--image-resolution", type=int, default=64)
    train.add_argument("--image-cache-dir", type=str, default="data/image_cache_option_a")
    train.add_argument(
        "--image-render-backend",
        type=str,
        default="fast",
        choices=["matplotlib", "fast"],
    )
    train.add_argument("--flat-hold-penalty", type=float, default=0.0)
    train.add_argument("--open-position-bonus", type=float, default=0.0)
    train.add_argument("--directional-bias-penalty", type=float, default=0.0)
    train.add_argument("--directional-symmetry-prob", type=float, default=0.0)
    train.add_argument("--action-balance-penalty", type=float, default=0.0)
    train.add_argument("--drawdown-penalty", type=float, default=0.0)
    train.add_argument("--random-reset-start", action="store_true", default=False)
    train.add_argument("--episode-horizon", type=int, default=0)
    train.add_argument(
        "--scalar-feature-mode",
        type=str,
        default="legacy4",
        choices=["legacy4", "extended_v1"],
    )
    train.add_argument(
        "--hold-action-mode",
        type=str,
        default="flat",
        choices=["flat", "maintain"],
    )
    train.add_argument("--learning-rate", type=float, default=2.5e-4)
    train.add_argument("--ent-coef", type=float, default=0.01)
    train.add_argument("--n-steps", type=int, default=2048)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--n-epochs", type=int, default=10)
    train.add_argument("--target-kl", type=float, default=-1.0)
    train.add_argument("--mirror-augmentation", action="store_true", default=False)
    train.add_argument(
        "--visual-backbone",
        type=str,
        default="cnn",
        choices=[
            "cnn",
            "dinov2_reg_s",
            "dinov2_reg_b",
            "dinov3_vits16",
            "siglip2_base_224",
            "hf_custom",
        ],
    )
    train.add_argument("--visual-backbone-model", type=str, default="")
    train.add_argument(
        "--visual-pooling",
        type=str,
        default="cls_patch_mean",
        choices=["cls", "mean", "cls_patch_mean"],
    )
    train.add_argument("--unfreeze-visual-backbone", action="store_true", default=False)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--n-envs", type=int, default=1)
    train.add_argument("--vec-env", type=str, default="dummy", choices=["dummy", "subproc"])
    train.add_argument("--save-path", type=str, default="checkpoints/ppo_option_a_smoke.zip")

    train_real = sub.add_parser(
        "train-real", help="Run Option A PPO training on real Binance market data"
    )
    train_real.add_argument("--symbol", type=str, default="BTCUSDT")
    train_real.add_argument("--start-date", type=str, required=True)
    train_real.add_argument("--end-date", type=str, required=True)
    train_real.add_argument("--timeframe", type=str, default="1m")
    train_real.add_argument("--market-type", type=str, default="futures")
    train_real.add_argument("--timesteps", type=int, default=2048)
    train_real.add_argument("--window-size", type=int, default=96)
    train_real.add_argument("--leverage", type=float, default=1.0)
    train_real.add_argument("--use-images", action="store_true", default=False)
    train_real.add_argument("--image-resolution", type=int, default=64)
    train_real.add_argument(
        "--image-cache-dir", type=str, default="data/image_cache_option_a_real"
    )
    train_real.add_argument(
        "--image-render-backend",
        type=str,
        default="fast",
        choices=["matplotlib", "fast"],
    )
    train_real.add_argument("--flat-hold-penalty", type=float, default=0.0)
    train_real.add_argument("--open-position-bonus", type=float, default=0.0)
    train_real.add_argument("--directional-bias-penalty", type=float, default=0.0)
    train_real.add_argument("--directional-symmetry-prob", type=float, default=0.0)
    train_real.add_argument("--action-balance-penalty", type=float, default=0.0)
    train_real.add_argument("--drawdown-penalty", type=float, default=0.0)
    train_real.add_argument("--random-reset-start", action="store_true", default=False)
    train_real.add_argument("--episode-horizon", type=int, default=0)
    train_real.add_argument(
        "--scalar-feature-mode",
        type=str,
        default="legacy4",
        choices=["legacy4", "extended_v1"],
    )
    train_real.add_argument(
        "--hold-action-mode",
        type=str,
        default="flat",
        choices=["flat", "maintain"],
    )
    train_real.add_argument("--learning-rate", type=float, default=2.5e-4)
    train_real.add_argument("--ent-coef", type=float, default=0.01)
    train_real.add_argument("--n-steps", type=int, default=2048)
    train_real.add_argument("--batch-size", type=int, default=256)
    train_real.add_argument("--n-epochs", type=int, default=10)
    train_real.add_argument("--target-kl", type=float, default=-1.0)
    train_real.add_argument("--mirror-augmentation", action="store_true", default=False)
    train_real.add_argument(
        "--visual-backbone",
        type=str,
        default="cnn",
        choices=[
            "cnn",
            "dinov2_reg_s",
            "dinov2_reg_b",
            "dinov3_vits16",
            "siglip2_base_224",
            "hf_custom",
        ],
    )
    train_real.add_argument("--visual-backbone-model", type=str, default="")
    train_real.add_argument(
        "--visual-pooling",
        type=str,
        default="cls_patch_mean",
        choices=["cls", "mean", "cls_patch_mean"],
    )
    train_real.add_argument("--unfreeze-visual-backbone", action="store_true", default=False)
    train_real.add_argument("--n-envs", type=int, default=1)
    train_real.add_argument(
        "--vec-env", type=str, default="dummy", choices=["dummy", "subproc"]
    )
    train_real.add_argument("--seed", type=int, default=42)
    train_real.add_argument("--save-path", type=str, default="checkpoints/ppo_option_a_real.zip")

    train_vlm = sub.add_parser(
        "train-vlm-smoke", help="Run Option B VLM GRPO smoke training"
    )
    train_vlm.add_argument("--model-name", type=str, default="auto")
    train_vlm.add_argument(
        "--source",
        type=str,
        default="synthetic",
        choices=["synthetic", "csv", "binance"],
    )
    train_vlm.add_argument("--input-csv", type=str, default="")
    train_vlm.add_argument("--timeframe", type=str, default="1m")
    train_vlm.add_argument("--symbol", type=str, default="BTCUSDT")
    train_vlm.add_argument("--start-date", type=str, default="")
    train_vlm.add_argument("--end-date", type=str, default="")
    train_vlm.add_argument("--market-type", type=str, default="futures")
    train_vlm.add_argument("--num-rows", type=int, default=8000)
    train_vlm.add_argument("--synthetic-drift", type=float, default=0.0)
    train_vlm.add_argument("--synthetic-regime-amplitude", type=float, default=0.0004)
    train_vlm.add_argument("--synthetic-regime-period", type=int, default=720)
    train_vlm.add_argument("--window-size", type=int, default=96)
    train_vlm.add_argument("--resolution", type=int, default=224)
    train_vlm.add_argument("--cache-dir", type=str, default="data/image_cache_vlm")
    train_vlm.add_argument("--max-samples", type=int, default=256)
    train_vlm.add_argument(
        "--modality",
        type=str,
        default="multimodal",
        choices=["multimodal", "text_only"],
    )
    train_vlm.add_argument(
        "--action-schema",
        type=str,
        default="buy_hold_sell",
        choices=["buy_hold_sell", "trade_gate", "trade_side"],
    )
    train_vlm.add_argument(
        "--prompt-style",
        type=str,
        default="numeric",
        choices=["numeric", "symbolic", "hybrid"],
    )
    train_vlm.add_argument(
        "--prompt-feature-mode",
        type=str,
        default="basic_v0",
        choices=["basic_v0", "engineered_v1"],
    )
    train_vlm.add_argument("--hold-band", type=float, default=0.0005)
    train_vlm.add_argument("--target-horizon", type=int, default=1)
    train_vlm.add_argument(
        "--label-mode",
        type=str,
        default="next_return",
        choices=["next_return", "utility"],
    )
    train_vlm.add_argument("--buy-reward-weight", type=float, default=1.0)
    train_vlm.add_argument("--hold-reward-weight", type=float, default=1.0)
    train_vlm.add_argument("--sell-reward-weight", type=float, default=1.0)
    train_vlm.add_argument(
        "--reward-mode",
        type=str,
        default="classification",
        choices=["classification", "utility"],
    )
    train_vlm.add_argument("--utility-hold-margin", type=float, default=0.0)
    train_vlm.add_argument("--utility-fee-rate", type=float, default=0.0005)
    train_vlm.add_argument("--utility-slippage-rate", type=float, default=0.0001)
    train_vlm.add_argument("--utility-leverage", type=float, default=1.0)
    train_vlm.add_argument("--utility-stop-loss", type=float, default=-1.0)
    train_vlm.add_argument("--utility-take-profit", type=float, default=-1.0)
    train_vlm.add_argument(
        "--utility-use-log-return",
        type=str,
        default="true",
        choices=["true", "false"],
    )
    train_vlm.add_argument("--utility-base-risk-weight", type=float, default=0.0)
    train_vlm.add_argument("--utility-regime-weight-volatility", type=float, default=0.0)
    train_vlm.add_argument("--utility-regime-weight-downtrend", type=float, default=0.0)
    train_vlm.add_argument("--utility-regime-weight-drawdown", type=float, default=0.0)
    train_vlm.add_argument("--utility-min-risk-weight", type=float, default=0.0)
    train_vlm.add_argument("--utility-max-risk-weight", type=float, default=1.0)
    train_vlm.add_argument("--utility-hold-reward-bias", type=float, default=0.0)
    train_vlm.add_argument("--utility-reward-scale", type=float, default=400.0)
    train_vlm.add_argument("--utility-gap-scale", type=float, default=400.0)
    train_vlm.add_argument(
        "--sample-mode",
        type=str,
        default="balanced",
        choices=["sequential", "random", "balanced", "uniform"],
    )
    train_vlm.add_argument("--sample-seed", type=int, default=42)
    train_vlm.add_argument("--output-dir", type=str, default="checkpoints/vlm_grpo_smoke")
    train_vlm.add_argument("--max-steps", type=int, default=10)
    train_vlm.add_argument("--learning-rate", type=float, default=1e-5)
    train_vlm.add_argument("--per-device-train-batch-size", type=int, default=1)
    train_vlm.add_argument("--num-generations", type=int, default=2)
    train_vlm.add_argument("--temperature", type=float, default=2.0)
    train_vlm.add_argument("--top-p", type=float, default=0.95)
    train_vlm.add_argument("--top-k", type=int, default=50)
    train_vlm.add_argument("--max-completion-length", type=int, default=8)
    train_vlm.add_argument("--min-new-tokens", type=int, default=1)
    train_vlm.add_argument(
        "--do-sample",
        type=str,
        default="true",
        choices=["true", "false"],
    )
    train_vlm.add_argument("--log-completions", action="store_true", default=False)
    train_vlm.add_argument("--num-completions-to-print", type=int, default=2)
    train_vlm.add_argument(
        "--scale-rewards",
        type=str,
        default="batch",
        choices=["group", "batch", "none"],
    )
    train_vlm.add_argument(
        "--reward-variance-guard",
        type=str,
        default="auto",
        choices=["auto", "off"],
    )
    train_vlm.add_argument("--load-in-4bit", action="store_true", default=False)
    train_vlm.add_argument("--lora-r", type=int, default=8)
    train_vlm.add_argument("--lora-alpha", type=int, default=16)
    train_vlm.add_argument("--dry-run", action="store_true", default=False)

    eval_vlm = sub.add_parser(
        "eval-vlm", help="Evaluate Option B VLM policy (base or LoRA adapter)"
    )
    eval_vlm.add_argument("--model-name", type=str, default="auto")
    eval_vlm.add_argument("--adapter-dir", type=str, default="")
    eval_vlm.add_argument(
        "--source",
        type=str,
        default="synthetic",
        choices=["synthetic", "csv", "binance"],
    )
    eval_vlm.add_argument("--input-csv", type=str, default="")
    eval_vlm.add_argument("--timeframe", type=str, default="1m")
    eval_vlm.add_argument("--symbol", type=str, default="BTCUSDT")
    eval_vlm.add_argument("--start-date", type=str, default="")
    eval_vlm.add_argument("--end-date", type=str, default="")
    eval_vlm.add_argument("--market-type", type=str, default="futures")
    eval_vlm.add_argument("--num-rows", type=int, default=8000)
    eval_vlm.add_argument("--synthetic-drift", type=float, default=0.0)
    eval_vlm.add_argument("--synthetic-regime-amplitude", type=float, default=0.0004)
    eval_vlm.add_argument("--synthetic-regime-period", type=int, default=720)
    eval_vlm.add_argument("--window-size", type=int, default=96)
    eval_vlm.add_argument("--resolution", type=int, default=224)
    eval_vlm.add_argument("--cache-dir", type=str, default="data/image_cache_vlm")
    eval_vlm.add_argument(
        "--modality",
        type=str,
        default="multimodal",
        choices=["multimodal", "text_only"],
    )
    eval_vlm.add_argument(
        "--action-schema",
        type=str,
        default="buy_hold_sell",
        choices=["buy_hold_sell", "trade_gate", "trade_side"],
    )
    eval_vlm.add_argument(
        "--prompt-style",
        type=str,
        default="numeric",
        choices=["numeric", "symbolic", "hybrid"],
    )
    eval_vlm.add_argument(
        "--prompt-feature-mode",
        type=str,
        default="basic_v0",
        choices=["basic_v0", "engineered_v1"],
    )
    eval_vlm.add_argument("--hold-band", type=float, default=0.0005)
    eval_vlm.add_argument("--target-horizon", type=int, default=1)
    eval_vlm.add_argument(
        "--label-mode",
        type=str,
        default="next_return",
        choices=["next_return", "utility"],
    )
    eval_vlm.add_argument("--utility-hold-margin", type=float, default=0.0)
    eval_vlm.add_argument("--utility-fee-rate", type=float, default=0.0005)
    eval_vlm.add_argument("--utility-slippage-rate", type=float, default=0.0001)
    eval_vlm.add_argument("--utility-leverage", type=float, default=1.0)
    eval_vlm.add_argument("--utility-stop-loss", type=float, default=-1.0)
    eval_vlm.add_argument("--utility-take-profit", type=float, default=-1.0)
    eval_vlm.add_argument(
        "--utility-use-log-return",
        type=str,
        default="true",
        choices=["true", "false"],
    )
    eval_vlm.add_argument("--utility-base-risk-weight", type=float, default=0.0)
    eval_vlm.add_argument("--utility-regime-weight-volatility", type=float, default=0.0)
    eval_vlm.add_argument("--utility-regime-weight-downtrend", type=float, default=0.0)
    eval_vlm.add_argument("--utility-regime-weight-drawdown", type=float, default=0.0)
    eval_vlm.add_argument("--utility-min-risk-weight", type=float, default=0.0)
    eval_vlm.add_argument("--utility-max-risk-weight", type=float, default=1.0)
    eval_vlm.add_argument("--utility-hold-reward-bias", type=float, default=0.0)
    eval_vlm.add_argument("--max-samples", type=int, default=128)
    eval_vlm.add_argument(
        "--sample-mode",
        type=str,
        default="balanced",
        choices=["sequential", "random", "balanced", "uniform"],
    )
    eval_vlm.add_argument("--sample-seed", type=int, default=42)
    eval_vlm.add_argument("--max-completion-length", type=int, default=8)
    eval_vlm.add_argument("--min-new-tokens", type=int, default=1)
    eval_vlm.add_argument(
        "--do-sample",
        type=str,
        default="false",
        choices=["true", "false"],
    )
    eval_vlm.add_argument("--temperature", type=float, default=1.0)
    eval_vlm.add_argument("--top-p", type=float, default=1.0)
    eval_vlm.add_argument("--top-k", type=int, default=0)
    eval_vlm.add_argument(
        "--decision-mode",
        type=str,
        default="generate",
        choices=["generate", "likelihood"],
    )
    eval_vlm.add_argument("--action-bias-buy", type=float, default=0.0)
    eval_vlm.add_argument("--action-bias-hold", type=float, default=0.0)
    eval_vlm.add_argument("--action-bias-sell", type=float, default=0.0)
    eval_vlm.add_argument(
        "--store-action-scores",
        type=str,
        default="false",
        choices=["true", "false"],
    )
    eval_vlm.add_argument("--load-in-4bit", action="store_true", default=False)
    eval_vlm.add_argument("--output", type=str, default="results/eval_vlm.json")

    select_vlm = sub.add_parser(
        "select-vlm-checkpoint",
        help="Train/eval multiple VLM checkpoints and select best step",
    )
    select_vlm.add_argument("--candidate-steps", type=str, default="20,30,40")
    select_vlm.add_argument(
        "--output-root", type=str, default="results/vlm_checkpoint_selection"
    )
    select_vlm.add_argument("--model-name", type=str, default="auto")
    select_vlm.add_argument(
        "--source",
        type=str,
        default="synthetic",
        choices=["synthetic", "csv", "binance"],
    )
    select_vlm.add_argument("--input-csv", type=str, default="")
    select_vlm.add_argument("--timeframe", type=str, default="1m")
    select_vlm.add_argument("--symbol", type=str, default="BTCUSDT")
    select_vlm.add_argument("--start-date", type=str, default="")
    select_vlm.add_argument("--end-date", type=str, default="")
    select_vlm.add_argument("--market-type", type=str, default="futures")
    select_vlm.add_argument("--num-rows", type=int, default=8000)
    select_vlm.add_argument("--synthetic-drift", type=float, default=0.0)
    select_vlm.add_argument("--synthetic-regime-amplitude", type=float, default=0.0004)
    select_vlm.add_argument("--synthetic-regime-period", type=int, default=720)
    select_vlm.add_argument("--window-size", type=int, default=96)
    select_vlm.add_argument("--resolution", type=int, default=224)
    select_vlm.add_argument("--cache-dir", type=str, default="data/image_cache_vlm")
    select_vlm.add_argument(
        "--prompt-style",
        type=str,
        default="numeric",
        choices=["numeric", "symbolic", "hybrid"],
    )
    select_vlm.add_argument(
        "--prompt-feature-mode",
        type=str,
        default="basic_v0",
        choices=["basic_v0", "engineered_v1"],
    )
    select_vlm.add_argument("--max-samples", type=int, default=300)
    select_vlm.add_argument("--hold-band", type=float, default=0.0005)
    select_vlm.add_argument("--target-horizon", type=int, default=1)
    select_vlm.add_argument(
        "--label-mode",
        type=str,
        default="next_return",
        choices=["next_return", "utility"],
    )
    select_vlm.add_argument("--buy-reward-weight", type=float, default=1.0)
    select_vlm.add_argument("--hold-reward-weight", type=float, default=1.0)
    select_vlm.add_argument("--sell-reward-weight", type=float, default=1.0)
    select_vlm.add_argument(
        "--reward-mode",
        type=str,
        default="classification",
        choices=["classification", "utility"],
    )
    select_vlm.add_argument("--utility-hold-margin", type=float, default=0.0)
    select_vlm.add_argument("--utility-fee-rate", type=float, default=0.0005)
    select_vlm.add_argument("--utility-slippage-rate", type=float, default=0.0001)
    select_vlm.add_argument("--utility-leverage", type=float, default=1.0)
    select_vlm.add_argument("--utility-stop-loss", type=float, default=-1.0)
    select_vlm.add_argument("--utility-take-profit", type=float, default=-1.0)
    select_vlm.add_argument(
        "--utility-use-log-return",
        type=str,
        default="true",
        choices=["true", "false"],
    )
    select_vlm.add_argument("--utility-base-risk-weight", type=float, default=0.0)
    select_vlm.add_argument("--utility-regime-weight-volatility", type=float, default=0.0)
    select_vlm.add_argument("--utility-regime-weight-downtrend", type=float, default=0.0)
    select_vlm.add_argument("--utility-regime-weight-drawdown", type=float, default=0.0)
    select_vlm.add_argument("--utility-min-risk-weight", type=float, default=0.0)
    select_vlm.add_argument("--utility-max-risk-weight", type=float, default=1.0)
    select_vlm.add_argument("--utility-hold-reward-bias", type=float, default=0.0)
    select_vlm.add_argument("--utility-reward-scale", type=float, default=400.0)
    select_vlm.add_argument("--utility-gap-scale", type=float, default=400.0)
    select_vlm.add_argument(
        "--sample-mode",
        type=str,
        default="balanced",
        choices=["sequential", "random", "balanced", "uniform"],
    )
    select_vlm.add_argument("--sample-seed", type=int, default=22)
    select_vlm.add_argument("--learning-rate", type=float, default=1e-5)
    select_vlm.add_argument("--per-device-train-batch-size", type=int, default=1)
    select_vlm.add_argument("--num-generations", type=int, default=2)
    select_vlm.add_argument("--temperature", type=float, default=2.0)
    select_vlm.add_argument("--top-p", type=float, default=0.95)
    select_vlm.add_argument("--top-k", type=int, default=50)
    select_vlm.add_argument(
        "--scale-rewards",
        type=str,
        default="batch",
        choices=["group", "batch", "none"],
    )
    select_vlm.add_argument(
        "--reward-variance-guard",
        type=str,
        default="auto",
        choices=["auto", "off"],
    )
    select_vlm.add_argument("--max-completion-length", type=int, default=8)
    select_vlm.add_argument("--min-new-tokens", type=int, default=1)
    select_vlm.add_argument(
        "--do-sample", type=str, default="true", choices=["true", "false"]
    )
    select_vlm.add_argument("--eval-max-samples", type=int, default=90)
    select_vlm.add_argument(
        "--eval-sample-mode",
        type=str,
        default="balanced",
        choices=["sequential", "random", "balanced", "uniform"],
    )
    select_vlm.add_argument("--eval-sample-seed", type=int, default=33)
    select_vlm.add_argument(
        "--eval-decision-mode",
        type=str,
        default="generate",
        choices=["generate", "likelihood"],
    )
    select_vlm.add_argument("--eval-action-bias-buy", type=float, default=0.0)
    select_vlm.add_argument("--eval-action-bias-hold", type=float, default=0.0)
    select_vlm.add_argument("--eval-action-bias-sell", type=float, default=0.0)
    select_vlm.add_argument(
        "--eval-store-action-scores",
        type=str,
        default="false",
        choices=["true", "false"],
    )

    calibrate_vlm = sub.add_parser(
        "calibrate-vlm-bias",
        help="Grid-search action biases for VLM likelihood decision mode",
    )
    calibrate_vlm.add_argument(
        "--action-schema",
        type=str,
        default="buy_hold_sell",
        choices=["buy_hold_sell", "trade_gate", "trade_side"],
    )
    calibrate_vlm.add_argument(
        "--input-report",
        type=str,
        action="append",
        required=True,
        help="eval-vlm JSON path containing action_scores (repeatable)",
    )
    calibrate_vlm.add_argument("--buy-min", type=float, default=-0.8)
    calibrate_vlm.add_argument("--buy-max", type=float, default=0.8)
    calibrate_vlm.add_argument("--buy-step", type=float, default=0.1)
    calibrate_vlm.add_argument("--hold-min", type=float, default=-0.6)
    calibrate_vlm.add_argument("--hold-max", type=float, default=0.6)
    calibrate_vlm.add_argument("--hold-step", type=float, default=0.1)
    calibrate_vlm.add_argument("--sell-min", type=float, default=-0.8)
    calibrate_vlm.add_argument("--sell-max", type=float, default=0.8)
    calibrate_vlm.add_argument("--sell-step", type=float, default=0.1)
    calibrate_vlm.add_argument("--trade-min", type=float, default=-1.5)
    calibrate_vlm.add_argument("--trade-max", type=float, default=1.5)
    calibrate_vlm.add_argument("--trade-step", type=float, default=0.1)
    calibrate_vlm.add_argument("--no-trade-min", type=float, default=-1.5)
    calibrate_vlm.add_argument("--no-trade-max", type=float, default=1.5)
    calibrate_vlm.add_argument("--no-trade-step", type=float, default=0.1)
    calibrate_vlm.add_argument("--long-min", type=float, default=-1.5)
    calibrate_vlm.add_argument("--long-max", type=float, default=1.5)
    calibrate_vlm.add_argument("--long-step", type=float, default=0.1)
    calibrate_vlm.add_argument("--short-min", type=float, default=-1.5)
    calibrate_vlm.add_argument("--short-max", type=float, default=1.5)
    calibrate_vlm.add_argument("--short-step", type=float, default=0.1)
    calibrate_vlm.add_argument("--top-k", type=int, default=10)
    calibrate_vlm.add_argument("--weight-accuracy", type=float, default=1.0)
    calibrate_vlm.add_argument("--weight-balanced-recall", type=float, default=0.15)
    calibrate_vlm.add_argument("--weight-directional-mean", type=float, default=0.05)
    calibrate_vlm.add_argument("--weight-directional-gap", type=float, default=0.10)
    calibrate_vlm.add_argument("--require-min-recall", type=str, action="append", default=[])
    calibrate_vlm.add_argument("--require-min-pred-frac", type=str, action="append", default=[])
    calibrate_vlm.add_argument("--require-max-pred-frac", type=str, action="append", default=[])
    calibrate_vlm.add_argument("--output", type=str, default="results/vlm_bias_calibration.json")

    compose_gate_side = sub.add_parser(
        "compose-gate-side",
        help="Compose trade_gate and trade_side eval reports into BUY/HOLD/SELL report",
    )
    compose_gate_side.add_argument("--gate-report", type=str, required=True)
    compose_gate_side.add_argument("--side-report", type=str, required=True)
    compose_gate_side.add_argument("--floor-score", type=float, default=-1.0e9)
    compose_gate_side.add_argument("--gate-margin-threshold", type=float, default=0.0)
    compose_gate_side.add_argument("--side-margin-threshold", type=float, default=0.0)
    compose_gate_side.add_argument("--side-weight", type=float, default=1.0)
    compose_gate_side.add_argument(
        "--output",
        type=str,
        default="results/composed_gate_side_policy.json",
    )

    walkforward = sub.add_parser(
        "optiona-walkforward",
        help="Run Option A walk-forward policy selection on real market periods",
    )
    walkforward.add_argument("--model-path", type=str, required=True)
    walkforward.add_argument("--source", type=str, default="binance")
    walkforward.add_argument("--symbol", type=str, default="BTCUSDT")
    walkforward.add_argument("--timeframe", type=str, default="1m")
    walkforward.add_argument("--market-type", type=str, default="futures")
    walkforward.add_argument("--window-size", type=int, default=96)
    walkforward.add_argument("--leverage", type=float, default=1.0)
    walkforward.add_argument("--initial-equity", type=float, default=1000.0)
    walkforward.add_argument("--fee-rate", type=float, default=0.0005)
    walkforward.add_argument("--slippage-rate", type=float, default=0.0001)
    walkforward.add_argument("--flat-hold-penalty", type=float, default=0.001)
    walkforward.add_argument(
        "--hold-action-mode",
        type=str,
        default="auto",
        choices=["auto", "flat", "maintain"],
    )
    walkforward.add_argument(
        "--use-images",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
    )
    walkforward.add_argument("--image-cache-dir", type=str, default="data/image_cache_backtest")
    walkforward.add_argument("--score-center-alpha", type=float, default=0.02)
    walkforward.add_argument("--trend-guard", type=str, default="off", choices=["off", "hard"])
    walkforward.add_argument("--trend-threshold", type=float, default=0.002)
    walkforward.add_argument(
        "--folds",
        type=str,
        default="",
        help="Optional folds spec: val_start:val_end:test_start:test_end[, ...]",
    )
    walkforward.add_argument("--output", type=str, default="results/optiona_walkforward.json")

    cnn_dry = sub.add_parser("cnn-dryrun", help="Run CNN PPO convergence dry-run")
    cnn_dry.add_argument("--output-json", type=str, default="results/cnn_convergence_dryrun.json")
    cnn_dry.add_argument("--train-rows", type=int, default=5000)
    cnn_dry.add_argument("--eval-rows", type=int, default=3000)
    cnn_dry.add_argument("--drift", type=float, default=0.0)
    cnn_dry.add_argument("--regime-amplitude", type=float, default=0.0008)
    cnn_dry.add_argument("--regime-period", type=int, default=480)
    cnn_dry.add_argument("--window-size", type=int, default=32)
    cnn_dry.add_argument("--image-resolution", type=int, default=64)
    cnn_dry.add_argument("--flat-hold-penalty", type=float, default=0.001)
    cnn_dry.add_argument("--directional-bias-penalty", type=float, default=0.0)
    cnn_dry.add_argument(
        "--hold-action-mode",
        type=str,
        default="flat",
        choices=["flat", "maintain"],
    )
    cnn_dry.add_argument("--chunks", type=int, default=8)
    cnn_dry.add_argument("--chunk-steps", type=int, default=4096)

    sweep = sub.add_parser(
        "robust-sweep", help="Run Option A robustness sweep over PPO hyperparameters"
    )
    sweep.add_argument("--output-json", type=str, default="results/robustness_sweep.json")
    sweep.add_argument("--train-rows", type=int, default=3000)
    sweep.add_argument("--eval-rows", type=int, default=2000)
    sweep.add_argument("--timesteps", type=int, default=4096)
    sweep.add_argument("--train-seed", type=int, default=42)
    sweep.add_argument("--eval-seeds", type=str, default="101,102,103")
    sweep.add_argument("--window-size", type=int, default=32)
    sweep.add_argument("--synthetic-drift", type=float, default=0.0005)
    sweep.add_argument("--synthetic-regime-amplitude", type=float, default=0.0004)
    sweep.add_argument("--synthetic-regime-period", type=int, default=720)
    sweep.add_argument("--max-candidates", type=int, default=8)

    bt = sub.add_parser("backtest", help="Backtest saved PPO checkpoint")
    bt.add_argument(
        "--source",
        type=str,
        default="synthetic",
        choices=["synthetic", "csv", "binance"],
    )
    bt.add_argument("--input-csv", type=str, default="")
    bt.add_argument("--timeframe", type=str, default="1m")
    bt.add_argument("--symbol", type=str, default="BTCUSDT")
    bt.add_argument("--start-date", type=str, default="")
    bt.add_argument("--end-date", type=str, default="")
    bt.add_argument("--market-type", type=str, default="futures")
    bt.add_argument("--num-rows", type=int, default=8000)
    bt.add_argument("--seed", type=int, default=123)
    bt.add_argument("--synthetic-drift", type=float, default=0.0)
    bt.add_argument("--synthetic-regime-amplitude", type=float, default=0.0004)
    bt.add_argument("--synthetic-regime-period", type=int, default=720)
    bt.add_argument("--model-path", type=str, default="checkpoints/ppo_option_a_smoke.zip")
    bt.add_argument("--window-size", type=int, default=96)
    bt.add_argument("--leverage", type=float, default=1.0)
    bt.add_argument("--initial-equity", type=float, default=1000.0)
    bt.add_argument("--fee-rate", type=float, default=0.0005)
    bt.add_argument("--slippage-rate", type=float, default=0.0001)
    bt.add_argument("--flat-hold-penalty", type=float, default=0.0)
    bt.add_argument(
        "--hold-action-mode",
        type=str,
        default="auto",
        choices=["auto", "flat", "maintain"],
    )
    bt.add_argument(
        "--use-images",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
    )
    bt.add_argument("--image-cache-dir", type=str, default="data/image_cache_backtest")
    bt.add_argument(
        "--image-render-backend",
        type=str,
        default="auto",
        choices=["auto", "matplotlib", "fast"],
    )
    bt.add_argument(
        "--scalar-feature-mode",
        type=str,
        default="auto",
        choices=["auto", "legacy4", "extended_v1"],
    )
    bt.add_argument(
        "--deterministic",
        type=str,
        default="true",
        choices=["true", "false"],
    )
    bt.add_argument(
        "--decision-mode",
        type=str,
        default="policy",
        choices=["policy", "score_band", "regime_switch", "blend_score_band"],
    )
    bt.add_argument(
        "--flat-start-policy",
        type=str,
        default="as_is",
        choices=["as_is", "prefer_entry"],
    )
    bt.add_argument("--directional-tie-hold-eps", type=float, default=0.0)
    bt.add_argument(
        "--debiased-action",
        type=str,
        default="off",
        choices=["off", "mirror_scalar"],
    )
    bt.add_argument("--blend-model-a", type=str, default="")
    bt.add_argument("--blend-model-b", type=str, default="")
    bt.add_argument(
        "--blend-weight-mode",
        type=str,
        default="static",
        choices=["static", "trend"],
    )
    bt.add_argument("--blend-weight-a", type=float, default=0.75)
    bt.add_argument("--blend-weight-up", type=float, default=0.90)
    bt.add_argument("--blend-weight-down", type=float, default=0.35)
    bt.add_argument("--blend-trend-threshold", type=float, default=0.002)
    bt.add_argument("--score-entry-threshold", type=float, default=0.02)
    bt.add_argument("--score-flip-threshold", type=float, default=0.05)
    bt.add_argument("--score-neutral-band", type=float, default=0.005)
    bt.add_argument(
        "--score-exit-threshold",
        type=float,
        default=-1.0,
        help="If >=0, force HOLD(flat) when in-position and |score| <= threshold.",
    )
    bt.add_argument(
        "--score-centering",
        type=str,
        default="off",
        choices=["off", "ema"],
    )
    bt.add_argument("--score-center-alpha", type=float, default=0.02)
    bt.add_argument(
        "--trend-guard",
        type=str,
        default="off",
        choices=["off", "hard", "align_flat", "long_flat", "short_flat"],
    )
    bt.add_argument("--trend-threshold", type=float, default=0.002)
    bt.add_argument(
        "--volatility-gate",
        type=str,
        default="off",
        choices=["off", "hard"],
        help="Optional high-volatility gate; hard forces HOLD(flat) above threshold.",
    )
    bt.add_argument("--volatility-threshold", type=float, default=0.0)
    bt.add_argument("--regime-model-up", type=str, default="")
    bt.add_argument("--regime-model-down", type=str, default="")
    bt.add_argument(
        "--regime-neutral-policy",
        type=str,
        default="hold",
        choices=["hold", "up", "down"],
    )
    bt.add_argument(
        "--regime-transition-mode",
        type=str,
        default="force_align",
        choices=["none", "force_align", "force_on_switch"],
    )
    bt.add_argument("--regime-ret-short", type=int, default=60)
    bt.add_argument("--regime-ret-long", type=int, default=240)
    bt.add_argument("--regime-ema-fast", type=int, default=60)
    bt.add_argument("--regime-ema-slow", type=int, default=240)
    bt.add_argument("--regime-vol-lookback", type=int, default=60)
    bt.add_argument("--regime-z-window", type=int, default=1440)
    bt.add_argument(
        "--regime-score-mode",
        type=str,
        default="zscore",
        choices=["zscore", "raw"],
    )
    bt.add_argument("--regime-weight-ret-long", type=float, default=0.45)
    bt.add_argument("--regime-weight-ema-slope", type=float, default=0.35)
    bt.add_argument("--regime-weight-ret-short", type=float, default=0.20)
    bt.add_argument("--regime-weight-vol", type=float, default=-0.25)
    bt.add_argument("--regime-weight-drawdown", type=float, default=-0.15)
    bt.add_argument("--regime-enter-threshold", type=float, default=0.7)
    bt.add_argument("--regime-confirm-bars", type=int, default=3)
    bt.add_argument("--eval-seeds", type=str, default="")
    bt.add_argument("--output", type=str, default="results/backtest.json")
    return parser


def cmd_prepare_dataset(args: argparse.Namespace) -> None:
    market_df = load_market_data(
        source=args.source,
        input_csv=args.input_csv or None,
        timeframe=args.timeframe,
        symbol=args.symbol,
        start_date=args.start_date or None,
        end_date=args.end_date or None,
        market_type=args.market_type,
        num_rows=args.num_rows,
        seed=args.seed,
    )
    dataset = prepare_observation_dataset(
        market_df=market_df,
        window_size=args.window_size,
        resolution=args.resolution,
        cache_dir=args.cache_dir,
        image_render_backend=args.image_render_backend,
    )
    out = save_dataset_npz(dataset, output_path=args.output)
    print(
        f"[prepare-dataset] saved={out} "
        f"images={dataset['images'].shape} scalars={dataset['scalars'].shape}"
    )


def cmd_train_smoke(args: argparse.Namespace) -> None:
    out = train_option_a_from_source(
        source=args.source,
        input_csv=args.input_csv or None,
        timeframe=args.timeframe,
        num_rows=args.num_rows,
        synthetic_drift=args.synthetic_drift,
        synthetic_regime_amplitude=args.synthetic_regime_amplitude,
        synthetic_regime_period=args.synthetic_regime_period,
        total_timesteps=args.timesteps,
        window_size=args.window_size,
        leverage=args.leverage,
        use_images=args.use_images,
        image_resolution=args.image_resolution,
        image_cache_dir=args.image_cache_dir,
        image_render_backend=args.image_render_backend,
        flat_hold_penalty=args.flat_hold_penalty,
        open_position_bonus=args.open_position_bonus,
        directional_bias_penalty=args.directional_bias_penalty,
        directional_symmetry_prob=args.directional_symmetry_prob,
        action_balance_penalty=args.action_balance_penalty,
        drawdown_penalty=args.drawdown_penalty,
        random_reset_start=args.random_reset_start,
        episode_horizon=args.episode_horizon,
        scalar_feature_mode=args.scalar_feature_mode,
        hold_action_mode=args.hold_action_mode,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        target_kl=None if args.target_kl < 0 else args.target_kl,
        mirror_augmentation=args.mirror_augmentation,
        visual_backbone=args.visual_backbone,
        visual_backbone_model=args.visual_backbone_model or None,
        visual_pooling=args.visual_pooling,
        freeze_visual_backbone=not args.unfreeze_visual_backbone,
        seed=args.seed,
        n_envs=args.n_envs,
        vec_env=args.vec_env,
        save_path=args.save_path,
    )
    print(f"[train-smoke] saved={out}")


def cmd_train_real(args: argparse.Namespace) -> None:
    out = train_option_a_from_source(
        source="binance",
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe,
        market_type=args.market_type,
        total_timesteps=args.timesteps,
        window_size=args.window_size,
        leverage=args.leverage,
        use_images=args.use_images,
        image_resolution=args.image_resolution,
        image_cache_dir=args.image_cache_dir,
        image_render_backend=args.image_render_backend,
        flat_hold_penalty=args.flat_hold_penalty,
        open_position_bonus=args.open_position_bonus,
        directional_bias_penalty=args.directional_bias_penalty,
        directional_symmetry_prob=args.directional_symmetry_prob,
        action_balance_penalty=args.action_balance_penalty,
        drawdown_penalty=args.drawdown_penalty,
        random_reset_start=args.random_reset_start,
        episode_horizon=args.episode_horizon,
        scalar_feature_mode=args.scalar_feature_mode,
        hold_action_mode=args.hold_action_mode,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        target_kl=None if args.target_kl < 0 else args.target_kl,
        mirror_augmentation=args.mirror_augmentation,
        visual_backbone=args.visual_backbone,
        visual_backbone_model=args.visual_backbone_model or None,
        visual_pooling=args.visual_pooling,
        freeze_visual_backbone=not args.unfreeze_visual_backbone,
        n_envs=args.n_envs,
        vec_env=args.vec_env,
        seed=args.seed,
        save_path=args.save_path,
    )
    print(f"[train-real] saved={out}")


def cmd_train_vlm_smoke(args: argparse.Namespace) -> None:
    from training.train_vlm_grpo import train_vlm_grpo_smoke

    utility_stop_loss = (
        None if float(args.utility_stop_loss) <= 0.0 else float(args.utility_stop_loss)
    )
    utility_take_profit = (
        None if float(args.utility_take_profit) <= 0.0 else float(args.utility_take_profit)
    )
    out = train_vlm_grpo_smoke(
        model_name=args.model_name,
        source=args.source,
        input_csv=args.input_csv or None,
        timeframe=args.timeframe,
        symbol=args.symbol,
        start_date=args.start_date or None,
        end_date=args.end_date or None,
        market_type=args.market_type,
        num_rows=args.num_rows,
        synthetic_drift=args.synthetic_drift,
        synthetic_regime_amplitude=args.synthetic_regime_amplitude,
        synthetic_regime_period=args.synthetic_regime_period,
        window_size=args.window_size,
        resolution=args.resolution,
        cache_dir=args.cache_dir or None,
        max_samples=args.max_samples,
        modality=args.modality,
        action_schema=args.action_schema,
        prompt_style=args.prompt_style,
        prompt_feature_mode=args.prompt_feature_mode,
        hold_band=args.hold_band,
        target_horizon=args.target_horizon,
        label_mode=args.label_mode,
        buy_reward_weight=args.buy_reward_weight,
        hold_reward_weight=args.hold_reward_weight,
        sell_reward_weight=args.sell_reward_weight,
        reward_mode=args.reward_mode,
        utility_hold_margin=args.utility_hold_margin,
        utility_fee_rate=args.utility_fee_rate,
        utility_slippage_rate=args.utility_slippage_rate,
        utility_leverage=args.utility_leverage,
        utility_stop_loss=utility_stop_loss,
        utility_take_profit=utility_take_profit,
        utility_use_log_return=args.utility_use_log_return == "true",
        utility_base_risk_weight=args.utility_base_risk_weight,
        utility_regime_weight_volatility=args.utility_regime_weight_volatility,
        utility_regime_weight_downtrend=args.utility_regime_weight_downtrend,
        utility_regime_weight_drawdown=args.utility_regime_weight_drawdown,
        utility_min_risk_weight=args.utility_min_risk_weight,
        utility_max_risk_weight=args.utility_max_risk_weight,
        utility_hold_reward_bias=args.utility_hold_reward_bias,
        utility_reward_scale=args.utility_reward_scale,
        utility_gap_scale=args.utility_gap_scale,
        sample_mode=args.sample_mode,
        sample_seed=args.sample_seed,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_completion_length=args.max_completion_length,
        min_new_tokens=args.min_new_tokens,
        do_sample=args.do_sample == "true",
        log_completions=args.log_completions,
        num_completions_to_print=args.num_completions_to_print,
        scale_rewards=args.scale_rewards,
        reward_variance_guard=args.reward_variance_guard,
        load_in_4bit=args.load_in_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        dry_run=args.dry_run,
    )
    print(f"[train-vlm-smoke] saved={out}")


def cmd_eval_vlm(args: argparse.Namespace) -> None:
    from training.eval_vlm_policy import evaluate_vlm_policy

    utility_stop_loss = (
        None if float(args.utility_stop_loss) <= 0.0 else float(args.utility_stop_loss)
    )
    utility_take_profit = (
        None if float(args.utility_take_profit) <= 0.0 else float(args.utility_take_profit)
    )
    report = evaluate_vlm_policy(
        model_name=args.model_name,
        adapter_dir=args.adapter_dir or None,
        source=args.source,
        input_csv=args.input_csv or None,
        timeframe=args.timeframe,
        symbol=args.symbol,
        start_date=args.start_date or None,
        end_date=args.end_date or None,
        market_type=args.market_type,
        num_rows=args.num_rows,
        synthetic_drift=args.synthetic_drift,
        synthetic_regime_amplitude=args.synthetic_regime_amplitude,
        synthetic_regime_period=args.synthetic_regime_period,
        window_size=args.window_size,
        resolution=args.resolution,
        cache_dir=args.cache_dir or None,
        modality=args.modality,
        action_schema=args.action_schema,
        prompt_style=args.prompt_style,
        prompt_feature_mode=args.prompt_feature_mode,
        hold_band=args.hold_band,
        target_horizon=args.target_horizon,
        label_mode=args.label_mode,
        utility_hold_margin=args.utility_hold_margin,
        utility_fee_rate=args.utility_fee_rate,
        utility_slippage_rate=args.utility_slippage_rate,
        utility_leverage=args.utility_leverage,
        utility_stop_loss=utility_stop_loss,
        utility_take_profit=utility_take_profit,
        utility_use_log_return=args.utility_use_log_return == "true",
        utility_base_risk_weight=args.utility_base_risk_weight,
        utility_regime_weight_volatility=args.utility_regime_weight_volatility,
        utility_regime_weight_downtrend=args.utility_regime_weight_downtrend,
        utility_regime_weight_drawdown=args.utility_regime_weight_drawdown,
        utility_min_risk_weight=args.utility_min_risk_weight,
        utility_max_risk_weight=args.utility_max_risk_weight,
        utility_hold_reward_bias=args.utility_hold_reward_bias,
        max_samples=args.max_samples,
        sample_mode=args.sample_mode,
        sample_seed=args.sample_seed,
        max_completion_length=args.max_completion_length,
        min_new_tokens=args.min_new_tokens,
        do_sample=args.do_sample == "true",
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        decision_mode=args.decision_mode,
        action_bias_buy=args.action_bias_buy,
        action_bias_hold=args.action_bias_hold,
        action_bias_sell=args.action_bias_sell,
        store_action_scores=args.store_action_scores == "true",
        output=args.output,
    )
    print(f"[eval-vlm] saved={Path(args.output).resolve()}")
    print(report["metrics"])


def cmd_compose_gate_side(args: argparse.Namespace) -> None:
    from training.compose_gate_side_policy import compose_gate_side_reports

    report = compose_gate_side_reports(
        gate_report_path=args.gate_report,
        side_report_path=args.side_report,
        output_path=args.output,
        floor_score=args.floor_score,
        gate_margin_threshold=args.gate_margin_threshold,
        side_margin_threshold=args.side_margin_threshold,
        side_weight=args.side_weight,
    )
    print(f"[compose-gate-side] saved={Path(args.output).resolve()}")
    print(report["metrics"])


def cmd_select_vlm_checkpoint(args: argparse.Namespace) -> None:
    from training.select_vlm_checkpoint import parse_step_list, select_best_vlm_checkpoint

    utility_stop_loss = (
        None if float(args.utility_stop_loss) <= 0.0 else float(args.utility_stop_loss)
    )
    utility_take_profit = (
        None if float(args.utility_take_profit) <= 0.0 else float(args.utility_take_profit)
    )
    report = select_best_vlm_checkpoint(
        candidate_steps=parse_step_list(args.candidate_steps),
        output_root=args.output_root,
        model_name=args.model_name,
        source=args.source,
        input_csv=args.input_csv or None,
        timeframe=args.timeframe,
        symbol=args.symbol,
        start_date=args.start_date or None,
        end_date=args.end_date or None,
        market_type=args.market_type,
        num_rows=args.num_rows,
        synthetic_drift=args.synthetic_drift,
        synthetic_regime_amplitude=args.synthetic_regime_amplitude,
        synthetic_regime_period=args.synthetic_regime_period,
        window_size=args.window_size,
        resolution=args.resolution,
        cache_dir=args.cache_dir or None,
        prompt_style=args.prompt_style,
        prompt_feature_mode=args.prompt_feature_mode,
        max_samples=args.max_samples,
        hold_band=args.hold_band,
        target_horizon=args.target_horizon,
        label_mode=args.label_mode,
        buy_reward_weight=args.buy_reward_weight,
        hold_reward_weight=args.hold_reward_weight,
        sell_reward_weight=args.sell_reward_weight,
        reward_mode=args.reward_mode,
        utility_hold_margin=args.utility_hold_margin,
        utility_fee_rate=args.utility_fee_rate,
        utility_slippage_rate=args.utility_slippage_rate,
        utility_leverage=args.utility_leverage,
        utility_stop_loss=utility_stop_loss,
        utility_take_profit=utility_take_profit,
        utility_use_log_return=args.utility_use_log_return == "true",
        utility_base_risk_weight=args.utility_base_risk_weight,
        utility_regime_weight_volatility=args.utility_regime_weight_volatility,
        utility_regime_weight_downtrend=args.utility_regime_weight_downtrend,
        utility_regime_weight_drawdown=args.utility_regime_weight_drawdown,
        utility_min_risk_weight=args.utility_min_risk_weight,
        utility_max_risk_weight=args.utility_max_risk_weight,
        utility_hold_reward_bias=args.utility_hold_reward_bias,
        utility_reward_scale=args.utility_reward_scale,
        utility_gap_scale=args.utility_gap_scale,
        sample_mode=args.sample_mode,
        sample_seed=args.sample_seed,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        scale_rewards=args.scale_rewards,
        reward_variance_guard=args.reward_variance_guard,
        max_completion_length=args.max_completion_length,
        min_new_tokens=args.min_new_tokens,
        do_sample=args.do_sample == "true",
        eval_max_samples=args.eval_max_samples,
        eval_sample_mode=args.eval_sample_mode,
        eval_sample_seed=args.eval_sample_seed,
        eval_decision_mode=args.eval_decision_mode,
        eval_action_bias_buy=args.eval_action_bias_buy,
        eval_action_bias_hold=args.eval_action_bias_hold,
        eval_action_bias_sell=args.eval_action_bias_sell,
        eval_store_action_scores=args.eval_store_action_scores == "true",
    )
    print(
        f"[select-vlm-checkpoint] best_step={report['best_step']} "
        f"best_score={report['best_score']:.6f} "
        f"best_checkpoint={report['best_checkpoint_dir']}"
    )


def cmd_calibrate_vlm_bias(args: argparse.Namespace) -> None:
    from models.option_b_vlm import get_action_labels
    from training.calibrate_vlm_bias import (
        _bias_grid_from_legacy_args,
        _parse_label_value_specs,
        calibrate_action_biases,
        load_action_scores,
    )

    rows = load_action_scores(args.input_report)
    labels = get_action_labels(args.action_schema)
    bias_grid = _bias_grid_from_legacy_args(
        labels,
        buy_min=args.buy_min,
        buy_max=args.buy_max,
        buy_step=args.buy_step,
        hold_min=args.hold_min,
        hold_max=args.hold_max,
        hold_step=args.hold_step,
        sell_min=args.sell_min,
        sell_max=args.sell_max,
        sell_step=args.sell_step,
        trade_min=args.trade_min,
        trade_max=args.trade_max,
        trade_step=args.trade_step,
        no_trade_min=args.no_trade_min,
        no_trade_max=args.no_trade_max,
        no_trade_step=args.no_trade_step,
        long_min=args.long_min,
        long_max=args.long_max,
        long_step=args.long_step,
        short_min=args.short_min,
        short_max=args.short_max,
        short_step=args.short_step,
    )
    report = calibrate_action_biases(
        rows=rows,
        labels=labels,
        bias_grid=bias_grid,
        min_recall_by_label=_parse_label_value_specs(args.require_min_recall, name="require-min-recall"),
        min_pred_frac_by_label=_parse_label_value_specs(args.require_min_pred_frac, name="require-min-pred-frac"),
        max_pred_frac_by_label=_parse_label_value_specs(args.require_max_pred_frac, name="require-max-pred-frac"),
        top_k=args.top_k,
        weight_accuracy=args.weight_accuracy,
        weight_balanced_recall=args.weight_balanced_recall,
        weight_directional_mean=args.weight_directional_mean,
        weight_directional_gap=args.weight_directional_gap,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"[calibrate-vlm-bias] saved={out.resolve()}")
    print(report["best"])


def cmd_optiona_walkforward(args: argparse.Namespace) -> None:
    from training.optiona_walkforward import (
        default_walkforward_folds,
        parse_folds_spec,
        run_optiona_walkforward,
    )

    folds = parse_folds_spec(args.folds) if args.folds.strip() else default_walkforward_folds()
    report = run_optiona_walkforward(
        model_path=args.model_path,
        source=args.source,
        symbol=args.symbol,
        timeframe=args.timeframe,
        market_type=args.market_type,
        window_size=args.window_size,
        leverage=args.leverage,
        initial_equity=args.initial_equity,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        flat_hold_penalty=args.flat_hold_penalty,
        hold_action_mode=args.hold_action_mode,
        use_images=args.use_images,
        image_cache_dir=args.image_cache_dir,
        score_center_alpha=args.score_center_alpha,
        trend_guard=args.trend_guard,
        trend_threshold=args.trend_threshold,
        folds=folds,
        output=args.output,
    )
    print(f"[optiona-walkforward] saved={Path(args.output).resolve()}")
    print(report["summary"])


def cmd_cnn_dryrun(args: argparse.Namespace) -> None:
    from training.cnn_convergence_dryrun import run_dryrun

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


def cmd_robust_sweep(args: argparse.Namespace) -> None:
    from training.robustness_sweep import run_sweep

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


def cmd_backtest(args: argparse.Namespace) -> None:
    common_kwargs = dict(
        source=args.source,
        input_csv=args.input_csv or None,
        timeframe=args.timeframe,
        symbol=args.symbol,
        start_date=args.start_date or None,
        end_date=args.end_date or None,
        market_type=args.market_type,
        num_rows=args.num_rows,
        window_size=args.window_size,
        leverage=args.leverage,
        initial_equity=args.initial_equity,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        flat_hold_penalty=args.flat_hold_penalty,
        hold_action_mode=(None if args.hold_action_mode == "auto" else args.hold_action_mode),
        use_images=(None if args.use_images == "auto" else args.use_images == "true"),
        image_cache_dir=args.image_cache_dir or None,
        image_render_backend=(
            None if args.image_render_backend == "auto" else args.image_render_backend
        ),
        scalar_feature_mode=(
            None if args.scalar_feature_mode == "auto" else args.scalar_feature_mode
        ),
        deterministic=args.deterministic == "true",
        decision_mode=args.decision_mode,
        flat_start_policy=args.flat_start_policy,
        directional_tie_hold_eps=args.directional_tie_hold_eps,
        debiased_action=args.debiased_action,
        blend_model_a=args.blend_model_a or None,
        blend_model_b=args.blend_model_b or None,
        blend_weight_mode=args.blend_weight_mode,
        blend_weight_a=args.blend_weight_a,
        blend_weight_up=args.blend_weight_up,
        blend_weight_down=args.blend_weight_down,
        blend_trend_threshold=args.blend_trend_threshold,
        score_entry_threshold=args.score_entry_threshold,
        score_flip_threshold=args.score_flip_threshold,
        score_neutral_band=args.score_neutral_band,
        score_exit_threshold=args.score_exit_threshold,
        score_centering=args.score_centering,
        score_center_alpha=args.score_center_alpha,
        trend_guard=args.trend_guard,
        trend_threshold=args.trend_threshold,
        volatility_gate=args.volatility_gate,
        volatility_threshold=args.volatility_threshold,
        regime_model_up=args.regime_model_up or None,
        regime_model_down=args.regime_model_down or None,
        regime_neutral_policy=args.regime_neutral_policy,
        regime_transition_mode=args.regime_transition_mode,
        regime_ret_short=args.regime_ret_short,
        regime_ret_long=args.regime_ret_long,
        regime_ema_fast=args.regime_ema_fast,
        regime_ema_slow=args.regime_ema_slow,
        regime_vol_lookback=args.regime_vol_lookback,
        regime_z_window=args.regime_z_window,
        regime_score_mode=args.regime_score_mode,
        regime_weight_ret_long=args.regime_weight_ret_long,
        regime_weight_ema_slope=args.regime_weight_ema_slope,
        regime_weight_ret_short=args.regime_weight_ret_short,
        regime_weight_vol=args.regime_weight_vol,
        regime_weight_drawdown=args.regime_weight_drawdown,
        regime_enter_threshold=args.regime_enter_threshold,
        regime_confirm_bars=args.regime_confirm_bars,
    )
    if args.eval_seeds.strip():
        seeds = [int(x.strip()) for x in args.eval_seeds.split(",") if x.strip()]
        report = run_backtest_multi_seed(
            model_path=args.model_path,
            seeds=seeds,
            **common_kwargs,
        )
    else:
        report = run_backtest(
            model_path=args.model_path,
            seed=args.seed,
            **common_kwargs,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(f"[backtest] saved={output.resolve()}")
    print(report)


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "prepare-dataset":
        cmd_prepare_dataset(args)
    elif args.cmd == "train-smoke":
        cmd_train_smoke(args)
    elif args.cmd == "train-real":
        cmd_train_real(args)
    elif args.cmd == "train-vlm-smoke":
        cmd_train_vlm_smoke(args)
    elif args.cmd == "eval-vlm":
        cmd_eval_vlm(args)
    elif args.cmd == "select-vlm-checkpoint":
        cmd_select_vlm_checkpoint(args)
    elif args.cmd == "calibrate-vlm-bias":
        cmd_calibrate_vlm_bias(args)
    elif args.cmd == "compose-gate-side":
        cmd_compose_gate_side(args)
    elif args.cmd == "optiona-walkforward":
        cmd_optiona_walkforward(args)
    elif args.cmd == "cnn-dryrun":
        cmd_cnn_dryrun(args)
    elif args.cmd == "robust-sweep":
        cmd_robust_sweep(args)
    elif args.cmd == "backtest":
        cmd_backtest(args)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
