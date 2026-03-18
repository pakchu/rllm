"""SB3 PPO training entrypoint for Option A baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from envs.trading_env import TradingEnv, TradingEnvConfig
from preprocessing.chart_generator import build_images
from training.data_sources import (
    load_market_data,
    make_synthetic_market_df,
)


def _augment_with_mirror_market(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a mirrored-price copy (inverse return path) and append to dataset.

    This helps reduce directional collapse (always-long/always-short) by exposing
    both bullish and bearish dynamics with similar volatility structure.
    """
    required_cols = {"date", "open", "high", "low", "close", "volume", "tic"}
    if not required_cols.issubset(market_df.columns):
        return market_df

    base_price = float(market_df["open"].iloc[0])
    eps = 1e-12

    mirrored = market_df.copy()
    open_m = (base_price * base_price) / np.maximum(market_df["open"].to_numpy(), eps)
    close_m = (base_price * base_price) / np.maximum(market_df["close"].to_numpy(), eps)
    high_tmp = (base_price * base_price) / np.maximum(market_df["low"].to_numpy(), eps)
    low_tmp = (base_price * base_price) / np.maximum(market_df["high"].to_numpy(), eps)
    high_m = np.maximum(high_tmp, low_tmp)
    low_m = np.minimum(high_tmp, low_tmp)

    mirrored["open"] = open_m
    mirrored["high"] = high_m
    mirrored["low"] = low_m
    mirrored["close"] = close_m

    dt = pd.to_datetime(market_df["date"])
    span = dt.iloc[-1] - dt.iloc[0]
    mirrored["date"] = dt + span + pd.Timedelta(minutes=1)

    out = pd.concat([market_df, mirrored], ignore_index=True)
    out = out.sort_values("date").reset_index(drop=True)
    return out


def _build_env(
    market_df: pd.DataFrame,
    window_size: int,
    leverage: float,
    images=None,
    image_resolution: int = 64,
    flat_hold_penalty: float = 0.0,
    open_position_bonus: float = 0.0,
    directional_bias_penalty: float = 0.0,
    directional_symmetry_prob: float = 0.0,
    action_balance_penalty: float = 0.0,
    drawdown_penalty: float = 0.0,
    random_reset_start: bool = False,
    episode_horizon: int = 0,
    scalar_feature_mode: str = "legacy4",
    hold_action_mode: str = "flat",
):
    config = TradingEnvConfig(
        window_size=window_size,
        leverage=leverage,
        flat_hold_penalty=flat_hold_penalty,
        open_position_bonus=open_position_bonus,
        directional_bias_penalty=directional_bias_penalty,
        directional_symmetry_prob=directional_symmetry_prob,
        action_balance_penalty=action_balance_penalty,
        drawdown_penalty=drawdown_penalty,
        random_reset_start=random_reset_start,
        episode_horizon=episode_horizon,
        scalar_feature_mode=scalar_feature_mode,
        hold_action_mode=hold_action_mode,
        image_shape=(3, image_resolution, image_resolution),
    )
    return TradingEnv(market_df=market_df, images=images, config=config)


def _make_env_factory(
    market_df: pd.DataFrame,
    window_size: int,
    leverage: float,
    images=None,
    image_resolution: int = 64,
    flat_hold_penalty: float = 0.0,
    open_position_bonus: float = 0.0,
    directional_bias_penalty: float = 0.0,
    directional_symmetry_prob: float = 0.0,
    action_balance_penalty: float = 0.0,
    drawdown_penalty: float = 0.0,
    random_reset_start: bool = False,
    episode_horizon: int = 0,
    scalar_feature_mode: str = "legacy4",
    hold_action_mode: str = "flat",
) -> Callable[[], TradingEnv]:
    def _factory():
        return _build_env(
            market_df=market_df,
            window_size=window_size,
            leverage=leverage,
            images=images,
            image_resolution=image_resolution,
            flat_hold_penalty=flat_hold_penalty,
            open_position_bonus=open_position_bonus,
            directional_bias_penalty=directional_bias_penalty,
            directional_symmetry_prob=directional_symmetry_prob,
            action_balance_penalty=action_balance_penalty,
            drawdown_penalty=drawdown_penalty,
            random_reset_start=random_reset_start,
            episode_horizon=episode_horizon,
            scalar_feature_mode=scalar_feature_mode,
            hold_action_mode=hold_action_mode,
        )

    return _factory


def _build_vec_env(
    market_df: pd.DataFrame,
    window_size: int,
    leverage: float,
    images=None,
    image_resolution: int = 64,
    flat_hold_penalty: float = 0.0,
    open_position_bonus: float = 0.0,
    directional_bias_penalty: float = 0.0,
    directional_symmetry_prob: float = 0.0,
    action_balance_penalty: float = 0.0,
    drawdown_penalty: float = 0.0,
    random_reset_start: bool = False,
    episode_horizon: int = 0,
    scalar_feature_mode: str = "legacy4",
    hold_action_mode: str = "flat",
    n_envs: int = 1,
    vec_env: str = "dummy",
):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    if n_envs < 1:
        raise ValueError(f"n_envs must be >= 1, got {n_envs}")

    env_fns = [
        (
            lambda f=_make_env_factory(
                market_df,
                window_size,
                leverage,
                images=images,
                image_resolution=image_resolution,
                flat_hold_penalty=flat_hold_penalty,
                open_position_bonus=open_position_bonus,
                directional_bias_penalty=directional_bias_penalty,
                directional_symmetry_prob=directional_symmetry_prob,
                action_balance_penalty=action_balance_penalty,
                drawdown_penalty=drawdown_penalty,
                random_reset_start=random_reset_start,
                episode_horizon=episode_horizon,
                scalar_feature_mode=scalar_feature_mode,
                hold_action_mode=hold_action_mode,
            ): Monitor(f())
        )
        for _ in range(n_envs)
    ]

    if vec_env == "subproc" and n_envs > 1:
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


def train_option_a_on_market(
    market_df: pd.DataFrame,
    total_timesteps: int = 100_000,
    window_size: int = 96,
    leverage: float = 1.0,
    use_images: bool = False,
    image_resolution: int = 64,
    image_cache_dir: str | None = "data/image_cache_option_a",
    image_render_backend: str = "fast",
    flat_hold_penalty: float = 0.0,
    open_position_bonus: float = 0.0,
    directional_bias_penalty: float = 0.0,
    directional_symmetry_prob: float = 0.0,
    action_balance_penalty: float = 0.0,
    drawdown_penalty: float = 0.0,
    random_reset_start: bool = True,
    episode_horizon: int = 0,
    scalar_feature_mode: str = "legacy4",
    hold_action_mode: str = "flat",
    learning_rate: float = 2.5e-4,
    ent_coef: float = 0.01,
    n_steps: int = 2048,
    batch_size: int = 256,
    n_epochs: int = 10,
    target_kl: float | None = None,
    mirror_augmentation: bool = False,
    visual_backbone: str = "cnn",
    visual_backbone_model: str | None = None,
    visual_pooling: str = "cls_patch_mean",
    freeze_visual_backbone: bool = True,
    n_envs: int = 1,
    vec_env: str = "dummy",
    seed: int = 42,
    save_path: str = "checkpoints/ppo_option_a.zip",
    tensorboard_log: str | None = None,
) -> str:
    """
    Train Option A PPO agent.

    Returns:
        Saved model path.
    """
    # Lazy imports to keep helper functions usable without full RL deps.
    from stable_baselines3 import PPO

    from models.option_a import VISUAL_BACKBONE_PRESETS, build_policy_kwargs

    images = None
    train_df = market_df
    if mirror_augmentation:
        train_df = _augment_with_mirror_market(market_df)
    if visual_backbone != "cnn" and not use_images:
        raise ValueError("visual_backbone requires use_images=True")
    if use_images:
        images = build_images(
            market_df=train_df,
            window_size=window_size,
            resolution=image_resolution,
            cache_dir=image_cache_dir,
            backend=image_render_backend,
        )

    env = _build_vec_env(
        market_df=train_df,
        window_size=window_size,
        leverage=leverage,
        images=images,
        image_resolution=image_resolution,
        flat_hold_penalty=flat_hold_penalty,
        open_position_bonus=open_position_bonus,
        directional_bias_penalty=directional_bias_penalty,
        directional_symmetry_prob=directional_symmetry_prob,
        action_balance_penalty=action_balance_penalty,
        drawdown_penalty=drawdown_penalty,
        random_reset_start=random_reset_start,
        episode_horizon=episode_horizon,
        scalar_feature_mode=scalar_feature_mode,
        hold_action_mode=hold_action_mode,
        n_envs=n_envs,
        vec_env=vec_env,
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=build_policy_kwargs(
            visual_backbone=visual_backbone,
            visual_backbone_model=visual_backbone_model,
            freeze_visual_backbone=freeze_visual_backbone,
            visual_pooling=visual_pooling,
        ),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.2,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=target_kl,
        seed=seed,
        verbose=1,
        tensorboard_log=tensorboard_log if tensorboard_log else None,
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    resolved_visual_backbone_model = (
        visual_backbone_model
        if visual_backbone_model
        else VISUAL_BACKBONE_PRESETS.get(visual_backbone)
    )

    metadata = {
        "source": "option_a_sb3",
        "use_images": bool(use_images),
        "image_resolution": int(image_resolution),
        "image_render_backend": str(image_render_backend),
        "window_size": int(window_size),
        "leverage": float(leverage),
        "flat_hold_penalty": float(flat_hold_penalty),
        "open_position_bonus": float(open_position_bonus),
        "directional_bias_penalty": float(directional_bias_penalty),
        "directional_symmetry_prob": float(directional_symmetry_prob),
        "action_balance_penalty": float(action_balance_penalty),
        "drawdown_penalty": float(drawdown_penalty),
        "random_reset_start": bool(random_reset_start),
        "episode_horizon": int(episode_horizon),
        "scalar_feature_mode": str(scalar_feature_mode),
        "hold_action_mode": str(hold_action_mode),
        "learning_rate": float(learning_rate),
        "ent_coef": float(ent_coef),
        "n_steps": int(n_steps),
        "batch_size": int(batch_size),
        "n_epochs": int(n_epochs),
        "target_kl": None if target_kl is None else float(target_kl),
        "mirror_augmentation": bool(mirror_augmentation),
        "visual_backbone": str(visual_backbone),
        "visual_backbone_model": (
            None
            if resolved_visual_backbone_model is None
            else str(resolved_visual_backbone_model)
        ),
        "visual_pooling": str(visual_pooling),
        "freeze_visual_backbone": bool(freeze_visual_backbone),
    }
    meta_path = output_path.with_name(output_path.name + ".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2))
    return str(output_path.resolve())


def train_option_a(
    total_timesteps: int = 100_000,
    window_size: int = 96,
    leverage: float = 1.0,
    use_images: bool = False,
    image_resolution: int = 64,
    image_cache_dir: str | None = "data/image_cache_option_a",
    image_render_backend: str = "fast",
    flat_hold_penalty: float = 0.0,
    open_position_bonus: float = 0.0,
    directional_bias_penalty: float = 0.0,
    directional_symmetry_prob: float = 0.0,
    action_balance_penalty: float = 0.0,
    drawdown_penalty: float = 0.0,
    random_reset_start: bool = True,
    episode_horizon: int = 0,
    scalar_feature_mode: str = "legacy4",
    hold_action_mode: str = "flat",
    learning_rate: float = 2.5e-4,
    ent_coef: float = 0.01,
    n_steps: int = 2048,
    batch_size: int = 256,
    n_epochs: int = 10,
    target_kl: float | None = None,
    mirror_augmentation: bool = False,
    visual_backbone: str = "cnn",
    visual_backbone_model: str | None = None,
    visual_pooling: str = "cls_patch_mean",
    freeze_visual_backbone: bool = True,
    n_envs: int = 1,
    vec_env: str = "dummy",
    seed: int = 42,
    save_path: str = "checkpoints/ppo_option_a.zip",
    tensorboard_log: str | None = None,
) -> str:
    """Train Option A PPO on synthetic data (backward-compatible helper)."""
    market_df = make_synthetic_market_df(seed=seed)
    return train_option_a_on_market(
        market_df=market_df,
        total_timesteps=total_timesteps,
        window_size=window_size,
        leverage=leverage,
        use_images=use_images,
        image_resolution=image_resolution,
        image_cache_dir=image_cache_dir,
        image_render_backend=image_render_backend,
        flat_hold_penalty=flat_hold_penalty,
        open_position_bonus=open_position_bonus,
        directional_bias_penalty=directional_bias_penalty,
        directional_symmetry_prob=directional_symmetry_prob,
        action_balance_penalty=action_balance_penalty,
        drawdown_penalty=drawdown_penalty,
        random_reset_start=random_reset_start,
        episode_horizon=episode_horizon,
        scalar_feature_mode=scalar_feature_mode,
        hold_action_mode=hold_action_mode,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        target_kl=target_kl,
        mirror_augmentation=mirror_augmentation,
        visual_backbone=visual_backbone,
        visual_backbone_model=visual_backbone_model,
        visual_pooling=visual_pooling,
        freeze_visual_backbone=freeze_visual_backbone,
        n_envs=n_envs,
        vec_env=vec_env,
        seed=seed,
        save_path=save_path,
        tensorboard_log=tensorboard_log,
    )


def train_option_a_from_source(
    source: str = "synthetic",
    input_csv: str | None = None,
    timeframe: str = "1m",
    symbol: str = "BTCUSDT",
    start_date: str | None = None,
    end_date: str | None = None,
    market_type: str = "futures",
    num_rows: int = 8_000,
    synthetic_drift: float = 0.0,
    synthetic_regime_amplitude: float = 0.0004,
    synthetic_regime_period: int = 720,
    total_timesteps: int = 100_000,
    window_size: int = 96,
    leverage: float = 1.0,
    use_images: bool = False,
    image_resolution: int = 64,
    image_cache_dir: str | None = "data/image_cache_option_a",
    image_render_backend: str = "fast",
    flat_hold_penalty: float = 0.0,
    open_position_bonus: float = 0.0,
    directional_bias_penalty: float = 0.0,
    directional_symmetry_prob: float = 0.0,
    action_balance_penalty: float = 0.0,
    drawdown_penalty: float = 0.0,
    random_reset_start: bool = True,
    episode_horizon: int = 0,
    scalar_feature_mode: str = "legacy4",
    hold_action_mode: str = "flat",
    learning_rate: float = 2.5e-4,
    ent_coef: float = 0.01,
    n_steps: int = 2048,
    batch_size: int = 256,
    n_epochs: int = 10,
    target_kl: float | None = None,
    mirror_augmentation: bool = False,
    visual_backbone: str = "cnn",
    visual_backbone_model: str | None = None,
    visual_pooling: str = "cls_patch_mean",
    freeze_visual_backbone: bool = True,
    n_envs: int = 1,
    vec_env: str = "dummy",
    seed: int = 42,
    save_path: str = "checkpoints/ppo_option_a.zip",
    tensorboard_log: str | None = None,
) -> str:
    """Train Option A PPO using configurable market data source."""
    market_df = load_market_data(
        source=source,
        input_csv=input_csv,
        timeframe=timeframe,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        market_type=market_type,
        num_rows=num_rows,
        synthetic_drift=synthetic_drift,
        synthetic_regime_amplitude=synthetic_regime_amplitude,
        synthetic_regime_period=synthetic_regime_period,
        seed=seed,
    )
    return train_option_a_on_market(
        market_df=market_df,
        total_timesteps=total_timesteps,
        window_size=window_size,
        leverage=leverage,
        use_images=use_images,
        image_resolution=image_resolution,
        image_cache_dir=image_cache_dir,
        image_render_backend=image_render_backend,
        flat_hold_penalty=flat_hold_penalty,
        open_position_bonus=open_position_bonus,
        directional_bias_penalty=directional_bias_penalty,
        directional_symmetry_prob=directional_symmetry_prob,
        action_balance_penalty=action_balance_penalty,
        drawdown_penalty=drawdown_penalty,
        random_reset_start=random_reset_start,
        episode_horizon=episode_horizon,
        scalar_feature_mode=scalar_feature_mode,
        hold_action_mode=hold_action_mode,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        target_kl=target_kl,
        mirror_augmentation=mirror_augmentation,
        visual_backbone=visual_backbone,
        visual_backbone_model=visual_backbone_model,
        visual_pooling=visual_pooling,
        freeze_visual_backbone=freeze_visual_backbone,
        n_envs=n_envs,
        vec_env=vec_env,
        seed=seed,
        save_path=save_path,
        tensorboard_log=tensorboard_log,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Option A SB3 PPO baseline.")
    parser.add_argument(
        "--source", type=str, default="synthetic", choices=["synthetic", "csv", "binance"]
    )
    parser.add_argument("--input-csv", type=str, default="")
    parser.add_argument("--timeframe", type=str, default="1m")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--start-date", type=str, default="")
    parser.add_argument("--end-date", type=str, default="")
    parser.add_argument("--market-type", type=str, default="futures")
    parser.add_argument("--num-rows", type=int, default=8000)
    parser.add_argument("--synthetic-drift", type=float, default=0.0)
    parser.add_argument("--synthetic-regime-amplitude", type=float, default=0.0004)
    parser.add_argument("--synthetic-regime-period", type=int, default=720)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--window-size", type=int, default=96)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--use-images", action="store_true", default=False)
    parser.add_argument("--image-resolution", type=int, default=64)
    parser.add_argument("--image-cache-dir", type=str, default="data/image_cache_option_a")
    parser.add_argument(
        "--image-render-backend",
        type=str,
        default="fast",
        choices=["matplotlib", "fast"],
    )
    parser.add_argument("--flat-hold-penalty", type=float, default=0.0)
    parser.add_argument("--open-position-bonus", type=float, default=0.0)
    parser.add_argument("--directional-bias-penalty", type=float, default=0.0)
    parser.add_argument("--directional-symmetry-prob", type=float, default=0.0)
    parser.add_argument("--action-balance-penalty", type=float, default=0.0)
    parser.add_argument("--drawdown-penalty", type=float, default=0.0)
    parser.add_argument("--random-reset-start", action="store_true", default=False)
    parser.add_argument("--episode-horizon", type=int, default=0)
    parser.add_argument(
        "--scalar-feature-mode",
        type=str,
        default="legacy4",
        choices=["legacy4", "extended_v1"],
    )
    parser.add_argument(
        "--hold-action-mode",
        type=str,
        default="flat",
        choices=["flat", "maintain"],
        help="Meaning of HOLD action: flat (target 0) or maintain (keep current side).",
    )
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--target-kl", type=float, default=-1.0)
    parser.add_argument("--mirror-augmentation", action="store_true", default=False)
    parser.add_argument(
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
    parser.add_argument("--visual-backbone-model", type=str, default="")
    parser.add_argument(
        "--visual-pooling",
        type=str,
        default="cls_patch_mean",
        choices=["cls", "mean", "cls_patch_mean"],
    )
    parser.add_argument("--unfreeze-visual-backbone", action="store_true", default=False)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--vec-env", type=str, default="dummy", choices=["dummy", "subproc"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="checkpoints/ppo_option_a.zip")
    parser.add_argument("--tensorboard-log", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved_path = train_option_a_from_source(
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
        tensorboard_log=args.tensorboard_log,
    )
    print(f"Saved model to: {saved_path}")


if __name__ == "__main__":
    main()
