"""Backtest utility for trained SB3 policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

from envs.trading_env import TradingEnv, TradingEnvConfig
from evaluation.metrics import summarize_metrics
from preprocessing.chart_generator import build_images
from training.data_sources import load_market_data


def build_underlying_curve(
    open_prices: np.ndarray, initial_equity: float
) -> np.ndarray:
    """Convert open-price series to an equity-like benchmark curve."""
    if len(open_prices) == 0:
        return np.asarray([], dtype=np.float64)
    return initial_equity * (open_prices / open_prices[0])


def periods_per_year_from_timeframe(timeframe: str) -> int:
    """Map timeframe to annualized period count for Sharpe scaling."""
    tf = timeframe.lower().strip()
    mapping = {
        "1m": 365 * 24 * 60,
        "3m": 365 * 24 * 20,
        "5m": 365 * 24 * 12,
        "15m": 365 * 24 * 4,
        "1h": 365 * 24,
        "1d": 365,
    }
    return mapping.get(tf, 365 * 24 * 60)


def normalize_action_probs(probs: np.ndarray) -> np.ndarray:
    """
    Sanitize and normalize action probabilities to a valid simplex.

    - non-finite values -> 0
    - negatives -> 0
    - zero-sum fallback -> uniform
    """
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    if p.shape[0] != 3:
        raise ValueError(f"Expected 3 action probabilities, got shape={p.shape}")
    p = np.where(np.isfinite(p), p, 0.0)
    p = np.clip(p, 0.0, None)
    s = float(np.sum(p))
    if s <= 0.0:
        return np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
    return p / s


def blend_weight_from_trend(
    *,
    mode: str,
    trend: float,
    weight_a: float = 0.75,
    weight_up: float = 0.90,
    weight_down: float = 0.35,
    trend_threshold: float = 0.002,
) -> float:
    """Compute blend weight for model A from trend scalar."""
    m = str(mode).lower().strip()
    if m == "static":
        return float(np.clip(weight_a, 0.0, 1.0))

    th = float(abs(trend_threshold))
    if float(trend) >= th:
        return float(np.clip(weight_up, 0.0, 1.0))
    if float(trend) <= -th:
        return float(np.clip(weight_down, 0.0, 1.0))
    up = float(np.clip(weight_up, 0.0, 1.0))
    down = float(np.clip(weight_down, 0.0, 1.0))
    return 0.5 * (up + down)


def ema_series(values: np.ndarray, span: int) -> np.ndarray:
    """Simple EMA series computed with recursive smoothing."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr.copy()
    s = max(1, int(span))
    alpha = 2.0 / (float(s) + 1.0)
    out = np.empty_like(arr, dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, arr.size):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def rolling_mean_std(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute rolling mean/std with variable warmup window (uses available history)."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    n = arr.size
    if n == 0:
        return arr.copy(), arr.copy()
    w = max(1, int(window))
    csum = np.concatenate(([0.0], np.cumsum(arr)))
    csum2 = np.concatenate(([0.0], np.cumsum(arr * arr)))
    means = np.zeros(n, dtype=np.float64)
    stds = np.zeros(n, dtype=np.float64)
    for i in range(n):
        st = max(0, i - w + 1)
        count = i - st + 1
        s = float(csum[i + 1] - csum[st])
        s2 = float(csum2[i + 1] - csum2[st])
        mean = s / float(count)
        var = max(s2 / float(count) - mean * mean, 0.0)
        means[i] = mean
        stds[i] = float(np.sqrt(var))
    return means, stds


def rolling_zscore(values: np.ndarray, window: int, eps: float = 1e-12) -> np.ndarray:
    """Rolling z-score computed from trailing window statistics."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    means, stds = rolling_mean_std(arr, window=max(1, int(window)))
    z = np.zeros_like(arr, dtype=np.float64)
    mask = stds > float(eps)
    z[mask] = (arr[mask] - means[mask]) / stds[mask]
    return z


def build_regime_scores(
    open_prices: np.ndarray,
    *,
    score_mode: str = "zscore",
    ret_short: int = 60,
    ret_long: int = 240,
    ema_fast: int = 60,
    ema_slow: int = 240,
    vol_lookback: int = 60,
    z_window: int = 1_440,
    weight_ret_long: float = 0.45,
    weight_ema_slope: float = 0.35,
    weight_ret_short: float = 0.20,
    weight_vol: float = -0.25,
    weight_drawdown: float = -0.15,
) -> dict:
    """
    Build regime features and scalar score using only present/past price information.
    """
    prices = np.asarray(open_prices, dtype=np.float64).reshape(-1)
    n = prices.size
    if n == 0:
        zeros = np.asarray([], dtype=np.float64)
        return {
            "ret_short": zeros,
            "ret_long": zeros,
            "ema_slope": zeros,
            "realized_vol": zeros,
            "drawdown": zeros,
            "score": zeros,
        }

    r_short = max(1, int(ret_short))
    r_long = max(1, int(ret_long))
    v_lb = max(2, int(vol_lookback))
    z_w = max(20, int(z_window))

    ret_s = np.zeros(n, dtype=np.float64)
    ret_l = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if i >= r_short and prices[i - r_short] > 0.0:
            ret_s[i] = prices[i] / prices[i - r_short] - 1.0
        if i >= r_long and prices[i - r_long] > 0.0:
            ret_l[i] = prices[i] / prices[i - r_long] - 1.0

    ema_f = ema_series(prices, span=max(1, int(ema_fast)))
    ema_sl = ema_series(prices, span=max(1, int(ema_slow)))
    ema_slope = np.zeros(n, dtype=np.float64)
    nz = prices > 0.0
    ema_slope[nz] = (ema_f[nz] - ema_sl[nz]) / prices[nz]

    logret = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if prices[i] > 0.0 and prices[i - 1] > 0.0:
            logret[i] = np.log(prices[i] / prices[i - 1])
    _, realized_vol = rolling_mean_std(logret, window=v_lb)

    peak = np.maximum.accumulate(prices)
    drawdown = np.zeros(n, dtype=np.float64)
    good = peak > 0.0
    drawdown[good] = (peak[good] - prices[good]) / peak[good]

    if str(score_mode) == "raw":
        f_ret_l = ret_l
        f_ema_slope = ema_slope
        f_ret_s = ret_s
        f_vol = realized_vol
        f_dd = drawdown
    else:
        f_ret_l = rolling_zscore(ret_l, window=z_w)
        f_ema_slope = rolling_zscore(ema_slope, window=z_w)
        f_ret_s = rolling_zscore(ret_s, window=z_w)
        f_vol = rolling_zscore(realized_vol, window=z_w)
        f_dd = rolling_zscore(drawdown, window=z_w)

    score = (
        float(weight_ret_long) * f_ret_l
        + float(weight_ema_slope) * f_ema_slope
        + float(weight_ret_short) * f_ret_s
        + float(weight_vol) * f_vol
        + float(weight_drawdown) * f_dd
    )
    return {
        "score_mode": str(score_mode),
        "ret_short": ret_s,
        "ret_long": ret_l,
        "ema_slope": ema_slope,
        "realized_vol": realized_vol,
        "drawdown": drawdown,
        "score": score,
    }


def build_regime_states(
    scores: np.ndarray,
    *,
    enter_threshold: float = 0.7,
    confirm_bars: int = 3,
) -> np.ndarray:
    """
    Convert regime scores to {-1,0,+1} states with hysteresis and confirmation bars.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    states = np.zeros(s.size, dtype=np.int8)
    c = max(1, int(confirm_bars))
    th = float(enter_threshold)
    up_count = 0
    down_count = 0
    state = 0
    for i, v in enumerate(s):
        if v >= th:
            up_count += 1
            down_count = 0
        elif v <= -th:
            down_count += 1
            up_count = 0
        else:
            up_count = 0
            down_count = 0
        if up_count >= c:
            state = 1
        elif down_count >= c:
            state = -1
        states[i] = state
    return states


def infer_expected_image_shape(model) -> tuple[int, int, int]:
    """Infer expected image observation shape from a loaded SB3 policy."""
    expected_image_shape = (3, 320, 320)
    obs_space = getattr(model, "observation_space", None)
    if obs_space is not None and hasattr(obs_space, "spaces"):
        img_space = obs_space.spaces.get("image")
        if img_space is not None and hasattr(img_space, "shape") and len(img_space.shape) == 3:
            expected_image_shape = tuple(int(x) for x in img_space.shape)
    return expected_image_shape


def extract_policy_action_probs(model, obs: dict, debiased_action: str = "off") -> np.ndarray:
    """
    Extract per-action probabilities from model policy with optional mirror debiasing.
    """
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    dist = model.policy.get_distribution(obs_tensor)
    probs_eff = dist.distribution.probs.detach().cpu().numpy().reshape(-1, 3)[0].astype(np.float64)

    if debiased_action == "mirror_scalar":
        obs_m = {
            "image": np.array(obs["image"], copy=True),
            "scalars": np.array(obs["scalars"], copy=True),
        }
        # Scalars = [side, unrealized_pnl_pct, range_volatility_pct, window_trend_pct]
        obs_m["scalars"][0] *= -1.0
        obs_m["scalars"][1] *= -1.0
        obs_m["scalars"][3] *= -1.0
        obs_m_tensor, _ = model.policy.obs_to_tensor(obs_m)
        dist_m = model.policy.get_distribution(obs_m_tensor)
        probs_m = (
            dist_m.distribution.probs.detach().cpu().numpy().reshape(-1, 3)[0]
        ).astype(np.float64)
        probs_eff = np.array(
            [
                0.5 * (probs_eff[0] + probs_m[2]),
                0.5 * (probs_eff[1] + probs_m[1]),
                0.5 * (probs_eff[2] + probs_m[0]),
            ],
            dtype=np.float64,
        )
    return normalize_action_probs(probs_eff)


def run_backtest(
    model_path: str,
    source: str = "synthetic",
    input_csv: str | None = None,
    timeframe: str = "1m",
    symbol: str = "BTCUSDT",
    start_date: str | None = None,
    end_date: str | None = None,
    market_type: str = "futures",
    num_rows: int = 8_000,
    seed: int = 123,
    synthetic_drift: float = 0.0,
    synthetic_regime_amplitude: float = 0.0004,
    synthetic_regime_period: int = 720,
    window_size: int = 96,
    leverage: float = 1.0,
    initial_equity: float = 1_000.0,
    fee_rate: float = 0.0005,
    slippage_rate: float = 0.0001,
    flat_hold_penalty: float = 0.0,
    hold_action_mode: str | None = None,
    use_images: bool | None = None,
    image_cache_dir: str | None = "data/image_cache_backtest",
    image_render_backend: str | None = None,
    scalar_feature_mode: str | None = None,
    deterministic: bool = True,
    decision_mode: str = "policy",
    flat_start_policy: str = "as_is",
    directional_tie_hold_eps: float = 0.0,
    debiased_action: str = "off",
    blend_model_a: str | None = None,
    blend_model_b: str | None = None,
    blend_weight_mode: str = "static",
    blend_weight_a: float = 0.75,
    blend_weight_up: float = 0.90,
    blend_weight_down: float = 0.35,
    blend_trend_threshold: float = 0.002,
    score_entry_threshold: float = 0.02,
    score_flip_threshold: float = 0.05,
    score_neutral_band: float = 0.005,
    score_exit_threshold: float = -1.0,
    score_centering: str = "off",
    score_center_alpha: float = 0.02,
    trend_guard: str = "off",
    trend_threshold: float = 0.002,
    volatility_gate: str = "off",
    volatility_threshold: float = 0.0,
    regime_model_up: str | None = None,
    regime_model_down: str | None = None,
    regime_neutral_policy: str = "hold",
    regime_transition_mode: str = "force_align",
    regime_score_mode: str = "zscore",
    regime_ret_short: int = 60,
    regime_ret_long: int = 240,
    regime_ema_fast: int = 60,
    regime_ema_slow: int = 240,
    regime_vol_lookback: int = 60,
    regime_z_window: int = 1_440,
    regime_weight_ret_long: float = 0.45,
    regime_weight_ema_slope: float = 0.35,
    regime_weight_ret_short: float = 0.20,
    regime_weight_vol: float = -0.25,
    regime_weight_drawdown: float = -0.15,
    regime_enter_threshold: float = 0.7,
    regime_confirm_bars: int = 3,
) -> Dict[str, float]:
    """Run deterministic backtest on configurable market data source."""
    from stable_baselines3 import PPO

    model = PPO.load(model_path)
    model_up = model
    model_down = model
    model_blend_a = model
    model_blend_b = model
    up_model_path_resolved = str(Path(regime_model_up or model_path).resolve())
    down_model_path_resolved = str(Path(regime_model_down or model_path).resolve())
    primary_model_path_resolved = str(Path(model_path).resolve())
    blend_model_a_path_resolved = str(Path(blend_model_a or model_path).resolve())
    blend_model_b_path_resolved = str(Path(blend_model_b or model_path).resolve())
    if decision_mode == "regime_switch":
        if up_model_path_resolved != primary_model_path_resolved:
            model_up = PPO.load(up_model_path_resolved)
        if down_model_path_resolved == up_model_path_resolved:
            model_down = model_up
        elif down_model_path_resolved != primary_model_path_resolved:
            model_down = PPO.load(down_model_path_resolved)
    elif decision_mode == "blend_score_band":
        if blend_model_a_path_resolved != primary_model_path_resolved:
            model_blend_a = PPO.load(blend_model_a_path_resolved)
        if blend_model_b_path_resolved == blend_model_a_path_resolved:
            model_blend_b = model_blend_a
        elif blend_model_b_path_resolved == primary_model_path_resolved:
            model_blend_b = model
        else:
            model_blend_b = PPO.load(blend_model_b_path_resolved)

    # Align env observation spec to what the trained model expects.
    expected_image_shape = infer_expected_image_shape(model)
    if decision_mode == "regime_switch":
        up_shape = infer_expected_image_shape(model_up)
        down_shape = infer_expected_image_shape(model_down)
        if up_shape != expected_image_shape or down_shape != expected_image_shape:
            raise ValueError(
                "All regime-switch models must share same image observation shape: "
                f"primary={expected_image_shape}, up={up_shape}, down={down_shape}"
            )
    elif decision_mode == "blend_score_band":
        a_shape = infer_expected_image_shape(model_blend_a)
        b_shape = infer_expected_image_shape(model_blend_b)
        if a_shape != expected_image_shape or b_shape != expected_image_shape:
            raise ValueError(
                "All blend models must share same image observation shape: "
                f"primary={expected_image_shape}, a={a_shape}, b={b_shape}"
            )

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

    model_meta = {}
    meta_path = Path(str(model_path) + ".meta.json")
    if meta_path.exists():
        try:
            model_meta = json.loads(meta_path.read_text())
        except Exception:
            model_meta = {}

    # Auto policy:
    # - If user specified use_images=True/False, respect it.
    # - Otherwise, infer from metadata if available.
    if use_images is None:
        use_images = False
        if isinstance(model_meta.get("use_images"), bool):
            use_images = bool(model_meta["use_images"])

    if image_render_backend is None:
        backend_from_meta = model_meta.get("image_render_backend")
        if backend_from_meta in {"matplotlib", "fast"}:
            image_render_backend = str(backend_from_meta)
        else:
            image_render_backend = "matplotlib"

    if scalar_feature_mode is None:
        scalar_mode_from_meta = model_meta.get("scalar_feature_mode")
        if scalar_mode_from_meta in {"legacy4", "extended_v1"}:
            scalar_feature_mode = str(scalar_mode_from_meta)
        else:
            scalar_feature_mode = "legacy4"

    if hold_action_mode is None:
        mode_from_meta = model_meta.get("hold_action_mode")
        if mode_from_meta in {"flat", "maintain"}:
            hold_action_mode = str(mode_from_meta)
        else:
            hold_action_mode = "flat"

    images = None
    if use_images:
        if expected_image_shape[1] != expected_image_shape[2]:
            raise ValueError(
                f"Backtest only supports square image shapes, got: {expected_image_shape}"
            )
        images = build_images(
            market_df,
            window_size=window_size,
            resolution=expected_image_shape[1],
            cache_dir=image_cache_dir,
            backend=image_render_backend,
        )

    env = TradingEnv(
        market_df=market_df,
        images=images,
        config=TradingEnvConfig(
            window_size=window_size,
            leverage=leverage,
            initial_equity=initial_equity,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            flat_hold_penalty=flat_hold_penalty,
            scalar_feature_mode=str(scalar_feature_mode),
            hold_action_mode=str(hold_action_mode),
            image_shape=expected_image_shape,
        ),
    )

    obs, info = env.reset()
    rng = np.random.default_rng(seed)
    score_ema = 0.0
    score_ema_init = False
    equity_curve = [info["equity"]]
    action_counts = {0: 0, 1: 0, 2: 0}
    action_prob_sum = np.zeros(3, dtype=np.float64)
    reward_sum = 0.0
    trade_rebalance_steps = 0
    trade_turnover_legs = 0
    trade_entries = 0
    trade_exits = 0
    trade_direct_flips = 0
    trade_long_entries = 0
    trade_short_entries = 0
    trade_long_exits = 0
    trade_short_exits = 0
    blend_weight_sum = 0.0
    blend_weight_count = 0
    blend_trend_up_count = 0
    blend_trend_down_count = 0
    blend_trend_neutral_count = 0
    regime_counts = {"up": 0, "neutral": 0, "down": 0}
    regime_model_usage = {"up_model": 0, "down_model": 0, "neutral_hold": 0}
    regime_forced_alignment_count = 0
    prev_regime_state = 0
    regime_score_arr = None
    regime_state_arr = None
    if decision_mode == "regime_switch":
        open_all = market_df["open"].to_numpy(dtype=np.float64)
        regime_features = build_regime_scores(
            open_all,
            score_mode=regime_score_mode,
            ret_short=regime_ret_short,
            ret_long=regime_ret_long,
            ema_fast=regime_ema_fast,
            ema_slow=regime_ema_slow,
            vol_lookback=regime_vol_lookback,
            z_window=regime_z_window,
            weight_ret_long=regime_weight_ret_long,
            weight_ema_slope=regime_weight_ema_slope,
            weight_ret_short=regime_weight_ret_short,
            weight_vol=regime_weight_vol,
            weight_drawdown=regime_weight_drawdown,
        )
        regime_score_arr = regime_features["score"]
        regime_state_arr = build_regime_states(
            regime_score_arr,
            enter_threshold=regime_enter_threshold,
            confirm_bars=regime_confirm_bars,
        )

    done = False
    while not done:
        action_i = 1
        probs_eff = None
        active_model = model
        forced_neutral_hold = False
        regime_state = 0

        if decision_mode == "blend_score_band":
            trend_now = 0.0
            try:
                trend_now = float(obs["scalars"][3])
            except Exception:
                trend_now = 0.0
            w_a = blend_weight_from_trend(
                mode=blend_weight_mode,
                trend=trend_now,
                weight_a=blend_weight_a,
                weight_up=blend_weight_up,
                weight_down=blend_weight_down,
                trend_threshold=blend_trend_threshold,
            )
            blend_weight_sum += float(w_a)
            blend_weight_count += 1
            th = float(abs(blend_trend_threshold))
            if trend_now >= th:
                blend_trend_up_count += 1
            elif trend_now <= -th:
                blend_trend_down_count += 1
            else:
                blend_trend_neutral_count += 1
            try:
                p_a = extract_policy_action_probs(
                    model_blend_a,
                    obs,
                    debiased_action=debiased_action,
                )
                p_b = extract_policy_action_probs(
                    model_blend_b,
                    obs,
                    debiased_action=debiased_action,
                )
                probs_eff = normalize_action_probs(float(w_a) * p_a + (1.0 - float(w_a)) * p_b)
                action_prob_sum += probs_eff
            except Exception:
                probs_eff = None
                active_model = model_blend_a
        elif decision_mode == "regime_switch":
            idx = int(info.get("step", window_size - 1))
            if regime_state_arr is not None:
                idx = min(max(0, idx), len(regime_state_arr) - 1)
                regime_state = int(regime_state_arr[idx])
            if regime_state > 0:
                regime_counts["up"] += 1
                regime_model_usage["up_model"] += 1
                active_model = model_up
            elif regime_state < 0:
                regime_counts["down"] += 1
                regime_model_usage["down_model"] += 1
                active_model = model_down
            else:
                regime_counts["neutral"] += 1
                if regime_neutral_policy == "up":
                    active_model = model_up
                    regime_model_usage["up_model"] += 1
                elif regime_neutral_policy == "down":
                    active_model = model_down
                    regime_model_usage["down_model"] += 1
                else:
                    forced_neutral_hold = True
                    regime_model_usage["neutral_hold"] += 1
        regime_transition = (
            decision_mode == "regime_switch"
            and regime_state != 0
            and regime_state != int(prev_regime_state)
        )

        if decision_mode != "blend_score_band" and forced_neutral_hold:
            probs_eff = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
            action_prob_sum += probs_eff
        elif decision_mode != "blend_score_band":
            try:
                probs_eff = extract_policy_action_probs(
                    active_model,
                    obs,
                    debiased_action=debiased_action,
                )
                action_prob_sum += probs_eff
            except Exception:
                probs_eff = None

        if probs_eff is not None:
            if decision_mode in {"score_band", "blend_score_band"}:
                side_now = int(info.get("position_side", 0))
                score_raw = float(probs_eff[0] - probs_eff[2])
                score = score_raw
                if score_centering == "ema":
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
                    if (
                        float(score_exit_threshold) >= 0.0
                        and abs(score) <= float(score_exit_threshold)
                    ):
                        action_i = 1
                    else:
                        action_i = 2 if score <= -float(score_flip_threshold) else 0
                else:
                    if (
                        float(score_exit_threshold) >= 0.0
                        and abs(score) <= float(score_exit_threshold)
                    ):
                        action_i = 1
                    else:
                        action_i = 0 if score >= float(score_flip_threshold) else 2
            elif deterministic:
                action_i = int(np.argmax(probs_eff))
            else:
                action_i = int(rng.choice([0, 1, 2], p=normalize_action_probs(probs_eff)))
        else:
            action, _ = active_model.predict(obs, deterministic=deterministic)
            action_i = int(action)

        if trend_guard in {"hard", "align_flat", "long_flat", "short_flat"}:
            try:
                trend = float(obs["scalars"][3])
                th = float(trend_threshold)
                if trend_guard == "hard":
                    if trend >= th:
                        action_i = 0  # BUY
                    elif trend <= -th:
                        action_i = 2  # SELL
                elif trend_guard == "align_flat":
                    if trend >= th:
                        action_i = 0
                    elif trend <= -th:
                        action_i = 2
                    else:
                        action_i = 1
                elif trend_guard == "long_flat":
                    action_i = 0 if trend >= th else 1
                elif trend_guard == "short_flat":
                    action_i = 2 if trend <= -th else 1
            except Exception:
                pass
        if volatility_gate == "hard":
            try:
                vol_now = float(obs["scalars"][2])
                if vol_now >= float(volatility_threshold):
                    action_i = 1  # HOLD -> flat in default mode
            except Exception:
                pass

        if decision_mode == "regime_switch" and regime_transition_mode == "force_align":
            side_now = int(info.get("position_side", 0))
            if regime_state > 0 and side_now <= 0 and action_i != 0:
                action_i = 0
                regime_forced_alignment_count += 1
            elif regime_state < 0 and side_now >= 0 and action_i != 2:
                action_i = 2
                regime_forced_alignment_count += 1
        elif decision_mode == "regime_switch" and regime_transition_mode == "force_on_switch":
            if regime_transition:
                side_now = int(info.get("position_side", 0))
                if regime_state > 0 and side_now <= 0 and action_i != 0:
                    action_i = 0
                    regime_forced_alignment_count += 1
                elif regime_state < 0 and side_now >= 0 and action_i != 2:
                    action_i = 2
                    regime_forced_alignment_count += 1

        if (
            deterministic
            and flat_start_policy == "prefer_entry"
            and int(info.get("position_side", 0)) == 0
            and action_i == 1
            and probs_eff is not None
        ):
            # Avoid degenerate "always HOLD while flat" by preferring directional entry.
            action_i = 0 if float(probs_eff[0]) >= float(probs_eff[2]) else 2
        if (
            deterministic
            and directional_tie_hold_eps > 0.0
            and probs_eff is not None
            and action_i in (0, 2)
            and abs(float(probs_eff[0]) - float(probs_eff[2]))
            < float(directional_tie_hold_eps)
        ):
            # Debias near-tie BUY/SELL logits: avoid arbitrary fixed direction by staying neutral.
            action_i = 1

        if action_i in action_counts:
            action_counts[action_i] += 1
        side_before = int(info.get("position_side", 0))
        obs, step_reward, terminated, truncated, info = env.step(action_i)
        side_after = int(info.get("position_side", 0))
        if side_after != side_before:
            trade_rebalance_steps += 1
            trade_turnover_legs += abs(side_after - side_before)
            if side_before == 0 and side_after != 0:
                trade_entries += 1
                if side_after > 0:
                    trade_long_entries += 1
                else:
                    trade_short_entries += 1
            elif side_before != 0 and side_after == 0:
                trade_exits += 1
                if side_before > 0:
                    trade_long_exits += 1
                else:
                    trade_short_exits += 1
            elif side_before != 0 and side_after != 0:
                # Direct flip (+1 <-> -1): close + open in one step.
                trade_direct_flips += 1
                trade_exits += 1
                trade_entries += 1
                if side_before > 0:
                    trade_long_exits += 1
                    trade_short_entries += 1
                else:
                    trade_short_exits += 1
                    trade_long_entries += 1
        prev_regime_state = int(regime_state)
        reward_sum += float(step_reward)
        equity_curve.append(info["equity"])
        done = terminated or truncated

    start_idx = window_size - 1
    end_idx = start_idx + len(equity_curve)
    open_prices = market_df["open"].iloc[start_idx:end_idx].to_numpy(dtype=np.float64)
    benchmark_curve = build_underlying_curve(open_prices, initial_equity=initial_equity)

    report = summarize_metrics(
        equity=equity_curve,
        underlying=benchmark_curve,
        periods_per_year=periods_per_year_from_timeframe(timeframe),
    )
    report["strict_mdd_pct"] = float(report.get("max_drawdown_pct", 0.0))
    report["num_steps"] = float(len(equity_curve) - 1)
    report["cumulative_reward"] = float(reward_sum)
    report["mean_step_reward"] = float(reward_sum / max(1, len(equity_curve) - 1))
    total_actions = max(1, int(len(equity_curve) - 1))
    report["action_counts"] = {
        "buy": float(action_counts[0]),
        "hold": float(action_counts[1]),
        "sell": float(action_counts[2]),
    }
    report["action_ratio"] = {
        "buy": action_counts[0] / total_actions,
        "hold": action_counts[1] / total_actions,
        "sell": action_counts[2] / total_actions,
    }
    report["mean_action_prob"] = {
        "buy": float(action_prob_sum[0] / total_actions),
        "hold": float(action_prob_sum[1] / total_actions),
        "sell": float(action_prob_sum[2] / total_actions),
    }
    report["deterministic"] = bool(deterministic)
    report["decision_mode"] = str(decision_mode)
    report["flat_start_policy"] = flat_start_policy
    report["directional_tie_hold_eps"] = float(directional_tie_hold_eps)
    report["debiased_action"] = debiased_action
    report["score_entry_threshold"] = float(score_entry_threshold)
    report["score_flip_threshold"] = float(score_flip_threshold)
    report["score_neutral_band"] = float(score_neutral_band)
    report["score_exit_threshold"] = float(score_exit_threshold)
    report["score_centering"] = str(score_centering)
    report["score_center_alpha"] = float(score_center_alpha)
    report["trend_guard"] = trend_guard
    report["trend_threshold"] = float(trend_threshold)
    report["volatility_gate"] = str(volatility_gate)
    report["volatility_threshold"] = float(volatility_threshold)
    report["hold_action_mode"] = str(hold_action_mode)
    report["scalar_feature_mode"] = str(scalar_feature_mode)
    report["source"] = source
    report["timeframe"] = timeframe
    report["window_size"] = int(window_size)
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
    if decision_mode == "blend_score_band":
        total_blend_steps = max(1, int(blend_weight_count))
        report["blend_score_band"] = {
            "model_a": blend_model_a_path_resolved,
            "model_b": blend_model_b_path_resolved,
            "weight_mode": str(blend_weight_mode),
            "weight_a": float(blend_weight_a),
            "weight_up": float(blend_weight_up),
            "weight_down": float(blend_weight_down),
            "trend_threshold": float(blend_trend_threshold),
            "mean_weight_a": float(blend_weight_sum / total_blend_steps),
            "trend_bucket_ratio": {
                "up": float(blend_trend_up_count / total_blend_steps),
                "neutral": float(blend_trend_neutral_count / total_blend_steps),
                "down": float(blend_trend_down_count / total_blend_steps),
            },
            "debiased_action": str(debiased_action),
            "score_thresholds": {
                "entry": float(score_entry_threshold),
                "flip": float(score_flip_threshold),
                "neutral": float(score_neutral_band),
                "exit": float(score_exit_threshold),
            },
        }
    if decision_mode == "regime_switch":
        total_regime_steps = max(
            1, int(regime_counts["up"] + regime_counts["neutral"] + regime_counts["down"])
        )
        report["regime_switch"] = {
            "model_up": up_model_path_resolved,
            "model_down": down_model_path_resolved,
            "neutral_policy": str(regime_neutral_policy),
            "transition_mode": str(regime_transition_mode),
            "enter_threshold": float(regime_enter_threshold),
            "confirm_bars": int(regime_confirm_bars),
            "weights": {
                "ret_long": float(regime_weight_ret_long),
                "ema_slope": float(regime_weight_ema_slope),
                "ret_short": float(regime_weight_ret_short),
                "vol": float(regime_weight_vol),
                "drawdown": float(regime_weight_drawdown),
            },
            "score_mode": str(regime_score_mode),
            "lookbacks": {
                "ret_short": int(regime_ret_short),
                "ret_long": int(regime_ret_long),
                "ema_fast": int(regime_ema_fast),
                "ema_slow": int(regime_ema_slow),
                "vol": int(regime_vol_lookback),
                "z_window": int(regime_z_window),
            },
            "counts": {
                "up": int(regime_counts["up"]),
                "neutral": int(regime_counts["neutral"]),
                "down": int(regime_counts["down"]),
            },
            "ratio": {
                "up": float(regime_counts["up"] / total_regime_steps),
                "neutral": float(regime_counts["neutral"] / total_regime_steps),
                "down": float(regime_counts["down"] / total_regime_steps),
            },
            "model_usage": {
                "up_model": int(regime_model_usage["up_model"]),
                "down_model": int(regime_model_usage["down_model"]),
                "neutral_hold": int(regime_model_usage["neutral_hold"]),
            },
            "forced_alignment_count": int(regime_forced_alignment_count),
            "mean_score": float(np.mean(regime_score_arr)) if regime_score_arr is not None else 0.0,
            "std_score": float(np.std(regime_score_arr)) if regime_score_arr is not None else 0.0,
        }
    return report


def run_backtest_multi_seed(
    model_path: str,
    seeds: list[int],
    **kwargs,
) -> dict:
    """
    Run repeated backtests across seeds and return per-seed + aggregate metrics.

    Primarily useful for synthetic source robustness checks.
    """
    if not seeds:
        raise ValueError("seeds must not be empty")

    reports = []
    for s in seeds:
        rep = run_backtest(model_path=model_path, seed=int(s), **kwargs)
        rep["seed"] = int(s)
        reports.append(rep)

    def _mean(key: str) -> float:
        vals = [float(r[key]) for r in reports]
        return float(np.mean(vals))

    def _std(key: str) -> float:
        vals = [float(r[key]) for r in reports]
        return float(np.std(vals))

    summary = {
        "num_seeds": float(len(seeds)),
        "mean_cumulative_return_pct": _mean("cumulative_return_pct"),
        "std_cumulative_return_pct": _std("cumulative_return_pct"),
        "mean_sharpe_ratio": _mean("sharpe_ratio"),
        "std_sharpe_ratio": _std("sharpe_ratio"),
        "mean_max_drawdown_pct": _mean("max_drawdown_pct"),
        "std_max_drawdown_pct": _std("max_drawdown_pct"),
    }
    return {"summary": summary, "reports": reports}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest trained PPO checkpoint.")
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
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--synthetic-drift", type=float, default=0.0)
    parser.add_argument("--synthetic-regime-amplitude", type=float, default=0.0004)
    parser.add_argument("--synthetic-regime-period", type=int, default=720)
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints/ppo_option_a_smoke.zip",
        help="Path to saved model zip",
    )
    parser.add_argument("--window-size", type=int, default=96)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--initial-equity", type=float, default=1000.0)
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-rate", type=float, default=0.0001)
    parser.add_argument("--flat-hold-penalty", type=float, default=0.0)
    parser.add_argument(
        "--hold-action-mode",
        type=str,
        default="auto",
        choices=["auto", "flat", "maintain"],
        help="Meaning of HOLD action. auto uses model metadata when available.",
    )
    parser.add_argument(
        "--use-images",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="Whether to render chart images for backtest observations.",
    )
    parser.add_argument("--image-cache-dir", type=str, default="data/image_cache_backtest")
    parser.add_argument(
        "--image-render-backend",
        type=str,
        default="auto",
        choices=["auto", "matplotlib", "fast"],
        help="Image renderer for chart observations. auto uses model metadata when available.",
    )
    parser.add_argument(
        "--scalar-feature-mode",
        type=str,
        default="auto",
        choices=["auto", "legacy4", "extended_v1"],
        help="Scalar feature set. auto uses model metadata when available.",
    )
    parser.add_argument(
        "--deterministic",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Use deterministic policy action (argmax) or sampled action.",
    )
    parser.add_argument(
        "--decision-mode",
        type=str,
        default="policy",
        choices=["policy", "score_band", "regime_switch", "blend_score_band"],
        help="Action selection mode: raw policy or score-band hysteresis.",
    )
    parser.add_argument(
        "--flat-start-policy",
        type=str,
        default="as_is",
        choices=["as_is", "prefer_entry"],
        help="When deterministic and flat, optionally force directional entry instead of HOLD.",
    )
    parser.add_argument(
        "--directional-tie-hold-eps",
        type=float,
        default=0.0,
        help="If |P(buy)-P(sell)| is below this threshold, force HOLD to avoid arbitrary side bias.",
    )
    parser.add_argument(
        "--debiased-action",
        type=str,
        default="off",
        choices=["off", "mirror_scalar"],
        help="Inference-time direction debiasing. mirror_scalar uses mirrored scalar pass.",
    )
    parser.add_argument(
        "--blend-model-a",
        type=str,
        default="",
        help="Model path for blend model A (blend_score_band mode).",
    )
    parser.add_argument(
        "--blend-model-b",
        type=str,
        default="",
        help="Model path for blend model B (blend_score_band mode).",
    )
    parser.add_argument(
        "--blend-weight-mode",
        type=str,
        default="static",
        choices=["static", "trend"],
    )
    parser.add_argument("--blend-weight-a", type=float, default=0.75)
    parser.add_argument("--blend-weight-up", type=float, default=0.90)
    parser.add_argument("--blend-weight-down", type=float, default=0.35)
    parser.add_argument("--blend-trend-threshold", type=float, default=0.002)
    parser.add_argument("--score-entry-threshold", type=float, default=0.02)
    parser.add_argument("--score-flip-threshold", type=float, default=0.05)
    parser.add_argument("--score-neutral-band", type=float, default=0.005)
    parser.add_argument(
        "--score-exit-threshold",
        type=float,
        default=-1.0,
        help="If >=0, force HOLD(flat) when in-position and |score| <= threshold.",
    )
    parser.add_argument(
        "--score-centering",
        type=str,
        default="off",
        choices=["off", "ema"],
        help="Optional centering for score-band mode.",
    )
    parser.add_argument("--score-center-alpha", type=float, default=0.02)
    parser.add_argument(
        "--trend-guard",
        type=str,
        default="off",
        choices=["off", "hard", "align_flat", "long_flat", "short_flat"],
        help="Optional heuristic override using window trend scalar.",
    )
    parser.add_argument("--trend-threshold", type=float, default=0.002)
    parser.add_argument(
        "--volatility-gate",
        type=str,
        default="off",
        choices=["off", "hard"],
    )
    parser.add_argument("--volatility-threshold", type=float, default=0.0)
    parser.add_argument(
        "--regime-model-up",
        type=str,
        default="",
        help="Model path to use when regime score indicates up-trend (regime_switch mode).",
    )
    parser.add_argument(
        "--regime-model-down",
        type=str,
        default="",
        help="Model path to use when regime score indicates down-trend (regime_switch mode).",
    )
    parser.add_argument(
        "--regime-neutral-policy",
        type=str,
        default="hold",
        choices=["hold", "up", "down"],
        help="How to act in neutral regime before confirmation.",
    )
    parser.add_argument(
        "--regime-transition-mode",
        type=str,
        default="force_align",
        choices=["none", "force_align", "force_on_switch"],
        help="Optional forced action alignment to regime direction.",
    )
    parser.add_argument("--regime-ret-short", type=int, default=60)
    parser.add_argument("--regime-ret-long", type=int, default=240)
    parser.add_argument("--regime-ema-fast", type=int, default=60)
    parser.add_argument("--regime-ema-slow", type=int, default=240)
    parser.add_argument("--regime-vol-lookback", type=int, default=60)
    parser.add_argument("--regime-z-window", type=int, default=1440)
    parser.add_argument(
        "--regime-score-mode",
        type=str,
        default="zscore",
        choices=["zscore", "raw"],
    )
    parser.add_argument("--regime-weight-ret-long", type=float, default=0.45)
    parser.add_argument("--regime-weight-ema-slope", type=float, default=0.35)
    parser.add_argument("--regime-weight-ret-short", type=float, default=0.20)
    parser.add_argument("--regime-weight-vol", type=float, default=-0.25)
    parser.add_argument("--regime-weight-drawdown", type=float, default=-0.15)
    parser.add_argument("--regime-enter-threshold", type=float, default=0.7)
    parser.add_argument("--regime-confirm-bars", type=int, default=3)
    parser.add_argument(
        "--eval-seeds",
        type=str,
        default="",
        help="Comma-separated seeds for repeated evaluation (e.g. 123,456,789).",
    )
    parser.add_argument("--output", type=str, default="results/backtest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    common_kwargs = dict(
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
        regime_score_mode=args.regime_score_mode,
        regime_ret_short=args.regime_ret_short,
        regime_ret_long=args.regime_ret_long,
        regime_ema_fast=args.regime_ema_fast,
        regime_ema_slow=args.regime_ema_slow,
        regime_vol_lookback=args.regime_vol_lookback,
        regime_z_window=args.regime_z_window,
        regime_weight_ret_long=args.regime_weight_ret_long,
        regime_weight_ema_slope=args.regime_weight_ema_slope,
        regime_weight_ret_short=args.regime_weight_ret_short,
        regime_weight_vol=args.regime_weight_vol,
        regime_weight_drawdown=args.regime_weight_drawdown,
        regime_enter_threshold=args.regime_enter_threshold,
        regime_confirm_bars=args.regime_confirm_bars,
    )
    if args.eval_seeds.strip():
        seeds = [
            int(x.strip()) for x in args.eval_seeds.split(",") if x.strip()
        ]
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
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"Saved backtest report to: {output_path.resolve()}")
    print(report)


if __name__ == "__main__":
    main()
