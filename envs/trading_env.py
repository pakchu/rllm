"""Gymnasium trading environment with leak-safe observation logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from preprocessing.market_features import CORE_MARKET_FEATURE_COLUMNS, build_market_feature_frame
from preprocessing.timeframe import make_window


ACTION_BUY = 0
ACTION_HOLD = 1
ACTION_SELL = 2


@dataclass
class TradingEnvConfig:
    """Configuration for :class:`TradingEnv`."""

    window_size: int = 96
    initial_equity: float = 1_000.0
    fee_rate: float = 0.0005  # 5bp
    slippage_rate: float = 0.0001  # 1bp
    leverage: float = 1.0
    max_leverage: float = 10.0
    liquidation_penalty: float = -10.0
    flat_hold_penalty: float = 0.0
    open_position_bonus: float = 0.0
    directional_bias_penalty: float = 0.0
    directional_symmetry_prob: float = 0.0
    action_balance_penalty: float = 0.0
    drawdown_penalty: float = 0.0
    random_reset_start: bool = False
    episode_horizon: int = 0  # 0 disables early truncation; >0 truncates after N env steps
    scalar_feature_mode: str = "legacy4"
    hold_action_mode: str = "flat"  # "flat" -> HOLD targets flat, "maintain" -> HOLD keeps side
    image_shape: Tuple[int, int, int] = (3, 320, 320)


class TradingEnv(gym.Env):
    """
    BTCUSDT-style trading environment.

    Policy:
        - Action at time `t` is executed at `t+1 open`
        - Position sizing uses 100% of equity (with leverage multiplier)
        - Reward is `log(Equity_{t+1} / Equity_t)`
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        market_df: pd.DataFrame,
        images: Optional[np.ndarray] = None,
        config: Optional[TradingEnvConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config or TradingEnvConfig()
        self.window_size = int(self.config.window_size)
        self.initial_equity = float(self.config.initial_equity)
        self.fee_rate = float(self.config.fee_rate)
        self.slippage_rate = float(self.config.slippage_rate)
        self.leverage = float(self.config.leverage)
        self.max_leverage = float(self.config.max_leverage)
        self.liquidation_penalty = float(self.config.liquidation_penalty)
        self.flat_hold_penalty = float(self.config.flat_hold_penalty)
        self.open_position_bonus = float(self.config.open_position_bonus)
        self.directional_bias_penalty = float(self.config.directional_bias_penalty)
        self.directional_symmetry_prob = float(self.config.directional_symmetry_prob)
        self.action_balance_penalty = float(self.config.action_balance_penalty)
        self.drawdown_penalty = float(self.config.drawdown_penalty)
        self.random_reset_start = bool(self.config.random_reset_start)
        self.episode_horizon = int(self.config.episode_horizon)
        self.scalar_feature_mode = str(self.config.scalar_feature_mode).lower().strip()
        self.hold_action_mode = str(self.config.hold_action_mode).lower().strip()
        self.image_shape = tuple(self.config.image_shape)

        if not (1.0 <= self.leverage <= self.max_leverage):
            raise ValueError(
                f"leverage must be in [1, {self.max_leverage}], got {self.leverage}"
            )
        if self.hold_action_mode not in {"flat", "maintain"}:
            raise ValueError(
                "hold_action_mode must be one of {'flat', 'maintain'}, "
                f"got {self.hold_action_mode}"
            )
        if not (0.0 <= self.directional_symmetry_prob <= 1.0):
            raise ValueError(
                "directional_symmetry_prob must be in [0, 1], "
                f"got {self.directional_symmetry_prob}"
            )
        if self.episode_horizon < 0:
            raise ValueError(
                f"episode_horizon must be >= 0, got {self.episode_horizon}"
            )
        if self.scalar_feature_mode not in {"legacy4", "extended_v1"}:
            raise ValueError(
                "scalar_feature_mode must be one of {'legacy4', 'extended_v1'}, "
                f"got {self.scalar_feature_mode}"
            )

        required_columns = {"date", "open", "high", "low", "close", "volume"}
        missing = required_columns.difference(market_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        self.market_df = market_df.copy().reset_index(drop=True)
        self.market_df["date"] = pd.to_datetime(self.market_df["date"])

        min_required_rows = self.window_size + 1
        if len(self.market_df) < min_required_rows:
            raise ValueError(
                f"Need at least {min_required_rows} rows, got {len(self.market_df)}"
            )

        self.images = images
        self.images_start_step = 0
        if images is not None:
            if len(images) == len(self.market_df):
                self.images_start_step = 0
            elif len(images) == len(self.market_df) - self.window_size + 1:
                self.images_start_step = self.window_size - 1
            else:
                raise ValueError(
                    "images length must be len(market_df) or "
                    "len(market_df)-window_size+1"
                )

        max_float = np.finfo(np.float32).max
        scalar_low, scalar_high = self._scalar_bounds(max_float=max_float)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0.0, high=1.0, shape=self.image_shape, dtype=np.float32
                ),
                "scalars": spaces.Box(
                    low=scalar_low,
                    high=scalar_high,
                    dtype=np.float32,
                ),
            }
        )

        self.current_step = 0
        self.equity = self.initial_equity
        self.side = 0  # -1: short, 0: flat, +1: long
        self.last_entry_price = 0.0
        self.total_fees = 0.0
        self.total_slippage = 0.0
        self.directional_exposure_sum = 0.0
        self.directional_steps = 0
        self.mirror_sign = 1
        self.policy_buy_count = 0
        self.policy_sell_count = 0
        self.peak_equity = self.initial_equity
        self.episode_steps = 0
        self.position_age_steps = 0
        self.market_feature_cache = self._build_market_feature_cache()

    def _scalar_bounds(self, max_float: float) -> tuple[np.ndarray, np.ndarray]:
        if self.scalar_feature_mode == "legacy4":
            low = np.array([-1.0, -max_float, 0.0, -max_float], dtype=np.float32)
            high = np.array([1.0, max_float, max_float, max_float], dtype=np.float32)
            return low, high

        low = np.array(
            [
                -1.0,       # side
                -max_float, # unrealized pnl
                0.0,        # range volatility
                -max_float, # trend 12
                -max_float, # trend 24
                -max_float, # trend 96
                -max_float, # close vs sma12
                -max_float, # close vs sma24
                -max_float, # close vs sma48
                -1.0,       # rsi14 normalized
                -1.0,       # mfi14 normalized
                -max_float, # bollinger z
                -1.0,       # range position
                0.0,        # equity drawdown fraction
                0.0,        # holding age fraction
            ],
            dtype=np.float32,
        )
        high = np.array(
            [
                1.0,
                max_float,
                max_float,
                max_float,
                max_float,
                max_float,
                max_float,
                max_float,
                max_float,
                1.0,
                1.0,
                max_float,
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )
        return low, high

    def _action_to_target_side(self, action: int) -> int:
        if action == ACTION_BUY:
            return 1
        if action == ACTION_SELL:
            return -1
        if action == ACTION_HOLD:
            if self.hold_action_mode == "flat":
                return 0
            return self.side
        raise ValueError(f"Invalid action: {action}")

    def _map_policy_action(self, action: int) -> int:
        """Map policy action to market action under optional mirrored episode."""
        if self.mirror_sign >= 0:
            return int(action)
        if int(action) == ACTION_BUY:
            return ACTION_SELL
        if int(action) == ACTION_SELL:
            return ACTION_BUY
        return ACTION_HOLD

    def _unrealized_pnl_pct(self) -> float:
        if self.side == 0 or self.last_entry_price <= 0.0:
            return 0.0
        curr_open = float(self.market_df.loc[self.current_step, "open"])
        if curr_open <= 0.0:
            return 0.0
        raw = (curr_open - self.last_entry_price) / self.last_entry_price
        return float(self.side * raw)

    def _slippage_fill_price(self, base_price: float, target_side: int) -> float:
        if target_side > 0:
            return base_price * (1.0 + self.slippage_rate)
        if target_side < 0:
            return base_price * (1.0 - self.slippage_rate)
        return base_price

    def _range_volatility_pct(self) -> float:
        return float(self.market_feature_cache[self.current_step, 0])

    def _window_trend_pct(self) -> float:
        return float(self.market_feature_cache[self.current_step, 3])

    def _build_market_feature_cache(self) -> np.ndarray:
        frame = build_market_feature_frame(self.market_df, window_size=self.window_size)
        cache = frame.loc[:, list(CORE_MARKET_FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        return np.nan_to_num(cache, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _lookback_return(close_arr: np.ndarray, lookback: int) -> float:
        if len(close_arr) < 2:
            return 0.0
        anchor_idx = max(0, len(close_arr) - int(lookback))
        anchor = float(close_arr[anchor_idx])
        last_close = float(close_arr[-1])
        if anchor == 0.0:
            return 0.0
        return (last_close - anchor) / anchor

    @staticmethod
    def _mean_ratio(close_arr: np.ndarray, lookback: int) -> float:
        if len(close_arr) == 0:
            return 0.0
        tail = close_arr[-min(len(close_arr), int(lookback)) :]
        ref = float(np.mean(tail))
        last_close = float(close_arr[-1])
        if ref == 0.0:
            return 0.0
        return (last_close - ref) / ref

    def _build_extended_scalars(self) -> np.ndarray:
        market = self.market_feature_cache[self.current_step]
        peak = max(float(self.peak_equity), 1e-12)
        equity_drawdown = max(0.0, 1.0 - float(self.equity) / peak)
        age_denom = float(self.episode_horizon if self.episode_horizon > 0 else self.window_size)
        holding_age_frac = min(float(self.position_age_steps) / max(age_denom, 1.0), 1.0)

        return np.array(
            [
                float(self.side * self.mirror_sign),
                float(self._unrealized_pnl_pct() * self.mirror_sign),
                float(market[0]),
                float(market[1] * self.mirror_sign),
                float(market[2] * self.mirror_sign),
                float(market[3] * self.mirror_sign),
                float(market[4] * self.mirror_sign),
                float(market[5] * self.mirror_sign),
                float(market[6] * self.mirror_sign),
                float(market[7] * self.mirror_sign),
                float(market[8] * self.mirror_sign),
                float(market[9] * self.mirror_sign),
                float(market[10] * self.mirror_sign),
                float(equity_drawdown),
                float(holding_age_frac),
            ],
            dtype=np.float32,
        )

    def _get_observation(self) -> dict:
        if self.images is None:
            image = np.zeros(self.image_shape, dtype=np.float32)
        else:
            image_idx = self.current_step - self.images_start_step
            image = np.asarray(self.images[image_idx], dtype=np.float32)
            if image.shape != self.image_shape:
                raise ValueError(
                    f"Expected image shape {self.image_shape}, got {image.shape}"
                )

        if self.scalar_feature_mode == "extended_v1":
            scalars = self._build_extended_scalars()
        else:
            scalars = np.array(
                [
                    float(self.side * self.mirror_sign),
                    float(self._unrealized_pnl_pct() * self.mirror_sign),
                    float(self._range_volatility_pct()),
                    float(self._window_trend_pct() * self.mirror_sign),
                ],
                dtype=np.float32,
            )
        return {"image": image, "scalars": scalars}

    def _get_info(self) -> dict:
        peak = max(float(self.peak_equity), 1e-12)
        dd = max(0.0, 1.0 - float(self.equity) / peak)
        return {
            "step": self.current_step,
            "episode_steps": int(self.episode_steps),
            "equity": float(self.equity),
            "equity_peak": float(self.peak_equity),
            "equity_drawdown_pct": float(dd * 100.0),
            "position_side": int(self.side),
            "last_entry_price": float(self.last_entry_price),
            "total_fees": float(self.total_fees),
            "total_slippage": float(self.total_slippage),
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        del options

        min_start = self.window_size - 1
        max_start = len(self.market_df) - 2  # must leave room for t+1 execution
        if self.episode_horizon > 0:
            horizon_limited_max_start = len(self.market_df) - 1 - self.episode_horizon
            if horizon_limited_max_start >= min_start:
                max_start = min(max_start, horizon_limited_max_start)
        if self.random_reset_start and max_start > min_start:
            self.current_step = int(self.np_random.integers(min_start, max_start + 1))
        else:
            self.current_step = min_start
        self.equity = self.initial_equity
        self.side = 0
        self.last_entry_price = 0.0
        self.total_fees = 0.0
        self.total_slippage = 0.0
        self.directional_exposure_sum = 0.0
        self.directional_steps = 0
        self.mirror_sign = 1
        self.policy_buy_count = 0
        self.policy_sell_count = 0
        self.peak_equity = self.initial_equity
        self.episode_steps = 0
        self.position_age_steps = 0
        if self.directional_symmetry_prob > 0.0:
            if float(self.np_random.random()) < self.directional_symmetry_prob:
                self.mirror_sign = -1
        return self._get_observation(), self._get_info()

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is not in action space")
        if self.current_step >= len(self.market_df) - 1:
            raise RuntimeError("Episode is finished. Call reset() before step().")

        equity_before = float(self.equity)
        open_t = float(self.market_df.loc[self.current_step, "open"])
        open_t1 = float(self.market_df.loc[self.current_step + 1, "open"])

        price_return = 0.0 if open_t == 0.0 else (open_t1 - open_t) / open_t
        pnl_return = self.side * self.leverage * price_return
        equity_after_pnl = max(equity_before * (1.0 + pnl_return), 0.0)

        side_before = int(self.side)
        market_action = self._map_policy_action(int(action))
        target_side = self._action_to_target_side(market_action)
        position_opened = side_before == 0 and target_side != 0
        held_flat_idle = side_before == 0 and target_side == 0 and market_action == ACTION_HOLD
        turnover_multiplier = abs(target_side - self.side)
        turnover_notional = turnover_multiplier * equity_after_pnl * self.leverage
        fee_cost = turnover_notional * self.fee_rate
        slippage_cost = turnover_notional * self.slippage_rate

        equity_after_trade = max(equity_after_pnl - fee_cost - slippage_cost, 0.0)
        self.total_fees += fee_cost
        self.total_slippage += slippage_cost

        if target_side != self.side:
            if target_side == 0:
                self.last_entry_price = 0.0
            else:
                self.last_entry_price = self._slippage_fill_price(open_t1, target_side)
        self.side = target_side
        if self.side == 0:
            self.position_age_steps = 0
        elif target_side != side_before:
            self.position_age_steps = 1
        else:
            self.position_age_steps += 1

        liquidated = equity_after_trade <= 0.0
        terminated = bool(liquidated)
        truncated = False
        self.episode_steps += 1

        if terminated:
            reward = float(self.liquidation_penalty)
            self.equity = 0.0
            self.peak_equity = max(float(self.peak_equity), float(self.equity))
            self.current_step = min(self.current_step + 1, len(self.market_df) - 1)
        else:
            self.equity = float(equity_after_trade)
            self.peak_equity = max(float(self.peak_equity), float(self.equity))
            self.current_step += 1
            if self.current_step >= len(self.market_df) - 1:
                truncated = True
            elif self.episode_horizon > 0 and self.episode_steps >= self.episode_horizon:
                truncated = True
            eps = 1e-12
            reward = float(np.log(max(self.equity, eps) / max(equity_before, eps)))
            if position_opened and self.open_position_bonus != 0.0:
                reward += float(self.open_position_bonus)
            if (
                self.flat_hold_penalty > 0.0
                and held_flat_idle
            ):
                reward -= float(self.flat_hold_penalty)
            self.directional_steps += 1
            self.directional_exposure_sum += float(self.side)
            directional_abs_mean = abs(
                self.directional_exposure_sum / max(1, self.directional_steps)
            )
            if self.directional_bias_penalty > 0.0:
                reward -= float(self.directional_bias_penalty) * directional_abs_mean
            policy_action = int(action)
            if policy_action == ACTION_BUY:
                self.policy_buy_count += 1
            elif policy_action == ACTION_SELL:
                self.policy_sell_count += 1
            directional_policy_actions = self.policy_buy_count + self.policy_sell_count
            policy_imbalance = 0.0
            if directional_policy_actions > 0:
                policy_imbalance = abs(
                    self.policy_buy_count - self.policy_sell_count
                ) / float(directional_policy_actions)
            if self.action_balance_penalty > 0.0:
                reward -= float(self.action_balance_penalty) * float(policy_imbalance)
            if self.drawdown_penalty > 0.0:
                peak = max(float(self.peak_equity), eps)
                curr_drawdown = max(0.0, 1.0 - float(self.equity) / peak)
                reward -= float(self.drawdown_penalty) * float(curr_drawdown)

        info = self._get_info()
        directional_policy_actions = self.policy_buy_count + self.policy_sell_count
        policy_imbalance = 0.0
        if directional_policy_actions > 0:
            policy_imbalance = abs(
                self.policy_buy_count - self.policy_sell_count
            ) / float(directional_policy_actions)
        info.update(
            {
                "fee_cost": float(fee_cost),
                "slippage_cost": float(slippage_cost),
                "liquidated": terminated,
                "directional_abs_mean": float(
                    abs(self.directional_exposure_sum / max(1, self.directional_steps))
                ),
                "mirror_sign": int(self.mirror_sign),
                "policy_directional_imbalance": float(policy_imbalance),
            }
        )
        return self._get_observation(), reward, terminated, truncated, info

    def render(self):
        """Simple text render."""
        print(
            f"step={self.current_step} equity={self.equity:.4f} "
            f"side={self.side} last_entry={self.last_entry_price:.4f}"
        )
