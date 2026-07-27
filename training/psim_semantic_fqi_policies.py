"""Numeric fitted-Q policies for frozen PSIM semantic representations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np


POSITION_NAMES = ("POSITION_SHORT", "POSITION_FLAT", "POSITION_LONG")
POSITION_INDEX = {name: index for index, name in enumerate(POSITION_NAMES)}
ACTION_NAMES = ("TARGET_FLAT", "TARGET_SHORT", "TARGET_LONG")
ACTION_VALUES = (0.0, -0.5, 0.5)
ACTION_TO_POSITION = (1, 0, 2)
INTERNAL_PERMUTED_ACTION_NAMES = (
    "TARGET_LONG",
    "TARGET_SHORT",
    "TARGET_FLAT",
)
GAMMA = 0.99
BELLMAN_ITERATIONS = 25
RIDGE_ALPHA = 100.0
RANDOM_SEED = 20_260_727
Q_TIE_TOLERANCE = 1e-12
CIRCULAR_SHIFT = 21
EXTRA_TREES_KWARGS = {
    "n_estimators": 512,
    "max_depth": 6,
    "min_samples_split": 24,
    "min_samples_leaf": 12,
    "max_features": "sqrt",
    "bootstrap": False,
    "criterion": "squared_error",
    "random_state": RANDOM_SEED,
    "n_jobs": 1,
}


class _RidgeMultiOutput:
    def __init__(self, coefficient: np.ndarray):
        self.coefficient = np.asarray(coefficient, dtype=np.float64)

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray) -> "_RidgeMultiOutput":
        design = np.column_stack([np.ones(len(x)), x])
        penalty = np.eye(design.shape[1], dtype=np.float64) * RIDGE_ALPHA
        penalty[0, 0] = 0.0
        coefficient = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ y,
        )
        return cls(coefficient)

    def predict(self, x: np.ndarray) -> np.ndarray:
        design = np.column_stack([np.ones(len(x)), x])
        return design @ self.coefficient


def _fit_estimator(
    algorithm: str,
    x: np.ndarray,
    y: np.ndarray,
) -> Any:
    if algorithm == "ridge":
        return _RidgeMultiOutput.fit(x, y)
    if algorithm == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        estimator = ExtraTreesRegressor(**EXTRA_TREES_KWARGS)
        estimator.fit(x, y)
        return estimator
    raise ValueError(f"unknown PSIM semantic FQI algorithm: {algorithm}")


def _position_features(
    features: np.ndarray,
    position_indices: np.ndarray,
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    positions = np.asarray(position_indices, dtype=np.int64)
    if values.ndim != 2 or positions.shape != (len(values),):
        raise ValueError("PSIM semantic state/position shape changed")
    if len(positions) and (
        positions.min() < 0 or positions.max() >= len(POSITION_NAMES)
    ):
        raise ValueError("PSIM semantic position index changed")
    one_hot = np.zeros((len(values), len(POSITION_NAMES)), dtype=np.float64)
    if len(values):
        one_hot[np.arange(len(values)), positions] = 1.0
    return np.concatenate([values, one_hot], axis=1)


@dataclass
class FittedQPolicy:
    algorithm: str
    feature_width: int
    estimator: Any
    action_names: tuple[str, str, str] = ACTION_NAMES

    def predict_q_many(
        self,
        features: np.ndarray,
        positions: Sequence[str],
    ) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.feature_width:
            raise ValueError("PSIM semantic inference feature shape changed")
        if len(values) != len(positions):
            raise ValueError("PSIM semantic inference rows do not align")
        try:
            indices = np.asarray(
                [POSITION_INDEX[position] for position in positions],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ValueError("PSIM semantic inference position changed") from exc
        predicted = np.asarray(
            self.estimator.predict(_position_features(values, indices)),
            dtype=np.float64,
        )
        if predicted.shape != (len(values), 3):
            raise ValueError("PSIM semantic fitted-Q output shape changed")
        return predicted

    def predict_q(
        self,
        feature: np.ndarray,
        position: str,
    ) -> np.ndarray:
        return self.predict_q_many(
            np.asarray(feature, dtype=np.float64).reshape(1, -1),
            [position],
        )[0]


def _complete_reward_indices(rewards: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isfinite(rewards).all(axis=(1, 2)))


def _month_key(value: Any) -> str:
    return str(value)[:7]


def _prepare_rewards(
    reward_tensor: np.ndarray,
    *,
    reward_mode: str,
    months: Sequence[Any] | None,
    action_indices: Sequence[int],
) -> np.ndarray:
    rewards = np.asarray(reward_tensor, dtype=np.float64)
    if rewards.ndim != 3 or rewards.shape[1:] != (3, 3):
        raise ValueError("PSIM semantic reward tensor shape changed")
    transformed = rewards.copy()
    complete = _complete_reward_indices(rewards)
    if reward_mode == "base":
        pass
    elif reward_mode == "circular_21":
        transformed[complete] = np.roll(
            rewards[complete],
            shift=CIRCULAR_SHIFT,
            axis=0,
        )
    elif reward_mode == "within_month_shuffled":
        if months is None or len(months) != len(rewards):
            raise ValueError("PSIM semantic reward months changed")
        for month in sorted({_month_key(month) for month in months}):
            indices = [
                index
                for index in complete.tolist()
                if _month_key(months[index]) == month
            ]
            ordered = sorted(
                indices,
                key=lambda index: hashlib.sha256(
                    f"{RANDOM_SEED}\x00{month}\x00{index}".encode("utf-8")
                ).digest(),
            )
            transformed[indices] = rewards[ordered]
    else:
        raise ValueError(f"unknown PSIM semantic reward mode: {reward_mode}")
    return transformed[:, :, list(action_indices)]


def fit_fitted_q(
    features: np.ndarray,
    reward_tensor: np.ndarray,
    terminal: np.ndarray,
    reachable_mask: np.ndarray,
    *,
    algorithm: str,
    reward_mode: str = "base",
    months: Sequence[Any] | None = None,
    internal_action_names: tuple[str, str, str] = ACTION_NAMES,
) -> FittedQPolicy:
    values = np.asarray(features, dtype=np.float64)
    rewards = np.asarray(reward_tensor, dtype=np.float64)
    terminal_array = np.asarray(terminal, dtype=bool)
    reachable = np.asarray(reachable_mask, dtype=bool)
    count = len(values)
    if (
        values.ndim != 2
        or not np.isfinite(values).all()
        or rewards.shape != (count, 3, 3)
        or terminal_array.shape != (count,)
        or reachable.shape != (count, 3)
        or set(internal_action_names) != set(ACTION_NAMES)
    ):
        raise ValueError("PSIM semantic FQI training arrays changed")
    action_indices = tuple(
        ACTION_NAMES.index(name) for name in internal_action_names
    )
    prepared = _prepare_rewards(
        rewards,
        reward_mode=reward_mode,
        months=months,
        action_indices=action_indices,
    )
    if not np.isfinite(prepared[reachable]).all():
        raise ValueError("PSIM semantic reachable rewards are non-finite")
    state_indices: list[int] = []
    position_indices: list[int] = []
    for state_index in range(count):
        for position_index in range(3):
            if reachable[state_index, position_index]:
                state_indices.append(state_index)
                position_indices.append(position_index)
    if not state_indices:
        raise ValueError("PSIM semantic FQI has no reachable states")
    state_index_array = np.asarray(state_indices, dtype=np.int64)
    position_index_array = np.asarray(position_indices, dtype=np.int64)
    x = _position_features(
        values[state_index_array],
        position_index_array,
    )
    y = np.vstack(
        [
            prepared[state_index, position_index]
            for state_index, position_index in zip(
                state_indices,
                position_indices,
            )
        ]
    )
    estimator = _fit_estimator(algorithm, x, y)
    policy = FittedQPolicy(
        algorithm=algorithm,
        feature_width=values.shape[1],
        estimator=estimator,
        action_names=internal_action_names,
    )
    for _ in range(1, BELLMAN_ITERATIONS):
        updated = y.copy()
        continuation_features: list[np.ndarray] = []
        continuation_positions: list[str] = []
        continuation_slots: list[tuple[int, int]] = []
        for train_index, state_index in enumerate(state_indices):
            if terminal_array[state_index] or state_index + 1 >= count:
                continue
            for action_index, action_name in enumerate(internal_action_names):
                continuation_features.append(values[state_index + 1])
                continuation_positions.append(
                    action_name.replace("TARGET", "POSITION")
                )
                continuation_slots.append((train_index, action_index))
        if continuation_features:
            next_q = policy.predict_q_many(
                np.vstack(continuation_features),
                continuation_positions,
            )
            updated = np.vstack(
                [
                    prepared[state_index, position_index].copy()
                    for state_index, position_index in zip(
                        state_indices,
                        position_indices,
                    )
                ]
            )
            for (train_index, action_index), q_values in zip(
                continuation_slots,
                next_q,
            ):
                finite = np.isfinite(q_values)
                continuation = (
                    float(np.max(q_values[finite])) if finite.any() else 0.0
                )
                updated[train_index, action_index] += GAMMA * continuation
        estimator = _fit_estimator(algorithm, x, updated)
        policy = FittedQPolicy(
            algorithm=algorithm,
            feature_width=values.shape[1],
            estimator=estimator,
            action_names=internal_action_names,
        )
        y = updated
    return policy


def choose_action_index(
    q_values: Sequence[float],
    current_position: str,
) -> int:
    values = np.asarray(q_values, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).any():
        return ACTION_NAMES.index("TARGET_FLAT")
    best = float(np.max(values[np.isfinite(values)]))
    candidates = {
        index
        for index, value in enumerate(values)
        if np.isfinite(value) and value >= best - Q_TIE_TOLERANCE
    }
    current_target = current_position.replace("POSITION", "TARGET")
    order = [
        ACTION_NAMES.index("TARGET_FLAT"),
        ACTION_NAMES.index(current_target),
        ACTION_NAMES.index("TARGET_SHORT"),
        ACTION_NAMES.index("TARGET_LONG"),
    ]
    return next(index for index in order if index in candidates)


def rollout_policy(
    policy: Any,
    features: np.ndarray,
    *,
    start_position: str = "POSITION_FLAT",
) -> list[str]:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("PSIM semantic rollout features changed")
    current = start_position
    targets: list[str] = []
    for feature in values:
        q_values = np.asarray(policy.predict_q(feature, current), dtype=float)
        target = ACTION_NAMES[choose_action_index(q_values, current)]
        targets.append(target)
        current = target.replace("TARGET", "POSITION")
    return targets


@dataclass(frozen=True)
class ConstantPolicy:
    target: str

    def predict_q(self, feature: np.ndarray, position: str) -> np.ndarray:
        output = np.full(3, -1.0, dtype=np.float64)
        output[ACTION_NAMES.index(self.target)] = 1.0
        return output


@dataclass(frozen=True)
class PersistencePolicy:
    def predict_q(self, feature: np.ndarray, position: str) -> np.ndarray:
        target = position.replace("POSITION", "TARGET")
        output = np.full(3, -1.0, dtype=np.float64)
        output[ACTION_NAMES.index(target)] = 1.0
        return output


@dataclass(frozen=True)
class ExactPayloadMemoryPolicy:
    table: Mapping[tuple[str, str], np.ndarray]
    signatures: Sequence[str]

    def predict_q_at(
        self,
        row_index: int,
        position: str,
    ) -> np.ndarray:
        value = self.table.get((self.signatures[row_index], position))
        if value is None:
            return np.array([0.0, -np.inf, -np.inf], dtype=np.float64)
        return np.asarray(value, dtype=np.float64)


def fit_exact_payload_memory(
    signatures: Sequence[str],
    reward_tensor: np.ndarray,
    reachable_mask: np.ndarray,
) -> dict[tuple[str, str], np.ndarray]:
    rewards = np.asarray(reward_tensor, dtype=np.float64)
    reachable = np.asarray(reachable_mask, dtype=bool)
    if rewards.shape != (len(signatures), 3, 3):
        raise ValueError("PSIM semantic memory reward shape changed")
    sums: dict[tuple[str, str], np.ndarray] = {}
    counts: dict[tuple[str, str], int] = {}
    for index, signature in enumerate(signatures):
        for position_index, position in enumerate(POSITION_NAMES):
            if not reachable[index, position_index]:
                continue
            key = (str(signature), position)
            sums[key] = sums.get(key, np.zeros(3)) + rewards[
                index,
                position_index,
            ]
            counts[key] = counts.get(key, 0) + 1
    return {key: value / counts[key] for key, value in sums.items()}


def rollout_exact_payload_memory(
    table: Mapping[tuple[str, str], np.ndarray],
    signatures: Sequence[str],
    *,
    start_position: str = "POSITION_FLAT",
) -> list[str]:
    current = start_position
    targets: list[str] = []
    for signature in signatures:
        q_values = table.get((str(signature), current))
        if q_values is None:
            target = "TARGET_FLAT"
        else:
            target = ACTION_NAMES[choose_action_index(q_values, current)]
        targets.append(target)
        current = target.replace("TARGET", "POSITION")
    return targets


@dataclass(frozen=True)
class DirectionFlipPolicy:
    base: Any

    def predict_q(self, feature: np.ndarray, position: str) -> np.ndarray:
        return np.asarray(
            self.base.predict_q(feature, position),
            dtype=np.float64,
        )[[0, 2, 1]]


@dataclass(frozen=True)
class CanonicalActionPermutationPolicy:
    base: FittedQPolicy
    internal_order: tuple[str, str, str] = INTERNAL_PERMUTED_ACTION_NAMES

    def predict_q(self, feature: np.ndarray, position: str) -> np.ndarray:
        internal = np.asarray(
            self.base.predict_q(feature, position),
            dtype=np.float64,
        )
        return internal[
            [self.internal_order.index(name) for name in ACTION_NAMES]
        ]


def fit_action_code_permutation(
    features: np.ndarray,
    reward_tensor: np.ndarray,
    terminal: np.ndarray,
    reachable_mask: np.ndarray,
    *,
    algorithm: str,
    months: Sequence[Any] | None = None,
) -> CanonicalActionPermutationPolicy:
    base = fit_fitted_q(
        features,
        reward_tensor,
        terminal,
        reachable_mask,
        algorithm=algorithm,
        months=months,
        internal_action_names=INTERNAL_PERMUTED_ACTION_NAMES,
    )
    return CanonicalActionPermutationPolicy(base)
