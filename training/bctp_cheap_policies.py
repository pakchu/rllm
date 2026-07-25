"""Frozen cheap policy machinery for BCTP-12H.

This module intentionally has no market, funding, or outcome-file access.  It
only consumes already-built categorical source states and per-transition reward
tensors supplied by the economic evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from training import preregister_block_clearing_relational_topology as bcrt
from training import preregister_block_clearing_target_position_mdp as bctp


POSITION_VALUES = (-0.5, 0.0, 0.5)
POSITIONS = ("POSITION_SHORT", "POSITION_FLAT", "POSITION_LONG")
POSITION_TO_INDEX = {name: index for index, name in enumerate(POSITIONS)}
POSITION_TO_TARGET_NAME = {
    "POSITION_SHORT": "TARGET_SHORT",
    "POSITION_FLAT": "TARGET_FLAT",
    "POSITION_LONG": "TARGET_LONG",
}
MODEL_ACTION_VALUES = (0.0, -0.5, 0.5)
MODEL_ACTION_NAMES = ("TARGET_FLAT", "TARGET_SHORT", "TARGET_LONG")
TARGET_ACTION_VALUES = MODEL_ACTION_VALUES
TARGET_ACTION_NAMES = MODEL_ACTION_NAMES
MODEL_ACTION_TO_POSITION = (1, 0, 2)  # flat->POSITION_FLAT, short->SHORT, long->LONG
INTERNAL_PERMUTED_ACTION_NAMES = (
    "TARGET_LONG",
    "TARGET_SHORT",
    "TARGET_FLAT",
)
TARGET_NAME_TO_VALUE = dict(zip(TARGET_ACTION_NAMES, TARGET_ACTION_VALUES))
TARGET_VALUE_TO_NAME = dict(zip(TARGET_ACTION_VALUES, TARGET_ACTION_NAMES))

GAMMA = 0.99
BELLMAN_ITERATIONS = 25
RCOND = 1e-12
RIDGE_ALPHA = 100.0
Q_TIE_TOLERANCE = 1e-12
RANDOM_SEED = 20_260_725
CIRCULAR_SHIFT = 21
MASKED_TOKEN_NAMES = ("order_transition", "leader_transition")

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

def _token_schema() -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(
        (name, tuple(values))
        for name, values in bcrt.TOKEN_SCHEMA
    )


def _token_columns() -> tuple[str, ...]:
    return tuple(name for name, _ in _token_schema())


def _source_token_columns() -> tuple[str, ...]:
    return tuple(bctp.SOURCE_TOKEN_COLUMNS)


SOURCE_TOKEN_COLUMNS = _source_token_columns()
REQUIRED_STATE_COLUMNS = ("entry_time", *SOURCE_TOKEN_COLUMNS)


def _as_records(states: Any) -> list[dict[str, Any]]:
    if hasattr(states, "to_dict"):
        return [dict(row) for row in states.to_dict("records")]
    return [dict(row) for row in states]


def validate_source_state_schema(states: Any, *, require_known: bool = False) -> list[dict[str, Any]]:
    """Validate and return source-state records.

    The schema is the frozen three-snapshot, 36-token BCTP source sequence.  By
    default this validates shape only; set ``require_known`` to reject token
    values outside the frozen BCRT vocabulary.  Policy inference itself fails
    unknown values closed to flat.
    """
    records = _as_records(states)
    missing: list[tuple[int, str]] = []
    vocab = dict(_token_schema())
    for row_index, row in enumerate(records):
        for column in REQUIRED_STATE_COLUMNS:
            if column not in row:
                missing.append((row_index, column))
        if require_known:
            for column in SOURCE_TOKEN_COLUMNS:
                if column not in row:
                    continue
                token = column.split("__", 1)[1]
                if str(row[column]) not in vocab[token]:
                    raise ValueError(f"unknown token in {column}: {row[column]!r}")
    if missing:
        sample = ", ".join(f"row {i} missing {c}" for i, c in missing[:5])
        raise ValueError(f"invalid BCTP source-state schema: {sample}")
    return records


def _row_has_known_vocabulary(row: Mapping[str, Any]) -> bool:
    vocabulary = dict(_token_schema())
    try:
        return all(
            str(row[column]) in vocabulary[column.split("__", 1)[1]]
            for column in SOURCE_TOKEN_COLUMNS
        )
    except (KeyError, TypeError, ValueError):
        return False


def load_source_state_schema(states: Any, *, require_known: bool = False) -> list[dict[str, Any]]:
    """Alias for schema validation used by runner code."""
    return validate_source_state_schema(states, require_known=require_known)


def load_source_states(path: str, *, require_known: bool = False) -> list[dict[str, Any]]:
    """Load source states from CSV/JSON/JSONL without opening market/funding data."""
    import csv
    import json

    p = Path(path)
    if p.suffix == ".jsonl":
        records = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    elif p.suffix == ".json":
        payload = json.loads(p.read_text())
        records = payload["states"] if isinstance(payload, dict) and "states" in payload else payload
    else:
        opener = gzip.open if p.suffixes[-2:] == [".csv", ".gz"] else Path.open
        with opener(p, mode="rt", newline="") as handle:
            records = list(csv.DictReader(handle))
    return validate_source_state_schema(records, require_known=require_known)


def _entry_sort_key(row: Mapping[str, Any]) -> str:
    value = row.get("entry_time")
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _entry_month_key(row: Mapping[str, Any]) -> str:
    value = row.get("entry_time")
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m")
    text = str(value)
    return text[:7]


def _transform_row(row: Mapping[str, Any], feature_mode: str) -> dict[str, str]:
    if feature_mode not in {"full", "current_only", "reversed", "masked"}:
        raise ValueError(f"unknown feature_mode: {feature_mode!r}")
    out: dict[str, str] = {}
    for column in SOURCE_TOKEN_COLUMNS:
        label, token = column.split("__", 1)
        source_column = column
        if feature_mode == "current_only" and label != "s_0":
            continue
        if feature_mode == "reversed":
            if label == "s_minus_2":
                source_column = f"s_0__{token}"
            elif label == "s_0":
                source_column = f"s_minus_2__{token}"
        value = row[source_column]
        if feature_mode == "masked" and token in MASKED_TOKEN_NAMES:
            value = "MASKED"
        out[column] = str(value)
    return out


@dataclass(frozen=True)
class OneHotEncoder:
    feature_columns: tuple[str, ...]
    vocab: tuple[tuple[str, tuple[str, ...]], ...]
    width: int

    @classmethod
    def fit(cls, records: Sequence[Mapping[str, Any]], feature_mode: str) -> "OneHotEncoder":
        if not records:
            raise ValueError("cannot fit BCTP encoder without records")
        token_vocab = dict(_token_schema())
        entries: list[tuple[str, tuple[str, ...]]] = []
        for column in SOURCE_TOKEN_COLUMNS:
            label, token = column.split("__", 1)
            if feature_mode == "current_only" and label != "s_0":
                continue
            if feature_mode == "masked" and token in MASKED_TOKEN_NAMES:
                values = ("MASKED",)
            else:
                values = token_vocab[token]
            entries.append((column, tuple(values)))
        entries.append(("__position__", POSITIONS))
        vocab = tuple(entries)
        width = sum(len(vals) for _, vals in vocab)
        return cls(tuple(column for column, _ in vocab), vocab, width)

    def encode_one(self, row: Mapping[str, Any], current_position: str, feature_mode: str) -> tuple[np.ndarray, bool]:
        if current_position not in POSITION_TO_INDEX:
            return np.zeros(self.width, dtype=float), False
        transformed = _transform_row(row, feature_mode)
        transformed["__position__"] = current_position
        x = np.zeros(self.width, dtype=float)
        offset = 0
        ok = True
        for column, vals in self.vocab:
            value = transformed.get(column)
            try:
                idx = vals.index(str(value))
            except ValueError:
                ok = False
                idx = None
            if idx is not None:
                x[offset + idx] = 1.0
            offset += len(vals)
        return x, ok

    def encode_many(self, rows: Sequence[Mapping[str, Any]], positions: Sequence[str], feature_mode: str) -> np.ndarray:
        xs = [self.encode_one(row, pos, feature_mode)[0] for row, pos in zip(rows, positions)]
        return np.vstack(xs) if xs else np.zeros((0, self.width), dtype=float)


class _LinearMultiOutput:
    def __init__(self, coef: np.ndarray):
        self.coef = coef

    @classmethod
    def fit_lstsq(cls, x: np.ndarray, y: np.ndarray) -> "_LinearMultiOutput":
        design = np.column_stack([np.ones(len(x)), x])
        coef = np.linalg.lstsq(design, y, rcond=RCOND)[0]
        return cls(coef)

    @classmethod
    def fit_ridge(cls, x: np.ndarray, y: np.ndarray) -> "_LinearMultiOutput":
        design = np.column_stack([np.ones(len(x)), x])
        penalty = np.eye(design.shape[1]) * RIDGE_ALPHA
        penalty[0, 0] = 0.0  # unpenalized intercept
        coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        return cls(coef)

    def predict(self, x: np.ndarray) -> np.ndarray:
        design = np.column_stack([np.ones(len(x)), x])
        return design @ self.coef


@dataclass
class FittedQPolicy:
    algorithm: str
    feature_mode: str
    encoder: OneHotEncoder
    estimator: Any
    action_names: tuple[str, str, str] = MODEL_ACTION_NAMES

    def default_q(self) -> np.ndarray:
        values = np.full(3, -np.inf, dtype=float)
        values[self.action_names.index("TARGET_FLAT")] = 0.0
        return values

    def predict_q(self, row: Mapping[str, Any], current_position: str) -> np.ndarray:
        x, ok = self.encoder.encode_one(row, current_position, self.feature_mode)
        if not ok:
            return self.default_q()
        q = np.asarray(self.estimator.predict(x.reshape(1, -1))[0], dtype=float)
        if q.shape != (3,) or not np.all(np.isfinite(q)):
            return self.default_q()
        return q

    def choose_model_index(self, row: Mapping[str, Any], current_position: str) -> int:
        return _choose_model_index(self.predict_q(row, current_position), current_position)


def _choose_model_index(q: Sequence[float], current_position: str) -> int:
    arr = np.asarray(q, dtype=float)
    finite = np.isfinite(arr)
    if arr.shape != (3,) or not finite.any():
        return 0
    best = np.max(arr[finite])
    candidates = {i for i, value in enumerate(arr) if np.isfinite(value) and value >= best - Q_TIE_TOLERANCE}
    order = [0]
    if current_position in POSITION_TO_TARGET_NAME:
        current_target = POSITION_TO_TARGET_NAME[current_position]
        order.append(MODEL_ACTION_NAMES.index(current_target))
    order.extend([1, 2])
    for index in order:
        if index in candidates:
            return index
    return min(candidates)


def _prepare_rewards(
    states: Sequence[Mapping[str, Any]],
    reward_tensor: Any,
    reward_mode: str,
    action_indices: Sequence[int],
) -> np.ndarray:
    rewards = np.asarray(reward_tensor, dtype=float)
    if rewards.ndim != 3 or rewards.shape[1:] != (3, 3):
        raise ValueError("reward_tensor must have shape [n, 3 positions, 3 actions]")
    if rewards.shape[0] != len(states):
        raise ValueError("reward_tensor length must match states")
    if reward_mode == "base":
        transformed = rewards.copy()
    elif reward_mode == "shuffled_month":
        out = rewards.copy()
        rng = np.random.default_rng(RANDOM_SEED)
        for month in sorted({_entry_month_key(row) for row in states}):
            idx = np.array(
                [
                    i
                    for i, row in enumerate(states)
                    if _entry_month_key(row) == month
                    and np.isfinite(rewards[i]).all()
                ],
                dtype=int,
            )
            perm = idx.copy()
            rng.shuffle(perm)
            out[idx] = rewards[perm]
        transformed = out
    elif reward_mode == "circular_21":
        transformed = rewards.copy()
        complete = np.flatnonzero(
            np.isfinite(rewards).all(axis=(1, 2))
        )
        transformed[complete] = np.roll(
            rewards[complete],
            shift=CIRCULAR_SHIFT,
            axis=0,
        )
    else:
        raise ValueError(f"unknown reward_mode: {reward_mode!r}")
    return transformed[:, :, list(action_indices)]


def _fit_estimator(algorithm: str, x: np.ndarray, y: np.ndarray) -> Any:
    if algorithm == "linear":
        return _LinearMultiOutput.fit_lstsq(x, y)
    if algorithm == "ridge":
        return _LinearMultiOutput.fit_ridge(x, y)
    if algorithm == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        model = ExtraTreesRegressor(**EXTRA_TREES_KWARGS)
        model.fit(x, y)
        return model
    raise ValueError(f"unknown algorithm: {algorithm!r}")


def _fit_fitted_q(
    states: Any,
    reward_tensor: Any,
    terminal: Any,
    reachable_mask: Any,
    *,
    algorithm: str = "linear",
    feature_mode: str = "full",
    reward_mode: str = "base",
    internal_action_names: tuple[str, str, str] = MODEL_ACTION_NAMES,
) -> FittedQPolicy:
    if set(internal_action_names) != set(MODEL_ACTION_NAMES):
        raise ValueError("internal action names must permute the frozen actions")
    input_records = validate_source_state_schema(states, require_known=True)
    raw_rewards = np.asarray(reward_tensor, dtype=float)
    raw_terminal = np.asarray(terminal, dtype=bool)
    raw_reachable = np.asarray(reachable_mask, dtype=bool)
    n_input = len(input_records)
    if raw_rewards.shape != (n_input, 3, 3):
        raise ValueError("reward_tensor must have shape [n, 3 positions, 3 actions]")
    if raw_terminal.shape != (n_input,):
        raise ValueError("terminal must have shape [n]")
    if raw_reachable.shape != (n_input, 3):
        raise ValueError("reachable_mask must have shape [n, 3]")
    order = sorted(range(n_input), key=lambda i: (_entry_sort_key(input_records[i]), i))
    records = [input_records[i] for i in order]
    action_indices = tuple(
        MODEL_ACTION_NAMES.index(name) for name in internal_action_names
    )
    rewards = _prepare_rewards(
        records,
        raw_rewards[order],
        reward_mode,
        action_indices,
    )
    terminal_arr = raw_terminal[order]
    reachable = raw_reachable[order]
    if not np.isfinite(rewards[reachable]).all():
        raise ValueError("reachable BCTP rewards must be finite")
    n = len(records)
    encoder = OneHotEncoder.fit(records, feature_mode)
    train_rows: list[Mapping[str, Any]] = []
    train_positions: list[str] = []
    train_indices: list[tuple[int, int]] = []
    for i, row in enumerate(records):
        for p, pos_name in enumerate(POSITIONS):
            if reachable[i, p]:
                train_rows.append(row)
                train_positions.append(pos_name)
                train_indices.append((i, p))
    if not train_rows:
        raise ValueError("reachable_mask leaves no training rows")
    x = encoder.encode_many(train_rows, train_positions, feature_mode)

    y = np.vstack([rewards[i, p] for i, p in train_indices])
    estimator = _fit_estimator(algorithm, x, y)
    policy = FittedQPolicy(
        algorithm,
        feature_mode,
        encoder,
        estimator,
        internal_action_names,
    )

    for _ in range(1, BELLMAN_ITERATIONS):
        previous = policy
        updated: list[np.ndarray] = []
        for i, p in train_indices:
            target = rewards[i, p].copy()
            if not terminal_arr[i] and i + 1 < n:
                for model_action_index, action_name in enumerate(
                    internal_action_names
                ):
                    next_position = action_name.replace("TARGET", "POSITION")
                    next_q = previous.predict_q(
                        records[i + 1],
                        next_position,
                    )
                    cont = np.max(next_q[np.isfinite(next_q)]) if np.isfinite(next_q).any() else 0.0
                    target[model_action_index] += GAMMA * cont
            updated.append(target)
        y = np.vstack(updated)
        estimator = _fit_estimator(algorithm, x, y)
        policy = FittedQPolicy(
            algorithm,
            feature_mode,
            encoder,
            estimator,
            internal_action_names,
        )
    return policy


def fit_fitted_q(
    states: Any,
    reward_tensor: Any,
    terminal: Any,
    reachable_mask: Any,
    *,
    algorithm: str = "linear",
    feature_mode: str = "full",
    reward_mode: str = "base",
) -> FittedQPolicy:
    """Fit deterministic FQI with reward actions in ``FLAT, SHORT, LONG`` order."""

    return _fit_fitted_q(
        states,
        reward_tensor,
        terminal,
        reachable_mask,
        algorithm=algorithm,
        feature_mode=feature_mode,
        reward_mode=reward_mode,
    )


def rollout_policy(policy: Any, states: Any, start_position: str = "POSITION_FLAT") -> list[str]:
    """Roll out targets in frozen model action order semantics.

    The returned target names are members of ``MODEL_ACTION_NAMES``.  Unknown or
    malformed rows fail closed to ``TARGET_FLAT``.
    """
    records = validate_source_state_schema(states)
    current = start_position
    targets: list[str] = []
    for row in records:
        if not _row_has_known_vocabulary(row):
            targets.append("TARGET_FLAT")
            current = "POSITION_FLAT"
            continue
        try:
            q = np.asarray(policy.predict_q(row, current), dtype=float)
            index = _choose_model_index(q, current)
        except Exception:
            index = 0
        target = MODEL_ACTION_NAMES[index]
        targets.append(target)
        current = target.replace("TARGET", "POSITION")
    return targets


@dataclass
class ConstantPolicy:
    target: str

    def predict_q(self, row: Mapping[str, Any], current_position: str) -> np.ndarray:
        q = np.zeros(3, dtype=float)
        q[:] = -1.0
        q[MODEL_ACTION_NAMES.index(self.target)] = 1.0
        return q


class PersistencePolicy:
    def predict_q(self, row: Mapping[str, Any], current_position: str) -> np.ndarray:
        target = POSITION_TO_TARGET_NAME.get(current_position, "TARGET_FLAT")
        q = np.zeros(3, dtype=float)
        q[:] = -1.0
        q[MODEL_ACTION_NAMES.index(target)] = 1.0
        return q


def constant_policy(target: str | float = "TARGET_FLAT") -> ConstantPolicy:
    if isinstance(target, (float, int)):
        target = TARGET_VALUE_TO_NAME[float(target)]
    if target not in MODEL_ACTION_NAMES:
        raise ValueError(f"unknown target: {target!r}")
    return ConstantPolicy(str(target))


def constant_schedule(states: Any, target: str | float = "TARGET_FLAT") -> list[str]:
    return rollout_policy(constant_policy(target), states)


def persistence_policy() -> PersistencePolicy:
    return PersistencePolicy()


def persistence_schedule(states: Any, start_position: str = "POSITION_FLAT") -> list[str]:
    return rollout_policy(PersistencePolicy(), states, start_position=start_position)


@dataclass
class ExactSignatureMemoryPolicy:
    table: dict[tuple[tuple[str, ...], str], np.ndarray]
    feature_mode: str = "full"

    def predict_q(self, row: Mapping[str, Any], current_position: str) -> np.ndarray:
        try:
            signature = tuple(_transform_row(row, self.feature_mode)[column] for column in SOURCE_TOKEN_COLUMNS)
        except Exception:
            return np.array([0.0, -np.inf, -np.inf])
        value = self.table.get((signature, current_position))
        if value is None or not np.all(np.isfinite(value)):
            return np.array([0.0, -np.inf, -np.inf])
        return value.copy()


def exact_signature_memory_policy(
    states: Any,
    reward_tensor: Any,
    reachable_mask: Any | None = None,
) -> ExactSignatureMemoryPolicy:
    records = validate_source_state_schema(states, require_known=True)
    rewards = _prepare_rewards(records, reward_tensor, "base", (0, 1, 2))
    reachable = np.ones((len(records), 3), dtype=bool) if reachable_mask is None else np.asarray(reachable_mask, dtype=bool)
    sums: dict[tuple[tuple[str, ...], str], np.ndarray] = {}
    counts: dict[tuple[tuple[str, ...], str], int] = {}
    for i, row in enumerate(records):
        signature = tuple(_transform_row(row, "full")[column] for column in SOURCE_TOKEN_COLUMNS)
        for p, pos in enumerate(POSITIONS):
            if reachable[i, p]:
                key = (signature, pos)
                sums[key] = sums.get(key, np.zeros(3, dtype=float)) + rewards[i, p]
                counts[key] = counts.get(key, 0) + 1
    table = {key: value / counts[key] for key, value in sums.items()}
    return ExactSignatureMemoryPolicy(table)


@dataclass
class DirectionFlipPolicy:
    base: Any

    def predict_q(self, row: Mapping[str, Any], current_position: str) -> np.ndarray:
        base_q = np.asarray(self.base.predict_q(row, current_position), dtype=float)
        return base_q[[0, 2, 1]]  # flat unchanged, short<->long


def direction_flip(policy: Any) -> DirectionFlipPolicy:
    return DirectionFlipPolicy(policy)


@dataclass
class ActionCodePermutationPolicy:
    base: Any
    internal_order: tuple[str, str, str] = INTERNAL_PERMUTED_ACTION_NAMES

    def predict_q(self, row: Mapping[str, Any], current_position: str) -> np.ndarray:
        internal = np.asarray(
            self.base.predict_q(row, current_position),
            dtype=float,
        )
        if internal.shape != (3,):
            return np.array([0.0, -np.inf, -np.inf], dtype=float)
        return internal[
            [self.internal_order.index(name) for name in MODEL_ACTION_NAMES]
        ]


@dataclass
class CanonicalToInternalActionView:
    base: Any
    internal_order: tuple[str, str, str] = INTERNAL_PERMUTED_ACTION_NAMES

    def predict_q(self, row: Mapping[str, Any], current_position: str) -> np.ndarray:
        canonical = np.asarray(
            self.base.predict_q(row, current_position),
            dtype=float,
        )
        return canonical[
            [MODEL_ACTION_NAMES.index(name) for name in self.internal_order]
        ]


def action_code_permutation(policy: Any) -> ActionCodePermutationPolicy:
    return ActionCodePermutationPolicy(CanonicalToInternalActionView(policy))


def neutral_action_code_permutation(policy: Any) -> ActionCodePermutationPolicy:
    return action_code_permutation(policy)


def action_code_permutation_policy(
    states: Any,
    reward_tensor: Any,
    terminal: Any,
    reachable_mask: Any,
    *,
    algorithm: str = "linear",
    feature_mode: str = "full",
    reward_mode: str = "base",
) -> ActionCodePermutationPolicy:
    return ActionCodePermutationPolicy(
        _fit_fitted_q(
            states,
            reward_tensor,
            terminal,
            reachable_mask,
            algorithm=algorithm,
            feature_mode=feature_mode,
            reward_mode=reward_mode,
            internal_action_names=INTERNAL_PERMUTED_ACTION_NAMES,
        )
    )


__all__ = [
    "POSITIONS",
    "POSITION_VALUES",
    "TARGET_ACTION_NAMES",
    "TARGET_ACTION_VALUES",
    "MODEL_ACTION_NAMES",
    "MODEL_ACTION_VALUES",
    "SOURCE_TOKEN_COLUMNS",
    "EXTRA_TREES_KWARGS",
    "validate_source_state_schema",
    "load_source_state_schema",
    "load_source_states",
    "fit_fitted_q",
    "rollout_policy",
    "exact_signature_memory_policy",
    "constant_policy",
    "constant_schedule",
    "persistence_policy",
    "persistence_schedule",
    "direction_flip",
    "action_code_permutation",
    "neutral_action_code_permutation",
    "action_code_permutation_policy",
]
