"""Frozen CARTA delayed contextual-bandit primitives.

The state prompt is causal and symbolic.  Future paths are used only by the
offline reward builder after the scheduled exit and never enter policy input.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from training.evaluate_metaorder_fragmentation_impact_curvature import (
    EvaluationConfig,
    simulate_schedule,
)
from training.preregister_causal_adaptive_relational_tokens import (
    TOKEN_COLUMNS,
    Config as StateConfig,
    compute_carta_state,
    nonoverlapping_carta_schedule,
    relational_tokens,
)


PREREGISTRATION_COMMIT = "1f8439b"
PREREGISTRATION_SOURCE = Path(
    "training/preregister_causal_adaptive_relational_tokens.py"
)
PREREGISTRATION_SOURCE_SHA256 = (
    "a3a0be1c8c4401bfb707176d9def951938471805597d51c66f92500bafc4f4af"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/causal-adaptive-relational-token-abstainer-preregistration-2026-07-14.md"
)
PREREGISTRATION_DOCUMENT_SHA256 = (
    "bf803614949a525cdae49475fba79a5ac992f788b877ecd8cae7453948fc18ba"
)
PREREGISTRATION_RESULT = Path(
    "results/causal_adaptive_relational_tokens_support_2026-07-14.json"
)
PREREGISTRATION_RESULT_SHA256 = (
    "77dfd1d0b0ad444744157972aa437f805901bc56428a4e5d76029bf64100d339"
)
EXECUTION_SOURCE = Path(
    "training/evaluate_metaorder_fragmentation_impact_curvature.py"
)
EXECUTION_SOURCE_SHA256 = (
    "1589a52605386570485a7e6be3b8f3aa9439a498abb60eaa42272ac62d4cbed3"
)
SOURCE_LOADER = Path(
    "training/preregister_metaorder_fragmentation_impact_curvature.py"
)
SOURCE_LOADER_SHA256 = (
    "51e99dbdc5ba13e6b4ac15e3915ec5b30e36dff89c1e5b31a5f3f7f272f01a59"
)
AGGTRADE_MANIFEST = Path(
    "data/binance_um_aggtrade_microstructure_btc_2020_2023/build_manifest.json"
)
AGGTRADE_MANIFEST_SHA256 = (
    "6eec40460a6146c58994e52f1af9ace4eecc0c085887d97af5ef17c30b9f7e73"
)
KLINE_MANIFEST = Path(
    "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
)
KLINE_MANIFEST_SHA256 = (
    "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
)

ACTION_NAMES = ("ABSTAIN", "FOLLOW", "FADE")
ACTION_PRIORITY = {action: -index for index, action in enumerate(ACTION_NAMES)}
INTERACTION_PAIRS = (
    ("capital_transition_relation", "crowd_transition_relation"),
    ("capital_transition_relation", "price_transition_relation"),
    ("crowd_transition_relation", "price_transition_relation"),
    ("tension_transition", "transition_alignment_count"),
    ("arrival_transition", "concentration_transition"),
    ("concentration_transition", "trade_span_transition"),
    ("origin_trend_24h_relation", "range_location_24h"),
    ("volatility_rank", "drawdown_rank"),
    ("reference_side_token", "origin_trend_24h_relation"),
    ("origin_structure_bits", "signal_structure_bits"),
)


@dataclass(frozen=True)
class BanditConfig:
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    held_path_drawdown_penalty: float = 1.0 / 3.0
    ridge_alpha: float = 100.0
    minimum_feature_count: int = 3
    prediction_utility_floor: float = 0.0
    naive_bayes_alpha: float = 1.0
    seed: int = 20_260_714


@dataclass(frozen=True)
class RidgePolicy:
    vocabulary: tuple[str, ...]
    coefficients: dict[str, tuple[float, ...]]
    utility_floor: float


@dataclass(frozen=True)
class NaiveBayesPolicy:
    class_counts: dict[str, int]
    feature_counts: dict[str, dict[str, int]]
    field_values: dict[str, tuple[str, ...]]
    alpha: float


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def verify_preregistration() -> dict[str, Any]:
    for path, expected in (
        (PREREGISTRATION_SOURCE, PREREGISTRATION_SOURCE_SHA256),
        (PREREGISTRATION_DOCUMENT, PREREGISTRATION_DOCUMENT_SHA256),
        (PREREGISTRATION_RESULT, PREREGISTRATION_RESULT_SHA256),
        (EXECUTION_SOURCE, EXECUTION_SOURCE_SHA256),
        (SOURCE_LOADER, SOURCE_LOADER_SHA256),
        (AGGTRADE_MANIFEST, AGGTRADE_MANIFEST_SHA256),
        (KLINE_MANIFEST, KLINE_MANIFEST_SHA256),
    ):
        if _sha256(path) != expected:
            raise ValueError(f"frozen CARTA artifact changed: {path}")
    result = json.loads(PREREGISTRATION_RESULT.read_text())
    if result.get("protocol", {}).get("outcomes_opened_for_carta") is not False:
        raise ValueError("CARTA support artifact opened outcomes")
    if result.get("all_support_gates_pass") is not True:
        raise ValueError("CARTA support gates are not frozen as passing")
    if result.get("config") != _jsonable(asdict(StateConfig())):
        raise ValueError("CARTA state config changed after preregistration")
    if result.get("token_schema") != list(TOKEN_COLUMNS):
        raise ValueError("CARTA token schema changed after preregistration")
    calibration = result.get("support_calibration", {})
    if calibration.get("selected_setup_tension_quantile") != 0.975:
        raise ValueError("CARTA support stopping rule changed")
    if calibration.get("further_support_repairs_allowed") is not False:
        raise ValueError("CARTA support artifact allows repair")
    return result


def prompt_from_tokens(tokens: dict[str, str]) -> str:
    if set(tokens) != set(TOKEN_COLUMNS):
        raise ValueError("CARTA prompt tokens differ from frozen schema")
    state = "\n".join(f"{key}={tokens[key]}" for key in TOKEN_COLUMNS)
    return (
        "Choose one CARTA action from ABSTAIN, FOLLOW, FADE.\n"
        "FOLLOW uses the reference capital direction; FADE uses its opposite.\n"
        "Reason only from these causal symbolic relations.\n"
        f"{state}\n"
        'Return exactly one JSON object: {"action":"ABSTAIN|FOLLOW|FADE"}'
    )


def _trade_outcome(
    frame: pd.DataFrame,
    schedule_row: Any,
    *,
    side: int,
    cfg: BanditConfig,
) -> dict[str, float | int]:
    if side not in (-1, 1):
        raise ValueError("trade side must be long or short")
    entry_position = int(schedule_row.entry_position)
    exit_position = int(schedule_row.exit_position)
    if not 0 <= entry_position < exit_position < len(frame):
        raise ValueError("CARTA reward positions are invalid")
    opens = frame["open"].to_numpy(float)
    highs = frame["high"].to_numpy(float)
    lows = frame["low"].to_numpy(float)
    entry_price = float(opens[entry_position])
    exit_price = float(opens[exit_position])
    if entry_price <= 0.0 or exit_price <= 0.0:
        raise ValueError("CARTA reward prices must be positive")
    per_side_cost = (cfg.fee_rate + cfg.slippage_rate) * cfg.leverage
    entry_equity = 1.0 - per_side_cost
    held_high = float(np.max(highs[entry_position:exit_position]))
    held_low = float(np.min(lows[entry_position:exit_position]))
    favorable_price = held_high if side > 0 else held_low
    adverse_price = held_low if side > 0 else held_high
    favorable_equity = max(
        0.0,
        entry_equity
        * (1.0 + cfg.leverage * side * (favorable_price / entry_price - 1.0)),
    )
    adverse_equity = max(
        0.0,
        entry_equity
        * (1.0 + cfg.leverage * side * (adverse_price / entry_price - 1.0)),
    )
    raw_return = side * (exit_price / entry_price - 1.0)
    multiplier = (
        (1.0 - per_side_cost)
        * max(0.0, 1.0 + cfg.leverage * raw_return)
        * (1.0 - per_side_cost)
    )
    intratrade_peak = max(1.0, favorable_equity)
    held_path_drawdown = max(
        per_side_cost,
        1.0 - adverse_equity / intratrade_peak,
        1.0 - multiplier / intratrade_peak,
    )
    utility = (
        math.log(max(multiplier, 1e-12))
        - cfg.held_path_drawdown_penalty * held_path_drawdown
    )
    return {
        "side": int(side),
        "underlying_raw_return": float(raw_return),
        "account_multiplier": float(multiplier),
        "account_net_return": float(multiplier - 1.0),
        "held_path_drawdown": float(held_path_drawdown),
        "utility": float(utility),
    }


def action_outcomes(
    frame: pd.DataFrame,
    schedule_row: Any,
    cfg: BanditConfig,
) -> dict[str, dict[str, float | int]]:
    reference_side = int(schedule_row.side)
    if reference_side not in (-1, 1):
        raise ValueError("CARTA reference side must be long or short")
    abstain: dict[str, float | int] = {
        "side": 0,
        "underlying_raw_return": 0.0,
        "account_multiplier": 1.0,
        "account_net_return": 0.0,
        "held_path_drawdown": 0.0,
        "utility": 0.0,
    }
    return {
        "ABSTAIN": abstain,
        "FOLLOW": _trade_outcome(
            frame, schedule_row, side=reference_side, cfg=cfg
        ),
        "FADE": _trade_outcome(
            frame, schedule_row, side=-reference_side, cfg=cfg
        ),
    }


def _best_action(outcomes: dict[str, dict[str, float | int]]) -> str:
    return max(
        ACTION_NAMES,
        key=lambda action: (
            float(outcomes[action]["utility"]),
            ACTION_PRIORITY[action],
        ),
    )


def build_bandit_rows(
    frame: pd.DataFrame,
    state: pd.DataFrame,
    schedule: pd.DataFrame,
    cfg: BanditConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for schedule_row in schedule.itertuples(index=False):
        signal_position = int(schedule_row.signal_position)
        tokens = relational_tokens(state.loc[signal_position])
        outcomes = action_outcomes(frame, schedule_row, cfg)
        rows.append(
            {
                "date": str(frame.loc[signal_position, "date"]),
                "signal_position": signal_position,
                "entry_position": int(schedule_row.entry_position),
                "exit_position": int(schedule_row.exit_position),
                "reference_side": int(schedule_row.side),
                "tokens": tokens,
                "prompt": prompt_from_tokens(tokens),
                "action_outcomes": outcomes,
                "oracle_best_action": _best_action(outcomes),
            }
        )
    return rows


def symbolic_features(tokens: dict[str, str]) -> list[str]:
    if set(tokens) != set(TOKEN_COLUMNS):
        raise ValueError("CARTA feature tokens differ from frozen schema")
    features = [f"main:{key}={tokens[key]}" for key in TOKEN_COLUMNS]
    features.extend(
        f"pair:{left}={tokens[left]}|{right}={tokens[right]}"
        for left, right in INTERACTION_PAIRS
    )
    return features


def _fit_vocabulary(
    rows: list[dict[str, Any]],
    *,
    minimum_count: int,
) -> tuple[str, ...]:
    counts: Counter[str] = Counter(
        feature
        for row in rows
        for feature in symbolic_features(row["tokens"])
    )
    return tuple(
        ["__BIAS__"]
        + sorted(
            feature for feature, count in counts.items() if count >= minimum_count
        )
    )


def _matrix(
    rows: list[dict[str, Any]],
    vocabulary: tuple[str, ...],
) -> np.ndarray:
    index = {feature: position for position, feature in enumerate(vocabulary)}
    matrix = np.zeros((len(rows), len(vocabulary)), dtype=np.float64)
    if "__BIAS__" in index:
        matrix[:, index["__BIAS__"]] = 1.0
    for row_position, row in enumerate(rows):
        for feature in symbolic_features(row["tokens"]):
            column = index.get(feature)
            if column is not None:
                matrix[row_position, column] = 1.0
    return matrix


def fit_ridge_policy(
    rows: list[dict[str, Any]],
    cfg: BanditConfig,
    *,
    shuffle_targets: bool = False,
) -> RidgePolicy:
    if not rows:
        raise ValueError("cannot fit CARTA ridge without rows")
    vocabulary = _fit_vocabulary(
        rows, minimum_count=cfg.minimum_feature_count
    )
    matrix = _matrix(rows, vocabulary)
    penalty = np.eye(len(vocabulary), dtype=np.float64) * cfg.ridge_alpha
    penalty[0, 0] = 0.0
    normal = matrix.T @ matrix + penalty
    rng = np.random.default_rng(cfg.seed)
    coefficients: dict[str, tuple[float, ...]] = {}
    for action in ("FOLLOW", "FADE"):
        target = np.asarray(
            [float(row["action_outcomes"][action]["utility"]) for row in rows],
            dtype=np.float64,
        )
        if shuffle_targets:
            target = target[rng.permutation(len(target))]
        fitted = np.linalg.solve(normal, matrix.T @ target)
        coefficients[action] = tuple(float(value) for value in fitted)
    return RidgePolicy(
        vocabulary=vocabulary,
        coefficients=coefficients,
        utility_floor=cfg.prediction_utility_floor,
    )


def predict_ridge(
    policy: RidgePolicy,
    rows: list[dict[str, Any]],
) -> list[str]:
    matrix = _matrix(rows, policy.vocabulary)
    values = {
        action: matrix @ np.asarray(policy.coefficients[action], dtype=np.float64)
        for action in ("FOLLOW", "FADE")
    }
    predictions: list[str] = []
    for index in range(len(rows)):
        action = max(
            ("FOLLOW", "FADE"),
            key=lambda name: (float(values[name][index]), ACTION_PRIORITY[name]),
        )
        predictions.append(
            action
            if float(values[action][index]) > policy.utility_floor
            else "ABSTAIN"
        )
    return predictions


def fit_naive_bayes(
    rows: list[dict[str, Any]],
    cfg: BanditConfig,
) -> NaiveBayesPolicy:
    class_counts: Counter[str] = Counter()
    feature_counts: dict[str, Counter[str]] = {
        action: Counter() for action in ACTION_NAMES
    }
    field_values: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        action = str(row["oracle_best_action"])
        class_counts[action] += 1
        for key in TOKEN_COLUMNS:
            value = str(row["tokens"][key])
            feature_counts[action][f"{key}={value}"] += 1
            field_values[key].add(value)
    return NaiveBayesPolicy(
        class_counts={action: int(class_counts[action]) for action in ACTION_NAMES},
        feature_counts={
            action: dict(feature_counts[action]) for action in ACTION_NAMES
        },
        field_values={
            key: tuple(sorted(values)) for key, values in field_values.items()
        },
        alpha=cfg.naive_bayes_alpha,
    )


def predict_naive_bayes(
    policy: NaiveBayesPolicy,
    rows: list[dict[str, Any]],
) -> list[str]:
    total = sum(policy.class_counts.values())
    predictions: list[str] = []
    for row in rows:
        scores: dict[str, float] = {}
        for action in ACTION_NAMES:
            class_count = policy.class_counts[action]
            score = math.log(
                (class_count + policy.alpha)
                / (total + policy.alpha * len(ACTION_NAMES))
            )
            for key in TOKEN_COLUMNS:
                value = str(row["tokens"][key])
                count = policy.feature_counts[action].get(f"{key}={value}", 0)
                categories = max(1, len(policy.field_values.get(key, ())))
                score += math.log(
                    (count + policy.alpha)
                    / (class_count + policy.alpha * categories)
                )
            scores[action] = score
        predictions.append(
            max(
                ACTION_NAMES,
                key=lambda action: (scores[action], ACTION_PRIORITY[action]),
            )
        )
    return predictions


def fit_signature_memory(rows: list[dict[str, Any]]) -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        signature = json.dumps(row["tokens"], sort_keys=True, separators=(",", ":"))
        grouped[signature].append(str(row["oracle_best_action"]))
    memory: dict[str, str] = {}
    for signature, actions in grouped.items():
        counts = Counter(actions)
        memory[signature] = max(
            ACTION_NAMES,
            key=lambda action: (counts[action], ACTION_PRIORITY[action]),
        )
    return memory


def predict_signature_memory(
    memory: dict[str, str],
    rows: list[dict[str, Any]],
) -> list[str]:
    return [
        memory.get(
            json.dumps(row["tokens"], sort_keys=True, separators=(",", ":")),
            "ABSTAIN",
        )
        for row in rows
    ]


def constant_predictions(action: str) -> Callable[[list[dict[str, Any]]], list[str]]:
    if action not in ACTION_NAMES:
        raise ValueError("unknown CARTA constant action")
    return lambda rows: [action] * len(rows)


def prediction_schedule(
    candidate_schedule: pd.DataFrame,
    predictions: list[str],
) -> pd.DataFrame:
    if len(candidate_schedule) != len(predictions):
        raise ValueError("CARTA predictions and candidates have different lengths")
    rows: list[dict[str, Any]] = []
    for row, action in zip(candidate_schedule.itertuples(index=False), predictions):
        if action == "ABSTAIN":
            continue
        if action not in {"FOLLOW", "FADE"}:
            raise ValueError("unknown CARTA prediction")
        item = row._asdict()
        reference = int(row.side)
        item["side"] = reference if action == "FOLLOW" else -reference
        item["branch"] = "continuation" if action == "FOLLOW" else "fade"
        rows.append(item)
    return pd.DataFrame(rows, columns=candidate_schedule.columns)


def evaluate_predictions(
    frame: pd.DataFrame,
    candidate_schedule: pd.DataFrame,
    predictions: list[str],
    *,
    start: str,
    end: str,
    cfg: BanditConfig,
) -> dict[str, Any]:
    schedule = prediction_schedule(candidate_schedule, predictions)
    execution = EvaluationConfig(
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
    )
    metrics = simulate_schedule(
        frame,
        schedule,
        start=start,
        end=end,
        cfg=execution,
    )
    action_counts = Counter(predictions)
    metrics["action_counts"] = {
        action: int(action_counts[action]) for action in ACTION_NAMES
    }
    metrics["follow_count"] = int(action_counts["FOLLOW"])
    if metrics["continuation_count"] != metrics["follow_count"]:
        raise AssertionError("CARTA FOLLOW branch accounting changed")
    if metrics["fade_count"] != int(action_counts["FADE"]):
        raise AssertionError("CARTA FADE branch accounting changed")
    metrics["candidate_count"] = int(len(candidate_schedule))
    return metrics


def build_window(
    frame: pd.DataFrame,
    state: pd.DataFrame,
    *,
    start: str,
    end: str,
    cfg: BanditConfig,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    schedule = nonoverlapping_carta_schedule(
        state, frame, start=start, end=end
    )
    return schedule, build_bandit_rows(frame, state, schedule, cfg)


def load_state_frame() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    verify_preregistration()
    from training.preregister_metaorder_fragmentation_impact_curvature import (
        Config as SourceConfig,
        load_causal_frame,
    )

    frame, source = load_causal_frame(SourceConfig())
    state = compute_carta_state(frame, StateConfig(), include_tokens=True)
    return frame, state, source
