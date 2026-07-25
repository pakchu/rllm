"""Freeze BCTP-12H economics and policy families before outcome access."""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import numpy as np


POLICY_ID = "BCTP-12H"
PROTOCOL_VERSION = "block_clearing_target_position_mdp_evaluator_freeze_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

IMPLEMENTATION = Path(
    "training/freeze_block_clearing_target_position_evaluator.py"
)
TESTS = Path(
    "tests/test_freeze_block_clearing_target_position_evaluator.py"
)
CONTRACT = Path(
    "docs/bctp-economic-cheap-policy-freeze-2026-07-25.md"
)
BOUNDARY = Path(
    "docs/block-clearing-target-position-mdp-boundary-2026-07-25.md"
)
PREREGISTRATION_SOURCE = Path(
    "training/preregister_block_clearing_target_position_mdp.py"
)
PREREGISTRATION = Path(
    "results/block_clearing_target_position_mdp_"
    "preregistration_2026-07-25.json"
)
SUPPORT_SOURCE = Path(
    "training/build_block_clearing_target_position_mdp_support.py"
)
SUPPORT = Path(
    "results/block_clearing_target_position_mdp_support_2026-07-25.json"
)
SEQUENCES = Path(
    "data/block_clearing_target_position_mdp_sequences_2020_2023.csv.gz"
)
BCRT_CLOCK = Path(
    "data/block_clearing_relational_topology_clocks_2020_2023.csv.gz"
)
DEFAULT_OUTPUT = Path(
    "results/block_clearing_target_position_mdp_"
    "evaluator_freeze_2026-07-25.json"
)

MARKET = Path(
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)
MARKET_SHA256 = (
    "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
)
MARKET_MANIFEST = Path(
    "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
)
MARKET_MANIFEST_SHA256 = (
    "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
)
FUNDING = Path(
    "data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz"
)
FUNDING_SHA256 = (
    "3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6"
)
FUNDING_MANIFEST = Path(
    "results/binance_um_btcusdt_funding_marks_2020_2023_"
    "manifest_2026-07-17.json"
)
FUNDING_MANIFEST_SHA256 = (
    "a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b"
)

EXPECTED_STATIC_SHA256 = {
    str(BOUNDARY): (
        "97f92a4b9e78fcdc50cb227f1a91c778e7dacec48799cda372c636cb5f58e16e"
    ),
    str(PREREGISTRATION_SOURCE): (
        "2bea5f3bfb5d0fd1985bf74ff7fe4cf7d43de7427378223af8c6f5cb65f2199c"
    ),
    str(PREREGISTRATION): (
        "dfdc18c61f578425ee4459ef30bdede97032c364af55f846365e0687694fdbc8"
    ),
    str(SUPPORT_SOURCE): (
        "1c3751134fd0eb851e1375dc32d02847ce9fa9210b8929fc28539a1d19023d9c"
    ),
    str(SUPPORT): (
        "00166aac18bd59e2b8c56ac285072fe5151c77159eca2e8c8e446ae0ed134ef8"
    ),
    str(SEQUENCES): (
        "00fd5a0fb5c238ca27109e49d6b3c7f11d16d6edbd37788f76a1bcaeeb86dd56"
    ),
    str(BCRT_CLOCK): (
        "c0420c7175410a822455a0d68bf877cba94a2ec17b31f6d9a588244cb893c909"
    ),
    str(MARKET_MANIFEST): MARKET_MANIFEST_SHA256,
    str(FUNDING_MANIFEST): FUNDING_MANIFEST_SHA256,
}

ACTIONS = (-0.5, 0.0, 0.5)
ACTION_NAMES = ("TARGET_SHORT", "TARGET_FLAT", "TARGET_LONG")
MODEL_ACTIONS = (0.0, -0.5, 0.5)
MODEL_ACTION_NAMES = ("TARGET_FLAT", "TARGET_SHORT", "TARGET_LONG")
POSITIONS = ("POSITION_SHORT", "POSITION_FLAT", "POSITION_LONG")
FAMILY_IDS = (
    "always_flat",
    "always_long",
    "always_short",
    "previous_target_persistence",
    "exact_signature_memory",
    "categorical_linear_fqi",
    "categorical_ridge_fqi",
    "extra_trees_fqi",
    "categorical_linear_fqi_current_only",
    "categorical_linear_fqi_reversed_sequence",
    "categorical_linear_fqi_masked_source",
    "categorical_ridge_fqi_current_only",
    "categorical_ridge_fqi_reversed_sequence",
    "categorical_ridge_fqi_masked_source",
    "extra_trees_fqi_current_only",
    "extra_trees_fqi_reversed_sequence",
    "extra_trees_fqi_masked_source",
    "categorical_linear_fqi_shuffled_reward",
    "categorical_linear_fqi_circular_21_reward",
    "categorical_ridge_fqi_shuffled_reward",
    "categorical_ridge_fqi_circular_21_reward",
    "extra_trees_fqi_shuffled_reward",
    "extra_trees_fqi_circular_21_reward",
    "categorical_linear_fqi_direction_flip",
    "categorical_linear_fqi_action_code_permutation",
    "categorical_ridge_fqi_direction_flip",
    "categorical_ridge_fqi_action_code_permutation",
    "extra_trees_fqi_direction_flip",
    "extra_trees_fqi_action_code_permutation",
    "bcrt_exact_six_hour_always_long",
    "bcrt_exact_six_hour_always_short",
)


@dataclass(frozen=True)
class AccountingConfig:
    target_absolute_gross: float = 0.5
    base_cost_rate: float = 0.0006
    stress_cost_rate: float = 0.0010
    bar_seconds: int = 300
    terminal_flat_bars_before_end: int = 1
    cagr_year_days: float = 365.2425
    equation_relative_tolerance: float = 1e-12
    funding_boundary_rule: str = "min(0,old_cash,new_cash)"
    held_bar_order: str = "favorable_then_adverse"


@dataclass(frozen=True)
class RewardConfig:
    minimum_multiplier: float = 1e-12
    downside_linear_coefficient: float = 1.0 / 3.0
    target_change_coefficient: float = 0.0010


@dataclass(frozen=True)
class CheapPolicyConfig:
    discount: float = 0.99
    fitted_q_iterations: int = 25
    random_seed: int = 20_260_725
    linear_lstsq_rcond: float = 1e-12
    extra_trees_estimators: int = 512
    extra_trees_max_depth: int = 6
    extra_trees_min_samples_split: int = 24
    extra_trees_min_samples_leaf: int = 12
    extra_trees_max_features: str = "sqrt"
    extra_trees_bootstrap: bool = False
    extra_trees_criterion: str = "squared_error"
    extra_trees_jobs: int = 1
    ridge_alpha: float = 100.0
    ridge_intercept_penalized: bool = False
    circular_reward_shift_decisions: int = 21
    masked_token_names: tuple[str, ...] = (
        "order_transition",
        "leader_transition",
    )
    q_tie_tolerance: float = 1e-12


@dataclass(frozen=True)
class StatisticalConfig:
    exact_cluster_max: int = 20
    draws: int = 100_000
    seed: int = 20_260_725
    batch_draws: int = 2_000
    cluster: str = "monday_00_utc_half_open_week"
    alternative: str = "one_sided_positive_studentized_mean"


@dataclass(frozen=True)
class GemmaConfig:
    repository: str = "google/gemma-4-E4B"
    revision: str = "9f9f0f28c85251b6616672841d041635e1763f13"
    tokenizer_sha256: str = (
        "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
    )
    vocabulary_size: int = 262_144
    action_tokens: tuple[str, ...] = ("A", "B", "C")
    action_token_ids: tuple[int, ...] = (236_776, 236_799, 236_780)
    action_permutations: int = 6
    quantization: str = "nf4_double_quant_bfloat16_compute"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    maximum_sequence_tokens: int = 768
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    warmup_fraction: float = 0.10
    gradient_clip: float = 1.0
    seeds: tuple[int, ...] = (20_260_725, 20_260_726)
    checkpoint_steps: tuple[int, ...] = (80, 160, 240)
    actor_kl_uniform_coefficient: float = 0.01
    q_centered_clip: float = 0.10


def _path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def json_normalized(payload: Any) -> Any:
    return json.loads(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"BCTP expected JSON object: {path}")
    return payload


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _assert_protocol_committed() -> str:
    paths = tuple(str(path) for path in (IMPLEMENTATION, TESTS, CONTRACT))
    tracked = _git("ls-files", "--error-unmatch", "--", *paths)
    if tracked.returncode:
        raise RuntimeError("BCTP evaluator protocol is not committed")
    clean = _git("diff", "--quiet", "HEAD", "--", *paths)
    if clean.returncode:
        raise RuntimeError("BCTP evaluator protocol differs from HEAD")
    head = _git("rev-parse", "HEAD")
    if head.returncode or len(head.stdout.strip()) != 40:
        raise RuntimeError("BCTP cannot identify evaluator freeze commit")
    return head.stdout.strip()


def solve_target_quantity(
    pre_equity: float,
    old_quantity: float,
    price: float,
    cost_rate: float,
    target: float,
    *,
    cfg: AccountingConfig = AccountingConfig(),
) -> float:
    """Solve the frozen post-cost target-gross equation analytically.

    Returning zero is the fail-closed action for an inadmissible input or
    branch.  The two nonzero candidates are independently checked against the
    original equation rather than trusted from their branch formula.
    """

    values = (pre_equity, old_quantity, price, cost_rate, target)
    if not all(math.isfinite(value) for value in values):
        return 0.0
    if (
        pre_equity <= 0.0
        or price <= 0.0
        or not 0.0 <= cost_rate < 0.1
        or target not in ACTIONS
    ):
        return 0.0
    if target == 0.0:
        return 0.0

    old_notional = old_quantity * price
    candidates: list[float] = []
    branches = (
        (
            target * (pre_equity + cost_rate * old_notional)
            / (1.0 + target * cost_rate),
            "ge",
        ),
        (
            target * (pre_equity - cost_rate * old_notional)
            / (1.0 - target * cost_rate),
            "le",
        ),
    )
    for new_notional, branch in branches:
        if not math.isfinite(new_notional):
            continue
        tolerance = cfg.equation_relative_tolerance * max(
            1.0, abs(new_notional), abs(old_notional)
        )
        if branch == "ge" and new_notional < old_notional - tolerance:
            continue
        if branch == "le" and new_notional > old_notional + tolerance:
            continue
        post_equity = (
            pre_equity
            - cost_rate * abs(new_notional - old_notional)
        )
        if post_equity <= 0.0 or new_notional * target <= 0.0:
            continue
        ratio = new_notional / post_equity
        if not math.isclose(
            ratio,
            target,
            rel_tol=cfg.equation_relative_tolerance,
            abs_tol=cfg.equation_relative_tolerance,
        ):
            continue
        quantity = new_notional / price
        if not any(
            math.isclose(
                quantity,
                existing,
                rel_tol=cfg.equation_relative_tolerance,
                abs_tol=cfg.equation_relative_tolerance,
            )
            for existing in candidates
        ):
            candidates.append(quantity)
    if len(candidates) != 1:
        return 0.0
    return float(candidates[0])


def solve_target_quantity_bisection(
    pre_equity: float,
    old_quantity: float,
    price: float,
    cost_rate: float,
    target: float,
    *,
    iterations: int = 256,
) -> float:
    """Independent signed-notional bisection used only as a test oracle."""

    values = (pre_equity, old_quantity, price, cost_rate, target)
    if not all(math.isfinite(value) for value in values):
        return 0.0
    if (
        pre_equity <= 0.0
        or price <= 0.0
        or not 0.0 <= cost_rate < 0.1
        or target not in ACTIONS
        or iterations < 1
    ):
        return 0.0
    if target == 0.0:
        return 0.0

    sign = math.copysign(1.0, target)
    old_notional = old_quantity * price

    def residual(magnitude: float) -> float:
        new_notional = sign * magnitude
        post = pre_equity - cost_rate * abs(
            new_notional - old_notional
        )
        if post <= 0.0:
            return math.inf
        return magnitude / post - abs(target)

    lower = 0.0
    upper = max(pre_equity * 2.0, abs(old_notional) * 2.0, 1.0)
    while residual(upper) < 0.0 and upper < pre_equity * 1e12:
        upper *= 2.0
    if not math.isfinite(residual(upper)) and residual(lower) > 0.0:
        return 0.0
    for _ in range(iterations):
        midpoint = (lower + upper) / 2.0
        if residual(midpoint) >= 0.0:
            upper = midpoint
        else:
            lower = midpoint
    candidate = sign * ((lower + upper) / 2.0)
    post = pre_equity - cost_rate * abs(candidate - old_notional)
    if post <= 0.0 or not math.isclose(
        candidate / post,
        target,
        rel_tol=1e-10,
        abs_tol=1e-12,
    ):
        return 0.0
    return float(candidate / price)


def conservative_boundary_funding_cash(
    old_quantity: float,
    new_quantity: float,
    settlement_mark: float,
    funding_rate: float,
) -> float:
    values = (
        old_quantity,
        new_quantity,
        settlement_mark,
        funding_rate,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("BCTP non-finite funding boundary input")
    if settlement_mark <= 0.0:
        raise ValueError("BCTP funding settlement mark must be positive")
    old_cash = -old_quantity * settlement_mark * funding_rate
    new_cash = -new_quantity * settlement_mark * funding_rate
    return float(min(0.0, old_cash, new_cash))


def transition_utility(
    multiplier: float,
    held_path_downside_fraction: float,
    old_target: float,
    new_target: float,
    *,
    cfg: RewardConfig = RewardConfig(),
) -> float:
    values = (
        multiplier,
        held_path_downside_fraction,
        old_target,
        new_target,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("BCTP utility input is non-finite")
    if (
        multiplier <= 0.0
        or held_path_downside_fraction < 0.0
        or old_target not in ACTIONS
        or new_target not in ACTIONS
    ):
        raise ValueError("BCTP utility input is outside its domain")
    return float(
        math.log(max(multiplier, cfg.minimum_multiplier))
        - cfg.downside_linear_coefficient
        * held_path_downside_fraction
        - cfg.target_change_coefficient * abs(new_target - old_target)
    )


def _studentized_mean(values: np.ndarray) -> float:
    sample = np.asarray(values, dtype=np.float64)
    if (
        sample.ndim != 1
        or len(sample) < 2
        or not np.isfinite(sample).all()
    ):
        raise ValueError("BCTP weekly statistic input is invalid")
    deviation = float(sample.std(ddof=1))
    if deviation <= 0.0:
        return float("-inf")
    return float(math.sqrt(len(sample)) * sample.mean() / deviation)


def _valid_week_key(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return (
        value.endswith("Z")
        and parsed.utcoffset() is not None
        and parsed.utcoffset().total_seconds() == 0.0
        and parsed.tzinfo is not None
        and parsed.weekday() == 0
        and parsed.hour == 0
        and parsed.minute == 0
        and parsed.second == 0
        and parsed.microsecond == 0
    )


def shared_weekly_max_stat(
    weekly_returns: Mapping[str, Sequence[tuple[str, float]]],
    *,
    cfg: StatisticalConfig = StatisticalConfig(),
) -> dict[str, Any]:
    """One-sided local and shared-family max-stat Rademacher inference."""

    if tuple(weekly_returns) != FAMILY_IDS:
        raise ValueError("BCTP weekly family order changed")
    week_keys: tuple[str, ...] | None = None
    rows: list[np.ndarray] = []
    for variant in FAMILY_IDS:
        observations = tuple(weekly_returns[variant])
        keys = tuple(str(key) for key, _ in observations)
        if (
            len(keys) < 2
            or len(set(keys)) != len(keys)
            or keys != tuple(sorted(keys))
            or not all(_valid_week_key(key) for key in keys)
        ):
            raise ValueError("BCTP weekly keys are invalid")
        if week_keys is None:
            week_keys = keys
        elif keys != week_keys:
            raise ValueError("BCTP weekly keys are misaligned")
        rows.append(
            np.asarray(
                [float(value) for _, value in observations],
                dtype=np.float64,
            )
        )
    if week_keys is None:
        raise ValueError("BCTP weekly family is empty")
    matrix = np.vstack(
        rows
    )
    if (
        matrix.ndim != 2
        or matrix.shape[1] < 2
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("BCTP weekly family matrix is invalid")
    observed = np.asarray(
        [_studentized_mean(row) for row in matrix],
        dtype=np.float64,
    )
    local_exceed = np.zeros(len(FAMILY_IDS), dtype=np.int64)
    max_exceed = np.zeros(len(FAMILY_IDS), dtype=np.int64)

    def consume(signs: np.ndarray) -> None:
        signed = signs[:, None, :] * matrix[None, :, :]
        means = signed.mean(axis=2)
        deviations = signed.std(axis=2, ddof=1)
        null_t = np.full_like(means, float("-inf"))
        np.divide(
            math.sqrt(matrix.shape[1]) * means,
            deviations,
            out=null_t,
            where=deviations > 0.0,
        )
        family_max = null_t.max(axis=1)
        for index, statistic in enumerate(observed):
            if not math.isfinite(statistic):
                continue
            local_exceed[index] += int(
                np.count_nonzero(null_t[:, index] >= statistic - 1e-15)
            )
            max_exceed[index] += int(
                np.count_nonzero(family_max >= statistic - 1e-15)
            )

    weeks = matrix.shape[1]
    if weeks <= cfg.exact_cluster_max:
        draws = 1 << weeks
        bit_positions = np.arange(weeks, dtype=np.uint64)
        for begin in range(0, draws, cfg.batch_draws):
            indices = np.arange(
                begin,
                min(draws, begin + cfg.batch_draws),
                dtype=np.uint64,
            )
            bits = (indices[:, None] >> bit_positions[None, :]) & 1
            consume(1.0 - 2.0 * bits.astype(np.float64))
        method = "exact_rademacher_enumeration"
        correction = 0
    else:
        draws = cfg.draws
        generator = np.random.default_rng(cfg.seed)
        completed = 0
        while completed < draws:
            current = min(cfg.batch_draws, draws - completed)
            consume(
                generator.choice(
                    (-1.0, 1.0),
                    size=(current, weeks),
                )
            )
            completed += current
        method = "monte_carlo_rademacher"
        correction = 1

    local: dict[str, float] = {}
    adjusted: dict[str, float] = {}
    for index, variant in enumerate(FAMILY_IDS):
        if not math.isfinite(observed[index]):
            local[variant] = 1.0
            adjusted[variant] = 1.0
            continue
        denominator = draws + correction
        local[variant] = float(
            (local_exceed[index] + correction) / denominator
        )
        adjusted[variant] = float(
            (max_exceed[index] + correction) / denominator
        )
        if adjusted[variant] + 1e-15 < local[variant]:
            raise RuntimeError("BCTP max-stat p is below local p")
    return {
        "method": method,
        "draws": int(draws),
        "seed": int(cfg.seed),
        "weeks": int(weeks),
        "week_keys": list(week_keys),
        "shared_signs": True,
        "family_ids": list(FAMILY_IDS),
        "observed_t": {
            variant: (
                float(observed[index])
                if math.isfinite(observed[index])
                else None
            )
            for index, variant in enumerate(FAMILY_IDS)
        },
        "local_p": local,
        "p_max": adjusted,
    }


def _verify_bound_inputs() -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for path, expected in EXPECTED_STATIC_SHA256.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"BCTP frozen input changed: {path}")
        audit[path] = {"path": path, "sha256": actual}

    preregistration = _load_json(PREREGISTRATION)
    if (
        preregistration.get("protocol_version")
        != "block_clearing_target_position_mdp_preregistration_v1"
        or preregistration.get("policy", {}).get("policy_id") != POLICY_ID
    ):
        raise RuntimeError("BCTP preregistration identity changed")
    support = _load_json(SUPPORT)
    if (
        support.get("protocol_version")
        != "block_clearing_target_position_mdp_support_v1"
        or support.get("policy_id") != POLICY_ID
        or support.get("decision")
        != "advance_to_frozen_economic_and_cheap_policy_evaluator"
        or support.get("authorized_next_stage")
        != "freeze_economic_evaluator_and_cheap_policy_family"
        or support.get("source_support_passed") is not True
        or support.get("artifact_eligible") is not True
    ):
        raise RuntimeError("BCTP source support does not authorize freeze")
    if support.get("market_loaded") is not False:
        raise RuntimeError("BCTP source support already opened market")
    if support.get("funding_loaded") is not False:
        raise RuntimeError("BCTP source support already opened funding")

    market_manifest = _load_json(MARKET_MANIFEST)
    if (
        market_manifest.get("combined_output") != str(MARKET)
        or market_manifest.get("combined_sha256") != MARKET_SHA256
    ):
        raise RuntimeError("BCTP market manifest binding changed")
    funding_manifest = _load_json(FUNDING_MANIFEST)
    funding_data = funding_manifest.get("data", {})
    if (
        funding_manifest.get("outcomes_opened") is not False
        or funding_data.get("path") != str(FUNDING)
        or funding_data.get("sha256") != FUNDING_SHA256
    ):
        raise RuntimeError("BCTP funding manifest binding changed")
    return audit


def _manifest_core(*, freeze_commit: str) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "as_of_date": "2026-07-25",
        "freeze_commit": freeze_commit,
        "implementation": {
            "source": str(IMPLEMENTATION),
            "source_sha256": sha256_file(IMPLEMENTATION),
            "tests": str(TESTS),
            "tests_sha256": sha256_file(TESTS),
            "contract": str(CONTRACT),
            "contract_sha256": sha256_file(CONTRACT),
        },
        "bound_inputs": _verify_bound_inputs(),
        "execution_sources": {
            "market": {
                "path": str(MARKET),
                "expected_sha256": MARKET_SHA256,
                "manifest": str(MARKET_MANIFEST),
                "manifest_sha256": MARKET_MANIFEST_SHA256,
                "payload_bytes_hashed_during_freeze": False,
            },
            "funding": {
                "path": str(FUNDING),
                "expected_sha256": FUNDING_SHA256,
                "manifest": str(FUNDING_MANIFEST),
                "manifest_sha256": FUNDING_MANIFEST_SHA256,
                "payload_bytes_hashed_during_freeze": False,
            },
        },
        "temporal_roles": {
            "fit": ["2020-01-01T00:00:00Z", "2021-01-01T00:00:00Z"],
            "transfer": [
                "2021-01-01T00:00:00Z",
                "2022-01-01T00:00:00Z",
            ],
            "refit": ["2020-01-01T00:00:00Z", "2022-01-01T00:00:00Z"],
            "gemma_selection": [
                "2022-01-01T00:00:00Z",
                "2023-01-01T00:00:00Z",
            ],
            "immutable_eval": [
                "2023-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
            ],
            "sealed_from": "2024-01-01T00:00:00Z",
        },
        "accounting": asdict(AccountingConfig()),
        "reward": {
            **asdict(RewardConfig()),
            "formula": (
                "log(max(E_end/E_pre,1e-12))"
                "-(1/3)*held_path_downside_fraction"
                "-0.001*abs(target_new-target_old)"
            ),
            "future_best_label_used": False,
        },
        "cheap_policy": asdict(CheapPolicyConfig()),
        "statistical_test": asdict(StatisticalConfig()),
        "family_ids": list(FAMILY_IDS),
        "family_size": len(FAMILY_IDS),
        "action_values": list(ACTIONS),
        "action_names": list(ACTION_NAMES),
        "model_action_values": list(MODEL_ACTIONS),
        "model_action_names": list(MODEL_ACTION_NAMES),
        "position_names": list(POSITIONS),
        "neutral_action_code_permutation": [
            "TARGET_LONG",
            "TARGET_SHORT",
            "TARGET_FLAT",
        ],
        "cheap_gates": {
            "promotable_primary_ids": [
                "categorical_linear_fqi",
                "categorical_ridge_fqi",
                "extra_trees_fqi",
            ],
            "selection_rule": [
                "largest_minimum_base_stress_delay_cagr_to_strict_mdd",
                "largest_base_cagr_to_strict_mdd",
                "largest_absolute_return",
                "lower_strict_mdd",
                "lexical_policy_id",
            ],
            "2021": {
                "absolute_return_positive": True,
                "cagr_to_strict_mdd_minimum": 1.0,
                "stress_return_positive": True,
                "delay_return_positive": True,
                "minimum_nonflat_intervals": 80,
                "minimum_each_direction_share_of_nonflat": 0.20,
                "beat_required_controls_on_return_and_ratio": True,
                "familywise_p_max_strictly_below": 0.25,
            },
            "2022": {
                "absolute_return_positive": True,
                "cagr_to_strict_mdd_minimum": 1.5,
                "stress_return_positive": True,
                "delay_return_positive": True,
                "minimum_nonflat_intervals": 80,
                "minimum_each_direction_share_of_nonflat": 0.20,
                "beat_required_controls_on_return_and_ratio": True,
                "familywise_p_max_strictly_below": 0.10,
            },
            "2022_reselection_allowed": False,
        },
        "gemma": {
            **asdict(GemmaConfig()),
            "authorized_now": False,
            "objective": (
                "-sum(pi*stop_gradient(center_clip(Q,0.10)))"
                "+0.01*KL(pi||uniform)"
            ),
            "thinking_enabled": False,
            "generated_text_decoded": False,
            "selection_gate": {
                "absolute_return_positive": True,
                "cagr_to_strict_mdd_minimum": 3.0,
                "strict_mdd_pct_maximum": 15.0,
                "minimum_nonflat_intervals": 100,
                "minimum_each_direction_share_of_nonflat": 0.20,
                "both_half_returns_positive": True,
                "stress_return_positive": True,
                "delay_return_positive": True,
                "familywise_p_max_strictly_below": 0.05,
            },
            "immutable_2023_one_policy_p_strictly_below": 0.10,
        },
        "stage_protocol": {
            "transfer_schedules_sealed_before_outcome_loader": True,
            "market_rows_streamed_only_inside_stage": True,
            "funding_rows_streamed_only_inside_stage": True,
            "stage_order": ["2020", "2021", "2022", "2023"],
            "2023_may_select_or_repair": False,
            "post_2023_opened": False,
        },
        "outcome_boundary": {
            "market_rows_parsed": 0,
            "funding_rows_parsed": 0,
            "market_or_funding_payload_bytes_hashed": False,
            "future_returns_created": 0,
            "rewards_created": 0,
            "models_fit": 0,
            "economic_metrics_computed": 0,
            "bctp_2023_market_outcomes_opened": False,
            "post_2023_source_or_outcomes_opened": False,
        },
        "mutable_parameters": [],
        "authorized_next_stage": "implement_and_verify_frozen_economic_runner",
    }


def build_manifest(*, freeze_commit: str) -> dict[str, Any]:
    core = json_normalized(
        _manifest_core(freeze_commit=freeze_commit)
    )
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_manifest(payload: Mapping[str, Any]) -> None:
    freeze_commit = str(payload.get("freeze_commit", ""))
    if (
        len(freeze_commit) != 40
        or any(character not in "0123456789abcdef" for character in freeze_commit)
    ):
        raise ValueError("BCTP evaluator freeze commit is invalid")
    expected = build_manifest(freeze_commit=freeze_commit)
    if dict(payload) != expected:
        raise ValueError("BCTP evaluator freeze differs from frozen contract")


def write_once(path: str | Path, payload: Mapping[str, Any]) -> str:
    target = _path(path)
    encoded = (
        json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if target.exists():
        if target.read_bytes() != encoded:
            raise RuntimeError(f"BCTP write-once freeze drift: {target}")
        return hashlib.sha256(encoded).hexdigest()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=target.parent,
        prefix=f".{target.name}.",
        delete=False,
    ) as handle:
        handle.write(encoded)
        temporary = Path(handle.name)
    temporary.replace(target)
    return hashlib.sha256(encoded).hexdigest()


def freeze_evaluator(
    output_path: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    commit = _assert_protocol_committed()
    payload = build_manifest(freeze_commit=commit)
    validate_manifest(payload)
    digest = write_once(output_path, payload)
    return {
        "decision": "frozen_before_outcomes",
        "output": str(output_path),
        "sha256": digest,
        "manifest_hash": payload["manifest_hash"],
        "authorized_next_stage": payload["authorized_next_stage"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            freeze_evaluator(args.output),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
