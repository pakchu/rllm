"""One-shot pre-2024 evaluation for frozen CLV v1.

The preregistration owns the feature and candidate clock. This module verifies
every frozen dependency before opening only calendar-2023 CLV returns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.evaluate_metaorder_fragmentation_impact_curvature import (
    simulate_schedule,
)
from training.preregister_cross_collateral_liquidity_vacuum import (
    Config as SignalConfig,
    classify_vacuum,
    support_summary,
)
from training.preregister_cross_collateral_liquidity_void_refill import (
    build_features,
    load_sources,
)
from training.preregister_metaorder_fragmentation_impact_curvature import (
    nonoverlapping_schedule,
)


PREREGISTRATION_COMMIT = "d2cf1b3"
PREREGISTRATION_SOURCE = Path(
    "training/preregister_cross_collateral_liquidity_vacuum.py"
)
PREREGISTRATION_SOURCE_SHA256 = (
    "4ea113bb09cb9a00d295c49729c2c876bb6b7f90378fce1b63de6907d4a6b7d7"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/cross-collateral-liquidity-vacuum-preregistration-2026-07-14.md"
)
PREREGISTRATION_DOCUMENT_SHA256 = (
    "b5bbcc0d7d54bbd11ee10711c8fb9d616645f38a8188daafaafb888c4a9c2c0b"
)
PREREGISTRATION_RESULT = Path(
    "results/cross_collateral_liquidity_vacuum_support_2026-07-14.json"
)
PREREGISTRATION_RESULT_SHA256 = (
    "3c661a85e90a6191385b1b69fafac6f8dd146888b6937394493b3aa44096c8d9"
)
FEATURE_SOURCE = Path(
    "training/preregister_cross_collateral_liquidity_void_refill.py"
)
FEATURE_SOURCE_SHA256 = (
    "8465af153f4e5a19299c7ee2b6104e7ea009feb8da80ad10d50cf49bddd7ad51"
)
SCHEDULER_SOURCE = Path(
    "training/preregister_metaorder_fragmentation_impact_curvature.py"
)
SCHEDULER_SOURCE_SHA256 = (
    "51e99dbdc5ba13e6b4ac15e3915ec5b30e36dff89c1e5b31a5f3f7f272f01a59"
)
EXECUTION_SOURCE = Path(
    "training/evaluate_metaorder_fragmentation_impact_curvature.py"
)
EXECUTION_SOURCE_SHA256 = (
    "1589a52605386570485a7e6be3b8f3aa9439a498abb60eaa42272ac62d4cbed3"
)
TRADE_STATS_SOURCE = Path("training/strict_bar_backtest.py")
TRADE_STATS_SOURCE_SHA256 = (
    "3e95ad320d8869755afa1f4907d2d478200a3ebfc015e4eaeace0be0b15f9682"
)
DEPTH_MANIFEST = Path(
    "results/binance_cross_collateral_book_depth_btc_2023_manifest.json"
)
DEPTH_MANIFEST_SHA256 = (
    "95ec6e133dfcc7ed3c058538f380d24d98552c0a921fc24a679d247159a4f080"
)
MARKET_MANIFEST = Path(
    "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
)
MARKET_MANIFEST_SHA256 = (
    "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
)
EVALUATION_SOURCE = Path(
    "training/evaluate_cross_collateral_liquidity_vacuum.py"
)

WINDOWS: dict[str, tuple[str, str]] = {
    "train2023_h1": ("2023-01-01", "2023-07-01"),
    "select2023_h2": ("2023-07-01", "2024-01-01"),
    "q1": ("2023-01-01", "2023-04-01"),
    "q2": ("2023-04-01", "2023-07-01"),
    "q3": ("2023-07-01", "2023-10-01"),
    "q4": ("2023-10-01", "2024-01-01"),
}
POLICY_NAMES = (
    "clv",
    "reverse",
    "always_long",
    "always_short",
    "permuted_sign",
)


@dataclass(frozen=True)
class EvaluationConfig:
    output: str = (
        "results/cross_collateral_liquidity_vacuum_selection_2026-07-14.json"
    )
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    cluster_permutations: int = 100_000
    cluster_seed: int = 20_260_714
    sign_permutation_seed: int = 20_260_714


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_preregistration() -> dict[str, Any]:
    for path, expected in (
        (PREREGISTRATION_SOURCE, PREREGISTRATION_SOURCE_SHA256),
        (PREREGISTRATION_DOCUMENT, PREREGISTRATION_DOCUMENT_SHA256),
        (PREREGISTRATION_RESULT, PREREGISTRATION_RESULT_SHA256),
        (FEATURE_SOURCE, FEATURE_SOURCE_SHA256),
        (SCHEDULER_SOURCE, SCHEDULER_SOURCE_SHA256),
        (EXECUTION_SOURCE, EXECUTION_SOURCE_SHA256),
        (TRADE_STATS_SOURCE, TRADE_STATS_SOURCE_SHA256),
        (DEPTH_MANIFEST, DEPTH_MANIFEST_SHA256),
        (MARKET_MANIFEST, MARKET_MANIFEST_SHA256),
    ):
        if _sha256(path) != expected:
            raise ValueError(f"frozen CLV dependency changed: {path}")

    result = json.loads(PREREGISTRATION_RESULT.read_text())
    protocol = result.get("protocol", {})
    calibration = result.get("support_calibration", {})
    frozen = result.get("frozen_artifacts", {})
    if protocol.get("outcomes_opened_for_clv") is not False:
        raise ValueError("CLV preregistration opened outcomes")
    if protocol.get("support_rejected") is not False:
        raise ValueError("CLV preregistration was support-rejected")
    if result.get("all_support_gates_pass") is not True:
        raise ValueError("CLV support gates are not frozen as passing")
    if result.get("config") != asdict(SignalConfig()):
        raise ValueError("CLV signal config differs from frozen support")
    if calibration.get("selected_score_quantile") != 0.975:
        raise ValueError("CLV support stopping rule changed")
    if calibration.get("outcomes_opened_for_clv") is not False:
        raise ValueError("CLV support calibration opened outcomes")
    if calibration.get("tested_score_quantiles") != [
        0.90,
        0.925,
        0.95,
        0.975,
        0.99,
        0.995,
    ]:
        raise ValueError("CLV support calibration grid changed")
    if calibration.get("all_other_parameters_fixed") is not True:
        raise ValueError("CLV calibration did not fix other parameters")
    if calibration.get("further_support_repairs_allowed") is not False:
        raise ValueError("CLV support artifact permits post-freeze repair")
    if protocol.get("sealed_windows") != ["test2024", "eval2025", "ytd2026"]:
        raise ValueError("CLV sealed-window contract changed")

    expected_frozen = {
        "preregistration_source_sha256": PREREGISTRATION_SOURCE_SHA256,
        "preregistration_document_sha256": PREREGISTRATION_DOCUMENT_SHA256,
        "feature_source_sha256": FEATURE_SOURCE_SHA256,
        "scheduler_source_sha256": SCHEDULER_SOURCE_SHA256,
        "depth_manifest_sha256": DEPTH_MANIFEST_SHA256,
        "market_manifest_sha256": MARKET_MANIFEST_SHA256,
    }
    for key, expected in expected_frozen.items():
        if frozen.get(key) != expected:
            raise ValueError(f"CLV frozen-artifact record changed: {key}")
    return result


def verify_signal_replay(
    signal: pd.DataFrame,
    market: pd.DataFrame,
    cfg: SignalConfig,
    preregistration: dict[str, Any],
) -> None:
    replayed_support = support_summary(signal, market, cfg)
    if replayed_support != preregistration.get("support"):
        raise ValueError("CLV support replay differs from frozen artifact")
    if int(signal["candidate"].sum()) != preregistration.get(
        "raw_candidate_count"
    ):
        raise ValueError("CLV raw candidate replay differs from frozen artifact")
    schedule = pd.concat(
        [
            nonoverlapping_schedule(
                signal,
                market,
                start=start,
                end=end,
            )
            for start, end in (
                ("2023-01-01", "2023-04-01"),
                ("2023-04-01", "2023-07-01"),
                ("2023-07-01", "2023-10-01"),
                ("2023-10-01", "2024-01-01"),
            )
        ],
        ignore_index=True,
    )
    side_counts = {
        "long": int(schedule["side"].gt(0).sum()),
        "short": int(schedule["side"].lt(0).sum()),
    }
    if side_counts != preregistration.get("scheduled_side_counts"):
        raise ValueError("CLV side-count replay differs from frozen artifact")


def policy_schedule(
    reserved_schedule: pd.DataFrame,
    policy: str,
    *,
    permutation_seed: int,
) -> pd.DataFrame:
    """Change actions only after reserving the non-overlap opportunity clock."""
    if policy not in POLICY_NAMES:
        raise ValueError(f"unknown CLV control policy: {policy}")
    output = reserved_schedule.copy()
    if output.empty:
        return output
    if not output["branch"].eq("vacuum").all():
        raise ValueError("CLV schedule contains an unknown branch")
    sides = output["side"].to_numpy(np.int8)
    if not np.isin(sides, [-1, 1]).all():
        raise ValueError("CLV schedule contains an invalid side")

    if policy == "reverse":
        output["side"] = -sides
    elif policy == "always_long":
        output["side"] = np.ones(len(output), dtype=np.int8)
    elif policy == "always_short":
        output["side"] = -np.ones(len(output), dtype=np.int8)
    elif policy == "permuted_sign":
        rng = np.random.default_rng(permutation_seed)
        output["side"] = rng.permutation(sides)
    return output.reset_index(drop=True)


def evaluate_policy(
    market: pd.DataFrame,
    reserved_schedule: pd.DataFrame,
    *,
    policy: str,
    start: str,
    end: str,
    cfg: EvaluationConfig,
) -> dict[str, Any]:
    schedule = policy_schedule(
        reserved_schedule,
        policy,
        permutation_seed=cfg.sign_permutation_seed,
    )
    metrics = simulate_schedule(
        market,
        schedule,
        start=start,
        end=end,
        cfg=cfg,
    )
    metrics.pop("continuation_count", None)
    metrics.pop("fade_count", None)
    metrics["reserved_candidate_count"] = int(len(reserved_schedule))
    metrics["executed_candidate_count"] = int(len(schedule))
    return metrics


def _qualification(windows: dict[str, Any]) -> dict[str, Any]:
    train = windows["train2023_h1"]["clv"]
    select = windows["select2023_h2"]["clv"]
    failures: list[str] = []

    for name, metrics in (
        ("train2023_h1", train),
        ("select2023_h2", select),
    ):
        if metrics["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}: non-positive absolute return")
        if metrics["cagr_to_strict_mdd"] < 3.0:
            failures.append(f"{name}: CAGR/strict-MDD below 3")
        if metrics["strict_mdd_pct"] > 15.0:
            failures.append(f"{name}: strict MDD above 15%")
        if metrics["trade_count"] < 180:
            failures.append(f"{name}: fewer than 180 trades")
        p_value = metrics["weekly_cluster_sign_flip"]["p_value_one_sided"]
        if p_value >= 0.10:
            failures.append(f"{name}: weekly-cluster p-value not below 0.10")

    for name in ("q1", "q2", "q3", "q4"):
        metrics = windows[name]["clv"]
        if metrics["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}: non-positive absolute return")
        if metrics["trade_count"] < 75:
            failures.append(f"{name}: fewer than 75 trades")

    clv_min_ratio = min(
        train["cagr_to_strict_mdd"],
        select["cagr_to_strict_mdd"],
    )
    for control in ("reverse", "always_long", "always_short"):
        control_min_ratio = min(
            windows["train2023_h1"][control]["cagr_to_strict_mdd"],
            windows["select2023_h2"][control]["cagr_to_strict_mdd"],
        )
        if clv_min_ratio <= control_min_ratio:
            failures.append(
                "clv: minimum train/select ratio does not beat " + control
            )
    return {
        "qualifies": not failures,
        "failures": failures,
        "clv_min_train_select_ratio": float(clv_min_ratio),
    }


def run_evaluation(cfg: EvaluationConfig) -> dict[str, Any]:
    preregistration = verify_preregistration()
    signal_cfg = SignalConfig()
    market, source = load_sources(signal_cfg)
    features = build_features(market, signal_cfg)
    signal = classify_vacuum(features, signal_cfg)
    verify_signal_replay(signal, market, signal_cfg, preregistration)

    windows: dict[str, Any] = {}
    for window_name, (start, end) in WINDOWS.items():
        reserved = nonoverlapping_schedule(
            signal,
            market,
            start=start,
            end=end,
        )
        windows[window_name] = {
            policy: evaluate_policy(
                market,
                reserved,
                policy=policy,
                start=start,
                end=end,
                cfg=cfg,
            )
            for policy in POLICY_NAMES
        }

    qualification = _qualification(windows)
    return {
        "protocol": {
            "name": "CLV v1 frozen pre-2024 selection evaluation",
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "preregistration_source_sha256": PREREGISTRATION_SOURCE_SHA256,
            "preregistration_document_sha256": PREREGISTRATION_DOCUMENT_SHA256,
            "preregistration_result_sha256": PREREGISTRATION_RESULT_SHA256,
            "evaluation_source_sha256": _sha256(EVALUATION_SOURCE),
            "outcomes_opened_for_clv": True,
            "opened_windows": list(WINDOWS),
            "sealed_windows": ["test2024", "eval2025", "ytd2026"],
            "signal_parameters_mutable": False,
            "candidate_clock_reserved_before_control": True,
            "entry": "next 5m open after completed depth and market bar",
            "exit": "scheduled open 12 completed 5m bars after entry",
            "strict_mdd": (
                "held path, favorable extreme first then adverse extreme"
            ),
            "cagr": "full wall-clock split including idle cash",
        },
        "evaluation_config": asdict(cfg),
        "signal_config": preregistration["config"],
        "source": source,
        "windows": windows,
        "qualification": qualification,
        "selection": {
            "selected_alpha": "clv" if qualification["qualifies"] else None,
            "rejected": not qualification["qualifies"],
            "reason": (
                "passed every frozen train/select gate"
                if qualification["qualifies"]
                else "CLV v1 failed at least one frozen train/select gate"
            ),
        },
    }


def _headline(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics[key]
        for key in (
            "absolute_return_pct",
            "cagr_pct",
            "strict_mdd_pct",
            "cagr_to_strict_mdd",
            "trade_count",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=EvaluationConfig.output)
    args = parser.parse_args()
    cfg = EvaluationConfig(output=args.output)
    result = run_evaluation(cfg)
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "selection": result["selection"],
                "qualification": result["qualification"],
                "clv": {
                    name: _headline(policies["clv"])
                    for name, policies in result["windows"].items()
                },
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
