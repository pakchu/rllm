"""One-shot pre-2024 evaluation for the frozen BIFT v1 protocol.

The signal and support clock live in the preregistration module.  This module
verifies every frozen dependency before opening only the 2020-2023 outcomes.
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
from training.preregister_bayesian_impact_flow_transition import (
    Config as SignalConfig,
    build_hourly_diagnostics,
    classify_candidates,
    nonoverlapping_bift_schedule,
    project_to_five_minute,
)
from training.preregister_metaorder_fragmentation_impact_curvature import (
    Config as SourceConfig,
    load_causal_frame,
)


PREREGISTRATION_COMMIT = "a1b9b0b"
PREREGISTRATION_SOURCE = Path(
    "training/preregister_bayesian_impact_flow_transition.py"
)
PREREGISTRATION_SOURCE_SHA256 = (
    "acd39e2c6b8bcf3cd6266b235c793c85b397ac646c10e52b7cb8d6404181cecc"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/bayesian-impact-flow-transition-preregistration-2026-07-14.md"
)
PREREGISTRATION_DOCUMENT_SHA256 = (
    "6b407f69081ebd74ad1b9c58f18736f8bb632e3e249d663045f010488246fa06"
)
PREREGISTRATION_RESULT = Path(
    "results/bayesian_impact_flow_transition_support_2026-07-14.json"
)
PREREGISTRATION_RESULT_SHA256 = (
    "4bde483193545d8398d8e62bc539e4d32d92f8d0e785c54d34389e7cab128242"
)
SOURCE_LOADER = Path(
    "training/preregister_metaorder_fragmentation_impact_curvature.py"
)
SOURCE_LOADER_SHA256 = (
    "51e99dbdc5ba13e6b4ac15e3915ec5b30e36dff89c1e5b31a5f3f7f272f01a59"
)
EXECUTION_SOURCE = Path(
    "training/evaluate_metaorder_fragmentation_impact_curvature.py"
)
EXECUTION_SOURCE_SHA256 = (
    "1589a52605386570485a7e6be3b8f3aa9439a498abb60eaa42272ac62d4cbed3"
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

WINDOWS: dict[str, tuple[str, str]] = {
    "train": ("2020-01-01", "2023-01-01"),
    "select2023": ("2023-01-01", "2024-01-01"),
    "select2023_h1": ("2023-01-01", "2023-07-01"),
    "select2023_h2": ("2023-07-01", "2024-01-01"),
}
POLICY_NAMES = (
    "bift",
    "always_follow",
    "always_fade",
    "propagation_only",
    "absorption_only",
    "permuted_branch",
)


@dataclass(frozen=True)
class EvaluationConfig:
    output: str = (
        "results/bayesian_impact_flow_transition_selection_2026-07-14.json"
    )
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    cluster_permutations: int = 100_000
    cluster_seed: int = 20_260_714
    branch_permutation_seed: int = 20_260_714


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_preregistration() -> dict[str, Any]:
    for path, expected in (
        (PREREGISTRATION_SOURCE, PREREGISTRATION_SOURCE_SHA256),
        (PREREGISTRATION_DOCUMENT, PREREGISTRATION_DOCUMENT_SHA256),
        (PREREGISTRATION_RESULT, PREREGISTRATION_RESULT_SHA256),
        (SOURCE_LOADER, SOURCE_LOADER_SHA256),
        (EXECUTION_SOURCE, EXECUTION_SOURCE_SHA256),
        (AGGTRADE_MANIFEST, AGGTRADE_MANIFEST_SHA256),
        (KLINE_MANIFEST, KLINE_MANIFEST_SHA256),
    ):
        if _sha256(path) != expected:
            raise ValueError(f"frozen BIFT dependency changed: {path}")
    result = json.loads(PREREGISTRATION_RESULT.read_text())
    protocol = result.get("protocol", {})
    calibration = result.get("support_calibration", {})
    if protocol.get("outcomes_opened_for_bift") is not False:
        raise ValueError("BIFT preregistration opened outcomes")
    if result.get("all_support_gates_pass") is not True:
        raise ValueError("BIFT support gates are not frozen as passing")
    if result.get("config") != asdict(SignalConfig()):
        raise ValueError("BIFT signal config differs from frozen support")
    if calibration.get("selected_change_quantile") != 0.925:
        raise ValueError("BIFT support stopping rule changed")
    if calibration.get("further_support_repairs_allowed") is not False:
        raise ValueError("BIFT support artifact permits post-freeze repair")
    if protocol.get("sealed_windows") != ["test2024", "eval2025", "ytd2026"]:
        raise ValueError("BIFT sealed-window contract changed")
    return result


def _reference_side(schedule: pd.DataFrame) -> np.ndarray:
    if schedule.empty:
        return np.empty(0, dtype=np.int8)
    side = schedule["side"].to_numpy(np.int8)
    branch = schedule["branch"].astype(str).to_numpy()
    known = np.isin(branch, ["propagation", "absorption"])
    if not known.all():
        raise ValueError("BIFT schedule contains an unknown branch")
    return np.where(branch == "propagation", side, -side).astype(np.int8)


def policy_schedule(
    schedule: pd.DataFrame,
    policy: str,
    *,
    permutation_seed: int,
) -> pd.DataFrame:
    """Apply a control action after the non-overlap candidate clock is fixed."""
    if policy not in POLICY_NAMES:
        raise ValueError(f"unknown BIFT control policy: {policy}")
    output = schedule.copy()
    if output.empty:
        return output
    reference = _reference_side(output)
    branches = output["branch"].astype(str).to_numpy()

    if policy == "always_follow":
        output["side"] = reference
    elif policy == "always_fade":
        output["side"] = -reference
    elif policy == "propagation_only":
        output = output.loc[branches == "propagation"].copy()
    elif policy == "absorption_only":
        output = output.loc[branches == "absorption"].copy()
    elif policy == "permuted_branch":
        rng = np.random.default_rng(permutation_seed)
        permuted_propagation = rng.permutation(branches == "propagation")
        output["side"] = np.where(
            permuted_propagation, reference, -reference
        ).astype(np.int8)
        output["branch"] = np.where(
            permuted_propagation, "propagation", "absorption"
        )
    return output.reset_index(drop=True)


def evaluate_policy(
    frame: pd.DataFrame,
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
        permutation_seed=cfg.branch_permutation_seed,
    )
    metrics = simulate_schedule(
        frame,
        schedule,
        start=start,
        end=end,
        cfg=cfg,
    )
    metrics.pop("continuation_count", None)
    metrics.pop("fade_count", None)
    metrics["reserved_candidate_count"] = int(len(reserved_schedule))
    metrics["executed_candidate_count"] = int(len(schedule))
    metrics["branch_counts"] = {
        name: int(value)
        for name, value in schedule["branch"].value_counts().items()
    } if not schedule.empty else {}
    return metrics


def _qualification(windows: dict[str, Any]) -> dict[str, Any]:
    train = windows["train"]["bift"]
    select = windows["select2023"]["bift"]
    h1 = windows["select2023_h1"]["bift"]
    h2 = windows["select2023_h2"]["bift"]
    failures: list[str] = []
    for name, metrics in (("train", train), ("select2023", select)):
        if metrics["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}: non-positive absolute return")
        if metrics["cagr_to_strict_mdd"] < 3.0:
            failures.append(f"{name}: CAGR/strict-MDD below 3")
        if metrics["strict_mdd_pct"] > 15.0:
            failures.append(f"{name}: strict MDD above 15%")
        if metrics["weekly_cluster_sign_flip"]["p_value_one_sided"] >= 0.10:
            failures.append(f"{name}: weekly-cluster p-value not below 0.10")
    for name, metrics in (("select2023_h1", h1), ("select2023_h2", h2)):
        if metrics["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}: non-positive absolute return")
        if metrics["trade_count"] < 30:
            failures.append(f"{name}: fewer than 30 trades")
    if select["trade_count"] < 80:
        failures.append("select2023: fewer than 80 trades")

    bift_min_ratio = min(
        train["cagr_to_strict_mdd"],
        select["cagr_to_strict_mdd"],
    )
    for control in ("always_follow", "always_fade"):
        control_min_ratio = min(
            windows["train"][control]["cagr_to_strict_mdd"],
            windows["select2023"][control]["cagr_to_strict_mdd"],
        )
        if bift_min_ratio <= control_min_ratio:
            failures.append(
                f"bift: minimum train/select ratio does not beat {control}"
            )
    return {
        "qualifies": not failures,
        "failures": failures,
        "bift_min_train_select_ratio": float(bift_min_ratio),
    }


def run_evaluation(cfg: EvaluationConfig) -> dict[str, Any]:
    preregistration = verify_preregistration()
    signal_cfg = SignalConfig()
    frame, source = load_causal_frame(SourceConfig())
    hourly = build_hourly_diagnostics(frame, signal_cfg)
    hourly_state = classify_candidates(hourly, signal_cfg)
    signal = project_to_five_minute(hourly_state, frame)

    windows: dict[str, Any] = {}
    for window_name, (start, end) in WINDOWS.items():
        reserved = nonoverlapping_bift_schedule(
            signal,
            frame,
            start=start,
            end=end,
        )
        windows[window_name] = {
            policy: evaluate_policy(
                frame,
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
            "name": "BIFT v1 frozen pre-2024 selection evaluation",
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "preregistration_source_sha256": PREREGISTRATION_SOURCE_SHA256,
            "preregistration_document_sha256": PREREGISTRATION_DOCUMENT_SHA256,
            "preregistration_result_sha256": PREREGISTRATION_RESULT_SHA256,
            "outcomes_opened_for_bift": True,
            "opened_windows": list(WINDOWS),
            "sealed_windows": ["test2024", "eval2025", "ytd2026"],
            "signal_parameters_mutable": False,
            "candidate_clock_reserved_before_control": True,
            "entry": "next 5m open",
            "exit": "scheduled 144-bar future 5m open",
            "strict_mdd": "held path, favorable extreme first then adverse extreme",
            "cagr": "full wall-clock split including idle cash",
        },
        "evaluation_config": asdict(cfg),
        "signal_config": preregistration["config"],
        "source": source,
        "windows": windows,
        "qualification": qualification,
        "selection": {
            "selected_alpha": "bift" if qualification["qualifies"] else None,
            "rejected": not qualification["qualifies"],
            "reason": (
                "passed every frozen train/select gate"
                if qualification["qualifies"]
                else "BIFT v1 failed at least one frozen train/select gate"
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
                "bift": {
                    name: _headline(policies["bift"])
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
