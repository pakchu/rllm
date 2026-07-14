"""One-shot pre-2024 evaluation for frozen CFCF v1.

The preregistration module owns the signal and reserved candidate clock. This
module verifies every frozen dependency before opening only 2021-2023 returns.
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
from training.preregister_cross_venue_funding_consensus_fracture import (
    Config as SignalConfig,
    build_settlement_features,
    classify_settlements,
    load_sources,
    nonoverlapping_cfcf_schedule,
    project_to_market,
    support_summary,
)


PREREGISTRATION_COMMIT = "17c6631"
PREREGISTRATION_SOURCE = Path(
    "training/preregister_cross_venue_funding_consensus_fracture.py"
)
PREREGISTRATION_SOURCE_SHA256 = (
    "a426174df6014b54a7a3966774c9eb08fb4c75997dfca561ba5023d110da3431"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/cross-venue-funding-consensus-fracture-preregistration-2026-07-14.md"
)
PREREGISTRATION_DOCUMENT_SHA256 = (
    "7b2f328e69dc6dc90ecef63ff567196472bc63f7fe89d769f17b151abf159448"
)
PREREGISTRATION_RESULT = Path(
    "results/cross_venue_funding_consensus_fracture_support_2026-07-14.json"
)
PREREGISTRATION_RESULT_SHA256 = (
    "99a40e9a03b754dc1064edc887dd480350ff037beb94d6a688dbc5464991f61b"
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
BINANCE_MANIFEST = Path(
    "results/binance_um_aux_btc_2021_2023_manifest.json"
)
BINANCE_MANIFEST_SHA256 = (
    "80c77f461be54b77c7554837a304a187321a052dd05cb39b4e0a3c80de5d2bdc"
)
BYBIT_MANIFEST = Path(
    "results/bybit_linear_aux_btc_2021_2023_manifest.json"
)
BYBIT_MANIFEST_SHA256 = (
    "2d57d60b17bfed75f0c48c557e2a51edc2eb449bf3f48f996070290199d7ae64"
)
MARKET_MANIFEST = Path(
    "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
)
MARKET_MANIFEST_SHA256 = (
    "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
)
EVALUATION_SOURCE = Path(
    "training/evaluate_cross_venue_funding_consensus_fracture.py"
)

WINDOWS: dict[str, tuple[str, str]] = {
    "train": ("2021-01-01", "2023-01-01"),
    "select2023": ("2023-01-01", "2024-01-01"),
    "select2023_h1": ("2023-01-01", "2023-07-01"),
    "select2023_h2": ("2023-07-01", "2024-01-01"),
}
POLICY_NAMES = (
    "cfcf",
    "reverse",
    "always_long",
    "always_short",
    "bybit_rich_only",
    "bybit_cheap_only",
    "permuted_branch",
)


@dataclass(frozen=True)
class EvaluationConfig:
    output: str = (
        "results/cross_venue_funding_consensus_fracture_selection_2026-07-14.json"
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
        (SCHEDULER_SOURCE, SCHEDULER_SOURCE_SHA256),
        (EXECUTION_SOURCE, EXECUTION_SOURCE_SHA256),
        (BINANCE_MANIFEST, BINANCE_MANIFEST_SHA256),
        (BYBIT_MANIFEST, BYBIT_MANIFEST_SHA256),
        (MARKET_MANIFEST, MARKET_MANIFEST_SHA256),
    ):
        if _sha256(path) != expected:
            raise ValueError(f"frozen CFCF dependency changed: {path}")

    result = json.loads(PREREGISTRATION_RESULT.read_text())
    protocol = result.get("protocol", {})
    calibration = result.get("support_calibration", {})
    frozen = result.get("frozen_artifacts", {})
    if protocol.get("outcomes_opened_for_cfcf") is not False:
        raise ValueError("CFCF preregistration opened outcomes")
    if result.get("all_support_gates_pass") is not True:
        raise ValueError("CFCF support gates are not frozen as passing")
    if result.get("config") != asdict(SignalConfig()):
        raise ValueError("CFCF signal config differs from frozen support")
    if calibration.get("selected_crowding_quantile") != 0.90:
        raise ValueError("CFCF support stopping rule changed")
    if calibration.get("outcomes_opened_for_cfcf") is not False:
        raise ValueError("CFCF support calibration opened outcomes")
    if calibration.get("tested_crowding_quantiles") != [
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.925,
        0.95,
        0.975,
    ]:
        raise ValueError("CFCF support calibration grid changed")
    if calibration.get("all_other_parameters_fixed") is not True:
        raise ValueError("CFCF support calibration did not fix other parameters")
    if calibration.get("further_support_repairs_allowed") is not False:
        raise ValueError("CFCF support artifact permits post-freeze repair")
    if protocol.get("sealed_windows") != ["test2024", "eval2025", "ytd2026"]:
        raise ValueError("CFCF sealed-window contract changed")
    expected_frozen = {
        "preregistration_source_sha256": PREREGISTRATION_SOURCE_SHA256,
        "preregistration_document_sha256": PREREGISTRATION_DOCUMENT_SHA256,
        "scheduler_source_sha256": SCHEDULER_SOURCE_SHA256,
        "binance_manifest_sha256": BINANCE_MANIFEST_SHA256,
        "bybit_manifest_sha256": BYBIT_MANIFEST_SHA256,
        "market_manifest_sha256": MARKET_MANIFEST_SHA256,
    }
    for key, expected in expected_frozen.items():
        if frozen.get(key) != expected:
            raise ValueError(f"CFCF frozen-artifact record changed: {key}")
    return result


def verify_signal_replay(
    state: pd.DataFrame,
    signal: pd.DataFrame,
    market: pd.DataFrame,
    cfg: SignalConfig,
    preregistration: dict[str, Any],
) -> None:
    replayed_support = support_summary(signal, market, cfg)
    if replayed_support != preregistration.get("support"):
        raise ValueError("CFCF support replay differs from frozen artifact")
    if int(state["candidate"].sum()) != preregistration.get(
        "raw_candidate_count"
    ):
        raise ValueError("CFCF raw candidate replay differs from frozen artifact")
    schedule = pd.concat(
        [
            nonoverlapping_cfcf_schedule(
                signal,
                market,
                start=f"{year}-01-01",
                end=f"{year + 1}-01-01",
            )
            for year in range(2021, 2024)
        ],
        ignore_index=True,
    )
    branch_counts = {
        name: int(value)
        for name, value in schedule["branch"].value_counts().items()
    }
    if branch_counts != preregistration.get("scheduled_branch_counts"):
        raise ValueError("CFCF branch-count replay differs from frozen artifact")


def policy_schedule(
    reserved_schedule: pd.DataFrame,
    policy: str,
    *,
    permutation_seed: int,
) -> pd.DataFrame:
    """Change actions only after the non-overlap opportunity clock is fixed."""
    if policy not in POLICY_NAMES:
        raise ValueError(f"unknown CFCF control policy: {policy}")
    output = reserved_schedule.copy()
    if output.empty:
        return output

    branches = output["branch"].astype(str).to_numpy()
    known = np.isin(branches, ["bybit_rich", "bybit_cheap"])
    if not known.all():
        raise ValueError("CFCF schedule contains an unknown branch")

    if policy == "reverse":
        output["side"] = -output["side"].to_numpy(np.int8)
    elif policy == "always_long":
        output["side"] = np.ones(len(output), dtype=np.int8)
    elif policy == "always_short":
        output["side"] = -np.ones(len(output), dtype=np.int8)
    elif policy == "bybit_rich_only":
        output = output.loc[branches == "bybit_rich"].copy()
    elif policy == "bybit_cheap_only":
        output = output.loc[branches == "bybit_cheap"].copy()
    elif policy == "permuted_branch":
        rng = np.random.default_rng(permutation_seed)
        permuted_rich = rng.permutation(branches == "bybit_rich")
        output["side"] = np.where(permuted_rich, -1, 1).astype(np.int8)
        output["branch"] = np.where(
            permuted_rich,
            "bybit_rich",
            "bybit_cheap",
        )
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
        permutation_seed=cfg.branch_permutation_seed,
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
    metrics["branch_counts"] = (
        {
            name: int(value)
            for name, value in schedule["branch"].value_counts().items()
        }
        if not schedule.empty
        else {}
    )
    return metrics


def _qualification(windows: dict[str, Any]) -> dict[str, Any]:
    train = windows["train"]["cfcf"]
    select = windows["select2023"]["cfcf"]
    h1 = windows["select2023_h1"]["cfcf"]
    h2 = windows["select2023_h2"]["cfcf"]
    failures: list[str] = []

    for name, metrics in (("train", train), ("select2023", select)):
        if metrics["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}: non-positive absolute return")
        if metrics["cagr_to_strict_mdd"] < 3.0:
            failures.append(f"{name}: CAGR/strict-MDD below 3")
        if metrics["strict_mdd_pct"] > 15.0:
            failures.append(f"{name}: strict MDD above 15%")
        p_value = metrics["weekly_cluster_sign_flip"]["p_value_one_sided"]
        if p_value >= 0.10:
            failures.append(f"{name}: weekly-cluster p-value not below 0.10")

    for name, metrics in (("select2023_h1", h1), ("select2023_h2", h2)):
        if metrics["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}: non-positive absolute return")
        if metrics["trade_count"] < 30:
            failures.append(f"{name}: fewer than 30 trades")
    if select["trade_count"] < 80:
        failures.append("select2023: fewer than 80 trades")

    cfcf_min_ratio = min(
        train["cagr_to_strict_mdd"],
        select["cagr_to_strict_mdd"],
    )
    for control in ("reverse", "always_long", "always_short"):
        control_min_ratio = min(
            windows["train"][control]["cagr_to_strict_mdd"],
            windows["select2023"][control]["cagr_to_strict_mdd"],
        )
        if cfcf_min_ratio <= control_min_ratio:
            failures.append(
                "cfcf: minimum train/select ratio does not beat " + control
            )
    return {
        "qualifies": not failures,
        "failures": failures,
        "cfcf_min_train_select_ratio": float(cfcf_min_ratio),
    }


def run_evaluation(cfg: EvaluationConfig) -> dict[str, Any]:
    preregistration = verify_preregistration()
    signal_cfg = SignalConfig()
    premium, funding, market, source = load_sources(signal_cfg)
    settlements = build_settlement_features(premium, funding, signal_cfg)
    state = classify_settlements(settlements, signal_cfg)
    signal = project_to_market(state, market)
    verify_signal_replay(state, signal, market, signal_cfg, preregistration)

    windows: dict[str, Any] = {}
    for window_name, (start, end) in WINDOWS.items():
        reserved = nonoverlapping_cfcf_schedule(
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
            "name": "CFCF v1 frozen pre-2024 selection evaluation",
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "preregistration_source_sha256": PREREGISTRATION_SOURCE_SHA256,
            "preregistration_document_sha256": PREREGISTRATION_DOCUMENT_SHA256,
            "preregistration_result_sha256": PREREGISTRATION_RESULT_SHA256,
            "evaluation_source_sha256": _sha256(EVALUATION_SOURCE),
            "outcomes_opened_for_cfcf": True,
            "opened_windows": list(WINDOWS),
            "sealed_windows": ["test2024", "eval2025", "ytd2026"],
            "signal_parameters_mutable": False,
            "candidate_clock_reserved_before_control": True,
            "entry": "next 5m open after completed premium hour",
            "exit": "next funding-boundary open, 84 bars after entry",
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
            "selected_alpha": "cfcf" if qualification["qualifies"] else None,
            "rejected": not qualification["qualifies"],
            "reason": (
                "passed every frozen train/select gate"
                if qualification["qualifies"]
                else "CFCF v1 failed at least one frozen train/select gate"
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
                "cfcf": {
                    name: _headline(policies["cfcf"])
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
