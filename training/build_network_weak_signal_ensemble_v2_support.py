"""Build the price-free NWE-8 feature clock before return labels exist."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from training import build_network_weak_signal_ensemble_support as base
from training import preregister_network_weak_signal_ensemble_v2 as prereg


DEFAULT_OUTPUT = "results/network_weak_signal_ensemble_v2_support_2026-07-17.json"
DEFAULT_CLOCK = "results/network_weak_signal_ensemble_v2_feature_clock_2026-07-17.csv"


def load_preregistration(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    prereg.validate_manifest(payload)
    if payload["outcomes_opened"] is not False:
        raise RuntimeError("NWE-8 support cannot run after outcomes are opened")
    if payload["policy"] != asdict(prereg.Policy()):
        raise RuntimeError("NWE-8 support policy differs from preregistration")
    return payload


def support_summary(clock: pd.DataFrame, payload: dict[str, Any]) -> dict[str, Any]:
    eligible = clock.loc[clock["prediction_eligible"].astype(bool)].copy()
    decision = pd.to_datetime(eligible["decision_date"])
    train_start = pd.Timestamp(payload["selection_protocol"]["train"][0])
    windows = {
        "train_2021_2022": (train_start, pd.Timestamp("2023-01-01")),
        "train_2021": (train_start, pd.Timestamp("2022-01-01")),
        "train_2022": (pd.Timestamp("2022-01-01"), pd.Timestamp("2023-01-01")),
        "selection_2023": (pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")),
        "selection_2023_h1": (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-07-01")),
        "selection_2023_h2": (pd.Timestamp("2023-07-01"), pd.Timestamp("2024-01-01")),
    }
    counts = {
        key: int(((decision >= start) & (decision < end)).sum())
        for key, (start, end) in windows.items()
    }
    history = clock.loc[
        (pd.to_datetime(clock["decision_date"]) < train_start)
        & (pd.to_datetime(clock["feature_available_at"]) <= train_start)
        & (pd.to_datetime(clock["exit_date"]) <= train_start)
        & clock["all_features_finite"].astype(bool)
    ]
    gate = payload["support_freeze_before_labels"]
    checks = {
        "train_total": counts["train_2021_2022"]
        >= gate["train_2021_2022_candidate_weeks_min"],
        "train_2021": counts["train_2021"] >= gate["train_2021_candidate_weeks_min"],
        "train_2022": counts["train_2022"] >= gate["train_2022_candidate_weeks_min"],
        "selection_2023": counts["selection_2023"]
        >= gate["selection_2023_candidate_weeks_min"],
        "selection_2023_h1": counts["selection_2023_h1"]
        >= gate["selection_2023_h1_candidate_weeks_min"],
        "selection_2023_h2": counts["selection_2023_h2"]
        >= gate["selection_2023_h2_candidate_weeks_min"],
        "initial_training_history": len(history)
        >= gate["initial_fully_available_training_samples_min"],
        "all_prediction_features_finite": bool(eligible["all_features_finite"].all()),
    }
    return {
        "candidate_counts": counts,
        "initial_fully_available_training_samples": int(len(history)),
        "first_prediction_decision": str(decision.min()) if len(decision) else None,
        "last_prediction_decision": str(decision.max()) if len(decision) else None,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def run(
    *,
    preregistration: str = prereg.DEFAULT_OUTPUT,
    output: str = DEFAULT_OUTPUT,
    clock_output: str = DEFAULT_CLOCK,
) -> dict[str, Any]:
    payload = load_preregistration(preregistration)
    policy = prereg.Policy(**payload["policy"])
    sources, source_summary = base.load_sources(payload)
    daily = base.build_daily_features(sources, policy)
    clock = base.build_feature_clock(daily, policy)
    support = support_summary(clock, payload)
    clock_path = Path(clock_output)
    clock_path.parent.mkdir(parents=True, exist_ok=True)
    clock.to_csv(clock_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    result_core: dict[str, Any] = {
        "protocol_version": "network_weak_signal_ensemble_v2_support_v1",
        "outcomes_opened": False,
        "policy": asdict(policy),
        "feature_columns": list(prereg.FEATURE_COLUMNS),
        "preregistration": preregistration,
        "preregistration_sha256": base.sha256_file(preregistration),
        "preregistration_manifest_hash": payload["manifest_hash"],
        "source": {
            **source_summary,
            "blockspace_sha256": base.sha256_file(payload["source_contract"]["blockspace"]),
            "network_sha256": base.sha256_file(payload["source_contract"]["network"]),
            "columns_loaded": [
                "observation_date",
                "available_at",
                "FeeTotNtv",
                "IssTotNtv",
                "BlkCnt",
                "TxCnt",
                "AdrActCnt",
                "TxTfrCnt",
            ],
            "market_or_return_rows_loaded": 0,
        },
        "daily_feature_frame": {
            "rows": int(len(daily)),
            "frame_hash": base.frame_hash(daily),
            "first_observation": str(daily["observation_date"].min()),
            "last_observation": str(daily["observation_date"].max()),
        },
        "feature_clock": {
            "path": clock_output,
            "sha256": base.sha256_file(clock_output),
            "frame_hash": base.frame_hash(clock),
            "rows": int(len(clock)),
            "prediction_eligible_rows": int(clock["prediction_eligible"].sum()),
        },
        "support_gate": support,
        "sealed": ["all_return_labels", "2024", "2025", "2026_ytd"],
        "failure_action": (
            "reject without constructing return labels"
            if not support["passed"]
            else "freeze this exact feature clock before implementing the evaluator"
        ),
    }
    result = {
        **result_core,
        "result_hash": prereg.canonical_hash(result_core),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", default=prereg.DEFAULT_OUTPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--clock-output", default=DEFAULT_CLOCK)
    args = parser.parse_args()
    result = run(
        preregistration=args.preregistration,
        output=args.output,
        clock_output=args.clock_output,
    )
    print(
        json.dumps(
            {
                "outcomes_opened": result["outcomes_opened"],
                "policy_id": result["policy"]["policy_id"],
                "support_gate": result["support_gate"],
                "feature_clock": result["feature_clock"],
                "result_hash": result["result_hash"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
