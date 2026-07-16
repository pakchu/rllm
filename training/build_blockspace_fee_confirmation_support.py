"""Build the outcome-blind BFC-3 support clock from Coin Metrics only."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_blockspace_fee_confirmation as prereg
from training.build_network_topology_broadening_support import (
    _strict_prior_z,
    frame_hash,
    sha256_file,
)


DEFAULT_OUTPUT = "results/blockspace_fee_confirmation_support_2026-07-17.json"
DEFAULT_CLOCK = "results/blockspace_fee_confirmation_clock_2026-07-17.csv"
SELECTION_END = pd.Timestamp("2024-01-01")
TRAIN_START = pd.Timestamp("2021-03-01")


def load_preregistration(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    prereg.validate_manifest(payload)
    if payload["outcomes_opened"] is not False:
        raise RuntimeError("BFC-3 support cannot run after outcomes are opened")
    if payload["policy"] != asdict(prereg.Policy()):
        raise RuntimeError("BFC-3 support policy differs from preregistration")
    return payload


def load_blockspace(payload: dict[str, Any]) -> pd.DataFrame:
    source = payload["source_contract"]
    if sha256_file(source["blockspace"]) != source["blockspace_sha256"]:
        raise RuntimeError("BFC-3 blockspace source hash mismatch")
    if sha256_file(source["blockspace_manifest"]) != source["blockspace_manifest_sha256"]:
        raise RuntimeError("BFC-3 blockspace manifest hash mismatch")
    manifest = json.loads(Path(source["blockspace_manifest"]).read_text())
    if manifest.get("sha256") != source["blockspace_sha256"]:
        raise RuntimeError("blockspace manifest does not identify frozen source")
    expected_columns = [
        "observation_date",
        "available_at",
        "FeeTotNtv",
        "IssTotNtv",
        "BlkCnt",
        "TxCnt",
    ]
    if manifest.get("columns") != expected_columns:
        raise RuntimeError("blockspace manifest columns differ from BFC-3 contract")
    frame = pd.read_csv(
        source["blockspace"],
        usecols=expected_columns,
        parse_dates=["observation_date", "available_at"],
    )
    if len(frame) != int(source["blockspace_rows"]):
        raise RuntimeError("blockspace row count differs from preregistration")
    if frame["observation_date"].duplicated().any():
        raise RuntimeError("blockspace observation dates must be unique")
    frame = frame.sort_values("observation_date").reset_index(drop=True)
    if frame["observation_date"].max() >= SELECTION_END:
        raise RuntimeError("BFC-3 support source crosses sealed 2024 boundary")
    if (frame["available_at"] < frame["observation_date"] + pd.Timedelta(days=1)).any():
        raise RuntimeError("blockspace metric became available before daily completion")
    for column in ("FeeTotNtv", "IssTotNtv", "BlkCnt", "TxCnt"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if (~np.isfinite(frame[column])).any():
            raise RuntimeError(f"blockspace column must be finite: {column}")
    if (frame["FeeTotNtv"] < 0.0).any():
        raise RuntimeError("FeeTotNtv must be nonnegative")
    if (frame[["IssTotNtv", "BlkCnt", "TxCnt"]] <= 0.0).any().any():
        raise RuntimeError("issuance, block count, and transaction count must be positive")
    return frame


def build_features(blockspace: pd.DataFrame, policy: prereg.Policy) -> pd.DataFrame:
    frame = blockspace[["observation_date", "available_at"]].copy()
    fee = blockspace["FeeTotNtv"].to_numpy(float)
    issuance = blockspace["IssTotNtv"].to_numpy(float)
    blocks = blockspace["BlkCnt"].to_numpy(float)
    transactions = blockspace["TxCnt"].to_numpy(float)
    fee_share = np.full(len(blockspace), np.nan, dtype=float)
    positive_fee = fee > 0.0
    fee_share[positive_fee] = np.log(fee[positive_fee] / issuance[positive_fee])
    transaction_density = np.log(transactions / blocks)
    available = frame["available_at"].to_numpy(dtype="datetime64[ns]")
    fee_z, fee_reference_count = _strict_prior_z(
        fee_share,
        available,
        reference_days=policy.reference_days,
        minimum=policy.reference_min_periods,
    )
    density_z, density_reference_count = _strict_prior_z(
        transaction_density,
        available,
        reference_days=policy.reference_days,
        minimum=policy.reference_min_periods,
    )
    frame["fee_share"] = fee_share
    frame["transaction_density"] = transaction_density
    frame["fee_share_z"] = fee_z
    frame["transaction_density_z"] = density_z
    frame["composite"] = fee_z + policy.transaction_density_weight * density_z
    frame["fee_reference_count"] = fee_reference_count
    frame["density_reference_count"] = density_reference_count
    frame["source_lag_days"] = (
        (frame["available_at"] - frame["observation_date"]).dt.total_seconds() / 86_400.0
    )
    finite = np.isfinite(frame["fee_share_z"]) & np.isfinite(frame["transaction_density_z"])
    frame["eligible"] = (
        finite
        & frame["source_lag_days"].le(policy.maximum_source_lag_days)
        & frame["fee_share_z"].ge(policy.fee_share_z_min)
        & frame["transaction_density_z"].ge(policy.transaction_density_z_min)
        & frame["composite"].ge(policy.composite_min)
    )
    frame["event"] = frame["eligible"] & ~frame["eligible"].shift(1, fill_value=False)
    return frame


def schedule_clock(features: pd.DataFrame, policy: prereg.Policy) -> pd.DataFrame:
    candidates = features.loc[features["event"]].copy()
    candidates["earliest_tradable_open"] = candidates["available_at"].dt.ceil("5min")
    candidates["entry_date"] = candidates["earliest_tradable_open"] + pd.to_timedelta(
        policy.entry_delay_bars * 5, unit="min"
    )
    candidates["exit_date"] = candidates["entry_date"] + pd.to_timedelta(
        policy.hold_bars * 5, unit="min"
    )
    accepted: list[int] = []
    next_entry = pd.Timestamp.min
    for index, row in candidates.sort_values("entry_date").iterrows():
        if row["entry_date"] < TRAIN_START or row["entry_date"] >= SELECTION_END:
            continue
        if row["entry_date"] < next_entry:
            continue
        accepted.append(index)
        next_entry = row["exit_date"]
    clock = candidates.loc[accepted].copy().sort_values("entry_date").reset_index(drop=True)
    clock.insert(0, "policy_id", policy.policy_id)
    clock.insert(1, "side", 1)
    return clock[
        [
            "policy_id",
            "side",
            "observation_date",
            "available_at",
            "earliest_tradable_open",
            "entry_date",
            "exit_date",
            "fee_share",
            "transaction_density",
            "fee_share_z",
            "transaction_density_z",
            "composite",
            "source_lag_days",
            "fee_reference_count",
            "density_reference_count",
        ]
    ]


def support_summary(clock: pd.DataFrame, payload: dict[str, Any]) -> dict[str, Any]:
    entry = pd.to_datetime(clock["entry_date"])
    windows = {
        "train_2021_2022": ("2021-03-01", "2023-01-01"),
        "train_2021": ("2021-03-01", "2022-01-01"),
        "train_2022": ("2022-01-01", "2023-01-01"),
        "selection_2023": ("2023-01-01", "2024-01-01"),
        "selection_2023_h1": ("2023-01-01", "2023-07-01"),
        "selection_2023_h2": ("2023-07-01", "2024-01-01"),
    }
    counts = {
        key: int(((entry >= pd.Timestamp(start)) & (entry < pd.Timestamp(end))).sum())
        for key, (start, end) in windows.items()
    }
    month_counts = entry.dt.to_period("M").value_counts().sort_index()
    maximum_month_share = float(month_counts.max() / len(clock)) if len(clock) else 1.0
    gate = payload["support_freeze_before_returns"]
    checks = {
        "train_total": counts["train_2021_2022"] >= gate["train_2021_2022_nonoverlap_min"],
        "train_2021": counts["train_2021"] >= gate["each_train_year_min"],
        "train_2022": counts["train_2022"] >= gate["each_train_year_min"],
        "selection_2023": counts["selection_2023"] >= gate["selection_2023_min"],
        "selection_2023_h1": counts["selection_2023_h1"] >= gate["selection_2023_h1_min"],
        "selection_2023_h2": counts["selection_2023_h2"] >= gate["selection_2023_h2_min"],
        "month_concentration": maximum_month_share <= gate["maximum_single_month_share"],
    }
    return {
        "counts": counts,
        "maximum_single_month_share": maximum_month_share,
        "monthly_counts": {str(key): int(value) for key, value in month_counts.items()},
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
    blockspace = load_blockspace(payload)
    features = build_features(blockspace, policy)
    clock = schedule_clock(features, policy)
    support = support_summary(clock, payload)
    clock_path = Path(clock_output)
    clock_path.parent.mkdir(parents=True, exist_ok=True)
    clock.to_csv(clock_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    result_core: dict[str, Any] = {
        "protocol_version": "blockspace_fee_confirmation_support_v1",
        "outcomes_opened": False,
        "policy": asdict(policy),
        "preregistration": preregistration,
        "preregistration_sha256": sha256_file(preregistration),
        "preregistration_manifest_hash": payload["manifest_hash"],
        "source": {
            "blockspace": payload["source_contract"]["blockspace"],
            "blockspace_sha256": sha256_file(payload["source_contract"]["blockspace"]),
            "blockspace_manifest_sha256": sha256_file(
                payload["source_contract"]["blockspace_manifest"]
            ),
            "columns_loaded": [
                "observation_date",
                "available_at",
                "FeeTotNtv",
                "IssTotNtv",
                "BlkCnt",
                "TxCnt",
            ],
            "market_or_funding_rows_loaded": 0,
            "first_observation": str(blockspace["observation_date"].min()),
            "last_observation": str(blockspace["observation_date"].max()),
            "rows": int(len(blockspace)),
        },
        "feature_support": {
            "finite_blockspace_rows": int(
                (
                    np.isfinite(features["fee_share_z"])
                    & np.isfinite(features["transaction_density_z"])
                ).sum()
            ),
            "eligible_state_rows": int(features["eligible"].sum()),
            "event_onsets_before_nonoverlap": int(features["event"].sum()),
            "accepted_nonoverlap_events": int(len(clock)),
        },
        "clock": {
            "path": clock_output,
            "sha256": sha256_file(clock_output),
            "frame_hash": frame_hash(clock),
            "rows": int(len(clock)),
            "first_entry": str(clock["entry_date"].min()) if len(clock) else None,
            "last_entry": str(clock["entry_date"].max()) if len(clock) else None,
        },
        "support_gate": support,
        "sealed": ["2024", "2025", "2026_ytd"],
        "failure_action": (
            "reject without loading any post-entry market outcome"
            if not support["passed"]
            else "freeze this exact clock before building the strict evaluator"
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
                "clock": result["clock"],
                "result_hash": result["result_hash"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
