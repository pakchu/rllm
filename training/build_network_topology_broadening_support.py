"""Build the outcome-blind NTB-7 support clock from Coin Metrics only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_network_topology_broadening as prereg


DEFAULT_OUTPUT = "results/network_topology_broadening_support_2026-07-17.json"
DEFAULT_CLOCK = "results/network_topology_broadening_clock_2026-07-17.csv"
SELECTION_END = pd.Timestamp("2024-01-01")
TRAIN_START = pd.Timestamp("2021-03-01")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_hash(frame: pd.DataFrame) -> str:
    canonical = frame.copy()
    for column in canonical:
        if pd.api.types.is_datetime64_any_dtype(canonical[column]):
            canonical[column] = canonical[column].astype("datetime64[ns]").astype("int64")
    digest = pd.util.hash_pandas_object(canonical, index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(digest.tobytes()).hexdigest()


def load_preregistration(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    prereg.validate_manifest(payload)
    if payload["outcomes_opened"] is not False:
        raise RuntimeError("NTB-7 support cannot run after outcomes are opened")
    if payload["policy"] != asdict(prereg.Policy()):
        raise RuntimeError("NTB-7 support policy differs from preregistration")
    return payload


def load_network(payload: dict[str, Any]) -> pd.DataFrame:
    source = payload["source_contract"]
    if sha256_file(source["network"]) != source["network_sha256"]:
        raise RuntimeError("NTB-7 network source hash mismatch")
    if sha256_file(source["network_manifest"]) != source["network_manifest_sha256"]:
        raise RuntimeError("NTB-7 network manifest hash mismatch")

    manifest = json.loads(Path(source["network_manifest"]).read_text())
    if manifest.get("sha256") != source["network_sha256"]:
        raise RuntimeError("network manifest does not identify frozen network source")
    expected_columns = [
        "observation_date",
        "available_at",
        "AdrActCnt",
        "TxCnt",
        "TxTfrCnt",
    ]
    if manifest.get("columns") != expected_columns:
        raise RuntimeError("network manifest columns differ from NTB-7 contract")

    frame = pd.read_csv(
        source["network"],
        usecols=expected_columns,
        parse_dates=["observation_date", "available_at"],
    )
    if len(frame) != int(source["network_rows"]):
        raise RuntimeError("network row count differs from NTB-7 preregistration")
    if frame["observation_date"].duplicated().any():
        raise RuntimeError("network observation dates must be unique")
    frame = frame.sort_values("observation_date").reset_index(drop=True)
    if not frame["observation_date"].is_monotonic_increasing:
        raise RuntimeError("network observation dates must be monotonic")
    if frame["observation_date"].max() >= SELECTION_END:
        raise RuntimeError("NTB-7 support source crosses the sealed 2024 boundary")
    if (frame["available_at"] < frame["observation_date"] + pd.Timedelta(days=1)).any():
        raise RuntimeError("network metric became available before daily completion")
    for column in ("AdrActCnt", "TxCnt", "TxTfrCnt"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if (~np.isfinite(frame[column]) | (frame[column] <= 0.0)).any():
            raise RuntimeError(f"network column must be finite and positive: {column}")
    return frame


def _strict_prior_z(
    values: np.ndarray,
    available_at: np.ndarray,
    *,
    reference_days: int,
    minimum: int,
) -> tuple[np.ndarray, np.ndarray]:
    zscore = np.full(len(values), np.nan, dtype=float)
    reference_count = np.zeros(len(values), dtype=np.int64)
    for index in range(len(values)):
        if not np.isfinite(values[index]):
            continue
        prior = np.arange(index)
        if len(prior) == 0:
            continue
        causal = prior[
            np.isfinite(values[prior]) & (available_at[prior] < available_at[index])
        ]
        if len(causal) > reference_days:
            causal = causal[-reference_days:]
        reference_count[index] = len(causal)
        if len(causal) < minimum:
            continue
        reference = values[causal]
        std = float(np.std(reference, ddof=1))
        if not math.isfinite(std) or std <= 0.0:
            continue
        zscore[index] = (values[index] - float(np.mean(reference))) / std
    return zscore, reference_count


def build_features(network: pd.DataFrame, policy: prereg.Policy) -> pd.DataFrame:
    frame = network[["observation_date", "available_at"]].copy()
    fanout = np.log(network["TxTfrCnt"].to_numpy(float) / network["TxCnt"].to_numpy(float))
    breadth = np.log(network["AdrActCnt"].to_numpy(float) / network["TxTfrCnt"].to_numpy(float))
    fanout_change = pd.Series(fanout).diff(policy.change_days).to_numpy(float)
    breadth_change = pd.Series(breadth).diff(policy.change_days).to_numpy(float)
    available = frame["available_at"].to_numpy(dtype="datetime64[ns]")
    fanout_z, fanout_reference_count = _strict_prior_z(
        fanout_change,
        available,
        reference_days=policy.reference_days,
        minimum=policy.reference_min_periods,
    )
    breadth_z, breadth_reference_count = _strict_prior_z(
        breadth_change,
        available,
        reference_days=policy.reference_days,
        minimum=policy.reference_min_periods,
    )
    frame["fanout"] = fanout
    frame["breadth"] = breadth
    frame["fanout_change"] = fanout_change
    frame["breadth_change"] = breadth_change
    frame["fanout_z"] = fanout_z
    frame["breadth_z"] = breadth_z
    frame["composite"] = breadth_z - fanout_z
    frame["fanout_reference_count"] = fanout_reference_count
    frame["breadth_reference_count"] = breadth_reference_count
    frame["source_lag_days"] = (
        (frame["available_at"] - frame["observation_date"]).dt.total_seconds() / 86_400.0
    )
    finite = np.isfinite(frame["fanout_z"]) & np.isfinite(frame["breadth_z"])
    frame["eligible"] = (
        finite
        & frame["source_lag_days"].le(policy.maximum_source_lag_days)
        & frame["breadth_z"].ge(policy.breadth_z_min)
        & frame["fanout_z"].le(policy.fanout_z_max)
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
    columns = [
        "policy_id",
        "side",
        "observation_date",
        "available_at",
        "earliest_tradable_open",
        "entry_date",
        "exit_date",
        "fanout",
        "breadth",
        "fanout_change",
        "breadth_change",
        "fanout_z",
        "breadth_z",
        "composite",
        "source_lag_days",
        "fanout_reference_count",
        "breadth_reference_count",
    ]
    return clock[columns]


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
    network = load_network(payload)
    features = build_features(network, policy)
    clock = schedule_clock(features, policy)
    support = support_summary(clock, payload)

    clock_path = Path(clock_output)
    clock_path.parent.mkdir(parents=True, exist_ok=True)
    clock.to_csv(clock_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    result_core: dict[str, Any] = {
        "protocol_version": "network_topology_broadening_support_v1",
        "outcomes_opened": False,
        "policy": asdict(policy),
        "preregistration": preregistration,
        "preregistration_sha256": sha256_file(preregistration),
        "preregistration_manifest_hash": payload["manifest_hash"],
        "source": {
            "network": payload["source_contract"]["network"],
            "network_sha256": sha256_file(payload["source_contract"]["network"]),
            "network_manifest_sha256": sha256_file(
                payload["source_contract"]["network_manifest"]
            ),
            "columns_loaded": [
                "observation_date",
                "available_at",
                "AdrActCnt",
                "TxCnt",
                "TxTfrCnt",
            ],
            "market_or_funding_rows_loaded": 0,
            "first_observation": str(network["observation_date"].min()),
            "last_observation": str(network["observation_date"].max()),
            "rows": int(len(network)),
        },
        "feature_support": {
            "finite_topology_rows": int(
                (np.isfinite(features["fanout_z"]) & np.isfinite(features["breadth_z"])).sum()
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
