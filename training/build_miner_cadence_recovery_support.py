"""Build the outcome-blind MCR-7 feature and entry clock."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_miner_cadence_recovery as prereg
from training.build_network_topology_broadening_support import (
    _strict_prior_z,
    frame_hash,
    sha256_file,
)


DEFAULT_OUTPUT = "results/miner_cadence_recovery_support_2026-07-17.json"
DEFAULT_CLOCK = "results/miner_cadence_recovery_clock_2026-07-17.csv"
TRAIN_START = pd.Timestamp("2021-03-01")
SELECTION_END = pd.Timestamp("2024-01-01")


def load_preregistration(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    prereg.validate_manifest(payload)
    if payload["outcomes_opened"] is not False:
        raise RuntimeError("MCR-7 support cannot run after outcomes are opened")
    if payload["policy"] != asdict(prereg.Policy()):
        raise RuntimeError("MCR-7 support policy differs from preregistration")
    return payload


def load_miner_security(payload: dict[str, Any]) -> pd.DataFrame:
    source = payload["source_contract"]
    if sha256_file(source["miner_security"]) != source["miner_security_sha256"]:
        raise RuntimeError("MCR-7 miner-security source hash mismatch")
    if (
        sha256_file(source["miner_security_manifest"])
        != source["miner_security_manifest_sha256"]
    ):
        raise RuntimeError("MCR-7 source-manifest hash mismatch")
    manifest = json.loads(Path(source["miner_security_manifest"]).read_text())
    if manifest.get("sha256") != source["miner_security_sha256"]:
        raise RuntimeError("MCR-7 source manifest does not identify the frozen data")
    expected_columns = [
        "observation_date",
        "available_at",
        "HashRate",
        "IssTotNtv",
        "FeeTotNtv",
        "BlkCnt",
    ]
    if manifest.get("columns") != expected_columns:
        raise RuntimeError("MCR-7 source columns differ from the frozen manifest")
    columns = source["columns_allowed"]
    frame = pd.read_csv(
        source["miner_security"],
        usecols=columns,
        parse_dates=["observation_date", "available_at"],
    )
    if len(frame) != int(source["rows"]):
        raise RuntimeError("MCR-7 source row count drifted")
    if frame["observation_date"].duplicated().any():
        raise RuntimeError("MCR-7 observation dates must be unique")
    frame = frame.sort_values("observation_date").reset_index(drop=True)
    if frame["observation_date"].max() >= SELECTION_END:
        raise RuntimeError("MCR-7 support source crossed the sealed 2024 boundary")
    if (frame["available_at"] < frame["observation_date"] + pd.Timedelta(days=1)).any():
        raise RuntimeError("MCR-7 source became available before daily completion")
    for column in ("HashRate", "BlkCnt"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if (~np.isfinite(frame[column]) | (frame[column] <= 0.0)).any():
            raise RuntimeError(f"MCR-7 source must be finite and positive: {column}")
    return frame


def causal_hash_change(
    log_hash_rate: np.ndarray,
    available_at: np.ndarray,
    days: int,
) -> np.ndarray:
    values = np.full(len(log_hash_rate), np.nan, dtype=float)
    for index in range(days, len(log_hash_rate)):
        prior = index - days
        if available_at[prior] < available_at[index]:
            values[index] = log_hash_rate[index] - log_hash_rate[prior]
    return values


def causal_cadence(
    log_blocks: np.ndarray,
    available_at: np.ndarray,
    *,
    short_days: int,
    reference_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    short = np.full(len(log_blocks), np.nan, dtype=float)
    reference = np.full(len(log_blocks), np.nan, dtype=float)
    gap = np.full(len(log_blocks), np.nan, dtype=float)
    for index in range(reference_days, len(log_blocks)):
        current_available = available_at[index]
        short_indices = np.arange(index - short_days + 1, index + 1)
        prior_indices = np.arange(index - reference_days, index)
        if not np.all(available_at[short_indices[:-1]] < current_available):
            continue
        if not np.all(available_at[prior_indices] < current_available):
            continue
        short[index] = float(np.mean(log_blocks[short_indices]))
        reference[index] = float(np.mean(log_blocks[prior_indices]))
        gap[index] = short[index] - reference[index]
    return short, reference, gap


def causal_recent_stress(
    zscore: np.ndarray,
    available_at: np.ndarray,
    *,
    lookback_days: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    present = np.zeros(len(zscore), dtype=bool)
    minimum = np.full(len(zscore), np.nan, dtype=float)
    for index in range(len(zscore)):
        start = max(0, index - lookback_days)
        prior = np.arange(start, index)
        causal = prior[
            np.isfinite(zscore[prior]) & (available_at[prior] < available_at[index])
        ]
        if len(causal) == 0:
            continue
        minimum[index] = float(np.min(zscore[causal]))
        present[index] = minimum[index] <= threshold
    return present, minimum


def build_features(frame: pd.DataFrame, policy: prereg.Policy) -> pd.DataFrame:
    features = frame[["observation_date", "available_at", "HashRate", "BlkCnt"]].copy()
    available = features["available_at"].to_numpy(dtype="datetime64[ns]")
    log_hash = np.log(features["HashRate"].to_numpy(float))
    log_blocks = np.log(features["BlkCnt"].to_numpy(float))
    hash_change = causal_hash_change(log_hash, available, policy.hash_change_days)
    hash_change_z, reference_count = _strict_prior_z(
        hash_change,
        available,
        reference_days=policy.reference_days,
        minimum=policy.reference_min_periods,
    )
    cadence_short, cadence_reference, cadence_gap = causal_cadence(
        log_blocks,
        available,
        short_days=policy.cadence_short_days,
        reference_days=policy.cadence_reference_days,
    )
    recent_stress, recent_stress_min = causal_recent_stress(
        hash_change_z,
        available,
        lookback_days=policy.stress_lookback_days,
        threshold=policy.stress_z_max,
    )
    prior_z = np.full(len(features), np.nan, dtype=float)
    for index in range(1, len(features)):
        if available[index - 1] < available[index]:
            prior_z[index] = hash_change_z[index - 1]
    source_lag_days = (
        (features["available_at"] - features["observation_date"]).dt.total_seconds()
        / 86_400.0
    )
    recovery_cross = (
        np.isfinite(hash_change_z)
        & np.isfinite(prior_z)
        & (hash_change_z >= policy.recovery_z_min)
        & (prior_z < policy.recovery_z_min)
    )
    eligible = (
        recovery_cross
        & recent_stress
        & np.isfinite(cadence_gap)
        & (cadence_gap >= 0.0)
        & source_lag_days.le(policy.maximum_source_lag_days).to_numpy()
    )
    features["hash_change"] = hash_change
    features["hash_change_z"] = hash_change_z
    features["prior_hash_change_z"] = prior_z
    features["hash_reference_count"] = reference_count
    features["recent_stress_min_z"] = recent_stress_min
    features["recent_stress"] = recent_stress
    features["cadence_short"] = cadence_short
    features["cadence_reference"] = cadence_reference
    features["cadence_gap"] = cadence_gap
    features["source_lag_days"] = source_lag_days
    features["recovery_cross"] = recovery_cross
    features["eligible"] = eligible
    features["event"] = eligible
    return features


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
            "HashRate",
            "BlkCnt",
            "hash_change",
            "hash_change_z",
            "prior_hash_change_z",
            "recent_stress_min_z",
            "cadence_short",
            "cadence_reference",
            "cadence_gap",
            "source_lag_days",
            "hash_reference_count",
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
        "monthly_counts": {str(key): int(value) for key, value in month_counts.items()},
        "maximum_single_month_share": maximum_month_share,
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
    source = load_miner_security(payload)
    features = build_features(source, policy)
    clock = schedule_clock(features, policy)
    support = support_summary(clock, payload)
    clock_path = Path(clock_output)
    clock_path.parent.mkdir(parents=True, exist_ok=True)
    clock.to_csv(clock_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    result_core: dict[str, Any] = {
        "protocol_version": "miner_cadence_recovery_support_v1",
        "outcomes_opened": False,
        "policy": asdict(policy),
        "preregistration": preregistration,
        "preregistration_sha256": sha256_file(preregistration),
        "preregistration_manifest_hash": payload["manifest_hash"],
        "source": {
            "miner_security": payload["source_contract"]["miner_security"],
            "miner_security_sha256": sha256_file(
                payload["source_contract"]["miner_security"]
            ),
            "miner_security_manifest_sha256": sha256_file(
                payload["source_contract"]["miner_security_manifest"]
            ),
            "columns_loaded": payload["source_contract"]["columns_allowed"],
            "market_or_funding_rows_loaded": 0,
            "rows": int(len(source)),
            "first_observation": str(source["observation_date"].min()),
            "last_observation": str(source["observation_date"].max()),
        },
        "feature_support": {
            "finite_hash_z_rows": int(np.isfinite(features["hash_change_z"]).sum()),
            "recovery_cross_rows": int(features["recovery_cross"].sum()),
            "eligible_events_before_nonoverlap": int(features["event"].sum()),
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
        "sealed": ["all_post_entry_market_outcomes", "2024", "2025", "2026_ytd"],
        "failure_action": (
            "reject without loading any post-entry market outcome"
            if not support["passed"]
            else "freeze this exact clock before implementing the strict evaluator"
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


def markdown(result: dict[str, Any]) -> str:
    support = result["support_gate"]
    status = "passed" if support["passed"] else "rejected"
    return f"""# MCR-7 outcome-blind support freeze

Support decision: **{status} before any post-entry market return was loaded**.

- accepted nonoverlap events: {result['clock']['rows']}
- train 2021-2022: {support['counts']['train_2021_2022']}
- train 2021: {support['counts']['train_2021']}
- train 2022: {support['counts']['train_2022']}
- selection 2023: {support['counts']['selection_2023']}
- selection H1/H2: {support['counts']['selection_2023_h1']} / {support['counts']['selection_2023_h2']}
- maximum month share: {support['maximum_single_month_share']:.4f}
- market/funding rows loaded: {result['source']['market_or_funding_rows_loaded']}

The exact entry clock and all source/protocol hashes are frozen. No price,
funding, return, CAGR, drawdown, or 2024+ source row was used in this decision.

Result hash: `{result['result_hash']}`
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", default=prereg.DEFAULT_OUTPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--clock-output", default=DEFAULT_CLOCK)
    parser.add_argument(
        "--docs", default="docs/miner-cadence-recovery-support-2026-07-17.md"
    )
    args = parser.parse_args()
    result = run(
        preregistration=args.preregistration,
        output=args.output,
        clock_output=args.clock_output,
    )
    docs_path = Path(args.docs)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(markdown(result))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
