"""Build the price-free NWE-7 weekly feature clock before return labels exist."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_network_weak_signal_ensemble as prereg
from training.build_network_topology_broadening_support import (
    _strict_prior_z,
    frame_hash,
    sha256_file,
)


DEFAULT_OUTPUT = "results/network_weak_signal_ensemble_support_2026-07-17.json"
DEFAULT_CLOCK = "results/network_weak_signal_ensemble_feature_clock_2026-07-17.csv"
SELECTION_END = pd.Timestamp("2024-01-01")


def load_preregistration(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    prereg.validate_manifest(payload)
    if payload["outcomes_opened"] is not False:
        raise RuntimeError("NWE-7 support cannot run after outcomes are opened")
    if payload["policy"] != asdict(prereg.Policy()):
        raise RuntimeError("NWE-7 support policy differs from preregistration")
    return payload


def _verify_manifest(
    *,
    data_path: str,
    data_hash: str,
    manifest_path: str,
    manifest_hash: str,
    expected_columns: list[str],
) -> dict[str, Any]:
    if sha256_file(data_path) != data_hash:
        raise RuntimeError(f"NWE-7 source hash mismatch: {data_path}")
    if sha256_file(manifest_path) != manifest_hash:
        raise RuntimeError(f"NWE-7 source manifest hash mismatch: {manifest_path}")
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("sha256") != data_hash:
        raise RuntimeError(f"NWE-7 manifest does not identify source: {data_path}")
    if manifest.get("columns") != expected_columns:
        raise RuntimeError(f"NWE-7 manifest columns changed: {data_path}")
    return manifest


def load_sources(payload: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = payload["source_contract"]
    blockspace_columns = [
        "observation_date",
        "available_at",
        "FeeTotNtv",
        "IssTotNtv",
        "BlkCnt",
        "TxCnt",
    ]
    network_columns = [
        "observation_date",
        "available_at",
        "AdrActCnt",
        "TxCnt",
        "TxTfrCnt",
    ]
    blockspace_manifest = _verify_manifest(
        data_path=source["blockspace"],
        data_hash=source["blockspace_sha256"],
        manifest_path=source["blockspace_manifest"],
        manifest_hash=source["blockspace_manifest_sha256"],
        expected_columns=blockspace_columns,
    )
    network_manifest = _verify_manifest(
        data_path=source["network"],
        data_hash=source["network_sha256"],
        manifest_path=source["network_manifest"],
        manifest_hash=source["network_manifest_sha256"],
        expected_columns=network_columns,
    )
    blockspace = pd.read_csv(
        source["blockspace"],
        usecols=blockspace_columns,
        parse_dates=["observation_date", "available_at"],
    ).rename(columns={"available_at": "blockspace_available_at"})
    network = pd.read_csv(
        source["network"],
        usecols=network_columns,
        parse_dates=["observation_date", "available_at"],
    ).rename(
        columns={
            "available_at": "network_available_at",
            "TxCnt": "network_TxCnt",
        }
    )
    for name, frame in (("blockspace", blockspace), ("network", network)):
        if frame["observation_date"].duplicated().any():
            raise RuntimeError(f"NWE-7 {name} dates must be unique")
        frame.sort_values("observation_date", inplace=True)
        frame.reset_index(drop=True, inplace=True)
        if frame["observation_date"].max() >= SELECTION_END:
            raise RuntimeError(f"NWE-7 {name} source crosses sealed 2024 boundary")
    merged = blockspace.merge(network, on="observation_date", how="inner", validate="one_to_one")
    if merged.empty:
        raise RuntimeError("NWE-7 sources have no common observation")
    merged["available_at"] = merged[
        ["blockspace_available_at", "network_available_at"]
    ].max(axis=1)
    if (
        merged["available_at"]
        < merged["observation_date"] + pd.Timedelta(days=1)
    ).any():
        raise RuntimeError("NWE-7 source became available before daily completion")
    numeric = [
        "FeeTotNtv",
        "IssTotNtv",
        "BlkCnt",
        "TxCnt",
        "AdrActCnt",
        "network_TxCnt",
        "TxTfrCnt",
    ]
    for column in numeric:
        merged[column] = pd.to_numeric(merged[column], errors="raise")
        if (~np.isfinite(merged[column]) | (merged[column] <= 0.0)).any():
            raise RuntimeError(f"NWE-7 source must be finite and positive: {column}")
    if not np.array_equal(
        merged["TxCnt"].to_numpy(float), merged["network_TxCnt"].to_numpy(float)
    ):
        raise RuntimeError("NWE-7 Coin Metrics TxCnt sources disagree")
    return merged, {
        "blockspace_manifest_rows": int(blockspace_manifest["rows"]),
        "network_manifest_rows": int(network_manifest["rows"]),
        "common_rows": int(len(merged)),
        "first_common_observation": str(merged["observation_date"].min()),
        "last_common_observation": str(merged["observation_date"].max()),
    }


def _z_pair(
    level: np.ndarray,
    available: np.ndarray,
    policy: prereg.Policy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    change = pd.Series(level).diff(policy.feature_change_days).to_numpy(float)
    level_z, level_count = _strict_prior_z(
        level,
        available,
        reference_days=policy.reference_days,
        minimum=policy.reference_min_periods,
    )
    change_z, change_count = _strict_prior_z(
        change,
        available,
        reference_days=policy.reference_days,
        minimum=policy.reference_min_periods,
    )
    return level_z, change_z, level_count, change_count


def build_daily_features(sources: pd.DataFrame, policy: prereg.Policy) -> pd.DataFrame:
    frame = sources[["observation_date", "available_at"]].copy()
    raw = {
        "fee_share": np.log(
            sources["FeeTotNtv"].to_numpy(float) / sources["IssTotNtv"].to_numpy(float)
        ),
        "transaction_density": np.log(
            sources["TxCnt"].to_numpy(float) / sources["BlkCnt"].to_numpy(float)
        ),
        "breadth": np.log(
            sources["AdrActCnt"].to_numpy(float) / sources["TxTfrCnt"].to_numpy(float)
        ),
        "fanout": np.log(
            sources["TxTfrCnt"].to_numpy(float) / sources["TxCnt"].to_numpy(float)
        ),
    }
    available = frame["available_at"].to_numpy(dtype="datetime64[ns]")
    minimum_counts: list[np.ndarray] = []
    for name, values in raw.items():
        level_z, change_z, level_count, change_count = _z_pair(values, available, policy)
        frame[f"{name}_level_z"] = np.clip(level_z, -policy.feature_clip, policy.feature_clip)
        frame[f"{name}_change_z"] = np.clip(change_z, -policy.feature_clip, policy.feature_clip)
        minimum_counts.extend([level_count, change_count])
    frame["minimum_reference_count"] = np.min(np.column_stack(minimum_counts), axis=1)
    return frame


def build_feature_clock(daily: pd.DataFrame, policy: prereg.Policy) -> pd.DataFrame:
    decisions = pd.date_range(
        pd.Timestamp(policy.fit_history_start),
        pd.Timestamp("2023-12-31"),
        freq="W-MON",
    ) + pd.Timedelta(hours=policy.decision_hour_utc, minutes=policy.decision_minute_utc)
    rows: list[dict[str, Any]] = []
    observation_dates = daily["observation_date"].to_numpy(dtype="datetime64[ns]")
    feature_columns = list(prereg.FEATURE_COLUMNS)
    for decision in decisions:
        # Coin Metrics daily rows use beginning-of-interval timestamps, so the
        # current UTC day is incomplete and cannot be the source observation.
        cutoff = decision.floor("D")
        source_position = int(
            np.searchsorted(observation_dates, np.datetime64(cutoff), side="left") - 1
        )
        if source_position < 0:
            continue
        source = daily.iloc[source_position]
        feature_available_at = pd.Timestamp(source["available_at"])
        observation_age_days = (
            decision - pd.Timestamp(source["observation_date"])
        ).total_seconds() / 86_400.0
        values = source[feature_columns].to_numpy(float)
        finite = bool(np.isfinite(values).all())
        prediction_eligible = bool(
            decision >= pd.Timestamp(policy.prediction_start)
            and decision < SELECTION_END
            and decision + pd.Timedelta(minutes=5 * policy.entry_delay_bars)
            + pd.Timedelta(minutes=5 * policy.hold_bars)
            < SELECTION_END
            and feature_available_at <= decision
            and observation_age_days <= policy.maximum_observation_age_days
            and finite
        )
        row: dict[str, Any] = {
            "policy_id": policy.policy_id,
            "decision_date": decision,
            "entry_date": decision + pd.Timedelta(minutes=5 * policy.entry_delay_bars),
            "exit_date": decision
            + pd.Timedelta(minutes=5 * (policy.entry_delay_bars + policy.hold_bars)),
            "source_observation_date": pd.Timestamp(source["observation_date"]),
            "feature_available_at": feature_available_at,
            "observation_age_days": observation_age_days,
            "minimum_reference_count": int(source["minimum_reference_count"]),
            "all_features_finite": finite,
            "prediction_eligible": prediction_eligible,
        }
        row.update({name: float(value) for name, value in zip(feature_columns, values)})
        rows.append(row)
    return pd.DataFrame(rows)


def support_summary(clock: pd.DataFrame, payload: dict[str, Any]) -> dict[str, Any]:
    eligible = clock.loc[clock["prediction_eligible"].astype(bool)].copy()
    decision = pd.to_datetime(eligible["decision_date"])
    windows = {
        "train_2021_2022": ("2021-03-01", "2023-01-01"),
        "train_2021": ("2021-03-01", "2022-01-01"),
        "train_2022": ("2022-01-01", "2023-01-01"),
        "selection_2023": ("2023-01-01", "2024-01-01"),
        "selection_2023_h1": ("2023-01-01", "2023-07-01"),
        "selection_2023_h2": ("2023-07-01", "2024-01-01"),
    }
    counts = {
        key: int(((decision >= pd.Timestamp(start)) & (decision < pd.Timestamp(end))).sum())
        for key, (start, end) in windows.items()
    }
    first_prediction = pd.Timestamp(payload["policy"]["prediction_start"])
    history = clock.loc[
        (pd.to_datetime(clock["decision_date"]) < first_prediction)
        & (pd.to_datetime(clock["feature_available_at"]) <= first_prediction)
        & (pd.to_datetime(clock["exit_date"]) <= first_prediction)
        & clock["all_features_finite"].astype(bool)
    ]
    gate = payload["support_freeze_before_labels"]
    checks = {
        "train_total": counts["train_2021_2022"] >= gate["train_2021_2022_candidate_weeks_min"],
        "train_2021": counts["train_2021"] >= gate["train_2021_candidate_weeks_min"],
        "train_2022": counts["train_2022"] >= gate["train_2022_candidate_weeks_min"],
        "selection_2023": counts["selection_2023"] >= gate["selection_2023_candidate_weeks_min"],
        "selection_2023_h1": counts["selection_2023_h1"] >= gate[
            "selection_2023_h1_candidate_weeks_min"
        ],
        "selection_2023_h2": counts["selection_2023_h2"] >= gate[
            "selection_2023_h2_candidate_weeks_min"
        ],
        "initial_training_history": len(history) >= payload["policy"]["minimum_train_samples"],
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
    sources, source_summary = load_sources(payload)
    daily = build_daily_features(sources, policy)
    clock = build_feature_clock(daily, policy)
    support = support_summary(clock, payload)
    clock_path = Path(clock_output)
    clock_path.parent.mkdir(parents=True, exist_ok=True)
    clock.to_csv(clock_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    result_core: dict[str, Any] = {
        "protocol_version": "network_weak_signal_ensemble_support_v1",
        "outcomes_opened": False,
        "policy": asdict(policy),
        "feature_columns": list(prereg.FEATURE_COLUMNS),
        "preregistration": preregistration,
        "preregistration_sha256": sha256_file(preregistration),
        "preregistration_manifest_hash": payload["manifest_hash"],
        "source": {
            **source_summary,
            "blockspace_sha256": sha256_file(payload["source_contract"]["blockspace"]),
            "network_sha256": sha256_file(payload["source_contract"]["network"]),
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
            "frame_hash": frame_hash(daily),
            "first_observation": str(daily["observation_date"].min()),
            "last_observation": str(daily["observation_date"].max()),
        },
        "feature_clock": {
            "path": clock_output,
            "sha256": sha256_file(clock_output),
            "frame_hash": frame_hash(clock),
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
