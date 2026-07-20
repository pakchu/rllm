"""Freeze and run outcome-blind source support for DFIA-72.

The preregistration stage opens neither complete source incidence nor BTC
market outcomes.  The support stage may read only the ignored pre-2024
Deribit funding-memory source and writes an event clock only when every frozen
coverage, dispersion, nonoverlap, and side-balance gate passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.download_deribit_btc_perpetual_funding_hourly import (
    Config as SourceConfig,
    ENDPOINT,
    INSTRUMENT,
    OFFICIAL_DOCS,
    OFFICIAL_USAGE_POLICY,
    OUTPUT_COLUMNS,
)


POLICY_ID = "DFIA-72"
SOURCE_DATA = Path("data/deribit_btc_perpetual_funding_2019_2023.csv.gz")
SOURCE_MANIFEST = Path(
    "results/deribit_btc_perpetual_funding_source_manifest_2026-07-20.json"
)
SOURCE_DECISION = Path(
    "docs/deribit-funding-impulse-absorption-mechanism-decision-2026-07-20.md"
)
SOURCE_DECISION_SHA256 = (
    "3aa1e0dbea5afa55b82e3693da945c384efe2187a971dfe4f8aa52d5677e7281"
)
SOURCE_DOWNLOADER = Path(
    "training/download_deribit_btc_perpetual_funding_hourly.py"
)
SOURCE_DOWNLOADER_SHA256 = (
    "ef166913bf398282056a046e15985e2b8f2a81d8f338376fcf5ee2f8cc21d00d"
)
LOWER_BOUND_CORRECTION = Path(
    "docs/deribit-funding-source-lower-bound-operational-correction-2026-07-20.md"
)
LOWER_BOUND_CORRECTION_SHA256 = (
    "ee4ae21d3785b9de7c37ba35f32cafb33b116f139c6b8cec32a36fc233d89e71"
)
ORIGINAL_PREREGISTRATION_ARTIFACT_HASH = (
    "911afce424443a7cd7e23b852357ce4bc1f6d0e837377b582515c8e88f6e7f41"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/deribit-funding-impulse-absorption-support-preregistration-2026-07-20.md"
)
PREREGISTRATION_SOURCE = Path(
    "training/preregister_deribit_funding_impulse_absorption_support.py"
)
SOURCE_COLUMNS = list(OUTPUT_COLUMNS)
EXPECTED_SOURCE_SEMANTICS = {
    "interest_1h": "Deribit one-hour perpetual interest rate",
    "interest_8h": "Deribit eight-hour perpetual interest rate",
    "index_price": "Deribit BTC index price at the hourly row",
    "prev_index_price": (
        "Deribit BTC index price at the prior hourly point; null only at the "
        "exact source lower boundary where prior history is unavailable"
    ),
}
EXPECTED_OUTCOME_BOUNDARY = {
    "binance_market_rows_loaded": 0,
    "binance_funding_rows_loaded": 0,
    "return_or_pnl_fields": 0,
    "post_2023_source_rows_loaded": 0,
    "raw_deribit_responses_persisted": False,
}
EVENT_COLUMNS = [
    "policy_id",
    "split",
    "source_timestamp",
    "feature_available_at",
    "earliest_observable_open",
    "entry_time",
    "exit_time",
    "side",
    "funding_impulse",
    "index_return_1h",
    "funding_impulse_z",
    "index_return_z",
    "impulse_reference_count",
    "index_reference_count",
    "memory_chain_ready",
]


@dataclass(frozen=True)
class Config:
    preregistration_output: str = (
        "results/deribit_funding_impulse_absorption_support_"
        "preregistration_2026-07-20.json"
    )
    support_output: str = (
        "results/deribit_funding_impulse_absorption_support_2026-07-20.json"
    )
    event_clock_output: str = (
        "results/deribit_funding_impulse_absorption_event_clock_2026-07-20.json"
    )
    reference_lookback_hours: int = 720
    minimum_prior_observations: int = 360
    standard_deviation_ddof: int = 0
    funding_impulse_z_threshold: float = 1.25
    index_response_z_boundary: float = 0.0
    minimum_contiguous_memory_hours: int = 8
    source_availability_delay_minutes: int = 5
    entry_latency_bars: int = 1
    hold_bars: int = 72
    train_start: str = "2020-01-01T00:00:00Z"
    train_end_exclusive: str = "2023-01-01T00:00:00Z"
    test_start: str = "2023-01-01T00:00:00Z"
    test_end_exclusive: str = "2024-01-01T00:00:00Z"
    source_first_exact: str = "2019-04-30T10:00:00Z"
    source_last_exact: str = "2023-12-31T23:00:00Z"
    minimum_source_coverage_ratio: float = 0.98
    minimum_source_month_coverage_ratio: float = 0.95
    maximum_source_gap_hours: int = 24
    minimum_total: int = 300
    minimum_train: int = 200
    minimum_each_train_year: int = 50
    minimum_test: int = 80
    minimum_each_test_half: int = 30
    minimum_each_quarter: int = 10
    minimum_active_months: int = 44
    minimum_side_share: float = 0.25
    maximum_month_share: float = 0.08


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _utc(value: Any, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"DFIA-72 {name} is not a timestamp") from exc
    if timestamp.tzinfo is None:
        raise RuntimeError(f"DFIA-72 {name} lacks an explicit timezone")
    return timestamp.tz_convert("UTC")


def _validate_config(cfg: Config) -> None:
    expected = Config(
        preregistration_output=cfg.preregistration_output,
        support_output=cfg.support_output,
        event_clock_output=cfg.event_clock_output,
    )
    if cfg != expected:
        raise ValueError("DFIA-72 signal and support configuration is frozen")
    anchors = {
        SOURCE_DECISION: SOURCE_DECISION_SHA256,
        SOURCE_DOWNLOADER: SOURCE_DOWNLOADER_SHA256,
        LOWER_BOUND_CORRECTION: LOWER_BOUND_CORRECTION_SHA256,
    }
    for path, expected_sha in anchors.items():
        if sha256_file(path) != expected_sha:
            raise ValueError(f"DFIA-72 frozen source anchor mismatch: {path}")


def protocol(cfg: Config) -> dict[str, Any]:
    return {
        "policy_id": POLICY_ID,
        "support_only": True,
        "outcomes_opened": False,
        "source": {
            "endpoint": ENDPOINT,
            "official_docs": OFFICIAL_DOCS,
            "official_usage_policy": OFFICIAL_USAGE_POLICY,
            "instrument": INSTRUMENT,
            "interval": [SourceConfig.start, SourceConfig.end_exclusive],
            "aggregate": str(SOURCE_DATA),
            "aggregate_sha256": "pending_outcome_blind_download",
            "manifest": str(SOURCE_MANIFEST),
            "manifest_hash": "pending_outcome_blind_download",
            "fields_used": SOURCE_COLUMNS,
            "fields_explicitly_unused": [
                "binance_open",
                "binance_high",
                "binance_low",
                "binance_close",
                "binance_funding",
                "future_return",
                "held_path",
                "pnl",
            ],
            "raw_responses_persisted": False,
            "rows_at_or_after_2024_loaded": False,
            "lower_boundary_prev_index_price": (
                "exactly one preserved null at the first source row; never "
                "imputed or used as a return"
            ),
        },
        "clock": {
            "timezone": "UTC",
            "historical_availability": (
                f"source timestamp + {cfg.source_availability_delay_minutes} "
                "minutes"
            ),
            "live_availability": (
                "actual first successful observation when later; never backdate"
            ),
            "earliest_observable_open": (
                "ceiling of feature availability to a five-minute boundary"
            ),
            "entry_latency_bars": cfg.entry_latency_bars,
            "entry": "one additional completed five-minute bar",
            "hold_bars": cfg.hold_bars,
            "hold_minutes": cfg.hold_bars * 5,
            "gap_rule": (
                f"require {cfg.minimum_contiguous_memory_hours} contiguous "
                "hourly source rows ending at the current row"
            ),
        },
        "feature": {
            "trailing_hourly_mean": "interest_8h / 8",
            "funding_impulse": "interest_1h - interest_8h / 8",
            "index_return_1h": "log(index_price / prev_index_price)",
            "reference": (
                f"memory-valid observations in [T-{cfg.reference_lookback_hours}h,T) "
                "with strictly prior availability; require "
                f"{cfg.minimum_prior_observations}"
            ),
            "standard_deviation_ddof": cfg.standard_deviation_ddof,
            "short": (
                f"funding_impulse_z >= {cfg.funding_impulse_z_threshold}, "
                "raw index_return_1h <= 0, and index_return_z <= 0"
            ),
            "long": (
                f"funding_impulse_z <= -{cfg.funding_impulse_z_threshold}, "
                "raw index_return_1h >= 0, and index_return_z >= 0"
            ),
            "threshold_grid": False,
            "sign_search": False,
            "repair_after_incidence": False,
        },
        "scheduler": {
            "candidate_clock": "every qualifying hourly row while flat",
            "transition_onset_only": False,
            "nonoverlap": (
                "chronological greedy within each split; next entry may equal "
                "the prior accepted exit"
            ),
            "complete_split_containment": True,
            "train": [cfg.train_start, cfg.train_end_exclusive],
            "test": [cfg.test_start, cfg.test_end_exclusive],
        },
        "source_gate": {
            "first_exact": cfg.source_first_exact,
            "last_exact": cfg.source_last_exact,
            "expected_hourly_timestamps": 40_958,
            "eligible_months": ["2020-01", "2023-12"],
            "minimum_overall_coverage_ratio": (
                cfg.minimum_source_coverage_ratio
            ),
            "minimum_each_eligible_month_coverage_ratio": (
                cfg.minimum_source_month_coverage_ratio
            ),
            "maximum_adjacent_observed_timestamp_delta_hours": (
                cfg.maximum_source_gap_hours
            ),
            "maximum_consecutive_missing_timestamps": (
                cfg.maximum_source_gap_hours - 1
            ),
            "memory_invariant_required": True,
            "lower_boundary_null_prev_index_price_exact": 1,
        },
        "candidate_support_gate": {
            "minimum_total": cfg.minimum_total,
            "minimum_train": cfg.minimum_train,
            "minimum_each_train_year": cfg.minimum_each_train_year,
            "minimum_test": cfg.minimum_test,
            "minimum_each_test_half": cfg.minimum_each_test_half,
            "minimum_each_quarter": cfg.minimum_each_quarter,
            "minimum_active_months": cfg.minimum_active_months,
            "minimum_side_share_all_train_test": cfg.minimum_side_share,
            "maximum_month_share": cfg.maximum_month_share,
            "count_clock": (
                "accepted entry timestamp after reference readiness, continuity, "
                "latency, split containment, and greedy nonoverlap"
            ),
            "failure_action": (
                "reject before BTC market outcomes; no threshold, sign, reference, "
                "latency, hold, scheduler, or support repair"
            ),
        },
        "later_evaluation_contract": {
            "open_sequence": ["train_2020_2022", "test_2023"],
            "sealed_sequential": ["2024", "2025", "2026_ytd"],
            "leverage": 0.5,
            "base_cost_notional_per_side": 0.0006,
            "stress_cost_notional_per_side": 0.0010,
            "funding": (
                "interior exact-time symmetric; exact entry/exit credits dropped "
                "and debits retained"
            ),
            "cagr": "full split wall clock including warmup and idle cash",
            "strict_mdd": (
                "global/pre-entry HWM, entry cost, exact funding, every held 5m "
                "path, virtual adverse exit fee, and actual exit"
            ),
            "primary_gates_each_opened_split": {
                "absolute_return_positive": True,
                "cagr_to_strict_mdd_min": 3.0,
                "strict_mdd_pct_max": 15.0,
                "stress_cost_absolute_return_positive": True,
                "one_bar_delayed_absolute_return_positive": True,
                "mean_gross_underlying_bp_min": 20.0,
                "weekly_cluster_signflip_p_max": 0.10,
            },
            "controls": [
                "exact side flip on candidate clock",
                (
                    "impulse-only abs(funding_impulse_z)>=1.25, side opposite "
                    "impulse, same scheduler"
                ),
                (
                    "index-only abs(index_return_z)>=1.25, side opposite index "
                    "return, same scheduler"
                ),
                (
                    "eight-hour-stale interaction uses already-causal "
                    "funding_impulse_z at exact T-8h and current index response"
                ),
                "one additional five-minute entry delay before rescheduling",
                "constant long and constant short on candidate clock",
                (
                    "deterministic within-year paired funding-impulse/index-return "
                    "source permutation and complete feature/clock rebuild"
                ),
            ],
            "control_clock_contract": (
                "six-hour hold, complete split containment, and chronological "
                "greedy nonoverlap; component/stale controls build their own "
                "clocks and exact-clock controls retain the candidate clock"
            ),
            "minimum_component_or_stale_control_trades_each_split": 30,
            "candidate_mean_gross_minus_best_component_or_stale_bp_min": 5.0,
        },
        "frozen_artifacts": {
            "source_decision": str(SOURCE_DECISION),
            "source_decision_sha256": SOURCE_DECISION_SHA256,
            "source_downloader": str(SOURCE_DOWNLOADER),
            "source_downloader_sha256": SOURCE_DOWNLOADER_SHA256,
            "lower_bound_correction": str(LOWER_BOUND_CORRECTION),
            "lower_bound_correction_sha256": (
                LOWER_BOUND_CORRECTION_SHA256
            ),
            "original_preregistration_artifact_hash": (
                ORIGINAL_PREREGISTRATION_ARTIFACT_HASH
            ),
            "preregistration_document": str(PREREGISTRATION_DOCUMENT),
            "preregistration_document_sha256": sha256_file(
                PREREGISTRATION_DOCUMENT
            ),
            "preregistration_source": str(PREREGISTRATION_SOURCE),
            "preregistration_source_sha256": sha256_file(
                PREREGISTRATION_SOURCE
            ),
        },
        "research_history_boundary": (
            "candidate-level freeze only; broad repository BTC history is open, "
            "bounded Deribit source probes and the lower-bound null diagnostic "
            "are disclosed, and complete source incidence, candidate incidence, "
            "matching outcomes, and 2024+ source rows remain unopened; 2023 is "
            "an outcome-blind support-screened test rather than a pristine "
            "global holdout"
        ),
    }


def write_preregistration(cfg: Config) -> dict[str, Any]:
    _validate_config(cfg)
    protocol_payload = protocol(cfg)
    core = {
        "protocol_version": "deribit_funding_impulse_absorption_preregistration_v2",
        "protocol": protocol_payload,
        "protocol_hash": canonical_hash(protocol_payload),
        "outcomes_opened": False,
        "bounded_source_schema_probes_opened": True,
        "source_lower_boundary_diagnostic_opened": True,
        "complete_source_incidence_opened": False,
        "candidate_incidence_opened": False,
        "supersedes_artifact_hash": ORIGINAL_PREREGISTRATION_ARTIFACT_HASH,
    }
    artifact = {**core, "artifact_hash": canonical_hash(core)}
    _write_json(cfg.preregistration_output, artifact)
    return artifact


def load_preregistration(cfg: Config) -> dict[str, Any]:
    artifact = json.loads(Path(cfg.preregistration_output).read_text())
    core = {
        key: value for key, value in artifact.items() if key != "artifact_hash"
    }
    if canonical_hash(core) != artifact.get("artifact_hash"):
        raise RuntimeError("DFIA-72 preregistration artifact hash mismatch")
    if artifact.get("protocol_version") != (
        "deribit_funding_impulse_absorption_preregistration_v2"
    ):
        raise RuntimeError("DFIA-72 preregistration version mismatch")
    expected_protocol = protocol(cfg)
    if artifact.get("protocol") != expected_protocol or artifact.get(
        "protocol_hash"
    ) != canonical_hash(expected_protocol):
        raise RuntimeError("DFIA-72 preregistration protocol drift")
    expected_flags = {
        "outcomes_opened": False,
        "bounded_source_schema_probes_opened": True,
        "source_lower_boundary_diagnostic_opened": True,
        "complete_source_incidence_opened": False,
        "candidate_incidence_opened": False,
    }
    for key, expected in expected_flags.items():
        if artifact.get(key) is not expected:
            raise RuntimeError(f"DFIA-72 preregistration boundary drift: {key}")
    if artifact.get("supersedes_artifact_hash") != (
        ORIGINAL_PREREGISTRATION_ARTIFACT_HASH
    ):
        raise RuntimeError("DFIA-72 preregistration predecessor mismatch")
    return artifact


def _manifest_core(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in manifest.items() if key != "manifest_hash"
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_source_manifest_metadata(
    manifest: dict[str, Any], cfg: Config
) -> dict[str, Any]:
    if manifest.get("protocol_version") != 1 or manifest.get(
        "candidate"
    ) != POLICY_ID:
        raise RuntimeError("DFIA-72 source manifest identity mismatch")
    if canonical_hash(_manifest_core(manifest)) != manifest.get("manifest_hash"):
        raise RuntimeError("DFIA-72 source manifest hash mismatch")
    if manifest.get("config") != asdict(SourceConfig()):
        raise RuntimeError("DFIA-72 source request contract mismatch")
    if (
        manifest.get("official_docs") != OFFICIAL_DOCS
        or manifest.get("official_usage_policy") != OFFICIAL_USAGE_POLICY
        or manifest.get("source_decision") != str(SOURCE_DECISION)
        or manifest.get("source_decision_sha256") != SOURCE_DECISION_SHA256
    ):
        raise RuntimeError("DFIA-72 source authority anchor mismatch")
    if (
        manifest.get("output") != str(SOURCE_DATA)
        or manifest.get("output_columns") != SOURCE_COLUMNS
        or not _is_sha256(manifest.get("output_sha256"))
    ):
        raise RuntimeError("DFIA-72 source aggregate metadata mismatch")
    if manifest.get("source_semantics") != EXPECTED_SOURCE_SEMANTICS:
        raise RuntimeError("DFIA-72 source semantics mismatch")
    expected_availability = {
        "historical_synthetic_delay_minutes": (
            cfg.source_availability_delay_minutes
        ),
        "live_rule": (
            "use actual first successful observation when later; never backdate"
        ),
        "gap_rule": "a missing hour breaks the feature chain and is never filled",
    }
    if manifest.get("causal_availability") != expected_availability:
        raise RuntimeError("DFIA-72 source availability contract mismatch")
    if manifest.get("outcome_boundary") != EXPECTED_OUTCOME_BOUNDARY:
        raise RuntimeError("DFIA-72 source outcome boundary mismatch")

    audit = manifest.get("source_audit")
    if not isinstance(audit, dict):
        raise RuntimeError("DFIA-72 source audit is missing")
    start = _utc(SourceConfig.start, "source start")
    end = _utc(SourceConfig.end_exclusive, "source end")
    expected_hours = int((end - start).total_seconds() // 3600)
    observed = audit.get("observed_rows")
    missing = audit.get("missing_hours")
    if (
        audit.get("requested_hours") != expected_hours
        or isinstance(observed, bool)
        or not isinstance(observed, int)
        or observed <= 0
        or isinstance(missing, bool)
        or not isinstance(missing, int)
        or missing < 0
        or observed + missing != expected_hours
    ):
        raise RuntimeError("DFIA-72 source row-count audit mismatch")
    coverage = audit.get("coverage_ratio")
    if (
        isinstance(coverage, bool)
        or not isinstance(coverage, (int, float))
        or not math.isfinite(float(coverage))
        or not math.isclose(
            float(coverage), observed / expected_hours, rel_tol=0.0, abs_tol=1e-15
        )
    ):
        raise RuntimeError("DFIA-72 source coverage audit mismatch")
    if (
        audit.get("first_observation") != cfg.source_first_exact
        or audit.get("last_observation") != cfg.source_last_exact
        or audit.get("unexpected_row_fields") != 0
        or audit.get("exact_boundary_duplicates") != 0
        or audit.get("conflicting_duplicates") != 0
        or audit.get("lower_boundary_null_prev_index_price") != 1
    ):
        raise RuntimeError("DFIA-72 source boundary/schema audit mismatch")
    maximum_gap = audit.get("maximum_observation_gap_hours")
    if (
        isinstance(maximum_gap, bool)
        or not isinstance(maximum_gap, int)
        or maximum_gap < 1
    ):
        raise RuntimeError("DFIA-72 source gap audit mismatch")
    memory = audit.get("memory_identity")
    if (
        not isinstance(memory, dict)
        or memory.get("all_windows_within_tolerance") is not True
        or memory.get("maximum_allowed_absolute_error") != "0.00005"
    ):
        raise RuntimeError("DFIA-72 source memory audit mismatch")
    lengths = audit.get("response_result_lengths")
    hashes = audit.get("response_result_sha256")
    windows = audit.get("request_windows")
    if (
        isinstance(windows, bool)
        or not isinstance(windows, int)
        or windows <= 0
        or not isinstance(lengths, list)
        or not isinstance(hashes, list)
        or len(lengths) != windows
        or len(hashes) != windows
        or any(
            isinstance(length, bool)
            or not isinstance(length, int)
            or length < 0
            or length >= 744
            for length in lengths
        )
        or any(not _is_sha256(value) for value in hashes)
        or audit.get("response_chain_sha256") != canonical_hash(hashes)
        or audit.get("request_window_hours_max") != SourceConfig.chunk_hours
    ):
        raise RuntimeError("DFIA-72 source response-chain audit mismatch")
    if audit.get("response_environment") != {
        "jsonrpc": "2.0",
        "testnet": False,
        "server_timing_validated_not_persisted": True,
    }:
        raise RuntimeError("DFIA-72 source environment audit mismatch")
    return audit


def validate_source_frame(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if list(frame.columns) != SOURCE_COLUMNS:
        raise RuntimeError("DFIA-72 source columns mismatch")
    checked = frame.copy()
    for column in ["timestamp", "available_at"]:
        checked[column] = pd.to_datetime(checked[column], utc=True, errors="raise")
    numeric_columns = [
        "interest_1h",
        "interest_8h",
        "index_price",
    ]
    for column in numeric_columns:
        checked[column] = pd.to_numeric(checked[column], errors="raise")
        if not np.isfinite(checked[column].to_numpy(float)).all():
            raise RuntimeError(f"DFIA-72 source {column} is not finite")
    raw_previous = checked["prev_index_price"]
    null_previous = raw_previous.isna() | raw_previous.astype(str).str.strip().eq(
        ""
    )
    parsed_previous = pd.to_numeric(raw_previous, errors="coerce")
    if (parsed_previous.isna() & ~null_previous).any():
        raise RuntimeError("DFIA-72 source prev_index_price is not numeric")
    expected_null = pd.Series(False, index=checked.index)
    if len(expected_null):
        expected_null.iloc[0] = True
    if not null_previous.equals(expected_null):
        raise RuntimeError(
            "DFIA-72 source must preserve one lower-bound null prior index"
        )
    checked["prev_index_price"] = parsed_previous
    if checked.empty or checked["timestamp"].duplicated().any():
        raise RuntimeError("DFIA-72 source clock is empty or duplicated")
    if not checked["timestamp"].is_monotonic_increasing:
        raise RuntimeError("DFIA-72 source clock is not chronological")
    timestamp = checked["timestamp"]
    if not (
        timestamp.dt.minute.eq(0)
        & timestamp.dt.second.eq(0)
        & timestamp.dt.microsecond.eq(0)
    ).all():
        raise RuntimeError("DFIA-72 source clock is not an exact UTC hour")
    if timestamp.iloc[0] != _utc(cfg.source_first_exact, "first source") or (
        timestamp.iloc[-1] != _utc(cfg.source_last_exact, "last source")
    ):
        raise RuntimeError("DFIA-72 source frame boundary mismatch")
    expected_available = timestamp + pd.Timedelta(
        minutes=cfg.source_availability_delay_minutes
    )
    if not checked["available_at"].equals(expected_available):
        raise RuntimeError("DFIA-72 source availability clock mismatch")
    if (
        checked["index_price"].le(0.0).any()
        or checked["prev_index_price"].iloc[1:].le(0.0).any()
        or not np.isfinite(
            checked["prev_index_price"].iloc[1:].to_numpy(float)
        ).all()
    ):
        raise RuntimeError("DFIA-72 source index price is not positive")

    gap_hours = timestamp.diff().dt.total_seconds().div(3600.0)
    contiguous = gap_hours.eq(1.0)
    prior_index = checked["index_price"].shift(1)
    if not np.array_equal(
        checked.loc[contiguous, "prev_index_price"].to_numpy(float),
        prior_index.loc[contiguous].to_numpy(float),
    ):
        raise RuntimeError("DFIA-72 source contiguous index-price chain changed")

    one_hour = checked["interest_1h"].to_numpy(float)
    eight_hour = checked["interest_8h"].to_numpy(float)
    timestamp_ns = timestamp.astype("int64").to_numpy()
    maximum_memory_error = float(SourceConfig.maximum_memory_abs_error)
    checked_windows = 0
    for right in range(7, len(checked)):
        if timestamp_ns[right] - timestamp_ns[right - 7] != 7 * 3_600_000_000_000:
            continue
        error = abs(float(one_hour[right - 7 : right + 1].sum()) - eight_hour[right])
        if error > maximum_memory_error:
            raise RuntimeError("DFIA-72 source memory invariant drifted")
        checked_windows += 1
    if checked_windows == 0:
        raise RuntimeError("DFIA-72 source has no contiguous memory window")
    return checked


def load_source(cfg: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    audit = validate_source_manifest_metadata(manifest, cfg)
    if sha256_file(SOURCE_DATA) != manifest.get("output_sha256"):
        raise RuntimeError("DFIA-72 source aggregate hash mismatch")
    frame = validate_source_frame(
        pd.read_csv(SOURCE_DATA, dtype=str, keep_default_na=False), cfg
    )
    if len(frame) != audit.get("observed_rows"):
        raise RuntimeError("DFIA-72 source aggregate row count mismatch")
    return frame, manifest


def strict_prior_z(
    values: np.ndarray,
    timestamps: pd.Series,
    *,
    lookback_hours: int,
    minimum: int,
    ddof: int,
) -> tuple[np.ndarray, np.ndarray]:
    numeric = np.asarray(values, dtype=float)
    clock = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    if len(numeric) != len(clock) or not clock.is_monotonic_increasing:
        raise ValueError("DFIA-72 z-score inputs must share a chronological clock")
    series = pd.Series(numeric, index=clock)
    count = (
        series.rolling(
            f"{lookback_hours}h", closed="left", min_periods=1
        )
        .count()
        .fillna(0.0)
        .to_numpy(dtype=np.int64)
    )
    reference = series.rolling(
        f"{lookback_hours}h", closed="left", min_periods=minimum
    )
    mean = reference.mean().to_numpy(float)
    deviation = reference.std(ddof=ddof).to_numpy(float)
    score = np.divide(
        numeric - mean,
        deviation,
        out=np.full(len(numeric), np.nan, dtype=float),
        where=np.isfinite(deviation) & (deviation > 0.0),
    )
    return score, count


def build_features(source: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    panel = validate_source_frame(source, cfg).reset_index(drop=True)
    panel["funding_impulse"] = (
        panel["interest_1h"] - panel["interest_8h"] / 8.0
    )
    panel["index_return_1h"] = np.log(
        panel["index_price"] / panel["prev_index_price"]
    )
    one_hour_step = panel["timestamp"].diff().eq(pd.Timedelta(hours=1))
    required_steps = cfg.minimum_contiguous_memory_hours - 1
    panel["memory_chain_ready"] = (
        one_hour_step.astype(np.int8)
        .rolling(required_steps, min_periods=required_steps)
        .sum()
        .eq(required_steps)
    )
    reference_impulse = panel["funding_impulse"].where(
        panel["memory_chain_ready"]
    )
    reference_index_return = panel["index_return_1h"].where(
        panel["memory_chain_ready"]
    )
    impulse_z, impulse_count = strict_prior_z(
        reference_impulse.to_numpy(float),
        panel["timestamp"],
        lookback_hours=cfg.reference_lookback_hours,
        minimum=cfg.minimum_prior_observations,
        ddof=cfg.standard_deviation_ddof,
    )
    index_z, index_count = strict_prior_z(
        reference_index_return.to_numpy(float),
        panel["timestamp"],
        lookback_hours=cfg.reference_lookback_hours,
        minimum=cfg.minimum_prior_observations,
        ddof=cfg.standard_deviation_ddof,
    )
    panel["funding_impulse_z"] = impulse_z
    panel["index_return_z"] = index_z
    panel["impulse_reference_count"] = impulse_count
    panel["index_reference_count"] = index_count
    panel["reference_ready"] = (
        panel["impulse_reference_count"].ge(cfg.minimum_prior_observations)
        & panel["index_reference_count"].ge(cfg.minimum_prior_observations)
        & np.isfinite(panel["funding_impulse_z"])
        & np.isfinite(panel["index_return_z"])
    )
    short = (
        panel["reference_ready"]
        & panel["memory_chain_ready"]
        & panel["funding_impulse_z"].ge(cfg.funding_impulse_z_threshold)
        & panel["index_return_1h"].le(0.0)
        & panel["index_return_z"].le(cfg.index_response_z_boundary)
    )
    long = (
        panel["reference_ready"]
        & panel["memory_chain_ready"]
        & panel["funding_impulse_z"].le(-cfg.funding_impulse_z_threshold)
        & panel["index_return_1h"].ge(0.0)
        & panel["index_return_z"].ge(cfg.index_response_z_boundary)
    )
    if (short & long).any():
        raise RuntimeError("DFIA-72 source row has contradictory sides")
    panel["side"] = np.select([long, short], [1, -1], default=0).astype(np.int8)
    panel["candidate"] = panel["side"].ne(0)
    panel["feature_available_at"] = panel["available_at"]
    panel["earliest_observable_open"] = panel[
        "feature_available_at"
    ].dt.ceil("5min")
    panel["entry_time"] = panel["earliest_observable_open"] + pd.Timedelta(
        minutes=5 * cfg.entry_latency_bars
    )
    panel["exit_time"] = panel["entry_time"] + pd.Timedelta(
        minutes=5 * cfg.hold_bars
    )
    return panel


def schedule_clock(features: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    accepted: list[tuple[int, str]] = []
    splits = [
        ("train", cfg.train_start, cfg.train_end_exclusive),
        ("test", cfg.test_start, cfg.test_end_exclusive),
    ]
    candidates = features.loc[features["candidate"]].sort_values("entry_time")
    for split, start_text, end_text in splits:
        start = _utc(start_text, f"{split} start")
        end = _utc(end_text, f"{split} end")
        contained = candidates.loc[
            candidates["entry_time"].ge(start)
            & candidates["exit_time"].le(end)
        ]
        previous_exit: pd.Timestamp | None = None
        for index, row in contained.iterrows():
            entry = _utc(row["entry_time"], "candidate entry")
            if previous_exit is not None and entry < previous_exit:
                continue
            accepted.append((int(index), split))
            previous_exit = _utc(row["exit_time"], "candidate exit")
    if not accepted:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    indices = [index for index, _ in accepted]
    clock = features.loc[indices].copy()
    clock.insert(0, "split", [split for _, split in accepted])
    clock.insert(0, "policy_id", POLICY_ID)
    clock = clock.rename(columns={"timestamp": "source_timestamp"})
    return clock[EVENT_COLUMNS].reset_index(drop=True)


def _count_window(entry: pd.Series, start: str, end_exclusive: str) -> int:
    return int(
        entry.ge(_utc(start, "count start"))
        .mul(entry.lt(_utc(end_exclusive, "count end")))
        .sum()
    )


def _side_shares(clock: pd.DataFrame) -> dict[str, float]:
    if len(clock) == 0:
        return {"long": 0.0, "short": 0.0}
    return {
        "long": float(clock["side"].eq(1).mean()),
        "short": float(clock["side"].eq(-1).mean()),
    }


def source_quality_summary(
    source: pd.DataFrame, manifest: dict[str, Any], cfg: Config
) -> dict[str, Any]:
    timestamp = source["timestamp"]
    start = _utc(SourceConfig.start, "source start")
    end = _utc(SourceConfig.end_exclusive, "source end")
    expected_hours = int((end - start).total_seconds() // 3600)
    overall_coverage = len(source) / expected_hours
    maximum_gap = int(
        timestamp.diff().dt.total_seconds().div(3600.0).max()
    )
    eligible = source.loc[
        timestamp.ge(_utc(cfg.train_start, "train start"))
        & timestamp.lt(_utc(cfg.test_end_exclusive, "test end"))
    ]
    observed_month = (
        eligible["timestamp"].dt.tz_localize(None).dt.to_period("M").value_counts()
    )
    months = pd.period_range("2020-01", "2023-12", freq="M")
    month_counts: dict[str, int] = {}
    month_coverage: dict[str, float] = {}
    for month in months:
        observed = int(observed_month.get(month, 0))
        expected = int(month.days_in_month * 24)
        month_counts[str(month)] = observed
        month_coverage[str(month)] = observed / expected
    audit = manifest["source_audit"]
    checks = {
        "first_exact": timestamp.iloc[0] == _utc(cfg.source_first_exact, "first"),
        "last_exact": timestamp.iloc[-1] == _utc(cfg.source_last_exact, "last"),
        "minimum_overall_coverage": (
            overall_coverage >= cfg.minimum_source_coverage_ratio
        ),
        "minimum_each_eligible_month_coverage": all(
            value >= cfg.minimum_source_month_coverage_ratio
            for value in month_coverage.values()
        ),
        "maximum_gap": maximum_gap <= cfg.maximum_source_gap_hours,
        "memory_invariant": (
            audit["memory_identity"]["all_windows_within_tolerance"] is True
        ),
        "outcome_boundary": manifest["outcome_boundary"]
        == EXPECTED_OUTCOME_BOUNDARY,
    }
    return {
        "expected_hours": expected_hours,
        "observed_hours": int(len(source)),
        "overall_coverage_ratio": float(overall_coverage),
        "first_observation": timestamp.iloc[0].isoformat(),
        "last_observation": timestamp.iloc[-1].isoformat(),
        "maximum_gap_hours": maximum_gap,
        "eligible_month_counts": month_counts,
        "eligible_month_coverage_ratio": month_coverage,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def support_summary(
    clock: pd.DataFrame,
    source_quality: dict[str, Any],
    cfg: Config,
) -> dict[str, Any]:
    entry = pd.to_datetime(clock["entry_time"], utc=True)
    counts = {
        "total_2020_2023": len(clock),
        "train_2020_2022": _count_window(
            entry, cfg.train_start, cfg.train_end_exclusive
        ),
        "train_2020": _count_window(entry, cfg.train_start, "2021-01-01T00:00:00Z"),
        "train_2021": _count_window(
            entry, "2021-01-01T00:00:00Z", "2022-01-01T00:00:00Z"
        ),
        "train_2022": _count_window(
            entry, "2022-01-01T00:00:00Z", cfg.train_end_exclusive
        ),
        "test_2023": _count_window(entry, cfg.test_start, cfg.test_end_exclusive),
        "test_2023_h1": _count_window(
            entry, cfg.test_start, "2023-07-01T00:00:00Z"
        ),
        "test_2023_h2": _count_window(
            entry, "2023-07-01T00:00:00Z", cfg.test_end_exclusive
        ),
    }
    entry_naive = entry.dt.tz_localize(None)
    months = pd.period_range("2020-01", "2023-12", freq="M")
    observed_months = entry_naive.dt.to_period("M").value_counts()
    month_counts = {
        str(month): int(observed_months.get(month, 0)) for month in months
    }
    quarters = pd.period_range("2020Q1", "2023Q4", freq="Q")
    observed_quarters = entry_naive.dt.to_period("Q").value_counts()
    quarter_counts = {
        str(quarter): int(observed_quarters.get(quarter, 0))
        for quarter in quarters
    }
    active_months = sum(count > 0 for count in month_counts.values())
    maximum_month_share = (
        max(month_counts.values()) / len(clock) if len(clock) else 1.0
    )
    side_shares = {
        "all": _side_shares(clock),
        "train": _side_shares(clock.loc[clock["split"].eq("train")]),
        "test": _side_shares(clock.loc[clock["split"].eq("test")]),
    }
    checks = {
        "source_quality": source_quality["passed"] is True,
        "minimum_total": counts["total_2020_2023"] >= cfg.minimum_total,
        "minimum_train": counts["train_2020_2022"] >= cfg.minimum_train,
        "minimum_each_train_year": min(
            counts["train_2020"], counts["train_2021"], counts["train_2022"]
        )
        >= cfg.minimum_each_train_year,
        "minimum_test": counts["test_2023"] >= cfg.minimum_test,
        "minimum_each_test_half": min(
            counts["test_2023_h1"], counts["test_2023_h2"]
        )
        >= cfg.minimum_each_test_half,
        "minimum_each_quarter": all(
            count >= cfg.minimum_each_quarter for count in quarter_counts.values()
        ),
        "minimum_active_months": active_months >= cfg.minimum_active_months,
        "minimum_side_share": all(
            share >= cfg.minimum_side_share
            for split in side_shares.values()
            for share in split.values()
        ),
        "maximum_month_share": maximum_month_share <= cfg.maximum_month_share,
    }
    return {
        "source_quality": source_quality,
        "counts": counts,
        "month_counts": month_counts,
        "quarter_counts": quarter_counts,
        "active_months": active_months,
        "maximum_month_share": float(maximum_month_share),
        "side_shares": side_shares,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _timestamp(value: Any) -> str:
    return _utc(value, "event timestamp").isoformat()


def _finite_float(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"DFIA-72 event field is not finite: {name}")
    return number


def event_records(clock: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in clock.itertuples(index=False):
        if row.side not in (-1, 1) or not bool(row.memory_chain_ready):
            raise RuntimeError("DFIA-72 event side or memory readiness drifted")
        records.append(
            {
                "policy_id": row.policy_id,
                "split": row.split,
                "source_timestamp": _timestamp(row.source_timestamp),
                "feature_available_at": _timestamp(row.feature_available_at),
                "earliest_observable_open": _timestamp(
                    row.earliest_observable_open
                ),
                "entry_time": _timestamp(row.entry_time),
                "exit_time": _timestamp(row.exit_time),
                "side": int(row.side),
                "funding_impulse": _finite_float(
                    row.funding_impulse, "funding_impulse"
                ),
                "index_return_1h": _finite_float(
                    row.index_return_1h, "index_return_1h"
                ),
                "funding_impulse_z": _finite_float(
                    row.funding_impulse_z, "funding_impulse_z"
                ),
                "index_return_z": _finite_float(
                    row.index_return_z, "index_return_z"
                ),
                "impulse_reference_count": int(row.impulse_reference_count),
                "index_reference_count": int(row.index_reference_count),
                "memory_chain_ready": True,
            }
        )
    return records


def event_clock_hash(
    events: list[dict[str, Any]],
    *,
    cfg: Config,
    preregistration_hash: str,
    source_manifest_hash: str,
    source_sha256: str,
) -> str:
    return canonical_hash(
        {
            "policy": asdict(cfg),
            "preregistration_hash": preregistration_hash,
            "source_manifest_hash": source_manifest_hash,
            "source_sha256": source_sha256,
            "events": events,
        }
    )


def run_support(cfg: Config) -> dict[str, Any]:
    _validate_config(cfg)
    preregistration = load_preregistration(cfg)
    source, source_manifest = load_source(cfg)
    features = build_features(source, cfg)
    clock = schedule_clock(features, cfg)
    source_quality = source_quality_summary(source, source_manifest, cfg)
    support = support_summary(clock, source_quality, cfg)
    source_sha = sha256_file(SOURCE_DATA)
    source_manifest_hash = str(source_manifest["manifest_hash"])
    records = event_records(clock)
    clock_hash = event_clock_hash(
        records,
        cfg=cfg,
        preregistration_hash=str(preregistration["artifact_hash"]),
        source_manifest_hash=source_manifest_hash,
        source_sha256=source_sha,
    )
    event_path = Path(cfg.event_clock_output)
    if support["passed"]:
        clock_core = {
            "protocol_version": "deribit_funding_impulse_absorption_event_clock_v1",
            "policy_id": POLICY_ID,
            "outcomes_opened": False,
            "policy": asdict(cfg),
            "preregistration_artifact_hash": preregistration["artifact_hash"],
            "source_manifest_hash": source_manifest_hash,
            "source_sha256": source_sha,
            "events": records,
            "event_clock_hash": clock_hash,
        }
        clock_artifact = {
            **clock_core,
            "artifact_hash": canonical_hash(clock_core),
        }
        _write_json(event_path, clock_artifact)
    else:
        event_path.unlink(missing_ok=True)

    core = {
        "protocol_version": "deribit_funding_impulse_absorption_support_v1",
        "policy_id": POLICY_ID,
        "outcomes_opened": False,
        "complete_source_incidence_opened": True,
        "candidate_incidence_opened": True,
        "policy": asdict(cfg),
        "preregistration": {
            "path": cfg.preregistration_output,
            "sha256": sha256_file(cfg.preregistration_output),
            "artifact_hash": preregistration["artifact_hash"],
        },
        "source": {
            "path": str(SOURCE_DATA),
            "sha256": source_sha,
            "manifest": str(SOURCE_MANIFEST),
            "manifest_sha256": sha256_file(SOURCE_MANIFEST),
            "manifest_hash": source_manifest_hash,
            "rows": len(source),
            "first_observation": _timestamp(source["timestamp"].min()),
            "last_observation": _timestamp(source["timestamp"].max()),
            "binance_market_or_funding_rows_loaded": 0,
            "post_2023_source_rows_loaded": 0,
        },
        "feature_incidence": {
            "reference_ready_rows": int(features["reference_ready"].sum()),
            "memory_ready_rows": int(features["memory_chain_ready"].sum()),
            "candidate_rows_before_nonoverlap": int(features["candidate"].sum()),
            "accepted_split_contained_nonoverlap_events": len(clock),
        },
        "support_gate": support,
        "event_clock": {
            "written": support["passed"],
            "path": cfg.event_clock_output if support["passed"] else None,
            "event_clock_hash": clock_hash if support["passed"] else None,
            "rows": len(clock) if support["passed"] else 0,
        },
        "sealed": [
            "all matching 2020-2023 Binance five-minute paths and funding",
            "all Deribit source rows at or after 2024-01-01",
            "2024",
            "2025",
            "2026_ytd",
        ],
        "failure_action": (
            "freeze exact clock and preregister strict evaluator"
            if support["passed"]
            else "reject before opening BTC market or funding outcomes; no repair"
        ),
    }
    result = {**core, "result_hash": canonical_hash(core)}
    _write_json(cfg.support_output, result)
    return result


def parse_args() -> tuple[str, Config]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("preregister", "support"), nargs="?", default="preregister"
    )
    parser.add_argument(
        "--preregistration-output", default=Config.preregistration_output
    )
    parser.add_argument("--support-output", default=Config.support_output)
    parser.add_argument("--event-clock-output", default=Config.event_clock_output)
    args = parser.parse_args()
    return args.mode, Config(
        preregistration_output=args.preregistration_output,
        support_output=args.support_output,
        event_clock_output=args.event_clock_output,
    )


def main() -> None:
    mode, cfg = parse_args()
    result = (
        write_preregistration(cfg) if mode == "preregister" else run_support(cfg)
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
