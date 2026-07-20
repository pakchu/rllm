"""Freeze and run outcome-blind source support for ARCR-864.

This module may read only the frozen Coin Metrics address source.  It never
loads BTC market bars, funding, post-entry returns, held paths, or PnL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.download_coinmetrics_btc_address_reservoir_daily import (
    Config as SourceConfig,
)
from training.download_coinmetrics_btc_address_reservoir_daily import (
    OFFICIAL_CATALOG_URL,
)
from training.download_coinmetrics_btc_address_reservoir_daily import (
    OUTPUT_COLUMNS as SOURCE_OUTPUT_COLUMNS,
)
from training.download_coinmetrics_btc_address_reservoir_daily import (
    canonical_hash,
    source_url,
)


POLICY_ID = "ARCR-864"
SOURCE_DATA = Path(
    "data/coinmetrics_btc_address_reservoir_2019_2023.csv.gz"
)
SOURCE_MANIFEST = Path(
    "results/coinmetrics_btc_address_reservoir_source_manifest_2026-07-20.json"
)
SOURCE_DECISION = Path(
    "docs/address-reservoir-capacitance-mechanism-decision-2026-07-20.md"
)
SOURCE_DECISION_SHA256 = (
    "9b22d60eef61ab7fc3b9f5332669be75af0a4bd54d331f3017bf9aeabf0eaa8f"
)
SOURCE_DOWNLOADER = Path(
    "training/download_coinmetrics_btc_address_reservoir_daily.py"
)
SOURCE_DOWNLOADER_SHA256 = (
    "40b759a06038782ebc6e676b3320b4f8eb360e097de9384e5f0f0db45c890c94"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/address-reservoir-capacitance-support-preregistration-2026-07-20.md"
)
PREREGISTRATION_SOURCE = Path(
    "training/preregister_address_reservoir_capacitance_support.py"
)
SOURCE_COLUMNS = list(SOURCE_OUTPUT_COLUMNS)
HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SOURCE_MANIFEST_KEYS = frozenset(
    {
        "protocol_version",
        "candidate",
        "config",
        "official_catalog_url",
        "source_audit",
        "output",
        "output_columns",
        "output_sha256",
        "source_semantics",
        "causal_availability",
        "revision_boundary",
        "outcome_boundary",
        "manifest_hash",
    }
)
SOURCE_AUDIT_KEYS = frozenset(
    {
        "source_url",
        "response_pages",
        "response_page_lengths",
        "response_page_sha256",
        "response_chain_sha256",
        "expected_rows",
        "observed_rows",
        "first_observation",
        "last_observation",
        "maximum_observation_gap_days",
        "duplicates",
        "missing_days",
        "unexpected_row_fields",
    }
)
EXPECTED_SOURCE_SEMANTICS = {
    "AdrBalCnt": (
        "unique addresses holding any positive native-unit balance at interval end"
    ),
    "AdrActCnt": (
        "unique addresses active as originator or recipient during the interval"
    ),
    "AssetEODCompletionTime": (
        "recorded completion timestamp used as source availability"
    ),
}
EXPECTED_CAUSAL_AVAILABILITY = (
    "available_at equals AssetEODCompletionTime and must be no earlier "
    "than observation UTC midnight plus one day"
)
EXPECTED_REVISION_BOUNDARY = (
    "the output hash freezes this downloaded source vintage; it is not "
    "a historical archive of Coin Metrics revisions"
)


@dataclass(frozen=True)
class Config:
    preregistration_output: str = (
        "results/address_reservoir_capacitance_support_preregistration_"
        "2026-07-20.json"
    )
    support_output: str = (
        "results/address_reservoir_capacitance_support_2026-07-20.json"
    )
    event_clock_output: str = (
        "results/address_reservoir_capacitance_event_clock_2026-07-20.json"
    )
    change_days: int = 7
    reference_lookback_days: int = 365
    minimum_prior_observations: int = 180
    reservoir_z_threshold: float = 0.75
    turnover_z_threshold: float = 0.75
    spread_z_threshold: float = 1.75
    maximum_source_lag_days: float = 3.0
    entry_latency_bars: int = 1
    hold_bars: int = 864
    train_start: str = "2021-07-01"
    train_end_exclusive: str = "2023-01-01"
    test_start: str = "2023-01-01"
    test_end_exclusive: str = "2024-01-01"
    expected_source_rows: int = 1826
    expected_source_first: str = "2019-01-01T00:00:00Z"
    expected_source_last: str = "2023-12-31T00:00:00Z"
    maximum_source_gap_days: int = 1
    minimum_total: int = 90
    minimum_train: int = 55
    minimum_train_2021h2: int = 15
    minimum_train_2022: int = 30
    minimum_test_2023: int = 30
    minimum_each_test_half: int = 12
    minimum_each_quarter: int = 5
    minimum_active_months: int = 25
    minimum_side_share: float = 0.25
    maximum_month_share: float = 0.15


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
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
    temporary = output.with_name(output.name + ".tmp")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_config(cfg: Config) -> None:
    expected = Config(
        preregistration_output=cfg.preregistration_output,
        support_output=cfg.support_output,
        event_clock_output=cfg.event_clock_output,
    )
    if cfg != expected:
        raise ValueError("ARCR-864 signal and support configuration is frozen")
    anchors = {
        SOURCE_DECISION: SOURCE_DECISION_SHA256,
        SOURCE_DOWNLOADER: SOURCE_DOWNLOADER_SHA256,
    }
    for path, expected_sha in anchors.items():
        if sha256_file(path) != expected_sha:
            raise ValueError(f"ARCR-864 frozen source anchor mismatch: {path}")


def protocol(cfg: Config) -> dict[str, Any]:
    return {
        "policy_id": POLICY_ID,
        "support_only": True,
        "outcomes_opened": False,
        "source": {
            "endpoint": (
                "https://community-api.coinmetrics.io/v4/timeseries/"
                "asset-metrics"
            ),
            "asset": "btc",
            "metrics": [
                "AdrBalCnt",
                "AdrActCnt",
                "AssetEODCompletionTime",
            ],
            "interval": ["2019-01-01", "2024-01-01"],
            "aggregate": str(SOURCE_DATA),
            "aggregate_sha256": "pending_outcome_blind_download",
            "manifest": str(SOURCE_MANIFEST),
            "manifest_hash": "pending_outcome_blind_download",
            "exact_output_columns": SOURCE_COLUMNS,
            "raw_responses_persisted": False,
            "rows_at_or_after_2024_loaded": False,
            "market_funding_return_or_pnl_fields": [],
        },
        "clock": {
            "timezone": "UTC",
            "row_availability": (
                "exact AssetEODCompletionTime, never earlier than observation "
                "midnight + 1 day"
            ),
            "feature_availability": (
                "maximum availability of current and seven-day-lag source rows"
            ),
            "reference_availability": (
                "prior feature availability must be strictly earlier than "
                "current feature availability"
            ),
            "earliest_observable_open": "ceil(feature_available_at, 5 minutes)",
            "entry_latency_bars": cfg.entry_latency_bars,
            "hold_bars": cfg.hold_bars,
            "hold_minutes": cfg.hold_bars * 5,
            "split_containment": True,
            "nonoverlap": "greedy independently inside train and test",
            "late_live_source": "delay or cancel; never backdate",
        },
        "feature": {
            "change_days": cfg.change_days,
            "reservoir_flux": "log(AdrBalCnt_t / AdrBalCnt_t-7)",
            "turnover": "log(AdrActCnt_t / AdrBalCnt_t)",
            "turnover_shift": "turnover_t - turnover_t-7",
            "activity_flux": "log(AdrActCnt_t / AdrActCnt_t-7)",
            "mechanical_identity": (
                "turnover_shift = activity_flux - reservoir_flux; the gated "
                "terms are an intentional contrast, not independent confirmations"
            ),
            "standardization": {
                "window": (
                    f"strictly prior {cfg.reference_lookback_days} calendar days"
                ),
                "minimum": cfg.minimum_prior_observations,
                "mean": "sample mean",
                "standard_deviation": "sample standard deviation, ddof=1",
                "current_excluded": True,
                "availability_filtered": True,
                "current_freshness_filter_applies_to_references": False,
                "clipping_or_winsorization": False,
                "insufficient_or_zero_std": "neutral score",
            },
            "spread": "reservoir_z - turnover_z",
            "long": {
                "reservoir_z_min": cfg.reservoir_z_threshold,
                "turnover_z_max": -cfg.turnover_z_threshold,
                "spread_z_min": cfg.spread_z_threshold,
            },
            "short": {
                "reservoir_z_max": -cfg.reservoir_z_threshold,
                "turnover_z_min": cfg.turnover_z_threshold,
                "spread_z_max": -cfg.spread_z_threshold,
            },
            "maximum_source_lag_days": cfg.maximum_source_lag_days,
            "maximum_source_lag_scope": "current candidate row only",
            "event": "first nonzero state after a neutral prior observation",
            "direct_reversal_without_neutral": False,
            "threshold_grid": False,
            "repair_after_incidence": False,
        },
        "support_gate": {
            "train": [cfg.train_start, cfg.train_end_exclusive],
            "test": [cfg.test_start, cfg.test_end_exclusive],
            "expected_source_rows": cfg.expected_source_rows,
            "expected_source_first": cfg.expected_source_first,
            "expected_source_last": cfg.expected_source_last,
            "maximum_source_gap_days": cfg.maximum_source_gap_days,
            "minimum_total": cfg.minimum_total,
            "minimum_train": cfg.minimum_train,
            "minimum_train_2021h2": cfg.minimum_train_2021h2,
            "minimum_train_2022": cfg.minimum_train_2022,
            "minimum_test_2023": cfg.minimum_test_2023,
            "minimum_each_test_half": cfg.minimum_each_test_half,
            "minimum_each_quarter": cfg.minimum_each_quarter,
            "minimum_active_months": cfg.minimum_active_months,
            "minimum_side_share_all_train_test": cfg.minimum_side_share,
            "maximum_month_share": cfg.maximum_month_share,
            "count_clock": (
                "accepted entry timestamp after current freshness, transition, "
                "latency, split containment, and greedy nonoverlap"
            ),
            "failure_action": (
                "reject before market outcomes; no feature, sign, threshold, "
                "transition, latency, hold, or support repair"
            ),
        },
        "later_evaluation_contract": {
            "ordered_opening": [
                "train_2021h2_2022",
                "test_2023",
                "2024",
                "2025",
                "2026_ytd",
            ],
            "stop_on_first_failure": True,
            "split_contained_holds": True,
            "leverage": 0.5,
            "base_cost_notional_per_side": 0.0006,
            "stress_cost_notional_per_side": 0.0010,
            "funding": (
                "interior exact-time symmetric; exact entry/exit credits "
                "dropped and debits retained"
            ),
            "cagr": "full split wall clock including warmup and idle cash",
            "strict_mdd": (
                "global/pre-entry HWM, entry cost, exact funding, every held "
                "5m path, virtual adverse exit fee, and actual exit"
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
                "exact side flip",
                (
                    "reservoir-only state: long at reservoir_z >= 0.75, "
                    "short at reservoir_z <= -0.75"
                ),
                (
                    "turnover-only state: long at turnover_z <= -0.75, "
                    "short at turnover_z >= 0.75"
                ),
                (
                    "activity-flux-only state: long at activity_z <= -0.75, "
                    "short at activity_z >= 0.75"
                ),
                (
                    "exact candidate event and side delayed seven calendar "
                    "days before split containment and nonoverlap"
                ),
                "one additional five-minute entry delay",
                "constant long on exact candidate clocks",
                "constant short on exact candidate clocks",
                (
                    "within-year permutation of paired reservoir_flux, "
                    "turnover_shift, activity_flux, and source_lag tuples; "
                    "NumPy default_rng seed is the first eight bytes of "
                    "SHA256('ARCR-864|<year>') interpreted as an unsigned "
                    "big-endian integer; then recompute references, states, "
                    "and clocks"
                ),
            ],
            "control_comparison_metric": (
                "mean gross underlying basis points per accepted trade, "
                "separately in every opened split"
            ),
            "candidate_mean_gross_minus_best_component_or_stale_control_bp_min": 5.0,
        },
        "frozen_artifacts": {
            "source_decision": str(SOURCE_DECISION),
            "source_decision_sha256": SOURCE_DECISION_SHA256,
            "source_downloader": str(SOURCE_DOWNLOADER),
            "source_downloader_sha256": SOURCE_DOWNLOADER_SHA256,
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
            "candidate-level freeze only; broad unrelated BTC history has been "
            "seen. A disclosed bounded source-schema probe is open, while "
            "complete source incidence, candidate incidence, all matching "
            "market outcomes, and every 2024+ source row remain unopened"
        ),
    }


def write_preregistration(cfg: Config) -> dict[str, Any]:
    _validate_config(cfg)
    protocol_payload = protocol(cfg)
    core = {
        "protocol_version": "address_reservoir_capacitance_preregistration_v1",
        "protocol": protocol_payload,
        "protocol_hash": canonical_hash(protocol_payload),
        "outcomes_opened": False,
        "bounded_source_schema_probe_opened": True,
        "complete_source_incidence_opened": False,
        "candidate_incidence_opened": False,
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
        raise RuntimeError("ARCR-864 preregistration artifact hash mismatch")
    if artifact.get("protocol") != protocol(cfg):
        raise RuntimeError("ARCR-864 preregistration protocol drift")
    if artifact.get("outcomes_opened") is not False:
        raise RuntimeError("ARCR-864 preregistration opened outcomes")
    if artifact.get("complete_source_incidence_opened") is not False:
        raise RuntimeError("ARCR-864 preregistration opened complete source")
    if artifact.get("candidate_incidence_opened") is not False:
        raise RuntimeError("ARCR-864 preregistration opened candidate incidence")
    return artifact


def validate_source_manifest_metadata(
    manifest: dict[str, Any], cfg: Config
) -> dict[str, Any]:
    if frozenset(manifest) != SOURCE_MANIFEST_KEYS:
        raise RuntimeError("ARCR-864 source manifest schema drift")
    if manifest.get("protocol_version") != 1:
        raise RuntimeError("ARCR-864 source manifest version mismatch")
    core = {
        key: value for key, value in manifest.items() if key != "manifest_hash"
    }
    if canonical_hash(core) != manifest.get("manifest_hash"):
        raise RuntimeError("ARCR-864 source manifest hash mismatch")
    if manifest.get("candidate") != POLICY_ID:
        raise RuntimeError("ARCR-864 source manifest candidate mismatch")
    if manifest.get("config") != asdict(SourceConfig()):
        raise RuntimeError("ARCR-864 source request contract mismatch")
    if manifest.get("official_catalog_url") != OFFICIAL_CATALOG_URL:
        raise RuntimeError("ARCR-864 source catalog contract mismatch")
    if manifest.get("output") != str(SOURCE_DATA):
        raise RuntimeError("ARCR-864 source path mismatch")
    if manifest.get("output_columns") != SOURCE_COLUMNS:
        raise RuntimeError("ARCR-864 source columns mismatch")
    output_sha = manifest.get("output_sha256")
    if not isinstance(output_sha, str) or HEX_SHA256_RE.fullmatch(output_sha) is None:
        raise RuntimeError("ARCR-864 source SHA-256 is invalid")

    audit = manifest.get("source_audit")
    if not isinstance(audit, dict):
        raise RuntimeError("ARCR-864 source audit is missing")
    if frozenset(audit) != SOURCE_AUDIT_KEYS:
        raise RuntimeError("ARCR-864 source audit schema drift")
    expected_audit = {
        "source_url": source_url(SourceConfig()),
        "expected_rows": cfg.expected_source_rows,
        "observed_rows": cfg.expected_source_rows,
        "first_observation": cfg.expected_source_first,
        "last_observation": cfg.expected_source_last,
        "maximum_observation_gap_days": cfg.maximum_source_gap_days,
        "duplicates": 0,
        "missing_days": 0,
        "unexpected_row_fields": 0,
    }
    for key, expected in expected_audit.items():
        if audit.get(key) != expected:
            raise RuntimeError(f"ARCR-864 source audit mismatch: {key}")
    pages = audit.get("response_pages")
    lengths = audit.get("response_page_lengths")
    hashes = audit.get("response_page_sha256")
    valid_lengths = (
        isinstance(lengths, list)
        and all(
            not isinstance(length, bool)
            and isinstance(length, int)
            and length > 0
            for length in lengths
        )
    )
    valid_hashes = (
        isinstance(hashes, list)
        and all(
            isinstance(value, str)
            and HEX_SHA256_RE.fullmatch(value) is not None
            for value in hashes
        )
    )
    if (
        isinstance(pages, bool)
        or not isinstance(pages, int)
        or pages <= 0
        or not valid_lengths
        or not valid_hashes
        or len(lengths) != pages
        or len(hashes) != pages
        or sum(lengths) != cfg.expected_source_rows
    ):
        raise RuntimeError("ARCR-864 source response-page audit is invalid")
    if canonical_hash(hashes) != audit.get("response_chain_sha256"):
        raise RuntimeError("ARCR-864 source response chain mismatch")

    if manifest.get("source_semantics") != EXPECTED_SOURCE_SEMANTICS:
        raise RuntimeError("ARCR-864 source semantics mismatch")
    if manifest.get("causal_availability") != EXPECTED_CAUSAL_AVAILABILITY:
        raise RuntimeError("ARCR-864 source availability semantics mismatch")
    if manifest.get("revision_boundary") != EXPECTED_REVISION_BOUNDARY:
        raise RuntimeError("ARCR-864 source revision boundary mismatch")

    expected_outcome_boundary = {
        "btc_market_rows_loaded": 0,
        "funding_rows_loaded": 0,
        "return_or_pnl_fields": 0,
        "post_2023_source_rows_loaded": 0,
        "raw_api_pages_persisted": False,
    }
    if manifest.get("outcome_boundary") != expected_outcome_boundary:
        raise RuntimeError("ARCR-864 source outcome boundary mismatch")
    return {"output_sha256": output_sha, "source_audit": audit}


def load_source(cfg: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    metadata = validate_source_manifest_metadata(manifest, cfg)
    if sha256_file(SOURCE_DATA) != metadata["output_sha256"]:
        raise RuntimeError("ARCR-864 source file hash mismatch")
    frame = pd.read_csv(SOURCE_DATA, dtype=str)
    if frame.columns.tolist() != SOURCE_COLUMNS:
        raise RuntimeError("ARCR-864 source file columns changed")
    if len(frame) != cfg.expected_source_rows:
        raise RuntimeError("ARCR-864 source row count changed")
    frame["observation_date"] = pd.to_datetime(
        frame["observation_date"], utc=True, errors="raise"
    )
    frame["available_at"] = pd.to_datetime(
        frame["available_at"], utc=True, errors="raise"
    )
    expected_dates = pd.date_range(
        "2019-01-01", "2023-12-31", freq="1D", tz="UTC"
    )
    if not frame["observation_date"].equals(
        pd.Series(expected_dates, name="observation_date")
    ):
        raise RuntimeError("ARCR-864 source daily grid changed")
    if (
        frame["available_at"]
        < frame["observation_date"] + pd.Timedelta(days=1)
    ).any():
        raise RuntimeError("ARCR-864 source availability moved before D+1")
    if frame["observation_date"].max() >= pd.Timestamp("2024-01-01", tz="UTC"):
        raise RuntimeError("ARCR-864 source crossed the sealed 2024 boundary")
    for column in ("AdrBalCnt", "AdrActCnt"):
        values = frame[column]
        if not values.str.fullmatch(r"[1-9]\d*").all():
            raise RuntimeError(
                f"ARCR-864 source column must be a positive integer: {column}"
            )
        frame[column] = values.map(int).astype(np.int64)
    return frame, manifest


def strict_prior_z(
    values: np.ndarray,
    observation_date: np.ndarray,
    feature_available_at: np.ndarray,
    *,
    lookback_days: int,
    minimum: int,
) -> tuple[np.ndarray, np.ndarray]:
    zscore = np.full(len(values), np.nan, dtype=float)
    reference_count = np.zeros(len(values), dtype=np.int64)
    lookback = np.timedelta64(lookback_days, "D")
    for index in range(len(values)):
        if not np.isfinite(values[index]):
            continue
        prior = np.arange(index)
        causal = prior[
            np.isfinite(values[prior])
            & (observation_date[prior] >= observation_date[index] - lookback)
            & (observation_date[prior] < observation_date[index])
            & (feature_available_at[prior] < feature_available_at[index])
        ]
        reference_count[index] = len(causal)
        if len(causal) < minimum:
            continue
        reference = values[causal]
        standard_deviation = float(np.std(reference, ddof=1))
        if not math.isfinite(standard_deviation) or standard_deviation <= 0.0:
            continue
        zscore[index] = (
            values[index] - float(np.mean(reference))
        ) / standard_deviation
    return zscore, reference_count


def build_features(source: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    frame = source[["observation_date", "available_at"]].copy()
    balance = source["AdrBalCnt"].to_numpy(float)
    active = source["AdrActCnt"].to_numpy(float)
    log_balance = np.log(balance)
    turnover = np.log(active / balance)
    reservoir_flux = np.full(len(frame), np.nan, dtype=float)
    activity_flux = np.full(len(frame), np.nan, dtype=float)
    turnover_shift = np.full(len(frame), np.nan, dtype=float)
    feature_available = np.full(
        len(frame), np.datetime64("NaT"), dtype="datetime64[ns]"
    )
    source_available = source["available_at"].to_numpy(dtype="datetime64[ns]")
    for index in range(cfg.change_days, len(frame)):
        lag = index - cfg.change_days
        reservoir_flux[index] = log_balance[index] - log_balance[lag]
        activity_flux[index] = math.log(active[index]) - math.log(active[lag])
        turnover_shift[index] = turnover[index] - turnover[lag]
        feature_available[index] = max(
            source_available[index], source_available[lag]
        )
    observation = source["observation_date"].to_numpy(dtype="datetime64[ns]")
    reservoir_z, reservoir_reference_count = strict_prior_z(
        reservoir_flux,
        observation,
        feature_available,
        lookback_days=cfg.reference_lookback_days,
        minimum=cfg.minimum_prior_observations,
    )
    turnover_z, turnover_reference_count = strict_prior_z(
        turnover_shift,
        observation,
        feature_available,
        lookback_days=cfg.reference_lookback_days,
        minimum=cfg.minimum_prior_observations,
    )
    activity_z, activity_reference_count = strict_prior_z(
        activity_flux,
        observation,
        feature_available,
        lookback_days=cfg.reference_lookback_days,
        minimum=cfg.minimum_prior_observations,
    )
    finite_identity = np.isfinite(turnover_shift)
    if not np.allclose(
        turnover_shift[finite_identity],
        (activity_flux - reservoir_flux)[finite_identity],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise AssertionError("ARCR-864 stock-flow identity failed")
    spread_z = reservoir_z - turnover_z
    feature_available_series = pd.Series(
        pd.to_datetime(feature_available, utc=True), index=frame.index
    )
    source_lag_days = (
        feature_available_series - frame["observation_date"]
    ).dt.total_seconds() / 86_400.0
    finite = (
        np.isfinite(reservoir_z)
        & np.isfinite(turnover_z)
        & np.isfinite(activity_z)
    )
    fresh = source_lag_days.le(cfg.maximum_source_lag_days).fillna(False)
    long_state = (
        finite
        & fresh.to_numpy()
        & (reservoir_z >= cfg.reservoir_z_threshold)
        & (turnover_z <= -cfg.turnover_z_threshold)
        & (spread_z >= cfg.spread_z_threshold)
    )
    short_state = (
        finite
        & fresh.to_numpy()
        & (reservoir_z <= -cfg.reservoir_z_threshold)
        & (turnover_z >= cfg.turnover_z_threshold)
        & (spread_z <= -cfg.spread_z_threshold)
    )
    if np.any(long_state & short_state):
        raise AssertionError("ARCR-864 long and short states overlap")
    state_side = np.where(long_state, 1, np.where(short_state, -1, 0))
    prior_side = np.concatenate(([0], state_side[:-1]))
    event = (state_side != 0) & (prior_side == 0)

    frame["feature_available_at"] = feature_available_series
    frame["reservoir_flux_7d"] = reservoir_flux
    frame["activity_flux_7d"] = activity_flux
    frame["turnover"] = turnover
    frame["turnover_shift_7d"] = turnover_shift
    frame["reservoir_z"] = reservoir_z
    frame["turnover_z"] = turnover_z
    frame["activity_z"] = activity_z
    frame["spread_z"] = spread_z
    frame["reservoir_reference_count"] = reservoir_reference_count
    frame["turnover_reference_count"] = turnover_reference_count
    frame["activity_reference_count"] = activity_reference_count
    frame["source_lag_days"] = source_lag_days
    frame["state_side"] = state_side
    frame["event"] = event
    return frame


EVENT_COLUMNS = [
    "policy_id",
    "split",
    "side",
    "observation_date",
    "feature_available_at",
    "earliest_observable_open",
    "entry_time",
    "exit_time",
    "reservoir_flux_7d",
    "activity_flux_7d",
    "turnover_shift_7d",
    "reservoir_z",
    "turnover_z",
    "activity_z",
    "spread_z",
    "source_lag_days",
    "reservoir_reference_count",
    "turnover_reference_count",
    "activity_reference_count",
]


def schedule_clock(features: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    candidates = features.loc[features["event"]].copy()
    candidates["earliest_observable_open"] = candidates[
        "feature_available_at"
    ].dt.ceil("5min")
    candidates["entry_time"] = candidates[
        "earliest_observable_open"
    ] + pd.to_timedelta(cfg.entry_latency_bars * 5, unit="min")
    candidates["exit_time"] = candidates["entry_time"] + pd.to_timedelta(
        cfg.hold_bars * 5, unit="min"
    )
    splits = {
        "train": (cfg.train_start, cfg.train_end_exclusive),
        "test": (cfg.test_start, cfg.test_end_exclusive),
    }
    selected: list[pd.DataFrame] = []
    for label, (start_text, end_text) in splits.items():
        start = pd.Timestamp(start_text, tz="UTC")
        end = pd.Timestamp(end_text, tz="UTC")
        contained = candidates.loc[
            candidates["entry_time"].ge(start)
            & candidates["exit_time"].lt(end)
        ].sort_values("entry_time")
        accepted: list[int] = []
        next_entry: pd.Timestamp | None = None
        for index, row in contained.iterrows():
            if next_entry is not None and row["entry_time"] < next_entry:
                continue
            accepted.append(index)
            next_entry = row["exit_time"]
        split = contained.loc[accepted].copy()
        split.insert(0, "policy_id", POLICY_ID)
        split.insert(1, "split", label)
        split.insert(2, "side", split["state_side"].astype(int))
        selected.append(split)
    if not selected:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    clock = (
        pd.concat(selected, ignore_index=True)
        .sort_values("entry_time")
        .reset_index(drop=True)
    )
    return clock[EVENT_COLUMNS]


def _count_window(
    entry: pd.Series, start: str, end_exclusive: str
) -> int:
    start_time = pd.Timestamp(start, tz="UTC")
    end_time = pd.Timestamp(end_exclusive, tz="UTC")
    return int(entry.ge(start_time).mul(entry.lt(end_time)).sum())


def _side_shares(clock: pd.DataFrame) -> dict[str, float]:
    if len(clock) == 0:
        return {"long": 0.0, "short": 0.0}
    return {
        "long": float(clock["side"].eq(1).mean()),
        "short": float(clock["side"].eq(-1).mean()),
    }


def support_summary(clock: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    entry = pd.to_datetime(clock["entry_time"], utc=True)
    counts = {
        "total_2021h2_2023": len(clock),
        "train_2021h2_2022": _count_window(
            entry, cfg.train_start, cfg.train_end_exclusive
        ),
        "train_2021h2": _count_window(
            entry, cfg.train_start, "2022-01-01"
        ),
        "train_2022": _count_window(
            entry, "2022-01-01", cfg.train_end_exclusive
        ),
        "test_2023": _count_window(
            entry, cfg.test_start, cfg.test_end_exclusive
        ),
        "test_2023_h1": _count_window(
            entry, cfg.test_start, "2023-07-01"
        ),
        "test_2023_h2": _count_window(
            entry, "2023-07-01", cfg.test_end_exclusive
        ),
    }
    entry_naive = entry.dt.tz_localize(None)
    months = pd.period_range("2021-07", "2023-12", freq="M")
    observed_months = entry_naive.dt.to_period("M").value_counts()
    month_counts = {
        str(month): int(observed_months.get(month, 0)) for month in months
    }
    quarters = pd.period_range("2021Q3", "2023Q4", freq="Q")
    observed_quarters = entry_naive.dt.to_period("Q").value_counts()
    quarter_counts = {
        str(quarter): int(observed_quarters.get(quarter, 0))
        for quarter in quarters
    }
    active_months = sum(count > 0 for count in month_counts.values())
    maximum_month_share = (
        max(month_counts.values()) / len(clock) if len(clock) else 1.0
    )
    train = clock.loc[clock["split"].eq("train")]
    test = clock.loc[clock["split"].eq("test")]
    side_shares = {
        "all": _side_shares(clock),
        "train": _side_shares(train),
        "test": _side_shares(test),
    }
    checks = {
        "minimum_total": counts["total_2021h2_2023"] >= cfg.minimum_total,
        "minimum_train": counts["train_2021h2_2022"] >= cfg.minimum_train,
        "minimum_train_2021h2": (
            counts["train_2021h2"] >= cfg.minimum_train_2021h2
        ),
        "minimum_train_2022": (
            counts["train_2022"] >= cfg.minimum_train_2022
        ),
        "minimum_test_2023": (
            counts["test_2023"] >= cfg.minimum_test_2023
        ),
        "minimum_test_2023_h1": (
            counts["test_2023_h1"] >= cfg.minimum_each_test_half
        ),
        "minimum_test_2023_h2": (
            counts["test_2023_h2"] >= cfg.minimum_each_test_half
        ),
        "minimum_each_quarter": all(
            count >= cfg.minimum_each_quarter
            for count in quarter_counts.values()
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
    return pd.Timestamp(value).isoformat()


def _finite_float(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"ARCR-864 event field must be finite: {name}")
    return number


def event_records(clock: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in clock.itertuples(index=False):
        records.append(
            {
                "policy_id": row.policy_id,
                "split": row.split,
                "side": int(row.side),
                "observation_date": _timestamp(row.observation_date),
                "feature_available_at": _timestamp(row.feature_available_at),
                "earliest_observable_open": _timestamp(
                    row.earliest_observable_open
                ),
                "entry_time": _timestamp(row.entry_time),
                "exit_time": _timestamp(row.exit_time),
                "reservoir_flux_7d": _finite_float(
                    row.reservoir_flux_7d, "reservoir_flux_7d"
                ),
                "activity_flux_7d": _finite_float(
                    row.activity_flux_7d, "activity_flux_7d"
                ),
                "turnover_shift_7d": _finite_float(
                    row.turnover_shift_7d, "turnover_shift_7d"
                ),
                "reservoir_z": _finite_float(row.reservoir_z, "reservoir_z"),
                "turnover_z": _finite_float(row.turnover_z, "turnover_z"),
                "activity_z": _finite_float(row.activity_z, "activity_z"),
                "spread_z": _finite_float(row.spread_z, "spread_z"),
                "source_lag_days": _finite_float(
                    row.source_lag_days, "source_lag_days"
                ),
                "reservoir_reference_count": int(
                    row.reservoir_reference_count
                ),
                "turnover_reference_count": int(row.turnover_reference_count),
                "activity_reference_count": int(row.activity_reference_count),
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
    support = support_summary(clock, cfg)
    source_sha = sha256_file(SOURCE_DATA)
    source_manifest_hash = source_manifest["manifest_hash"]
    records = event_records(clock)
    clock_hash = event_clock_hash(
        records,
        cfg=cfg,
        preregistration_hash=preregistration["artifact_hash"],
        source_manifest_hash=source_manifest_hash,
        source_sha256=source_sha,
    )
    event_clock_path = Path(cfg.event_clock_output)
    if support["passed"]:
        clock_core = {
            "protocol_version": "address_reservoir_capacitance_event_clock_v1",
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
        _write_json(event_clock_path, clock_artifact)
    else:
        event_clock_path.unlink(missing_ok=True)

    core = {
        "protocol_version": "address_reservoir_capacitance_support_v1",
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
            "first_observation": _timestamp(source["observation_date"].min()),
            "last_observation": _timestamp(source["observation_date"].max()),
            "market_or_funding_rows_loaded": 0,
            "return_or_pnl_fields_loaded": 0,
            "post_2023_source_rows_loaded": 0,
        },
        "feature_support": {
            "finite_zscore_rows": int(
                (
                    np.isfinite(features["reservoir_z"])
                    & np.isfinite(features["turnover_z"])
                    & np.isfinite(features["activity_z"])
                ).sum()
            ),
            "nonzero_state_rows": int(features["state_side"].ne(0).sum()),
            "event_onsets_before_nonoverlap": int(features["event"].sum()),
            "accepted_split_contained_nonoverlap_events": len(clock),
        },
        "support_gate": support,
        "event_clock": {
            "written": support["passed"],
            "path": cfg.event_clock_output if support["passed"] else None,
            "event_clock_hash": clock_hash if support["passed"] else None,
            "rows": len(clock) if support["passed"] else 0,
        },
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
