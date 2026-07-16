"""Freeze the 2020-2022 CVTT feature, market, and funding source prefixes."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.export_coinbase_spot_leadership_sources import (
    expected_grid,
    git_commit_for,
    range_frame,
    raw_input_metadata,
    resolve_existing,
    validate_funding,
)
from training.export_tail_arrival_selection_prefix import validate_market
from training.export_wikimedia_attention_source import (
    deterministic_gzip_csv,
    sha256_file,
)
from training.preregister_cross_venue_temporal_torsion_alpha import (
    DEFAULT_OUTPUT as DEFAULT_PREREGISTRATION,
    EXPECTED_FEATURE_SHA256,
    EXPECTED_SOURCE_AUDIT_SHA256,
    EXPECTED_SOURCE_MANIFEST_SHA256,
    SELECTION_END,
    canonical_hash,
    validate_manifest as validate_preregistration,
)


DEFAULT_FEATURE_INPUT = (
    "/home/pakchu/rllm/data/binance_cross_venue_minute_leadership_btc_2020_2023/"
    "BTCUSDT_cross_venue_minute_leadership_5m_2020-01_2023-12.csv.gz"
)
DEFAULT_FEATURE_MANIFEST = (
    "/home/pakchu/rllm/data/binance_cross_venue_minute_leadership_btc_2020_2023/"
    "build_manifest.json"
)
DEFAULT_SOURCE_AUDIT = (
    "results/binance_cross_venue_minute_leadership_audit_2026-07-14.json"
)
DEFAULT_MARKET_INPUT = (
    "/home/pakchu/rllm/data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)
DEFAULT_MARKET_MANIFEST = (
    "/home/pakchu/rllm/data/binance_um_kline_reference_btc_2020_2023/"
    "build_manifest.json"
)
DEFAULT_FUNDING_INPUT = (
    "/home/pakchu/rllm/data/binance_um_aux_btc_2020_2026/"
    "BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
)
DEFAULT_FEATURE_OUTPUT = "data/cross_venue_temporal_torsion_features_2020_2022.csv.gz"
DEFAULT_MARKET_OUTPUT = "data/cross_venue_temporal_torsion_market_2020_2022.csv.gz"
DEFAULT_FUNDING_OUTPUT = "data/cross_venue_temporal_torsion_funding_2020_2022.csv.gz"
DEFAULT_MANIFEST = (
    "results/cross_venue_temporal_torsion_source_manifest_2026-07-16.json"
)
DEFAULT_START = "2020-01-01"
QUARANTINE_FOLLOWING_BARS = 24
FEATURE_COLUMNS = [
    "date",
    "feature_available_time_utc",
    "trade_earliest_time_utc",
    "spot_flow_fraction",
    "um_flow_fraction",
    "spot_log_return_5m",
    "um_log_return_5m",
    "spot_flow_time_centroid",
    "um_flow_time_centroid",
    "spot_return_time_centroid",
    "um_return_time_centroid",
    "source_complete",
    "cross_venue_feature_valid",
    "feature_invalid_reason",
]
SIGNAL_COLUMNS = [
    "spot_flow_fraction",
    "um_flow_fraction",
    "spot_log_return_5m",
    "um_log_return_5m",
    "spot_flow_time_centroid",
    "um_flow_time_centroid",
    "spot_return_time_centroid",
    "um_return_time_centroid",
]


@dataclass(frozen=True)
class Config:
    feature_input: str = DEFAULT_FEATURE_INPUT
    feature_manifest: str = DEFAULT_FEATURE_MANIFEST
    source_audit: str = DEFAULT_SOURCE_AUDIT
    market_input: str = DEFAULT_MARKET_INPUT
    market_manifest: str = DEFAULT_MARKET_MANIFEST
    funding_input: str = DEFAULT_FUNDING_INPUT
    preregistration: str = DEFAULT_PREREGISTRATION
    feature_output: str = DEFAULT_FEATURE_OUTPUT
    market_output: str = DEFAULT_MARKET_OUTPUT
    funding_output: str = DEFAULT_FUNDING_OUTPUT
    manifest_output: str = DEFAULT_MANIFEST
    start: str = DEFAULT_START
    end: str = SELECTION_END


def parse_bool(values: pd.Series, column: str) -> pd.Series:
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if not normalized.isin(mapping).all():
        raise ValueError(f"invalid boolean value in {column}")
    return normalized.map(mapping).astype(bool)


def blank_or_na(value: object) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip().lower() in {"", "na", "nan", "none", "null"}


def quarantine(invalid: pd.Series) -> pd.Series:
    return invalid.astype(bool).rolling(
        QUARANTINE_FOLLOWING_BARS + 1, min_periods=1
    ).max().astype(bool)


def validate_features(raw: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    attrs = dict(raw.attrs)
    frame = raw.sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any() or not frame["date"].equals(pd.Series(grid)):
        raise RuntimeError("CVTT feature prefix is not the exact five-minute grid")
    for column in ("feature_available_time_utc", "trade_earliest_time_utc"):
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    expected_available = frame["date"] + pd.Timedelta("5min")
    if not frame["feature_available_time_utc"].equals(expected_available):
        raise RuntimeError("CVTT feature availability is not bucket close time")
    if not frame["trade_earliest_time_utc"].equals(expected_available):
        raise RuntimeError("upstream CVTT earliest-trade timestamp differs from availability")

    frame["source_complete"] = parse_bool(frame["source_complete"], "source_complete")
    frame["cross_venue_feature_valid"] = parse_bool(
        frame["cross_venue_feature_valid"], "cross_venue_feature_valid"
    )
    valid_current = (
        frame["source_complete"]
        & frame["cross_venue_feature_valid"]
        & frame["feature_invalid_reason"].eq("ok")
    )
    raw_invalid = frame.loc[~valid_current, SIGNAL_COLUMNS]
    if not raw_invalid.apply(lambda values: values.map(blank_or_na)).all().all():
        raise ValueError("invalid CVTT source row retained a raw signal descriptor")
    numeric = pd.DataFrame(np.nan, index=frame.index, columns=SIGNAL_COLUMNS)
    for column in SIGNAL_COLUMNS:
        numeric.loc[valid_current, column] = pd.to_numeric(
            frame.loc[valid_current, column], errors="raise"
        )
    frame.loc[:, SIGNAL_COLUMNS] = numeric
    valid_values = frame.loc[valid_current, SIGNAL_COLUMNS]
    if not np.isfinite(valid_values.to_numpy(float)).all():
        raise ValueError("valid CVTT source row contains a non-finite descriptor")
    centroid_columns = [column for column in SIGNAL_COLUMNS if "centroid" in column]
    if not frame.loc[valid_current, centroid_columns].apply(
        lambda values: values.between(0.0, 1.0)
    ).all().all():
        raise ValueError("CVTT centroid lies outside [0,1]")
    flow_columns = ["spot_flow_fraction", "um_flow_fraction"]
    if not frame.loc[valid_current, flow_columns].apply(
        lambda values: values.between(-1.0, 1.0)
    ).all().all():
        raise ValueError("CVTT flow fraction lies outside [-1,1]")

    quarantined = quarantine(~valid_current)
    frame.loc[quarantined, SIGNAL_COLUMNS] = np.nan
    frame["source_valid_current"] = valid_current.astype(np.int8)
    frame["source_quarantined"] = quarantined.astype(np.int8)
    frame["source_available"] = (~quarantined).astype(np.int8)
    selection_reason = pd.Series("ok", index=frame.index, dtype="object")
    selection_reason.loc[~valid_current] = (
        "upstream:" + frame.loc[~valid_current, "feature_invalid_reason"].astype(str)
    )
    selection_reason.loc[quarantined & valid_current] = (
        "post_invalid_24bar_quarantine"
    )
    frame["selection_invalid_reason"] = selection_reason
    frame["source_complete"] = frame["source_complete"].astype(np.int8)
    frame["cross_venue_feature_valid"] = frame[
        "cross_venue_feature_valid"
    ].astype(np.int8)
    frame["strategy_entry_earliest_time_utc"] = frame["date"] + pd.Timedelta(
        "10min"
    )
    frame = frame.drop(columns=["trade_earliest_time_utc"])
    frame.attrs.update(attrs)
    return frame


def validate_upstream_sources(cfg: Config) -> dict[str, dict[str, Any]]:
    feature_path = resolve_existing(cfg.feature_input)
    feature_manifest_path = resolve_existing(cfg.feature_manifest)
    audit_path = resolve_existing(cfg.source_audit)
    if sha256_file(feature_path) != EXPECTED_FEATURE_SHA256:
        raise RuntimeError("CVTT feature source differs from preregistered hash")
    if sha256_file(feature_manifest_path) != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise RuntimeError("CVTT feature manifest differs from preregistered hash")
    if sha256_file(audit_path) != EXPECTED_SOURCE_AUDIT_SHA256:
        raise RuntimeError("CVTT source audit differs from preregistered hash")
    feature_manifest = json.loads(feature_manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    if feature_manifest.get("combined_sha256") != EXPECTED_FEATURE_SHA256:
        raise RuntimeError("CVTT combined source hash does not match manifest")
    if feature_manifest.get("protocol", {}).get("outcomes_opened") is not False:
        raise RuntimeError("CVTT upstream feature build opened outcomes")
    if audit.get("passed") is not True or audit.get("failed_checks") != []:
        raise RuntimeError("CVTT upstream audit did not pass cleanly")
    if (
        audit.get("manifest_diagnostics", {}).get("combined_sha256")
        != EXPECTED_FEATURE_SHA256
    ):
        raise RuntimeError("CVTT audit and feature source hash disagree")

    market_manifest_path = resolve_existing(cfg.market_manifest)
    market_path = resolve_existing(cfg.market_input)
    market_manifest = json.loads(market_manifest_path.read_text())
    if market_manifest.get("protocol", {}).get("outcomes_opened") is not False:
        raise RuntimeError("CVTT upstream market manifest opened outcomes")
    if sha256_file(market_path) != market_manifest.get("combined_sha256"):
        raise RuntimeError("CVTT market source hash does not match manifest")
    return {"feature": feature_manifest, "audit": audit, "market": market_manifest}


def run(cfg: Config) -> dict[str, Any]:
    if cfg.start != DEFAULT_START or cfg.end != SELECTION_END:
        raise RuntimeError("CVTT source freezer is locked to 2020-2022")
    prereg_path = resolve_existing(cfg.preregistration)
    prereg = json.loads(prereg_path.read_text())
    validate_preregistration(prereg)
    upstream = validate_upstream_sources(cfg)
    grid = expected_grid(cfg.start, cfg.end)

    raw_features = range_frame(
        resolve_existing(cfg.feature_input),
        date_column="date",
        start=cfg.start,
        end=cfg.end,
        usecols=FEATURE_COLUMNS,
    )
    features = validate_features(raw_features, grid)
    raw_market = range_frame(
        resolve_existing(cfg.market_input),
        date_column="date",
        start=cfg.start,
        end=cfg.end,
        usecols=["date", "open", "high", "low", "close"],
    )
    market = validate_market(raw_market, grid)
    raw_funding = range_frame(
        resolve_existing(cfg.funding_input),
        date_column="date",
        start=cfg.start,
        end=cfg.end,
        usecols=["date", "funding_rate"],
    )
    funding = validate_funding(raw_funding)

    deterministic_gzip_csv(features, cfg.feature_output)
    deterministic_gzip_csv(market, cfg.market_output)
    deterministic_gzip_csv(funding, cfg.funding_output)
    outputs: dict[str, Any] = {}
    for name, path, rows in (
        ("features", cfg.feature_output, len(features)),
        ("market", cfg.market_output, len(market)),
        ("funding", cfg.funding_output, len(funding)),
    ):
        outputs[name] = {
            "path": path,
            "rows": int(rows),
            "bytes": Path(path).stat().st_size,
            "sha256": sha256_file(path),
        }

    availability = features["source_available"].eq(1)
    core: dict[str, Any] = {
        "protocol_version": "cross_venue_temporal_torsion_source_v1",
        "phase": "selection_inputs_only_2020_2022",
        "forward_trade_outcomes_opened": False,
        "start_inclusive": cfg.start,
        "end_exclusive": cfg.end,
        "preregistration": {
            "path": str(prereg_path),
            "file_sha256": sha256_file(prereg_path),
            "manifest_hash": prereg["manifest_hash"],
            "git_commit": git_commit_for(cfg.preregistration),
        },
        "upstream_manifests": {
            "features": {
                "path": cfg.feature_manifest,
                "sha256": sha256_file(resolve_existing(cfg.feature_manifest)),
                "combined_sha256": upstream["feature"]["combined_sha256"],
            },
            "source_audit": {
                "path": cfg.source_audit,
                "sha256": sha256_file(resolve_existing(cfg.source_audit)),
                "passed": upstream["audit"]["passed"],
            },
            "market": {
                "path": cfg.market_manifest,
                "sha256": sha256_file(resolve_existing(cfg.market_manifest)),
                "combined_sha256": upstream["market"]["combined_sha256"],
            },
        },
        "raw_inputs": {
            "features": raw_input_metadata(cfg.feature_input),
            "market": raw_input_metadata(cfg.market_input),
            "funding": raw_input_metadata(cfg.funding_input),
        },
        "prefix_materialization_contract": {
            "reader": "chronological raw-line stream with first date field parsed alone",
            "stop": "before parsing any non-date field at or after 2023-01-01",
            "future_non_date_fields_csv_parsed": 0,
            "feature_cutoff_sentinel_date": raw_features.attrs.get(
                "cutoff_sentinel_date"
            ),
            "market_cutoff_sentinel_date": raw_market.attrs.get(
                "cutoff_sentinel_date"
            ),
            "funding_cutoff_sentinel_date": raw_funding.attrs.get(
                "cutoff_sentinel_date"
            ),
        },
        "quality": {
            "expected_five_minute_rows": len(grid),
            "source_valid_current_rows": int(features["source_valid_current"].sum()),
            "source_quarantined_rows": int(features["source_quarantined"].sum()),
            "source_available_rows": int(availability.sum()),
            "source_unavailable_rows": int((~availability).sum()),
            "source_available_by_year": {
                str(year): int(
                    (availability & features["date"].dt.year.eq(year)).sum()
                )
                for year in (2020, 2021, 2022)
            },
            "market_complete_rows": len(market),
            "funding_rows": len(funding),
            "funding_maximum_grid_offset_seconds": funding.attrs.get(
                "maximum_grid_offset_seconds"
            ),
        },
        "outputs": outputs,
        "source_freezer": {
            "code_sha256": sha256_file(__file__),
            "git_commit": git_commit_for(__file__),
        },
        "future_data_requested": False,
    }
    payload = {
        **core,
        "manifest_hash": canonical_hash(core),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output = Path(cfg.manifest_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Config()
    for name, value in asdict(defaults).items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    return parser.parse_args()


def main() -> None:
    payload = run(Config(**vars(parse_args())))
    print(
        json.dumps(
            {
                "manifest_hash": payload["manifest_hash"],
                "quality": payload["quality"],
                "outputs": payload["outputs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
