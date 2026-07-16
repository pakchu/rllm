"""Freeze 2020-2022 tail-arrival feature, market, and funding prefixes."""
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
    git_commit_for,
    range_frame,
    raw_input_metadata,
    resolve_existing,
    validate_funding,
)
from training.export_wikimedia_attention_source import (
    deterministic_gzip_csv,
    sha256_file,
)
from training.preregister_tail_arrival_absorption_alpha import (
    DEFAULT_OUTPUT as DEFAULT_PREREGISTRATION,
    SELECTION_END,
    canonical_hash,
    validate_manifest as validate_preregistration,
)


DEFAULT_FEATURE_INPUT = (
    "data/binance_um_aggtrade_microstructure_btc_2020_2023/"
    "BTCUSDT_aggtrade_5m_2020-01-01_2023-12-31.csv.gz"
)
DEFAULT_FEATURE_MANIFEST = (
    "data/binance_um_aggtrade_microstructure_btc_2020_2023/build_manifest.json"
)
DEFAULT_MARKET_INPUT = (
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)
DEFAULT_MARKET_MANIFEST = "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
DEFAULT_FUNDING_INPUT = (
    "/home/pakchu/rllm/data/binance_um_aux_btc_2020_2026/"
    "BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
)
DEFAULT_FEATURE_OUTPUT = "data/tail_arrival_features_2020_2022.csv.gz"
DEFAULT_MARKET_OUTPUT = "data/tail_arrival_market_2020_2022.csv.gz"
DEFAULT_FUNDING_OUTPUT = "data/tail_arrival_funding_2020_2022.csv.gz"
DEFAULT_MANIFEST = "results/tail_arrival_source_manifest_2026-07-16.json"
DEFAULT_START = "2020-01-01"
FEATURE_COLUMNS = [
    "date",
    "agg_trade_count",
    "event_notional_mean",
    "event_notional_std",
    "event_notional_p50",
    "event_notional_p90",
    "event_notional_p99",
    "event_notional_max",
    "interarrival_mean_ms",
    "interarrival_std_ms",
    "buy_sell_event_size_log_ratio",
    "micro_log_return",
]


@dataclass(frozen=True)
class Config:
    feature_input: str = DEFAULT_FEATURE_INPUT
    feature_manifest: str = DEFAULT_FEATURE_MANIFEST
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


def expected_grid(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="5min", inclusive="left")


def source_gap_days(manifest: dict[str, Any], cutoff: str) -> set[str]:
    cutoff_date = pd.Timestamp(cutoff).date()
    archives = [
        archive
        for month in manifest.get("months", [])
        for archive in month.get("archives", [])
        if pd.Timestamp(archive["date"]).date() < cutoff_date
    ]
    gaps: set[str] = set()
    for archive in archives:
        missing_ids = (
            int(archive["last_agg_trade_id"])
            - int(archive["first_agg_trade_id"])
            + 1
            - int(archive["agg_trade_rows"])
        )
        if missing_ids > 0:
            gaps.add(str(archive["date"]))
        if missing_ids < 0:
            raise RuntimeError("aggregate trade archive has negative ID gap")
    for previous, current in zip(archives, archives[1:]):
        delta = int(current["first_agg_trade_id"]) - int(previous["last_agg_trade_id"]) - 1
        if delta > 0:
            gaps.update({str(previous["date"]), str(current["date"])})
        if delta < 0:
            raise RuntimeError("aggregate trade IDs overlap across source days")
    return gaps


def normalize_features(
    raw: pd.DataFrame,
    grid: pd.DatetimeIndex,
    gap_days: set[str],
) -> pd.DataFrame:
    raw = raw.sort_values("date").reset_index(drop=True)
    if raw["date"].duplicated().any() or not raw["date"].isin(grid).all():
        raise RuntimeError("tail-arrival source timestamps are invalid")
    for column in FEATURE_COLUMNS[1:]:
        raw[column] = pd.to_numeric(raw[column], errors="raise")
        if not np.isfinite(raw[column]).all():
            raise ValueError(f"tail-arrival source contains non-finite {column}")
    positive = [
        "event_notional_mean",
        "event_notional_p50",
        "event_notional_p90",
        "event_notional_p99",
        "event_notional_max",
    ]
    if (raw[positive] <= 0.0).any().any():
        raise ValueError("tail-arrival event notionals must be positive")
    if (raw[["event_notional_std", "interarrival_mean_ms", "interarrival_std_ms"]] < 0.0).any().any():
        raise ValueError("tail-arrival dispersion/timing values must be nonnegative")
    if (
        (raw["event_notional_p50"] > raw["event_notional_p90"]).any()
        or (raw["event_notional_p90"] > raw["event_notional_p99"]).any()
        or (raw["event_notional_p99"] > raw["event_notional_max"]).any()
    ):
        raise ValueError("tail-arrival event-notional quantiles are incoherent")
    if (raw["agg_trade_count"] < 1).any():
        raise ValueError("complete aggTrade bars must contain at least one event")
    normalized = raw.set_index("date").reindex(grid)
    normalized.index.name = "date"
    normalized.insert(0, "source_complete", normalized["agg_trade_count"].notna().astype(np.int8))
    normalized.insert(
        1,
        "source_gap_day",
        pd.Series(grid.strftime("%Y-%m-%d").isin(gap_days).astype(np.int8), index=grid),
    )
    return normalized.reset_index()


def validate_market(raw: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    frame = raw.sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any() or not frame["date"].equals(pd.Series(grid)):
        raise RuntimeError("tail-arrival market is not the exact 5m prefix grid")
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all() or (frame[column] <= 0).any():
            raise ValueError(f"invalid tail-arrival market column: {column}")
    if (frame["low"] > frame[["open", "close"]].min(axis=1)).any():
        raise ValueError("market low is incoherent")
    if (frame["high"] < frame[["open", "close"]].max(axis=1)).any():
        raise ValueError("market high is incoherent")
    return frame


def verify_upstream(path: str | Path, output_path: Path) -> dict[str, Any]:
    manifest = json.loads(resolve_existing(path).read_text())
    if manifest.get("protocol", {}).get("outcomes_opened") is not False:
        raise RuntimeError("upstream source manifest did not preserve unopened outcomes")
    if sha256_file(output_path) != manifest.get("combined_sha256"):
        raise RuntimeError("upstream combined source hash mismatch")
    return manifest


def run(cfg: Config) -> dict[str, Any]:
    if cfg.start != DEFAULT_START or cfg.end != SELECTION_END:
        raise RuntimeError("tail-arrival source prefix is locked to 2020-2022")
    prereg_path = resolve_existing(cfg.preregistration)
    prereg = json.loads(prereg_path.read_text())
    validate_preregistration(prereg)
    feature_input = resolve_existing(cfg.feature_input)
    market_input = resolve_existing(cfg.market_input)
    funding_input = resolve_existing(cfg.funding_input)
    feature_manifest = verify_upstream(cfg.feature_manifest, feature_input)
    market_manifest = verify_upstream(cfg.market_manifest, market_input)
    grid = expected_grid(cfg.start, cfg.end)
    gap_days = source_gap_days(feature_manifest, cfg.end)
    feature_raw = range_frame(
        feature_input,
        date_column="date",
        start=cfg.start,
        end=cfg.end,
        usecols=FEATURE_COLUMNS,
    )
    features = normalize_features(feature_raw, grid, gap_days)
    market_raw = range_frame(
        market_input,
        date_column="date",
        start=cfg.start,
        end=cfg.end,
        usecols=["date", "open", "high", "low", "close"],
    )
    market = validate_market(market_raw, grid)
    funding_raw = range_frame(
        funding_input,
        date_column="date",
        start=cfg.start,
        end=cfg.end,
        usecols=["date", "funding_rate"],
    )
    funding = validate_funding(funding_raw)
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
    core: dict[str, Any] = {
        "protocol_version": "tail_arrival_selection_source_v1",
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
            "feature": {
                "path": cfg.feature_manifest,
                "sha256": sha256_file(resolve_existing(cfg.feature_manifest)),
                "combined_sha256": feature_manifest["combined_sha256"],
            },
            "market": {
                "path": cfg.market_manifest,
                "sha256": sha256_file(resolve_existing(cfg.market_manifest)),
                "combined_sha256": market_manifest["combined_sha256"],
            },
        },
        "raw_inputs": {
            "features": raw_input_metadata(cfg.feature_input),
            "market": raw_input_metadata(cfg.market_input),
            "funding": raw_input_metadata(cfg.funding_input),
        },
        "prefix_materialization_contract": {
            "reader": "chronological raw-line stream with first date field parsed alone",
            "stop": "before CSV parsing any non-date field at or after 2023-01-01",
            "future_non_date_fields_csv_parsed": 0,
            "feature_cutoff_sentinel_date": feature_raw.attrs.get("cutoff_sentinel_date"),
            "market_cutoff_sentinel_date": market_raw.attrs.get("cutoff_sentinel_date"),
            "funding_cutoff_sentinel_date": funding_raw.attrs.get("cutoff_sentinel_date"),
        },
        "quality": {
            "expected_five_minute_rows": len(grid),
            "feature_complete_rows": int(features["source_complete"].sum()),
            "feature_missing_rows": int(features["source_complete"].eq(0).sum()),
            "source_gap_days": sorted(gap_days),
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
