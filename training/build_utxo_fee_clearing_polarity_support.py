"""Build outcome-blind UFCP support clocks from hash-bound Bitcoin UTXO fee source.

The builder intentionally reads only the preregistered UTXO source CSV and never
opens market, funding, return, or incidence inputs.  It freezes the primary
calendar clock plus diagnostic/control clocks needed before any outcome review.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
import pandas as pd

from training import preregister_utxo_fee_clearing_polarity as prereg


POLICY_ID = "UFCP-1"
PROTOCOL_VERSION = "utxo_fee_clearing_polarity_support_v1"
DEFAULT_PREREGISTRATION = Path("results/utxo_fee_clearing_polarity_preregistration_2026-07-20.json")
DEFAULT_OUTPUT = Path("results/utxo_fee_clearing_polarity_support_2026-07-20.json")
DEFAULT_PRIMARY_CLOCK = Path("results/utxo_fee_clearing_polarity_primary_clock_2026-07-20.csv")
DEFAULT_CONTROL_CLOCKS = Path("results/utxo_fee_clearing_polarity_control_clocks_2026-07-20.csv.gz")
SUPPORT_BUILDER = Path("training/build_utxo_fee_clearing_polarity_support.py")
EXPECTED_PREREGISTRATION_SHA256 = "160efdd2eb857c47a80ec0ed4a976a659a1ee3dd3c930093d197798e619d65c9"
EXPECTED_PREREGISTRATION_MANIFEST_HASH = "95cd5911171b033923603d5d845949df6a7ef28020f2591c41ab1e0d1293da5b"
EXPECTED_POLICY_HASH = "9945815de1e3f88ab1d59e2dec7dc7923294c91807cea8071f10e55c60a0daef"
EXPECTED_PREREGISTRATION_SOURCE_SHA256 = "10311773ae2e9baac1b27d537bcab7cf5b8687d7dc84a0dd1b0bcd8d2673fa05"
SOURCE_COLUMNS = list(prereg.SOURCE_COLUMNS)
CLOCK_COLUMNS = [
    "policy_id",
    "clock",
    "source_day",
    "available_time",
    "entry_time",
    "exit_time",
    "side",
    "source_start_height",
    "source_end_height",
    "successor_end_height",
    "block_count",
    "edges",
    "fees",
    "fee_burden",
    "utxo_polarity",
    "fee_rank",
    "polarity_rank",
]
CONTROL_NAMES = [
    "direction_flip",
    "constant_long_same_clock",
    "constant_short_same_clock",
    "topology_only",
    "low_fee_mirror",
    "stale_7d",
    "year_side_stratified_random_clock",
    "one_bar_delayed_entry",
]
OUTCOME_BOUNDARY = {
    "market_rows_loaded": 0,
    "funding_rows_loaded": 0,
    "return_rows_loaded": 0,
    "market_values_read": 0,
    "funding_values_read": 0,
    "profit_loss_fields": 0,
}
FORBIDDEN_SOURCE_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "price",
    "return",
    "returns",
    "pnl",
    "funding",
    "funding_rate",
    "incidence",
}


@dataclass(frozen=True)
class Config:
    preregistration: str = str(DEFAULT_PREREGISTRATION)
    output: str = str(DEFAULT_OUTPUT)
    primary_clock: str = str(DEFAULT_PRIMARY_CLOCK)
    control_clocks: str = str(DEFAULT_CONTROL_CLOCKS)


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("UFCP preregistration must be a JSON object")
    return payload


def validate_preregistration(path: str | Path) -> dict[str, Any]:
    artifact = _read_json(path)
    core = {key: value for key, value in artifact.items() if key != "manifest_hash"}
    if artifact.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("UFCP preregistration manifest hash mismatch")
    if artifact.get("protocol_version") != prereg.PROTOCOL_VERSION:
        raise RuntimeError("UFCP preregistration protocol drift")
    if artifact.get("policy_id") != POLICY_ID or artifact.get("policy") != prereg.policy():
        raise RuntimeError("UFCP preregistration policy drift")
    if artifact.get("outcomes_opened") is not False:
        raise RuntimeError("UFCP preregistration opened outcomes")
    if artifact.get("outcome_boundary") != prereg.PREREGISTRATION_OUTCOME_BOUNDARY:
        raise RuntimeError("UFCP preregistration outcome boundary drift")
    source_manifest = artifact.get("source_manifest")
    if not isinstance(source_manifest, dict):
        raise RuntimeError("UFCP preregistration source binding missing")
    source_output = source_manifest.get("source_output")
    if not isinstance(source_output, dict):
        raise RuntimeError("UFCP preregistration source output binding missing")
    if source_output.get("columns") != SOURCE_COLUMNS:
        raise RuntimeError("UFCP source schema binding drift")
    source_path = source_output.get("path")
    source_sha = source_output.get("sha256")
    if not isinstance(source_path, str) or not isinstance(source_sha, str):
        raise RuntimeError("UFCP source path/hash binding missing")
    return artifact


def validate_frozen_preregistration(path: str | Path) -> dict[str, Any]:
    preregistration_path = Path(path)
    if preregistration_path.resolve() != DEFAULT_PREREGISTRATION.resolve():
        raise RuntimeError("UFCP preregistration path differs from the frozen artifact")
    if sha256_file(preregistration_path) != EXPECTED_PREREGISTRATION_SHA256:
        raise RuntimeError("UFCP frozen preregistration file SHA drift")
    artifact = validate_preregistration(preregistration_path)
    if artifact.get("manifest_hash") != EXPECTED_PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("UFCP frozen preregistration manifest-hash drift")
    if artifact.get("policy_hash") != EXPECTED_POLICY_HASH:
        raise RuntimeError("UFCP frozen policy-hash drift")
    if canonical_hash(artifact["policy"]) != EXPECTED_POLICY_HASH:
        raise RuntimeError("UFCP frozen policy content drift")
    expected_source = {
        "path": str(prereg.PREREGISTRATION_SOURCE),
        "sha256": EXPECTED_PREREGISTRATION_SOURCE_SHA256,
    }
    if artifact.get("preregistration_source") != expected_source:
        raise RuntimeError("UFCP preregistration source binding drift")
    if sha256_file(prereg.PREREGISTRATION_SOURCE) != EXPECTED_PREREGISTRATION_SOURCE_SHA256:
        raise RuntimeError("UFCP preregistration source file drift")
    source_manifest = artifact["source_manifest"]
    expected_source_output = {
        "path": str(prereg.EXPECTED_SOURCE_OUTPUT),
        "sha256": prereg.EXPECTED_SOURCE_OUTPUT_SHA256,
        "bytes": prereg.EXPECTED_SOURCE_OUTPUT_BYTES,
        "columns": SOURCE_COLUMNS,
    }
    if source_manifest.get("path") != str(prereg.SOURCE_MANIFEST):
        raise RuntimeError("UFCP frozen source-manifest path drift")
    if source_manifest.get("sha256") != prereg.EXPECTED_SOURCE_MANIFEST_SHA256:
        raise RuntimeError("UFCP frozen source-manifest file SHA drift")
    if source_manifest.get("manifest_hash") != prereg.EXPECTED_SOURCE_MANIFEST_HASH:
        raise RuntimeError("UFCP frozen source-manifest hash drift")
    if source_manifest.get("protocol_version") != prereg.SOURCE_PROTOCOL_VERSION:
        raise RuntimeError("UFCP frozen source-manifest protocol drift")
    if source_manifest.get("source_output") != expected_source_output:
        raise RuntimeError("UFCP frozen source-output binding drift")
    audit = source_manifest.get("source_audit")
    expected_audit = {
        "observed_rows": prereg.FROZEN_ROWS,
        "start_height": prereg.FROZEN_START_HEIGHT,
        "end_height": prereg.FROZEN_END_HEIGHT,
        "latest_eligible_packet_end": prereg.FROZEN_END_HEIGHT - 6,
        "height_links_checked": prereg.FROZEN_ROWS - 1,
        "complete_inclusive_height_range": True,
        "unique_block_hashes": True,
        "all_rows_pre_cutoff": True,
        "utxo_identity_checked": True,
        "end_timestamp_exclusive": prereg.FROZEN_END_TIMESTAMP_EXCLUSIVE,
    }
    if audit != expected_audit:
        raise RuntimeError("UFCP frozen source audit drift")
    if prereg._validate_source_manifest(prereg.SOURCE_MANIFEST) != source_manifest:
        raise RuntimeError("UFCP frozen source manifest no longer revalidates")
    return artifact


def _source_output_binding(preregistration: dict[str, Any]) -> dict[str, Any]:
    return preregistration["source_manifest"]["source_output"]


def _validate_config(cfg: Config, preregistration: dict[str, Any]) -> None:
    writes = {
        "support output": Path(cfg.output),
        "primary clock": Path(cfg.primary_clock),
        "control clocks": Path(cfg.control_clocks),
    }
    if Path(cfg.output).suffix != ".json":
        raise ValueError("UFCP support output must be a JSON file")
    if Path(cfg.primary_clock).suffix != ".csv":
        raise ValueError("UFCP primary clock must be a CSV file")
    if not str(cfg.control_clocks).endswith(".csv.gz"):
        raise ValueError("UFCP control clocks must be a .csv.gz file")

    resolved_writes = {name: path.resolve() for name, path in writes.items()}
    if len(set(resolved_writes.values())) != len(resolved_writes):
        raise ValueError("UFCP output paths must be distinct")

    source_manifest = preregistration["source_manifest"]
    protected = {
        Path(cfg.preregistration).resolve(),
        Path(_source_output_binding(preregistration)["path"]).resolve(),
        Path(source_manifest["path"]).resolve(),
        SUPPORT_BUILDER.resolve(),
        prereg.PREREGISTRATION_SOURCE.resolve(),
        prereg.SOURCE_DECISION.resolve(),
        prereg.SOURCE_BUILDER.resolve(),
        prereg.EXPECTED_REFERENCE.resolve(),
    }
    for name, resolved in resolved_writes.items():
        if resolved in protected:
            raise ValueError(f"UFCP {name} path must not overwrite a frozen input")


def load_source_frame(preregistration: dict[str, Any]) -> pd.DataFrame:
    binding = _source_output_binding(preregistration)
    source_path = Path(binding["path"])
    if not source_path.is_file():
        raise RuntimeError("UFCP hash-bound source CSV is missing")
    if sha256_file(source_path) != binding["sha256"]:
        raise RuntimeError("UFCP hash-bound source CSV SHA mismatch")
    if binding.get("bytes") is not None and source_path.stat().st_size != binding["bytes"]:
        raise RuntimeError("UFCP hash-bound source CSV byte-size mismatch")
    header = pd.read_csv(source_path, nrows=0).columns.tolist()
    if header != SOURCE_COLUMNS:
        raise RuntimeError("UFCP source CSV schema drift")
    forbidden = FORBIDDEN_SOURCE_COLUMNS.intersection({column.lower() for column in header})
    if forbidden:
        raise RuntimeError(f"UFCP source CSV contains outcome-like columns: {sorted(forbidden)}")
    frame = pd.read_csv(source_path, usecols=SOURCE_COLUMNS, dtype={"id": "string", "previousblockhash": "string"})
    return validate_source_frame(frame)


def validate_source_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.columns.tolist() != SOURCE_COLUMNS:
        raise RuntimeError("UFCP source frame schema drift")
    if len(frame) == 0:
        raise RuntimeError("UFCP source frame is empty")
    out = frame.copy()
    int_columns = [
        "height",
        "timestamp",
        "mediantime",
        "tx_count",
        "size",
        "weight",
        "total_fees",
        "total_inputs",
        "total_outputs",
        "utxo_set_change",
    ]
    for column in int_columns:
        values = pd.to_numeric(out[column], errors="raise")
        if values.isna().any() or (values % 1).ne(0).any():
            raise RuntimeError(f"UFCP source {column} must be exact integers")
        out[column] = values.astype(np.int64)
    out["id"] = out["id"].astype(str)
    out["previousblockhash"] = out["previousblockhash"].astype(str)
    out = out.reset_index(drop=True)
    expected_heights = np.arange(int(out["height"].iloc[0]), int(out["height"].iloc[0]) + len(out), dtype=np.int64)
    if len(out) != len(expected_heights) or not np.array_equal(out["height"].to_numpy(), expected_heights):
        raise RuntimeError("UFCP source rows must be exact contiguous heights")
    if out["id"].duplicated().any():
        raise RuntimeError("UFCP source block ids must be unique")
    if not np.array_equal(out["previousblockhash"].iloc[1:].to_numpy(), out["id"].iloc[:-1].to_numpy()):
        raise RuntimeError("UFCP source hash-chain linkage failed")
    if not (out["total_outputs"] - out["total_inputs"]).eq(out["utxo_set_change"]).all():
        raise RuntimeError("UFCP source UTXO identity failed")
    if out[["timestamp", "mediantime", "tx_count", "size", "weight"]].le(0).any().any():
        raise RuntimeError("UFCP source contains non-positive block header fields")
    return out


def strict_prior_midrank(current: float, prior: Iterable[float]) -> float:
    values = list(prior)
    if not values:
        raise ValueError("strict-prior midrank requires at least one prior value")
    less = sum(value < current for value in values)
    equal = sum(value == current for value in values)
    return (less + 0.5 * equal) / len(values)


def build_daily_features(frame: pd.DataFrame) -> pd.DataFrame:
    source = validate_source_frame(frame)
    source["source_day"] = pd.to_datetime(source["timestamp"], unit="s", utc=True).dt.floor("D")
    grouped = source.groupby("source_day", sort=True, observed=True)
    rows: list[dict[str, Any]] = []
    for day, group in grouped:
        group = group.sort_values("height")
        final_index = int(group.index[-1])
        successor_index = final_index + 6
        successor_available = successor_index < len(source)
        edges = int((group["total_inputs"] + group["total_outputs"]).sum())
        fees = int(group["total_fees"].sum())
        block_count = int(len(group))
        valid = bool(block_count >= 72 and edges > 0 and fees > 0 and successor_available)
        rows.append(
            {
                "source_day": day,
                "source_start_height": int(group["height"].iloc[0]),
                "source_end_height": int(group["height"].iloc[-1]),
                "successor_end_height": int(source["height"].iloc[successor_index]) if successor_available else pd.NA,
                "block_count": block_count,
                "edges": edges,
                "fees": fees,
                "utxo_sum": int(group["utxo_set_change"].sum()),
                "valid_source_day": valid,
            }
        )
    daily = pd.DataFrame(rows).sort_values("source_day").reset_index(drop=True)
    if daily.empty:
        raise RuntimeError("UFCP source produced no UTC days")
    expected_days = pd.date_range(daily["source_day"].iloc[0], daily["source_day"].iloc[-1], freq="D", tz="UTC")
    if len(expected_days) != len(daily) or not daily["source_day"].reset_index(drop=True).eq(pd.Series(expected_days)).all():
        raise RuntimeError("UFCP source has a missing UTC day")
    daily["fee_burden"] = np.where(daily["valid_source_day"], np.log(daily["fees"] / daily["edges"]), np.nan)
    daily["utxo_polarity"] = np.where(daily["valid_source_day"], daily["utxo_sum"] / daily["edges"], np.nan)
    daily["available_time"] = daily["source_day"] + pd.Timedelta(days=2)
    daily["entry_time"] = daily["available_time"] + pd.Timedelta(minutes=5)
    daily["exit_time"] = daily["entry_time"] + pd.Timedelta(minutes=5 * 288)
    return daily


def attach_strict_prior_ranks(daily: pd.DataFrame, *, lookback: int = 180, minimum_prior: int = 120) -> pd.DataFrame:
    ranked = daily.sort_values("source_day").reset_index(drop=True).copy()
    fee_ranks: list[float] = []
    polarity_ranks: list[float] = []
    prior_fee: list[float] = []
    prior_pol: list[float] = []
    for row in ranked.itertuples(index=False):
        if bool(row.valid_source_day) and len(prior_fee) >= minimum_prior:
            fee_window = prior_fee[-lookback:]
            polarity_window = prior_pol[-lookback:]
            fee_ranks.append(strict_prior_midrank(float(row.fee_burden), fee_window))
            polarity_ranks.append(strict_prior_midrank(float(row.utxo_polarity), polarity_window))
        else:
            fee_ranks.append(np.nan)
            polarity_ranks.append(np.nan)
        if bool(row.valid_source_day):
            prior_fee.append(float(row.fee_burden))
            prior_pol.append(float(row.utxo_polarity))
    ranked["fee_rank"] = fee_ranks
    ranked["polarity_rank"] = polarity_ranks
    ranked["rank_ready"] = ranked["fee_rank"].notna() & ranked["polarity_rank"].notna()
    return ranked


def _side_from_ranks(fee_rank: float, polarity_rank: float, *, mode: str) -> int:
    if math.isnan(fee_rank) or math.isnan(polarity_rank):
        return 0
    if mode == "primary":
        if fee_rank >= 0.75 and polarity_rank >= 0.75:
            return 1
        if fee_rank >= 0.75 and polarity_rank <= 0.25:
            return -1
    elif mode == "topology_only":
        if polarity_rank >= 0.75:
            return 1
        if polarity_rank <= 0.25:
            return -1
    elif mode == "low_fee_mirror":
        if fee_rank <= 0.25 and polarity_rank >= 0.75:
            return 1
        if fee_rank <= 0.25 and polarity_rank <= 0.25:
            return -1
    else:
        raise ValueError(f"unknown side mode: {mode}")
    return 0


def _clock_from_candidates(candidates: pd.DataFrame, *, clock_name: str, side_mode: str = "primary") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    prior_exit: pd.Timestamp | None = None
    ordered = candidates.sort_values(["entry_time", "source_day"]).reset_index(drop=True)
    for row in ordered.itertuples(index=False):
        side = _side_from_ranks(float(row.fee_rank), float(row.polarity_rank), mode=side_mode)
        if side == 0:
            continue
        entry = row.entry_time
        exit_time = row.exit_time
        if prior_exit is not None and entry < prior_exit:
            continue
        record = {column: getattr(row, column) for column in CLOCK_COLUMNS if column not in {"policy_id", "clock", "side"}}
        record.update({"policy_id": POLICY_ID, "clock": clock_name, "side": side})
        rows.append(record)
        prior_exit = exit_time
    clock = pd.DataFrame(rows, columns=CLOCK_COLUMNS)
    return clock


def build_primary_clock(ranked_daily: pd.DataFrame) -> pd.DataFrame:
    eligible = ranked_daily[ranked_daily["valid_source_day"] & ranked_daily["rank_ready"]].copy()
    return _clock_from_candidates(eligible, clock_name="primary", side_mode="primary")


def _same_calendar(primary: pd.DataFrame, clock_name: str, side_values: Iterable[int]) -> pd.DataFrame:
    control = primary.copy()
    control["clock"] = clock_name
    control["side"] = list(side_values)
    return control[CLOCK_COLUMNS]


def _random_clock(ranked_daily: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    pool = ranked_daily[ranked_daily["valid_source_day"] & ranked_daily["rank_ready"]].copy()
    rng = random.Random(20260720)
    selected_frames: list[pd.DataFrame] = []
    for (year, side), group in primary.groupby([primary["entry_time"].dt.year, "side"], sort=True):
        candidates = pool[pool["entry_time"].dt.year == int(year)].copy()
        if len(candidates) < len(group):
            raise RuntimeError("UFCP random control cannot preserve year/side counts without replacement")
        indexes = rng.sample(list(candidates.index), len(group))
        sampled = pool.loc[indexes].copy()
        sampled["side"] = int(side)
        selected_frames.append(sampled)
        pool = pool.drop(index=indexes)
    if not selected_frames:
        return pd.DataFrame(columns=CLOCK_COLUMNS)
    control = pd.concat(selected_frames, ignore_index=True).sort_values(["entry_time", "source_day"]).reset_index(drop=True)
    control["clock"] = "year_side_stratified_random_clock"
    control["policy_id"] = POLICY_ID
    return control[CLOCK_COLUMNS]

def build_control_clocks(ranked_daily: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    controls: list[pd.DataFrame] = []
    controls.append(_same_calendar(primary, "direction_flip", -primary["side"].astype(int)))
    controls.append(_same_calendar(primary, "constant_long_same_clock", [1] * len(primary)))
    controls.append(_same_calendar(primary, "constant_short_same_clock", [-1] * len(primary)))
    controls.append(_clock_from_candidates(ranked_daily[ranked_daily["valid_source_day"] & ranked_daily["rank_ready"]], clock_name="topology_only", side_mode="topology_only"))
    controls.append(_clock_from_candidates(ranked_daily[ranked_daily["valid_source_day"] & ranked_daily["rank_ready"]], clock_name="low_fee_mirror", side_mode="low_fee_mirror"))

    stale = ranked_daily.copy()
    stale[["fee_rank", "polarity_rank"]] = stale[["fee_rank", "polarity_rank"]].shift(7)
    stale["rank_ready"] = stale["fee_rank"].notna() & stale["polarity_rank"].notna()
    controls.append(_clock_from_candidates(stale[stale["valid_source_day"] & stale["rank_ready"]], clock_name="stale_7d", side_mode="primary"))
    controls.append(_random_clock(ranked_daily, primary))
    delayed = primary.copy()
    delayed["clock"] = "one_bar_delayed_entry"
    delayed["entry_time"] = delayed["entry_time"] + pd.Timedelta(minutes=5)
    delayed["exit_time"] = delayed["exit_time"] + pd.Timedelta(minutes=5)
    controls.append(delayed[CLOCK_COLUMNS])
    combined = pd.concat(controls, ignore_index=True) if controls else pd.DataFrame(columns=CLOCK_COLUMNS)
    observed = set(combined["clock"].unique()) if len(combined) else set()
    if observed != set(CONTROL_NAMES):
        raise RuntimeError("UFCP control-count drift")
    return combined[CLOCK_COLUMNS].sort_values(["clock", "entry_time", "source_day"]).reset_index(drop=True)


def _window(entry: pd.Series, name: str) -> pd.Series:
    if name == "train":
        return (entry >= pd.Timestamp("2021-01-01T00:00:00Z")) & (entry < pd.Timestamp("2023-01-01T00:00:00Z"))
    if name == "selection":
        return (entry >= pd.Timestamp("2023-01-01T00:00:00Z")) & (entry < pd.Timestamp("2024-01-01T00:00:00Z"))
    raise ValueError(name)


def _side_share_ok(frame: pd.DataFrame) -> bool:
    if len(frame) == 0:
        return False
    shares = frame["side"].value_counts(normalize=True)
    return all(0.25 <= float(shares.get(side, 0.0)) <= 0.75 for side in (-1, 1))


def _max_month_share(frame: pd.DataFrame) -> float:
    if len(frame) == 0:
        return 1.0
    month = frame["entry_time"].dt.strftime("%Y-%m")
    return float(month.value_counts(normalize=True).max())


def _has_contiguous_utc_days(daily: pd.DataFrame) -> bool:
    if "source_day" not in daily.columns or daily.empty:
        return False
    days = pd.to_datetime(daily["source_day"], utc=True).sort_values().reset_index(drop=True)
    if days.duplicated().any():
        return False
    expected = pd.Series(pd.date_range(days.iloc[0], days.iloc[-1], freq="D", tz="UTC"))
    return bool(len(days) == len(expected) and days.eq(expected).all())


def support_gate_summary(primary: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Any]:
    train = primary[_window(primary["entry_time"], "train")]
    selection = primary[_window(primary["entry_time"], "selection")]
    h1_2023 = primary[(primary["entry_time"] >= pd.Timestamp("2023-01-01T00:00:00Z")) & (primary["entry_time"] < pd.Timestamp("2023-07-01T00:00:00Z"))]
    h2_2023 = primary[(primary["entry_time"] >= pd.Timestamp("2023-07-01T00:00:00Z")) & (primary["entry_time"] < pd.Timestamp("2024-01-01T00:00:00Z"))]
    by_year = primary.groupby(primary["entry_time"].dt.year).size().to_dict() if len(primary) else {}
    checks = {
        "train_2021_2022_total_minimum": len(train) >= 60,
        "train_2021_minimum": int(by_year.get(2021, 0)) >= 24,
        "train_2022_minimum": int(by_year.get(2022, 0)) >= 24,
        "selection_2023_total_minimum": len(selection) >= 24,
        "selection_2023_h1_minimum": len(h1_2023) >= 10,
        "selection_2023_h2_minimum": len(h2_2023) >= 10,
        "train_side_share_25_75": _side_share_ok(train),
        "selection_side_share_25_75": _side_share_ok(selection),
        "train_max_entry_month_share": _max_month_share(train) <= 0.15,
        "selection_max_entry_month_share": _max_month_share(selection) <= 0.15,
        "all_usable_days_have_min72_blocks": bool(daily.loc[daily["valid_source_day"], "block_count"].ge(72).all()),
        "all_usable_days_no_missing_utc_day": _has_contiguous_utc_days(daily),
        "all_usable_days_edges_positive": bool(daily.loc[daily["valid_source_day"], "edges"].gt(0).all()),
        "all_usable_days_fees_positive": bool(daily.loc[daily["valid_source_day"], "fees"].gt(0).all()),
    }
    return {
        "passed": all(bool(value) for value in checks.values()),
        "checks": checks,
        "counts": {
            "primary_total": int(len(primary)),
            "train_2021_2022": int(len(train)),
            "selection_2023": int(len(selection)),
            "2021": int(by_year.get(2021, 0)),
            "2022": int(by_year.get(2022, 0)),
            "2023H1": int(len(h1_2023)),
            "2023H2": int(len(h2_2023)),
        },
        "side_counts": {
            "train": {str(k): int(v) for k, v in train["side"].value_counts().to_dict().items()},
            "selection": {str(k): int(v) for k, v in selection["side"].value_counts().to_dict().items()},
        },
        "max_entry_month_share": {"train": _max_month_share(train), "selection": _max_month_share(selection)},
    }


def _format_clock_for_csv(clock: pd.DataFrame) -> pd.DataFrame:
    out = clock.copy()
    for column in ["source_day", "available_time", "entry_time", "exit_time"]:
        out[column] = pd.to_datetime(out[column], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return out[CLOCK_COLUMNS]


def _frame_hash(frame: pd.DataFrame) -> str:
    return canonical_hash(_format_clock_for_csv(frame).to_dict(orient="records"))


def write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    _format_clock_for_csv(frame).to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, output)


def write_csv_gz(path: str | Path, frame: pd.DataFrame) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            _format_clock_for_csv(frame).to_csv(gz, index=False, lineterminator="\n")
    os.replace(temporary, output)


def build_support_artifacts(cfg: Config) -> dict[str, Any]:
    preregistration = validate_frozen_preregistration(cfg.preregistration)
    _validate_config(cfg, preregistration)
    preregistration_sha256 = sha256_file(cfg.preregistration)
    support_builder_sha256 = sha256_file(SUPPORT_BUILDER)
    source = load_source_frame(preregistration)
    daily = attach_strict_prior_ranks(build_daily_features(source))
    primary = build_primary_clock(daily)
    controls = build_control_clocks(daily, primary)
    gates = support_gate_summary(primary, daily)
    write_csv(cfg.primary_clock, primary)
    write_csv_gz(cfg.control_clocks, controls)
    source_binding = _source_output_binding(preregistration)
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "config": asdict(cfg),
        "support_builder": {"path": str(SUPPORT_BUILDER), "sha256": support_builder_sha256},
        "preregistration": {"path": str(cfg.preregistration), "sha256": preregistration_sha256, "manifest_hash": preregistration["manifest_hash"]},
        "source": {
            "path": source_binding["path"],
            "sha256": source_binding["sha256"],
            "rows_read": int(len(source)),
            "columns": SOURCE_COLUMNS,
        },
        "artifacts": {
            "primary_clock": {"path": str(cfg.primary_clock), "sha256": sha256_file(cfg.primary_clock), "frame_hash": _frame_hash(primary), "rows": int(len(primary)), "columns": CLOCK_COLUMNS},
            "control_clocks": {"path": str(cfg.control_clocks), "sha256": sha256_file(cfg.control_clocks), "frame_hash": _frame_hash(controls), "rows": int(len(controls)), "columns": CLOCK_COLUMNS, "clock_counts": {str(k): int(v) for k, v in controls["clock"].value_counts().sort_index().to_dict().items()}},
        },
        "support_gates": gates,
        "outcome_boundary": OUTCOME_BOUNDARY,
        "market_rows_loaded": 0,
        "funding_rows_loaded": 0,
        "return_rows_loaded": 0,
        "market_values_read": 0,
        "funding_values_read": 0,
        "return_values_read": 0,
        "profit_loss_fields": 0,
    }
    artifact = {**core, "manifest_hash": canonical_hash(core)}
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return artifact


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", default=Config.preregistration)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--primary-clock", default=Config.primary_clock)
    parser.add_argument("--control-clocks", default=Config.control_clocks)
    return Config(**vars(parser.parse_args()))


def main() -> None:
    print(json.dumps(build_support_artifacts(parse_args()), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
