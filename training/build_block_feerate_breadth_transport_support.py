"""Build outcome-blind BFRT-288 feature and support clocks.

This stage opens only the hash-bound mined-block fee-rate source.  It derives
the preregistered source features, primary incidence, and diagnostic controls;
it never loads BTC market data, funding, returns, or PnL.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd

from training import preregister_block_feerate_breadth_transport as prereg


POLICY_ID = "BFRT-288"
PROTOCOL_VERSION = "block_feerate_breadth_transport_support_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREGISTRATION = Path(
    "results/block_feerate_breadth_transport_preregistration_2026-07-20.json"
)
DEFAULT_OUTPUT = Path(
    "results/block_feerate_breadth_transport_support_2026-07-20.json"
)
DEFAULT_PRIMARY_CLOCK = Path(
    "results/block_feerate_breadth_transport_primary_clock_2026-07-20.csv.gz"
)
DEFAULT_CONTROL_CLOCKS = Path(
    "results/block_feerate_breadth_transport_control_clocks_2026-07-20.csv.gz"
)
SUPPORT_BUILDER = Path(
    "training/build_block_feerate_breadth_transport_support.py"
)
EXPECTED_PREREGISTRATION_FILE_SHA256 = (
    "73b06b94db2f844d993dcd76d6ad9e60f9b6d332d4c4d6ba8d41a3de970dae42"
)
EXPECTED_PREREGISTRATION_MANIFEST_HASH = (
    "2d708231cbecaa5621c756f6cd9b7fbd259feb8baf32268464d68838487b9ebc"
)
EXPECTED_POLICY_HASH = (
    "06d2284866781a3c751857a6a049769cece2e64ea56e00d8e2180b5992825925"
)
EXPECTED_PREREGISTRATION_SOURCE_SHA256 = (
    "2cc282d40e6791b0e4a5b5c1fb6f5081335eceb9a144db17faf157a8965deb5f"
)
SOURCE_COLUMNS = list(prereg.SOURCE_COLUMNS)
INNER_FEE_COLUMNS = ["fee_p10", "fee_p25", "fee_p50", "fee_p75", "fee_p90"]
INTEGER_COLUMNS = ["avg_height", "avg_timestamp", *SOURCE_COLUMNS[5:]]
FEATURE_COLUMNS = [
    "location",
    "signed_coherence",
    "coherence",
    "tail_divergence",
    "magnitude_rank",
    "tail_divergence_rank",
]
CLOCK_COLUMNS = [
    "policy_id",
    "clock",
    "window",
    "bucket_start_utc",
    "source_available_at_utc",
    "entry_time_utc",
    "exit_time_utc",
    "side",
    *FEATURE_COLUMNS,
]
CONTROL_NAMES = [
    "direction_flip",
    "magnitude_only",
    "tail_only",
    "constant_long_same_clock",
    "constant_short_same_clock",
    "stale_7d",
    "month_side_stratified_random_clock",
    "one_bar_delayed_entry",
]
OUTCOME_BOUNDARY = {
    "source_values_read": True,
    "source_feature_rows_derived": True,
    "signal_incidence_rows_derived": True,
    "market_rows_loaded": 0,
    "funding_rows_loaded": 0,
    "premium_or_oi_rows_loaded": 0,
    "return_rows_loaded": 0,
    "market_values_read": 0,
    "funding_values_read": 0,
    "return_or_pnl_fields": 0,
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
    "premium",
    "open_interest",
    "oi",
}
WINDOWS = {
    "train": (
        pd.Timestamp("2023-11-01T00:00:00Z"),
        pd.Timestamp("2025-01-01T00:00:00Z"),
    ),
    "test": (
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-01T00:00:00Z"),
    ),
    "eval": (
        pd.Timestamp("2026-01-01T00:00:00Z"),
        pd.Timestamp("2026-07-20T00:00:00Z"),
    ),
}


@dataclass(frozen=True)
class Config:
    preregistration: str = str(DEFAULT_PREREGISTRATION)
    output: str = str(DEFAULT_OUTPUT)
    primary_clock: str = str(DEFAULT_PRIMARY_CLOCK)
    control_clocks: str = str(DEFAULT_CONTROL_CLOCKS)
    artifact_root: str = "results"


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return candidate.resolve()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_preregistration(path: str | Path) -> dict[str, Any]:
    artifact_path = _repository_path(path)
    if artifact_path != _repository_path(DEFAULT_PREREGISTRATION):
        raise RuntimeError("BFRT preregistration path differs from frozen artifact")
    if sha256_file(artifact_path) != EXPECTED_PREREGISTRATION_FILE_SHA256:
        raise RuntimeError("BFRT preregistration file SHA drift")
    artifact = prereg.load_preregistration(artifact_path)
    if artifact.get("manifest_hash") != EXPECTED_PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("BFRT preregistration manifest hash drift")
    if artifact.get("policy_hash") != EXPECTED_POLICY_HASH:
        raise RuntimeError("BFRT policy hash drift")
    if canonical_hash(artifact.get("policy")) != EXPECTED_POLICY_HASH:
        raise RuntimeError("BFRT policy content drift")
    expected_source = {
        "path": str(prereg.PREREGISTRATION_SOURCE),
        "sha256": EXPECTED_PREREGISTRATION_SOURCE_SHA256,
    }
    if artifact.get("preregistration_source") != expected_source:
        raise RuntimeError("BFRT preregistration source binding drift")
    if (
        sha256_file(prereg.PREREGISTRATION_SOURCE)
        != EXPECTED_PREREGISTRATION_SOURCE_SHA256
    ):
        raise RuntimeError("BFRT preregistration source file drift")
    if artifact.get("policy") != prereg.policy():
        raise RuntimeError("BFRT policy singleton drift")
    source = artifact.get("source_manifest")
    if not isinstance(source, dict):
        raise RuntimeError("BFRT source binding missing")
    if source != prereg._validate_source_manifest(prereg.SOURCE_MANIFEST):
        raise RuntimeError("BFRT frozen source binding no longer revalidates")
    return artifact


def _source_binding(artifact: dict[str, Any]) -> dict[str, Any]:
    source = artifact["source_manifest"].get("normalized_artifact")
    if not isinstance(source, dict):
        raise RuntimeError("BFRT normalized source binding missing")
    return source


def _validate_config(cfg: Config, preregistration: dict[str, Any]) -> None:
    paths = {
        "output": _repository_path(cfg.output),
        "primary_clock": _repository_path(cfg.primary_clock),
        "control_clocks": _repository_path(cfg.control_clocks),
    }
    if not str(cfg.output).endswith(".json"):
        raise ValueError("BFRT support output must be JSON")
    for name in ("primary_clock", "control_clocks"):
        if not str(getattr(cfg, name)).endswith(".csv.gz"):
            raise ValueError(f"BFRT {name} must be .csv.gz")
    if len(set(paths.values())) != len(paths):
        raise ValueError("BFRT support output paths must be distinct")
    protected = {
        _repository_path(cfg.preregistration),
        _repository_path(prereg.SOURCE_MANIFEST),
        _repository_path(_source_binding(preregistration)["path"]),
        _repository_path(prereg.EXPECTED_RAW_ARTIFACT["path"]),
        _repository_path(prereg.SOURCE_DECISION),
        _repository_path(prereg.SOURCE_FREEZE),
        _repository_path(prereg.SOURCE_BUILDER),
        _repository_path(prereg.PREREGISTRATION_SOURCE),
        _repository_path(prereg.PREREGISTRATION_DOCUMENT),
        _repository_path(SUPPORT_BUILDER),
    }
    if set(paths.values()) & protected:
        raise ValueError("BFRT support output aliases a frozen input")
    artifact_root = _repository_path(cfg.artifact_root)
    outside = [
        name for name, path in paths.items() if not path.is_relative_to(artifact_root)
    ]
    if outside:
        raise ValueError(
            "BFRT support outputs must stay under the explicit artifact root; "
            f"outside={outside!r}"
        )
    existing = [name for name, path in paths.items() if path.exists()]
    if existing:
        raise FileExistsError(
            f"BFRT support artifacts are immutable; existing={existing!r}"
        )


def validate_source_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.columns.tolist() != SOURCE_COLUMNS:
        raise RuntimeError("BFRT source frame schema drift")
    if frame.empty:
        raise RuntimeError("BFRT source frame is empty")
    forbidden = FORBIDDEN_SOURCE_COLUMNS.intersection(
        {str(column).lower() for column in frame.columns}
    )
    if forbidden:
        raise RuntimeError(f"BFRT source contains outcome-like columns: {forbidden!r}")
    out = frame.copy()
    for column in INTEGER_COLUMNS:
        values = pd.to_numeric(out[column], errors="raise")
        if values.isna().any() or (values % 1).ne(0).any():
            raise RuntimeError(f"BFRT source {column} must contain exact integers")
        if (values < 0).any():
            raise RuntimeError(f"BFRT source {column} must be non-negative")
        out[column] = values.astype(np.int64)
    if (out["avg_height"] <= 0).any() or (out["avg_timestamp"] <= 0).any():
        raise RuntimeError("BFRT source height/timestamp must be positive")

    for column in ("bucket_start_utc", "bucket_end_utc", "available_at_utc"):
        parsed = pd.to_datetime(out[column], utc=True, errors="raise")
        if parsed.isna().any():
            raise RuntimeError(f"BFRT source {column} contains null timestamps")
        out[column] = parsed
    if out["bucket_start_utc"].duplicated().any():
        raise RuntimeError("BFRT source bucket starts must be unique")
    if not out["bucket_start_utc"].is_monotonic_increasing:
        raise RuntimeError("BFRT source bucket starts must be strictly increasing")
    out = out.reset_index(drop=True)
    if (
        not out["avg_height"].is_monotonic_increasing
        or out["avg_height"].duplicated().any()
    ):
        raise RuntimeError("BFRT source average heights must be strictly increasing")
    if (
        not out["avg_timestamp"].is_monotonic_increasing
        or out["avg_timestamp"].duplicated().any()
    ):
        raise RuntimeError("BFRT source average timestamps must be strictly increasing")
    expected_end = out["bucket_start_utc"] + pd.Timedelta(hours=12)
    expected_available = expected_end + pd.Timedelta(hours=48)
    if not out["bucket_end_utc"].equals(expected_end):
        raise RuntimeError("BFRT source bucket-end clock drift")
    if not out["available_at_utc"].equals(expected_available):
        raise RuntimeError("BFRT source availability clock drift")
    unix_start = (
        out["bucket_start_utc"].astype("int64") // 1_000_000_000
    ).astype(np.int64)
    unix_end = (out["bucket_end_utc"].astype("int64") // 1_000_000_000).astype(np.int64)
    timestamp_in_bucket = (out["avg_timestamp"] >= unix_start) & (
        out["avg_timestamp"] < unix_end
    )
    if not timestamp_in_bucket.all():
        raise RuntimeError("BFRT average timestamp escaped its derived bucket")
    deltas = out["bucket_start_utc"].diff().dropna()
    if not deltas.eq(pd.Timedelta(hours=12)).all():
        raise RuntimeError("BFRT source has a missing or irregular 12h bucket")
    fees = out[["fee_p0", *INNER_FEE_COLUMNS, "fee_p100"]].to_numpy(np.int64)
    if not np.all(fees[:, 1:] >= fees[:, :-1]):
        raise RuntimeError("BFRT source percentile ordering drift")
    return out


def load_source_frame(preregistration: dict[str, Any]) -> pd.DataFrame:
    binding = _source_binding(preregistration)
    path = _repository_path(binding["path"])
    if path.stat().st_size != binding["bytes"]:
        raise RuntimeError("BFRT normalized source byte-size drift")
    if sha256_file(path) != binding["sha256"]:
        raise RuntimeError("BFRT normalized source SHA drift")
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if header != SOURCE_COLUMNS:
        raise RuntimeError("BFRT normalized source header drift")
    return validate_source_frame(pd.read_csv(path, usecols=SOURCE_COLUMNS))


def strict_prior_midrank(current: float, prior: Iterable[float]) -> float:
    values = list(prior)
    if not values:
        raise ValueError("BFRT strict-prior midrank requires prior values")
    less = sum(value < current for value in values)
    equal = sum(value == current for value in values)
    return (less + 0.5 * equal) / len(values)


def build_features(
    source: pd.DataFrame,
    *,
    lookback: int = 180,
    minimum_prior: int = 120,
) -> pd.DataFrame:
    frame = validate_source_frame(source)
    if lookback < minimum_prior or minimum_prior <= 0:
        raise ValueError("BFRT rank lookback/minimum contract is invalid")
    records: list[dict[str, Any]] = []
    prior_magnitude: list[float] = []
    prior_tail: list[float] = []
    prior_available: list[pd.Timestamp] = []
    for index, row in frame.iterrows():
        record = row.to_dict()
        valid = False
        location = signed = coherence = tail = math.nan
        if index >= 2:
            starts = frame.loc[index - 2 : index, "bucket_start_utc"].tolist()
            contiguous = (
                starts[1] - starts[0] == pd.Timedelta(hours=12)
                and starts[2] - starts[1] == pd.Timedelta(hours=12)
            )
            if contiguous:
                current = [math.log1p(int(row[column])) for column in INNER_FEE_COLUMNS]
                lagged = [
                    math.log1p(int(frame.at[index - 2, column]))
                    for column in INNER_FEE_COLUMNS
                ]
                delta = [now - old for now, old in zip(current, lagged)]
                l1_motion = math.fsum(abs(value) for value in delta)
                location = float(np.median(np.asarray(delta, dtype=np.float64)))
                if l1_motion > 0.0 and location != 0.0:
                    signed = math.fsum(delta) / l1_motion
                    coherence = abs(signed)
                    tail = (
                        abs((delta[4] - delta[3]) - (delta[1] - delta[0]))
                        / l1_motion
                    )
                    valid = bool(
                        all(
                            math.isfinite(value)
                            for value in (location, signed, coherence, tail)
                        )
                        and signed != 0.0
                        and math.copysign(1.0, signed)
                        == math.copysign(1.0, location)
                    )
        magnitude_rank = tail_rank = math.nan
        available = pd.Timestamp(record["available_at_utc"])
        if valid and len(prior_magnitude) >= minimum_prior:
            if prior_available and not prior_available[-1] < available:
                raise RuntimeError("BFRT strict-prior availability order drift")
            magnitude_rank = strict_prior_midrank(
                abs(location), prior_magnitude[-lookback:]
            )
            tail_rank = strict_prior_midrank(tail, prior_tail[-lookback:])
        if valid:
            prior_magnitude.append(abs(location))
            prior_tail.append(tail)
            prior_available.append(available)
        entry = available.ceil("5min") + pd.Timedelta(minutes=5)
        record.update(
            {
                "entry_time_utc": entry,
                "exit_time_utc": entry + pd.Timedelta(hours=24),
                "location": location,
                "signed_coherence": signed,
                "coherence": coherence,
                "tail_divergence": tail,
                "magnitude_rank": magnitude_rank,
                "tail_divergence_rank": tail_rank,
                "feature_valid": valid,
                "rank_ready": bool(
                    valid
                    and math.isfinite(magnitude_rank)
                    and math.isfinite(tail_rank)
                ),
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _window_name(entry: pd.Timestamp, exit_time: pd.Timestamp) -> str | None:
    for name, (start, end) in WINDOWS.items():
        if start <= entry and exit_time <= end:
            return name
    return None


def _candidate_side(row: Any, mode: str) -> int:
    if not bool(row.rank_ready):
        return 0
    magnitude = float(row.magnitude_rank)
    coherence = float(row.coherence)
    tail = float(row.tail_divergence_rank)
    if mode == "primary":
        eligible = magnitude >= 0.75 and coherence >= 0.60 and tail <= 0.75
    elif mode == "magnitude_only":
        eligible = magnitude >= 0.75
    elif mode == "tail_only":
        eligible = magnitude >= 0.75 and coherence >= 0.60 and tail > 0.75
    elif mode == "rank_ready":
        eligible = True
    else:
        raise ValueError(f"unknown BFRT candidate mode {mode!r}")
    if not eligible:
        return 0
    return 1 if float(row.signed_coherence) > 0.0 else -1


def _clock_record(row: Any, *, clock: str, window: str, side: int) -> dict[str, Any]:
    return {
        "policy_id": POLICY_ID,
        "clock": clock,
        "window": window,
        "bucket_start_utc": row.bucket_start_utc,
        "source_available_at_utc": row.available_at_utc,
        "entry_time_utc": row.entry_time_utc,
        "exit_time_utc": row.exit_time_utc,
        "side": int(side),
        **{column: float(getattr(row, column)) for column in FEATURE_COLUMNS},
    }


def build_clock(features: pd.DataFrame, *, mode: str, clock: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    prior_exit: dict[str, pd.Timestamp] = {}
    ordered = features.sort_values(
        ["entry_time_utc", "bucket_start_utc"], kind="stable"
    )
    for row in ordered.itertuples(index=False):
        side = _candidate_side(row, mode)
        if side == 0:
            continue
        entry = pd.Timestamp(row.entry_time_utc)
        exit_time = pd.Timestamp(row.exit_time_utc)
        window = _window_name(entry, exit_time)
        if window is None:
            continue
        last_exit = prior_exit.get(window)
        if last_exit is not None and entry < last_exit:
            continue
        records.append(_clock_record(row, clock=clock, window=window, side=side))
        prior_exit[window] = exit_time
    return pd.DataFrame.from_records(records, columns=CLOCK_COLUMNS)


def _same_clock(primary: pd.DataFrame, name: str, sides: Iterable[int]) -> pd.DataFrame:
    control = primary.copy()
    control["clock"] = name
    control["side"] = list(sides)
    return control[CLOCK_COLUMNS]


def _stale_clock(features: pd.DataFrame) -> pd.DataFrame:
    stale = features.copy()
    shifted = features[["bucket_start_utc", *FEATURE_COLUMNS, "rank_ready"]].shift(14)
    exact_lag = (
        stale["bucket_start_utc"] - shifted["bucket_start_utc"]
    ).eq(pd.Timedelta(days=7))
    for column in [*FEATURE_COLUMNS, "rank_ready"]:
        stale[column] = shifted[column]
    stale["rank_ready"] = shifted["rank_ready"].eq(True) & exact_lag
    return build_clock(stale, mode="primary", clock="stale_7d")


def _random_clock(features: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    pool = features[features["rank_ready"]].copy()
    baseline = build_clock(pool, mode="rank_ready", clock="random_pool")
    rng = random.Random(20260720)
    records: list[pd.DataFrame] = []
    if primary.empty:
        return pd.DataFrame(columns=CLOCK_COLUMNS)
    primary_month = primary["entry_time_utc"].dt.strftime("%Y-%m")
    baseline_month = baseline["entry_time_utc"].dt.strftime("%Y-%m")
    for month in sorted(primary_month.unique()):
        target = primary.loc[primary_month == month]
        candidates = baseline.loc[baseline_month == month]
        if len(candidates) < len(target):
            raise RuntimeError("BFRT random control month pool is too small")
        chosen = rng.sample(list(candidates.index), len(target))
        sampled = baseline.loc[chosen].sort_values("entry_time_utc").copy()
        sides = target["side"].astype(int).tolist()
        rng.shuffle(sides)
        sampled["side"] = sides
        sampled["clock"] = "month_side_stratified_random_clock"
        records.append(sampled)
    return (
        pd.concat(records, ignore_index=True)[CLOCK_COLUMNS]
        .sort_values(["entry_time_utc", "bucket_start_utc"], kind="stable")
        .reset_index(drop=True)
    )


def build_control_clocks(features: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    controls = [
        _same_clock(primary, "direction_flip", -primary["side"].astype(int)),
        build_clock(features, mode="magnitude_only", clock="magnitude_only"),
        build_clock(features, mode="tail_only", clock="tail_only"),
        _same_clock(primary, "constant_long_same_clock", [1] * len(primary)),
        _same_clock(primary, "constant_short_same_clock", [-1] * len(primary)),
        _stale_clock(features),
        _random_clock(features, primary),
    ]
    delayed = primary.copy()
    delayed["clock"] = "one_bar_delayed_entry"
    delayed["entry_time_utc"] += pd.Timedelta(minutes=5)
    delayed["exit_time_utc"] += pd.Timedelta(minutes=5)
    for row in delayed.itertuples(index=False):
        if _window_name(row.entry_time_utc, row.exit_time_utc) != row.window:
            raise RuntimeError("BFRT delayed control crossed a split boundary")
    controls.append(delayed[CLOCK_COLUMNS])
    nonempty = [control for control in controls if not control.empty]
    combined = (
        pd.concat(nonempty, ignore_index=True)
        if nonempty
        else pd.DataFrame(columns=CLOCK_COLUMNS)
    )
    observed = set(combined["clock"].unique())
    if not observed.issubset(set(CONTROL_NAMES)):
        raise RuntimeError("BFRT control family drift")
    return combined.sort_values(
        ["clock", "entry_time_utc", "bucket_start_utc"], kind="stable"
    ).reset_index(drop=True)


def _subset(
    clock: pd.DataFrame, start: str, end: str
) -> pd.DataFrame:
    entry = clock["entry_time_utc"]
    return clock[
        (entry >= pd.Timestamp(start)) & (entry < pd.Timestamp(end))
    ]


def _max_month_share(clock: pd.DataFrame) -> float:
    if clock.empty:
        return 1.0
    return float(
        clock["entry_time_utc"].dt.strftime("%Y-%m").value_counts(normalize=True).max()
    )


def support_gate_summary(primary: pd.DataFrame, source: pd.DataFrame) -> dict[str, Any]:
    train = primary[primary["window"] == "train"]
    test = primary[primary["window"] == "test"]
    evaluation = primary[primary["window"] == "eval"]
    periods = {
        "train_2023_nov_dec": _subset(
            primary, "2023-11-01T00:00:00Z", "2024-01-01T00:00:00Z"
        ),
        "train_2024_h1": _subset(
            primary, "2024-01-01T00:00:00Z", "2024-07-01T00:00:00Z"
        ),
        "train_2024_h2": _subset(
            primary, "2024-07-01T00:00:00Z", "2025-01-01T00:00:00Z"
        ),
        "test_2025_h1": _subset(
            primary, "2025-01-01T00:00:00Z", "2025-07-01T00:00:00Z"
        ),
        "test_2025_h2": _subset(
            primary, "2025-07-01T00:00:00Z", "2026-01-01T00:00:00Z"
        ),
        "eval_2026_h1": _subset(
            primary, "2026-01-01T00:00:00Z", "2026-07-01T00:00:00Z"
        ),
    }
    quarters = {
        quarter: _subset(primary, start, end)
        for quarter, start, end in (
            ("2025Q1", "2025-01-01T00:00:00Z", "2025-04-01T00:00:00Z"),
            ("2025Q2", "2025-04-01T00:00:00Z", "2025-07-01T00:00:00Z"),
            ("2025Q3", "2025-07-01T00:00:00Z", "2025-10-01T00:00:00Z"),
            ("2025Q4", "2025-10-01T00:00:00Z", "2026-01-01T00:00:00Z"),
        )
    }
    side_counts = {
        name: {
            "long": int((clock["side"] == 1).sum()),
            "short": int((clock["side"] == -1).sum()),
        }
        for name, clock in (("train", train), ("test", test), ("eval", evaluation))
    }
    checks = {
        "train_total_minimum": len(train) >= 80,
        "train_long_minimum": side_counts["train"]["long"] >= 25,
        "train_short_minimum": side_counts["train"]["short"] >= 25,
        "train_2023_nov_dec_minimum": len(periods["train_2023_nov_dec"]) >= 8,
        "train_2024_h1_minimum": len(periods["train_2024_h1"]) >= 14,
        "train_2024_h2_minimum": len(periods["train_2024_h2"]) >= 14,
        "train_maximum_month_share": _max_month_share(train) <= 0.15,
        "test_total_minimum": len(test) >= 35,
        "test_long_minimum": side_counts["test"]["long"] >= 12,
        "test_short_minimum": side_counts["test"]["short"] >= 12,
        "test_each_half_minimum": all(
            len(periods[name]) >= 14 for name in ("test_2025_h1", "test_2025_h2")
        ),
        "test_each_quarter_minimum": all(
            len(clock) >= 6 for clock in quarters.values()
        ),
        "test_maximum_month_share": _max_month_share(test) <= 0.20,
        "eval_total_minimum": len(evaluation) >= 20,
        "eval_long_minimum": side_counts["eval"]["long"] >= 6,
        "eval_short_minimum": side_counts["eval"]["short"] >= 6,
        "eval_2026_h1_minimum": len(periods["eval_2026_h1"]) >= 18,
        "eval_maximum_month_share": _max_month_share(evaluation) <= 0.25,
        "missing_12h_source_buckets": bool(
            source["bucket_start_utc"].diff().dropna().eq(pd.Timedelta(hours=12)).all()
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "primary_total": int(len(primary)),
            "train": int(len(train)),
            "test": int(len(test)),
            "eval": int(len(evaluation)),
            **{name: int(len(clock)) for name, clock in periods.items()},
            **{name: int(len(clock)) for name, clock in quarters.items()},
        },
        "side_counts": side_counts,
        "maximum_month_share": {
            "train": _max_month_share(train),
            "test": _max_month_share(test),
            "eval": _max_month_share(evaluation),
        },
    }


def _format_clock(clock: pd.DataFrame) -> pd.DataFrame:
    out = clock.copy()
    for column in (
        "bucket_start_utc",
        "source_available_at_utc",
        "entry_time_utc",
        "exit_time_utc",
    ):
        out[column] = pd.to_datetime(out[column], utc=True).dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    return out[CLOCK_COLUMNS]


def _temporary_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    handle.close()
    return Path(handle.name)


def _write_clock(path: Path, clock: pd.DataFrame) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                _format_clock(clock).to_csv(
                    text,
                    index=False,
                    lineterminator="\n",
                    float_format="%.17g",
                )


def _frame_hash(clock: pd.DataFrame) -> str:
    return canonical_hash(_format_clock(clock).to_dict(orient="records"))


def _publish_new(temporary: Path, final: Path) -> None:
    os.link(temporary, final)
    temporary.unlink()


def build_support_artifacts(cfg: Config) -> dict[str, Any]:
    registration = validate_frozen_preregistration(cfg.preregistration)
    _validate_config(cfg, registration)
    source = validate_source_frame(load_source_frame(registration))
    features = build_features(source)
    primary = build_clock(features, mode="primary", clock="primary")
    controls = build_control_clocks(features, primary)
    gates = support_gate_summary(primary, source)

    output = _repository_path(cfg.output)
    primary_path = _repository_path(cfg.primary_clock)
    controls_path = _repository_path(cfg.control_clocks)
    output_tmp = _temporary_path(output)
    primary_tmp = _temporary_path(primary_path)
    controls_tmp = _temporary_path(controls_path)
    try:
        _write_clock(primary_tmp, primary)
        _write_clock(controls_tmp, controls)
        binding = _source_binding(registration)
        core = {
            "protocol_version": PROTOCOL_VERSION,
            "policy_id": POLICY_ID,
            "config": asdict(cfg),
            "support_builder": {
                "path": str(SUPPORT_BUILDER),
                "sha256": sha256_file(SUPPORT_BUILDER),
            },
            "preregistration": {
                "path": str(DEFAULT_PREREGISTRATION),
                "sha256": EXPECTED_PREREGISTRATION_FILE_SHA256,
                "manifest_hash": EXPECTED_PREREGISTRATION_MANIFEST_HASH,
                "policy_hash": EXPECTED_POLICY_HASH,
            },
            "source": {
                "path": binding["path"],
                "sha256": binding["sha256"],
                "bytes": binding["bytes"],
                "rows_read": int(len(source)),
                "columns": SOURCE_COLUMNS,
            },
            "feature_audit": {
                "feature_rows": int(len(features)),
                "valid_feature_rows": int(features["feature_valid"].sum()),
                "rank_ready_rows": int(features["rank_ready"].sum()),
                "lookback_valid_feature_buckets": 180,
                "minimum_prior_valid_feature_buckets": 120,
                "source_values_summarized": False,
            },
            "artifacts": {
                "primary_clock": {
                    "path": cfg.primary_clock,
                    "sha256": sha256_file(primary_tmp),
                    "frame_hash": _frame_hash(primary),
                    "rows": int(len(primary)),
                    "columns": CLOCK_COLUMNS,
                },
                "control_clocks": {
                    "path": cfg.control_clocks,
                    "sha256": sha256_file(controls_tmp),
                    "frame_hash": _frame_hash(controls),
                    "rows": int(len(controls)),
                    "columns": CLOCK_COLUMNS,
                    "clock_counts": {
                        name: int((controls["clock"] == name).sum())
                        for name in CONTROL_NAMES
                    },
                },
            },
            "support_gates": gates,
            "outcome_boundary": OUTCOME_BOUNDARY,
            "performance_values_opened": False,
            "stopping_rule": (
                "reject permanently without outcomes on any support failure"
            ),
        }
        artifact = {**core, "manifest_hash": canonical_hash(core)}
        output_tmp.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        published: list[Path] = []
        try:
            _publish_new(primary_tmp, primary_path)
            published.append(primary_path)
            _publish_new(controls_tmp, controls_path)
            published.append(controls_path)
            _publish_new(output_tmp, output)
            published.append(output)
        except BaseException:
            for path in reversed(published):
                path.unlink(missing_ok=True)
            raise
        return artifact
    finally:
        for path in (output_tmp, primary_tmp, controls_tmp):
            path.unlink(missing_ok=True)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", default=Config.preregistration)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--primary-clock", default=Config.primary_clock)
    parser.add_argument("--control-clocks", default=Config.control_clocks)
    parser.add_argument("--artifact-root", default=Config.artifact_root)
    return Config(**vars(parser.parse_args()))


def main() -> None:
    print(
        json.dumps(
            build_support_artifacts(parse_args()),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
