"""Build outcome-blind source support for frozen HVVBTR-6."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import (
    preregister_high_volatility_volume_bucket_toxicity_reversal as prereg,
)

ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path(
    "training/build_high_volatility_volume_bucket_toxicity_reversal_support.py"
)
PREREG_SHA = "bd558bc00d6e8ac84a4e57c49832e374e6b2e96fceb8995a47d07b738d3fbee9"
QUERY_START = pd.Timestamp("2023-04-01T00:00:00Z")
FIRST_BUCKET_START = pd.Timestamp("2023-04-02T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
REGISTRATION = prereg.build()
POLICY = REGISTRATION["policy"]
SPLITS = {
    name: (pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1]))
    for name, bounds in REGISTRATION["stages"].items()
}
SUPPORT_GATES = REGISTRATION["source_support_gates"]
MINIMUM_EVENTS = SUPPORT_GATES["minimum_events"]
CONTROLS = tuple(REGISTRATION["diagnostic_controls"]["names"])

QUERY = """SELECT ts,open,high,low,close,quote_asset_volume,taker_buy_quote
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts"""

SOURCE_DIR = Path(
    "data/high_volatility_volume_bucket_toxicity_reversal_sources_2023_2026"
)
PAIR_PANEL = SOURCE_DIR / "volume_bucket_pairs.csv.gz"
FEATURE_PANEL = SOURCE_DIR / "volume_bucket_features.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path(
    "data/high_volatility_volume_bucket_toxicity_reversal_clocks_2023_2026.csv.gz"
)
CONTROL_DIR = Path(
    "data/high_volatility_volume_bucket_toxicity_reversal_controls_2023_2026"
)
RESULT = Path(
    "results/high_volatility_volume_bucket_toxicity_reversal_support_2026-08-10.json"
)

SOURCE_COLUMNS = (
    "ts",
    "open",
    "high",
    "low",
    "close",
    "quote_asset_volume",
    "taker_buy_quote",
)
PAIR_COLUMNS = (
    "bucket_id",
    "bucket_start_time",
    "bucket_final_bar_time",
    "feature_available_time",
    "source_valid",
    "bucket_valid",
    "variation_valid",
    "target_prior_minute_count",
    "bucket_minute_count",
    "target_quote_volume",
    "bucket_quote_volume",
    "bucket_overshoot_quote_volume",
    "bucket_signed_flow",
    "bucket_imbalance",
    "btc_variation",
    "terminal_failure_reason",
)
FEATURE_COLUMNS = (
    *PAIR_COLUMNS,
    "toxicity",
    "toxicity_rank",
    "direction_consensus",
    "consensus_sign",
    "variation_rank",
    "eligible_state",
    "source_valid_onset",
)
CLOCK_COLUMNS = (
    "candidate",
    "control",
    "split",
    "bucket_id",
    "bucket_start_time",
    "bucket_final_bar_time",
    "feature_available_time",
    "decision_time",
    "entry_time",
    "exit_time",
    "side",
    "bucket_minute_count",
    "target_quote_volume",
    "bucket_quote_volume",
    "bucket_signed_flow",
    "bucket_imbalance",
    "toxicity",
    "toxicity_rank",
    "consensus_sign",
    "btc_variation",
    "variation_rank",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def strict_prior_midrank(
    values: pd.Series,
    lookback: int = POLICY["history_buckets"],
    minimum: int = POLICY["minimum_history_buckets"],
) -> pd.Series:
    """Rank finite values against at most ``lookback`` finite strict priors."""
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    result = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if math.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior, dtype=float)
            result.at[index] = float(
                (
                    np.count_nonzero(array < current)
                    + 0.5 * np.count_nonzero(array == current)
                )
                / len(array)
            )
        if math.isfinite(current):
            history.append(float(current))
    return result


def postgres_engine():
    from sqlalchemy import create_engine

    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(
        postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10}
    )


def load_source() -> pd.DataFrame:
    """Query only the preregistered completed one-minute source fields."""
    from sqlalchemy import text

    engine = postgres_engine()
    try:
        with engine.connect() as connection:
            return pd.read_sql_query(
                text(QUERY),
                connection,
                params={
                    "start": QUERY_START.to_pydatetime(),
                    "end": END.to_pydatetime(),
                },
            )
    finally:
        engine.dispose()


def prepare_source(bars: pd.DataFrame) -> pd.DataFrame:
    """Validate schema and mark every minute against the frozen source contract."""
    if bars.columns.tolist() != list(SOURCE_COLUMNS):
        raise RuntimeError("HVVBTR source schema drift")
    frame = bars.copy()
    frame["ts"] = pd.to_datetime(frame.ts, utc=True, errors="coerce")
    for column in SOURCE_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.ts.isna().any() or frame.ts.duplicated().any():
        raise RuntimeError("HVVBTR invalid or duplicate source timestamps")

    prices = frame[["open", "high", "low", "close"]]
    quote = frame.quote_asset_volume
    taker = frame.taker_buy_quote
    frame["row_valid"] = (
        np.isfinite(prices).all(axis=1)
        & prices.gt(0).all(axis=1)
        & frame.high.ge(prices[["open", "close"]].max(axis=1))
        & frame.low.le(prices[["open", "close"]].min(axis=1))
        & frame.high.ge(frame.low)
        & np.isfinite(quote)
        & quote.ge(0)
        & np.isfinite(taker)
        & taker.ge(0)
        & taker.le(quote)
    )
    return frame.sort_values("ts", kind="mergesort").set_index("ts")


def _window(source: pd.DataFrame, start: pd.Timestamp, periods: int) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="1min")
    return source.reindex(index)


def _terminal_pair(
    bucket_id: int,
    start: pd.Timestamp,
    *,
    target_count: int,
    bucket_count: int,
    target: float = math.nan,
    quote: float = math.nan,
    reason: str,
) -> dict[str, Any]:
    return {
        "bucket_id": bucket_id,
        "bucket_start_time": start,
        "bucket_final_bar_time": pd.NaT,
        "feature_available_time": pd.NaT,
        "source_valid": False,
        "bucket_valid": False,
        "variation_valid": False,
        "target_prior_minute_count": target_count,
        "bucket_minute_count": bucket_count,
        "target_quote_volume": target,
        "bucket_quote_volume": quote,
        "bucket_overshoot_quote_volume": math.nan,
        "bucket_signed_flow": math.nan,
        "bucket_imbalance": math.nan,
        "btc_variation": math.nan,
        "terminal_failure_reason": reason,
    }


def build_pair_panel(
    bars: pd.DataFrame,
    first_bucket_start: pd.Timestamp = FIRST_BUCKET_START,
    end: pd.Timestamp = END,
) -> pd.DataFrame:
    """Construct exact, whole-minute, causally target-frozen volume buckets."""
    source = prepare_source(bars)
    start = pd.Timestamp(first_bucket_start)
    end = pd.Timestamp(end)
    rows: list[dict[str, Any]] = []
    bucket_id = 0
    prior_minutes = int(POLICY["target_prior_minutes"])
    maximum_minutes = int(POLICY["maximum_bucket_minutes"])

    while start < end:
        prior = _window(
            source, start - pd.Timedelta(minutes=prior_minutes), prior_minutes
        )
        prior_valid = prior.row_valid.eq(True)
        prior_count = int(prior_valid.sum())
        if not bool(prior_valid.all()):
            rows.append(
                _terminal_pair(
                    bucket_id,
                    start,
                    target_count=prior_count,
                    bucket_count=0,
                    reason="invalid_exact_target_window",
                )
            )
            break
        target = float(prior.quote_asset_volume.sum()) / float(
            POLICY["target_buckets_per_day"]
        )
        if not math.isfinite(target) or target <= 0:
            rows.append(
                _terminal_pair(
                    bucket_id,
                    start,
                    target_count=prior_count,
                    bucket_count=0,
                    target=target,
                    reason="nonpositive_target",
                )
            )
            break

        available_minutes = min(
            maximum_minutes, max(0, int((end - start) / pd.Timedelta(minutes=1)))
        )
        candidate = _window(source, start, available_minutes)
        cumulative_quote = 0.0
        final_position: int | None = None
        invalid_position: int | None = None
        for position, (_, minute) in enumerate(candidate.iterrows()):
            row_valid = minute.get("row_valid", False)
            if pd.isna(row_valid) or not bool(row_valid):
                invalid_position = position
                break
            cumulative_quote += float(minute.quote_asset_volume)
            if cumulative_quote >= target:
                final_position = position
                break

        if invalid_position is not None:
            rows.append(
                _terminal_pair(
                    bucket_id,
                    start,
                    target_count=prior_count,
                    bucket_count=invalid_position,
                    target=target,
                    quote=cumulative_quote,
                    reason="invalid_or_missing_constituent_minute",
                )
            )
            break
        if final_position is None:
            # A partial bucket censored by the preregistered source end is not a
            # duration failure. Exactly 360 available minutes without completion is.
            if available_minutes < maximum_minutes:
                break
            rows.append(
                _terminal_pair(
                    bucket_id,
                    start,
                    target_count=prior_count,
                    bucket_count=maximum_minutes,
                    target=target,
                    quote=cumulative_quote,
                    reason="maximum_duration_target_not_reached",
                )
            )
            break

        bucket = candidate.iloc[: final_position + 1]
        final_bar = start + pd.Timedelta(minutes=final_position)
        available = final_bar + pd.Timedelta(minutes=1)
        bucket_quote = float(bucket.quote_asset_volume.sum())
        signed_flow = float(
            (2.0 * bucket.taker_buy_quote - bucket.quote_asset_volume).sum()
        )
        imbalance = signed_flow / bucket_quote
        variation_window = _window(
            source,
            available - pd.Timedelta(minutes=prior_minutes),
            prior_minutes,
        )
        variation_rows = variation_window.row_valid.eq(True)
        variation = math.nan
        variation_valid = bool(variation_rows.all())
        if variation_valid:
            minute_returns = np.log(
                variation_window.close.to_numpy(float)
                / variation_window.open.to_numpy(float)
            )
            variation = float(np.square(minute_returns).sum())
            variation_valid = math.isfinite(variation) and variation > 0
        bucket_valid = (
            math.isfinite(bucket_quote)
            and bucket_quote > 0
            and math.isfinite(signed_flow)
            and math.isfinite(imbalance)
            and imbalance != 0
        )
        source_valid = bool(bucket_valid and variation_valid)
        reason = (
            ""
            if source_valid
            else (
                "zero_or_invalid_bucket_imbalance"
                if not bucket_valid
                else "invalid_exact_variation_window"
            )
        )
        rows.append(
            {
                "bucket_id": bucket_id,
                "bucket_start_time": start,
                "bucket_final_bar_time": final_bar,
                "feature_available_time": available,
                "source_valid": source_valid,
                "bucket_valid": bool(bucket_valid),
                "variation_valid": bool(variation_valid),
                "target_prior_minute_count": prior_count,
                "bucket_minute_count": final_position + 1,
                "target_quote_volume": target,
                "bucket_quote_volume": bucket_quote,
                "bucket_overshoot_quote_volume": bucket_quote - target,
                "bucket_signed_flow": signed_flow,
                "bucket_imbalance": imbalance,
                "btc_variation": variation,
                "terminal_failure_reason": reason,
            }
        )
        if not source_valid:
            break
        start = available
        bucket_id += 1

    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def build_features(pair: pd.DataFrame) -> pd.DataFrame:
    """Derive frozen toxicity, consensus, causal ranks, and primary onset."""
    if pair.columns.tolist() != list(PAIR_COLUMNS):
        raise RuntimeError("HVVBTR pair-panel schema drift")
    features = (
        pair.sort_values("bucket_id", kind="mergesort").reset_index(drop=True).copy()
    )
    ids = pd.to_numeric(features.bucket_id, errors="coerce")
    if ids.isna().any() or ids.duplicated().any() or not ids.is_monotonic_increasing:
        raise RuntimeError("HVVBTR bucket identity drift")
    source_valid = features.source_valid.eq(True)
    imbalance = pd.to_numeric(features.bucket_imbalance, errors="coerce")
    valid_imbalance = imbalance.where(
        source_valid & np.isfinite(imbalance) & imbalance.ne(0)
    )
    toxicity_count = int(POLICY["toxicity_buckets"])
    direction_count = int(POLICY["direction_buckets"])
    features["toxicity"] = (
        valid_imbalance.abs().rolling(toxicity_count, min_periods=toxicity_count).mean()
    )
    signs = np.sign(valid_imbalance)
    sign_sum = signs.rolling(direction_count, min_periods=direction_count).sum()
    features["direction_consensus"] = sign_sum.abs().eq(direction_count)
    features["consensus_sign"] = (
        np.sign(sign_sum).where(features.direction_consensus, 0).fillna(0).astype(int)
    )
    features["toxicity_rank"] = strict_prior_midrank(
        features.toxicity.where(source_valid)
    )
    features["variation_rank"] = strict_prior_midrank(
        pd.to_numeric(features.btc_variation, errors="coerce").where(source_valid)
    )
    eligible = (
        source_valid
        & features.toxicity_rank.ge(POLICY["toxicity_rank_min"])
        & features.variation_rank.ge(POLICY["variation_rank_min"])
        & features.direction_consensus
    )
    adjacent = ids.shift(1).add(1).eq(ids)
    onset = (
        eligible
        & adjacent
        & source_valid.shift(1, fill_value=False)
        & ~eligible.shift(1, fill_value=False)
    )
    features["eligible_state"] = eligible
    features["source_valid_onset"] = onset
    return features.loc[:, FEATURE_COLUMNS]


def active_and_side(
    features: pd.DataFrame, control: str = "primary"
) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """Return control-specific eligibility, source-valid onset, and frozen side."""
    if control not in ("primary", *CONTROLS):
        raise ValueError(f"unknown HVVBTR control: {control}")
    ordered = features.sort_values("bucket_id", kind="mergesort").reset_index(drop=True)
    used = (
        ordered.shift(1) if control == "one_bucket_stale_features" else ordered.copy()
    )
    source_valid = used.source_valid.eq(True)
    toxicity = pd.to_numeric(used.toxicity, errors="coerce")
    toxicity_rank = pd.to_numeric(used.toxicity_rank, errors="coerce")
    variation_rank = pd.to_numeric(used.variation_rank, errors="coerce")
    consensus = used.direction_consensus.eq(True)
    consensus_sign = (
        pd.to_numeric(used.consensus_sign, errors="coerce").fillna(0).astype(int)
    )
    if control == "single_last_bucket_direction":
        last = pd.to_numeric(used.bucket_imbalance, errors="coerce")
        consensus_sign = (
            pd.Series(np.sign(last), index=used.index).fillna(0).astype(int)
        )
        consensus = consensus_sign.ne(0)
    toxicity_gate = (
        pd.Series(True, index=used.index)
        if control == "no_toxicity_tail"
        else toxicity_rank.ge(POLICY["toxicity_rank_min"])
    )
    variation_gate = (
        pd.Series(True, index=used.index)
        if control == "no_variation_gate"
        else variation_rank.ge(POLICY["variation_rank_min"])
    )
    eligible = (
        source_valid
        & np.isfinite(toxicity)
        & consensus
        & toxicity_gate
        & variation_gate
    )
    ids = pd.to_numeric(ordered.bucket_id, errors="coerce")
    adjacent = ids.shift(1).add(1).eq(ids)
    onset = (
        eligible
        & adjacent
        & source_valid.shift(1, fill_value=False)
        & ~eligible.shift(1, fill_value=False)
    )
    side = -consensus_sign
    if control == "direction_flip":
        side = -side
    elif control == "forced_long":
        side = side.where(side.eq(0), 1)
    return eligible, onset, side, used


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    """Build split-safe clocks with exact availability and global half-open reservation."""
    ordered = features.sort_values("bucket_id", kind="mergesort").reset_index(drop=True)
    _, onset, sides, used = active_and_side(ordered, control)
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for index in ordered.index[onset]:
        available = pd.Timestamp(ordered.at[index, "feature_available_time"])
        if pd.isna(available):
            continue
        decision = available.ceil("5min")
        entry = decision + pd.Timedelta(minutes=POLICY["entry_delay_minutes"])
        exit_time = entry + pd.Timedelta(hours=POLICY["hold_hours"])
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next(
            (
                name
                for name, (start, end) in SPLITS.items()
                if entry >= start and exit_time <= end
            ),
            None,
        )
        if split is None:
            continue
        reserved_until = exit_time
        rows.append(
            {
                "candidate": "HVVBTR-6",
                "control": control,
                "split": split,
                "bucket_id": int(ordered.at[index, "bucket_id"]),
                "bucket_start_time": ordered.at[index, "bucket_start_time"],
                "bucket_final_bar_time": ordered.at[index, "bucket_final_bar_time"],
                "feature_available_time": available,
                "decision_time": decision,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": int(sides.at[index]),
                "bucket_minute_count": int(used.at[index, "bucket_minute_count"]),
                "target_quote_volume": float(used.at[index, "target_quote_volume"]),
                "bucket_quote_volume": float(used.at[index, "bucket_quote_volume"]),
                "bucket_signed_flow": float(used.at[index, "bucket_signed_flow"]),
                "bucket_imbalance": float(used.at[index, "bucket_imbalance"]),
                "toxicity": float(used.at[index, "toxicity"]),
                "toxicity_rank": float(used.at[index, "toxicity_rank"]),
                "consensus_sign": int(used.at[index, "consensus_sign"]),
                "btc_variation": float(used.at[index, "btc_variation"]),
                "variation_rank": float(used.at[index, "variation_rank"]),
            }
        )
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def support_stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = clock[clock.split.eq(split)]
    if selected.empty:
        return {
            "events": 0,
            "longs": 0,
            "shorts": 0,
            "minority_side_share": 0.0,
            "max_month_share": 0.0,
        }
    longs = int(selected.side.eq(1).sum())
    shorts = int(selected.side.eq(-1).sum())
    months = (
        pd.to_datetime(selected.entry_time, utc=True)
        .dt.strftime("%Y-%m")
        .value_counts()
    )
    return {
        "events": len(selected),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": int(months.max()) / len(selected),
    }


def deterministic_csv_gzip(frame: pd.DataFrame) -> bytes:
    text = frame.to_csv(index=False, float_format="%.12g", lineterminator="\n").encode()
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=6, mtime=0) as output:
        output.write(text)
    return buffer.getvalue()


def deterministic_json(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()


def write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(
                f"refusing to overwrite immutable HVVBTR artifact: {path}"
            )
        return
    path.write_bytes(content)


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVVBTR preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    if registration != REGISTRATION:
        raise RuntimeError("HVVBTR committed preregistration payload drift")
    if tuple(registration["diagnostic_controls"]["names"]) != CONTROLS:
        raise RuntimeError("HVVBTR diagnostic-control drift")

    bars = load_source()
    pair = build_pair_panel(bars)
    features = build_features(pair)
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}

    write_immutable(PAIR_PANEL, deterministic_csv_gzip(pair))
    write_immutable(FEATURE_PANEL, deterministic_csv_gzip(features))
    write_immutable(CLOCK, deterministic_csv_gzip(primary))
    for name, frame in controls.items():
        write_immutable(CONTROL_DIR / f"{name}.csv.gz", deterministic_csv_gzip(frame))

    terminal_failures = int(pair.terminal_failure_reason.astype(str).ne("").sum())
    source_core = {
        "protocol_version": "hvvbtr_6_btc_volume_bucket_source_v1",
        "query": QUERY,
        "query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(),
        "table": "bars_binance",
        "symbol": "BTCUSDT",
        "interval": "1m",
        "columns": list(SOURCE_COLUMNS),
        "window": [QUERY_START.isoformat(), END.isoformat()],
        "physical_rows": len(bars),
        "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
        "pair_panel": {
            "path": str(PAIR_PANEL),
            "sha256": sha(PAIR_PANEL),
            "rows": len(pair),
        },
        "feature_panel": {
            "path": str(FEATURE_PANEL),
            "sha256": sha(FEATURE_PANEL),
            "rows": len(features),
            "valid_rows": int(features.source_valid.sum()),
        },
        "bucket_construction": {
            "first_start": FIRST_BUCKET_START.isoformat(),
            "target_prior_minutes": POLICY["target_prior_minutes"],
            "target_buckets_per_day": POLICY["target_buckets_per_day"],
            "maximum_bucket_minutes": POLICY["maximum_bucket_minutes"],
            "whole_minutes_only": True,
            "target_frozen_at_bucket_start": True,
            "terminal_failure_rows": terminal_failures,
        },
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "execution_prices_opened": False,
        "funding_values_opened": False,
        "gross9_rows_opened": False,
        "no_imputation": True,
        "deterministic_immutable_artifacts": True,
    }
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    write_immutable(SOURCE_MANIFEST, deterministic_json(source_manifest))

    support = {name: support_stats(primary, name) for name in SPLITS}
    checks = {
        check: bool(passed)
        for name, item in support.items()
        for check, passed in (
            (f"{name}_minimum_events", item["events"] >= MINIMUM_EVENTS[name]),
            (
                f"{name}_side_balance",
                item["minority_side_share"] >= SUPPORT_GATES["minority_side_share_min"],
            ),
            (
                f"{name}_month_concentration",
                item["max_month_share"] <= SUPPORT_GATES["max_month_share"],
            ),
        )
    }
    passed = bool(all(checks.values()))
    core = {
        "protocol_version": "hvvbtr_6_source_support_v1",
        "policy_id": "HVVBTR-6",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": PREREG_SHA,
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifest": {
            "path": str(SOURCE_MANIFEST),
            "sha256": sha(SOURCE_MANIFEST),
            "manifest_hash": source_manifest["manifest_hash"],
        },
        "ranking": {
            "lookback_valid_buckets": POLICY["history_buckets"],
            "minimum_prior_valid_buckets": POLICY["minimum_history_buckets"],
            "current_excluded": True,
            "ties": "midrank",
        },
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "execution_prices_opened": False,
        "funding_values_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "reservation": {
            "scope": "global",
            "hours": POLICY["hold_hours"],
            "interval": "half_open",
            "equal_open_after_exit_allowed": True,
            "split_crossing_action": "skip",
        },
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(frame),
                "promotion_authorized": False,
            }
            for name, frame in controls.items()
        },
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
        "deterministic_immutable_artifacts": True,
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    write_immutable(RESULT, deterministic_json(result))
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    report = run()
    print(
        json.dumps(
            {"passed": report["support_passed"], "support": report["support"]}, indent=2
        )
    )
