"""Build source-only CLTCR-6 confirmation-ladder clocks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_confirmation_ladder_tempo_compression_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.download_fetls_block_summaries import OUTPUT as BLOCKS


ENV_FILE = "/home/pakchu/rllm/.env"
PREREG_SHA = "265d183d080238f66081d8a957dd38eee6184deea6f5a3fb49d00175f37faa4a"
QUERY_START = pd.Timestamp("2023-05-01T00:00Z")
END = pd.Timestamp("2026-08-01T00:00Z")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00Z"), pd.Timestamp("2024-01-01T00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00Z"), pd.Timestamp("2025-01-01T00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00Z"), pd.Timestamp("2026-01-01T00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("late_unanimity_only", "tempo_compression_only", "one_anchor_stale_ladder", "direction_flip")
QUERY = """SELECT ts,open,high,low,close,count(*) OVER (PARTITION BY ts) AS duplicate_count FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"""
SOURCE_DIR = Path("data/confirmation_ladder_tempo_compression_relay_sources_2023_2026")
FEATURES = SOURCE_DIR / "confirmation_ladders.csv.gz"
MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/confirmation_ladder_tempo_compression_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/confirmation_ladder_tempo_compression_relay_controls_2023_2026")
RESULT = Path("results/confirmation_ladder_tempo_compression_relay_support_2026-08-09.json")
RETURN_COLUMNS = tuple(f"interval_return_{number}" for number in range(1, 7))
DURATION_COLUMNS = tuple(f"interval_duration_{number}" for number in range(1, 7))
FEATURE_COLUMNS = (
    "anchor_height", "confirmation_height", "feature_available_time", "source_valid",
    *RETURN_COLUMNS, *DURATION_COLUMNS, "late_return", "late_unanimous",
    "tempo_compression", "eligible_state",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "anchor_height", "confirmation_height",
    "feature_available_time", "entry_time", "exit_time", "side", "late_return",
    "late_unanimous", "tempo_compression",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def ceil_5m(seconds: int) -> pd.Timestamp:
    return pd.Timestamp(((seconds + 299) // 300) * 300, unit="s", tz="UTC")


def engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env
    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_minutes() -> pd.DataFrame:
    from sqlalchemy import text
    db = engine()
    try:
        with db.connect() as connection:
            return pd.read_sql_query(text(QUERY), connection, params={"start": QUERY_START.to_pydatetime(), "end": END.to_pydatetime()})
    finally:
        db.dispose()


def validate_blocks(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"height", "id", "previousblockhash", "timestamp", "mediantime", "tx_count", "size", "weight"}
    if set(frame) != required:
        raise ValueError("CLTCR block schema drift")
    blocks = frame.sort_values("height", kind="mergesort").reset_index(drop=True)
    for column in ("height", "timestamp", "mediantime"):
        blocks[column] = pd.to_numeric(blocks[column], errors="raise").astype("int64")
    if not np.array_equal(np.diff(blocks.height), np.ones(len(blocks) - 1, dtype=np.int64)):
        raise RuntimeError("CLTCR block heights are not contiguous")
    if not blocks.previousblockhash.iloc[1:].reset_index(drop=True).equals(blocks.id.iloc[:-1].reset_index(drop=True)):
        raise RuntimeError("CLTCR canonical previous-hash continuity failed")
    return blocks


def prepare_minutes(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"ts", "open", "high", "low", "close", "duplicate_count"}
    if set(frame) != required:
        raise ValueError("CLTCR minute schema drift")
    result = frame.copy(); result["ts"] = pd.to_datetime(result.ts, utc=True, errors="coerce")
    numeric = ["open", "high", "low", "close", "duplicate_count"]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    valid = (
        result.ts.notna() & np.isfinite(result[numeric]).all(axis=1)
        & result[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & result.high.ge(result[["open", "close"]].max(axis=1))
        & result.low.le(result[["open", "close"]].min(axis=1))
        & result.high.ge(result.low) & result.duplicate_count.eq(1)
    )
    result["source_valid"] = valid
    if result.ts.duplicated().any():
        raise RuntimeError("CLTCR duplicate timestamps")
    return result.sort_values("ts", kind="mergesort").set_index("ts")


def interval_return(market: pd.DataFrame, left_seconds: int, right_seconds: int):
    duration = right_seconds - left_seconds
    if not 60 <= duration <= 1800:
        return None
    first = ((left_seconds + 59) // 60) * 60
    end = (right_seconds // 60) * 60
    if first >= end:
        return None
    expected = pd.date_range(pd.Timestamp(first, unit="s", tz="UTC"), pd.Timestamp(end, unit="s", tz="UTC"), freq="1min", inclusive="left")
    window = market.reindex(expected)
    if window.source_valid.isna().any() or not bool(window.source_valid.all()):
        return None
    value = float(np.log(window.close.iloc[-1] / window.open.iloc[0]))
    if not np.isfinite(value) or value == 0:
        return None
    return value, len(expected)


def build_features(blocks: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    blocks = validate_blocks(blocks); market = prepare_minutes(market)
    timestamp = blocks.timestamp.to_numpy(np.int64); mediantime = blocks.mediantime.to_numpy(np.int64)
    prefix_time = np.maximum.accumulate(np.maximum(timestamp, mediantime)); rows = []
    for index in range(0, len(blocks) - 6):
        height = int(blocks.height.iloc[index])
        if height % 36:
            continue
        returns: list[float] = []; durations: list[int] = []; valid = True
        for offset in range(1, 7):
            left, right = int(timestamp[index + offset - 1]), int(timestamp[index + offset])
            values = interval_return(market, left, right)
            if values is None:
                valid = False; break
            value, _ = values; returns.append(value); durations.append(right - left)
        late = float(np.sum(returns[3:])) if valid else np.nan
        valid = bool(valid and np.isfinite(late) and late != 0)
        late_signs = np.sign(returns[3:]) if valid else np.array([])
        late_unanimous = bool(
            valid and len(late_signs) == 3
            and np.all(late_signs == late_signs[0]) and late_signs[0] != 0
        )
        tempo_compression = bool(valid and durations[3] > durations[4] > durations[5])
        rows.append({
            "anchor_height": height, "confirmation_height": int(blocks.height.iloc[index + 6]),
            "feature_available_time": ceil_5m(int(prefix_time[index + 6]) + 7200), "source_valid": valid,
            **{column: returns[position] if valid else np.nan for position, column in enumerate(RETURN_COLUMNS)},
            **{column: durations[position] if valid else np.nan for position, column in enumerate(DURATION_COLUMNS)},
            "late_return": late if valid else np.nan, "late_unanimous": late_unanimous,
            "tempo_compression": tempo_compression,
        })
    output = pd.DataFrame(rows)
    output["eligible_state"] = output.source_valid & output.late_unanimous & output.tempo_compression
    return output[list(FEATURE_COLUMNS)]

def active_and_side(features: pd.DataFrame, control: str = "primary"):
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    primary = features.eligible_state.astype(bool); late = features.late_return.copy()
    if control == "late_unanimity_only":
        state = features.source_valid & features.late_unanimous
    elif control == "tempo_compression_only":
        state = features.source_valid & features.tempo_compression
    elif control == "one_anchor_stale_ladder":
        state = primary.shift(1, fill_value=False); late = late.shift(1)
    else:
        state = primary
    active = pd.Series(False, index=features.index); previous_valid_state = False
    for index in features.index:
        if not bool(features.at[index, "source_valid"]):
            continue
        current = bool(state.at[index]); active.at[index] = current and not previous_valid_state; previous_valid_state = current
    side = np.sign(late)
    if control == "direction_flip":
        side = -side
    return active & late.ne(0), pd.Series(side, index=features.index).astype("Int64").fillna(0).astype(int)

def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, sides = active_and_side(features, control); rows = []; reserved = None
    for index in features.index[active & sides.ne(0)]:
        available = pd.Timestamp(features.at[index, "feature_available_time"]); entry = available + pd.Timedelta(minutes=5); exit_time = entry + pd.Timedelta(hours=6)
        if reserved is not None and entry < reserved:
            continue
        split = next((name for name, (left, right) in SPLITS.items() if entry >= left and exit_time <= right), None)
        if split is None:
            continue
        reserved = exit_time
        rows.append({
            "candidate": prereg.POLICY_ID, "control": control, "split": split,
            "anchor_height": int(features.at[index, "anchor_height"]), "confirmation_height": int(features.at[index, "confirmation_height"]),
            "feature_available_time": available, "entry_time": entry, "exit_time": exit_time, "side": int(sides.at[index]),
            "late_return": float(features.at[index, "late_return"]),
            "late_unanimous": bool(features.at[index, "late_unanimous"]), "tempo_compression": bool(features.at[index, "tempo_compression"]),
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    frame = clock[clock.split.eq(split)]
    if frame.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(frame.side.eq(1).sum()); shorts = int(frame.side.eq(-1).sum())
    months = pd.to_datetime(frame.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(frame), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(frame), "max_month_share": int(months.max()) / len(frame)}


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("CLTCR preregistration drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text()); prereg.validate(registration)
    blocks = pd.read_csv(BLOCKS, compression="gzip"); market = load_minutes()
    features = build_features(blocks, market); primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(features, FEATURES); _write_gzip_csv(primary, CLOCK)
    for name, clock in controls.items():
        _write_gzip_csv(clock, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {
        "protocol_version": "cltcr_6_source_v1", "query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(), "window": [QUERY_START.isoformat(), END.isoformat()],
        "physical_rows": {"blocks": len(blocks), "market": len(market)},
        "features": {"path": str(FEATURES), "sha256": sha(FEATURES), "rows": len(features)},
        "completed_preentry_sources_opened": True, "execution_prices_opened": False,
        "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "rv20_opened": False,
    }
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    MANIFEST.write_text(json.dumps(source_manifest, indent=2, allow_nan=False) + "\n")
    support = {name: stats(primary, name) for name in SPLITS}; checks = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.2
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "cltcr_6_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(MANIFEST), "sha256": sha(MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "execution_prices_opened": False,
        "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "rv20_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(clock), "promotion_authorized": False} for name, clock in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
