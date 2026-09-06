"""Build source-only TSPI-12 weekly clocks from completed one-minute bars."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_temporal_sign_persistence_imbalance as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
PREREG_SHA = "2ed7738d0ed934d7e01a8d0bfd9b1f0c81b0bf75bcf43d3a8b49a7c294aaf633"
START = pd.Timestamp("2023-06-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("body_count", "body_magnitude", "maximum_run", "direction_flip")
QUERY = """
SELECT ts,open,high,low,close
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
  AND EXTRACT(ISODOW FROM ts)=2
ORDER BY ts
"""
SOURCE_DIR = Path("data/temporal_sign_persistence_imbalance_sources_2023_2026")
FEATURES = SOURCE_DIR / "weekly_run_geometry.csv.gz"
MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/temporal_sign_persistence_imbalance_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/temporal_sign_persistence_imbalance_controls_2023_2026")
RESULT = Path("results/temporal_sign_persistence_imbalance_support_2026-08-09.json")
FEATURE_COLUMNS = (
    "source_day", "decision_time", "feature_available_time", "source_valid",
    "positive_bodies", "negative_bodies", "positive_run_mass", "negative_run_mass",
    "maximum_positive_run", "maximum_negative_run", "body_magnitude",
    "persistence_imbalance",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "source_day", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "positive_bodies", "negative_bodies",
    "positive_run_mass", "negative_run_mass", "maximum_positive_run",
    "maximum_negative_run", "body_magnitude", "persistence_imbalance",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env
    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_source() -> pd.DataFrame:
    from sqlalchemy import text
    db = engine()
    try:
        with db.connect() as connection:
            return pd.read_sql_query(text(QUERY), connection, params={"start": START.to_pydatetime(), "end": END.to_pydatetime()})
    finally:
        db.dispose()


def run_geometry(signs: np.ndarray) -> dict[str, int]:
    positive_mass = negative_mass = positive_max = negative_max = 0
    prior = run_sign = run_length = 0
    for raw in signs:
        current = int(raw)
        if current == 0:
            if run_sign > 0: positive_mass += run_length ** 2; positive_max = max(positive_max, run_length)
            elif run_sign < 0: negative_mass += run_length ** 2; negative_max = max(negative_max, run_length)
            prior = run_sign = run_length = 0
        elif current == prior:
            run_length += 1
        else:
            if run_sign > 0: positive_mass += run_length ** 2; positive_max = max(positive_max, run_length)
            elif run_sign < 0: negative_mass += run_length ** 2; negative_max = max(negative_max, run_length)
            prior = run_sign = current; run_length = 1
    if run_sign > 0: positive_mass += run_length ** 2; positive_max = max(positive_max, run_length)
    elif run_sign < 0: negative_mass += run_length ** 2; negative_max = max(negative_max, run_length)
    return {"positive_run_mass": positive_mass, "negative_run_mass": negative_mass,
            "maximum_positive_run": positive_max, "maximum_negative_run": negative_max}


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    required = ["ts", "open", "high", "low", "close"]
    if not set(required).issubset(raw.columns): raise ValueError("TSPI source schema drift")
    frame = raw[required].copy(); frame.ts = pd.to_datetime(frame.ts, utc=True, errors="coerce")
    for column in required[1:]: frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values("ts", kind="mergesort"); frame["source_day"] = frame.ts.dt.floor("D")
    rows = []
    for day, group in frame.groupby("source_day", sort=True):
        expected = pd.date_range(day, day + pd.Timedelta(days=1), freq="1min", inclusive="left")
        prices = group[["open", "high", "low", "close"]]
        valid = bool(
            pd.Timestamp(day).weekday() == 1 and len(group) == 1440 and not group.ts.duplicated().any()
            and group.ts.reset_index(drop=True).equals(pd.Series(expected, name="ts"))
            and np.isfinite(prices).all(axis=1).all() and prices.gt(0).all(axis=1).all()
            and group.high.ge(group[["open", "close"]].max(axis=1)).all()
            and group.low.le(group[["open", "close"]].min(axis=1)).all() and group.high.ge(group.low).all()
        )
        signs = np.sign(np.log(group.close.to_numpy(float) / group.open.to_numpy(float))) if valid else np.array([])
        positive, negative = int(np.count_nonzero(signs > 0)), int(np.count_nonzero(signs < 0))
        valid = bool(valid and positive > 0 and negative > 0)
        geometry = run_geometry(signs) if valid else {"positive_run_mass": 0, "negative_run_mass": 0, "maximum_positive_run": 0, "maximum_negative_run": 0}
        magnitude = float(np.log(group.close / group.open).sum()) if valid else np.nan
        rows.append({
            "source_day": day, "decision_time": day + pd.Timedelta(days=1),
            "feature_available_time": day + pd.Timedelta(days=1), "source_valid": valid,
            "positive_bodies": positive, "negative_bodies": negative, **geometry,
            "body_magnitude": magnitude,
            "persistence_imbalance": geometry["positive_run_mass"] - geometry["negative_run_mass"],
        })
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def signal(features: pd.DataFrame, control: str = "primary") -> pd.Series:
    if control not in ("primary", *CONTROLS): raise ValueError(control)
    if control == "body_count": statistic = features.positive_bodies - features.negative_bodies
    elif control == "body_magnitude": statistic = features.body_magnitude
    elif control == "maximum_run": statistic = features.maximum_positive_run - features.maximum_negative_run
    else: statistic = features.persistence_imbalance
    side = np.sign(statistic).astype("Int64").fillna(0).astype(int)
    if control == "direction_flip": side = -side
    return side.where(features.source_valid & pd.Series(statistic, index=features.index).ne(0), 0)


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    sides = signal(features, control); rows = []; reserved_until = None
    for index in features.index[sides.ne(0)]:
        decision = pd.Timestamp(features.at[index, "decision_time"]); entry = decision + pd.Timedelta(minutes=5); exit_time = entry + pd.Timedelta(hours=12)
        if reserved_until is not None and entry < reserved_until: continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None: continue
        reserved_until = exit_time
        rows.append({
            "candidate": prereg.POLICY_ID, "control": control, "split": split,
            "source_day": features.at[index, "source_day"], "decision_time": decision,
            "feature_available_time": features.at[index, "feature_available_time"],
            "entry_time": entry, "exit_time": exit_time, "side": int(sides.at[index]),
            **{column: features.at[index, column] for column in CLOCK_COLUMNS[9:]},
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = clock[clock.split.eq(split)]
    if selected.empty: return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(selected.side.eq(1).sum()), int(selected.side.eq(-1).sum())
    months = pd.to_datetime(selected.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(selected), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(selected), "max_month_share": int(months.max()) / len(selected)}


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA: raise RuntimeError("TSPI preregistration drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text()); prereg.validate(registration)
    raw = load_source(); features = build_features(raw); primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(features, FEATURES); _write_gzip_csv(primary, CLOCK)
    for name, value in controls.items(): _write_gzip_csv(value, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {
        "protocol_version": "tspi_12_source_v1", "query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(),
        "window": [START.isoformat(), END.isoformat()], "physical_rows": len(raw),
        "features": {"path": str(FEATURES), "sha256": sha(FEATURES), "rows": len(features)},
        "completed_preentry_sources_opened": True, "execution_prices_opened": False,
        "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "rv20_opened": False,
    }
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    MANIFEST.write_text(json.dumps(source_manifest, indent=2, allow_nan=False) + "\n")
    support = {name: stats(primary, name) for name in SPLITS}; checks = {}
    for name, value in support.items():
        checks[f"{name}_minimum_events"] = value["events"] >= MINIMUM[name]
        checks[f"{name}_side_balance"] = value["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = value["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "tspi_12_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(MANIFEST), "sha256": sha(MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "execution_prices_opened": False,
        "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "rv20_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(value), "promotion_authorized": False} for name, value in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}; RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n"); return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args(); report = run(); print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
