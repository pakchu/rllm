"""Build source-only PPCSR-6 premium-polarity clocks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_premium_polarity_cross_settlement_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
PREREG_SHA = "755a7f244193f66247a693d413ee6a4c7c65e0995a04361b7a32e8418895dda5"
QUERY_START = pd.Timestamp("2023-05-01T00:00Z")
END = pd.Timestamp("2026-08-01T00:00Z")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00Z"), pd.Timestamp("2024-01-01T00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00Z"), pd.Timestamp("2025-01-01T00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00Z"), pd.Timestamp("2026-01-01T00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("thirty_minute_persistence", "one_event_stale_cross", "direction_flip", "same_clock_forced_long")
QUERY = """SELECT ts,open,high,low,close,count(*) OVER (PARTITION BY ts) AS duplicate_count FROM bars_binance_premium WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"""
SOURCE_DIR = Path("data/premium_polarity_cross_settlement_relay_sources_2023_2026")
FEATURES = SOURCE_DIR / "premium_crosses.csv.gz"
MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/premium_polarity_cross_settlement_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/premium_polarity_cross_settlement_relay_controls_2023_2026")
RESULT = Path("results/premium_polarity_cross_settlement_relay_support_2026-08-09.json")
FEATURE_COLUMNS = (
    "decision_time", "feature_available_time", "source_valid", "current_close",
    "prior_sign", "persistent_60", "persistent_30", "eligible_state",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "current_close", "prior_sign",
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


def load_premium() -> pd.DataFrame:
    from sqlalchemy import text
    db = engine()
    try:
        with db.connect() as connection:
            return pd.read_sql_query(text(QUERY), connection, params={"start": QUERY_START.to_pydatetime(), "end": END.to_pydatetime()})
    finally:
        db.dispose()


def prepare_premium(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"ts", "open", "high", "low", "close", "duplicate_count"}
    if set(frame) != required:
        raise ValueError("PPCSR premium schema drift")
    result = frame.copy(); result["ts"] = pd.to_datetime(result.ts, utc=True, errors="coerce")
    numeric = ["open", "high", "low", "close", "duplicate_count"]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    valid = (
        result.ts.notna() & np.isfinite(result[numeric]).all(axis=1)
        & result.high.ge(result[["open", "close"]].max(axis=1))
        & result.low.le(result[["open", "close"]].min(axis=1))
        & result.high.ge(result.low) & result.duplicate_count.eq(1)
    )
    result["source_valid"] = valid
    if result.ts.duplicated().any():
        raise RuntimeError("PPCSR duplicate timestamps")
    return result.sort_values("ts", kind="mergesort").set_index("ts")


def polarity_cross(premium: pd.DataFrame, decision: pd.Timestamp):
    expected = pd.date_range(decision - pd.Timedelta(minutes=61), decision, freq="1min", inclusive="left")
    window = premium.reindex(expected)
    if len(expected) != 61 or window.source_valid.isna().any() or not bool(window.source_valid.all()):
        return None
    closes = pd.to_numeric(window.close, errors="coerce").to_numpy(float)
    prior, current = closes[:60], float(closes[-1])
    if current == 0 or np.any(prior == 0) or not np.isfinite(closes).all():
        return current, 0, False, False, 0
    prior_signs = np.sign(prior)
    sign60 = int(prior_signs[0]) if np.all(prior_signs == prior_signs[0]) else 0
    signs30 = prior_signs[-30:]
    sign30 = int(signs30[0]) if np.all(signs30 == signs30[0]) else 0
    current_sign = int(np.sign(current))
    persistent60 = sign60 != 0 and current_sign == -sign60
    persistent30 = sign30 != 0 and current_sign == -sign30
    return current, sign60, persistent60, persistent30, current_sign if persistent60 else 0


def build_features(premium: pd.DataFrame) -> pd.DataFrame:
    premium = prepare_premium(premium); rows = []
    first = QUERY_START.ceil("5min") + pd.Timedelta(minutes=65)
    for decision in pd.date_range(first, END, freq="5min", inclusive="left"):
        values = polarity_cross(premium, decision)
        valid = values is not None
        current, prior_sign, persistent60, persistent30, state = values if valid else (np.nan, 0, False, False, 0)
        rows.append({
            "decision_time": decision, "feature_available_time": decision + pd.Timedelta(minutes=5),
            "source_valid": valid, "current_close": current, "prior_sign": prior_sign,
            "persistent_60": persistent60, "persistent_30": persistent30,
            "eligible_state": state,
        })
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def active_and_side(features: pd.DataFrame, control: str = "primary"):
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    primary = pd.to_numeric(features.eligible_state, errors="raise").astype(int)
    if control == "thirty_minute_persistence":
        current_sign = np.sign(pd.to_numeric(features.current_close, errors="coerce")).fillna(0).astype(int)
        state = current_sign.where(features.source_valid & features.persistent_30, 0)
    elif control == "one_event_stale_cross":
        state = primary.shift(1, fill_value=0)
    else:
        state = primary
    active = features.source_valid & state.ne(0)
    side = state.copy()
    if control == "direction_flip":
        side = -side
    elif control == "same_clock_forced_long":
        side = side.where(side.eq(0), 1)
    return active & side.ne(0), side.astype(int)


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, sides = active_and_side(features, control); rows = []; reserved = None
    for index in features.index[active & sides.ne(0)]:
        available = pd.Timestamp(features.at[index, "feature_available_time"])
        entry = available + pd.Timedelta(minutes=5); exit_time = entry + pd.Timedelta(hours=6)
        if reserved is not None and entry < reserved:
            continue
        split = next((name for name, (left, right) in SPLITS.items() if entry >= left and exit_time <= right), None)
        if split is None:
            continue
        reserved = exit_time
        rows.append({
            "candidate": prereg.POLICY_ID, "control": control, "split": split,
            "decision_time": pd.Timestamp(features.at[index, "decision_time"]),
            "feature_available_time": available, "entry_time": entry, "exit_time": exit_time,
            "side": int(sides.at[index]), "current_close": float(features.at[index, "current_close"]),
            "prior_sign": int(features.at[index, "prior_sign"]),
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
        raise RuntimeError("PPCSR preregistration drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text()); prereg.validate(registration)
    premium = load_premium()
    features = build_features(premium); primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(features, FEATURES); _write_gzip_csv(primary, CLOCK)
    for name, clock in controls.items():
        _write_gzip_csv(clock, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {
        "protocol_version": "ppcsr_6_source_v1", "query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(), "window": [QUERY_START.isoformat(), END.isoformat()],
        "physical_rows": {"premium": len(premium)},
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
        "protocol_version": "ppcsr_6_source_support_v1", "policy_id": prereg.POLICY_ID,
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
