"""Materialize funding-only source support for frozen CAFACR-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_cross_alt_funding_acceleration_contradiction_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
PREREG_SHA = "b80da6fd1df1bd2148340261b87d322151e248af8bd7a5c077751493b699c96c"
START = pd.Timestamp("2023-06-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
SYMBOLS = ("BTCUSDT", *prereg.ALTS)
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_btc_contradiction", "btc_change_only", "follow_alt_majority", "direction_flip")
QUERY = """
SELECT symbol,funding_time,funding_rate
FROM funding_rates_binance
WHERE symbol = ANY(:symbols) AND funding_time>=:start AND funding_time<:end
ORDER BY funding_time,symbol
"""
SOURCE_DIR = Path("data/cross_alt_funding_acceleration_contradiction_relay_sources_2023_2026")
FEATURES = SOURCE_DIR / "common_funding_changes.csv.gz"
MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/cross_alt_funding_acceleration_contradiction_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/cross_alt_funding_acceleration_contradiction_relay_controls_2023_2026")
RESULT = Path("results/cross_alt_funding_acceleration_contradiction_relay_support_2026-08-09.json")
DELTA_COLUMNS = tuple(f"delta_{symbol}" for symbol in SYMBOLS)
FEATURE_COLUMNS = (
    "settlement_time", "feature_available_time", "common_valid", "prior_common_consecutive",
    *DELTA_COLUMNS, "positive_alt_count", "negative_alt_count", "alt_majority_side",
    "btc_change_side", "btc_contradiction",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "settlement_time", "feature_available_time",
    "entry_time", "exit_time", "side", "positive_alt_count", "negative_alt_count",
    "alt_majority_side", "btc_change_side", "btc_contradiction", *DELTA_COLUMNS,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


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
            return pd.read_sql_query(
                text(QUERY), connection,
                params={"symbols": list(SYMBOLS), "start": START.to_pydatetime(), "end": END.to_pydatetime()},
            )
    finally:
        db.dispose()


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    required = ["symbol", "funding_time", "funding_rate"]
    if not set(required).issubset(raw.columns):
        raise ValueError("CAFACR funding source schema drift")
    frame = raw[required].copy()
    frame["funding_time"] = pd.to_datetime(frame.funding_time, utc=True, errors="coerce")
    frame["funding_rate"] = pd.to_numeric(frame.funding_rate, errors="coerce")
    frame = frame[frame.symbol.isin(SYMBOLS)].sort_values(["funding_time", "symbol"], kind="mergesort")
    rows: list[dict[str, Any]] = []
    for timestamp, group in frame.groupby("funding_time", sort=True, dropna=False):
        timestamp = pd.Timestamp(timestamp)
        counts = group.symbol.value_counts()
        valid = bool(
            timestamp is not pd.NaT
            and timestamp.minute == timestamp.second == timestamp.microsecond == 0
            and timestamp.hour in (0, 8, 16)
            and set(counts.index) == set(SYMBOLS)
            and counts.eq(1).all()
            and np.isfinite(group.funding_rate).all()
        )
        rates = group.set_index("symbol").funding_rate.to_dict() if valid else {}
        rows.append({
            "settlement_time": timestamp, "feature_available_time": timestamp,
            "common_valid": valid, **{symbol: float(rates.get(symbol, np.nan)) for symbol in SYMBOLS},
        })
    common = pd.DataFrame(rows).sort_values("settlement_time", kind="mergesort").reset_index(drop=True)
    if common.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    output: list[dict[str, Any]] = []
    for index, row in common.iterrows():
        prior_ok = bool(
            index >= 1 and row.common_valid and common.at[index - 1, "common_valid"]
            and row.settlement_time - common.at[index - 1, "settlement_time"] == pd.Timedelta(hours=8)
        )
        deltas = {
            symbol: float(row[symbol] - common.at[index - 1, symbol]) if prior_ok else np.nan
            for symbol in SYMBOLS
        }
        alt = np.asarray([deltas[symbol] for symbol in prereg.ALTS], dtype=float)
        positive = int(np.count_nonzero(alt > 0)) if prior_ok else 0
        negative = int(np.count_nonzero(alt < 0)) if prior_ok else 0
        majority = 1 if positive >= 4 else -1 if negative >= 4 else 0
        btc_side = int(np.sign(deltas["BTCUSDT"])) if prior_ok and deltas["BTCUSDT"] != 0 else 0
        output.append({
            "settlement_time": row.settlement_time, "feature_available_time": row.feature_available_time,
            "common_valid": bool(row.common_valid), "prior_common_consecutive": prior_ok,
            **{f"delta_{symbol}": deltas[symbol] for symbol in SYMBOLS},
            "positive_alt_count": positive, "negative_alt_count": negative,
            "alt_majority_side": majority, "btc_change_side": btc_side,
            "btc_contradiction": bool(majority != 0 and btc_side == -majority),
        })
    return pd.DataFrame(output, columns=FEATURE_COLUMNS)


def signal(features: pd.DataFrame, control: str = "primary") -> pd.Series:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    base_valid = features.common_valid & features.prior_common_consecutive
    if control == "btc_change_only":
        eligible = base_valid & features.btc_change_side.ne(0)
        side = -features.btc_change_side
    else:
        eligible = base_valid & features.alt_majority_side.ne(0)
        if control != "no_btc_contradiction":
            eligible &= features.btc_contradiction
        side = -features.alt_majority_side
        if control in ("follow_alt_majority", "direction_flip"):
            side = features.alt_majority_side
    return side.astype(int).where(eligible, 0)


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    sides = signal(features, control)
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for index in features.index[sides.ne(0)]:
        settlement = pd.Timestamp(features.at[index, "settlement_time"])
        available = pd.Timestamp(features.at[index, "feature_available_time"])
        entry = settlement + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=8)
        if available >= entry or (reserved_until is not None and entry < reserved_until):
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        reserved_until = exit_time
        rows.append({
            "candidate": prereg.POLICY_ID, "control": control, "split": split,
            "settlement_time": settlement, "feature_available_time": available,
            "entry_time": entry, "exit_time": exit_time, "side": int(sides.at[index]),
            **{column: features.at[index, column] for column in CLOCK_COLUMNS[8:]},
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = clock[clock.split.eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(selected.side.eq(1).sum()), int(selected.side.eq(-1).sum())
    months = pd.to_datetime(selected.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(selected), "longs": longs, "shorts": shorts,
            "minority_side_share": min(longs, shorts) / len(selected),
            "max_month_share": int(months.max()) / len(selected)}


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("CAFACR preregistration drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    raw = load_source()
    features = build_features(raw)
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(features, FEATURES); _write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items(): _write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {
        "protocol_version": "cafacr_8_funding_source_v1", "query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(),
        "window": [START.isoformat(), END.isoformat()], "symbols": list(SYMBOLS),
        "rows": {"physical": len(raw), "common_features": len(features)},
        "feature_output": {"path": str(FEATURES), "sha256": sha(FEATURES)},
        "funding_incidence_opened": True, "btc_price_rows_opened": False,
        "execution_prices_opened": False, "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False,
    }
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    MANIFEST.write_text(json.dumps(source_manifest, indent=2, allow_nan=False) + "\n")
    support = {name: stats(primary, name) for name in SPLITS}; checks = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "cafacr_8_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(MANIFEST), "sha256": sha(MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
        "funding_incidence_opened": True, "btc_price_rows_opened": False, "execution_prices_opened": False,
        "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()},
        "rv20_q90": {"entry_filter": False, "opened": False}, "support": support, "support_checks": checks,
        "support_passed": passed, "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    result = run()
    print(json.dumps({"passed": result["support_passed"], "support": result["support"]}, indent=2))
