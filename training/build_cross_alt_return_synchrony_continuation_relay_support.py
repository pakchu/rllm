"""Materialize outcome-blind source support for frozen CARSC-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_cross_alt_return_synchrony_continuation_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

PREREG_SHA = "fa9ed9a30e8cbd0532b690acd585d371c91c91a725e7c1b8f4f3187544d0c8e4"
ENV = Path("/home/pakchu/rllm/.env")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT")
ALTS = SYMBOLS[1:]
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
SOURCE_DIR = Path("data/cross_alt_return_synchrony_continuation_relay_sources_2023_2026")
SNAPSHOT = SOURCE_DIR / "return_synchrony_features.csv.gz"
MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/cross_alt_return_synchrony_continuation_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/cross_alt_return_synchrony_continuation_relay_controls_2023_2026")
RESULT = Path("results/cross_alt_return_synchrony_continuation_relay_support_2026-08-10.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_synchrony_gate", "raw_median_correlation_above_half",
    "one_block_stale_synchrony", "direction_flip", "forced_long",
)
QUERY = """SELECT ts,open,high,low,close FROM bars_binance WHERE symbol=:symbol AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"""
COLUMNS = (
    "candidate", "control", "split", "block_start", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "synchrony", "synchrony_rank", "btc_return",
    "alt_breadth", "btc_variation", "btc_variation_rank",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def strict_prior_rank(values: pd.Series, lookback: int = 270, minimum: int = 180) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    output = np.full(len(numeric), np.nan)
    history: list[float] = []
    for index, current in enumerate(numeric):
        prior = history[-lookback:]
        if np.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior)
            output[index] = (np.sum(array < current) + 0.5 * np.sum(array == current)) / len(array)
        if np.isfinite(current):
            history.append(float(current))
    return pd.Series(output, index=values.index)


def engine():
    from preprocessing.live_db_features import sqlalchemy_engine_from_env

    return sqlalchemy_engine_from_env(ENV)


def load_symbol(connection: Any, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sqlalchemy import text
    raw = pd.read_sql_query(
        text(QUERY), connection,
        params={"symbol": symbol, "start": START.to_pydatetime(), "end": END.to_pydatetime()},
    )
    raw["ts"] = pd.to_datetime(raw.ts, utc=True, errors="raise")
    for column in ("open", "high", "low", "close"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    if raw.ts.duplicated().any():
        raise RuntimeError(f"duplicate source timestamp: {symbol}")
    raw["coherent"] = (
        np.isfinite(raw[["open", "high", "low", "close"]]).all(axis=1)
        & raw[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & raw.high.ge(raw[["open", "close"]].max(axis=1))
        & raw.low.le(raw[["open", "close"]].min(axis=1))
        & raw.high.ge(raw.low)
    )
    raw["minute_return"] = np.log(raw.close / raw.open).where(raw.coherent)
    raw["block_start"] = raw.ts.dt.floor("8h")
    grouped = raw.groupby("block_start", sort=True)
    summary = grouped.agg(
        source_rows=("ts", "size"), first_ts=("ts", "min"), last_ts=("ts", "max"),
        coherent=("coherent", "all"), block_open=("open", "first"), block_close=("close", "last"),
        minute_variation_sq=("minute_return", lambda x: float(np.square(x).sum())),
    )
    summary["valid"] = (
        summary.source_rows.eq(480)
        & summary.first_ts.eq(summary.index)
        & summary.last_ts.eq(summary.index + pd.Timedelta(hours=7, minutes=59))
        & summary.coherent
    )
    summary["block_return"] = np.log(summary.block_close / summary.block_open).where(summary.valid)
    returns = raw.set_index("ts")[["minute_return", "block_start"]]
    return returns, summary


def build_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    connection_engine = engine()
    rows_read = 0
    try:
        with connection_engine.connect() as connection:
            btc_minutes, btc_summary = load_symbol(connection, "BTCUSDT")
            rows_read += len(btc_minutes)
            summaries = {"BTCUSDT": btc_summary}
            correlations: dict[str, pd.Series] = {}
            for symbol in ALTS:
                alt_minutes, alt_summary = load_symbol(connection, symbol)
                rows_read += len(alt_minutes)
                summaries[symbol] = alt_summary
                paired = btc_minutes[["minute_return", "block_start"]].join(
                    alt_minutes[["minute_return"]], how="inner", rsuffix="_alt"
                )
                correlations[symbol] = paired.groupby("block_start").apply(
                    lambda x: x.minute_return.corr(x.minute_return_alt)
                )
    finally:
        connection_engine.dispose()
    index = btc_summary.index
    features = pd.DataFrame(index=index)
    features["source_valid"] = pd.concat(
        [summaries[s].valid.reindex(index) for s in SYMBOLS], axis=1
    ).fillna(False).all(axis=1)
    corr = pd.DataFrame(correlations).reindex(index)
    features["synchrony"] = corr.median(axis=1, skipna=False)
    features["synchrony_rank"] = strict_prior_rank(features.synchrony.where(features.source_valid))
    features["btc_return"] = btc_summary.block_return.reindex(index)
    btc_side = np.sign(features.btc_return)
    alt_returns = pd.concat(
        [summaries[s].block_return.reindex(index).rename(s) for s in ALTS], axis=1
    )
    alt_signs = np.sign(alt_returns)
    features["alt_breadth"] = alt_signs.eq(btc_side, axis=0).sum(axis=1)
    features["btc_variation"] = np.sqrt(btc_summary.minute_variation_sq.reindex(index))
    features["btc_variation_rank"] = strict_prior_rank(features.btc_variation.where(features.source_valid))
    features = features.reset_index()
    features["decision_time"] = features.block_start + pd.Timedelta(hours=8)
    return features, {
        "query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(),
        "rows_read": rows_read,
        "valid_joint_blocks": int(features.source_valid.sum()),
        "first_block": features.block_start.min().isoformat(),
        "last_block": features.block_start.max().isoformat(),
    }


def conditions(features: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series]:
    synchrony = features.synchrony
    synchrony_rank = features.synchrony_rank
    if control == "one_block_stale_synchrony":
        synchrony = synchrony.shift(1)
        synchrony_rank = synchrony_rank.shift(1)
    if control == "no_synchrony_gate":
        synchrony_gate = pd.Series(True, index=features.index)
    elif control == "raw_median_correlation_above_half":
        synchrony_gate = synchrony.gt(0.5)
    else:
        synchrony_gate = synchrony_rank.ge(0.70)
    active = (
        features.source_valid
        & np.isfinite(synchrony)
        & synchrony_gate
        & features.btc_return.ne(0)
        & features.alt_breadth.ge(4)
        & features.btc_variation_rank.ge(0.65)
    )
    active = active & ~active.shift(1, fill_value=False) & features.source_valid.shift(1, fill_value=False)
    side = np.sign(features.btc_return)
    if control == "direction_flip":
        side = -side
    elif control == "forced_long":
        side = pd.Series(1, index=features.index)
    return active, side


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    active, side = conditions(features, control)
    rows = []
    next_allowed = None
    for index in features.index[active]:
        decision = pd.Timestamp(features.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=8)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next(
            (name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None
        )
        if split is None:
            continue
        next_allowed = exit_time
        rows.append({
            "candidate": "CARSC-8", "control": control, "split": split,
            "block_start": features.at[index, "block_start"], "decision_time": decision,
            "feature_available_time": decision, "entry_time": entry, "exit_time": exit_time,
            "side": int(side.at[index]), "synchrony": float(features.at[index, "synchrony"]),
            "synchrony_rank": float(features.at[index, "synchrony_rank"]),
            "btc_return": float(features.at[index, "btc_return"]),
            "alt_breadth": int(features.at[index, "alt_breadth"]),
            "btc_variation": float(features.at[index, "btc_variation"]),
            "btc_variation_rank": float(features.at[index, "btc_variation_rank"]),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = clock[clock.split.eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(selected.side.eq(1).sum())
    shorts = int(selected.side.eq(-1).sum())
    months = selected.entry_time.dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(selected), "longs": longs, "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": int(months.max()) / len(selected),
    }


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("CARSC preregistration drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    features, source = build_features()
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(features, SNAPSHOT)
    _write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items():
        _write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    manifest_core = {
        "protocol_version": "carsc_8_preentry_source_v1",
        "database": {"env_file": str(ENV), "table": "bars_binance", "read_only": True, **source},
        "features": {"path": str(SNAPSHOT), "sha256": sha(SNAPSHOT), "rows": len(features)},
        "postentry_outcomes_opened": False, "gross9_rows_opened": False,
    }
    manifest = {**manifest_core, "manifest_hash": chash(manifest_core)}
    MANIFEST.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {
        key: value
        for name, values in support.items()
        for key, value in (
            (f"{name}_minimum_events", values["events"] >= MINIMUM[name]),
            (f"{name}_side_balance", values["minority_side_share"] >= 0.2),
            (f"{name}_month_concentration", values["max_month_share"] <= 0.45),
        )
    }
    passed = all(checks.values())
    core = {
        "protocol_version": "carsc_8_source_support_v1", "policy_id": "CARSC-8",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(MANIFEST), "sha256": sha(MANIFEST), "manifest_hash": manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {
            name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False}
            for name, frame in controls.items()
        },
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": chash(core)}
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    outcome = run()
    print(json.dumps({"passed": outcome["support_passed"], "support": outcome["support"]}, indent=2))
