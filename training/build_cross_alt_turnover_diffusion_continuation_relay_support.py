"""Materialize outcome-blind source support for frozen CATDCR-8."""
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

from training import preregister_cross_alt_turnover_diffusion_continuation_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

PREREG_SHA = "190840cd984208a873a727cff2796d4b8a6c671f5415ef8e229826e034429f27"
ENV = Path("/home/pakchu/rllm/.env")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT")
ALTS = SYMBOLS[1:]
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
SOURCE_DIR = Path("data/cross_alt_turnover_diffusion_continuation_relay_sources_2023_2026")
SNAPSHOT = SOURCE_DIR / "turnover_diffusion_features.csv.gz"
MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/cross_alt_turnover_diffusion_continuation_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/cross_alt_turnover_diffusion_continuation_relay_controls_2023_2026")
RESULT = Path("results/cross_alt_turnover_diffusion_continuation_relay_support_2026-08-10.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_diffusion_gate", "raw_turnover_entropy", "one_block_stale_diffusion", "direction_flip")
QUERY = """SELECT symbol,date_trunc('day',ts)+(floor(extract(hour from ts)/8)*interval '8 hours') AS block_start,(array_agg(open ORDER BY ts))[1] AS block_open,(array_agg(close ORDER BY ts DESC))[1] AS block_close,max(high) AS block_high,min(low) AS block_low,sum(quote_asset_volume) AS quote_turnover,sum(power(ln(close/open),2)) AS minute_variation_sq,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND high>=open AND high>=close AND low<=open AND low<=close AND low>0 AND close>0 AND quote_asset_volume>=0) AS coherent FROM bars_binance WHERE symbol = ANY(:symbols) AND interval='1m' AND ts>=:start AND ts<:end GROUP BY symbol,block_start ORDER BY block_start,symbol"""
COLUMNS = (
    "candidate", "control", "split", "block_start", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "turnover_entropy", "diffusion_rank", "btc_return",
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


def strict_prior_median_ratio(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    output = np.full(len(numeric), np.nan)
    history: list[float] = []
    for index, current in enumerate(numeric):
        prior = history[-270:]
        if np.isfinite(current) and current > 0 and len(prior) >= 180:
            median = float(np.median(prior))
            output[index] = current / median if median > 0 else np.nan
        if np.isfinite(current) and current > 0:
            history.append(float(current))
    return pd.Series(output, index=values.index)


def entropy(frame: pd.DataFrame) -> pd.Series:
    total = frame.sum(axis=1)
    shares = frame.div(total.replace(0, np.nan), axis=0)
    terms = shares.where(shares > 0)
    return -(terms * np.log(terms)).sum(axis=1, min_count=len(ALTS)) / np.log(len(ALTS))


def engine():
    from preprocessing.live_db_features import sqlalchemy_engine_from_env

    return sqlalchemy_engine_from_env(ENV)


def build_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    from sqlalchemy import text

    connection_engine = engine()
    try:
        with connection_engine.connect() as connection:
            raw = pd.read_sql_query(
                text(QUERY), connection,
                params={"symbols": list(SYMBOLS), "start": START.to_pydatetime(), "end": END.to_pydatetime()},
            )
    finally:
        connection_engine.dispose()
    for column in ("block_start", "first_ts", "last_ts"):
        raw[column] = pd.to_datetime(raw[column], utc=True, errors="raise")
    numeric = ("block_open", "block_close", "block_high", "block_low", "quote_turnover", "minute_variation_sq")
    for column in numeric:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw["valid"] = (
        raw.source_rows.eq(480)
        & raw.distinct_rows.eq(480)
        & raw.first_ts.eq(raw.block_start)
        & raw.last_ts.eq(raw.block_start + pd.Timedelta(hours=7, minutes=59))
        & raw.coherent.eq(True)
        & np.isfinite(raw[list(numeric)]).all(axis=1)
        & raw[["block_open", "block_close", "block_high", "block_low", "quote_turnover"]].gt(0).all(axis=1)
        & raw.minute_variation_sq.ge(0)
    )
    raw["return"] = np.log(raw.block_close / raw.block_open).where(raw.valid)
    raw["normalized_turnover"] = raw.groupby("symbol", group_keys=False).quote_turnover.apply(
        strict_prior_median_ratio
    ).where(raw.valid)
    raw_turnover = raw.pivot(index="block_start", columns="symbol", values="quote_turnover").reindex(columns=SYMBOLS)
    normalized = raw.pivot(index="block_start", columns="symbol", values="normalized_turnover").reindex(columns=SYMBOLS)
    returns = raw.pivot(index="block_start", columns="symbol", values="return").reindex(columns=SYMBOLS)
    validity = raw.pivot(index="block_start", columns="symbol", values="valid").reindex(columns=SYMBOLS).fillna(False)
    variation_sq = raw.pivot(index="block_start", columns="symbol", values="minute_variation_sq").reindex(columns=SYMBOLS)
    features = pd.DataFrame(index=returns.index)
    features["source_valid"] = validity.all(axis=1)
    features["turnover_entropy"] = entropy(normalized[list(ALTS)])
    features["raw_turnover_entropy"] = entropy(raw_turnover[list(ALTS)])
    features["diffusion_rank"] = strict_prior_rank(features.turnover_entropy)
    features["raw_entropy_rank"] = strict_prior_rank(features.raw_turnover_entropy)
    features["btc_return"] = returns.BTCUSDT
    btc_side = np.sign(features.btc_return)
    alt_signs = np.sign(returns[list(ALTS)])
    features["alt_breadth"] = alt_signs.eq(btc_side, axis=0).sum(axis=1)
    features["btc_variation"] = np.sqrt(variation_sq.BTCUSDT)
    features["btc_variation_rank"] = strict_prior_rank(features.btc_variation)
    features = features.reset_index()
    features["decision_time"] = features.block_start + pd.Timedelta(hours=8)
    return features, {
        "query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(),
        "rows_read": len(raw),
        "valid_symbol_blocks": int(raw.valid.sum()),
        "first_block": raw.block_start.min().isoformat(),
        "last_block": raw.block_start.max().isoformat(),
    }


def conditions(features: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series]:
    entropy_value = features.turnover_entropy
    diffusion_rank = features.diffusion_rank
    if control == "raw_turnover_entropy":
        entropy_value = features.raw_turnover_entropy
        diffusion_rank = features.raw_entropy_rank
    elif control == "one_block_stale_diffusion":
        entropy_value = entropy_value.shift(1)
        diffusion_rank = diffusion_rank.shift(1)
    diffusion = pd.Series(True, index=features.index) if control == "no_diffusion_gate" else diffusion_rank.ge(0.70)
    active = (
        features.source_valid
        & np.isfinite(entropy_value)
        & diffusion
        & features.btc_return.ne(0)
        & features.alt_breadth.ge(4)
        & features.btc_variation_rank.ge(0.65)
    )
    side = np.sign(features.btc_return)
    if control == "direction_flip":
        side = -side
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
            "candidate": "CATDCR-8", "control": control, "split": split,
            "block_start": features.at[index, "block_start"], "decision_time": decision,
            "feature_available_time": decision, "entry_time": entry, "exit_time": exit_time,
            "side": int(side.at[index]), "turnover_entropy": float(features.at[index, "turnover_entropy"]),
            "diffusion_rank": float(features.at[index, "diffusion_rank"]),
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
        raise RuntimeError("CATDCR preregistration drift")
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
        "protocol_version": "catdcr_8_preentry_source_v1",
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
        "protocol_version": "catdcr_8_source_support_v1", "policy_id": "CATDCR-8",
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
