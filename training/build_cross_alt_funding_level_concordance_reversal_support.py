"""Materialize outcome-blind source support for frozen CAFLCR-8."""
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

from training import preregister_cross_alt_funding_level_concordance_reversal as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

ENV_FILE = "/home/pakchu/rllm/.env"
PREREG_SHA = "a6a6543c0a1a3bde9dcea3f1f6fa2ab9cce6b187f0bf133c8c5f84b38364fc89"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT")
ALTS = SYMBOLS[1:]
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_crowding_rank", "no_variation_gate", "alt_only_breadth", "direction_flip")
FUNDING_QUERY = """SELECT symbol,funding_time,funding_rate,mark_price FROM funding_rates_binance WHERE symbol = ANY(:symbols) AND funding_time>=:start AND funding_time<:end ORDER BY funding_time,symbol"""
MARKET_QUERY = """SELECT date_bin('1 hour',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS hour_start,sum(power(ln(close/open),2)) AS minute_variation_sq,count(*) AS source_rows,count(DISTINCT ts) AS distinct_rows,min(ts) AS first_ts,max(ts) AS last_ts,bool_and(open>0 AND close>0) AS coherent FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end GROUP BY hour_start ORDER BY hour_start"""
SOURCE_DIR = Path("data/cross_alt_funding_level_concordance_reversal_sources_2023_2026")
FEATURES = SOURCE_DIR / "common_funding_residual_features.csv.gz"
MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/cross_alt_funding_level_concordance_reversal_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/cross_alt_funding_level_concordance_reversal_controls_2023_2026")
RESULT = Path("results/cross_alt_funding_level_concordance_reversal_support_2026-08-10.json")
CLOCK_COLUMNS = (
    "candidate", "control", "split", "funding_time", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "alt_common_sign", "alt_breadth", "btc_residual",
    "crowding_magnitude", "crowding_rank", "btc_variation", "btc_variation_rank",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
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


def funding_residual(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    output = np.full(len(numeric), np.nan)
    history: list[float] = []
    for index, current in enumerate(numeric):
        prior = history[-90:]
        if np.isfinite(current) and len(prior) >= 60:
            output[index] = current - float(np.median(prior))
        if np.isfinite(current):
            history.append(float(current))
    return pd.Series(output, index=values.index)


def engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    from sqlalchemy import text

    connection_engine = engine()
    try:
        with connection_engine.connect() as connection:
            funding = pd.read_sql_query(
                text(FUNDING_QUERY), connection,
                params={"symbols": list(SYMBOLS), "start": START.to_pydatetime(), "end": END.to_pydatetime()},
            )
            market = pd.read_sql_query(
                text(MARKET_QUERY), connection,
                params={"start": (START - pd.Timedelta(hours=24)).to_pydatetime(), "end": END.to_pydatetime()},
            )
    finally:
        connection_engine.dispose()
    return funding, market


def build_features(funding: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    if funding.columns.tolist() != ["symbol", "funding_time", "funding_rate", "mark_price"]:
        raise RuntimeError("CAFLCR funding schema drift")
    if funding.duplicated(["symbol", "funding_time"]).any():
        raise RuntimeError("CAFLCR duplicate funding row")
    funding = funding.copy()
    funding["funding_time"] = pd.to_datetime(funding.funding_time, utc=True, errors="raise")
    for column in ("funding_rate", "mark_price"):
        funding[column] = pd.to_numeric(funding[column], errors="coerce")
    funding["valid"] = np.isfinite(funding[["funding_rate", "mark_price"]]).all(axis=1) & funding.mark_price.gt(0)
    funding["residual"] = funding.groupby("symbol", group_keys=False).funding_rate.apply(funding_residual).where(funding.valid)
    rates = funding.pivot(index="funding_time", columns="symbol", values="funding_rate").reindex(columns=SYMBOLS)
    marks = funding.pivot(index="funding_time", columns="symbol", values="mark_price").reindex(columns=SYMBOLS)
    residuals = funding.pivot(index="funding_time", columns="symbol", values="residual").reindex(columns=SYMBOLS)
    valid = funding.pivot(index="funding_time", columns="symbol", values="valid").reindex(columns=SYMBOLS).fillna(False)

    market = market.copy()
    for column in ("hour_start", "first_ts", "last_ts"):
        market[column] = pd.to_datetime(market[column], utc=True, errors="raise")
    for column in ("minute_variation_sq", "source_rows", "distinct_rows"):
        market[column] = pd.to_numeric(market[column], errors="coerce")
    market["valid"] = (
        market.source_rows.eq(60) & market.distinct_rows.eq(60) & market.coherent.eq(True)
        & market.first_ts.eq(market.hour_start)
        & market.last_ts.eq(market.hour_start + pd.Timedelta(minutes=59))
        & np.isfinite(market.minute_variation_sq) & market.minute_variation_sq.ge(0)
    )
    hourly = market.set_index("hour_start").sort_index()
    variation = hourly.minute_variation_sq.where(hourly.valid).rolling(24, min_periods=24).sum()
    variation.index = variation.index + pd.Timedelta(hours=1)

    features = pd.DataFrame(index=rates.index)
    features["source_valid"] = (
        valid.all(axis=1) & np.isfinite(rates).all(axis=1) & marks.gt(0).all(axis=1)
        & np.isfinite(residuals).all(axis=1)
    )
    alt_signs = np.sign(residuals[list(ALTS)])
    positive = alt_signs.gt(0).sum(axis=1)
    negative = alt_signs.lt(0).sum(axis=1)
    features["alt_common_sign"] = np.where(positive >= 4, 1, np.where(negative >= 4, -1, 0))
    features["alt_breadth"] = np.maximum(positive, negative)
    features["btc_residual"] = residuals.BTCUSDT
    features["btc_confirmed"] = np.sign(features.btc_residual).eq(features.alt_common_sign) & features.alt_common_sign.ne(0)
    features["crowding_magnitude"] = residuals[list(ALTS)].abs().median(axis=1)
    features["crowding_rank"] = strict_prior_rank(features.crowding_magnitude.where(features.source_valid))
    features["btc_variation"] = np.sqrt(variation.reindex(features.index))
    features["btc_variation_rank"] = strict_prior_rank(features.btc_variation.where(features.source_valid))
    features = features.reset_index()
    features["decision_time"] = features.funding_time
    features["feature_available_time"] = features.funding_time
    return features


def conditions(features: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series]:
    crowding_ok = pd.Series(True, index=features.index) if control == "no_crowding_rank" else features.crowding_rank.ge(0.70)
    variation_ok = pd.Series(True, index=features.index) if control == "no_variation_gate" else features.btc_variation_rank.ge(0.65)
    confirmation = pd.Series(True, index=features.index) if control == "alt_only_breadth" else features.btc_confirmed
    active = (
        features.source_valid & features.alt_common_sign.ne(0) & confirmation & crowding_ok & variation_ok
        & np.isfinite(features.btc_residual) & np.isfinite(features.btc_variation)
    )
    side = -features.alt_common_sign
    if control == "direction_flip":
        side = -side
    return active, side


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    active, side = conditions(features, control)
    rows = []
    reserved_until = None
    for index in features.index[active]:
        decision = pd.Timestamp(features.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=8)
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next(
            (name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None
        )
        if split is None:
            continue
        reserved_until = exit_time
        rows.append({
            "candidate": "CAFLCR-8", "control": control, "split": split,
            "funding_time": features.at[index, "funding_time"], "decision_time": decision,
            "feature_available_time": features.at[index, "feature_available_time"],
            "entry_time": entry, "exit_time": exit_time, "side": int(side.at[index]),
            **{column: features.at[index, column] for column in CLOCK_COLUMNS[9:]},
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = clock[clock.split.eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(selected.side.eq(1).sum())
    shorts = int(selected.side.eq(-1).sum())
    months = pd.to_datetime(selected.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(selected), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(selected), "max_month_share": int(months.max()) / len(selected)}


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("CAFLCR preregistration drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text()); prereg.validate(registration)
    funding, market = load_sources(); features = build_features(funding, market)
    primary = build_clock(features); controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(features, FEATURES); _write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items(): _write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {
        "protocol_version": "caflcr_8_source_v1",
        "queries": {"funding_sha256": hashlib.sha256(FUNDING_QUERY.encode()).hexdigest(), "market_sha256": hashlib.sha256(MARKET_QUERY.encode()).hexdigest()},
        "window": [START.isoformat(), END.isoformat()], "funding_rows": len(funding), "market_hour_rows": len(market),
        "features": {"path": str(FEATURES), "sha256": sha(FEATURES), "rows": len(features)},
        "completed_preentry_sources_opened": True, "execution_prices_opened": False,
        "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "rv20_opened": False,
    }
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    MANIFEST.write_text(json.dumps(source_manifest, indent=2, allow_nan=False) + "\n")
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {key: value for name, values in support.items() for key, value in ((f"{name}_minimum_events", values["events"] >= MINIMUM[name]), (f"{name}_side_balance", values["minority_side_share"] >= 0.2), (f"{name}_month_concentration", values["max_month_share"] <= 0.45))}
    passed = all(checks.values())
    core = {
        "protocol_version": "caflcr_8_source_support_v1", "policy_id": "CAFLCR-8",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(MANIFEST), "sha256": sha(MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "execution_prices_opened": False,
        "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "rv20_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n"); return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args(); outcome = run()
    print(json.dumps({"passed": outcome["support_passed"], "support": outcome["support"]}, indent=2))
