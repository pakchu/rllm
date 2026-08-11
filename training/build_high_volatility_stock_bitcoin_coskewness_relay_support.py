"""Materialize outcome-blind source support for frozen HVSBCR-24."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from training import preregister_high_volatility_stock_bitcoin_coskewness_relay as prereg
from training.build_bitcoin_stock_correlation_break_relay_support import (
    CLOSED_DATES,
    EARLY_CLOSES,
    cash_close_time,
    expected_session_dates,
    write_gzip_csv,
)


ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path("training/build_high_volatility_stock_bitcoin_coskewness_relay_support.py")
PREREG_SHA = "4724150e918fe5408f7c9dc9bebfec1fdfd65e982ebb5c8d2e1f13b17518d8a5"
SOURCE_DIR = Path("data/high_volatility_stock_bitcoin_coskewness_relay_sources_2022_2026")
RAW_SPY = SOURCE_DIR / "spy_yahoo_chart.json"
SPY_PANEL = SOURCE_DIR / "spy_sessions.csv.gz"
FEATURE_PANEL = SOURCE_DIR / "hvsbcr_preentry_features.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_stock_bitcoin_coskewness_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_stock_bitcoin_coskewness_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_stock_bitcoin_coskewness_relay_support_2026-08-11.json")
SOURCE_START = pd.Timestamp("2022-01-01T00:00:00Z")
SOURCE_END = pd.Timestamp("2026-08-02T00:00:00Z")
BTC_START = pd.Timestamp("2022-01-02T00:00:00Z")
BTC_END = pd.Timestamp("2026-08-01T00:00:00Z")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_btc_volatility_gate", "raw_coskewness_level", "btc_skew_only", "one_session_stale_coskewness", "direction_flip")
CLOCK_COLUMNS = (
    "candidate", "control", "split", "session_date", "cash_close_time", "feature_available_time",
    "entry_time", "exit_time", "side", "spy_daily_return", "btc_daily_return", "coskewness",
    "coskewness_z", "btc_skewness", "btc_skewness_z", "btc_realized_variation", "btc_variation_rank",
)
QUERY = """
SELECT ts,open,close
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts
""".strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def strict_prior_midrank(values: pd.Series, lookback: int = 252, minimum: int = 126) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if np.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior, dtype=float)
            output.at[index] = (
                np.count_nonzero(array < current) + 0.5 * np.count_nonzero(array == current)
            ) / len(array)
        if np.isfinite(current):
            history.append(float(current))
    return output


def strict_prior_zscore(values: pd.Series, lookback: int = 252, minimum: int = 126) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if np.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior, dtype=float)
            deviation = float(array.std(ddof=0))
            if np.isfinite(deviation) and deviation > 0.0:
                output.at[index] = (float(current) - float(array.mean())) / deviation
        if np.isfinite(current):
            history.append(float(current))
    return output


def _download_equity(symbol: str, base_url: str) -> tuple[bytes, pd.DataFrame]:
    query = urlencode(
        {
            "period1": int(SOURCE_START.timestamp()),
            "period2": int(SOURCE_END.timestamp()),
            "interval": "1d",
            "events": "div,splits",
        }
    )
    request = Request(f"{base_url}?{query}", headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=60) as response:
        parsed = json.loads(response.read())
    result = parsed["chart"]["result"][0]
    meta = result["meta"]
    if meta["symbol"] != symbol or meta["exchangeTimezoneName"] != "America/New_York":
        raise RuntimeError(f"HVSBCR {symbol} Yahoo metadata drift")
    quote = result["indicators"]["quote"][0]
    stable_source = {
        "meta": {
            key: meta.get(key)
            for key in (
                "currency", "symbol", "exchangeName", "fullExchangeName", "instrumentType",
                "firstTradeDate", "timezone", "exchangeTimezoneName", "dataGranularity",
            )
        },
        "timestamp": result["timestamp"],
        "quote": quote,
    }
    payload = json.dumps(
        stable_source, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    frame = pd.DataFrame({"timestamp": result["timestamp"], **quote})
    local = pd.to_datetime(frame.pop("timestamp"), unit="s", utc=True).dt.tz_convert("America/New_York")
    frame.insert(0, "session_date", pd.to_datetime(local.dt.date))
    # Yahoo can vary object-key order while returning identical values. Freeze the
    # research snapshot's tabular order so byte hashes do not depend on transport order.
    frame = frame[["session_date", "open", "high", "low", "close", "volume"]]
    frame = frame.sort_values("session_date").reset_index(drop=True)
    if frame.empty or frame.session_date.duplicated().any():
        raise RuntimeError(f"HVSBCR {symbol} session dates invalid")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    valid = np.isfinite(frame[["open", "high", "low", "close", "volume"]]).all(axis=1)
    valid &= frame[["open", "high", "low", "close"]].gt(0.0).all(axis=1)
    if not valid.all():
        dates = frame.loc[~valid, "session_date"].dt.strftime("%Y-%m-%d").tolist()
        raise RuntimeError(f"HVSBCR {symbol} invalid source rows: {dates[:5]}")
    return payload, frame


def spy_session_panel(spy: pd.DataFrame) -> pd.DataFrame:
    frame = spy.copy()
    expected = expected_session_dates(frame.session_date.min(), frame.session_date.max())
    observed = pd.DatetimeIndex(frame.session_date)
    if not observed.equals(expected):
        missing = expected.difference(observed).strftime("%Y-%m-%d").tolist()
        extra = observed.difference(expected).strftime("%Y-%m-%d").tolist()
        raise RuntimeError(f"HVSBCR SPY/NYSE calendar mismatch missing={missing[:5]} extra={extra[:5]}")
    closes = frame.session_date.map(cash_close_time)
    frame["cash_close_time"] = [value[0] for value in closes]
    frame["close_local_time"] = [value[1] for value in closes]
    frame["early_close"] = [value[2] for value in closes]
    frame["spy_daily_return"] = np.log(frame.close / frame.close.shift(1))
    return frame


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_bars() -> pd.DataFrame:
    from sqlalchemy import text

    engine = postgres_engine()
    try:
        frame = pd.read_sql_query(
            text(QUERY), engine, params={"start": BTC_START.to_pydatetime(), "end": BTC_END.to_pydatetime()}
        )
    finally:
        engine.dispose()
    if frame.columns.tolist() != ["ts", "open", "close"]:
        raise RuntimeError("HVSBCR BTC schema drift")
    frame.ts = pd.to_datetime(frame.ts, utc=True, errors="raise")
    frame = frame.sort_values("ts").reset_index(drop=True)
    expected = pd.date_range(BTC_START, BTC_END, freq="1min", inclusive="left")
    if len(frame) != len(expected) or not frame.ts.equals(pd.Series(expected, name="ts")):
        raise RuntimeError("HVSBCR BTC source is not the exact requested 1m grid")
    for column in ("open", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame[["open", "close"]]).all(axis=None) or not frame[["open", "close"]].gt(0).all(axis=None):
        raise RuntimeError("HVSBCR BTC prices invalid")
    return frame.set_index("ts")


def rolling_coskewness(btc: pd.Series, spy: pd.Series, window: int = 63) -> pd.Series:
    output = pd.Series(np.nan, index=btc.index, dtype=float)
    for index in range(window - 1, len(output)):
        b = btc.iloc[index - window + 1:index + 1].to_numpy(dtype=float)
        s = spy.iloc[index - window + 1:index + 1].to_numpy(dtype=float)
        if not np.isfinite(b).all() or not np.isfinite(s).all():
            continue
        bsd, svar = float(b.std(ddof=0)), float(s.var(ddof=0))
        if bsd > 0.0 and svar > 0.0:
            output.iloc[index] = float(np.mean((b - b.mean()) * np.square(s - s.mean())) / (bsd * svar))
    return output


def rolling_skewness(values: pd.Series, window: int = 63) -> pd.Series:
    output = pd.Series(np.nan, index=values.index, dtype=float)
    for index in range(window - 1, len(output)):
        sample = values.iloc[index - window + 1:index + 1].to_numpy(dtype=float)
        if np.isfinite(sample).all() and float(sample.std(ddof=0)) > 0.0:
            output.iloc[index] = float(np.mean(np.power((sample - sample.mean()) / sample.std(ddof=0), 3)))
    return output


def build_features(spy: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    frame = spy.copy()
    variations: list[float] = []
    btc_closes: list[float] = []
    for close in frame.cash_close_time:
        close = pd.Timestamp(close)
        window = bars.loc[close - pd.Timedelta(hours=24):close - pd.Timedelta(minutes=1)]
        value = np.nan
        if len(window) == 1440:
            value = float(np.sqrt(np.square(np.log(window.close.to_numpy() / window.open.to_numpy())).sum()))
        variations.append(value)
        btc_closes.append(float(bars.at[close - pd.Timedelta(minutes=1), "close"]) if close - pd.Timedelta(minutes=1) in bars.index else np.nan)
    frame["btc_realized_variation"] = variations
    frame["btc_close"] = btc_closes
    frame["btc_daily_return"] = np.log(frame.btc_close / frame.btc_close.shift(1))
    frame["coskewness"] = rolling_coskewness(frame.btc_daily_return, frame.spy_daily_return)
    frame["coskewness_z"] = strict_prior_zscore(frame.coskewness)
    frame["btc_skewness"] = rolling_skewness(frame.btc_daily_return)
    frame["btc_skewness_z"] = strict_prior_zscore(frame.btc_skewness)
    frame["btc_variation_rank"] = strict_prior_midrank(frame.btc_realized_variation)
    return frame


def signal(features: pd.DataFrame, control: str) -> pd.Series:
    score = features.coskewness_z.copy()
    zscore = features.coskewness_z.copy()
    variation_gate = features.btc_variation_rank.ge(0.65)
    if control == "raw_coskewness_level":
        score = features.coskewness
    elif control == "btc_skew_only":
        score, zscore = features.btc_skewness_z, features.btc_skewness_z
    elif control == "one_session_stale_coskewness":
        score, zscore = score.shift(1), zscore.shift(1)
    eligible = score.ne(0) & score.notna() & zscore.abs().ge(0.75) & variation_gate
    if control == "no_btc_volatility_gate":
        eligible = score.ne(0) & score.notna() & zscore.abs().ge(0.75)
    side = -np.sign(score).astype("Int64").fillna(0).astype(int).where(eligible, 0)
    return -side if control == "direction_flip" else side


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    sides = signal(features, control)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in features.index[sides.ne(0)]:
        close = pd.Timestamp(features.at[index, "cash_close_time"])
        feature = close + pd.Timedelta(minutes=5)
        entry = feature + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=24)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next(
            (name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None
        )
        if split is None:
            continue
        next_allowed = exit_time
        rows.append(
            {
                "candidate": "HVSBCR-24", "control": control, "split": split,
                "session_date": features.at[index, "session_date"], "cash_close_time": close,
                "feature_available_time": feature, "entry_time": entry, "exit_time": exit_time,
                "side": int(sides.at[index]),
                "spy_daily_return": float(features.at[index, "spy_daily_return"]),
                "btc_daily_return": float(features.at[index, "btc_daily_return"]),
                "coskewness": float(features.at[index, "coskewness"]),
                "coskewness_z": float(features.at[index, "coskewness_z"]),
                "btc_skewness": float(features.at[index, "btc_skewness"]),
                "btc_skewness_z": float(features.at[index, "btc_skewness_z"]),
                "btc_realized_variation": float(features.at[index, "btc_realized_variation"]),
                "btc_variation_rank": float(features.at[index, "btc_variation_rank"]),
            }
        )
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = clock[clock.split.eq(split)].copy()
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    entries = pd.to_datetime(subset.entry_time, utc=True)
    longs, shorts = int(subset.side.eq(1).sum()), int(subset.side.eq(-1).sum())
    return {
        "events": len(subset), "longs": longs, "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(subset),
        "max_month_share": int(entries.dt.strftime("%Y-%m").value_counts().max()) / len(subset),
    }


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVSBCR preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    spy_payload, spy = _download_equity("SPY", prereg.SPY_YAHOO_URL)
    sessions = spy_session_panel(spy)
    bars = load_bars()
    features = build_features(sessions, bars)
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    RAW_SPY.write_bytes(spy_payload)
    write_gzip_csv(sessions, SPY_PANEL)
    write_gzip_csv(features, FEATURE_PANEL)
    write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items():
        write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {
        "protocol_version": "hvsbcr_24_sources_v1",
        "spy_url": prereg.SPY_YAHOO_URL,
        "source_window": [SOURCE_START.isoformat(), SOURCE_END.isoformat()],
        "btc_query": QUERY, "btc_window": [BTC_START.isoformat(), BTC_END.isoformat()],
        "btc_rows": len(bars),
        "nyse_calendar": {"closed_dates": sorted(CLOSED_DATES), "early_close_dates": sorted(EARLY_CLOSES)},
        "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
        "outputs": {
            "raw_spy": {"path": str(RAW_SPY), "sha256": sha(RAW_SPY)},
            "spy_sessions": {"path": str(SPY_PANEL), "sha256": sha(SPY_PANEL), "rows": len(sessions)},
            "features": {"path": str(FEATURE_PANEL), "sha256": sha(FEATURE_PANEL), "rows": len(features)},
        },
        "candidate_outcomes_opened": False, "no_imputation": True,
    }
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    SOURCE_MANIFEST.write_text(json.dumps(source_manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    support = {name: stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "hvsbcr_24_source_support_v1", "policy_id": "HVSBCR-24",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA,
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifest": {
            "path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST),
            "manifest_hash": source_manifest["manifest_hash"],
        },
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame),
                "promotion_authorized": False,
            }
            for name, frame in controls.items()
        },
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    result = run()
    print(json.dumps({"passed": result["support_passed"], "support": result["support"]}, indent=2))
