"""Materialize outcome-blind source support for frozen HVKATR-24."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from training import preregister_high_volatility_kospi_asymmetric_transition_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path("training/build_high_volatility_kospi_asymmetric_transition_relay_support.py")
PREREG_SHA = "614ff1d00ec458bcc9e0f9f6797ed681804de7ed80458695d8c744e888b49fd8"
SOURCE_DIR = Path("data/high_volatility_kospi_asymmetric_transition_relay_sources_2021_2026")
RAW_SOURCE = SOURCE_DIR / "ks11_yahoo_chart.json"
KOSPI_PANEL = SOURCE_DIR / "kospi_sessions.csv.gz"
FEATURE_PANEL = SOURCE_DIR / "hvkatr_preentry_features.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_kospi_asymmetric_transition_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_kospi_asymmetric_transition_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_kospi_asymmetric_transition_relay_support_2026-08-12.json")
SOURCE_START = pd.Timestamp("2021-01-01T00:00:00Z")
SOURCE_END = pd.Timestamp("2026-08-02T00:00:00Z")
SESSION_CUTOFF = pd.Timestamp("2026-08-01")
BTC_START = pd.Timestamp("2021-01-02T06:30:00Z")
BTC_END = pd.Timestamp("2026-08-01T06:31:00Z")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_btc_volatility_gate",
    "no_kospi_shock_gate",
    "one_session_stale_transition",
    "return_level_without_transition",
    "direction_flip",
    "same_clock_forced_long",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "kospi_session_date", "cash_close_time",
    "feature_available_time", "entry_time", "exit_time", "side", "kospi_return",
    "prior_kospi_return", "kospi_shock_rank", "btc_realized_variation", "btc_variation_rank",
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


def strict_prior_midrank(values: pd.Series, lookback: int, minimum: int) -> pd.Series:
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


def _stable_yahoo_payload(result: Mapping[str, Any]) -> bytes:
    meta = result.get("meta") or {}
    indicators = result.get("indicators") or {}
    stable = {
        "meta": {
            key: meta.get(key)
            for key in (
                "currency", "symbol", "exchangeName", "fullExchangeName", "instrumentType",
                "firstTradeDate", "timezone", "exchangeTimezoneName", "dataGranularity",
            )
        },
        "timestamp": result.get("timestamp"),
        "quote": (indicators.get("quote") or [None])[0],
        "events": result.get("events") or {},
    }
    return json.dumps(
        stable, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def normalize_yahoo_chart(payload: bytes) -> tuple[bytes, pd.DataFrame, dict[str, Any]]:
    parsed = json.loads(payload)
    chart = parsed.get("chart") or {}
    if chart.get("error") is not None:
        raise RuntimeError("HVKATR Yahoo chart error")
    results = chart.get("result") or []
    if len(results) != 1:
        raise RuntimeError("HVKATR Yahoo result count drift")
    result = results[0]
    meta = result.get("meta") or {}
    if meta.get("symbol") != "^KS11" or meta.get("exchangeTimezoneName") != "Asia/Seoul":
        raise RuntimeError("HVKATR Yahoo metadata drift")
    timestamps = result.get("timestamp") or []
    quotes = (result.get("indicators") or {}).get("quote") or []
    if len(quotes) != 1 or not timestamps:
        raise RuntimeError("HVKATR Yahoo chart incomplete")
    quote = quotes[0]
    vectors = {name: quote.get(name) for name in ("open", "high", "low", "close", "volume")}
    if any(values is None or len(values) != len(timestamps) for values in vectors.values()):
        raise RuntimeError("HVKATR Yahoo vector length drift")
    local = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("Asia/Seoul")
    frame = pd.DataFrame({"kospi_session_date": pd.to_datetime(local.date), **vectors})
    frame = frame.loc[
        frame.kospi_session_date.ge(SOURCE_START.tz_localize(None))
        & frame.kospi_session_date.lt(SESSION_CUTOFF)
    ].sort_values("kospi_session_date").reset_index(drop=True)
    if frame.empty or frame.kospi_session_date.duplicated().any():
        raise RuntimeError("HVKATR KOSPI session dates invalid")
    if frame.kospi_session_date.dt.dayofweek.ge(5).any():
        raise RuntimeError("HVKATR KOSPI weekend session drift")
    gaps = frame.kospi_session_date.diff().dt.days.dropna()
    if gaps.gt(14).any():
        raise RuntimeError("HVKATR KOSPI implausible session gap")
    for column in vectors:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    prices = frame[["open", "high", "low", "close"]]
    valid = np.isfinite(frame[list(vectors)].to_numpy(dtype=float)).all(axis=1)
    valid &= prices.gt(0).all(axis=1) & frame.volume.ge(0)
    if not valid.all():
        dates = frame.loc[~valid, "kospi_session_date"].dt.strftime("%Y-%m-%d").tolist()
        raise RuntimeError(f"HVKATR invalid KOSPI rows: {dates[:5]}")
    close_local = pd.to_datetime(
        frame.kospi_session_date.dt.strftime("%Y-%m-%d") + " 15:30:00"
    ).dt.tz_localize("Asia/Seoul")
    frame["cash_close_time"] = close_local.dt.tz_convert("UTC")
    metadata = {
        "symbol": "^KS11", "exchange_timezone": "Asia/Seoul", "rows": int(len(frame)),
        "first_session": str(frame.kospi_session_date.iloc[0].date()),
        "last_session": str(frame.kospi_session_date.iloc[-1].date()),
        "native_close_local": "15:30",
    }
    return _stable_yahoo_payload(result), frame, metadata


def download_kospi() -> tuple[bytes, pd.DataFrame, dict[str, Any]]:
    query = urlencode({
        "period1": int(SOURCE_START.timestamp()), "period2": int(SOURCE_END.timestamp()),
        "interval": "1d", "events": "div,splits", "includeAdjustedClose": "false",
    })
    request = Request(f"{prereg.KOSPI_YAHOO_URL}?{query}", headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=60) as response:
        payload = response.read()
    return normalize_yahoo_chart(payload)


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
            text(QUERY), engine,
            params={"start": BTC_START.to_pydatetime(), "end": BTC_END.to_pydatetime()},
        )
    finally:
        engine.dispose()
    return normalize_btc_bars(frame).set_index("ts")


def normalize_btc_bars(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.columns.tolist() != ["ts", "open", "close"]:
        raise RuntimeError("HVKATR BTC schema drift")
    frame = raw.copy()
    frame.ts = pd.to_datetime(frame.ts, utc=True, errors="raise")
    frame = frame.sort_values("ts").reset_index(drop=True)
    expected = pd.date_range(BTC_START, BTC_END, freq="1min", inclusive="left")
    if len(frame) != len(expected) or not frame.ts.equals(pd.Series(expected, name="ts")):
        raise RuntimeError("HVKATR BTC source is not the exact requested 1m grid")
    for column in ("open", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    values = frame[["open", "close"]]
    if not np.isfinite(values.to_numpy(dtype=float)).all() or not values.gt(0).all(axis=None):
        raise RuntimeError("HVKATR BTC prices invalid")
    return frame


def build_features(kospi: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    previous_close: float | None = None
    for row in kospi.itertuples(index=False):
        close_time = pd.Timestamp(row.cash_close_time)
        if close_time < BTC_START + pd.Timedelta(hours=24) or close_time >= BTC_END:
            continue
        window = bars.loc[close_time - pd.Timedelta(hours=24):close_time - pd.Timedelta(minutes=1)]
        variation = (
            float(np.sqrt(np.square(np.log(window.close.to_numpy() / window.open.to_numpy())).sum()))
            if len(window) == 1440 else np.nan
        )
        current_close = float(row.close)
        records.append({
            "kospi_session_date": row.kospi_session_date,
            "cash_close_time": close_time,
            "feature_available_time": close_time + pd.Timedelta(minutes=5),
            "kospi_close": current_close,
            "kospi_return": np.log(current_close / previous_close) if previous_close is not None else np.nan,
            "btc_realized_variation": variation,
        })
        previous_close = current_close
    frame = pd.DataFrame(records)
    frame["prior_kospi_return"] = frame.kospi_return.shift(1)
    frame["kospi_shock_rank"] = strict_prior_midrank(frame.kospi_return.abs(), 252, 126)
    frame["btc_variation_rank"] = strict_prior_midrank(frame.btc_realized_variation, 270, 180)
    return frame


def _signal(features: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series]:
    current = features.kospi_return
    prior = features.prior_kospi_return
    transition = current.ne(0) & prior.ne(0) & current.notna() & prior.notna() & (np.sign(current) != np.sign(prior))
    eligible = transition & features.kospi_shock_rank.ge(0.60) & features.btc_variation_rank.ge(0.65)
    side = np.sign(current).astype("Int64").fillna(0).astype(int)
    if control == "no_btc_volatility_gate":
        eligible = transition & features.kospi_shock_rank.ge(0.60)
    elif control == "no_kospi_shock_gate":
        eligible = transition & features.btc_variation_rank.ge(0.65)
    elif control == "one_session_stale_transition":
        stale_current, stale_prior = current.shift(1), prior.shift(1)
        stale = stale_current.ne(0) & stale_prior.ne(0) & stale_current.notna() & stale_prior.notna() & (np.sign(stale_current) != np.sign(stale_prior))
        eligible = stale & features.kospi_shock_rank.shift(1).ge(0.60) & features.btc_variation_rank.ge(0.65)
        side = np.sign(stale_current).astype("Int64").fillna(0).astype(int)
    elif control == "return_level_without_transition":
        eligible = current.ne(0) & current.notna() & features.kospi_shock_rank.ge(0.60) & features.btc_variation_rank.ge(0.65)
    active_side = side.where(eligible, 0)
    if control == "direction_flip":
        active_side = -active_side
    elif control == "same_clock_forced_long":
        active_side = active_side.ne(0).astype(int)
    return eligible, active_side


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    _, sides = _signal(features, control)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in features.index[sides.ne(0)]:
        available = pd.Timestamp(features.at[index, "feature_available_time"])
        entry = available + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=24)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        next_allowed = exit_time
        rows.append({
            "candidate": "HVKATR-24", "control": control, "split": split,
            "kospi_session_date": str(pd.Timestamp(features.at[index, "kospi_session_date"]).date()),
            "cash_close_time": features.at[index, "cash_close_time"],
            "feature_available_time": available, "entry_time": entry, "exit_time": exit_time,
            "side": int(sides.at[index]), "kospi_return": float(features.at[index, "kospi_return"]),
            "prior_kospi_return": float(features.at[index, "prior_kospi_return"]),
            "kospi_shock_rank": float(features.at[index, "kospi_shock_rank"]),
            "btc_realized_variation": float(features.at[index, "btc_realized_variation"]),
            "btc_variation_rank": float(features.at[index, "btc_variation_rank"]),
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def support_stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = clock[clock.split.eq(split)].copy()
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    entries = pd.to_datetime(subset.entry_time, utc=True)
    longs, shorts = int(subset.side.eq(1).sum()), int(subset.side.eq(-1).sum())
    return {
        "events": int(len(subset)), "longs": longs, "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(subset),
        "max_month_share": int(entries.dt.strftime("%Y-%m").value_counts().max()) / len(subset),
    }


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVKATR preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    payload, kospi, yahoo_meta = download_kospi()
    bars = load_bars()
    features = build_features(kospi, bars)
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    RAW_SOURCE.write_bytes(payload)
    _write_gzip_csv(kospi, KOSPI_PANEL)
    _write_gzip_csv(features, FEATURE_PANEL)
    _write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items():
        _write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {
        "protocol_version": "hvkatr_24_sources_v1", "kospi_url": prereg.KOSPI_YAHOO_URL,
        "source_window": [SOURCE_START.isoformat(), SOURCE_END.isoformat()], "yahoo_metadata": yahoo_meta,
        "btc_query": QUERY, "btc_window": [BTC_START.isoformat(), BTC_END.isoformat()], "btc_rows": len(bars),
        "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
        "outputs": {
            "raw_kospi": {"path": str(RAW_SOURCE), "sha256": sha(RAW_SOURCE)},
            "kospi_panel": {"path": str(KOSPI_PANEL), "sha256": sha(KOSPI_PANEL), "rows": len(kospi)},
            "features": {"path": str(FEATURE_PANEL), "sha256": sha(FEATURE_PANEL), "rows": len(features)},
        },
        "candidate_outcomes_opened": False, "no_imputation": True,
    }
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    SOURCE_MANIFEST.write_text(json.dumps(source_manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    support = {name: support_stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "hvkatr_24_source_support_v1", "policy_id": "HVKATR-24",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()},
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
