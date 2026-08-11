"""Materialize outcome-blind source support for frozen HVDXYFT-24."""
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

from training import preregister_high_volatility_usd_index_forecast_threshold_relay as prereg
from training.build_bitcoin_stock_correlation_break_relay_support import (
    CLOSED_DATES,
    EARLY_CLOSES,
    cash_close_time,
    expected_session_dates,
    write_gzip_csv,
)
from training.build_high_volatility_stock_bitcoin_coskewness_relay_support import (
    strict_prior_midrank,
)


ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path("training/build_high_volatility_usd_index_forecast_threshold_relay_support.py")
PREREG_SHA256 = "40e4ffc42dc3359b3bea52ffd990a6c56cbbd11e1f14b6370bbad504003322cd"
SYMBOLS = ("UUP",)
SOURCE_DIR = Path("data/high_volatility_usd_index_forecast_threshold_relay_sources_2022_2026")
RAW_PATHS = {symbol: SOURCE_DIR / f"{symbol.lower()}_yahoo_adjusted_chart.json" for symbol in SYMBOLS}
EQUITY_PANEL = SOURCE_DIR / "common_adjusted_sessions.csv.gz"
BTC_SOURCE = SOURCE_DIR / "btc_1m_ts_open_close.csv.gz"
FEATURE_PANEL = SOURCE_DIR / "hvdxyft_preentry_features.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_usd_index_forecast_threshold_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_usd_index_forecast_threshold_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_usd_index_forecast_threshold_relay_support_2026-08-12.json")
YAHOO_CHART_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
SOURCE_START = pd.Timestamp("2022-01-01T00:00:00Z")
SOURCE_END = pd.Timestamp("2026-08-02T00:00:00Z")
SESSION_CUTOFF = pd.Timestamp("2026-08-01")
BTC_START = pd.Timestamp("2022-01-02T00:00:00Z")
BTC_END = pd.Timestamp("2026-08-01T00:00:00Z")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_btc_volatility_gate",
    "direction_flip",
    "one_session_stale_forecast",
    "uup_sign_only",
    "btc_ar_only",
    "same_clock_forced_long",
)
BTC_QUERY = """
SELECT ts,open,close
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts
""".strip()
CLOCK_COLUMNS = (
    "candidate",
    "control",
    "split",
    "session_date",
    "cash_close_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
    "btc_forecast",
    "forecast_magnitude_rank",
    "uup_return",
    "btc_return",
    "btc_realized_variation",
    "btc_variation_rank",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _stable_yahoo_payload(result: Mapping[str, Any]) -> bytes:
    """Retain only research fields, with transport-independent object ordering."""
    meta = result.get("meta") or {}
    indicators = result.get("indicators") or {}
    stable = {
        "meta": {
            key: meta.get(key)
            for key in (
                "currency",
                "symbol",
                "exchangeName",
                "fullExchangeName",
                "instrumentType",
                "firstTradeDate",
                "timezone",
                "exchangeTimezoneName",
                "dataGranularity",
            )
        },
        "timestamp": result.get("timestamp"),
        "quote": (indicators.get("quote") or [None])[0],
        "adjclose": (indicators.get("adjclose") or [None])[0],
        "events": result.get("events") or {},
    }
    return json.dumps(
        stable, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def normalize_yahoo_chart(payload: bytes, symbol: str) -> tuple[bytes, pd.DataFrame, dict[str, Any]]:
    parsed = json.loads(payload)
    chart = parsed.get("chart") or {}
    if chart.get("error") is not None:
        raise RuntimeError(f"HVDXYFT Yahoo chart error for {symbol}")
    results = chart.get("result") or []
    if len(results) != 1:
        raise RuntimeError(f"HVDXYFT Yahoo result count drift for {symbol}")
    result = results[0]
    meta = result.get("meta") or {}
    if meta.get("symbol") != symbol or meta.get("exchangeTimezoneName") != "America/New_York":
        raise RuntimeError(f"HVDXYFT {symbol} Yahoo metadata drift")
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    quotes = indicators.get("quote") or []
    adjusted = indicators.get("adjclose") or []
    if len(quotes) != 1 or len(adjusted) != 1 or not timestamps:
        raise RuntimeError(f"HVDXYFT {symbol} adjusted chart incomplete")
    quote = quotes[0]
    vectors = {name: quote.get(name) for name in ("open", "high", "low", "close", "volume")}
    vectors["adjusted_close"] = adjusted[0].get("adjclose")
    if any(values is None or len(values) != len(timestamps) for values in vectors.values()):
        raise RuntimeError(f"HVDXYFT {symbol} Yahoo vector length drift")
    local = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("America/New_York")
    frame = pd.DataFrame({"session_date": pd.to_datetime(local.date), **vectors})
    frame = frame.loc[
        frame.session_date.ge(SOURCE_START.tz_localize(None))
        & frame.session_date.lt(SESSION_CUTOFF)
    ].sort_values("session_date").reset_index(drop=True)
    if frame.empty or frame.session_date.duplicated().any():
        raise RuntimeError(f"HVDXYFT {symbol} session dates invalid")
    for column in (*vectors,):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    prices = frame[["open", "high", "low", "close", "adjusted_close"]]
    valid = np.isfinite(frame[[*vectors]].to_numpy(dtype=float)).all(axis=1)
    valid &= prices.gt(0.0).all(axis=1) & frame.volume.ge(0.0)
    if not valid.all():
        dates = frame.loc[~valid, "session_date"].dt.strftime("%Y-%m-%d").tolist()
        raise RuntimeError(f"HVDXYFT {symbol} invalid adjusted rows: {dates[:5]}")
    metadata = {
        "symbol": symbol,
        "exchange_timezone": "America/New_York",
        "rows": int(len(frame)),
        "first_session": str(frame.session_date.iloc[0].date()),
        "last_session": str(frame.session_date.iloc[-1].date()),
        "adjusted_close_read": True,
    }
    return _stable_yahoo_payload(result), frame, metadata


def download_equity(symbol: str, base_url: str = YAHOO_CHART_BASE) -> tuple[bytes, pd.DataFrame, dict[str, Any]]:
    query = urlencode(
        {
            "period1": int(SOURCE_START.timestamp()),
            "period2": int(SOURCE_END.timestamp()),
            "interval": "1d",
            "events": "div,splits",
            "includeAdjustedClose": "true",
        }
    )
    request = Request(f"{base_url}/{symbol}?{query}", headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=60) as response:
        payload = response.read()
    return normalize_yahoo_chart(payload, symbol)


def build_equity_panel(equities: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    if set(equities) != set(SYMBOLS):
        raise RuntimeError("HVDXYFT equity symbol set drift")
    expected: pd.DatetimeIndex | None = None
    panel: pd.DataFrame | None = None
    for symbol in SYMBOLS:
        source = equities[symbol].copy().sort_values("session_date").reset_index(drop=True)
        observed = pd.DatetimeIndex(pd.to_datetime(source.session_date))
        official = expected_session_dates(observed.min(), observed.max())
        if not observed.equals(official):
            missing = official.difference(observed).strftime("%Y-%m-%d").tolist()
            extra = observed.difference(official).strftime("%Y-%m-%d").tolist()
            raise RuntimeError(
                f"HVDXYFT {symbol}/NYSE calendar mismatch missing={missing[:5]} extra={extra[:5]}"
            )
        if expected is None:
            expected = observed
        elif not observed.equals(expected):
            raise RuntimeError(f"HVDXYFT {symbol} does not share every official common session")
        columns = source[["session_date", "adjusted_close"]].rename(
            columns={"adjusted_close": f"{symbol.lower()}_adjusted_close"}
        )
        panel = columns if panel is None else panel.merge(
            columns, on="session_date", how="inner", validate="one_to_one"
        )
    assert panel is not None and expected is not None
    if len(panel) != len(expected):
        raise RuntimeError("HVDXYFT common-session intersection dropped an official session")
    closes = panel.session_date.map(cash_close_time)
    panel["cash_close_time"] = [item[0] for item in closes]
    panel["close_local_time"] = [item[1] for item in closes]
    panel["early_close"] = [item[2] for item in closes]
    observed_early = set(panel.loc[panel.early_close, "session_date"].dt.strftime("%Y-%m-%d"))
    expected_early = set(EARLY_CLOSES).intersection(panel.session_date.dt.strftime("%Y-%m-%d"))
    if observed_early != expected_early:
        raise RuntimeError("HVDXYFT frozen early-close schedule drift")
    return panel


def normalize_btc_bars(
    raw: pd.DataFrame,
    start: pd.Timestamp = BTC_START,
    end: pd.Timestamp = BTC_END,
) -> pd.DataFrame:
    if raw.columns.tolist() != ["ts", "open", "close"]:
        raise RuntimeError("HVDXYFT BTC schema drift")
    frame = raw.copy()
    frame.ts = pd.to_datetime(frame.ts, utc=True, errors="raise")
    frame = frame.sort_values("ts").reset_index(drop=True)
    expected = pd.date_range(start, end, freq="1min", inclusive="left")
    if len(frame) != len(expected) or not frame.ts.equals(pd.Series(expected, name="ts")):
        raise RuntimeError("HVDXYFT BTC source is not the exact requested 1m grid")
    for column in ("open", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    values = frame[["open", "close"]]
    if not np.isfinite(values.to_numpy(dtype=float)).all() or not values.gt(0.0).all(axis=None):
        raise RuntimeError("HVDXYFT BTC prices invalid")
    return frame


def load_btc_bars(env_file: str = ENV_FILE) -> pd.DataFrame:
    from sqlalchemy import create_engine, text

    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(env_file)
    engine = create_engine(postgres_url_from_env(env_file), connect_args={"connect_timeout": 10})
    try:
        raw = pd.read_sql_query(
            text(BTC_QUERY),
            engine,
            params={"start": BTC_START.to_pydatetime(), "end": BTC_END.to_pydatetime()},
        )
    finally:
        engine.dispose()
    return normalize_btc_bars(raw)


def causal_var_forecasts(frame: pd.DataFrame, trailing: int = 252) -> pd.Series:
    required = frame[["btc_return", "uup_return"]].to_numpy(dtype=float)
    forecasts = pd.Series(np.nan, index=frame.index, dtype=float)
    for index in range(trailing + 1, len(frame)):
        dependent = required[index - trailing:index]
        lagged = required[index - trailing - 1:index - 1]
        if not np.isfinite(dependent).all() or not np.isfinite(lagged).all():
            continue
        design = np.column_stack([np.ones(trailing), lagged])
        coefficients, _, rank, _ = np.linalg.lstsq(design, dependent, rcond=None)
        current = np.array([1.0, required[index, 0], required[index, 1]])
        if rank == 3 and np.isfinite(current).all():
            forecasts.iloc[index] = float(current @ coefficients[:, 0])
    return forecasts


def build_features(sessions: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    frame = sessions.copy().sort_values("session_date").reset_index(drop=True)
    indexed = bars.set_index("ts") if "ts" in bars.columns else bars.copy()
    if not isinstance(indexed.index, pd.DatetimeIndex) or indexed.index.tz is None:
        raise RuntimeError("HVDXYFT BTC index must be timezone-aware")
    btc_closes: list[float] = []
    variations: list[float] = []
    for cash_close in frame.cash_close_time:
        close = pd.Timestamp(cash_close)
        minute = close - pd.Timedelta(minutes=1)
        btc_closes.append(float(indexed.at[minute, "close"]) if minute in indexed.index else np.nan)
        window = indexed.loc[(indexed.index >= close - pd.Timedelta(hours=24)) & (indexed.index < close)]
        if len(window) != 1440:
            variations.append(np.nan)
        else:
            component = np.log(window.close.to_numpy(dtype=float) / window.open.to_numpy(dtype=float))
            variations.append(float(np.sqrt(np.square(component).sum())))
    frame["btc_realized_variation"] = variations
    frame["btc_close"] = btc_closes
    frame["btc_return"] = np.log(frame.btc_close / frame.btc_close.shift(1))
    frame["uup_return"] = np.log(frame.uup_adjusted_close / frame.uup_adjusted_close.shift(1))
    frame["btc_forecast"] = causal_var_forecasts(frame)
    frame["forecast_magnitude_rank"] = strict_prior_midrank(
        frame.btc_forecast.abs(), lookback=252, minimum=126
    )
    frame["uup_magnitude_rank"] = strict_prior_midrank(
        frame.uup_return.abs(), lookback=252, minimum=126
    )
    frame["btc_variation_rank"] = strict_prior_midrank(
        frame.btc_realized_variation, lookback=270, minimum=180
    )
    return frame


def _signal(frame: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series]:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    score = frame.btc_forecast.copy()
    rank = frame.forecast_magnitude_rank.copy()
    if control == "one_session_stale_forecast":
        score, rank = score.shift(1), rank.shift(1)
    elif control == "uup_sign_only":
        score = frame.uup_return
        rank = frame.uup_magnitude_rank
    elif control == "btc_ar_only":
        reduced = frame[["btc_return", "uup_return"]].copy()
        reduced["uup_return"] = 0.0
        score = causal_var_forecasts(reduced)
        rank = strict_prior_midrank(score.abs(), lookback=252, minimum=126)
    active = score.notna() & score.ne(0.0) & rank.ge(0.60)
    if control != "no_btc_volatility_gate":
        active &= frame.btc_variation_rank.ge(0.65)
    side = np.sign(score).astype("Int64")
    if control == "direction_flip":
        side = -side
    elif control == "same_clock_forced_long":
        side = pd.Series(1, index=frame.index, dtype="Int64")
    active &= side.ne(0) & side.notna()
    return active.fillna(False), side


def build_clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, sides = _signal(frame, control)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in frame.index[active]:
        close = pd.Timestamp(frame.at[index, "cash_close_time"])
        feature_available = close + pd.Timedelta(minutes=5)
        entry = feature_available + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=24)
        if next_allowed is not None and entry < next_allowed:
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
        next_allowed = exit_time
        rows.append(
            {
                "candidate": "HVDXYFT-24",
                "control": control,
                "split": split,
                "session_date": frame.at[index, "session_date"],
                "cash_close_time": close,
                "feature_available_time": feature_available,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": int(sides.at[index]),
                "btc_forecast": float(frame.at[index, "btc_forecast"]),
                "forecast_magnitude_rank": float(frame.at[index, "forecast_magnitude_rank"]),
                "uup_return": float(frame.at[index, "uup_return"]),
                "btc_return": float(frame.at[index, "btc_return"]),
                "btc_realized_variation": float(frame.at[index, "btc_realized_variation"]),
                "btc_variation_rank": float(frame.at[index, "btc_variation_rank"]),
            }
        )
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def support_stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    selected = clock.loc[clock.split.eq(split)]
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
    months = pd.to_datetime(selected.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {
        "events": int(len(selected)),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": int(months.max()) / len(selected),
    }


def run(env_file: str = ENV_FILE) -> dict[str, Any]:
    if sha256(prereg.DEFAULT_OUTPUT) != PREREG_SHA256:
        raise RuntimeError("HVDXYFT preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)

    raw_payloads: dict[str, bytes] = {}
    equity_frames: dict[str, pd.DataFrame] = {}
    equity_metadata: dict[str, dict[str, Any]] = {}
    for symbol in SYMBOLS:
        raw_payloads[symbol], equity_frames[symbol], equity_metadata[symbol] = download_equity(symbol)
    sessions = build_equity_panel(equity_frames)
    bars = load_btc_bars(env_file)
    features = build_features(sessions, bars)
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    for symbol in SYMBOLS:
        RAW_PATHS[symbol].write_bytes(raw_payloads[symbol])
    write_gzip_csv(sessions, EQUITY_PANEL)
    write_gzip_csv(bars, BTC_SOURCE)
    write_gzip_csv(features, FEATURE_PANEL)
    write_gzip_csv(primary, CLOCK)
    for name, control_clock in controls.items():
        write_gzip_csv(control_clock, CONTROL_DIR / f"{name}.csv.gz")

    source_core = {
        "protocol_version": "hvdxyft_24_sources_v1",
        "preregistration_sha256": PREREG_SHA256,
        "outcomes_opened": False,
        "postentry_return_pnl_execution_price_opened": False,
        "equities": {
            symbol: {
                "path": str(RAW_PATHS[symbol]),
                "sha256": sha256(RAW_PATHS[symbol]),
                "provider": "Yahoo chart API current adjusted history",
                "metadata": equity_metadata[symbol],
            }
            for symbol in SYMBOLS
        },
        "equity_panel": {
            "path": str(EQUITY_PANEL),
            "sha256": sha256(EQUITY_PANEL),
            "rows": int(len(sessions)),
            "closed_dates": sorted(CLOSED_DATES),
            "early_close_dates": sorted(EARLY_CLOSES),
        },
        "btc": {
            "path": str(BTC_SOURCE),
            "sha256": sha256(BTC_SOURCE),
            "query": BTC_QUERY,
            "columns": ["ts", "open", "close"],
            "rows": int(len(bars)),
            "read_only": True,
        },
        "features": {
            "path": str(FEATURE_PANEL),
            "sha256": sha256(FEATURE_PANEL),
            "rows": int(len(features)),
        },
        "builder": {"path": str(BUILDER), "sha256": sha256(BUILDER)},
        "no_imputation": True,
    }
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    SOURCE_MANIFEST.write_text(
        json.dumps(source_manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )

    support = {name: support_stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "hvdxyft_24_source_support_v1",
        "policy_id": "HVDXYFT-24",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": PREREG_SHA256,
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifest": {
            "path": str(SOURCE_MANIFEST),
            "sha256": sha256(SOURCE_MANIFEST),
            "manifest_hash": source_manifest["manifest_hash"],
        },
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": int(len(primary))},
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": int(len(control_clock)),
                "promotion_authorized": False,
            }
            for name, control_clock in controls.items()
        },
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    report = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=ENV_FILE)
    args = parser.parse_args()
    report = run(args.env_file)
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))


if __name__ == "__main__":
    main()
