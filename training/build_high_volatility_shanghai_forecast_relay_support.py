"""Materialize outcome-blind source support for frozen HVSFR-24."""
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

from training import preregister_high_volatility_shanghai_forecast_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path("training/build_high_volatility_shanghai_forecast_relay_support.py")
PREREG_SHA = "ee87c778c3d7f6176ca83e624a4970d5eeb7af2a80d456354da7fc6867549b68"
SOURCE_DIR = Path("data/high_volatility_shanghai_forecast_relay_sources_2021_2026")
RAW_SOURCE = SOURCE_DIR / "000001_ss_yahoo_chart.json"
SSE_PANEL = SOURCE_DIR / "shanghai_composite_sessions.csv.gz"
FEATURE_PANEL = SOURCE_DIR / "hvsfr_preentry_features.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_shanghai_forecast_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_shanghai_forecast_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_shanghai_forecast_relay_support_2026-08-11.json")
SOURCE_START = pd.Timestamp("2021-01-01T00:00:00Z")
SOURCE_END = pd.Timestamp("2026-08-02T00:00:00Z")
BTC_START = pd.Timestamp("2021-01-03T07:00:00Z")
BTC_END = pd.Timestamp("2026-08-01T07:01:00Z")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_btc_volatility_gate", "btc_ar_only", "sse_sign_only", "direction_flip")
CLOCK_COLUMNS = (
    "candidate", "control", "split", "sse_session_date", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "btc_forecast", "sse_return", "btc_return",
    "btc_realized_variation", "btc_variation_rank",
)
QUERY = """
SELECT ts,open,close
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts
"""


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


def causal_var_forecasts(frame: pd.DataFrame, trailing: int = 252) -> pd.Series:
    required = frame[["btc_return", "sse_return"]].to_numpy(dtype=float)
    forecasts = pd.Series(np.nan, index=frame.index, dtype=float)
    for index in range(trailing + 1, len(frame)):
        dependent = required[index - trailing:index]
        lagged = required[index - trailing - 1:index - 1]
        if not np.isfinite(dependent).all() or not np.isfinite(lagged).all():
            continue
        design = np.column_stack([np.ones(trailing), lagged])
        coefficients, _, rank, _ = np.linalg.lstsq(design, dependent, rcond=None)
        if rank != 3:
            continue
        current = np.array([1.0, required[index, 0], required[index, 1]])
        if np.isfinite(current).all():
            forecasts.iat[index] = float(current @ coefficients[:, 0])
    return forecasts


def _download_sse() -> tuple[bytes, pd.DataFrame]:
    query = urlencode({
        "period1": int(SOURCE_START.timestamp()),
        "period2": int(SOURCE_END.timestamp()),
        "interval": "1d",
        "events": "div,splits",
    })
    request = Request(f"{prereg.SSE_YAHOO_URL}?{query}", headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=60) as response:
        payload = response.read()
    parsed = json.loads(payload)
    result = parsed["chart"]["result"][0]
    if result["meta"]["symbol"] != "000001.SS" or result["meta"]["exchangeTimezoneName"] != "Asia/Shanghai":
        raise RuntimeError("HVSFR Yahoo metadata drift")
    quote = result["indicators"]["quote"][0]
    frame = pd.DataFrame({"timestamp": result["timestamp"], **quote})
    frame["sse_session_date"] = pd.to_datetime(frame.timestamp, unit="s", utc=True).dt.tz_convert("Asia/Shanghai").dt.date
    frame["close_time"] = pd.to_datetime(frame.sse_session_date.astype(str) + " 15:00:00").dt.tz_localize("Asia/Shanghai").dt.tz_convert("UTC")
    frame = frame[["sse_session_date", "close_time", "open", "high", "low", "close", "volume"]]
    frame = frame.sort_values("sse_session_date").reset_index(drop=True)
    if frame.empty or frame.sse_session_date.duplicated().any():
        raise RuntimeError("HVSFR Shanghai session dates invalid")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame[["open", "high", "low", "close", "volume"]]).all(axis=None):
        raise RuntimeError("HVSFR Shanghai values are incomplete")
    if not frame[["open", "high", "low", "close"]].gt(0).all(axis=None):
        raise RuntimeError("HVSFR Shanghai prices are nonpositive")
    return payload, frame


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
        raise RuntimeError("HVSFR BTC schema drift")
    frame.ts = pd.to_datetime(frame.ts, utc=True, errors="raise")
    frame = frame.sort_values("ts").reset_index(drop=True)
    expected = pd.date_range(BTC_START, BTC_END, freq="1min", inclusive="left")
    if len(frame) != len(expected) or not frame.ts.equals(pd.Series(expected, name="ts")):
        raise RuntimeError("HVSFR BTC source is not the exact requested 1m grid")
    for column in ("open", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame[["open", "close"]]).all(axis=None) or not frame[["open", "close"]].gt(0).all(axis=None):
        raise RuntimeError("HVSFR BTC prices invalid")
    return frame.set_index("ts")


def build_features(sse: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    previous_close: float | None = None
    previous_btc_open: float | None = None
    for row in sse.itertuples(index=False):
        close_time = pd.Timestamp(row.close_time)
        if close_time < BTC_START + pd.Timedelta(days=1) or close_time >= BTC_END:
            continue
        btc_open = float(bars.at[close_time, "open"])
        window = bars.loc[close_time - pd.Timedelta(hours=24):close_time - pd.Timedelta(minutes=1)]
        variation = float(np.sqrt(np.square(np.log(window.close.to_numpy() / window.open.to_numpy())).sum())) if len(window) == 1440 else np.nan
        records.append({
            "sse_session_date": row.sse_session_date,
            "close_time": close_time,
            "decision_time": close_time + pd.Timedelta(minutes=5),
            "sse_close": float(row.close),
            "btc_close_anchor": btc_open,
            "sse_return": np.log(float(row.close) / previous_close) if previous_close is not None else np.nan,
            "btc_return": np.log(btc_open / previous_btc_open) if previous_btc_open is not None else np.nan,
            "btc_realized_variation": variation,
        })
        previous_close, previous_btc_open = float(row.close), btc_open
    frame = pd.DataFrame(records)
    frame["btc_variation_rank"] = strict_prior_midrank(frame.btc_realized_variation)
    frame["btc_forecast"] = causal_var_forecasts(frame)
    return frame


def signal(features: pd.DataFrame, control: str) -> pd.Series:
    forecast = features.btc_forecast.copy()
    eligible = forecast.ne(0) & forecast.notna() & features.btc_variation_rank.ge(0.65)
    side = np.sign(forecast).astype("Int64").fillna(0).astype(int)
    if control == "no_btc_volatility_gate":
        eligible = forecast.ne(0) & forecast.notna()
    elif control == "btc_ar_only":
        reduced = features[["btc_return", "sse_return"]].copy()
        reduced["sse_return"] = 0.0
        forecast = causal_var_forecasts(reduced)
        eligible = forecast.ne(0) & forecast.notna() & features.btc_variation_rank.ge(0.65)
        side = np.sign(forecast).astype("Int64").fillna(0).astype(int)
    elif control == "sse_sign_only":
        eligible = features.sse_return.ne(0) & features.sse_return.notna() & features.btc_variation_rank.ge(0.65)
        side = np.sign(features.sse_return).astype("Int64").fillna(0).astype(int)
    side = side.where(eligible, 0)
    return -side if control == "direction_flip" else side


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    sides = signal(features, control)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in features.index[sides.ne(0)]:
        decision = pd.Timestamp(features.at[index, "decision_time"])
        entry, exit_time = decision + pd.Timedelta(minutes=5), decision + pd.Timedelta(hours=24, minutes=5)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        next_allowed = exit_time
        rows.append({
            "candidate": "HVSFR-24", "control": control, "split": split,
            "sse_session_date": str(features.at[index, "sse_session_date"]),
            "decision_time": decision, "feature_available_time": decision,
            "entry_time": entry, "exit_time": exit_time, "side": int(sides.at[index]),
            "btc_forecast": float(features.at[index, "btc_forecast"]),
            "sse_return": float(features.at[index, "sse_return"]),
            "btc_return": float(features.at[index, "btc_return"]),
            "btc_realized_variation": float(features.at[index, "btc_realized_variation"]),
            "btc_variation_rank": float(features.at[index, "btc_variation_rank"]),
        })
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
        raise RuntimeError("HVSFR preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    payload, sse = _download_sse()
    bars = load_bars()
    features = build_features(sse, bars)
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    RAW_SOURCE.write_bytes(payload)
    _write_gzip_csv(sse, SSE_PANEL)
    _write_gzip_csv(features, FEATURE_PANEL)
    _write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items():
        _write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {
        "protocol_version": "hvsfr_24_sources_v1",
        "sse_url": prereg.SSE_YAHOO_URL,
        "source_window": [SOURCE_START.isoformat(), SOURCE_END.isoformat()],
        "btc_query": QUERY,
        "btc_window": [BTC_START.isoformat(), BTC_END.isoformat()],
        "btc_rows": len(bars),
        "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
        "outputs": {
            "raw_sse": {"path": str(RAW_SOURCE), "sha256": sha(RAW_SOURCE)},
            "sse_panel": {"path": str(SSE_PANEL), "sha256": sha(SSE_PANEL), "rows": len(sse)},
            "features": {"path": str(FEATURE_PANEL), "sha256": sha(FEATURE_PANEL), "rows": len(features)},
        },
        "candidate_outcomes_opened": False,
        "no_imputation": True,
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
        "protocol_version": "hvsfr_24_source_support_v1", "policy_id": "HVSFR-24",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
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
