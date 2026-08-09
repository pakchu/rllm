"""Materialize outcome-blind source support for frozen CMMTCR-24."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from training import preregister_cross_market_multi_horizon_trend_concordance_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path("training/build_cross_market_multi_horizon_trend_concordance_relay_support.py")
PREREG_SHA = "11f244c7dd4e0791c81eedb6b37eda41a7c82319e2064edec4ccf67674b8e4d3"
SOURCE_DIR = Path("data/cross_market_multi_horizon_trend_concordance_relay_sources_2023_2026")
FEATURE_PANEL = SOURCE_DIR / "cross_market_multi_horizon_trend_concordance_relay_preentry_features.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/cross_market_multi_horizon_trend_concordance_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/cross_market_multi_horizon_trend_concordance_relay_controls_2023_2026")
RESULT = Path("results/cross_market_multi_horizon_trend_concordance_relay_support_2026-08-09.json")
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_volatility_gate", "btc_horizons_only", "btc_6h_eth_6h_only", "one_day_stale_geometry", "direction_flip", "forced_long")
SYMBOLS = ("BTCUSDT", "ETHUSDT")
ECONOMIC_OUTCOMES_AUTHORIZED = False
CLOCK_COLUMNS = ("candidate", "control", "split", "session_date", "decision_time", "feature_available_time", "entry_time", "exit_time", "side", "btc_6h_return", "btc_24h_return", "eth_6h_return", "btc_realized_variation", "variation_rank")

def query() -> str:
    return (
        "SELECT ts,open,high,low,close FROM bars_binance "
        "WHERE symbol=:symbol AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"
    )


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def strict_prior_midrank(values: pd.Series, lookback: int = 270, minimum: int = 180) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if np.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior)
            output.at[index] = (np.count_nonzero(array < current) + 0.5 * np.count_nonzero(array == current)) / len(array)
        if np.isfinite(current):
            history.append(float(current))
    return output


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_bars(symbol: str) -> pd.DataFrame:
    from sqlalchemy import text

    engine = postgres_engine()
    try:
        frame = pd.read_sql_query(text(query()), engine, params={"symbol": symbol, "start": START.to_pydatetime(), "end": END.to_pydatetime()})
    finally:
        engine.dispose()
    if frame.columns.tolist() != ["ts", "open", "high", "low", "close"]:
        raise RuntimeError(f"CMMTCR {symbol} schema drift")
    frame["ts"] = pd.to_datetime(frame.ts, utc=True, errors="raise")
    frame = frame.sort_values("ts").reset_index(drop=True)
    if frame.empty or frame.ts.duplicated().any():
        raise RuntimeError(f"CMMTCR {symbol} empty or duplicated timestamps")
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    prices = frame[["open", "high", "low", "close"]]
    coherent = frame.high.ge(frame[["open", "close", "low"]].max(axis=1)) & frame.low.le(frame[["open", "close", "high"]].min(axis=1))
    if not np.isfinite(prices).all(axis=None) or not prices.gt(0).all(axis=None) or not coherent.all():
        raise RuntimeError(f"CMMTCR {symbol} contains invalid or incoherent OHLC")
    return frame.set_index("ts")


def _exact_window(bars: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    expected = pd.date_range(start, end, freq="1min", inclusive="left")
    window = bars.loc[(bars.index >= start) & (bars.index < end)]
    return window if len(window) == len(expected) and window.index.equals(expected) else None


def build_features(bars: dict[str, pd.DataFrame], lookback: int = 90, minimum: int = 60) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for decision in pd.date_range(START + pd.Timedelta(hours=2), END, freq="1D", inclusive="left"):
        btc24 = _exact_window(bars["BTCUSDT"], decision - pd.Timedelta(hours=24), decision)
        btc6 = _exact_window(bars["BTCUSDT"], decision - pd.Timedelta(hours=6), decision)
        eth6 = _exact_window(bars["ETHUSDT"], decision - pd.Timedelta(hours=6), decision)
        if btc24 is None or btc6 is None or eth6 is None: continue
        btc24_return = float(np.log(btc24.close.iloc[-1] / btc24.open.iloc[0]))
        btc6_return = float(np.log(btc6.close.iloc[-1] / btc6.open.iloc[0]))
        eth6_return = float(np.log(eth6.close.iloc[-1] / eth6.open.iloc[0]))
        variation = float(np.sqrt(np.square(np.log(btc24.close.to_numpy(float) / btc24.open.to_numpy(float))).sum()))
        signs = np.sign([btc6_return, btc24_return, eth6_return])
        concordant = bool(np.all(signs == signs[0]) and signs[0] != 0)
        rows.append({"session_date": decision.date().isoformat(), "decision_time": decision, "btc_6h_return": btc6_return, "btc_24h_return": btc24_return, "eth_6h_return": eth6_return, "trend_sign": int(signs[0]) if concordant else 0, "concordant": concordant, "btc_realized_variation": variation})
    frame = pd.DataFrame(rows)
    frame["variation_rank"] = strict_prior_midrank(frame.btc_realized_variation, lookback, minimum)
    return frame


def conditions(frame: pd.DataFrame, control: str = "primary") -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    if control not in ("primary", *CONTROLS): raise ValueError(control)
    used = frame.shift(1) if control == "one_day_stale_geometry" else frame
    btc6, btc24, eth6 = used.btc_6h_return, used.btc_24h_return, used.eth_6h_return
    if control == "btc_horizons_only": concordant=np.sign(btc6).eq(np.sign(btc24))&btc6.ne(0)&btc24.ne(0);side=np.sign(btc6)
    elif control == "btc_6h_eth_6h_only": concordant=np.sign(btc6).eq(np.sign(eth6))&btc6.ne(0)&eth6.ne(0);side=np.sign(btc6)
    else: concordant=used.concordant.fillna(False).astype(bool);side=used.trend_sign
    volatility = pd.Series(True,index=frame.index) if control=="no_volatility_gate" else used.variation_rank.ge(.65)
    active=concordant&volatility&np.isfinite(side)&pd.Series(side,index=frame.index).ne(0)
    side=pd.Series(side,index=frame.index).fillna(0).astype(int)
    if control=="direction_flip":side=-side
    if control=="forced_long":side=pd.Series(1,index=frame.index)
    return active,side,used


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, sides, used = conditions(features, control)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in features.index[active]:
        decision = pd.Timestamp(features.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=24)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        source = used.loc[index]
        next_allowed = exit_time
        rows.append({
            "candidate": "CMMTCR-24", "control": control, "split": split,
            "session_date": source.session_date, "decision_time": decision,
            "feature_available_time": decision, "entry_time": entry, "exit_time": exit_time,
            "side": int(sides.at[index]), "btc_6h_return": float(source.btc_6h_return), "btc_24h_return": float(source.btc_24h_return), "eth_6h_return": float(source.eth_6h_return), "btc_realized_variation": float(source.btc_realized_variation), "variation_rank": float(source.variation_rank),
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def split_stats(clock: pd.DataFrame, split: str) -> dict[str, int | float]:
    selected = clock[clock.split.eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(selected.side.eq(1).sum()), int(selected.side.eq(-1).sum())
    entries = pd.to_datetime(selected.entry_time, utc=True)
    return {"events": len(selected), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(selected), "max_month_share": int(entries.dt.strftime("%Y-%m").value_counts().max()) / len(selected)}


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("CMMTCR preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    registration_core = {key: value for key, value in registration.items() if key != "manifest_hash"}
    if registration.get("manifest_hash") != prereg.canonical_hash(registration_core):
        raise RuntimeError("CMMTCR preregistration manifest drift")
    bars = {symbol: load_bars(symbol) for symbol in SYMBOLS}
    features = build_features(bars)
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(features, FEATURE_PANEL)
    _write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items():
        _write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {
        "protocol_version": "cmmtcr_8_sources_v1",
        "query": query(), "table": "bars_binance", "symbols": list(SYMBOLS),
        "window": [START.isoformat(), END.isoformat()],
        "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
        "features": {"path": str(FEATURE_PANEL), "sha256": sha(FEATURE_PANEL), "rows": len(features)},
        "candidate_outcomes_opened": False, "gross9_rows_opened": False, "no_imputation": True,
    }
    manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    SOURCE_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    support = {name: split_stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "cmmtcr_8_source_support_v1", "policy_id": "CMMTCR-24",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": ECONOMIC_OUTCOMES_AUTHORIZED,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    output = run()
    print(json.dumps({"passed": output["support_passed"], "support": output["support"]}, indent=2))
