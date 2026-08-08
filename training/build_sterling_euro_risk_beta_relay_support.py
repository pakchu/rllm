"""Materialize outcome-blind source support for frozen SERBR-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_sterling_euro_risk_beta_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path("training/build_sterling_euro_risk_beta_relay_support.py")
PREREG_SHA = "2ecf95b99acc2e62b96ea717e373eabaa48781e1017803f19e3ba2a2da849550"
SOURCE_DIR = Path("data/sterling_euro_risk_beta_relay_sources_2023_2026")
FEATURES = SOURCE_DIR / "serbr_preentry_features.csv.gz"
MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/sterling_euro_risk_beta_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/sterling_euro_risk_beta_relay_controls_2023_2026")
RESULT = Path("results/sterling_euro_risk_beta_relay_support_2026-08-09.json")
BTC_START = pd.Timestamp("2022-12-29T00:00:00Z")
FX_START = pd.Timestamp("2023-01-01T00:00:00Z")
SOURCE_END = pd.Timestamp("2026-08-01T00:00:00Z")
LONDON = ZoneInfo("Europe/London")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_volatility_gate",
    "common_dollar_basket",
    "one_session_stale_relative_return",
    "direction_flip",
)
COLUMNS = (
    "candidate", "control", "split", "session_date", "decision_time",
    "feature_available_time", "entry_time", "exit_time", "side",
    "gbpusd_return", "eurusd_return", "relative_return",
    "btc_realized_variation", "btc_variation_rank",
)
BTC_QUERY = (
    "SELECT ts,open,close FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' "
    "AND ts>=:start AND ts<:end ORDER BY ts"
)
FX_QUERY = (
    "SELECT symbol,ts,open,close FROM bars_polygon WHERE symbol IN ('GBPUSD','EURUSD') "
    "AND interval='1m' AND ts>=:start AND ts<:end ORDER BY symbol,ts"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def strict_prior_rank(values: pd.Series, lookback: int = 252, minimum: int = 126) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if np.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior)
            output.at[index] = (
                np.count_nonzero(array < current) + 0.5 * np.count_nonzero(array == current)
            ) / len(array)
        if np.isfinite(current):
            history.append(float(current))
    return output


def engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_sources() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    from sqlalchemy import text

    connection_engine = engine()
    try:
        btc = pd.read_sql_query(
            text(BTC_QUERY), connection_engine,
            params={"start": BTC_START.to_pydatetime(), "end": SOURCE_END.to_pydatetime()},
        )
        fx = pd.read_sql_query(
            text(FX_QUERY), connection_engine,
            params={"start": FX_START.to_pydatetime(), "end": SOURCE_END.to_pydatetime()},
        )
    finally:
        connection_engine.dispose()
    if btc.columns.tolist() != ["ts", "open", "close"]:
        raise RuntimeError("SERBR BTC schema drift")
    if fx.columns.tolist() != ["symbol", "ts", "open", "close"]:
        raise RuntimeError("SERBR FX schema drift")
    frames: dict[str, pd.DataFrame] = {}
    for label, frame in [("BTCUSDT", btc), *[(symbol, fx[fx.symbol.eq(symbol)].drop(columns="symbol")) for symbol in ("GBPUSD", "EURUSD")]]:
        frame = frame.copy()
        frame["ts"] = pd.to_datetime(frame.ts, utc=True, errors="raise")
        frame.sort_values("ts", inplace=True)
        frame.reset_index(drop=True, inplace=True)
        if frame.ts.duplicated().any():
            raise RuntimeError(f"SERBR {label} duplicate timestamp")
        for column in ("open", "close"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[["open", "close"]]).all(axis=None) or not frame[["open", "close"]].gt(0).all(axis=None):
            raise RuntimeError(f"SERBR {label} invalid price")
        frames[label] = frame.set_index("ts")
    expected = pd.date_range(BTC_START, SOURCE_END, freq="1min", inclusive="left")
    btc_frame = frames.pop("BTCUSDT")
    if len(btc_frame) != len(expected) or not btc_frame.index.equals(expected):
        raise RuntimeError("SERBR BTC source not exact 1m grid")
    return btc_frame, frames


def build_features(btc: pd.DataFrame, fx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for day in pd.date_range("2023-01-02", "2026-07-31", freq="B"):
        local_day = pd.Timestamp(day).tz_localize(LONDON)
        start = (local_day + pd.Timedelta(hours=8)).tz_convert("UTC")
        decision = (local_day + pd.Timedelta(hours=16)).tz_convert("UTC")
        expected = pd.date_range(start, decision, freq="1min", inclusive="left")
        sessions = {symbol: frame.reindex(expected) for symbol, frame in fx.items()}
        if any(session[["open", "close"]].isna().any(axis=None) for session in sessions.values()):
            continue
        btc_window = btc.loc[decision - pd.Timedelta(hours=24): decision - pd.Timedelta(minutes=1)]
        if len(btc_window) != 1440:
            continue
        returns = {
            symbol: float(np.log(session.close.iloc[-1] / session.open.iloc[0]))
            for symbol, session in sessions.items()
        }
        variation = float(
            np.sqrt(np.square(np.log(btc_window.close.to_numpy() / btc_window.open.to_numpy())).sum())
        )
        rows.append({
            "session_date": day,
            "decision_time": decision,
            "gbpusd_return": returns["GBPUSD"],
            "eurusd_return": returns["EURUSD"],
            "relative_return": returns["GBPUSD"] - returns["EURUSD"],
            "btc_realized_variation": variation,
        })
    features = pd.DataFrame(rows)
    features["btc_variation_rank"] = strict_prior_rank(features.btc_realized_variation)
    return features


def signal(features: pd.DataFrame, control: str) -> pd.Series:
    relative = features.relative_return
    side = np.sign(relative).astype("Int64").fillna(0).astype(int)
    eligible = relative.ne(0) & features.btc_variation_rank.ge(0.65)
    if control == "no_volatility_gate":
        eligible = relative.ne(0)
    elif control == "common_dollar_basket":
        common = (features.gbpusd_return + features.eurusd_return) / 2
        side = np.sign(common).astype("Int64").fillna(0).astype(int)
        eligible = common.ne(0) & features.btc_variation_rank.ge(0.65)
    elif control == "one_session_stale_relative_return":
        relative = relative.shift(1)
        side = np.sign(relative).astype("Int64").fillna(0).astype(int)
        eligible = relative.ne(0) & features.btc_variation_rank.ge(0.65)
    side = side.where(eligible, 0)
    return -side if control == "direction_flip" else side


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    sides = signal(features, control)
    rows = []
    next_allowed = None
    for index in features.index[sides.ne(0)]:
        decision = pd.Timestamp(features.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=12)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        next_allowed = exit_time
        rows.append({
            "candidate": "SERBR-12", "control": control, "split": split,
            "session_date": pd.Timestamp(features.at[index, "session_date"]),
            "decision_time": decision, "feature_available_time": decision,
            "entry_time": entry, "exit_time": exit_time, "side": int(sides.at[index]),
            "gbpusd_return": float(features.at[index, "gbpusd_return"]),
            "eurusd_return": float(features.at[index, "eurusd_return"]),
            "relative_return": float(features.at[index, "relative_return"]),
            "btc_realized_variation": float(features.at[index, "btc_realized_variation"]),
            "btc_variation_rank": float(features.at[index, "btc_variation_rank"]),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = clock[clock.split.eq(split)].copy()
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    entries = pd.to_datetime(selected.entry_time, utc=True)
    longs = int(selected.side.eq(1).sum())
    shorts = int(selected.side.eq(-1).sum())
    return {
        "events": len(selected), "longs": longs, "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": int(entries.dt.strftime("%Y-%m").value_counts().max()) / len(selected),
    }


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("SERBR prereg drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    btc, fx = load_sources()
    features = build_features(btc, fx)
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(features, FEATURES)
    _write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items():
        _write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {
        "protocol_version": "serbr_12_sources_v1",
        "queries": {"btc": BTC_QUERY, "fx": FX_QUERY},
        "windows": {"btc": [BTC_START.isoformat(), SOURCE_END.isoformat()], "fx": [FX_START.isoformat(), SOURCE_END.isoformat()]},
        "rows": {"btc": len(btc), "gbpusd": len(fx["GBPUSD"]), "eurusd": len(fx["EURUSD"]), "features": len(features)},
        "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
        "feature_output": {"path": str(FEATURES), "sha256": sha(FEATURES)},
        "candidate_outcomes_opened": False, "no_imputation": True,
    }
    source_manifest = {**source_core, "manifest_hash": chash(source_core)}
    MANIFEST.write_text(json.dumps(source_manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    support = {name: stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.2
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "serbr_12_source_support_v1", "policy_id": "SERBR-12",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(MANIFEST), "sha256": sha(MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": chash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    outcome = run()
    print(json.dumps({"passed": outcome["support_passed"], "support": outcome["support"]}, indent=2))
