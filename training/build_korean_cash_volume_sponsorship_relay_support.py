"""Materialize outcome-blind source support for frozen KCVSR-12."""
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

from training import preregister_korean_cash_volume_sponsorship_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path("training/build_korean_cash_volume_sponsorship_relay_support.py")
PREREG_SHA = "7f01e72a3aa092f51617aa5aaff97417c3773e8841d2d7058560eba0e52cd969"
SOURCE_DIR = Path("data/korean_cash_volume_sponsorship_relay_sources_2023_2026")
FEATURE_PANEL = SOURCE_DIR / "korean_cash_volume_sponsorship_relay_preentry_features.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/korean_cash_volume_sponsorship_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/korean_cash_volume_sponsorship_relay_controls_2023_2026")
RESULT = Path("results/korean_cash_volume_sponsorship_relay_support_2026-08-09.json")
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_volume_gate", "no_variation_gate", "one_day_stale_features", "direction_flip")
ECONOMIC_OUTCOMES_AUTHORIZED = False
CLOCK_COLUMNS = (
    "candidate", "control", "split", "session_date", "decision_time",
    "feature_available_time", "entry_time", "exit_time", "side",
    "upbit_base_volume", "upbit_volume_rank", "perp_return",
    "perp_realized_variation", "variation_rank",
)


def query(table: str) -> str:
    if table not in {"bars_upbit", "bars_binance"}:
        raise ValueError(table)
    return (
        f"SELECT ts,open,high,low,close,volume FROM {table} "
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


def load_bars(table: str, symbol: str) -> pd.DataFrame:
    from sqlalchemy import text

    engine = postgres_engine()
    try:
        frame = pd.read_sql_query(text(query(table)), engine, params={"symbol": symbol, "start": START.to_pydatetime(), "end": END.to_pydatetime()})
    finally:
        engine.dispose()
    if frame.columns.tolist() != ["ts", "open", "high", "low", "close", "volume"]:
        raise RuntimeError(f"KCVSR {table} schema drift")
    frame["ts"] = pd.to_datetime(frame.ts, utc=True, errors="raise")
    frame = frame.sort_values("ts").reset_index(drop=True)
    if frame.empty or frame.ts.duplicated().any():
        raise RuntimeError(f"KCVSR {table} empty or duplicated timestamps")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    prices = frame[["open", "high", "low", "close"]]
    coherent = frame.high.ge(frame[["open", "close", "low"]].max(axis=1)) & frame.low.le(frame[["open", "close", "high"]].min(axis=1))
    if not np.isfinite(prices).all(axis=None) or not prices.gt(0).all(axis=None) or not coherent.all():
        raise RuntimeError(f"KCVSR {table} contains invalid or incoherent OHLC")
    if not np.isfinite(frame.volume).all() or not frame.volume.ge(0).all():
        raise RuntimeError(f"KCVSR {table} contains invalid volume")
    return frame.set_index("ts")


def _exact_session(bars: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame | None:
    end = start + pd.Timedelta(hours=8)
    expected = pd.date_range(start, end, freq="1min", inclusive="left")
    window = bars.loc[(bars.index >= start) & (bars.index < end)]
    if len(window) != 480 or not window.index.equals(expected):
        return None
    return window


def build_features(upbit: pd.DataFrame, perp: pd.DataFrame, lookback: int = 270, minimum: int = 180) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for start in pd.date_range(START, END, freq="1D", inclusive="left"):
        u = _exact_session(upbit, start)
        p = _exact_session(perp, start)
        if u is None or p is None:
            continue
        upbit_volume = float(u.volume.sum())
        perp_return = float(np.log(p.close.iloc[-1] / p.open.iloc[0]))
        variation = float(np.sqrt(np.square(np.log(p.close.to_numpy() / p.open.to_numpy())).sum()))
        rows.append({
            "session_date": start.date().isoformat(),
            "decision_time": start + pd.Timedelta(hours=8),
            "upbit_base_volume": upbit_volume,
            "perp_return": perp_return,
            "perp_realized_variation": variation,
        })
    frame = pd.DataFrame(rows)
    frame["upbit_volume_rank"] = strict_prior_midrank(frame.upbit_base_volume, lookback, minimum)
    frame["variation_rank"] = strict_prior_midrank(frame.perp_realized_variation, lookback, minimum)
    return frame


def conditions(frame: pd.DataFrame, control: str = "primary") -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    used = frame.shift(1) if control == "one_day_stale_features" else frame
    valid = used.upbit_base_volume.gt(0) & used.perp_return.ne(0)
    volume = pd.Series(True, index=frame.index) if control == "no_volume_gate" else used.upbit_volume_rank.ge(0.75)
    volatility = pd.Series(True, index=frame.index) if control == "no_variation_gate" else used.variation_rank.ge(0.65)
    active = valid & volume & volatility
    side = np.sign(used.perp_return).fillna(0).astype(int)
    if control == "direction_flip":
        side = -side
    return active, side, used


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, sides, used = conditions(features, control)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in features.index[active]:
        decision = pd.Timestamp(features.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=12)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        source = used.loc[index]
        next_allowed = exit_time
        rows.append({
            "candidate": "KCVSR-12", "control": control, "split": split,
            "session_date": source.session_date, "decision_time": decision,
            "feature_available_time": decision, "entry_time": entry, "exit_time": exit_time,
            "side": int(sides.at[index]),
            "upbit_base_volume": float(source.upbit_base_volume),
            "upbit_volume_rank": float(source.upbit_volume_rank),
            "perp_return": float(source.perp_return),
            "perp_realized_variation": float(source.perp_realized_variation),
            "variation_rank": float(source.variation_rank),
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
        raise RuntimeError("KCVSR preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    upbit = load_bars("bars_upbit", "KRW-BTC")
    perp = load_bars("bars_binance", "BTCUSDT")
    features = build_features(upbit, perp)
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
        "protocol_version": "kcvsr_12_sources_v1",
        "queries": {"upbit": query("bars_upbit"), "perpetual": query("bars_binance")},
        "symbols": {"upbit": "KRW-BTC", "perpetual": "BTCUSDT"},
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
        "protocol_version": "kcvsr_12_source_support_v1", "policy_id": "KCVSR-12",
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
