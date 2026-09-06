"""Build source-only STCR-72 clocks from completed BTC bars."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_scheduled_trend_concordance_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = Path("/home/pakchu/rllm/.env")
EXTENSION_START = pd.Timestamp("2026-05-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
SOURCE_DIR = Path("data/scheduled_trend_concordance_relay_sources_2020_2026")
DAILY = SOURCE_DIR / "daily_close_states.csv.gz"
MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/scheduled_trend_concordance_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/scheduled_trend_concordance_relay_controls_2023_2026")
RESULT = Path("results/scheduled_trend_concordance_relay_support_2026-08-09.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), END),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("three_day_only", "fourteen_day_only", "direction_flip")
LIVE_QUERY = """
SELECT ts,open,high,low,close
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts
"""
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "return_3d", "return_14d", "rv20",
    "rv20_threshold", "rv20_q90_active",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def postgres_engine():
    from preprocessing.live_db_features import sqlalchemy_engine_from_env
    return sqlalchemy_engine_from_env(ENV_FILE)


def _five_minute_extension(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame.pop("ts"), utc=True)
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.date.duplicated().any():
        raise RuntimeError("STCR live source has duplicate timestamps")
    frame = frame.sort_values("date").set_index("date")
    expected = pd.date_range(EXTENSION_START, END, freq="1min", inclusive="left")
    if not frame.index.equals(expected):
        raise RuntimeError("STCR live source is not an exact minute grid")
    valid = (
        np.isfinite(frame).all(axis=1)
        & frame.gt(0).all(axis=1)
        & frame.high.ge(frame[["open", "close"]].max(axis=1))
        & frame.low.le(frame[["open", "close"]].min(axis=1))
        & frame.high.ge(frame.low)
    )
    if not bool(valid.all()):
        raise RuntimeError("STCR live OHLC validity drift")
    grouped = frame.resample("5min", origin="epoch", closed="left", label="left")
    bars = grouped.agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), source_rows=("open", "size"),
    ).reset_index()
    if not bars.source_rows.eq(5).all():
        raise RuntimeError("STCR live five-minute aggregation is incomplete")
    return bars.drop(columns="source_rows")


def load_market() -> tuple[pd.DataFrame, dict[str, Any]]:
    historical = pd.read_csv(prereg.MARKET, usecols=["date", "open", "high", "low", "close"])
    historical["date"] = pd.to_datetime(historical.date, utc=True)
    from sqlalchemy import text
    engine = postgres_engine()
    try:
        with engine.connect() as connection:
            raw = pd.read_sql_query(text(LIVE_QUERY), connection, params={"start": EXTENSION_START.to_pydatetime(), "end": END.to_pydatetime()})
    finally:
        engine.dispose()
    live = _five_minute_extension(raw)
    market = pd.concat([historical, live], ignore_index=True).sort_values("date").drop_duplicates("date", keep="last")
    market = market[market.date.lt(END)].reset_index(drop=True)
    if market.date.duplicated().any() or not market.date.diff().dropna().eq(pd.Timedelta(minutes=5)).all():
        raise RuntimeError("STCR combined source continuity drift")
    return market, {
        "historical_path": str(prereg.MARKET), "historical_sha256": sha(prereg.MARKET),
        "historical_rows": len(historical), "live_query_sha256": hashlib.sha256(LIVE_QUERY.encode()).hexdigest(),
        "live_one_minute_rows": len(raw), "live_five_minute_rows": len(live),
        "combined_rows": len(market), "first": str(market.date.iloc[0]), "last": str(market.date.iloc[-1]),
    }


def daily_states(market: pd.DataFrame) -> pd.DataFrame:
    frame = market.set_index("date").sort_index()
    rows = []
    for day, window in frame.groupby(frame.index.floor("D"), sort=True):
        expected = pd.date_range(day, day + pd.Timedelta(days=1), freq="5min", inclusive="left")
        window = window.reindex(expected)
        valid = bool(
            len(window) == 288
            and np.isfinite(window[["open", "high", "low", "close"]]).all(axis=1).all()
            and window[["open", "high", "low", "close"]].gt(0).all(axis=1).all()
            and window.high.ge(window[["open", "close"]].max(axis=1)).all()
            and window.low.le(window[["open", "close"]].min(axis=1)).all()
            and window.high.ge(window.low).all()
        )
        rows.append({"source_day": day, "decision_time": day + pd.Timedelta(days=1), "source_valid": valid, "day_close": float(window.close.iloc[-1]) if valid else np.nan})
    states = pd.DataFrame(rows)
    states["daily_return"] = np.log(states.day_close / states.day_close.shift(1))
    states["return_3d"] = np.log(states.day_close / states.day_close.shift(3))
    states["return_14d"] = np.log(states.day_close / states.day_close.shift(14))
    states["rv20"] = states.daily_return.rolling(20, min_periods=20).apply(lambda x: math.sqrt(365.0 * float(np.mean(np.square(x)))), raw=True)
    states["rv20_threshold"] = states.rv20.rolling(756, min_periods=756).quantile(0.90, interpolation="linear").shift(1)
    states["rv20_q90_active"] = states.rv20.ge(states.rv20_threshold)
    return states


def build_clock(states: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for _, item in states.iterrows():
        decision = pd.Timestamp(item.decision_time)
        if decision.weekday() not in (0, 3):
            continue
        r3, r14 = float(item.return_3d), float(item.return_14d)
        finite = bool(item.source_valid and np.isfinite([r3, r14]).all() and r3 != 0 and r14 != 0)
        if not finite:
            continue
        if control == "three_day_only":
            side = int(np.sign(r3))
        elif control == "fourteen_day_only":
            side = int(np.sign(r14))
        else:
            if np.sign(r3) != np.sign(r14):
                continue
            side = int(np.sign(r3))
        if control == "direction_flip":
            side = -side
        entry, exit_ = decision + pd.Timedelta(minutes=5), decision + pd.Timedelta(hours=72, minutes=5)
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_ <= end), None)
        if split is None:
            continue
        reserved_until = exit_
        rows.append({
            "candidate": prereg.POLICY_ID, "control": control, "split": split,
            "decision_time": decision, "feature_available_time": decision,
            "entry_time": entry, "exit_time": exit_, "side": side,
            "return_3d": r3, "return_14d": r14, "rv20": float(item.rv20),
            "rv20_threshold": float(item.rv20_threshold), "rv20_q90_active": bool(item.rv20_q90_active),
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    subset = clock[clock.split.eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0, "rv20_q90_events": 0}
    longs, shorts = int(subset.side.eq(1).sum()), int(subset.side.eq(-1).sum())
    months = pd.to_datetime(subset.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(subset), "longs": longs, "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(subset),
        "max_month_share": int(months.max()) / len(subset),
        "rv20_q90_events": int(subset.rv20_q90_active.sum()),
    }


def run() -> dict[str, Any]:
    market, source = load_market()
    states = daily_states(market)
    primary = build_clock(states)
    controls = {name: build_clock(states, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(states, DAILY)
    _write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items():
        _write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    source_core = {
        "protocol_version": "stcr_72_source_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": sha(prereg.DEFAULT_OUTPUT), "manifest_hash": registration["manifest_hash"]},
        "source": source, "daily_states": {"path": str(DAILY), "sha256": sha(DAILY), "rows": len(states)},
        "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
    }
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    MANIFEST.write_text(json.dumps(source_manifest, indent=2, allow_nan=False) + "\n")
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "stcr_72_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": source_core["preregistration"],
        "source_manifest": {"path": str(MANIFEST), "sha256": sha(MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False, "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    result = run()
    print(json.dumps({"passed": result["support_passed"], "support": result["support"]}, indent=2))
