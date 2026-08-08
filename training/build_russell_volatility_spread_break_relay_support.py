"""Materialize outcome-blind source support for frozen RVSBR-12."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any
from urllib.request import urlopen
from zoneinfo import ZoneInfo

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from training import preregister_russell_volatility_spread_break_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path("training/build_russell_volatility_spread_break_relay_support.py")
PREREG_SHA = "20dad547af804c1f2ff662f98ac1f446fa8e859eb22aa8934f458d35f8315368"
SOURCE_DIR = Path("data/russell_volatility_spread_break_relay_sources_2022_2026")
CBOE_PANEL = SOURCE_DIR / "cboe_rvx_vix_panel.csv.gz"
FEATURE_PANEL = SOURCE_DIR / "rvsbr_preentry_features.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/russell_volatility_spread_break_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/russell_volatility_spread_break_relay_controls_2023_2026")
RESULT = Path("results/russell_volatility_spread_break_relay_support_2026-08-09.json")
SOURCE_START = pd.Timestamp("2022-12-29T00:00:00Z")
SOURCE_END = pd.Timestamp("2026-08-01T00:00:00Z")
CBOE_START = pd.Timestamp("2022-01-01")
CBOE_END = pd.Timestamp("2026-08-01")
NY = ZoneInfo("America/New_York")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_btc_volatility_gate", "vix_change_only", "one_session_stale_spread", "direction_flip")
CLOCK_COLUMNS = (
    "candidate", "control", "split", "cboe_observation_date", "next_cboe_source_date",
    "decision_time", "feature_available_time", "entry_time", "exit_time", "side",
    "relative_volatility_shock", "shock_z", "btc_realized_variation", "btc_variation_rank",
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
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def causal_z(values: pd.Series, lookback: int = 252, minimum: int = 126) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    prior = numeric.shift(1).rolling(lookback, min_periods=minimum)
    mean = prior.mean()
    std = prior.std(ddof=1)
    return (numeric - mean) / std.where(std.gt(0))


def strict_prior_midrank(values: pd.Series, lookback: int = 252, minimum: int = 126) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    finite_history: list[float] = []
    for index, current in numeric.items():
        history = finite_history[-lookback:]
        if np.isfinite(current) and len(history) >= minimum:
            array = np.asarray(history, dtype=float)
            output.at[index] = (np.count_nonzero(array < current) + 0.5 * np.count_nonzero(array == current)) / len(array)
        if np.isfinite(current):
            finite_history.append(float(current))
    return output


def _download_cboe(symbol: str) -> pd.DataFrame:
    with urlopen(prereg.SOURCES[symbol], timeout=60) as response:
        payload = response.read()
    frame = pd.read_csv(io.BytesIO(payload), usecols=["DATE", "CLOSE"])
    frame["observation_date"] = pd.to_datetime(frame.DATE, format="%m/%d/%Y", errors="raise")
    frame[f"{symbol}_close"] = pd.to_numeric(frame.CLOSE, errors="coerce")
    frame = frame[(frame.observation_date >= CBOE_START) & (frame.observation_date < CBOE_END)]
    frame = frame[["observation_date", f"{symbol}_close"]].sort_values("observation_date").reset_index(drop=True)
    if frame.empty or frame.observation_date.duplicated().any() or not frame.observation_date.is_monotonic_increasing:
        raise RuntimeError(f"RVSBR {symbol} dates invalid")
    if not np.isfinite(frame[f"{symbol}_close"]).all() or not frame[f"{symbol}_close"].gt(0).all():
        raise RuntimeError(f"RVSBR {symbol} values invalid")
    return frame


def load_cboe() -> pd.DataFrame:
    rvx = _download_cboe("RVX")
    vix = _download_cboe("VIX")
    frame = rvx.merge(vix, on="observation_date", validate="one_to_one").sort_values("observation_date").reset_index(drop=True)
    frame["relative_volatility"] = np.log(frame.RVX_close / frame.VIX_close)
    frame["relative_volatility_shock"] = frame.relative_volatility.diff()
    frame["vix_change"] = np.log(frame.VIX_close).diff()
    frame["shock_z"] = causal_z(frame.relative_volatility_shock)
    frame["vix_change_z"] = causal_z(frame.vix_change)
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
        frame = pd.read_sql_query(text(QUERY), engine, params={"start": SOURCE_START.to_pydatetime(), "end": SOURCE_END.to_pydatetime()})
    finally:
        engine.dispose()
    required = ["ts", "open", "close"]
    if frame.columns.tolist() != required:
        raise RuntimeError(f"RVSBR BTC schema must be exactly {required}")
    frame.ts = pd.to_datetime(frame.ts, utc=True, errors="raise")
    frame = frame.sort_values("ts").reset_index(drop=True)
    expected = pd.date_range(SOURCE_START, SOURCE_END, freq="1min", inclusive="left")
    if len(frame) != len(expected) or not frame.ts.equals(pd.Series(expected, name="ts")):
        raise RuntimeError("RVSBR BTC source is not the exact requested 1m grid")
    for column in ("open", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame[["open", "close"]]).all(axis=None) or not frame[["open", "close"]].gt(0).all(axis=None):
        raise RuntimeError("RVSBR BTC source contains invalid prices")
    return frame.set_index("ts")


def build_features(cboe: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    variations: list[float] = []
    records: list[dict[str, Any]] = []
    for index in range(1, len(cboe)):
        next_date = cboe.at[index, "observation_date"]
        decision = pd.Timestamp(next_date).tz_localize(NY) + pd.Timedelta(hours=9, minutes=30)
        decision = decision.tz_convert("UTC")
        window = bars.loc[decision - pd.Timedelta(hours=24): decision - pd.Timedelta(minutes=1)]
        if len(window) != 1440:
            variation = np.nan
        else:
            variation = float(np.sqrt(np.square(np.log(window.close.to_numpy() / window.open.to_numpy())).sum()))
        variations.append(variation)
        records.append({
            "source_index": index - 1,
            "cboe_observation_date": cboe.at[index - 1, "observation_date"],
            "next_cboe_source_date": next_date,
            "decision_time": decision,
            "relative_volatility_shock": cboe.at[index - 1, "relative_volatility_shock"],
            "shock_z": cboe.at[index - 1, "shock_z"],
            "vix_change": cboe.at[index - 1, "vix_change"],
            "vix_change_z": cboe.at[index - 1, "vix_change_z"],
            "btc_realized_variation": variation,
        })
    frame = pd.DataFrame(records)
    frame["btc_variation_rank"] = strict_prior_midrank(pd.Series(variations), lookback=252, minimum=126)
    return frame


def signal(features: pd.DataFrame, control: str) -> pd.Series:
    shock = features.relative_volatility_shock
    eligible = shock.ne(0) & features.shock_z.abs().ge(1.0) & features.btc_variation_rank.ge(0.65)
    side = -np.sign(shock).astype("Int64").fillna(0).astype(int)
    if control == "no_btc_volatility_gate":
        eligible = shock.ne(0) & features.shock_z.abs().ge(1.0)
    elif control == "vix_change_only":
        eligible = features.vix_change.ne(0) & features.vix_change_z.abs().ge(1.0) & features.btc_variation_rank.ge(0.65)
        side = -np.sign(features.vix_change).astype("Int64").fillna(0).astype(int)
    elif control == "one_session_stale_spread":
        stale_shock, stale_z = shock.shift(1), features.shock_z.shift(1)
        eligible = stale_shock.ne(0) & stale_z.abs().ge(1.0) & features.btc_variation_rank.ge(0.65)
        side = -np.sign(stale_shock).astype("Int64").fillna(0).astype(int)
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
        entry, exit_time = decision + pd.Timedelta(minutes=5), decision + pd.Timedelta(hours=12, minutes=5)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        next_allowed = exit_time
        rows.append({
            "candidate": "RVSBR-12", "control": control, "split": split,
            "cboe_observation_date": features.at[index, "cboe_observation_date"].date().isoformat(),
            "next_cboe_source_date": features.at[index, "next_cboe_source_date"].date().isoformat(),
            "decision_time": decision, "feature_available_time": decision, "entry_time": entry, "exit_time": exit_time,
            "side": int(sides.at[index]), "relative_volatility_shock": float(features.at[index, "relative_volatility_shock"]),
            "shock_z": float(features.at[index, "shock_z"]), "btc_realized_variation": float(features.at[index, "btc_realized_variation"]),
            "btc_variation_rank": float(features.at[index, "btc_variation_rank"]),
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = clock[clock.split.eq(split)].copy()
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    entries = pd.to_datetime(subset.entry_time, utc=True)
    longs, shorts = int(subset.side.eq(1).sum()), int(subset.side.eq(-1).sum())
    return {"events": len(subset), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(subset), "max_month_share": int(entries.dt.strftime("%Y-%m").value_counts().max()) / len(subset)}


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("RVSBR preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    cboe, bars = load_cboe(), load_bars()
    features = build_features(cboe, bars)
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True); CLOCK.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(cboe, CBOE_PANEL); _write_gzip_csv(features, FEATURE_PANEL); _write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items():
        _write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {"protocol_version": "rvsbr_12_sources_v1", "cboe_urls": prereg.SOURCES, "cboe_window": [str(CBOE_START.date()), str(CBOE_END.date())], "btc_query": QUERY, "btc_window": [SOURCE_START.isoformat(), SOURCE_END.isoformat()], "btc_rows": len(bars), "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)}, "outputs": {"cboe": {"path": str(CBOE_PANEL), "sha256": sha(CBOE_PANEL), "rows": len(cboe)}, "features": {"path": str(FEATURE_PANEL), "sha256": sha(FEATURE_PANEL), "rows": len(features)}}, "candidate_outcomes_opened": False, "no_imputation": True}
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    SOURCE_MANIFEST.write_text(json.dumps(source_manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    support = {name: stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {"protocol_version": "rvsbr_12_source_support_v1", "policy_id": "RVSBR-12", "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]}, "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]}, "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False, "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)}, "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()}, "support": support, "support_checks": checks, "support_passed": passed, "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False, "decision": "pass_to_novelty" if passed else "terminal_source_support_reject"}
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    result = run()
    print(json.dumps({"passed": result["support_passed"], "support": result["support"]}, indent=2))
