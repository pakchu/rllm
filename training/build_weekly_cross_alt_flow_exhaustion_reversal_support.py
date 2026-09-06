"""Materialize outcome-blind source support for frozen WCAFER-24."""
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

from training import preregister_weekly_cross_alt_flow_exhaustion_reversal as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path("training/build_weekly_cross_alt_flow_exhaustion_reversal_support.py")
PREREG_SHA = "929f0bc729ad56eab4b63af9a495d270c1d4c315a517bd155d0e14f68966698b"
SOURCE_DIR = Path("data/weekly_cross_alt_flow_exhaustion_reversal_sources_2023_2026")
FEATURE_PANEL = SOURCE_DIR / "weekly_cross_alt_flow_exhaustion_reversal_preentry_features.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/weekly_cross_alt_flow_exhaustion_reversal_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/weekly_cross_alt_flow_exhaustion_reversal_controls_2023_2026")
RESULT = Path("results/weekly_cross_alt_flow_exhaustion_reversal_support_2026-08-09.json")
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_volatility_gate", "three_of_six_breadth", "btc_weekly_confirmation",
    "one_week_stale_acceleration", "direction_flip",
)
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT")
ALT_SYMBOLS = SYMBOLS[1:]
ECONOMIC_OUTCOMES_AUTHORIZED = False
CLOCK_COLUMNS = (
    "candidate", "control", "split", "session_date", "decision_time",
    "feature_available_time", "entry_time", "exit_time", "side",
    "btc_flow_acceleration", "confirming_alts", "alt_majority_sign",
    "btc_realized_variation", "variation_rank",
)


def query() -> str:
    return """
SELECT date_trunc('week',ts-interval '4 days')+interval '4 days' AS week_start,
       count(*) AS row_count,
       count(DISTINCT ts) AS distinct_ts,
       min(ts) AS min_ts,
       max(ts) AS max_ts,
       bool_and(ts=date_trunc('minute',ts)) AS minute_aligned,
       bool_and(open IS NOT NULL AND high IS NOT NULL AND low IS NOT NULL AND close IS NOT NULL
                AND open>0 AND high>0 AND low>0 AND close>0
                AND high>=greatest(open,low,close)
                AND low<=least(open,high,close)) AS ohlc_valid,
       bool_and(quote_asset_volume IS NOT NULL AND taker_buy_quote IS NOT NULL
                AND quote_asset_volume>=0 AND taker_buy_quote>=0
                AND taker_buy_quote<=quote_asset_volume) AS flow_valid,
       sum(2*taker_buy_quote-quote_asset_volume)
         / nullif(sum(quote_asset_volume),0) AS flow_imbalance,
       sqrt(sum(power(ln(close/open),2))) AS realized_variation
FROM bars_binance
WHERE symbol=:symbol AND interval='1m' AND ts>=:start AND ts<:end
GROUP BY 1 ORDER BY 1
""".strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def strict_prior_midrank(values: pd.Series, lookback: int = 52, minimum: int = 26) -> pd.Series:
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


def load_weekly(symbol: str) -> pd.DataFrame:
    from sqlalchemy import text

    engine = postgres_engine()
    try:
        frame = pd.read_sql_query(text(query()), engine, params={"symbol": symbol, "start": START.to_pydatetime(), "end": END.to_pydatetime()})
    finally:
        engine.dispose()
    expected_columns = ["week_start", "row_count", "distinct_ts", "min_ts", "max_ts", "minute_aligned", "ohlc_valid", "flow_valid", "flow_imbalance", "realized_variation"]
    if frame.columns.tolist() != expected_columns:
        raise RuntimeError(f"WCAFER {symbol} schema drift")
    for column in ("week_start", "min_ts", "max_ts"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    frame["flow_imbalance"] = pd.to_numeric(frame.flow_imbalance, errors="coerce")
    frame["realized_variation"] = pd.to_numeric(frame.realized_variation, errors="coerce")
    exact = frame.row_count.eq(10080) & frame.distinct_ts.eq(10080) & frame.minute_aligned & frame.ohlc_valid & frame.flow_valid
    exact &= frame.min_ts.eq(frame.week_start) & frame.max_ts.eq(frame.week_start + pd.Timedelta(minutes=10079))
    exact &= np.isfinite(frame.flow_imbalance) & np.isfinite(frame.realized_variation)
    return frame.loc[exact, ["week_start", "flow_imbalance", "realized_variation"]].copy()


def build_features(weekly: dict[str, pd.DataFrame], lookback: int = 52, minimum: int = 26) -> pd.DataFrame:
    frame = weekly["BTCUSDT"].rename(columns={"flow_imbalance": "btcusdt_flow", "realized_variation": "btc_realized_variation"})
    for symbol in ALT_SYMBOLS:
        part = weekly[symbol][["week_start", "flow_imbalance"]].rename(columns={"flow_imbalance": f"{symbol.lower()}_flow"})
        frame = frame.merge(part, on="week_start", how="inner", validate="one_to_one")
    flow_columns = ["btcusdt_flow", *[f"{symbol.lower()}_flow" for symbol in ALT_SYMBOLS]]
    consecutive = frame.week_start.diff().eq(pd.Timedelta(days=7))
    for column in flow_columns:
        frame[f"{column}_acceleration"] = frame[column].diff().where(consecutive)
    alt_columns = [f"{symbol.lower()}_flow_acceleration" for symbol in ALT_SYMBOLS]
    signs = np.sign(frame[alt_columns])
    positive = signs.eq(1).sum(axis=1); negative = signs.eq(-1).sum(axis=1)
    frame["confirming_alts"] = np.maximum(positive, negative)
    frame["alt_majority_sign"] = np.select([positive.gt(negative), negative.gt(positive)], [1, -1], default=0)
    frame["btc_flow_acceleration"] = frame.pop("btcusdt_flow_acceleration")
    frame["session_date"] = frame.week_start.dt.date.astype(str)
    frame["decision_time"] = frame.week_start + pd.Timedelta(days=7)
    frame = frame.drop(columns=["week_start", *flow_columns])
    frame["variation_rank"] = strict_prior_midrank(frame.btc_realized_variation, lookback, minimum)
    return frame


def conditions(frame: pd.DataFrame, control: str = "primary") -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    used = frame.shift(1) if control == "one_week_stale_acceleration" else frame
    breadth_min = 3 if control == "three_of_six_breadth" else 4
    majority_sign = used.alt_majority_sign
    breadth = majority_sign.ne(0) & used.confirming_alts.ge(breadth_min)
    btc_confirmation = np.sign(used.btc_flow_acceleration).eq(majority_sign) if control == "btc_weekly_confirmation" else pd.Series(True, index=frame.index)
    volatility = pd.Series(True, index=frame.index) if control == "no_volatility_gate" else frame.variation_rank.ge(0.65)
    active = breadth & btc_confirmation & volatility
    side = -majority_sign.fillna(0).astype(int)
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
        exit_time = entry + pd.Timedelta(hours=24)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        source = used.loc[index]
        next_allowed = exit_time
        rows.append({
            "candidate": "WCAFER-24", "control": control, "split": split,
            "session_date": source.session_date, "decision_time": decision,
            "feature_available_time": decision, "entry_time": entry, "exit_time": exit_time,
            "side": int(sides.at[index]), "btc_flow_acceleration": float(source.btc_flow_acceleration),
            "confirming_alts": int(source.confirming_alts),
            "alt_majority_sign": float(source.alt_majority_sign),
            "btc_realized_variation": float(features.at[index, "btc_realized_variation"]),
            "variation_rank": float(features.at[index, "variation_rank"]),
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
        raise RuntimeError("WCAFER preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    weekly = {symbol: load_weekly(symbol) for symbol in SYMBOLS}
    features = build_features(weekly)
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
        "protocol_version": "wcafer_24_sources_v1",
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
        "protocol_version": "wcafer_24_source_support_v1", "policy_id": "WCAFER-24",
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
