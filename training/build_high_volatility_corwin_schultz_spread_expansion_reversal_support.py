"""Materialize outcome-blind source support for frozen HVCSER-12."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from training import preregister_high_volatility_corwin_schultz_spread_expansion_reversal as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path("training/build_high_volatility_corwin_schultz_spread_expansion_reversal_support.py")
PREREG_SHA = "172bb81c4adc31ca9701d646144b803cd54a8ef7fba60e20ca4b81f25c1f6115"
START = pd.Timestamp("2023-03-30T21:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
SOURCE_DIR = Path("data/high_volatility_corwin_schultz_spread_expansion_reversal_sources_2023_2026")
FEATURES = SOURCE_DIR / "hvcser_preentry_features.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_corwin_schultz_spread_expansion_reversal_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_corwin_schultz_spread_expansion_reversal_controls_2023_2026")
RESULT = Path("results/high_volatility_corwin_schultz_spread_expansion_reversal_support_2026-08-10.json")
QUERY = "SELECT ts,open,high,low,close FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_spread_expansion_gate",
    "no_volatility_gate",
    "raw_positive_expansion",
    "one_day_stale_expansion",
    "direction_flip",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time", "entry_time", "exit_time", "side",
    "prior_high", "prior_low", "current_high", "current_low", "beta", "gamma", "alpha", "implied_spread",
    "spread_expansion", "spread_expansion_rank", "realized_variation", "variation_rank", "completed_return",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def strict_prior_midrank(values: pd.Series, lookback: int = 90, minimum: int = 60) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if math.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior)
            output.at[index] = (np.count_nonzero(array < current) + 0.5 * np.count_nonzero(array == current)) / len(array)
        if math.isfinite(current):
            history.append(float(current))
    return output


def corwin_schultz_spread(prior_high: float, prior_low: float, current_high: float, current_low: float) -> tuple[float, float, float, float]:
    prices = np.asarray([prior_high, prior_low, current_high, current_low], dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any() or prior_high < prior_low or current_high < current_low:
        return math.nan, math.nan, math.nan, math.nan
    beta = math.log(prior_high / prior_low) ** 2 + math.log(current_high / current_low) ** 2
    gamma = math.log(max(prior_high, current_high) / min(prior_low, current_low)) ** 2
    denominator = 3 - 2 * math.sqrt(2)
    alpha = (math.sqrt(2 * beta) - math.sqrt(beta)) / denominator - math.sqrt(gamma / denominator)
    spread_alpha = max(alpha, 0.0)
    spread = 2 * (math.exp(spread_alpha) - 1) / (1 + math.exp(spread_alpha))
    return beta, gamma, alpha, spread


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_source() -> pd.DataFrame:
    from sqlalchemy import text

    database = postgres_engine()
    try:
        bars = pd.read_sql_query(text(QUERY), database, params={"start": START.to_pydatetime(), "end": END.to_pydatetime()})
    finally:
        database.dispose()
    return validate_source(bars)


def validate_source(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.columns.tolist() != ["ts", "open", "high", "low", "close"]:
        raise RuntimeError("HVCSER source schema drift")
    frame = bars.copy()
    frame["ts"] = pd.to_datetime(frame.ts, utc=True, errors="raise")
    frame = frame.sort_values("ts").reset_index(drop=True)
    expected = pd.date_range(START, END, freq="1min", inclusive="left")
    if len(frame) != len(expected) or not frame.ts.equals(pd.Series(expected, name="ts")):
        raise RuntimeError("HVCSER source is not the exact unique 1m grid")
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    prices = frame[["open", "high", "low", "close"]]
    coherent = (
        np.isfinite(prices).all(axis=1)
        & prices.gt(0).all(axis=1)
        & frame.high.ge(prices[["open", "close"]].max(axis=1))
        & frame.low.le(prices[["open", "close"]].min(axis=1))
        & frame.high.ge(frame.low)
    )
    if not coherent.all():
        raise RuntimeError("HVCSER source contains incoherent OHLC")
    return frame.set_index("ts")


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    frame = bars.copy()
    if "ts" in frame.columns:
        frame["ts"] = pd.to_datetime(frame.ts, utc=True)
        frame = frame.set_index("ts")
    frame.index = pd.to_datetime(frame.index, utc=True)
    minute_returns = np.log(pd.to_numeric(frame.close) / pd.to_numeric(frame.open))
    rows: list[dict[str, Any]] = []
    for decision in pd.date_range(START + pd.Timedelta(days=2), END, freq="24h", inclusive="left"):
        prior = frame.loc[decision - pd.Timedelta(hours=48):decision - pd.Timedelta(hours=24, minutes=1)]
        current = frame.loc[decision - pd.Timedelta(hours=24):decision - pd.Timedelta(minutes=1)]
        valid = len(prior) == 1440 and len(current) == 1440
        if valid:
            prior_high, prior_low = float(prior.high.max()), float(prior.low.min())
            current_high, current_low = float(current.high.max()), float(current.low.min())
            beta, gamma, alpha, spread = corwin_schultz_spread(prior_high, prior_low, current_high, current_low)
            current_minute_returns = minute_returns.reindex(current.index).to_numpy(float)
            variation = float(np.sqrt(np.square(current_minute_returns).sum()))
            completed_return = float(math.log(float(current.close.iloc[-1]) / float(current.open.iloc[0])))
            valid = np.isfinite([beta, gamma, alpha, spread, variation, completed_return]).all()
        if not valid:
            prior_high = prior_low = current_high = current_low = beta = gamma = alpha = spread = variation = completed_return = math.nan
        rows.append({
            "decision_time": decision, "source_valid": bool(valid), "prior_high": prior_high, "prior_low": prior_low,
            "current_high": current_high, "current_low": current_low, "beta": beta, "gamma": gamma, "alpha": alpha,
            "implied_spread": spread, "realized_variation": variation, "completed_return": completed_return,
        })
    features = pd.DataFrame(rows)
    previous_valid_spread = math.nan
    expansions: list[float] = []
    for row in features.itertuples():
        expansion = float(row.implied_spread - previous_valid_spread) if row.source_valid and math.isfinite(previous_valid_spread) else math.nan
        expansions.append(expansion)
        if row.source_valid:
            previous_valid_spread = float(row.implied_spread)
    features["spread_expansion"] = expansions
    features["spread_expansion_rank"] = strict_prior_midrank(features.spread_expansion)
    features["variation_rank"] = strict_prior_midrank(features.realized_variation.where(features.source_valid))
    return features


def conditions(features: pd.DataFrame, control: str = "primary") -> tuple[pd.Series, pd.Series]:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    expansion = features.spread_expansion
    expansion_rank = features.spread_expansion_rank
    if control == "one_day_stale_expansion":
        expansion = expansion.shift(1)
        expansion_rank = expansion_rank.shift(1)
    finite_expansion = np.isfinite(expansion)
    if control == "no_spread_expansion_gate":
        spread_gate = finite_expansion
    elif control == "raw_positive_expansion":
        spread_gate = finite_expansion & expansion.gt(0)
    else:
        spread_gate = finite_expansion & expansion.gt(0) & expansion_rank.ge(0.70)
    volatility_gate = pd.Series(True, index=features.index) if control == "no_volatility_gate" else features.variation_rank.ge(0.65)
    completed_return = pd.to_numeric(features.completed_return, errors="coerce")
    active = features.source_valid.fillna(False).astype(bool) & spread_gate & volatility_gate & np.isfinite(completed_return) & completed_return.ne(0)
    side = -np.sign(completed_return).astype("Int64").fillna(0).astype(int)
    if control == "direction_flip":
        side = -side
    return active, side


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, side = conditions(features, control)
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
        next_allowed = exit_time
        row = {"candidate": "HVCSER-12", "control": control, "split": split, "decision_time": decision, "feature_available_time": decision, "entry_time": entry, "exit_time": exit_time, "side": int(side.at[index])}
        for column in CLOCK_COLUMNS[8:]:
            row[column] = float(features.at[index, column])
        rows.append(row)
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    rows = clock[clock.split.eq(split)]
    if rows.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(rows.side.eq(1).sum()), int(rows.side.eq(-1).sum())
    months = pd.to_datetime(rows.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(rows), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(rows), "max_month_share": int(months.max()) / len(rows)}


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVCSER preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    if registration != prereg.build():
        raise RuntimeError("HVCSER preregistration content drift")
    bars = load_source()
    features = build_features(bars)
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(features, FEATURES)
    _write_gzip_csv(primary, CLOCK)
    for name, rows in controls.items():
        _write_gzip_csv(rows, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {
        "protocol_version": "hvcser_12_btc_ohlc_source_v1", "query": QUERY, "table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m",
        "window": [START.isoformat(), END.isoformat()], "source_columns": ["ts", "open", "high", "low", "close"], "rows": len(bars),
        "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)}, "feature_output": {"path": str(FEATURES), "sha256": sha(FEATURES), "rows": len(features)},
        "outcomes_opened": False, "candidate_incidence_opened_before_materialization": False, "no_imputation": True,
    }
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    SOURCE_MANIFEST.write_text(json.dumps(source_manifest, indent=2, allow_nan=False) + "\n")
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {
        check: passed
        for name, item in support.items()
        for check, passed in (
            (f"{name}_minimum_events", item["events"] >= MINIMUM_EVENTS[name]),
            (f"{name}_side_balance", item["minority_side_share"] >= 0.20),
            (f"{name}_month_concentration", item["max_month_share"] <= 0.45),
        )
    }
    passed = all(checks.values())
    core = {
        "protocol_version": "hvcser_12_source_support_v1", "policy_id": "HVCSER-12",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(rows), "promotion_authorized": False} for name, rows in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed, "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False, "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    report = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    result = run()
    print(json.dumps({"passed": result["support_passed"], "support": result["support"]}, indent=2))
