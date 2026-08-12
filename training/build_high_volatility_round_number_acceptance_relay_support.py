"""Source-only support gate for frozen HVRNAR-8."""
from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_round_number_acceptance_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "5ad360c0ad9d39dc183769379496eb3c362f8a97f1a15f2fdf2c077e8609894d"
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_variation_gate",
    "single_close_cross_instead_of_acceptance",
    "one_decision_stale_acceptance",
    "direction_flip",
    "forced_long",
)
ROOT = Path("data/high_volatility_round_number_acceptance_sources_2023_2026")
PANEL = ROOT / "states.csv.gz"
MANIFEST = ROOT / "manifest.json"
CLOCK = Path("data/high_volatility_round_number_acceptance_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_round_number_acceptance_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_round_number_acceptance_relay_support_2026-08-12.json")
QUERY = """SELECT ts,open,high,low,close
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def prior_rank(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    output = np.full(len(values), np.nan)
    ordered: list[float] = []
    history: deque[float] = deque()
    for index, value in enumerate(values):
        if math.isfinite(value) and len(history) >= 6048:
            left = bisect.bisect_left(ordered, value)
            right = bisect.bisect_right(ordered, value)
            output[index] = (left + 0.5 * (right - left)) / len(ordered)
        if math.isfinite(value):
            bisect.insort(ordered, float(value))
            history.append(float(value))
            if len(history) > 8640:
                old = history.popleft()
                ordered.pop(bisect.bisect_left(ordered, old))
    return pd.Series(output, index=series.index)


def engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def materialize() -> tuple[pd.DataFrame, dict[str, Any]]:
    from sqlalchemy import text

    database = engine()
    with database.connect() as connection:
        raw = pd.read_sql_query(text(QUERY), connection, params={"start": START.to_pydatetime(), "end": END.to_pydatetime()})
    database.dispose()
    raw["ts"] = pd.to_datetime(raw["ts"], utc=True)
    for column in ("open", "high", "low", "close"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    if raw["ts"].duplicated().any():
        raise RuntimeError("duplicate HVRNAR source timestamps")
    raw = raw.set_index("ts").sort_index()
    grouped = raw.resample("5min", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), rows=("close", "size")
    )
    coherent = (
        grouped["rows"].eq(5) & np.isfinite(grouped[["open", "high", "low", "close"]]).all(axis=1)
        & grouped[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & grouped["high"].ge(grouped[["open", "close"]].max(axis=1))
        & grouped["low"].le(grouped[["open", "close"]].min(axis=1))
        & grouped["high"].ge(grouped["low"])
    )
    bar_return = np.log(grouped["close"] / grouped["open"]).where(coherent)
    variation = np.sqrt(bar_return.pow(2).rolling(288, min_periods=288).sum())
    closes = grouped["close"].where(coherent)
    prior_max = closes.shift(6).rolling(6, min_periods=6).max()
    prior_min = closes.shift(6).rolling(6, min_periods=6).min()
    current_max = closes.rolling(6, min_periods=6).max()
    current_min = closes.rolling(6, min_periods=6).min()
    upward_low = np.floor(prior_max / 1000.0).astype("Int64") + 1
    upward_high = np.ceil(current_min / 1000.0).astype("Int64") - 1
    downward_low = np.floor(current_max / 1000.0).astype("Int64") + 1
    downward_high = np.ceil(prior_min / 1000.0).astype("Int64") - 1
    upward_count = (upward_high - upward_low + 1).clip(lower=0).astype("Int64")
    downward_count = (downward_high - downward_low + 1).clip(lower=0).astype("Int64")
    side = pd.Series(0, index=grouped.index, dtype=int)
    side.loc[upward_count.eq(1)] = 1
    side.loc[downward_count.eq(1)] = -1
    ambiguous = upward_count.add(downward_count, fill_value=0).ne(1)
    side.loc[ambiguous] = 0
    level = pd.Series(np.nan, index=grouped.index)
    level.loc[side.eq(1)] = upward_low.loc[side.eq(1)].astype(float) * 1000.0
    level.loc[side.eq(-1)] = downward_low.loc[side.eq(-1)].astype(float) * 1000.0
    source_valid = coherent.rolling(288, min_periods=288).sum().eq(288) & variation.gt(0)
    states = pd.DataFrame({
        "decision_time": grouped.index + pd.Timedelta("5m"),
        "source_valid": source_valid.to_numpy(bool),
        "realized_variation": variation.to_numpy(float),
        "acceptance_side": side.to_numpy(int),
        "round_level": level.to_numpy(float),
        "prior_last_close": closes.shift(6).to_numpy(float),
        "current_last_close": closes.to_numpy(float),
    })
    states["variation_rank"] = prior_rank(states["realized_variation"].where(states["source_valid"]))
    ROOT.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(states, PANEL)
    core = {
        "protocol_version": "hvrnar_source_v1", "query": QUERY,
        "window": [START.isoformat(), END.isoformat()], "outcomes_opened": False,
        "candidate_incidence_opened_before_materialization": False,
        "output": {"path": str(PANEL), "sha256": sha256(PANEL), "rows": len(states), "valid_rows": int(states["source_valid"].sum())},
    }
    manifest = {**core, "manifest_hash": canonical_hash(core)}
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return states, manifest


def active(states: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series]:
    side = states["acceptance_side"].copy()
    if control == "single_close_cross_instead_of_acceptance":
        before = states["prior_last_close"]
        after = states["current_last_close"]
        side = pd.Series(0, index=states.index, dtype=int)
        side.loc[np.floor(before / 1000).lt(np.floor(after / 1000))] = 1
        side.loc[np.floor(before / 1000).gt(np.floor(after / 1000))] = -1
    if control == "one_decision_stale_acceptance":
        side = side.shift(1, fill_value=0)
    variation_gate = pd.Series(True, index=states.index) if control == "no_variation_gate" else states["variation_rank"].ge(0.65)
    return states["source_valid"] & side.ne(0) & variation_gate, side


def clock(states: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    eligible, side = active(states, control)
    rows: list[dict[str, Any]] = []
    next_available: pd.Timestamp | None = None
    for index in states.index[eligible]:
        decision = pd.Timestamp(states.at[index, "decision_time"])
        entry = decision + pd.Timedelta("5m")
        exit_time = entry + pd.Timedelta("8h")
        if next_available is not None and entry < next_available:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        direction = int(side.at[index])
        if control == "direction_flip": direction = -direction
        elif control == "forced_long": direction = 1
        next_available = exit_time
        rows.append({
            "candidate": POLICY_ID, "control": control, "split": split,
            "decision_time": decision, "feature_available_time": decision,
            "entry_time": entry, "exit_time": exit_time, "side": direction,
            "realized_variation": float(states.at[index, "realized_variation"]),
            "variation_rank": float(states.at[index, "variation_rank"]),
            "round_level": float(states.at[index, "round_level"]) if math.isfinite(states.at[index, "round_level"]) else math.nan,
        })
    return pd.DataFrame(rows, columns=["candidate", "control", "split", "decision_time", "feature_available_time", "entry_time", "exit_time", "side", "realized_variation", "variation_rank", "round_level"])


def support_stats(candidate_clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = candidate_clock[candidate_clock["split"].eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(selected["side"].eq(1).sum()), int(selected["side"].eq(-1).sum())
    return {"events": len(selected), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(selected), "max_month_share": int(selected["entry_time"].dt.strftime("%Y-%m").value_counts().max()) / len(selected)}


def run() -> dict[str, Any]:
    if sha256(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVRNAR preregistration drift")
    states, source_manifest = materialize()
    primary = clock(states)
    controls = {name: clock(states, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, value in controls.items(): _write_gzip_csv(value, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: support_stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {
        "protocol_version": "hvrnar_8_source_support_v1", "policy_id": POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": sha256(prereg.DEFAULT_OUTPUT), "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(MANIFEST), "sha256": sha256(MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(value), "promotion_authorized": False} for name, value in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
