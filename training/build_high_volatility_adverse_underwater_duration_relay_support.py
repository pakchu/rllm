"""Source-only support gate for frozen HVAUD-8."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_adverse_underwater_duration_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-04-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "5a8d279c9b6900482ed9f43ba8054aa368498d4bad7a96af4acffc2bb0ee8286"
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_duration_tail",
    "no_variation_gate",
    "raw_duration_at_most_half",
    "one_block_stale_geometry",
    "direction_flip",
    "forced_long",
)
ROOT = Path("data/high_volatility_adverse_underwater_duration_sources_2023_2026")
PANEL = ROOT / "states.csv.gz"
MANIFEST = ROOT / "manifest.json"
CLOCK = Path("data/high_volatility_adverse_underwater_duration_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_adverse_underwater_duration_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_adverse_underwater_duration_relay_support_2026-08-10.json")
QUERY = """SELECT ts,open,high,low,close
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def prior_rank(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    output = np.full(len(values), np.nan)
    history: list[float] = []
    for index, value in enumerate(values):
        prior = np.asarray(history[-270:], dtype=float)
        if math.isfinite(value) and len(prior) >= 180:
            output[index] = (np.sum(prior < value) + 0.5 * np.sum(prior == value)) / len(prior)
        if math.isfinite(value):
            history.append(float(value))
    return pd.Series(output, index=series.index)


def longest_true_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def adverse_duration(close: np.ndarray, side: int) -> float:
    close = np.asarray(close, dtype=float)
    if len(close) != 480 or not np.isfinite(close).all() or np.any(close <= 0) or side not in (-1, 1):
        return math.nan
    underwater = close < np.maximum.accumulate(close) if side == 1 else close > np.minimum.accumulate(close)
    return longest_true_run(underwater) / len(close)

def engine():
    from sqlalchemy import create_engine

    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def materialize() -> tuple[pd.DataFrame, dict[str, Any]]:
    from sqlalchemy import text

    database = engine()
    with database.connect() as connection:
        frame = pd.read_sql_query(
            text(QUERY),
            connection,
            params={"start": START.to_pydatetime(), "end": END.to_pydatetime()},
        )
    database.dispose()
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame["ts"].duplicated().any():
        raise RuntimeError("duplicate HVAUD source timestamps")
    frame = frame.set_index("ts").sort_index()
    rows: list[dict[str, Any]] = []
    for decision in pd.date_range(START.ceil("8h"), END, freq="8h", inclusive="left"):
        index = pd.date_range(decision - pd.Timedelta("8h"), decision, freq="1min", inclusive="left")
        window = frame.reindex(index)
        valid = (
            len(window) == 480
            and np.isfinite(window).all().all()
            and window.gt(0).all().all()
            and window["high"].ge(window[["open", "close"]].max(axis=1)).all()
            and window["low"].le(window[["open", "close"]].min(axis=1)).all()
            and window["high"].ge(window["low"]).all()
        )
        if valid:
            returns = np.diff(np.log(window["close"].to_numpy(float)))
            variation = float(np.sqrt(np.square(returns).sum()))
            block_return = float(np.log(window["close"].iloc[-1] / window["open"].iloc[0]))
            duration = adverse_duration(window["close"].to_numpy(float), int(np.sign(block_return)))
            late_return = float(np.log(window["close"].iloc[-1] / window["open"].iloc[-120]))
            valid = variation > 0 and block_return != 0 and math.isfinite(duration)
        if not valid:
            variation = duration = block_return = late_return = math.nan
        rows.append(
            {
                "decision_time": decision,
                "source_valid": valid,
                "realized_variation": variation,
                "adverse_duration": duration,
                "block_return": block_return,
                "late_return": late_return,
            }
        )
    states = pd.DataFrame(rows)
    states["variation_rank"] = prior_rank(states["realized_variation"].where(states["source_valid"]))
    states["duration_rank"] = prior_rank(states["adverse_duration"].where(states["source_valid"]))
    ROOT.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(states, PANEL)
    core = {
        "protocol_version": "hvaud_source_v1",
        "query": QUERY,
        "window": [START.isoformat(), END.isoformat()],
        "outcomes_opened": False,
        "candidate_incidence_opened_before_materialization": False,
        "output": {
            "path": str(PANEL),
            "sha256": sha256(PANEL),
            "rows": len(states),
            "valid_rows": int(states["source_valid"].sum()),
        },
    }
    manifest = {**core, "manifest_hash": canonical_hash(core)}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return states, manifest


def active(states: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series]:
    block_return = states["block_return"]
    late_return = states["late_return"]
    variation_rank = states["variation_rank"]
    duration_rank = states["duration_rank"]
    duration = states["adverse_duration"]
    if control == "one_block_stale_geometry":
        block_return, late_return, variation_rank, duration_rank, duration = [
            value.shift(1)
            for value in (block_return, late_return, variation_rank, duration_rank, duration)
        ]
    variation_gate = (
        pd.Series(True, index=states.index)
        if control == "no_variation_gate"
        else variation_rank.ge(0.65)
    )
    if control == "no_duration_tail":
        duration_gate = pd.Series(True, index=states.index)
    elif control == "raw_duration_at_most_half":
        duration_gate = duration.le(0.5)
    else:
        duration_gate = duration_rank.le(0.25)
    agreement = (
        block_return.ne(0)
        & late_return.ne(0)
        & np.sign(block_return).eq(np.sign(late_return))
    )
    eligible = states["source_valid"] & variation_gate & duration_gate & agreement
    onset = eligible & ~eligible.shift(1, fill_value=False) & states["source_valid"].shift(1, fill_value=False)
    return onset, np.sign(block_return)


def clock(states: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    onset, side = active(states, control)
    rows: list[dict[str, Any]] = []
    next_available: pd.Timestamp | None = None
    for index in states.index[onset]:
        decision = pd.Timestamp(states.at[index, "decision_time"])
        entry = decision + pd.Timedelta("5m")
        exit_time = entry + pd.Timedelta("8h")
        if next_available is not None and entry < next_available:
            continue
        split = next(
            (name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end),
            None,
        )
        if split is None:
            continue
        direction = int(side.at[index])
        if control == "direction_flip":
            direction = -direction
        elif control == "forced_long":
            direction = 1
        next_available = exit_time
        rows.append(
            {
                "candidate": "HVAUD-8",
                "control": control,
                "split": split,
                "decision_time": decision,
                "feature_available_time": decision,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": direction,
                "realized_variation": float(states.at[index, "realized_variation"]),
                "variation_rank": float(states.at[index, "variation_rank"]),
                "adverse_duration": float(states.at[index, "adverse_duration"]),
                "duration_rank": float(states.at[index, "duration_rank"]),
            }
        )
    columns = [
        "candidate", "control", "split", "decision_time", "feature_available_time",
        "entry_time", "exit_time", "side", "realized_variation", "variation_rank",
        "adverse_duration", "duration_rank",
    ]
    return pd.DataFrame(rows, columns=columns)


def support_stats(candidate_clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = candidate_clock[candidate_clock["split"].eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(selected["side"].eq(1).sum())
    shorts = int(selected["side"].eq(-1).sum())
    return {
        "events": len(selected),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": int(selected["entry_time"].dt.strftime("%Y-%m").value_counts().max()) / len(selected),
    }


def run() -> dict[str, Any]:
    if sha256(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVAUD preregistration drift")
    states, source_manifest = materialize()
    primary = clock(states)
    controls = {name: clock(states, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, value in controls.items():
        _write_gzip_csv(value, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: support_stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {
        "protocol_version": "hvaud_8_source_support_v1",
        "policy_id": "HVAUD-8",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": sha256(prereg.DEFAULT_OUTPUT),
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifest": {
            "path": str(MANIFEST),
            "sha256": sha256(MANIFEST),
            "manifest_hash": source_manifest["manifest_hash"],
        },
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)},
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(value),
                "promotion_authorized": False,
            }
            for name, value in controls.items()
        },
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
