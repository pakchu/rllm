"""Build outcome-blind source support for frozen HVIGDR-6."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_interbar_gap_dominance_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path("training/build_high_volatility_interbar_gap_dominance_relay_support.py")
PREREG_SHA = "aa850bcee14c51a9defecb4e1f828660b226a01e5c84284914085c68e95183d0"
QUERY_START = pd.Timestamp("2023-04-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
REGISTRATION = prereg.build()
SPLITS = {
    name: (pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1]))
    for name, bounds in REGISTRATION["stages"].items()
}
POLICY = REGISTRATION["policy"]
SUPPORT_GATES = REGISTRATION["source_support_gates"]
MINIMUM_EVENTS = SUPPORT_GATES["minimum_events"]
CONTROLS = tuple(REGISTRATION["diagnostic_controls"]["names"])

QUERY = """SELECT ts,open,high,low,close,count(*) OVER (PARTITION BY ts) AS duplicate_count
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts"""

SOURCE_DIR = Path("data/high_volatility_interbar_gap_dominance_relay_sources_2023_2026")
FEATURES = SOURCE_DIR / "hourly_preentry_features.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_interbar_gap_dominance_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_interbar_gap_dominance_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_interbar_gap_dominance_relay_support_2026-08-10.json")

FEATURE_COLUMNS = (
    "decision_time", "feature_available_time", "source_valid", "valid_minute_count",
    "gap_count", "gap_energy", "dominant_gap_index", "dominant_gap_time",
    "dominant_gap", "latest_dominant_gap_index", "latest_dominant_gap_time",
    "latest_dominant_gap", "gap_dominance", "block_return",
    "direction_alignment", "latest_direction_alignment", "btc_realized_variation",
    "gap_dominance_rank", "block_return_rank", "variation_rank",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "gap_count", "gap_energy",
    "dominant_gap_index", "dominant_gap_time", "dominant_gap", "gap_dominance",
    "gap_dominance_rank", "block_return", "abs_block_return", "block_return_rank",
    "direction_alignment", "btc_realized_variation", "variation_rank",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def strict_prior_midrank(
    values: pd.Series, lookback: int = 2160, minimum: int = 1440
) -> pd.Series:
    """Rank each finite current value against finite strictly prior values."""
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    result = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if math.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior, dtype=float)
            result.at[index] = float(
                (np.count_nonzero(array < current) + 0.5 * np.count_nonzero(array == current))
                / len(array)
            )
        if math.isfinite(current):
            history.append(float(current))
    return result


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_source() -> pd.DataFrame:
    """Read only the preregistered BTCUSDT one-minute OHLC source."""
    from sqlalchemy import text

    engine = postgres_engine()
    try:
        with engine.connect() as connection:
            return pd.read_sql_query(
                text(QUERY),
                connection,
                params={"start": QUERY_START.to_pydatetime(), "end": END.to_pydatetime()},
            )
    finally:
        engine.dispose()


def prepare_source(bars: pd.DataFrame) -> pd.DataFrame:
    required = ["ts", "open", "high", "low", "close", "duplicate_count"]
    if bars.columns.tolist() != required:
        raise RuntimeError("HVIGDR source schema drift")
    frame = bars.copy()
    frame["ts"] = pd.to_datetime(frame.ts, utc=True, errors="coerce")
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    prices = frame[["open", "high", "low", "close"]]
    frame["source_valid"] = (
        frame.ts.notna()
        & np.isfinite(prices).all(axis=1)
        & prices.gt(0).all(axis=1)
        & frame.high.ge(prices[["open", "close"]].max(axis=1))
        & frame.low.le(prices[["open", "close"]].min(axis=1))
        & frame.high.ge(frame.low)
        & frame.duplicate_count.eq(1)
    )
    if frame.ts.duplicated().any():
        raise RuntimeError("HVIGDR duplicate source timestamps")
    return frame.sort_values("ts", kind="mergesort").set_index("ts")


def _invalid_feature(valid_minute_count: int) -> dict[str, Any]:
    return {
        "source_valid": False,
        "valid_minute_count": valid_minute_count,
        "gap_count": 0,
        "gap_energy": np.nan,
        "dominant_gap_index": np.nan,
        "dominant_gap_time": pd.NaT,
        "dominant_gap": np.nan,
        "latest_dominant_gap_index": np.nan,
        "latest_dominant_gap_time": pd.NaT,
        "latest_dominant_gap": np.nan,
        "gap_dominance": np.nan,
        "block_return": np.nan,
        "direction_alignment": False,
        "latest_direction_alignment": False,
        "btc_realized_variation": np.nan,
    }


def boundary_features(bars: pd.DataFrame, decision: pd.Timestamp) -> dict[str, Any]:
    """Compute the frozen six-hour gap state and exact prior-24-hour variation."""
    block_index = pd.date_range(
        decision - pd.Timedelta(hours=6), decision, freq="1min", inclusive="left"
    )
    day_index = pd.date_range(
        decision - pd.Timedelta(hours=24), decision, freq="1min", inclusive="left"
    )
    day = bars.reindex(day_index)
    valid_rows = day.source_valid.eq(True)
    valid_count = int(valid_rows.sum())
    if len(day) != 1440 or not bool(valid_rows.all()):
        return _invalid_feature(valid_count)
    block = day.reindex(block_index)

    opens = pd.to_numeric(block.open, errors="coerce").to_numpy(float)
    closes = pd.to_numeric(block.close, errors="coerce").to_numpy(float)
    gaps = np.log(opens[1:] / closes[:-1])
    abs_gaps = np.abs(gaps)
    gap_energy = float(np.square(gaps).sum())
    block_return = float(math.log(closes[-1] / opens[0]))
    minute_returns = np.log(
        pd.to_numeric(day.close, errors="coerce").to_numpy(float)
        / pd.to_numeric(day.open, errors="coerce").to_numpy(float)
    )
    variation = float(math.sqrt(float(np.square(minute_returns).sum())))
    finite = np.isfinite(gaps).all() and np.isfinite([gap_energy, block_return, variation]).all()
    if not finite or gap_energy <= 0 or block_return == 0:
        return _invalid_feature(valid_count)

    maximum = float(abs_gaps.max())
    ties = np.flatnonzero(abs_gaps == maximum)
    earliest_index = int(ties[0])
    latest_index = int(ties[-1])
    dominant_gap = float(gaps[earliest_index])
    latest_dominant_gap = float(gaps[latest_index])
    if dominant_gap == 0:
        return _invalid_feature(valid_count)
    dominant_time = block_index[earliest_index + 1]
    latest_time = block_index[latest_index + 1]
    return {
        "source_valid": True,
        "valid_minute_count": valid_count,
        "gap_count": len(gaps),
        "gap_energy": gap_energy,
        "dominant_gap_index": earliest_index + 1,
        "dominant_gap_time": dominant_time,
        "dominant_gap": dominant_gap,
        "latest_dominant_gap_index": latest_index + 1,
        "latest_dominant_gap_time": latest_time,
        "latest_dominant_gap": latest_dominant_gap,
        "gap_dominance": dominant_gap * dominant_gap / gap_energy,
        "block_return": block_return,
        "direction_alignment": bool(np.sign(dominant_gap) == np.sign(block_return)),
        "latest_direction_alignment": bool(
            np.sign(latest_dominant_gap) == np.sign(block_return)
        ),
        "btc_realized_variation": variation,
    }


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    source = prepare_source(bars)
    rows = [
        {
            "decision_time": decision,
            "feature_available_time": decision,
            **boundary_features(source, decision),
        }
        for decision in pd.date_range(
            QUERY_START + pd.Timedelta(hours=24), END, freq="1h", inclusive="left"
        )
    ]
    features = pd.DataFrame(rows)
    valid = features.source_valid.astype(bool)
    features["gap_dominance_rank"] = strict_prior_midrank(
        features.gap_dominance.where(valid),
        POLICY["history_hours"],
        POLICY["minimum_history_hours"],
    )
    features["block_return_rank"] = strict_prior_midrank(
        features.block_return.abs().where(valid),
        POLICY["history_hours"],
        POLICY["minimum_history_hours"],
    )
    features["variation_rank"] = strict_prior_midrank(
        features.btc_realized_variation.where(valid),
        POLICY["history_hours"],
        POLICY["minimum_history_hours"],
    )
    return features.loc[:, FEATURE_COLUMNS]


def active_and_side(
    features: pd.DataFrame, control: str = "primary"
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    if control not in ("primary", *CONTROLS):
        raise ValueError(f"unknown HVIGDR control: {control}")
    used = features.copy()
    if control == "latest_dominant_gap_tie_break":
        used["dominant_gap_index"] = features.latest_dominant_gap_index
        used["dominant_gap_time"] = features.latest_dominant_gap_time
        used["dominant_gap"] = features.latest_dominant_gap
        used["direction_alignment"] = features.latest_direction_alignment

    dominance_gate = (
        pd.Series(True, index=features.index)
        if control == "no_gap_dominance_gate"
        else pd.to_numeric(features.gap_dominance_rank, errors="coerce").ge(
            POLICY["gap_dominance_rank_min"]
        )
    )
    return_gate = (
        pd.Series(True, index=features.index)
        if control == "no_block_return_tail"
        else pd.to_numeric(features.block_return_rank, errors="coerce").ge(
            POLICY["block_return_rank_min"]
        )
    )
    variation_gate = (
        pd.Series(True, index=features.index)
        if control == "no_volatility_gate"
        else pd.to_numeric(features.variation_rank, errors="coerce").ge(
            POLICY["variation_rank_min"]
        )
    )
    side = np.sign(pd.to_numeric(features.block_return, errors="coerce")).fillna(0).astype(int)
    active = (
        features.source_valid.fillna(False).astype(bool)
        & dominance_gate
        & return_gate
        & variation_gate
        & used.direction_alignment.fillna(False).astype(bool)
        & side.ne(0)
    )
    if control == "direction_flip":
        side = -side
    return active, side, used


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    ordered = features.sort_values("decision_time", kind="mergesort").reset_index(drop=True)
    active, sides, used = active_and_side(ordered, control)
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for index in ordered.index[active]:
        decision = pd.Timestamp(ordered.at[index, "decision_time"])
        if decision.minute != 0 or decision.second != 0 or decision.microsecond != 0:
            raise RuntimeError("HVIGDR decision grid drift")
        entry = decision + pd.Timedelta(minutes=POLICY["entry_delay_minutes"])
        exit_time = entry + pd.Timedelta(hours=POLICY["hold_hours"])
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next(
            (
                name
                for name, (start, end) in SPLITS.items()
                if entry >= start and exit_time <= end
            ),
            None,
        )
        if split is None:
            continue
        reserved_until = exit_time
        block_return = float(ordered.at[index, "block_return"])
        row = {
            "candidate": "HVIGDR-6",
            "control": control,
            "split": split,
            "decision_time": decision,
            "feature_available_time": pd.Timestamp(ordered.at[index, "feature_available_time"]),
            "entry_time": entry,
            "exit_time": exit_time,
            "side": int(sides.at[index]),
            "gap_count": int(ordered.at[index, "gap_count"]),
            "gap_energy": float(ordered.at[index, "gap_energy"]),
            "dominant_gap_index": int(used.at[index, "dominant_gap_index"]),
            "dominant_gap_time": pd.Timestamp(used.at[index, "dominant_gap_time"]),
            "dominant_gap": float(used.at[index, "dominant_gap"]),
            "gap_dominance": float(ordered.at[index, "gap_dominance"]),
            "gap_dominance_rank": float(ordered.at[index, "gap_dominance_rank"]),
            "block_return": block_return,
            "abs_block_return": abs(block_return),
            "block_return_rank": float(ordered.at[index, "block_return_rank"]),
            "direction_alignment": bool(used.at[index, "direction_alignment"]),
            "btc_realized_variation": float(ordered.at[index, "btc_realized_variation"]),
            "variation_rank": float(ordered.at[index, "variation_rank"]),
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def support_stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = clock[clock.split.eq(split)]
    if selected.empty:
        return {
            "events": 0,
            "longs": 0,
            "shorts": 0,
            "minority_side_share": 0.0,
            "max_month_share": 0.0,
        }
    longs = int(selected.side.eq(1).sum())
    shorts = int(selected.side.eq(-1).sum())
    months = pd.to_datetime(selected.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(selected),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": int(months.max()) / len(selected),
    }


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVIGDR preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    if tuple(registration["diagnostic_controls"]["names"]) != CONTROLS:
        raise RuntimeError("HVIGDR diagnostic-control drift")

    bars = load_source()
    features = build_features(bars)
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(features, FEATURES)
    _write_gzip_csv(primary, CLOCK)
    for name, control_clock in controls.items():
        _write_gzip_csv(control_clock, CONTROL_DIR / f"{name}.csv.gz")

    source_core = {
        "protocol_version": "hvigdr_6_btc_ohlc_source_v1",
        "query": QUERY,
        "query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(),
        "table": "bars_binance",
        "symbol": "BTCUSDT",
        "interval": "1m",
        "columns": ["ts", "open", "high", "low", "close"],
        "window": [QUERY_START.isoformat(), END.isoformat()],
        "physical_rows": len(bars),
        "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
        "features": {
            "path": str(FEATURES),
            "sha256": sha(FEATURES),
            "rows": len(features),
            "valid_rows": int(features.source_valid.sum()),
        },
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "funding_values_opened": False,
        "gross9_rows_opened": False,
        "no_imputation": True,
    }
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    SOURCE_MANIFEST.write_text(json.dumps(source_manifest, indent=2, allow_nan=False) + "\n")

    support = {name: support_stats(primary, name) for name in SPLITS}
    checks = {
        check: passed
        for name, item in support.items()
        for check, passed in (
            (f"{name}_minimum_events", item["events"] >= MINIMUM_EVENTS[name]),
            (
                f"{name}_side_balance",
                item["minority_side_share"] >= SUPPORT_GATES["minority_side_share_min"],
            ),
            (
                f"{name}_month_concentration",
                item["max_month_share"] <= SUPPORT_GATES["max_month_share"],
            ),
        )
    }
    passed = all(checks.values())
    core = {
        "protocol_version": "hvigdr_6_source_support_v1",
        "policy_id": "HVIGDR-6",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": PREREG_SHA,
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifest": {
            "path": str(SOURCE_MANIFEST),
            "sha256": sha(SOURCE_MANIFEST),
            "manifest_hash": source_manifest["manifest_hash"],
        },
        "ranking": {
            "lookback_hours": POLICY["history_hours"],
            "minimum_prior_hours": POLICY["minimum_history_hours"],
            "current_excluded": True,
        },
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "funding_values_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "reservation": {
            "scope": "global",
            "interval": "half_open",
            "equal_open_after_exit_allowed": True,
            "split_crossing_action": "skip",
        },
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(control_clock),
                "promotion_authorized": False,
            }
            for name, control_clock in controls.items()
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
    argparse.ArgumentParser().parse_args()
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
