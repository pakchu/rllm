"""Build outcome-blind source support for frozen HVRNRR-6."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_round_number_rejection_reversal as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path("training/build_high_volatility_round_number_rejection_reversal_support.py")
PREREG_SHA = "ccdde7ff713e7a41932339970c1f94422eb99c904e175b7519d9bd5a544919d2"
QUERY_START = pd.Timestamp("2023-04-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
REGISTRATION = prereg.build()
POLICY = REGISTRATION["policy"]
SPLITS = {
    name: (pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1]))
    for name, bounds in REGISTRATION["stages"].items()
}
SUPPORT_GATES = REGISTRATION["source_support_gates"]
MINIMUM_EVENTS = SUPPORT_GATES["minimum_events"]
CONTROLS = tuple(REGISTRATION["diagnostic_controls"]["names"])

QUERY = """SELECT ts,open,high,low,close,count(*) OVER (PARTITION BY ts) AS duplicate_count
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts"""

SOURCE_DIR = Path("data/high_volatility_round_number_rejection_reversal_sources_2023_2026")
FEATURES = SOURCE_DIR / "hourly_preentry_features.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_round_number_rejection_reversal_clocks_2023_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_round_number_rejection_reversal_split_clocks_2023_2026")
CONTROL_DIR = Path("data/high_volatility_round_number_rejection_reversal_controls_2023_2026")
RESULT = Path("results/high_volatility_round_number_rejection_reversal_support_2026-08-10.json")

GEOMETRY_COLUMNS = (
    "geometry_decision_time", "hour_first_open", "nearest_level", "hour_high",
    "hour_low", "hour_last_close", "opening_side", "penetration", "rejection",
    "rejection_side", "penetration_rank",
)
FEATURE_COLUMNS = (
    "decision_time", "feature_available_time", "source_valid", "hour_valid",
    "variation_valid", "hour_minute_count", "variation_minute_count",
    *GEOMETRY_COLUMNS, "btc_realized_variation", "variation_rank",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", *GEOMETRY_COLUMNS,
    "btc_realized_variation", "variation_rank",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def nearest_round_level(first_open: float, quantum: float = 1000.0) -> float:
    """Nearest lattice level, with exact half-quantum ties assigned lower."""
    if not math.isfinite(first_open) or first_open <= 0 or quantum <= 0:
        return math.nan
    return float(quantum * math.ceil(first_open / quantum - 0.5))


def strict_prior_midrank(
    values: pd.Series, lookback: int = 2160, minimum: int = 1440
) -> pd.Series:
    """Rank a finite current value against finite strictly prior values only."""
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
    """Read only frozen BTCUSDT one-minute OHLC source columns."""
    from sqlalchemy import text

    engine = postgres_engine()
    try:
        with engine.connect() as connection:
            return pd.read_sql_query(
                text(QUERY), connection,
                params={"start": QUERY_START.to_pydatetime(), "end": END.to_pydatetime()},
            )
    finally:
        engine.dispose()


def prepare_source(bars: pd.DataFrame) -> pd.DataFrame:
    required = ["ts", "open", "high", "low", "close", "duplicate_count"]
    if bars.columns.tolist() != required:
        raise RuntimeError("HVRNRR source schema drift")
    frame = bars.copy()
    frame["ts"] = pd.to_datetime(frame.ts, utc=True, errors="coerce")
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    prices = frame[["open", "high", "low", "close"]]
    frame["row_valid"] = (
        frame.ts.notna()
        & np.isfinite(prices).all(axis=1)
        & prices.gt(0).all(axis=1)
        & frame.high.ge(prices[["open", "close"]].max(axis=1))
        & frame.low.le(prices[["open", "close"]].min(axis=1))
        & frame.high.ge(frame.low)
        & frame.duplicate_count.eq(1)
    )
    if frame.ts.duplicated().any():
        raise RuntimeError("HVRNRR duplicate source timestamps")
    return frame.sort_values("ts", kind="mergesort").set_index("ts")


def boundary_features(bars: pd.DataFrame, decision: pd.Timestamp) -> dict[str, Any]:
    """Compute exact [D-1h,D) rejection and exact [D-24h,D) variation."""
    hour_index = pd.date_range(
        decision - pd.Timedelta(hours=1), decision, freq="1min", inclusive="left"
    )
    day_index = pd.date_range(
        decision - pd.Timedelta(hours=24), decision, freq="1min", inclusive="left"
    )
    hour = bars.reindex(hour_index)
    day = bars.reindex(day_index)
    hour_valid_rows = hour.row_valid.eq(True)
    variation_valid_rows = day.row_valid.eq(True)
    hour_valid = len(hour) == 60 and bool(hour_valid_rows.all())
    variation_valid = len(day) == 1440 and bool(variation_valid_rows.all())
    hour_count = int(hour_valid_rows.sum())
    variation_count = int(variation_valid_rows.sum())

    invalid = {
        "source_valid": False, "hour_valid": hour_valid,
        "variation_valid": variation_valid, "hour_minute_count": hour_count,
        "variation_minute_count": variation_count,
        "geometry_decision_time": decision, "hour_first_open": np.nan,
        "nearest_level": np.nan, "hour_high": np.nan, "hour_low": np.nan,
        "hour_last_close": np.nan, "opening_side": 0, "penetration": np.nan,
        "rejection": False, "rejection_side": 0,
        "btc_realized_variation": np.nan,
    }
    if not (hour_valid and variation_valid):
        return invalid

    first_open = float(hour.open.iloc[0])
    hour_high = float(pd.to_numeric(hour.high, errors="coerce").max())
    hour_low = float(pd.to_numeric(hour.low, errors="coerce").min())
    last_close = float(hour.close.iloc[-1])
    level = nearest_round_level(first_open, POLICY["round_quantum_usd"])
    minute_returns = np.log(
        pd.to_numeric(day.close, errors="coerce").to_numpy(float)
        / pd.to_numeric(day.open, errors="coerce").to_numpy(float)
    )
    variation = float(math.sqrt(float(np.square(minute_returns).sum())))
    if not np.isfinite([first_open, hour_high, hour_low, last_close, level, variation]).all():
        return invalid

    opening_side = -1 if first_open < level else 1 if first_open > level else 0
    if opening_side < 0:
        penetration = max(0.0, (hour_high - level) / level)
        rejection = penetration > 0 and last_close < level
        rejection_side = -1 if rejection else 0
    elif opening_side > 0:
        penetration = max(0.0, (level - hour_low) / level)
        rejection = penetration > 0 and last_close > level
        rejection_side = 1 if rejection else 0
    else:
        penetration = 0.0
        rejection = False
        rejection_side = 0
    return {
        "source_valid": True, "hour_valid": True, "variation_valid": True,
        "hour_minute_count": hour_count, "variation_minute_count": variation_count,
        "geometry_decision_time": decision, "hour_first_open": first_open,
        "nearest_level": level, "hour_high": hour_high, "hour_low": hour_low,
        "hour_last_close": last_close, "opening_side": opening_side,
        "penetration": float(penetration), "rejection": bool(rejection),
        "rejection_side": rejection_side, "btc_realized_variation": variation,
    }


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    source = prepare_source(bars)
    rows = [
        {"decision_time": decision, "feature_available_time": decision,
         **boundary_features(source, decision)}
        for decision in pd.date_range(
            QUERY_START + pd.Timedelta(hours=24), END, freq="1h", inclusive="left"
        )
    ]
    features = pd.DataFrame(rows)
    valid = features.source_valid.astype(bool)
    features["penetration_rank"] = strict_prior_midrank(
        features.penetration.where(valid),
        POLICY["history_hours"], POLICY["minimum_history_hours"],
    )
    features["variation_rank"] = strict_prior_midrank(
        features.btc_realized_variation.where(valid),
        POLICY["history_hours"], POLICY["minimum_history_hours"],
    )
    return features.loc[:, FEATURE_COLUMNS]


def active_and_side(
    features: pd.DataFrame, control: str = "primary"
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    if control not in ("primary", *CONTROLS):
        raise ValueError(f"unknown HVRNRR control: {control}")
    used = features.copy()
    geometry_valid = features.source_valid.fillna(False).astype(bool)
    if control == "one_hour_stale_rejection":
        shifted = features.loc[:, GEOMETRY_COLUMNS].shift(1)
        for column in GEOMETRY_COLUMNS:
            used[column] = shifted[column]
        prior_decision = pd.to_datetime(features.decision_time, utc=True).shift(1)
        geometry_valid = (
            features.source_valid.shift(1).eq(True)
            & prior_decision.add(pd.Timedelta(hours=1)).eq(
                pd.to_datetime(features.decision_time, utc=True)
            )
        )

    penetration_gate = (
        pd.Series(True, index=features.index)
        if control == "no_penetration_rank_gate"
        else pd.to_numeric(used.penetration_rank, errors="coerce").ge(
            POLICY["penetration_rank_min"]
        )
    )
    variation_gate = (
        pd.Series(True, index=features.index)
        if control == "no_volatility_gate"
        else pd.to_numeric(features.variation_rank, errors="coerce").ge(
            POLICY["variation_rank_min"]
        )
    )
    side = pd.to_numeric(used.rejection_side, errors="coerce").fillna(0).astype(int)
    active = (
        geometry_valid & features.source_valid.fillna(False).astype(bool)
        & used.rejection.eq(True)
        & pd.to_numeric(used.penetration, errors="coerce").gt(0)
        & penetration_gate & variation_gate & side.ne(0)
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
            raise RuntimeError("HVRNRR decision grid drift")
        entry = decision + pd.Timedelta(minutes=POLICY["entry_delay_minutes"])
        exit_time = entry + pd.Timedelta(hours=POLICY["hold_hours"])
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next(
            (name for name, (start, end) in SPLITS.items()
             if entry >= start and exit_time <= end),
            None,
        )
        if split is None:
            continue
        reserved_until = exit_time
        row = {
            "candidate": "HVRNRR-6", "control": control, "split": split,
            "decision_time": decision,
            "feature_available_time": pd.Timestamp(ordered.at[index, "feature_available_time"]),
            "entry_time": entry, "exit_time": exit_time, "side": int(sides.at[index]),
            "btc_realized_variation": float(ordered.at[index, "btc_realized_variation"]),
            "variation_rank": float(ordered.at[index, "variation_rank"]),
        }
        for column in GEOMETRY_COLUMNS:
            row[column] = used.at[index, column]
        rows.append(row)
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def support_stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = clock[clock.split.eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0,
                "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(selected.side.eq(1).sum())
    shorts = int(selected.side.eq(-1).sum())
    months = pd.to_datetime(selected.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(selected), "longs": longs, "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": int(months.max()) / len(selected),
    }


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVRNRR preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    if tuple(registration["diagnostic_controls"]["names"]) != CONTROLS:
        raise RuntimeError("HVRNRR diagnostic-control drift")

    bars = load_source()
    features = build_features(bars)
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(features, FEATURES)
    _write_gzip_csv(primary, CLOCK)
    split_frames = {name: primary[primary.split.eq(name)].copy() for name in SPLITS}
    for name, split_frame in split_frames.items():
        _write_gzip_csv(split_frame, SPLIT_DIR / f"{name}.csv.gz")
    for name, control_clock in controls.items():
        _write_gzip_csv(control_clock, CONTROL_DIR / f"{name}.csv.gz")

    source_core = {
        "protocol_version": "hvrnrr_6_btc_ohlc_source_v1", "query": QUERY,
        "query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(),
        "table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m",
        "columns": ["ts", "open", "high", "low", "close"],
        "window": [QUERY_START.isoformat(), END.isoformat()], "physical_rows": len(bars),
        "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
        "features": {"path": str(FEATURES), "sha256": sha(FEATURES),
                     "rows": len(features), "valid_rows": int(features.source_valid.sum())},
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "funding_values_opened": False, "gross9_rows_opened": False,
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
            (f"{name}_side_balance", item["minority_side_share"] >= SUPPORT_GATES["minority_side_share_min"]),
            (f"{name}_month_concentration", item["max_month_share"] <= SUPPORT_GATES["max_month_share"]),
        )
    }
    passed = all(checks.values())
    core = {
        "protocol_version": "hvrnrr_6_source_support_v1", "policy_id": "HVRNRR-6",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA,
                            "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST),
                            "manifest_hash": source_manifest["manifest_hash"]},
        "ranking": {"lookback_hours": POLICY["history_hours"],
                    "minimum_prior_hours": POLICY["minimum_history_hours"],
                    "current_excluded": True},
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "funding_values_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "split_artifacts": {
            name: {"path": str(SPLIT_DIR / f"{name}.csv.gz"),
                   "sha256": sha(SPLIT_DIR / f"{name}.csv.gz"), "rows": len(frame)}
            for name, frame in split_frames.items()
        },
        "reservation": {"scope": "global", "interval": "half_open",
                        "equal_open_after_exit_allowed": True, "split_crossing_action": "skip"},
        "controls": {
            name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"),
                   "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"),
                   "rows": len(frame), "promotion_authorized": False}
            for name, frame in controls.items()
        },
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
