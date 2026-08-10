"""Build outcome-blind source support for frozen HVPFPR-6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_pre_funding_pressure_release_reversal as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
PREREG_SHA = "a211fe87f549d4df4c74c6d52cea648b6304051b5fd434edce6a3194de49ae84"
QUERY_START = pd.Timestamp("2023-04-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
SPLITS = {
    name: (pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1]))
    for name, bounds in prereg.build()["stages"].items()
}
MINIMUM = prereg.build()["source_support_gates"]["minimum_events"]
SUPPORT_GATES = prereg.build()["source_support_gates"]
CONTROLS = (
    "no_return_tail",
    "no_volatility_gate",
    "no_premium_alignment",
    "one_boundary_stale_pressure",
    "direction_flip",
)

BTC_QUERY = """SELECT ts,open,high,low,close,count(*) OVER (PARTITION BY ts) AS duplicate_count
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts"""
PREMIUM_QUERY = """SELECT ts,open,high,low,close,count(*) OVER (PARTITION BY ts) AS duplicate_count
FROM bars_binance_premium
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts"""

SOURCE_DIR = Path("data/high_volatility_pre_funding_pressure_release_reversal_sources_2023_2026")
FEATURES = SOURCE_DIR / "pre_boundary_features.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_pre_funding_pressure_release_reversal_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_pre_funding_pressure_release_reversal_controls_2023_2026")
RESULT = Path("results/high_volatility_pre_funding_pressure_release_reversal_support_2026-08-10.json")

FEATURE_COLUMNS = (
    "decision_time", "feature_available_time", "source_valid", "btc_return",
    "abs_btc_return", "premium_pressure", "pressure_alignment",
    "btc_realized_variation", "return_tail_rank", "variation_rank",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "btc_return", "premium_pressure",
    "pressure_alignment", "return_tail_rank", "btc_realized_variation", "variation_rank",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def strict_prior_midrank(
    values: pd.Series, lookback: int = 270, minimum: int = 180
) -> pd.Series:
    """Rank each finite value against finite strictly prior source-valid values."""
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = np.asarray(history[-lookback:], dtype=float)
        if np.isfinite(current) and len(prior) >= minimum:
            result.at[index] = (
                np.count_nonzero(prior < current)
                + 0.5 * np.count_nonzero(prior == current)
            ) / len(prior)
        if np.isfinite(current):
            history.append(float(current))
    return result


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read only the two preregistered one-minute OHLC sources."""
    from sqlalchemy import text

    engine = postgres_engine()
    try:
        with engine.connect() as connection:
            params = {"start": QUERY_START.to_pydatetime(), "end": END.to_pydatetime()}
            btc = pd.read_sql_query(text(BTC_QUERY), connection, params=params)
            premium = pd.read_sql_query(text(PREMIUM_QUERY), connection, params=params)
    finally:
        engine.dispose()
    return btc, premium


def prepare_ohlc(frame: pd.DataFrame, *, positive: bool) -> pd.DataFrame:
    required = {"ts", "open", "high", "low", "close", "duplicate_count"}
    if set(frame.columns) != required:
        raise ValueError("HVPFPR source schema drift")
    result = frame.copy()
    result["ts"] = pd.to_datetime(result.ts, utc=True, errors="coerce")
    numeric = ["open", "high", "low", "close", "duplicate_count"]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    coherent = (
        result.ts.notna()
        & np.isfinite(result[numeric]).all(axis=1)
        & result.high.ge(result[["open", "close"]].max(axis=1))
        & result.low.le(result[["open", "close"]].min(axis=1))
        & result.high.ge(result.low)
        & result.duplicate_count.eq(1)
    )
    if positive:
        coherent &= result[["open", "high", "low", "close"]].gt(0).all(axis=1)
    result["source_valid"] = coherent
    if result.ts.duplicated().any():
        raise RuntimeError("HVPFPR duplicate timestamps")
    return result.sort_values("ts", kind="mergesort").set_index("ts")


def boundary_features(
    btc: pd.DataFrame, premium: pd.DataFrame, decision: pd.Timestamp
) -> dict[str, Any]:
    hour_index = pd.date_range(
        decision - pd.Timedelta(hours=1), decision, freq="1min", inclusive="left"
    )
    day_index = pd.date_range(
        decision - pd.Timedelta(hours=24), decision, freq="1min", inclusive="left"
    )
    btc_hour = btc.reindex(hour_index)
    premium_hour = premium.reindex(hour_index)
    btc_day = btc.reindex(day_index)
    valid = (
        len(hour_index) == 60
        and len(day_index) == 1440
        and not btc_hour.source_valid.isna().any()
        and not premium_hour.source_valid.isna().any()
        and not btc_day.source_valid.isna().any()
        and bool(btc_hour.source_valid.all())
        and bool(premium_hour.source_valid.all())
        and bool(btc_day.source_valid.all())
    )
    if not valid:
        return {
            "source_valid": False, "btc_return": np.nan, "abs_btc_return": np.nan,
            "premium_pressure": np.nan, "pressure_alignment": False,
            "btc_realized_variation": np.nan,
        }
    btc_return = float(np.log(float(btc_hour.close.iloc[-1]) / float(btc_hour.open.iloc[0])))
    premium_pressure = float(pd.to_numeric(premium_hour.close, errors="coerce").mean())
    minute_returns = np.log(
        pd.to_numeric(btc_day.close, errors="coerce").to_numpy(float)
        / pd.to_numeric(btc_day.open, errors="coerce").to_numpy(float)
    )
    variation = float(np.sqrt(np.sum(np.square(minute_returns))))
    source_valid = bool(
        np.isfinite([btc_return, premium_pressure, variation]).all()
        and btc_return != 0
        and premium_pressure != 0
    )
    alignment = bool(
        source_valid
        and np.sign(premium_pressure) == np.sign(btc_return)
    )
    return {
        "source_valid": source_valid,
        "btc_return": btc_return if source_valid else np.nan,
        "abs_btc_return": abs(btc_return) if source_valid else np.nan,
        "premium_pressure": premium_pressure if source_valid else np.nan,
        "pressure_alignment": alignment,
        "btc_realized_variation": variation if source_valid else np.nan,
    }


def build_features(btc: pd.DataFrame, premium: pd.DataFrame) -> pd.DataFrame:
    btc = prepare_ohlc(btc, positive=True)
    premium = prepare_ohlc(premium, positive=False)
    rows: list[dict[str, Any]] = []
    for decision in pd.date_range(
        QUERY_START + pd.Timedelta(hours=24), END, freq="8h", inclusive="left"
    ):
        values = boundary_features(btc, premium, decision)
        rows.append({
            "decision_time": decision,
            "feature_available_time": decision,
            **values,
        })
    features = pd.DataFrame(rows)
    valid = features.source_valid.astype(bool)
    features["return_tail_rank"] = strict_prior_midrank(features.abs_btc_return.where(valid))
    features["variation_rank"] = strict_prior_midrank(
        features.btc_realized_variation.where(valid)
    )
    return features.loc[:, FEATURE_COLUMNS]


def active_and_side(
    features: pd.DataFrame, control: str = "primary"
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    current_return = pd.to_numeric(features.btc_return, errors="coerce")
    pressure = pd.to_numeric(features.premium_pressure, errors="coerce")
    valid_pressure = features.source_valid.fillna(False).astype(bool)
    if control == "one_boundary_stale_pressure":
        pressure = pressure.shift(1)
        prior_valid = valid_pressure.shift(1, fill_value=False)
        decisions = pd.to_datetime(features.decision_time, utc=True)
        adjacent = decisions.sub(decisions.shift(1)).eq(pd.Timedelta(hours=8))
        valid_pressure = valid_pressure & prior_valid & adjacent
    nonzero = pressure.ne(0) & current_return.ne(0)
    aligned = np.sign(pressure).eq(np.sign(current_return)) & nonzero
    return_gate = (
        pd.Series(True, index=features.index)
        if control == "no_return_tail"
        else pd.to_numeric(features.return_tail_rank, errors="coerce").ge(0.75)
    )
    variation_gate = (
        pd.Series(True, index=features.index)
        if control == "no_volatility_gate"
        else pd.to_numeric(features.variation_rank, errors="coerce").ge(0.65)
    )
    alignment_gate = (
        pd.Series(True, index=features.index)
        if control == "no_premium_alignment"
        else aligned
    )
    active = valid_pressure & nonzero & return_gate & variation_gate & alignment_gate
    side = -np.sign(current_return).fillna(0).astype(int)
    if control == "direction_flip":
        side = -side
    return active & side.ne(0), side, pressure


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, sides, pressures = active_and_side(features, control)
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for index in features.index[active]:
        decision = pd.Timestamp(features.at[index, "decision_time"])
        if decision.minute != 0 or decision.second != 0 or decision.hour not in (0, 8, 16):
            raise RuntimeError("HVPFPR decision grid drift")
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=6)
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next(
            (name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end),
            None,
        )
        if split is None:
            continue
        reserved_until = exit_time
        rows.append({
            "candidate": "HVPFPR-6", "control": control, "split": split,
            "decision_time": decision,
            "feature_available_time": pd.Timestamp(features.at[index, "feature_available_time"]),
            "entry_time": entry, "exit_time": exit_time, "side": int(sides.at[index]),
            "btc_return": float(features.at[index, "btc_return"]),
            "premium_pressure": float(pressures.at[index]),
            "pressure_alignment": bool(
                np.sign(pressures.at[index]) == np.sign(features.at[index, "btc_return"])
            ),
            "return_tail_rank": float(features.at[index, "return_tail_rank"]),
            "btc_realized_variation": float(features.at[index, "btc_realized_variation"]),
            "variation_rank": float(features.at[index, "variation_rank"]),
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    frame = clock[clock.split.eq(split)]
    if frame.empty:
        return {
            "events": 0, "longs": 0, "shorts": 0,
            "minority_side_share": 0.0, "max_month_share": 0.0,
        }
    longs = int(frame.side.eq(1).sum())
    shorts = int(frame.side.eq(-1).sum())
    months = pd.to_datetime(frame.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(frame), "longs": longs, "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(frame),
        "max_month_share": int(months.max()) / len(frame),
    }


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVPFPR preregistration drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    if tuple(registration["diagnostic_controls"]["names"]) != CONTROLS:
        raise RuntimeError("HVPFPR diagnostic-control drift")
    btc, premium = load_sources()
    features = build_features(btc, premium)
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(features, FEATURES)
    _write_gzip_csv(primary, CLOCK)
    for name, control_clock in controls.items():
        _write_gzip_csv(control_clock, CONTROL_DIR / f"{name}.csv.gz")

    source_core = {
        "protocol_version": "hvpfpr_6_sources_v1",
        "queries": {
            "btc_ohlc": BTC_QUERY, "premium_ohlc": PREMIUM_QUERY,
        },
        "query_sha256": {
            "btc_ohlc": hashlib.sha256(BTC_QUERY.encode()).hexdigest(),
            "premium_ohlc": hashlib.sha256(PREMIUM_QUERY.encode()).hexdigest(),
        },
        "tables": ["bars_binance", "bars_binance_premium"],
        "window": [QUERY_START.isoformat(), END.isoformat()],
        "physical_rows": {"btc": len(btc), "premium": len(premium)},
        "features": {"path": str(FEATURES), "sha256": sha(FEATURES), "rows": len(features)},
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "funding_values_opened": False,
        "no_imputation": True,
    }
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    SOURCE_MANIFEST.write_text(
        json.dumps(source_manifest, indent=2, allow_nan=False) + "\n"
    )

    support = {name: stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM[name]
        checks[f"{name}_side_balance"] = (
            values["minority_side_share"] >= SUPPORT_GATES["minority_side_share_min"]
        )
        checks[f"{name}_month_concentration"] = (
            values["max_month_share"] <= SUPPORT_GATES["max_month_share"]
        )
    passed = all(checks.values())
    core = {
        "protocol_version": "hvpfpr_6_source_support_v1", "policy_id": "HVPFPR-6",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA,
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifest": {
            "path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST),
            "manifest_hash": source_manifest["manifest_hash"],
        },
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False, "funding_values_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(control_clock), "promotion_authorized": False,
            }
            for name, control_clock in controls.items()
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
