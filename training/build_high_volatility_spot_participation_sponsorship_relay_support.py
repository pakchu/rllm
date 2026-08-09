"""Build outcome-blind source support for frozen HVSPSR-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_spot_participation_sponsorship_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
PREREG_SHA = "f6adc656d9de242175df7e6a42a88e1d7012058eababa9e1b0e945f83e439460"
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
SOURCE_DIR = Path("data/high_volatility_spot_participation_sponsorship_relay_sources_2023_2026")
FEATURES = SOURCE_DIR / "daily_preentry_features.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_spot_participation_sponsorship_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_spot_participation_sponsorship_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_spot_participation_sponsorship_relay_support_2026-08-09.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_spot_participation_gate",
    "no_btc_variation_gate",
    "no_direction_agreement",
    "one_day_stale_features",
    "direction_flip",
    "same_clock_forced_long",
)
SPOT_QUERY = """
SELECT date_trunc('day',ts) AS source_day,
       count(*) AS spot_source_rows, count(DISTINCT ts) AS spot_distinct_timestamps,
       min(ts) AS spot_first_ts, max(ts) AS spot_last_ts,
       (array_agg(open ORDER BY ts))[1] AS spot_day_open,
       (array_agg(close ORDER BY ts DESC))[1] AS spot_day_close,
       sum(quote_asset_volume) AS spot_quote_volume,
       bool_and(open>0 AND close>0 AND high>=greatest(open,close,low)
                AND low<=least(open,close,high) AND quote_asset_volume>=0) AS spot_coherent
FROM bars_binance_spot
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
GROUP BY 1 ORDER BY 1
"""
PERP_QUERY = """
SELECT date_trunc('day',ts) AS source_day,
       count(*) AS perp_source_rows, count(DISTINCT ts) AS perp_distinct_timestamps,
       min(ts) AS perp_first_ts, max(ts) AS perp_last_ts,
       (array_agg(open ORDER BY ts))[1] AS perp_day_open,
       (array_agg(close ORDER BY ts DESC))[1] AS perp_day_close,
       sum(quote_asset_volume) AS perp_quote_volume,
       sqrt(sum(power(ln(close/open),2))) AS btc_realized_variation,
       bool_and(open>0 AND close>0 AND high>=greatest(open,close,low)
                AND low<=least(open,close,high) AND quote_asset_volume>=0) AS perp_coherent
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
GROUP BY 1 ORDER BY 1
"""
CLOCK_COLUMNS = (
    "candidate", "control", "split", "source_day", "decision_time",
    "feature_available_time", "entry_time", "exit_time", "side",
    "spot_return", "perp_return", "spot_quote_volume", "perp_quote_volume",
    "spot_participation_share", "spot_participation_rank",
    "btc_realized_variation", "btc_variation_rank",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chash(payload: Any) -> str:
    return prereg.canonical_hash(payload)


def strict_prior_midrank(values: pd.Series, lookback: int = 270, minimum: int = 180) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = np.asarray(history[-lookback:], dtype=float)
        if np.isfinite(current) and len(prior) >= minimum:
            output.at[index] = (np.sum(prior < current) + 0.5 * np.sum(prior == current)) / len(prior)
        if np.isfinite(current):
            history.append(float(current))
    return output


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_daily_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    from sqlalchemy import text

    engine = postgres_engine()
    try:
        with engine.connect() as connection:
            spot = pd.read_sql_query(text(SPOT_QUERY), connection, params={"start": START, "end": END})
            perp = pd.read_sql_query(text(PERP_QUERY), connection, params={"start": START, "end": END})
    finally:
        engine.dispose()
    for frame in (spot, perp):
        frame["source_day"] = pd.to_datetime(frame.source_day, utc=True)
    return spot, perp


def build_features(spot: pd.DataFrame, perp: pd.DataFrame) -> pd.DataFrame:
    frame = spot.merge(perp, on="source_day", how="inner", validate="one_to_one")
    numeric = (
        (frame.spot_source_rows == 1440)
        & (frame.spot_distinct_timestamps == 1440)
        & (frame.perp_source_rows == 1440)
        & (frame.perp_distinct_timestamps == 1440)
        & frame.spot_coherent.fillna(False).astype(bool)
        & frame.perp_coherent.fillna(False).astype(bool)
        & pd.to_numeric(frame.btc_realized_variation, errors="coerce").gt(0)
        & pd.to_numeric(frame.spot_day_open, errors="coerce").gt(0)
        & pd.to_numeric(frame.spot_day_close, errors="coerce").gt(0)
        & pd.to_numeric(frame.perp_day_open, errors="coerce").gt(0)
        & pd.to_numeric(frame.perp_day_close, errors="coerce").gt(0)
        & pd.to_numeric(frame.spot_quote_volume, errors="coerce").ge(0)
        & pd.to_numeric(frame.perp_quote_volume, errors="coerce").ge(0)
    )
    expected_last = frame.source_day + pd.Timedelta(hours=23, minutes=59)
    total_quote_volume = (
        pd.to_numeric(frame.spot_quote_volume, errors="coerce")
        + pd.to_numeric(frame.perp_quote_volume, errors="coerce")
    )
    frame["source_valid"] = (
        numeric
        & frame.spot_first_ts.eq(frame.source_day)
        & frame.spot_last_ts.eq(expected_last)
        & frame.perp_first_ts.eq(frame.source_day)
        & frame.perp_last_ts.eq(expected_last)
        & total_quote_volume.gt(0)
    )
    frame["spot_return"] = np.log(
        pd.to_numeric(frame.spot_day_close, errors="coerce") / pd.to_numeric(frame.spot_day_open, errors="coerce")
    ).where(frame.source_valid)
    frame["perp_return"] = np.log(
        pd.to_numeric(frame.perp_day_close, errors="coerce") / pd.to_numeric(frame.perp_day_open, errors="coerce")
    ).where(frame.source_valid)
    frame["spot_participation_share"] = (
        pd.to_numeric(frame.spot_quote_volume, errors="coerce") / total_quote_volume
    ).where(frame.source_valid)
    frame["spot_participation_rank"] = strict_prior_midrank(
        frame.spot_participation_share
    )
    frame["btc_variation_rank"] = strict_prior_midrank(
        pd.to_numeric(frame.btc_realized_variation, errors="coerce").where(frame.source_valid)
    )
    return frame.sort_values("source_day").reset_index(drop=True)


def conditions(frame: pd.DataFrame, control: str = "primary") -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    used = frame.shift(1) if control == "one_day_stale_features" else frame
    valid = (
        used.source_valid.fillna(False).astype(bool)
        & used.spot_return.ne(0)
        & used.perp_return.ne(0)
    )
    agreement_gate = (
        pd.Series(True, index=frame.index)
        if control == "no_direction_agreement"
        else np.sign(used.spot_return).eq(np.sign(used.perp_return))
    )
    participation_gate = (
        pd.Series(True, index=frame.index)
        if control == "no_spot_participation_gate"
        else used.spot_participation_rank.ge(0.75)
    )
    variation_gate = (
        pd.Series(True, index=frame.index)
        if control == "no_btc_variation_gate"
        else used.btc_variation_rank.ge(0.65)
    )
    active = valid & agreement_gate & participation_gate & variation_gate
    side = np.sign(used.spot_return).fillna(0).astype(int)
    if control == "direction_flip":
        side = -side
    elif control == "same_clock_forced_long":
        side = pd.Series(1, index=frame.index)
    return active, side, used


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, sides, used = conditions(features, control)
    rows: list[dict[str, Any]] = []
    for index in features.index[active]:
        decision = pd.Timestamp(features.at[index, "source_day"]) + pd.Timedelta(days=1)
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=12)
        split = next(
            (name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end),
            None,
        )
        if split is None:
            continue
        source = used.loc[index]
        rows.append({
            "candidate": "HVSPSR-12", "control": control, "split": split,
            "source_day": source.source_day, "decision_time": decision,
            "feature_available_time": decision, "entry_time": entry, "exit_time": exit_time,
            "side": int(sides.at[index]),
            "spot_return": float(source.spot_return), "perp_return": float(source.perp_return),
            "spot_quote_volume": float(source.spot_quote_volume),
            "perp_quote_volume": float(source.perp_quote_volume),
            "spot_participation_share": float(source.spot_participation_share),
            "spot_participation_rank": float(source.spot_participation_rank),
            "btc_realized_variation": float(source.btc_realized_variation),
            "btc_variation_rank": float(source.btc_variation_rank),
        })
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
        raise RuntimeError("HVSPSR preregistration drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text()); prereg.validate(registration)
    spot, perp = load_daily_sources()
    features = build_features(spot, perp)
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True); CLOCK.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(features, FEATURES); _write_gzip_csv(primary, CLOCK)
    for name, control in controls.items():
        _write_gzip_csv(control, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {
        "protocol_version": "hvspsr_12_sources_v1",
        "queries": {"spot_daily": SPOT_QUERY, "perpetual_daily": PERP_QUERY},
        "tables": ["bars_binance_spot", "bars_binance"],
        "window": [START.isoformat(), END.isoformat()],
        "features": {"path": str(FEATURES), "sha256": sha(FEATURES), "rows": len(features), "valid_rows": int(features.source_valid.sum())},
        "outcomes_opened": False, "gross9_rows_opened": False, "no_imputation": True,
    }
    manifest = {**source_core, "manifest_hash": chash(source_core)}
    SOURCE_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {
        check: passed
        for name, item in support.items()
        for check, passed in (
            (f"{name}_minimum_events", item["events"] >= MINIMUM[name]),
            (f"{name}_side_balance", item["minority_side_share"] >= 0.20),
            (f"{name}_month_concentration", item["max_month_share"] <= 0.45),
        )
    }
    passed = all(checks.values())
    core = {
        "protocol_version": "hvspsr_12_source_support_v1", "policy_id": "HVSPSR-12",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(control), "promotion_authorized": False} for name, control in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": chash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    report = run(); print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
