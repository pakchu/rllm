"""Outcome-blind source-support gate for frozen HVOIPSCR-8."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import (
    preregister_high_volatility_oi_price_sign_concordance_relay as prereg,
)
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
PREREG_SHA = "432c2b34251957108020d0769f61be8ba7a98a51e5d9e4af90ca8a65b2d44c5d"
START = pd.Timestamp("2023-04-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
SPLITS = {
    "train": (
        pd.Timestamp("2023-07-01T00:00:00Z"),
        pd.Timestamp("2024-01-01T00:00:00Z"),
    ),
    "test": (
        pd.Timestamp("2024-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-01T00:00:00Z"),
    ),
    "eval": (
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-01T00:00:00Z"),
    ),
    "final": (
        pd.Timestamp("2026-01-01T00:00:00Z"),
        END,
    ),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_concordance_gate",
    "no_gross_oi_activity_gate",
    "no_variation_gate",
    "one_block_stale_features",
    "direction_flip",
    "forced_long",
)
BAR_QUERY = """
SELECT date_bin('5 minutes',ts,TIMESTAMPTZ '1970-01-01 00:00:00+00') AS bar_time,
       (array_agg(open ORDER BY ts))[1] AS bar_open,
       max(high) AS bar_high, min(low) AS bar_low,
       (array_agg(close ORDER BY ts DESC))[1] AS bar_close,
       count(*) AS source_rows, count(DISTINCT ts) AS distinct_rows,
       min(ts) AS first_ts, max(ts) AS last_ts,
       bool_and(open>0 AND high>0 AND low>0 AND close>0
                AND high>=greatest(open,close,low)
                AND low<=least(open,close,high)) AS coherent
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
GROUP BY 1 ORDER BY 1
"""
OI_QUERY = """
SELECT ts, sum_open_interest, count(*) OVER (PARTITION BY ts) AS duplicate_count
FROM open_interest_binance
WHERE symbol='BTCUSDT' AND period='5m' AND source='open_interest_hist'
  AND ts>=:start AND ts<:end
ORDER BY ts
"""
SOURCE_DIR = Path(
    "data/high_volatility_oi_price_sign_concordance_relay_sources_2023_2026"
)
FEATURES = SOURCE_DIR / "eight_hour_features.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path(
    "data/high_volatility_oi_price_sign_concordance_relay_clocks_2023_2026.csv.gz"
)
CONTROL_DIR = Path(
    "data/high_volatility_oi_price_sign_concordance_relay_controls_2023_2026"
)
RESULT = Path(
    "results/high_volatility_oi_price_sign_concordance_relay_support_2026-08-12.json"
)
FEATURE_COLUMNS = (
    "decision_time",
    "feature_available_time",
    "source_valid",
    "completed_return",
    "realized_variation",
    "gross_oi_activity",
    "sign_concordance",
    "nonzero_pairs",
    "gross_oi_activity_rank",
    "concordance_rank",
    "variation_rank",
)
CLOCK_COLUMNS = (
    "candidate",
    "control",
    "split",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
    *FEATURE_COLUMNS[3:],
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return prereg.canonical_hash(value)


def strict_prior_midrank(
    values: pd.Series, lookback: int = 270, minimum: int = 180
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = np.asarray(history[-lookback:], dtype=float)
        if math.isfinite(current) and len(prior) >= minimum:
            output.at[index] = (
                np.sum(prior < current) + 0.5 * np.sum(prior == current)
            ) / len(prior)
        if math.isfinite(current):
            history.append(float(current))
    return output


def postgres_engine():
    from sqlalchemy import create_engine

    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(
        postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10}
    )


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    from sqlalchemy import text

    database = postgres_engine()
    try:
        with database.connect() as connection:
            bars = pd.read_sql_query(
                text(BAR_QUERY), connection, params={"start": START, "end": END}
            )
            oi = pd.read_sql_query(
                text(OI_QUERY), connection, params={"start": START, "end": END}
            )
    finally:
        database.dispose()
    return bars, oi


def build_features(bars: pd.DataFrame, oi: pd.DataFrame) -> pd.DataFrame:
    required_bars = {
        "bar_time",
        "bar_open",
        "bar_high",
        "bar_low",
        "bar_close",
        "source_rows",
        "distinct_rows",
        "first_ts",
        "last_ts",
        "coherent",
    }
    required_oi = {"ts", "sum_open_interest", "duplicate_count"}
    if not required_bars.issubset(bars) or not required_oi.issubset(oi):
        raise ValueError("HVOIPSCR source schema drift")
    prices = bars.copy()
    inventory = oi.copy()
    for column in ("bar_time", "first_ts", "last_ts"):
        prices[column] = pd.to_datetime(prices[column], utc=True, errors="coerce")
    for column in (
        "bar_open",
        "bar_high",
        "bar_low",
        "bar_close",
        "source_rows",
        "distinct_rows",
    ):
        prices[column] = pd.to_numeric(prices[column], errors="coerce")
    inventory["ts"] = pd.to_datetime(inventory["ts"], utc=True, errors="coerce")
    inventory["sum_open_interest"] = pd.to_numeric(
        inventory["sum_open_interest"], errors="coerce"
    )
    inventory["duplicate_count"] = pd.to_numeric(
        inventory["duplicate_count"], errors="coerce"
    )
    prices = prices.sort_values("bar_time", kind="mergesort").set_index("bar_time")
    inventory = inventory.sort_values("ts", kind="mergesort").set_index("ts")
    rows: list[dict[str, Any]] = []
    decisions = pd.date_range(
        START + pd.Timedelta(hours=10), END, freq="8h", inclusive="left"
    )
    for decision in decisions:
        price_index = pd.date_range(
            decision - pd.Timedelta(hours=8), decision, freq="5min", inclusive="left"
        )
        oi_index = pd.date_range(
            decision - pd.Timedelta(hours=8), decision, freq="5min", inclusive="both"
        )
        price = prices.reindex(price_index)
        oi_path = inventory.reindex(oi_index)
        expected_first = pd.Series(price_index, index=price_index)
        expected_last = pd.Series(
            price_index + pd.Timedelta(minutes=4), index=price_index
        )
        price_ok = (
            np.isfinite(
                price[
                    [
                        "bar_open",
                        "bar_high",
                        "bar_low",
                        "bar_close",
                        "source_rows",
                        "distinct_rows",
                    ]
                ]
            ).all(axis=1)
            & price["bar_open"].gt(0)
            & price["bar_high"].gt(0)
            & price["bar_low"].gt(0)
            & price["bar_close"].gt(0)
            & price["source_rows"].eq(5)
            & price["distinct_rows"].eq(5)
            & price["coherent"].eq(True)
            & price["first_ts"].eq(expected_first)
            & price["last_ts"].eq(expected_last)
        )
        oi_ok = (
            np.isfinite(oi_path[["sum_open_interest", "duplicate_count"]]).all(axis=1)
            & oi_path["sum_open_interest"].gt(0)
            & oi_path["duplicate_count"].eq(1)
        )
        valid = bool(
            len(price) == 96
            and len(oi_path) == 97
            and price_ok.all()
            and oi_ok.all()
        )
        completed_return = variation = gross_activity = sign_concordance = nonzero_pairs = math.nan
        if valid:
            bar_returns = np.log(
                price["bar_close"].to_numpy(float)
                / price["bar_open"].to_numpy(float)
            )
            oi_changes = np.diff(np.log(oi_path["sum_open_interest"].to_numpy(float)))
            gross_activity = float(np.abs(oi_changes).sum())
            sign_pairs = np.sign(oi_changes) * np.sign(bar_returns)
            nonzero_pairs = int(np.count_nonzero(sign_pairs))
            sign_concordance = (
                float(np.count_nonzero(sign_pairs > 0) / nonzero_pairs)
                if nonzero_pairs >= 72 else math.nan
            )
            variation = float(np.sqrt(np.square(bar_returns).sum()))
            completed_return = float(
                np.log(price["bar_close"].iloc[-1] / price["bar_open"].iloc[0])
            )
            valid = bool(
                completed_return != 0
                and variation > 0
                and gross_activity > 0
                and np.isfinite(
                    [completed_return, variation, gross_activity, sign_concordance, nonzero_pairs]
                ).all()
            )
        if not valid:
            completed_return = variation = gross_activity = sign_concordance = nonzero_pairs = math.nan
        rows.append(
            {
                "decision_time": decision,
                "feature_available_time": decision,
                "source_valid": valid,
                "completed_return": completed_return,
                "realized_variation": variation,
                "gross_oi_activity": gross_activity,
                "sign_concordance": sign_concordance,
                "nonzero_pairs": nonzero_pairs,
            }
        )
    frame = pd.DataFrame(rows)
    frame["gross_oi_activity_rank"] = strict_prior_midrank(
        frame["gross_oi_activity"].where(frame["source_valid"])
    )
    frame["concordance_rank"] = strict_prior_midrank(
        frame["sign_concordance"].where(frame["source_valid"])
    )
    frame["variation_rank"] = strict_prior_midrank(
        frame["realized_variation"].where(frame["source_valid"])
    )
    return frame[list(FEATURE_COLUMNS)]


def conditions(
    frame: pd.DataFrame, control: str = "primary"
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    used = frame.shift(1) if control == "one_block_stale_features" else frame
    valid = used["source_valid"].eq(True) & used["completed_return"].ne(0)
    gross_gate = (
        pd.Series(True, index=frame.index)
        if control == "no_gross_oi_activity_gate"
        else used["gross_oi_activity_rank"].ge(0.60)
    )
    concordance_gate = (
        pd.Series(True, index=frame.index)
        if control == "no_concordance_gate"
        else used["concordance_rank"].ge(0.80)
    )
    variation_gate = (
        pd.Series(True, index=frame.index)
        if control == "no_variation_gate"
        else used["variation_rank"].ge(0.65)
    )
    eligible = valid & gross_gate & concordance_gate & variation_gate
    onset = (
        eligible
        & ~eligible.shift(1, fill_value=False)
        & valid.shift(1, fill_value=False)
    )
    side = np.sign(used["completed_return"]).fillna(0).astype(int)
    if control == "direction_flip":
        side = -side
    elif control == "forced_long":
        side = pd.Series(1, index=frame.index)
    return onset, side, used


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    active, sides, used = conditions(features, control)
    rows: list[dict[str, Any]] = []
    next_available: pd.Timestamp | None = None
    for index in features.index[active]:
        decision = pd.Timestamp(features.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=8)
        if next_available is not None and entry < next_available:
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
        source = used.loc[index]
        next_available = exit_time
        rows.append(
            {
                "candidate": prereg.POLICY_ID,
                "control": control,
                "split": split,
                "decision_time": decision,
                "feature_available_time": decision,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": int(sides.at[index]),
                **{column: source[column] for column in FEATURE_COLUMNS[3:]},
            }
        )
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def support_stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = clock[clock["split"].eq(split)]
    if selected.empty:
        return {
            "events": 0,
            "longs": 0,
            "shorts": 0,
            "minority_side_share": 0.0,
            "max_month_share": 0.0,
        }
    longs = int(selected["side"].eq(1).sum())
    shorts = int(selected["side"].eq(-1).sum())
    months = pd.to_datetime(selected["entry_time"], utc=True).dt.strftime(
        "%Y-%m"
    ).value_counts()
    return {
        "events": len(selected),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": int(months.max()) / len(selected),
    }


def run() -> dict[str, Any]:
    if sha256(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVOIPSCR preregistration drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    bars, oi = load_sources()
    features = build_features(bars, oi)
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(features, FEATURES)
    _write_gzip_csv(primary, CLOCK)
    for name, value in controls.items():
        _write_gzip_csv(value, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {
        "protocol_version": "hvoipscr_8_sources_v1",
        "query_sha256": {
            "bars": hashlib.sha256(BAR_QUERY.encode()).hexdigest(),
            "oi": hashlib.sha256(OI_QUERY.encode()).hexdigest(),
        },
        "tables": ["bars_binance", "open_interest_binance"],
        "window": [START.isoformat(), END.isoformat()],
        "physical_rows": {
            "bars_1m": int(pd.to_numeric(bars["source_rows"]).sum()),
            "oi_5m": len(oi),
        },
        "features": {
            "path": str(FEATURES),
            "sha256": sha256(FEATURES),
            "rows": len(features),
            "valid_rows": int(features["source_valid"].sum()),
        },
        "outcomes_opened": False,
        "gross9_rows_opened": False,
        "no_imputation": True,
    }
    manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    SOURCE_MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    support = {name: support_stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, item in support.items():
        checks[f"{name}_minimum_events"] = item["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = item["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = item["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "hvoipscr_8_source_support_v1",
        "policy_id": prereg.POLICY_ID,
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": PREREG_SHA,
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifest": {
            "path": str(SOURCE_MANIFEST),
            "sha256": sha256(SOURCE_MANIFEST),
            "manifest_hash": manifest["manifest_hash"],
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
    RESULT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    report = run()
    print(
        json.dumps(
            {"passed": report["support_passed"], "support": report["support"]},
            indent=2,
        )
    )
