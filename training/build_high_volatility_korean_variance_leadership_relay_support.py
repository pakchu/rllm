"""Build outcome-blind source support for frozen HVKVLR-6."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_korean_variance_leadership_relay as prereg


ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path(
    "training/build_high_volatility_korean_variance_leadership_relay_support.py"
)
PREREG_SHA = "adef4c9ba8f020c410692a55328b1f57b0a6f150b19bfa229818e8e613080d1e"
START = pd.Timestamp("2023-04-01T00:00:00Z")
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

UPBIT_QUERY = """SELECT ts,open,high,low,close
FROM bars_upbit
WHERE symbol='KRW-BTC' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts"""
BINANCE_QUERY = """SELECT ts,open,high,low,close
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts"""

SOURCE_DIR = Path(
    "data/high_volatility_korean_variance_leadership_relay_sources_2023_2026"
)
PAIR_PANEL = SOURCE_DIR / "hourly_aligned_variance_pairs.csv.gz"
FEATURE_PANEL = SOURCE_DIR / "hourly_preentry_features.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path(
    "data/high_volatility_korean_variance_leadership_relay_clocks_2023_2026.csv.gz"
)
CONTROL_DIR = Path(
    "data/high_volatility_korean_variance_leadership_relay_controls_2023_2026"
)
RESULT = Path(
    "results/high_volatility_korean_variance_leadership_relay_support_2026-08-10.json"
)

PAIR_COLUMNS = (
    "decision_time", "feature_available_time", "source_valid",
    "upbit_source_rows", "binance_source_rows", "upbit_variation",
    "binance_variation", "variance_leadership", "upbit_final_hour_return",
    "binance_final_hour_return", "return_magnitude_leadership",
)
FEATURE_COLUMNS = (
    *PAIR_COLUMNS, "variance_leadership_rank", "binance_variation_rank",
    "return_magnitude_leadership_rank",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "upbit_variation", "binance_variation",
    "variance_leadership", "variance_leadership_rank", "binance_variation_rank",
    "upbit_final_hour_return", "binance_final_hour_return",
    "return_magnitude_leadership", "return_magnitude_leadership_rank",
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
    """Rank finite values against only the most recent finite prior decisions."""
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    result = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if math.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior, dtype=float)
            result.at[index] = float(
                (
                    np.count_nonzero(array < current)
                    + 0.5 * np.count_nonzero(array == current)
                )
                / len(array)
            )
        if math.isfinite(current):
            history.append(float(current))
    return result


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(
        postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10}
    )


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read only the preregistered OHLC fields from the two frozen sources."""
    from sqlalchemy import text

    engine = postgres_engine()
    params = {"start": START.to_pydatetime(), "end": END.to_pydatetime()}
    try:
        with engine.connect() as connection:
            upbit = pd.read_sql_query(text(UPBIT_QUERY), connection, params=params)
            binance = pd.read_sql_query(text(BINANCE_QUERY), connection, params=params)
        return upbit, binance
    finally:
        engine.dispose()


def prepare_source(bars: pd.DataFrame, venue: str) -> pd.DataFrame:
    required = ["ts", "open", "high", "low", "close"]
    if bars.columns.tolist() != required:
        raise RuntimeError(f"HVKVLR {venue} source schema drift")
    frame = bars.copy()
    frame["ts"] = pd.to_datetime(frame.ts, utc=True, errors="coerce")
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.ts.isna().any() or frame.ts.duplicated().any():
        raise RuntimeError(f"HVKVLR invalid or duplicate {venue} source timestamps")
    prices = frame[["open", "high", "low", "close"]]
    frame["row_valid"] = (
        np.isfinite(prices).all(axis=1)
        & prices.gt(0).all(axis=1)
        & frame.high.ge(prices[["open", "close"]].max(axis=1))
        & frame.low.le(prices[["open", "close"]].min(axis=1))
        & frame.high.ge(frame.low)
    )
    return frame.sort_values("ts", kind="mergesort").set_index("ts")


def _invalid_pair(upbit_rows: int, binance_rows: int) -> dict[str, Any]:
    return {
        "source_valid": False,
        "upbit_source_rows": upbit_rows,
        "binance_source_rows": binance_rows,
        "upbit_variation": np.nan,
        "binance_variation": np.nan,
        "variance_leadership": np.nan,
        "upbit_final_hour_return": np.nan,
        "binance_final_hour_return": np.nan,
        "return_magnitude_leadership": np.nan,
    }


def boundary_pair(
    upbit: pd.DataFrame, binance: pd.DataFrame, decision: pd.Timestamp
) -> dict[str, Any]:
    """Compute the exact aligned [D-6h,D) frozen source geometry."""
    expected = pd.date_range(
        decision - pd.Timedelta(hours=POLICY["window_hours"]),
        decision,
        freq="1min",
        inclusive="left",
    )
    upbit_window = upbit.reindex(expected)
    binance_window = binance.reindex(expected)
    upbit_rows = int(upbit_window.row_valid.eq(True).sum())
    binance_rows = int(binance_window.row_valid.eq(True).sum())
    if not (
        len(expected) == 360
        and len(upbit_window) == 360
        and len(binance_window) == 360
        and bool(upbit_window.row_valid.eq(True).all())
        and bool(binance_window.row_valid.eq(True).all())
    ):
        return _invalid_pair(upbit_rows, binance_rows)

    upbit_minute_returns = np.log(
        upbit_window.close.to_numpy(float) / upbit_window.open.to_numpy(float)
    )
    binance_minute_returns = np.log(
        binance_window.close.to_numpy(float) / binance_window.open.to_numpy(float)
    )
    upbit_variation = float(np.square(upbit_minute_returns).sum())
    binance_variation = float(np.square(binance_minute_returns).sum())
    if not (
        math.isfinite(upbit_variation)
        and math.isfinite(binance_variation)
        and upbit_variation > 0
        and binance_variation > 0
    ):
        return _invalid_pair(upbit_rows, binance_rows)

    final_start = 360 - 60
    upbit_final = float(
        np.log(
            float(upbit_window.close.iloc[-1])
            / float(upbit_window.open.iloc[final_start])
        )
    )
    binance_final = float(
        np.log(
            float(binance_window.close.iloc[-1])
            / float(binance_window.open.iloc[final_start])
        )
    )
    leadership = float(np.log(upbit_variation / binance_variation))
    if upbit_final == 0 or binance_final == 0:
        return_magnitude = math.nan
    else:
        return_magnitude = float(np.log(abs(upbit_final) / abs(binance_final)))
    return {
        "source_valid": bool(math.isfinite(leadership)),
        "upbit_source_rows": upbit_rows,
        "binance_source_rows": binance_rows,
        "upbit_variation": upbit_variation,
        "binance_variation": binance_variation,
        "variance_leadership": leadership,
        "upbit_final_hour_return": upbit_final,
        "binance_final_hour_return": binance_final,
        "return_magnitude_leadership": return_magnitude,
    }


def build_pair_panel(upbit_bars: pd.DataFrame, binance_bars: pd.DataFrame) -> pd.DataFrame:
    upbit = prepare_source(upbit_bars, "upbit")
    binance = prepare_source(binance_bars, "binance")
    rows = [
        {
            "decision_time": decision,
            "feature_available_time": decision,
            **boundary_pair(upbit, binance, decision),
        }
        for decision in pd.date_range(START, END, freq="1h", inclusive="left")
    ]
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def build_features(pair: pd.DataFrame) -> pd.DataFrame:
    if pair.columns.tolist() != list(PAIR_COLUMNS):
        raise RuntimeError("HVKVLR pair-panel schema drift")
    features = pair.sort_values("decision_time", kind="mergesort").reset_index(drop=True).copy()
    decisions = pd.to_datetime(features.decision_time, utc=True, errors="coerce")
    exact_hours = (
        decisions.dt.minute.eq(0)
        & decisions.dt.second.eq(0)
        & decisions.dt.microsecond.eq(0)
    )
    if (
        decisions.isna().any()
        or decisions.duplicated().any()
        or not decisions.is_monotonic_increasing
        or not bool(exact_hours.all())
    ):
        raise RuntimeError("HVKVLR pair-panel decision grid invalid")
    valid = features.source_valid.fillna(False).astype(bool)
    features["variance_leadership_rank"] = strict_prior_midrank(
        features.variance_leadership.where(valid),
        POLICY["history_hours"],
        POLICY["minimum_history_hours"],
    )
    features["binance_variation_rank"] = strict_prior_midrank(
        features.binance_variation.where(valid),
        POLICY["history_hours"],
        POLICY["minimum_history_hours"],
    )
    magnitude_valid = valid & np.isfinite(features.return_magnitude_leadership)
    features["return_magnitude_leadership_rank"] = strict_prior_midrank(
        features.return_magnitude_leadership.where(magnitude_valid),
        POLICY["history_hours"],
        POLICY["minimum_history_hours"],
    )
    return features.loc[:, FEATURE_COLUMNS]


def active_and_side(
    features: pd.DataFrame, control: str = "primary"
) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """Return frozen eligibility, source-valid onset, side, and used geometry."""
    if control not in ("primary", *CONTROLS):
        raise ValueError(f"unknown HVKVLR control: {control}")
    ordered = features.copy()
    used = ordered.copy()
    if control == "one_hour_stale_features":
        used = ordered.shift(1)
        used["decision_time"] = ordered.decision_time
        used["feature_available_time"] = pd.to_datetime(
            ordered.feature_available_time, utc=True, errors="coerce"
        ).shift(1)

    source_valid = used.source_valid.eq(True)
    upbit_return = pd.to_numeric(used.upbit_final_hour_return, errors="coerce")
    binance_return = pd.to_numeric(used.binance_final_hour_return, errors="coerce")
    same_long = upbit_return.gt(0) & binance_return.gt(0)
    same_short = upbit_return.lt(0) & binance_return.lt(0)
    side = pd.Series(np.where(same_long, 1, np.where(same_short, -1, 0)), index=used.index)

    if control == "return_magnitude_leadership":
        leadership = pd.to_numeric(
            used.return_magnitude_leadership_rank, errors="coerce"
        )
    else:
        leadership = pd.to_numeric(used.variance_leadership_rank, errors="coerce")
    leadership_gate = (
        pd.Series(True, index=used.index)
        if control == "no_variance_leadership_tail"
        else leadership.ge(POLICY["leadership_rank_min"])
    )
    variation_gate = (
        pd.Series(True, index=used.index)
        if control == "no_binance_variation_gate"
        else pd.to_numeric(used.binance_variation_rank, errors="coerce").ge(
            POLICY["binance_variation_rank_min"]
        )
    )
    eligible = source_valid & leadership_gate & variation_gate & side.ne(0)

    decisions = pd.to_datetime(ordered.decision_time, utc=True, errors="coerce")
    adjacent = decisions.shift(1).add(pd.Timedelta(hours=1)).eq(decisions)
    onset = (
        eligible
        & adjacent
        & source_valid.shift(1, fill_value=False)
        & ~eligible.shift(1, fill_value=False)
    )
    if control == "direction_flip":
        side = -side
    elif control == "forced_long":
        side = side.where(side.eq(0), 1)
    return eligible, onset, side.astype(int), used


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    ordered = features.sort_values("decision_time", kind="mergesort").reset_index(drop=True)
    _, onset, sides, used = active_and_side(ordered, control)
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for index in ordered.index[onset]:
        decision = pd.Timestamp(ordered.at[index, "decision_time"])
        if (
            decision.minute != 0
            or decision.second != 0
            or decision.microsecond != 0
        ):
            raise RuntimeError("HVKVLR decision grid drift")
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
        rows.append(
            {
                "candidate": "HVKVLR-6",
                "control": control,
                "split": split,
                "decision_time": decision,
                "feature_available_time": pd.Timestamp(
                    used.at[index, "feature_available_time"]
                ),
                "entry_time": entry,
                "exit_time": exit_time,
                "side": int(sides.at[index]),
                "upbit_variation": float(used.at[index, "upbit_variation"]),
                "binance_variation": float(used.at[index, "binance_variation"]),
                "variance_leadership": float(used.at[index, "variance_leadership"]),
                "variance_leadership_rank": float(
                    used.at[index, "variance_leadership_rank"]
                ),
                "binance_variation_rank": float(
                    used.at[index, "binance_variation_rank"]
                ),
                "upbit_final_hour_return": float(
                    used.at[index, "upbit_final_hour_return"]
                ),
                "binance_final_hour_return": float(
                    used.at[index, "binance_final_hour_return"]
                ),
                "return_magnitude_leadership": float(
                    used.at[index, "return_magnitude_leadership"]
                ),
                "return_magnitude_leadership_rank": float(
                    used.at[index, "return_magnitude_leadership_rank"]
                ),
            }
        )
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
    months = pd.to_datetime(
        selected.entry_time, utc=True
    ).dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(selected),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(selected),
        "max_month_share": int(months.max()) / len(selected),
    }


def deterministic_csv_gzip(frame: pd.DataFrame) -> bytes:
    text = frame.to_csv(
        index=False, float_format="%.12g", lineterminator="\n"
    ).encode()
    buffer = io.BytesIO()
    with gzip.GzipFile(
        fileobj=buffer, mode="wb", compresslevel=6, mtime=0
    ) as output:
        output.write(text)
    return buffer.getvalue()


def deterministic_json(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()


def write_immutable(path: Path, content: bytes) -> None:
    """Create once, permit byte-identical reruns, and reject drift."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite immutable HVKVLR artifact: {path}")
        return
    path.write_bytes(content)


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVKVLR preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    if registration != REGISTRATION:
        raise RuntimeError("HVKVLR committed preregistration payload drift")
    if tuple(registration["diagnostic_controls"]["names"]) != CONTROLS:
        raise RuntimeError("HVKVLR diagnostic-control drift")

    upbit, binance = load_sources()
    pair = build_pair_panel(upbit, binance)
    features = build_features(pair)
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}

    write_immutable(PAIR_PANEL, deterministic_csv_gzip(pair))
    write_immutable(FEATURE_PANEL, deterministic_csv_gzip(features))
    write_immutable(CLOCK, deterministic_csv_gzip(primary))
    for name, frame in controls.items():
        write_immutable(CONTROL_DIR / f"{name}.csv.gz", deterministic_csv_gzip(frame))

    source_core = {
        "protocol_version": "hvkvlr_6_aligned_venue_variance_source_v1",
        "queries": {"upbit": UPBIT_QUERY, "binance": BINANCE_QUERY},
        "query_sha256": {
            "upbit": hashlib.sha256(UPBIT_QUERY.encode()).hexdigest(),
            "binance": hashlib.sha256(BINANCE_QUERY.encode()).hexdigest(),
        },
        "sources": {
            "upbit": {
                "table": "bars_upbit", "symbol": "KRW-BTC", "interval": "1m",
                "physical_rows": len(upbit),
            },
            "binance": {
                "table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m",
                "physical_rows": len(binance),
            },
        },
        "columns": ["ts", "open", "high", "low", "close"],
        "window": [START.isoformat(), END.isoformat()],
        "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
        "pair_panel": {
            "path": str(PAIR_PANEL), "sha256": sha(PAIR_PANEL), "rows": len(pair),
        },
        "feature_panel": {
            "path": str(FEATURE_PANEL), "sha256": sha(FEATURE_PANEL),
            "rows": len(features), "valid_rows": int(features.source_valid.sum()),
        },
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "funding_values_opened": False,
        "gross9_rows_opened": False,
        "no_imputation": True,
        "deterministic_immutable_artifacts": True,
    }
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    write_immutable(SOURCE_MANIFEST, deterministic_json(source_manifest))

    support = {name: support_stats(primary, name) for name in SPLITS}
    checks = {
        check: passed
        for name, item in support.items()
        for check, passed in (
            (f"{name}_minimum_events", item["events"] >= MINIMUM_EVENTS[name]),
            (
                f"{name}_side_balance",
                item["minority_side_share"]
                >= SUPPORT_GATES["minority_side_share_min"],
            ),
            (
                f"{name}_month_concentration",
                item["max_month_share"] <= SUPPORT_GATES["max_month_share"],
            ),
        )
    }
    passed = all(checks.values())
    core = {
        "protocol_version": "hvkvlr_6_source_support_v1",
        "policy_id": "HVKVLR-6",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA,
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifest": {
            "path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST),
            "manifest_hash": source_manifest["manifest_hash"],
        },
        "ranking": {
            "lookback_valid_decisions": POLICY["history_hours"],
            "minimum_prior_valid_decisions": POLICY["minimum_history_hours"],
            "current_excluded": True,
            "ties": "midrank",
        },
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "funding_values_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "reservation": {
            "scope": "global", "hours": POLICY["hold_hours"],
            "interval": "half_open", "equal_open_after_exit_allowed": True,
            "split_crossing_action": "skip",
        },
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(frame), "promotion_authorized": False,
            }
            for name, frame in controls.items()
        },
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
        "deterministic_immutable_artifacts": True,
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    write_immutable(RESULT, deterministic_json(result))
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
