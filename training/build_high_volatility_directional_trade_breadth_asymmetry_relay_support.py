"""Build outcome-blind source support for frozen HVDTBA-6."""
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

from training import preregister_high_volatility_directional_trade_breadth_asymmetry_relay as prereg


ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path(
    "training/build_high_volatility_directional_trade_breadth_asymmetry_relay_support.py"
)
PREREG_SHA = "21adc9b086b13be8ed798709f0e49ed2d062fb80722679b60a6f550ef8b0a5db"
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

QUERY = """SELECT ts,open,high,low,close,number_of_trades,quote_asset_volume
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts"""

SOURCE_DIR = Path(
    "data/high_volatility_directional_trade_breadth_asymmetry_relay_sources_2023_2026"
)
PAIR_PANEL = SOURCE_DIR / "hourly_directional_pairs.csv.gz"
FEATURE_PANEL = SOURCE_DIR / "hourly_preentry_features.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path(
    "data/high_volatility_directional_trade_breadth_asymmetry_relay_clocks_2023_2026.csv.gz"
)
SPLIT_DIR = Path(
    "data/high_volatility_directional_trade_breadth_asymmetry_relay_split_clocks_2023_2026"
)
CONTROL_DIR = Path(
    "data/high_volatility_directional_trade_breadth_asymmetry_relay_controls_2023_2026"
)
RESULT = Path(
    "results/high_volatility_directional_trade_breadth_asymmetry_relay_support_2026-08-10.json"
)

PAIR_COLUMNS = (
    "decision_time", "feature_available_time", "source_valid", "breadth_valid",
    "variation_valid", "quote_breadth_valid", "breadth_minute_count",
    "variation_minute_count", "positive_minute_count", "negative_minute_count",
    "zero_return_minute_count", "up_count", "down_count",
    "directional_trade_breadth", "up_quote_volume", "down_quote_volume",
    "quote_volume_directional_breadth", "btc_realized_variation",
)
FEATURE_COLUMNS = (
    *PAIR_COLUMNS, "absolute_breadth_rank", "absolute_quote_breadth_rank",
    "variation_rank",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "positive_minute_count",
    "negative_minute_count", "up_count", "down_count", "directional_trade_breadth",
    "absolute_breadth_rank", "up_quote_volume", "down_quote_volume",
    "quote_volume_directional_breadth", "absolute_quote_breadth_rank",
    "btc_realized_variation", "variation_rank",
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
    """Rank each finite current value against finite valid decisions strictly prior."""
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
    return create_engine(
        postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10}
    )


def load_source() -> pd.DataFrame:
    """Read only the preregistered BTCUSDT one-minute source fields."""
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
    required = [
        "ts", "open", "high", "low", "close", "number_of_trades",
        "quote_asset_volume",
    ]
    if bars.columns.tolist() != required:
        raise RuntimeError("HVDTBA source schema drift")
    frame = bars.copy()
    frame["ts"] = pd.to_datetime(frame.ts, utc=True, errors="coerce")
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.ts.isna().any() or frame.ts.duplicated().any():
        raise RuntimeError("HVDTBA invalid or duplicate source timestamps")

    prices = frame[["open", "high", "low", "close"]]
    trades = frame.number_of_trades
    frame["primary_row_valid"] = (
        np.isfinite(prices).all(axis=1)
        & prices.gt(0).all(axis=1)
        & frame.high.ge(prices[["open", "close"]].max(axis=1))
        & frame.low.le(prices[["open", "close"]].min(axis=1))
        & frame.high.ge(frame.low)
        & np.isfinite(trades)
        & trades.ge(0)
        & trades.eq(np.floor(trades))
    )
    frame["quote_row_valid"] = (
        frame.primary_row_valid
        & np.isfinite(frame.quote_asset_volume)
        & frame.quote_asset_volume.ge(0)
    )
    return frame.sort_values("ts", kind="mergesort").set_index("ts")


def _invalid_pair(
    breadth_count: int,
    variation_count: int,
    *,
    breadth_valid: bool = False,
    variation_valid: bool = False,
    quote_breadth_valid: bool = False,
) -> dict[str, Any]:
    return {
        "source_valid": False,
        "breadth_valid": breadth_valid,
        "variation_valid": variation_valid,
        "quote_breadth_valid": quote_breadth_valid,
        "breadth_minute_count": breadth_count,
        "variation_minute_count": variation_count,
        "positive_minute_count": 0,
        "negative_minute_count": 0,
        "zero_return_minute_count": 0,
        "up_count": np.nan,
        "down_count": np.nan,
        "directional_trade_breadth": np.nan,
        "up_quote_volume": np.nan,
        "down_quote_volume": np.nan,
        "quote_volume_directional_breadth": np.nan,
        "btc_realized_variation": np.nan,
    }


def boundary_pair(source: pd.DataFrame, decision: pd.Timestamp) -> dict[str, Any]:
    """Compute exact [D-6h,D) directional breadth and [D-24h,D) variation."""
    breadth_index = pd.date_range(
        decision - pd.Timedelta(hours=POLICY["breadth_hours"]),
        decision,
        freq="1min",
        inclusive="left",
    )
    day_index = pd.date_range(
        decision - pd.Timedelta(hours=POLICY["variation_hours"]),
        decision,
        freq="1min",
        inclusive="left",
    )
    day = source.reindex(day_index)
    block = day.reindex(breadth_index)
    day_rows = day.primary_row_valid.eq(True)
    block_rows = block.primary_row_valid.eq(True)
    variation_count = int(day_rows.sum())
    breadth_count = int(block_rows.sum())
    variation_grid_valid = len(day) == 1440 and bool(day_rows.all())
    breadth_grid_valid = len(block) == 360 and bool(block_rows.all())
    if not (variation_grid_valid and breadth_grid_valid):
        return _invalid_pair(breadth_count, variation_count)

    block_returns = np.log(block.close.to_numpy(float) / block.open.to_numpy(float))
    day_returns = np.log(day.close.to_numpy(float) / day.open.to_numpy(float))
    positive = block_returns > 0
    negative = block_returns < 0
    positive_count = int(np.count_nonzero(positive))
    negative_count = int(np.count_nonzero(negative))
    zero_count = int(len(block_returns) - positive_count - negative_count)
    variation = float(np.square(day_returns).sum())
    variation_valid = math.isfinite(variation) and variation > 0

    trades = block.number_of_trades.to_numpy(float)
    up_count = int(trades[positive].sum())
    down_count = int(trades[negative].sum())
    count_denominator = up_count + down_count
    support_valid = (
        positive_count >= POLICY["minimum_minutes_each_direction"]
        and negative_count >= POLICY["minimum_minutes_each_direction"]
    )
    trade_breadth = (
        (up_count - down_count) / count_denominator if count_denominator > 0 else math.nan
    )
    breadth_valid = (
        support_valid
        and count_denominator > 0
        and math.isfinite(trade_breadth)
        and trade_breadth != 0
    )

    quote_rows_valid = bool(block.quote_row_valid.eq(True).all())
    quote = block.quote_asset_volume.to_numpy(float)
    up_quote = float(quote[positive].sum()) if quote_rows_valid else math.nan
    down_quote = float(quote[negative].sum()) if quote_rows_valid else math.nan
    quote_denominator = up_quote + down_quote
    quote_breadth = (
        (up_quote - down_quote) / quote_denominator
        if quote_rows_valid and quote_denominator > 0
        else math.nan
    )
    quote_valid = (
        support_valid
        and quote_rows_valid
        and math.isfinite(quote_breadth)
        and quote_breadth != 0
    )
    return {
        "source_valid": bool(breadth_valid and variation_valid),
        "breadth_valid": bool(breadth_valid),
        "variation_valid": bool(variation_valid),
        "quote_breadth_valid": bool(quote_valid),
        "breadth_minute_count": breadth_count,
        "variation_minute_count": variation_count,
        "positive_minute_count": positive_count,
        "negative_minute_count": negative_count,
        "zero_return_minute_count": zero_count,
        "up_count": up_count,
        "down_count": down_count,
        "directional_trade_breadth": float(trade_breadth),
        "up_quote_volume": up_quote,
        "down_quote_volume": down_quote,
        "quote_volume_directional_breadth": float(quote_breadth),
        "btc_realized_variation": variation,
    }


def build_pair_panel(bars: pd.DataFrame) -> pd.DataFrame:
    source = prepare_source(bars)
    rows = [
        {
            "decision_time": decision,
            "feature_available_time": decision,
            **boundary_pair(source, decision),
        }
        for decision in pd.date_range(
            QUERY_START + pd.Timedelta(hours=24), END, freq="1h", inclusive="left"
        )
    ]
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def build_features(pair: pd.DataFrame) -> pd.DataFrame:
    if pair.columns.tolist() != list(PAIR_COLUMNS):
        raise RuntimeError("HVDTBA pair-panel schema drift")
    features = pair.sort_values("decision_time", kind="mergesort").reset_index(drop=True).copy()
    decisions = pd.to_datetime(features.decision_time, utc=True, errors="coerce")
    if decisions.isna().any() or decisions.duplicated().any() or not decisions.is_monotonic_increasing:
        raise RuntimeError("HVDTBA pair-panel decision order invalid")
    valid = features.source_valid.fillna(False).astype(bool)
    quote_valid = (
        features.quote_breadth_valid.fillna(False).astype(bool)
        & features.variation_valid.fillna(False).astype(bool)
    )
    features["absolute_breadth_rank"] = strict_prior_midrank(
        features.directional_trade_breadth.abs().where(valid),
        POLICY["history_hours"], POLICY["minimum_history_hours"],
    )
    features["absolute_quote_breadth_rank"] = strict_prior_midrank(
        features.quote_volume_directional_breadth.abs().where(quote_valid),
        POLICY["history_hours"], POLICY["minimum_history_hours"],
    )
    features["variation_rank"] = strict_prior_midrank(
        features.btc_realized_variation.where(valid),
        POLICY["history_hours"], POLICY["minimum_history_hours"],
    )
    return features.loc[:, FEATURE_COLUMNS]


def active_and_side(
    features: pd.DataFrame, control: str = "primary"
) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """Return frozen eligibility, source-valid onset, side, and selected geometry."""
    if control not in ("primary", *CONTROLS):
        raise ValueError(f"unknown HVDTBA control: {control}")
    ordered = features.copy()
    used = ordered.copy()
    available = pd.to_datetime(ordered.feature_available_time, utc=True, errors="coerce")
    if control == "one_hour_stale_features":
        used = ordered.shift(1)
        used["decision_time"] = ordered.decision_time
        used["feature_available_time"] = available.shift(1)

    if control == "quote_volume_directional_breadth":
        breadth = pd.to_numeric(used.quote_volume_directional_breadth, errors="coerce")
        breadth_rank = pd.to_numeric(used.absolute_quote_breadth_rank, errors="coerce")
        source_valid = (
            used.quote_breadth_valid.eq(True) & used.variation_valid.eq(True)
        )
    else:
        breadth = pd.to_numeric(used.directional_trade_breadth, errors="coerce")
        breadth_rank = pd.to_numeric(used.absolute_breadth_rank, errors="coerce")
        source_valid = used.source_valid.eq(True)

    breadth_gate = (
        pd.Series(True, index=used.index)
        if control == "no_breadth_magnitude_gate"
        else breadth_rank.ge(POLICY["absolute_breadth_rank_min"])
    )
    variation_gate = (
        pd.Series(True, index=used.index)
        if control == "no_variation_gate"
        else pd.to_numeric(used.variation_rank, errors="coerce").ge(
            POLICY["variation_rank_min"]
        )
    )
    side = pd.Series(np.sign(breadth), index=used.index).fillna(0).astype(int)
    eligible = source_valid & breadth.ne(0) & breadth_gate & variation_gate & side.ne(0)

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
    return eligible, onset, side, used


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    ordered = features.sort_values("decision_time", kind="mergesort").reset_index(drop=True)
    _, onset, sides, used = active_and_side(ordered, control)
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for index in ordered.index[onset]:
        decision = pd.Timestamp(ordered.at[index, "decision_time"])
        if decision.minute != 0 or decision.second != 0 or decision.microsecond != 0:
            raise RuntimeError("HVDTBA decision grid drift")
        entry = decision + pd.Timedelta(minutes=POLICY["entry_delay_minutes"])
        exit_time = entry + pd.Timedelta(hours=POLICY["hold_hours"])
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next(
            (
                name for name, (start, end) in SPLITS.items()
                if entry >= start and exit_time <= end
            ),
            None,
        )
        if split is None:
            continue
        reserved_until = exit_time
        rows.append(
            {
                "candidate": "HVDTBA-6",
                "control": control,
                "split": split,
                "decision_time": decision,
                "feature_available_time": pd.Timestamp(used.at[index, "feature_available_time"]),
                "entry_time": entry,
                "exit_time": exit_time,
                "side": int(sides.at[index]),
                "positive_minute_count": int(used.at[index, "positive_minute_count"]),
                "negative_minute_count": int(used.at[index, "negative_minute_count"]),
                "up_count": int(used.at[index, "up_count"]),
                "down_count": int(used.at[index, "down_count"]),
                "directional_trade_breadth": float(used.at[index, "directional_trade_breadth"]),
                "absolute_breadth_rank": float(used.at[index, "absolute_breadth_rank"]),
                "up_quote_volume": float(used.at[index, "up_quote_volume"]),
                "down_quote_volume": float(used.at[index, "down_quote_volume"]),
                "quote_volume_directional_breadth": float(
                    used.at[index, "quote_volume_directional_breadth"]
                ),
                "absolute_quote_breadth_rank": float(
                    used.at[index, "absolute_quote_breadth_rank"]
                ),
                "btc_realized_variation": float(used.at[index, "btc_realized_variation"]),
                "variation_rank": float(used.at[index, "variation_rank"]),
            }
        )
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def support_stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = clock[clock.split.eq(split)]
    if selected.empty:
        return {
            "events": 0, "longs": 0, "shorts": 0,
            "minority_side_share": 0.0, "max_month_share": 0.0,
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


def deterministic_csv_gzip(frame: pd.DataFrame) -> bytes:
    text = frame.to_csv(index=False, float_format="%.12g", lineterminator="\n").encode()
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=6, mtime=0) as output:
        output.write(text)
    return buffer.getvalue()


def deterministic_json(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()


def write_immutable(path: Path, content: bytes) -> None:
    """Create an artifact once; identical reruns are allowed, drift is rejected."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite immutable HVDTBA artifact: {path}")
        return
    path.write_bytes(content)


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVDTBA preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    if registration != REGISTRATION:
        raise RuntimeError("HVDTBA committed preregistration payload drift")
    if tuple(registration["diagnostic_controls"]["names"]) != CONTROLS:
        raise RuntimeError("HVDTBA diagnostic-control drift")

    bars = load_source()
    pair = build_pair_panel(bars)
    features = build_features(pair)
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}
    split_frames = {name: primary[primary.split.eq(name)].copy() for name in SPLITS}

    write_immutable(PAIR_PANEL, deterministic_csv_gzip(pair))
    write_immutable(FEATURE_PANEL, deterministic_csv_gzip(features))
    write_immutable(CLOCK, deterministic_csv_gzip(primary))
    for name, frame in split_frames.items():
        write_immutable(SPLIT_DIR / f"{name}.csv.gz", deterministic_csv_gzip(frame))
    for name, frame in controls.items():
        write_immutable(CONTROL_DIR / f"{name}.csv.gz", deterministic_csv_gzip(frame))

    source_core = {
        "protocol_version": "hvdtba_6_btc_directional_trade_breadth_source_v1",
        "query": QUERY,
        "query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(),
        "table": "bars_binance",
        "symbol": "BTCUSDT",
        "interval": "1m",
        "columns": [
            "ts", "open", "high", "low", "close", "number_of_trades",
            "quote_asset_volume",
        ],
        "window": [QUERY_START.isoformat(), END.isoformat()],
        "physical_rows": len(bars),
        "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
        "pair_panel": {"path": str(PAIR_PANEL), "sha256": sha(PAIR_PANEL), "rows": len(pair)},
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
        "protocol_version": "hvdtba_6_source_support_v1",
        "policy_id": "HVDTBA-6",
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
        "split_artifacts": {
            name: {
                "path": str(SPLIT_DIR / f"{name}.csv.gz"),
                "sha256": sha(SPLIT_DIR / f"{name}.csv.gz"), "rows": len(frame),
            }
            for name, frame in split_frames.items()
        },
        "reservation": {
            "scope": "global", "hours": POLICY["hold_hours"], "interval": "half_open",
            "equal_open_after_exit_allowed": True, "split_crossing_action": "skip",
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
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
