"""Build outcome-blind source support for frozen HVCACFR-8."""
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

from training import preregister_high_volatility_cross_alt_correlation_fracture_resolution_relay as prereg


ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path("training/build_high_volatility_cross_alt_correlation_fracture_resolution_relay_support.py")
PREREG_SHA = "65032a3cbc8c0ac43fe73ce283970f2166d604d718030de67c263a7628dd08bc"
QUERY_START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
REGISTRATION = prereg.build()
POLICY = REGISTRATION["policy"]
SYMBOLS = tuple(REGISTRATION["features"]["universe"])
ALTS = tuple(prereg.ALTS)
SPLITS = {
    name: (pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1]))
    for name, bounds in REGISTRATION["stages"].items()
}
SUPPORT_GATES = REGISTRATION["source_support_gates"]
MINIMUM_EVENTS = SUPPORT_GATES["minimum_events"]
CONTROLS = tuple(REGISTRATION["diagnostic_controls"]["names"])

QUERY = """SELECT ts,symbol,open,high,low,close
FROM bars_binance
WHERE symbol IN ('BTCUSDT','ADAUSDT','BNBUSDT','DOGEUSDT','ETHUSDT','SOLUSDT','XRPUSDT')
AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts,symbol"""

SOURCE_DIR = Path("data/high_volatility_cross_alt_correlation_fracture_resolution_relay_sources_2023_2026")
PAIR_PANEL = SOURCE_DIR / "daily_correlation_pairs.csv.gz"
FEATURE_PANEL = SOURCE_DIR / "daily_preentry_features.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_cross_alt_correlation_fracture_resolution_relay_clocks_2023_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_cross_alt_correlation_fracture_resolution_relay_split_clocks_2023_2026")
CONTROL_DIR = Path("data/high_volatility_cross_alt_correlation_fracture_resolution_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_cross_alt_correlation_fracture_resolution_relay_support_2026-08-10.json")

PAIR_COLUMNS = (
    "decision_time", "feature_available_time", "source_valid", "minute_count",
    "first_half_correlation", "second_half_correlation", "correlation_fracture",
    "full_window_correlation", "btc_variation", "final_120_alt_return",
    "direction_side",
)
FEATURE_COLUMNS = (
    *PAIR_COLUMNS, "correlation_fracture_rank", "full_window_correlation_rank",
    "btc_variation_rank",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "first_half_correlation",
    "second_half_correlation", "correlation_fracture", "correlation_fracture_rank",
    "full_window_correlation", "full_window_correlation_rank", "btc_variation",
    "btc_variation_rank", "final_120_alt_return", "direction_side",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def strict_prior_midrank(
    values: pd.Series, lookback: int = 180, minimum: int = 90
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
    """Read only the preregistered completed one-minute OHLC source fields."""
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
    required = ["ts", "symbol", "open", "high", "low", "close"]
    if bars.columns.tolist() != required:
        raise RuntimeError("HVCACFR source schema drift")
    frame = bars.copy()
    frame["ts"] = pd.to_datetime(frame.ts, utc=True, errors="coerce")
    frame["symbol"] = frame.symbol.astype(str)
    for column in required[2:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.ts.isna().any() or frame.duplicated(["ts", "symbol"]).any():
        raise RuntimeError("HVCACFR invalid or duplicate source key")
    if not frame.symbol.isin(SYMBOLS).all():
        raise RuntimeError("HVCACFR unexpected source symbol")
    prices = frame[["open", "high", "low", "close"]]
    frame["row_valid"] = (
        np.isfinite(prices).all(axis=1)
        & prices.gt(0).all(axis=1)
        & frame.high.ge(prices[["open", "close"]].max(axis=1))
        & frame.low.le(prices[["open", "close"]].min(axis=1))
        & frame.high.ge(frame.low)
    )
    return frame.sort_values(["ts", "symbol"], kind="mergesort").set_index(["ts", "symbol"])


def _invalid_pair(minute_count: int) -> dict[str, Any]:
    return {
        "source_valid": False, "minute_count": minute_count,
        "first_half_correlation": np.nan, "second_half_correlation": np.nan,
        "correlation_fracture": np.nan, "full_window_correlation": np.nan,
        "btc_variation": np.nan, "final_120_alt_return": np.nan,
        "direction_side": 0,
    }


def _finite_positive_variance(*vectors: np.ndarray) -> bool:
    return all(
        np.isfinite(vector).all() and float(np.var(vector, ddof=0)) > 0
        for vector in vectors
    )


def population_pearson(first: np.ndarray, second: np.ndarray) -> float:
    """Return Pearson covariance divided by population standard deviations."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape or not _finite_positive_variance(first, second):
        return float("nan")
    centered_first = first - first.mean()
    centered_second = second - second.mean()
    covariance = float(np.mean(centered_first * centered_second))
    denominator = math.sqrt(
        float(np.mean(np.square(centered_first)))
        * float(np.mean(np.square(centered_second)))
    )
    return covariance / denominator


def boundary_pair(source: pd.DataFrame, decision: pd.Timestamp) -> dict[str, Any]:
    """Compute the exact aligned twelve-hour correlation-fracture geometry."""
    minutes = pd.date_range(
        decision - pd.Timedelta(minutes=POLICY["window_minutes"]),
        decision, freq="1min", inclusive="left",
    )
    expected = pd.MultiIndex.from_product([minutes, SYMBOLS], names=["ts", "symbol"])
    block = source.reindex(expected)
    valid_rows = block.row_valid.eq(True)
    minute_count = int(valid_rows.sum())
    if (
        len(minutes) != POLICY["window_minutes"]
        or len(block) != POLICY["window_minutes"] * len(SYMBOLS)
        or not bool(valid_rows.all())
    ):
        return _invalid_pair(minute_count)

    returns = {
        symbol: np.log(
            block.xs(symbol, level="symbol").close.to_numpy(float)
            / block.xs(symbol, level="symbol").open.to_numpy(float)
        )
        for symbol in SYMBOLS
    }
    btc = returns["BTCUSDT"]
    alt = np.median(np.column_stack([returns[symbol] for symbol in ALTS]), axis=1)
    half = POLICY["half_window_minutes"]
    first = population_pearson(btc[:half], alt[:half])
    second = population_pearson(btc[half:], alt[half:])
    full = population_pearson(btc, alt)
    if not np.isfinite([first, second, full]).all():
        return _invalid_pair(minute_count)

    btc_variation = float(np.square(btc).sum())
    final_alt = float(alt[-POLICY["direction_window_minutes"]:].sum())
    if not math.isfinite(btc_variation) or not math.isfinite(final_alt):
        return _invalid_pair(minute_count)
    return {
        "source_valid": True, "minute_count": minute_count,
        "first_half_correlation": first, "second_half_correlation": second,
        "correlation_fracture": first - second,
        "full_window_correlation": full, "btc_variation": btc_variation,
        "final_120_alt_return": final_alt,
        "direction_side": int(np.sign(final_alt)),
    }


def build_pair_panel(bars: pd.DataFrame) -> pd.DataFrame:
    source = prepare_source(bars)
    first_decision = QUERY_START.normalize() + pd.Timedelta(days=1, hours=3)
    rows = [
        {"decision_time": decision, "feature_available_time": decision,
        **boundary_pair(source, decision)}
        for decision in pd.date_range(
            first_decision, END, freq="1D", inclusive="left",
        )
    ]
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def build_features(pair: pd.DataFrame) -> pd.DataFrame:
    if pair.columns.tolist() != list(PAIR_COLUMNS):
        raise RuntimeError("HVCACFR pair-panel schema drift")
    features = pair.sort_values("decision_time", kind="mergesort").reset_index(drop=True).copy()
    decisions = pd.to_datetime(features.decision_time, utc=True, errors="coerce")
    if decisions.isna().any() or decisions.duplicated().any() or not decisions.is_monotonic_increasing:
        raise RuntimeError("HVCACFR pair-panel decision order invalid")
    if not bool(
        (
            decisions.dt.hour.eq(3)
            & decisions.dt.minute.eq(0)
            & decisions.dt.second.eq(0)
            & decisions.dt.microsecond.eq(0)
        ).all()
    ):
        raise RuntimeError("HVCACFR pair-panel decision grid invalid")
    valid = features.source_valid.fillna(False).astype(bool)
    rank_args = (POLICY["prior_valid_days"], POLICY["minimum_prior_valid_days"])
    features["correlation_fracture_rank"] = strict_prior_midrank(
        features.correlation_fracture.where(valid), *rank_args
    )
    features["full_window_correlation_rank"] = strict_prior_midrank(
        features.full_window_correlation.where(valid), *rank_args
    )
    features["btc_variation_rank"] = strict_prior_midrank(
        features.btc_variation.where(valid), *rank_args
    )
    return features.loc[:, FEATURE_COLUMNS]


def _previous_source_valid_features(ordered: pd.DataFrame) -> pd.DataFrame:
    """Map each decision to the complete immediately previous source-valid geometry."""
    rows: list[dict[str, Any]] = []
    previous: int | None = None
    valid = ordered.source_valid.eq(True)
    for index in ordered.index:
        rows.append(
            ({column: pd.NA for column in ordered.columns}
             if previous is None else ordered.loc[previous].to_dict())
        )
        if bool(valid.at[index]):
            previous = int(index)
    used = pd.DataFrame(rows, index=ordered.index, columns=ordered.columns)
    used["decision_time"] = ordered.decision_time
    return used


def _source_valid_onset(eligible: pd.Series, source_valid: pd.Series) -> pd.Series:
    """Trigger only after the immediately previous source-valid decision was ineligible."""
    onset = pd.Series(False, index=eligible.index, dtype=bool)
    previous_valid: int | None = None
    for index in eligible.index:
        if not bool(source_valid.at[index]):
            continue
        if bool(eligible.at[index]) and previous_valid is not None:
            onset.at[index] = not bool(eligible.at[previous_valid])
        previous_valid = int(index)
    return onset


def active_and_side(
    features: pd.DataFrame, control: str = "primary"
) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """Return frozen eligibility, exact source-valid onset, side, and used geometry."""
    if control not in ("primary", *CONTROLS):
        raise ValueError(f"unknown HVCACFR control: {control}")
    ordered = features.copy()
    used = ordered.copy()
    if control == "one_day_stale_features":
        used = _previous_source_valid_features(ordered)

    source_valid = used.source_valid.eq(True)
    side = pd.to_numeric(used.direction_side, errors="coerce").fillna(0).astype(int)
    direction_gate = side.ne(0)
    if control == "no_correlation_fracture_gate":
        fracture_gate = pd.Series(True, index=used.index)
    elif control == "contemporaneous_full_window_correlation":
        fracture_gate = pd.to_numeric(
            used.full_window_correlation_rank, errors="coerce"
        ).ge(POLICY["correlation_fracture_rank_min"])
    else:
        fracture_gate = pd.to_numeric(
            used.correlation_fracture_rank, errors="coerce"
        ).ge(
            POLICY["correlation_fracture_rank_min"]
        )
    btc_variation_gate = (
        pd.Series(True, index=used.index)
        if control == "no_btc_variation_gate"
        else pd.to_numeric(used.btc_variation_rank, errors="coerce").ge(
            POLICY["btc_variation_rank_min"]
        )
    )
    eligible = source_valid & fracture_gate & btc_variation_gate & direction_gate
    onset = _source_valid_onset(eligible, source_valid)
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
        if (
            decision.hour != 3 or decision.minute != 0
            or decision.second != 0 or decision.microsecond != 0
        ):
            raise RuntimeError("HVCACFR decision grid drift")
        entry = decision + pd.Timedelta(minutes=POLICY["entry_delay_minutes"])
        exit_time = entry + pd.Timedelta(hours=POLICY["hold_hours"])
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
            "candidate": prereg.POLICY_ID, "control": control, "split": split,
            "decision_time": decision,
            "feature_available_time": pd.Timestamp(used.at[index, "feature_available_time"]),
            "entry_time": entry, "exit_time": exit_time, "side": int(sides.at[index]),
            **{
                column: float(used.at[index, column])
                for column in CLOCK_COLUMNS[8:-1]
            },
            "direction_side": int(used.at[index, "direction_side"]),
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def support_stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = clock[clock.split.eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(selected.side.eq(1).sum())
    shorts = int(selected.side.eq(-1).sum())
    months = pd.to_datetime(selected.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(selected), "longs": longs, "shorts": shorts,
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
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite immutable HVCACFR artifact: {path}")
        return
    path.write_bytes(content)


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVCACFR preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    if registration != REGISTRATION:
        raise RuntimeError("HVCACFR committed preregistration payload drift")
    if tuple(registration["diagnostic_controls"]["names"]) != CONTROLS:
        raise RuntimeError("HVCACFR diagnostic-control drift")

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
        "protocol_version": "hvcacfr_8_cross_alt_correlation_fracture_resolution_source_v1",
        "query": QUERY, "query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(),
        "table": "bars_binance", "symbols": list(SYMBOLS), "interval": "1m",
        "columns": ["ts", "symbol", "open", "high", "low", "close"],
        "window": [QUERY_START.isoformat(), END.isoformat()], "physical_rows": len(bars),
        "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
        "pair_panel": {"path": str(PAIR_PANEL), "sha256": sha(PAIR_PANEL), "rows": len(pair)},
        "feature_panel": {
            "path": str(FEATURE_PANEL), "sha256": sha(FEATURE_PANEL),
            "rows": len(features), "valid_rows": int(features.source_valid.sum()),
        },
        "completed_preentry_sources_opened": True, "candidate_incidence_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "funding_values_opened": False, "gross9_rows_opened": False,
        "no_imputation": True, "deterministic_immutable_artifacts": True,
    }
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    write_immutable(SOURCE_MANIFEST, deterministic_json(source_manifest))

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
        "protocol_version": "hvcacfr_8_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA,
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifest": {
            "path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST),
            "manifest_hash": source_manifest["manifest_hash"],
        },
        "ranking": {
            "lookback_valid_decisions": POLICY["prior_valid_days"],
            "minimum_prior_valid_decisions": POLICY["minimum_prior_valid_days"],
            "current_excluded": True, "ties": "midrank",
        },
        "completed_preentry_sources_opened": True, "candidate_incidence_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "funding_values_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "split_artifacts": {
            name: {"path": str(SPLIT_DIR / f"{name}.csv.gz"),
                   "sha256": sha(SPLIT_DIR / f"{name}.csv.gz"), "rows": len(frame)}
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
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
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
