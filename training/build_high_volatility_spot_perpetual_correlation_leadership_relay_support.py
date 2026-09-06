"""Build outcome-blind source support for frozen HVSPCL-8."""
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

from training import preregister_high_volatility_spot_perpetual_correlation_leadership_relay as prereg


ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path("training/build_high_volatility_spot_perpetual_correlation_leadership_relay_support.py")
PREREG_SHA = "780443153cd5587dee3faaf99354710edfaba7e789296d9dde9d12a69068d756"
QUERY_START = pd.Timestamp("2023-04-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
REGISTRATION = prereg.build()
POLICY = REGISTRATION["policy"]
VENUES = ("spot", "perpetual")
SPLITS = {
    name: (pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1]))
    for name, bounds in REGISTRATION["stages"].items()
}
SUPPORT_GATES = REGISTRATION["source_support_gates"]
MINIMUM_EVENTS = SUPPORT_GATES["minimum_events"]
CONTROLS = tuple(REGISTRATION["diagnostic_controls"]["names"])

QUERY = """SELECT ts,'spot' AS venue,open,high,low,close
FROM bars_binance_spot
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
UNION ALL
SELECT ts,'perpetual' AS venue,open,high,low,close
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts,venue"""

SOURCE_DIR = Path("data/high_volatility_spot_perpetual_correlation_leadership_relay_sources_2023_2026")
PAIR_PANEL = SOURCE_DIR / "hourly_leadership_pairs.csv.gz"
FEATURE_PANEL = SOURCE_DIR / "hourly_preentry_features.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_spot_perpetual_correlation_leadership_relay_clocks_2023_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_spot_perpetual_correlation_leadership_relay_split_clocks_2023_2026")
CONTROL_DIR = Path("data/high_volatility_spot_perpetual_correlation_leadership_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_spot_perpetual_correlation_leadership_relay_support_2026-08-13.json")

PAIR_COLUMNS = (
    "decision_time", "feature_available_time", "source_valid", "minute_count",
    "spot_leads_perpetual", "perpetual_leads_spot", "leadership_advantage",
    "same_minute_correlation", "perpetual_variation", "perpetual_final_hour_return",
    "spot_final_hour_return", "direction_side",
)
FEATURE_COLUMNS = (
    *PAIR_COLUMNS, "leadership_rank", "same_minute_correlation_rank",
    "perpetual_variation_rank",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "spot_leads_perpetual", "perpetual_leads_spot",
    "leadership_advantage", "leadership_rank", "same_minute_correlation",
    "same_minute_correlation_rank", "perpetual_variation", "perpetual_variation_rank",
    "perpetual_final_hour_return", "spot_final_hour_return", "direction_side",
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
    required = ["ts", "venue", "open", "high", "low", "close"]
    if bars.columns.tolist() != required:
        raise RuntimeError("HVSPCL source schema drift")
    frame = bars.copy()
    frame["ts"] = pd.to_datetime(frame.ts, utc=True, errors="coerce")
    frame["venue"] = frame.venue.astype(str)
    for column in required[2:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.ts.isna().any() or frame.duplicated(["ts", "venue"]).any():
        raise RuntimeError("HVSPCL invalid or duplicate source key")
    if not frame.venue.isin(VENUES).all():
        raise RuntimeError("HVSPCL unexpected source venue")
    prices = frame[["open", "high", "low", "close"]]
    frame["row_valid"] = (
        np.isfinite(prices).all(axis=1)
        & prices.gt(0).all(axis=1)
        & frame.high.ge(prices[["open", "close"]].max(axis=1))
        & frame.low.le(prices[["open", "close"]].min(axis=1))
        & frame.high.ge(frame.low)
    )
    return frame.sort_values(["ts", "venue"], kind="mergesort").set_index(["ts", "venue"])


def _invalid_pair(minute_count: int) -> dict[str, Any]:
    return {
        "source_valid": False, "minute_count": minute_count,
        "spot_leads_perpetual": np.nan, "perpetual_leads_spot": np.nan,
        "leadership_advantage": np.nan, "same_minute_correlation": np.nan,
        "perpetual_variation": np.nan, "perpetual_final_hour_return": np.nan,
        "spot_final_hour_return": np.nan, "direction_side": 0,
    }


def _finite_positive_variance(*vectors: np.ndarray) -> bool:
    return all(
        np.isfinite(vector).all() and float(np.var(vector, ddof=0)) > 0
        for vector in vectors
    )


def boundary_pair(source: pd.DataFrame, decision: pd.Timestamp) -> dict[str, Any]:
    """Compute exact aligned [D-6h,D) lead-lag geometry and return sums."""
    minutes = pd.date_range(
        decision - pd.Timedelta(minutes=POLICY["window_minutes"]),
        decision, freq="1min", inclusive="left",
    )
    expected = pd.MultiIndex.from_product([minutes, VENUES], names=["ts", "venue"])
    block = source.reindex(expected)
    valid_rows = block.row_valid.eq(True)
    minute_count = int(valid_rows.sum())
    if len(minutes) != 360 or len(block) != 360 * len(VENUES) or not bool(valid_rows.all()):
        return _invalid_pair(minute_count)

    returns = {
        venue: np.log(
            block.xs(venue, level="venue").close.to_numpy(float)
            / block.xs(venue, level="venue").open.to_numpy(float)
        )
        for venue in VENUES
    }
    spot = returns["spot"]
    perpetual = returns["perpetual"]
    spot_now, perpetual_next = spot[:-1], perpetual[1:]
    perpetual_now, spot_next = perpetual[:-1], spot[1:]
    if not (
        _finite_positive_variance(spot_now, perpetual_next)
        and _finite_positive_variance(perpetual_now, spot_next)
        and _finite_positive_variance(spot, perpetual)
    ):
        return _invalid_pair(minute_count)

    spot_leads_perpetual = float(np.corrcoef(spot_now, perpetual_next)[0, 1])
    perpetual_leads_spot = float(np.corrcoef(perpetual_now, spot_next)[0, 1])
    contemporaneous = float(np.corrcoef(spot, perpetual)[0, 1])
    perpetual_variation = float(np.sqrt(np.square(perpetual).sum()))
    perpetual_final_hour = float(perpetual[-60:].sum())
    spot_final_hour = float(spot[-60:].sum())
    leadership_advantage = spot_leads_perpetual - perpetual_leads_spot
    metrics = np.asarray([
        spot_leads_perpetual, perpetual_leads_spot, contemporaneous, perpetual_variation,
        perpetual_final_hour, spot_final_hour, leadership_advantage,
    ])
    if not np.isfinite(metrics).all() or perpetual_variation <= 0 or leadership_advantage <= 0:
        return _invalid_pair(minute_count)
    same_sign = (
        int(np.sign(perpetual_final_hour))
        if perpetual_final_hour != 0 and np.sign(perpetual_final_hour) == np.sign(spot_final_hour)
        else 0
    )
    return {
        "source_valid": True, "minute_count": minute_count,
        "spot_leads_perpetual": spot_leads_perpetual, "perpetual_leads_spot": perpetual_leads_spot,
        "leadership_advantage": leadership_advantage,
        "same_minute_correlation": contemporaneous,
        "perpetual_variation": perpetual_variation,
        "perpetual_final_hour_return": perpetual_final_hour,
        "spot_final_hour_return": spot_final_hour, "direction_side": same_sign,
    }


def build_pair_panel(bars: pd.DataFrame) -> pd.DataFrame:
    source = prepare_source(bars)
    rows = [
        {"decision_time": decision, "feature_available_time": decision,
         **boundary_pair(source, decision)}
        for decision in pd.date_range(
            QUERY_START + pd.Timedelta(minutes=POLICY["window_minutes"]),
            END, freq="1h", inclusive="left",
        )
    ]
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def build_features(pair: pd.DataFrame) -> pd.DataFrame:
    if pair.columns.tolist() != list(PAIR_COLUMNS):
        raise RuntimeError("HVSPCL pair-panel schema drift")
    features = pair.sort_values("decision_time", kind="mergesort").reset_index(drop=True).copy()
    decisions = pd.to_datetime(features.decision_time, utc=True, errors="coerce")
    if decisions.isna().any() or decisions.duplicated().any() or not decisions.is_monotonic_increasing:
        raise RuntimeError("HVSPCL pair-panel decision order invalid")
    valid = features.source_valid.fillna(False).astype(bool)
    rank_args = (POLICY["history_hours"], POLICY["minimum_history_hours"])
    features["leadership_rank"] = strict_prior_midrank(
        features.leadership_advantage.where(valid), *rank_args
    )
    features["same_minute_correlation_rank"] = strict_prior_midrank(
        features.same_minute_correlation.where(valid), *rank_args
    )
    features["perpetual_variation_rank"] = strict_prior_midrank(
        features.perpetual_variation.where(valid), *rank_args
    )
    return features.loc[:, FEATURE_COLUMNS]


def active_and_side(
    features: pd.DataFrame, control: str = "primary"
) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """Return frozen eligibility, exact source-valid onset, side, and used geometry."""
    if control not in ("primary", *CONTROLS):
        raise ValueError(f"unknown HVSPCL control: {control}")
    ordered = features.copy()
    used = ordered.copy()
    if control == "one_hour_stale_leadership":
        used = ordered.shift(1)
        used["decision_time"] = ordered.decision_time
        used["feature_available_time"] = pd.to_datetime(
            ordered.feature_available_time, utc=True, errors="coerce"
        ).shift(1)

    source_valid = used.source_valid.eq(True)
    side = pd.to_numeric(used.direction_side, errors="coerce").fillna(0).astype(int)
    direction_gate = side.ne(0)
    if control == "no_leadership_tail":
        leadership_gate = pd.Series(True, index=used.index)
    elif control == "same_minute_correlation":
        leadership_gate = pd.to_numeric(
            used.same_minute_correlation_rank, errors="coerce"
        ).ge(POLICY["leadership_rank_min"])
    else:
        leadership_gate = pd.to_numeric(used.leadership_rank, errors="coerce").ge(
            POLICY["leadership_rank_min"]
        )
    perpetual_variation_gate = (
        pd.Series(True, index=used.index)
        if control == "no_variation_gate"
        else pd.to_numeric(used.perpetual_variation_rank, errors="coerce").ge(
            POLICY["variation_rank_min"]
        )
    )
    eligible = source_valid & leadership_gate & perpetual_variation_gate & direction_gate
    decisions = pd.to_datetime(ordered.decision_time, utc=True, errors="coerce")
    adjacent = decisions.shift(1).add(pd.Timedelta(hours=1)).eq(decisions)
    onset = (
        eligible & adjacent & source_valid.shift(1, fill_value=False)
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
            raise RuntimeError("HVSPCL decision grid drift")
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
            raise RuntimeError(f"refusing to overwrite immutable HVSPCL artifact: {path}")
        return
    path.write_bytes(content)


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVSPCL preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    if registration != REGISTRATION:
        raise RuntimeError("HVSPCL committed preregistration payload drift")
    if tuple(registration["diagnostic_controls"]["names"]) != CONTROLS:
        raise RuntimeError("HVSPCL diagnostic-control drift")

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
        "protocol_version": "hvspcl_8_spot_perpetual_correlation_leadership_source_v1",
        "query": QUERY, "query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(),
        "tables": ["bars_binance_spot", "bars_binance"], "venues": list(VENUES), "interval": "1m",
        "symbol": "BTCUSDT", "columns": ["ts", "venue", "open", "high", "low", "close"],
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
        "protocol_version": "hvspcl_8_source_support_v1", "policy_id": prereg.POLICY_ID,
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
