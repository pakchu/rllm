"""Materialize source data and build outcome-blind support for PVIAR-6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.live_db_features import postgres_url_from_env
from training import preregister_premium_volatility_ignition_acceleration_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


START = pd.Timestamp("2023-06-20T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
ENV_FILE = "/home/pakchu/rllm/.env"
NONPRICE_DIR = Path("data/options_crowding_deleveraging_relay_sources_v4_2023_2026")
PREMIUM_DIR = Path("data/premium_volatility_ignition_sources_2023_2026")
PREMIUM_PATH = PREMIUM_DIR / "premium_1m.csv.gz"
PREMIUM_MANIFEST = PREMIUM_DIR / "manifest.json"
CLOCK = Path("data/premium_volatility_ignition_acceleration_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/premium_volatility_ignition_acceleration_relay_controls_2023_2026")
RESULT = Path("results/premium_volatility_ignition_acceleration_relay_support_2026-08-08.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_joint_expansion",
    "no_first_half_tail",
    "no_same_direction",
    "no_acceleration",
    "direction_flip",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time",
    "entry_time", "exit_time", "side", "bvol_body", "dvol_body",
    "first_half_move", "second_half_move", "prior_abs_first_half_q60",
    "acceleration_ratio",
)
QUERY_CONTRACT = """
SELECT ts, open, high, low, close, close_time
FROM bars_binance_premium
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts
""".strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def query_premium(env_file: str = ENV_FILE) -> pd.DataFrame:
    """Read only the preregistered completed premium source columns."""
    from sqlalchemy import create_engine, text

    engine = create_engine(
        postgres_url_from_env(env_file), connect_args={"connect_timeout": 10}
    )
    try:
        with engine.connect() as connection:
            frame = pd.read_sql_query(
                text(QUERY_CONTRACT),
                connection,
                params={"start": START.to_pydatetime(), "end": END.to_pydatetime()},
            )
    finally:
        engine.dispose()
    return normalize_raw_premium(frame)


def normalize_raw_premium(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["ts", "open", "high", "low", "close", "close_time"]
    if list(frame.columns) != required:
        raise ValueError(f"premium schema drift: {list(frame.columns)}")
    result = frame.copy()
    result["ts"] = pd.to_datetime(result["ts"], utc=True, format="mixed")
    if result["ts"].isna().any() or result["ts"].duplicated().any():
        raise ValueError("premium timestamps must be non-null and unique")
    result = result.sort_values("ts").reset_index(drop=True)
    if not result["ts"].is_monotonic_increasing:
        raise ValueError("premium timestamps must increase")
    if ((result["ts"] < START) | (result["ts"] >= END)).any():
        raise ValueError("premium timestamp outside frozen window")
    for column in ("open", "high", "low", "close"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["row_valid"] = (
        result["ts"].dt.second.eq(0)
        & result["ts"].dt.microsecond.eq(0)
        & np.isfinite(result[["open", "high", "low", "close"]]).all(axis=1)
    )
    return result


def hourly_premium_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate exact minute offsets without filling or imputing missing rows."""
    raw = raw.copy()
    raw["hour_start"] = raw["ts"].dt.floor("h")
    raw["minute_offset"] = ((raw["ts"] - raw["hour_start"]) / pd.Timedelta(minutes=1)).astype(int)
    grouped = raw.groupby("hour_start", sort=True)
    records: list[dict[str, Any]] = []
    expected_offsets = set(range(60))
    for hour_start, group in grouped:
        offsets = set(group["minute_offset"].tolist())
        valid = (
            len(group) == 60
            and offsets == expected_offsets
            and bool(group["row_valid"].all())
        )
        by_offset = group.set_index("minute_offset")
        records.append({
            "hour_start": hour_start,
            "source_rows": len(group),
            "distinct_timestamps": int(group["ts"].nunique()),
            "premium_valid": valid,
            "hour_open": float(by_offset.at[0, "open"]) if valid else np.nan,
            "first_half_close": float(by_offset.at[29, "close"]) if valid else np.nan,
            "second_half_open": float(by_offset.at[30, "open"]) if valid else np.nan,
            "hour_close": float(by_offset.at[59, "close"]) if valid else np.nan,
        })
    observed = pd.DataFrame.from_records(records)
    grid = pd.DataFrame({
        "hour_start": pd.date_range(START, END, freq="1h", inclusive="left")
    })
    hourly = grid.merge(observed, on="hour_start", how="left", validate="one_to_one")
    hourly[["source_rows", "distinct_timestamps"]] = hourly[
        ["source_rows", "distinct_timestamps"]
    ].fillna(0).astype(int)
    hourly["premium_valid"] = hourly["premium_valid"].fillna(False).astype(bool)
    hourly["decision_time"] = hourly["hour_start"] + pd.Timedelta(hours=1)
    hourly["first_half_move"] = hourly["first_half_close"] - hourly["hour_open"]
    hourly["second_half_move"] = hourly["hour_close"] - hourly["second_half_open"]
    hourly["prior_abs_first_half_q60"] = (
        hourly["first_half_move"].abs().where(hourly["premium_valid"])
        .shift(1).rolling(720, min_periods=672).quantile(0.60)
    )
    return hourly


def load_volatility() -> pd.DataFrame:
    bvol = pd.read_csv(NONPRICE_DIR / "bvol_hourly.csv.gz", compression="gzip")
    dvol = pd.read_csv(NONPRICE_DIR / "dvol_hourly.csv.gz", compression="gzip")
    b = pd.DataFrame({
        "decision_time": pd.to_datetime(bvol["feature_available_time_utc"], utc=True, format="mixed"),
        "bvol_open": pd.to_numeric(bvol["open"], errors="coerce"),
        "bvol_close": pd.to_numeric(bvol["close"], errors="coerce"),
        "bvol_valid": bvol["feature_valid"].astype(str).str.lower().eq("true"),
    })
    d = pd.DataFrame({
        "decision_time": pd.to_datetime(dvol["close_time"], utc=True, format="mixed"),
        "dvol_open": pd.to_numeric(dvol["open"], errors="coerce"),
        "dvol_close": pd.to_numeric(dvol["close"], errors="coerce"),
    })
    result = b.merge(d, on="decision_time", how="inner", validate="one_to_one")
    numeric = ["bvol_open", "bvol_close", "dvol_open", "dvol_close"]
    result["vol_valid"] = (
        result["bvol_valid"]
        & np.isfinite(result[numeric]).all(axis=1)
        & result[numeric].gt(0).all(axis=1)
    )
    result["bvol_body"] = result["bvol_close"] / result["bvol_open"] - 1.0
    result["dvol_body"] = result["dvol_close"] / result["dvol_open"] - 1.0
    return result


def joined_features(raw: pd.DataFrame) -> pd.DataFrame:
    premium = hourly_premium_features(raw)
    frame = premium.merge(load_volatility(), on="decision_time", validate="one_to_one")
    moves = ["first_half_move", "second_half_move"]
    frame["signal_valid"] = (
        frame["premium_valid"] & frame["vol_valid"]
        & np.isfinite(frame[moves]).all(axis=1)
        & frame[moves].ne(0).all(axis=1)
    )
    return frame.sort_values("decision_time").reset_index(drop=True)


def conditions(frame: pd.DataFrame, control: str) -> tuple[pd.Series, pd.Series]:
    volatility = (
        pd.Series(True, index=frame.index)
        if control == "no_joint_expansion"
        else frame["bvol_body"].gt(0) & frame["dvol_body"].gt(0)
    )
    shock = frame["first_half_move"].ne(0)
    if control != "no_first_half_tail":
        shock &= (
            frame["prior_abs_first_half_q60"].notna()
            & frame["first_half_move"].abs().ge(frame["prior_abs_first_half_q60"])
        )
    same_direction = frame["second_half_move"].ne(0)
    if control != "no_same_direction":
        same_direction &= np.sign(frame["second_half_move"]).eq(
            np.sign(frame["first_half_move"])
        )
    ratio = frame["second_half_move"].abs() / frame["first_half_move"].abs()
    acceleration = (
        pd.Series(True, index=frame.index)
        if control == "no_acceleration" else ratio.ge(1.0)
    )
    active = frame["signal_valid"] & volatility & shock & same_direction & acceleration
    onset = (
        active & ~active.shift(1, fill_value=False)
        & frame["signal_valid"].shift(1, fill_value=False)
        & frame["decision_time"].diff().eq(pd.Timedelta(hours=1))
    )
    return onset, ratio


def build_clock(frame: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    onset, ratio = conditions(frame, control)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in frame.index[onset]:
        decision = pd.Timestamp(frame.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=6)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next((
            name for name, (start, end) in SPLITS.items()
            if entry >= start and exit_time <= end
        ), None)
        if split is None:
            continue
        side = int(np.sign(frame.at[index, "second_half_move"]))
        if control == "direction_flip":
            side *= -1
        next_allowed = exit_time
        rows.append({
            "candidate": "PVIAR-6", "control": control, "split": split,
            "decision_time": decision, "feature_available_time": decision,
            "entry_time": entry, "exit_time": exit_time, "side": side,
            "bvol_body": float(frame.at[index, "bvol_body"]),
            "dvol_body": float(frame.at[index, "dvol_body"]),
            "first_half_move": float(frame.at[index, "first_half_move"]),
            "second_half_move": float(frame.at[index, "second_half_move"]),
            "prior_abs_first_half_q60": float(frame.at[index, "prior_abs_first_half_q60"]),
            "acceleration_ratio": float(ratio.at[index]),
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def split_stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = clock[clock["split"].eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(subset["side"].eq(1).sum())
    shorts = int(subset["side"].eq(-1).sum())
    months = subset["entry_time"].dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(subset), "longs": longs, "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(subset),
        "max_month_share": int(months.max()) / len(subset),
    }


def materialize_source(raw: pd.DataFrame) -> dict[str, Any]:
    PREMIUM_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(raw, PREMIUM_PATH)
    expected_minutes = len(pd.date_range(START, END, freq="1min", inclusive="left"))
    core = {
        "protocol_version": "pviar_6_premium_source_snapshot_v1",
        "query_contract": QUERY_CONTRACT,
        "table": "bars_binance_premium", "symbol": "BTCUSDT", "interval": "1m",
        "window": [START.isoformat(), END.isoformat()],
        "columns": ["ts", "open", "high", "low", "close", "close_time"],
        "outcomes_opened": False, "gross9_rows_opened": False,
        "no_imputation": True,
        "output": {
            "path": str(PREMIUM_PATH), "sha256": sha256(PREMIUM_PATH),
            "rows": len(raw), "expected_minutes": expected_minutes,
            "missing_minutes": expected_minutes - len(raw),
            "valid_rows": int(raw["row_valid"].sum()),
        },
    }
    report = {**core, "manifest_hash": canonical_hash(core)}
    PREMIUM_MANIFEST.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def run(env_file: str = ENV_FILE) -> dict[str, Any]:
    raw = query_premium(env_file)
    source_manifest = materialize_source(raw)
    frame = joined_features(raw)
    primary = build_clock(frame)
    controls = {name: build_clock(frame, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, control in controls.items():
        _write_gzip_csv(control, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: split_stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, stats in support.items():
        checks[f"{name}_minimum_events"] = stats["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = stats["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = stats["max_month_share"] <= 0.45
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    nonprice_manifest = NONPRICE_DIR / "manifest.json"
    passed = all(checks.values())
    core = {
        "protocol_version": "pviar_6_source_support_v1", "policy_id": "PVIAR-6",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT), "sha256": sha256(prereg.DEFAULT_OUTPUT),
            "manifest_hash": registration["manifest_hash"],
        },
        "source_manifests": {
            "premium": {"path": str(PREMIUM_MANIFEST), "sha256": sha256(PREMIUM_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
            "volatility": {"path": str(nonprice_manifest), "sha256": sha256(nonprice_manifest)},
        },
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)},
        "controls": {
            name: {
                "path": str(CONTROL_DIR / f"{name}.csv.gz"),
                "sha256": sha256(CONTROL_DIR / f"{name}.csv.gz"),
                "rows": len(control), "promotion_authorized": False,
            }
            for name, control in controls.items()
        },
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    report = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=ENV_FILE)
    args = parser.parse_args()
    result = run(args.env_file)
    print(json.dumps({"passed": result["support_passed"], "support": result["support"]}, indent=2))
