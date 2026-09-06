"""Materialize outcome-blind source support for frozen ESVRR-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from training import preregister_emerging_safe_haven_volatility_rotation_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv

ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path("training/build_emerging_safe_haven_volatility_rotation_relay_support.py")
PREREG_SHA = "7b7cba82f94709c412f9aa7fdb9ee27f5ff378783df88d9baa23c0766e15ad50"
SOURCE_DIR = Path("data/emerging_safe_haven_volatility_rotation_relay_sources_2022_2026")
FEATURE_PANEL = SOURCE_DIR / "emerging_safe_haven_volatility_rotation_relay_preentry_features.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = SOURCE_DIR / "emerging_safe_haven_volatility_rotation_relay_clocks_2023_2026.csv.gz"
CONTROL_DIR = SOURCE_DIR / "controls"
RESULT = Path("results/emerging_safe_haven_volatility_rotation_relay_support_2026-08-10.json")
SOURCE_START = pd.Timestamp("2022-12-29T00:00:00Z")
SOURCE_END = pd.Timestamp("2026-08-01T00:00:00Z")
NY = ZoneInfo("America/New_York")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_btc_variation_gate", "no_relative_change_tail", "vxeem_minus_gvz_raw",
    "one_session_stale_relative_change", "direction_flip", "forced_long",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "cboe_observation_date", "next_cboe_source_date",
    "decision_time", "feature_available_time", "entry_time", "exit_time", "side",
    "relative_volatility", "relative_volatility_change", "absolute_change_rank",
    "btc_realized_variation", "btc_variation_rank",
)
QUERY = """
SELECT ts,open,close
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts
"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode()).hexdigest()


def strict_prior_midrank(values: pd.Series, lookback: int = 252, minimum: int = 126) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    finite_history: list[float] = []
    for index, current in numeric.items():
        history = finite_history[-lookback:]
        if np.isfinite(current) and len(history) >= minimum:
            array = np.asarray(history, dtype=float)
            output.at[index] = (
                np.count_nonzero(array < current) + 0.5 * np.count_nonzero(array == current)
            ) / len(array)
        if np.isfinite(current):
            finite_history.append(float(current))
    return output


def _validated_source(path: Path, expected_sha: str) -> None:
    if sha(path) != expected_sha:
        raise RuntimeError(f"ESVRR source hash drift: {path}")


def load_cboe() -> pd.DataFrame:
    vxeem_spec, gvz_spec = prereg.SOURCES["vxeem_panel"], prereg.SOURCES["gvz"]
    vxeem_path, gvz_path = Path(vxeem_spec["path"]), Path(gvz_spec["path"])
    _validated_source(vxeem_path, vxeem_spec["sha256"])
    _validated_source(gvz_path, gvz_spec["sha256"])
    vxeem = pd.read_csv(vxeem_path, usecols=["observation_date", "VXEEM_close"])
    vxeem["observation_date"] = pd.to_datetime(vxeem.observation_date, errors="raise")
    gvz = pd.read_csv(gvz_path, usecols=["DATE", "GVZ"])
    gvz["observation_date"] = pd.to_datetime(gvz.DATE, format="%m/%d/%Y", errors="raise")
    gvz["GVZ_close"] = pd.to_numeric(gvz.GVZ, errors="coerce")
    gvz = gvz[["observation_date", "GVZ_close"]]
    frame = vxeem.merge(gvz, on="observation_date", validate="one_to_one")
    frame = frame.sort_values("observation_date").reset_index(drop=True)
    if frame.empty or frame.observation_date.duplicated().any():
        raise RuntimeError("ESVRR common source dates invalid")
    values = frame[["VXEEM_close", "GVZ_close"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values).all(axis=None) or not values.gt(0).all(axis=None):
        raise RuntimeError("ESVRR source closes invalid")
    frame[["VXEEM_close", "GVZ_close"]] = values
    frame["relative_volatility"] = np.log(frame.VXEEM_close / frame.GVZ_close)
    frame["relative_volatility_change"] = frame.relative_volatility.diff()
    frame["absolute_change_rank"] = strict_prior_midrank(frame.relative_volatility_change.abs())
    return frame


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env
    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_bars() -> pd.DataFrame:
    from sqlalchemy import text
    engine = postgres_engine()
    try:
        frame = pd.read_sql_query(text(QUERY), engine, params={
            "start": SOURCE_START.to_pydatetime(), "end": SOURCE_END.to_pydatetime(),
        })
    finally:
        engine.dispose()
    required = ["ts", "open", "close"]
    if frame.columns.tolist() != required:
        raise RuntimeError(f"ESVRR BTC schema must be exactly {required}")
    frame.ts = pd.to_datetime(frame.ts, utc=True, errors="raise")
    frame = frame.sort_values("ts").reset_index(drop=True)
    expected = pd.date_range(SOURCE_START, SOURCE_END, freq="1min", inclusive="left")
    if len(frame) != len(expected) or not frame.ts.equals(pd.Series(expected, name="ts")):
        raise RuntimeError("ESVRR BTC source is not the exact requested 1m grid")
    for column in ("open", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame[["open", "close"]]).all(axis=None) or not frame[["open", "close"]].gt(0).all(axis=None):
        raise RuntimeError("ESVRR BTC source contains invalid prices")
    return frame.set_index("ts")


def build_features(cboe: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    variations: list[float] = []
    records: list[dict[str, Any]] = []
    for index in range(1, len(cboe)):
        next_date = cboe.at[index, "observation_date"]
        decision = (pd.Timestamp(next_date).tz_localize(NY) + pd.Timedelta(hours=9, minutes=30)).tz_convert("UTC")
        window = bars.loc[decision - pd.Timedelta(hours=24):decision - pd.Timedelta(minutes=1)]
        variation = np.nan if len(window) != 1440 else float(np.sqrt(np.square(
            np.log(window.close.to_numpy() / window.open.to_numpy())
        ).sum()))
        variations.append(variation)
        records.append({
            "source_index": index - 1,
            "cboe_observation_date": cboe.at[index - 1, "observation_date"],
            "next_cboe_source_date": next_date,
            "decision_time": decision,
            "relative_volatility": cboe.at[index - 1, "relative_volatility"],
            "relative_volatility_change": cboe.at[index - 1, "relative_volatility_change"],
            "absolute_change_rank": cboe.at[index - 1, "absolute_change_rank"],
            "vxeem_minus_gvz_raw": cboe.at[index - 1, "VXEEM_close"] - cboe.at[index - 1, "GVZ_close"],
            "btc_realized_variation": variation,
        })
    frame = pd.DataFrame(records)
    frame["btc_variation_rank"] = strict_prior_midrank(pd.Series(variations), lookback=252, minimum=126)
    return frame


def signal(features: pd.DataFrame, control: str = "primary") -> pd.Series:
    change = features.relative_volatility_change
    eligible = change.ne(0) & features.absolute_change_rank.ge(0.70) & features.btc_variation_rank.ge(0.65)
    side = -np.sign(change).astype("Int64").fillna(0).astype(int)
    if control == "no_btc_variation_gate":
        eligible = change.ne(0) & features.absolute_change_rank.ge(0.70)
    elif control == "no_relative_change_tail":
        eligible = change.ne(0) & features.btc_variation_rank.ge(0.65)
    elif control == "vxeem_minus_gvz_raw":
        raw_change = features.vxeem_minus_gvz_raw.diff()
        eligible = raw_change.ne(0) & features.absolute_change_rank.ge(0.70) & features.btc_variation_rank.ge(0.65)
        side = -np.sign(raw_change).astype("Int64").fillna(0).astype(int)
    elif control == "one_session_stale_relative_change":
        stale_change, stale_rank = change.shift(1), features.absolute_change_rank.shift(1)
        eligible = stale_change.ne(0) & stale_rank.ge(0.70) & features.btc_variation_rank.ge(0.65)
        side = -np.sign(stale_change).astype("Int64").fillna(0).astype(int)
    side = side.where(eligible, 0)
    if control == "direction_flip":
        side = -side
    elif control == "forced_long":
        side = side.abs()
    return side


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    sides, rows, next_allowed = signal(features, control), [], None
    for index in features.index[sides.ne(0)]:
        decision = pd.Timestamp(features.at[index, "decision_time"])
        entry, exit_time = decision + pd.Timedelta(minutes=5), decision + pd.Timedelta(hours=12, minutes=5)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next((name for name, (start, end) in SPLITS.items()
                      if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        next_allowed = exit_time
        rows.append({
            "candidate": "ESVRR-12", "control": control, "split": split,
            "cboe_observation_date": features.at[index, "cboe_observation_date"].date().isoformat(),
            "next_cboe_source_date": features.at[index, "next_cboe_source_date"].date().isoformat(),
            "decision_time": decision, "feature_available_time": decision,
            "entry_time": entry, "exit_time": exit_time, "side": int(sides.at[index]),
            "relative_volatility": float(features.at[index, "relative_volatility"]),
            "relative_volatility_change": float(features.at[index, "relative_volatility_change"]),
            "absolute_change_rank": float(features.at[index, "absolute_change_rank"]),
            "btc_realized_variation": float(features.at[index, "btc_realized_variation"]),
            "btc_variation_rank": float(features.at[index, "btc_variation_rank"]),
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = clock[clock.split.eq(split)].copy()
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    entries = pd.to_datetime(subset.entry_time, utc=True)
    longs, shorts = int(subset.side.eq(1).sum()), int(subset.side.eq(-1).sum())
    return {"events": len(subset), "longs": longs, "shorts": shorts,
            "minority_side_share": min(longs, shorts) / len(subset),
            "max_month_share": int(entries.dt.strftime("%Y-%m").value_counts().max()) / len(subset)}


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("ESVRR preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    cboe, bars = load_cboe(), load_bars()
    features = build_features(cboe, bars)
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(features, FEATURE_PANEL)
    _write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items():
        _write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {
        "protocol_version": "esvrr_12_sources_v1",
        "frozen_sources": prereg.SOURCES,
        "btc_query": QUERY,
        "btc_window": [SOURCE_START.isoformat(), SOURCE_END.isoformat()],
        "btc_rows": len(bars),
        "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
        "outputs": {"features": {"path": str(FEATURE_PANEL), "sha256": sha(FEATURE_PANEL), "rows": len(features)}},
        "candidate_outcomes_opened": False, "no_imputation": True,
    }
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    SOURCE_MANIFEST.write_text(json.dumps(source_manifest, indent=2, allow_nan=False) + "\n")
    support = {name: stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "esvrr_12_source_support_v1", "policy_id": "ESVRR-12",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA,
                            "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST),
                            "manifest_hash": source_manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"),
                            "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame),
                            "promotion_authorized": False} for name, frame in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    result = run()
    print(json.dumps({"passed": result["support_passed"], "support": result["support"]}, indent=2))
