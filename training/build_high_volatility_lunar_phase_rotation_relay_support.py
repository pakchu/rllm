"""Materialize outcome-blind source support for frozen HVLPR-24."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from training import preregister_high_volatility_lunar_phase_rotation_relay as prereg


ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path("training/build_high_volatility_lunar_phase_rotation_relay_support.py")
PREREG_SHA = "4d7f933e740839cc8cd28409872ab16e629882974c14e1e35f576e4fde0eda6a"
SOURCE_DIR = Path("data/high_volatility_lunar_phase_rotation_relay_sources_2023_2026")
RAW_PHASES = SOURCE_DIR / "usno_phases_2023_2026.json"
PHASE_PANEL = SOURCE_DIR / "eligible_phases.csv.gz"
FEATURE_PANEL = SOURCE_DIR / "daily_preentry_states.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_lunar_phase_rotation_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_lunar_phase_rotation_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_lunar_phase_rotation_relay_support_2026-08-12.json")
BTC_START = pd.Timestamp("2023-01-01T00:00:00Z")
BTC_END = pd.Timestamp("2026-08-01T00:00:00Z")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_btc_volatility_gate", "phase_direction_flip", "one_day_stale_phase_window",
    "new_moon_only", "full_moon_only", "same_clock_forced_long",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "entry_time", "exit_time", "side",
    "phase", "phase_time", "phase_distance_hours", "btc_realized_variation", "btc_variation_rank",
)
QUERY = """
SELECT
  date_trunc('day', ts) AS source_day,
  count(*) AS source_rows,
  count(DISTINCT ts) AS distinct_timestamps,
  min(ts) AS first_ts,
  max(ts) AS last_ts,
  bool_and(open > 0 AND close > 0) AS positive_prices,
  sqrt(sum(power(ln(close / open), 2))) AS realized_variation
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
GROUP BY 1 ORDER BY 1
""".strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def write_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    text = frame.to_csv(index=False, lineterminator="\n", float_format="%.12g", date_format="%Y-%m-%dT%H:%M:%SZ")
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as handle:
        handle.write(text.encode())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.getvalue())


def strict_prior_midrank(values: pd.Series, lookback: int = 270, minimum: int = 180) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if np.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior, dtype=float)
            output.at[index] = (np.count_nonzero(array < current) + 0.5 * np.count_nonzero(array == current)) / len(array)
        if np.isfinite(current):
            history.append(float(current))
    return output


def download_phases() -> tuple[bytes, pd.DataFrame, str]:
    documents: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    api_version: str | None = None
    for year in (2023, 2024, 2025, 2026):
        url = prereg.USNO_YEAR_URL.format(year=year)
        request = Request(url, headers={"User-Agent": "rllm-hvlpr-source-support/1.0"})
        with urlopen(request, timeout=60) as response:
            document = json.loads(response.read())
        if document.get("year") != year or document.get("numphases") != len(document.get("phasedata", [])):
            raise RuntimeError(f"HVLPR malformed USNO year {year}")
        if api_version is None:
            api_version = str(document.get("apiversion"))
        if str(document.get("apiversion")) != api_version:
            raise RuntimeError("HVLPR mixed USNO API versions")
        documents.append(document)
        for item in document["phasedata"]:
            if item.get("phase") not in ("New Moon", "Full Moon"):
                continue
            timestamp = pd.Timestamp(
                year=int(item["year"]), month=int(item["month"]), day=int(item["day"]),
                hour=int(str(item["time"])[:2]), minute=int(str(item["time"])[3:]), tz="UTC",
            )
            rows.append({"phase": item["phase"], "phase_time": timestamp})
    frame = pd.DataFrame(rows).sort_values("phase_time").reset_index(drop=True)
    if frame.empty or frame.phase_time.duplicated().any() or not frame.phase.isin(["New Moon", "Full Moon"]).all():
        raise RuntimeError("HVLPR eligible phase rows invalid")
    payload = json.dumps(documents, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return payload, frame, str(api_version)


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_daily_variation() -> pd.DataFrame:
    from sqlalchemy import text

    engine = postgres_engine()
    try:
        frame = pd.read_sql_query(text(QUERY), engine, params={"start": BTC_START.to_pydatetime(), "end": BTC_END.to_pydatetime()})
    finally:
        engine.dispose()
    frame.source_day = pd.to_datetime(frame.source_day, utc=True, errors="raise")
    frame = frame.sort_values("source_day").reset_index(drop=True)
    expected = pd.date_range(BTC_START, BTC_END, freq="1D", inclusive="left")
    if len(frame) != len(expected) or not frame.source_day.equals(pd.Series(expected, name="source_day")):
        raise RuntimeError("HVLPR BTC daily source grid incomplete")
    expected_first = frame.source_day
    expected_last = frame.source_day + pd.Timedelta(hours=23, minutes=59)
    valid = frame.source_rows.eq(1440) & frame.distinct_timestamps.eq(1440) & frame.positive_prices.eq(True)
    valid &= pd.to_datetime(frame.first_ts, utc=True).eq(expected_first)
    valid &= pd.to_datetime(frame.last_ts, utc=True).eq(expected_last)
    frame.realized_variation = pd.to_numeric(frame.realized_variation, errors="coerce")
    valid &= np.isfinite(frame.realized_variation) & frame.realized_variation.gt(0)
    if not valid.all():
        raise RuntimeError(f"HVLPR invalid BTC source days: {frame.loc[~valid, 'source_day'].head().tolist()}")
    return frame


def build_features(phases: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily[["source_day", "realized_variation"]].copy()
    frame["decision_time"] = frame.source_day + pd.Timedelta(days=1)
    frame = frame[frame.decision_time.lt(BTC_END)].reset_index(drop=True)
    frame.rename(columns={"realized_variation": "btc_realized_variation"}, inplace=True)
    frame["btc_variation_rank"] = strict_prior_midrank(frame.btc_realized_variation)
    phase_times = phases.phase_time.tolist()
    phase_names = phases.phase.tolist()
    names: list[str | None] = []
    times: list[pd.Timestamp | pd.NaT] = []
    distances: list[float] = []
    for decision in frame.decision_time:
        matches = [(name, time, abs((decision - time).total_seconds()) / 3600.0) for name, time in zip(phase_names, phase_times) if abs((decision - time).total_seconds()) <= 36 * 3600]
        if len(matches) > 1:
            raise RuntimeError("HVLPR overlapping eligible phases")
        if matches:
            name, time, distance = matches[0]
            names.append(name); times.append(time); distances.append(distance)
        else:
            names.append(None); times.append(pd.NaT); distances.append(np.nan)
    frame["phase"] = names
    frame["phase_time"] = times
    frame["phase_distance_hours"] = distances
    frame["phase_side"] = frame.phase.map({"New Moon": 1, "Full Moon": -1}).fillna(0).astype(int)
    return frame


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    side = features.phase_side.copy()
    if control == "one_day_stale_phase_window":
        side = side.shift(1, fill_value=0)
    if control == "phase_direction_flip":
        side = -side
    if control == "new_moon_only":
        side = side.where(side.eq(1), 0)
    if control == "full_moon_only":
        side = side.where(side.eq(-1), 0)
    eligible = side.ne(0) & features.btc_variation_rank.ge(0.65)
    if control == "no_btc_volatility_gate":
        eligible = side.ne(0)
    if control == "same_clock_forced_long":
        side = side.where(~eligible, 1)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in features.index[eligible]:
        decision = pd.Timestamp(features.at[index, "decision_time"])
        entry = decision + pd.Timedelta(minutes=5)
        exit_time = entry + pd.Timedelta(hours=24)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        next_allowed = exit_time
        phase_index = index - 1 if control == "one_day_stale_phase_window" else index
        rows.append({
            "candidate": "HVLPR-24", "control": control, "split": split,
            "decision_time": decision, "entry_time": entry, "exit_time": exit_time, "side": int(side.at[index]),
            "phase": features.at[phase_index, "phase"], "phase_time": features.at[phase_index, "phase_time"],
            "phase_distance_hours": float(features.at[phase_index, "phase_distance_hours"]),
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
    return {
        "events": len(subset), "longs": longs, "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(subset),
        "max_month_share": int(entries.dt.strftime("%Y-%m").value_counts().max()) / len(subset),
    }


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVLPR preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    phase_payload, phases, api_version = download_phases()
    daily = load_daily_variation()
    features = build_features(phases, daily)
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    RAW_PHASES.write_bytes(phase_payload)
    write_gzip_csv(phases, PHASE_PANEL)
    write_gzip_csv(features, FEATURE_PANEL)
    write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items():
        write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {
        "protocol_version": "hvlpr_24_sources_v1", "usno_api_version": api_version,
        "usno_urls": [prereg.USNO_YEAR_URL.format(year=year) for year in (2023, 2024, 2025, 2026)],
        "btc_query": QUERY, "btc_window": [BTC_START.isoformat(), BTC_END.isoformat()], "btc_days": len(daily),
        "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
        "outputs": {
            "raw_phases": {"path": str(RAW_PHASES), "sha256": sha(RAW_PHASES)},
            "eligible_phases": {"path": str(PHASE_PANEL), "sha256": sha(PHASE_PANEL), "rows": len(phases)},
            "features": {"path": str(FEATURE_PANEL), "sha256": sha(FEATURE_PANEL), "rows": len(features)},
        },
        "candidate_outcomes_opened": False, "no_imputation": True,
    }
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    SOURCE_MANIFEST.write_text(json.dumps(source_manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    support = {name: stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "hvlpr_24_source_support_v1", "policy_id": "HVLPR-24",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True,
        "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
