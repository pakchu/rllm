"""Materialize outcome-blind source support for frozen HVAPPR-24."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import http.client
import io
import json
import math
from pathlib import Path
import threading
from typing import Any

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from training import preregister_high_volatility_air_pollution_penalty_rotation_relay as prereg


ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path("training/build_high_volatility_air_pollution_penalty_rotation_relay_support.py")
PREREG_SHA = "d886f157c5c8b56d78736a63749fb0ed6c96fc753f9d6b385ae0597eb5ea1383"
SOURCE_DIR = Path("data/high_volatility_air_pollution_penalty_rotation_relay_sources_2023_2026")
AQI_PANEL = SOURCE_DIR / "nyc_pm25_daily_aqi.csv.gz"
FEATURE_PANEL = SOURCE_DIR / "daily_preentry_states.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_air_pollution_penalty_rotation_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_air_pollution_penalty_rotation_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_air_pollution_penalty_rotation_relay_support_2026-08-12.json")
SOURCE_START = pd.Timestamp("2023-04-30T00:00:00Z")
SOURCE_END = pd.Timestamp("2026-08-01T00:00:00Z")
RANGE_FRACTIONS = (0.48, 0.68)
NYC_COUNTIES = {"005", "047", "061", "081", "085"}
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_btc_volatility_gate", "pollution_direction_flip", "one_day_stale_pollution",
    "aqi_rise_only", "aqi_fall_only", "same_clock_forced_long",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "entry_time", "exit_time", "side",
    "source_day", "city_pm25_aqi", "aqi_change", "aqi_change_rank",
    "btc_realized_variation", "btc_variation_rank",
)
QUERY = """
SELECT
  date_trunc('day', ts - interval '1 hour') + interval '1 day 1 hour' AS decision_time,
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
_thread = threading.local()


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


def strict_prior_midrank(values: pd.Series, lookback: int = 180, minimum: int = 60) -> pd.Series:
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


def pm25_aqi(concentration: float) -> int:
    value = math.floor(float(concentration) * 10.0) / 10.0
    bands = ((0.0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
             (55.5, 125.4, 151, 200), (125.5, 225.4, 201, 300), (225.5, 325.4, 301, 500))
    if value > 325.4:
        return 500
    for low, high, aqi_low, aqi_high in bands:
        if low <= value <= high:
            return int(round((aqi_high - aqi_low) / (high - low) * (value - low) + aqi_low))
    raise RuntimeError(f"HVAPPR invalid PM2.5 concentration {concentration}")


def parse_range_body(body: bytes) -> list[tuple[str, str, float]]:
    lines = body.split(b"\n")[1:-1]
    state_rows: list[bytes] = []
    for line in lines:
        fields = line.rstrip(b"\r").split(b"|")
        if len(fields) != 9:
            continue
        site = fields[2].decode("ascii", "strict")
        if len(site) == 9 and site.isdigit() and site[:2] == "36":
            state_rows.append(line.rstrip(b"\r"))
    if not state_rows:
        raise RuntimeError("HVAPPR byte range did not contain New York state rows")
    rows: list[tuple[str, str, float]] = []
    for line in state_rows:
        fields = line.split(b"|")
        site = fields[2].decode()
        pollutant, unit = fields[5].decode(), fields[6].decode()
        if site[2:5] not in NYC_COUNTIES or pollutant != "PM2.5" or unit != "UG/M3":
            continue
        value = float(fields[7])
        # AirNow uses negative numeric sentinels for unavailable observations.
        # They are not valid monitor-hours and are handled by the frozen >=18
        # valid-hour monitor-day completeness rule below.
        if not math.isfinite(value) or value < 0:
            continue
        rows.append((fields[0].decode(), site, value))
    return rows


def connection() -> http.client.HTTPSConnection:
    item = getattr(_thread, "connection", None)
    if item is None:
        item = http.client.HTTPSConnection("files.airnowtech.org", timeout=45)
        _thread.connection = item
    return item


def fetch_hour(day: pd.Timestamp, hour: int) -> tuple[list[tuple[str, str, float]], str]:
    stamp = day.strftime("%Y%m%d")
    path = f"/airnow/{day.year}/{stamp}/HourlyData_{stamp}{hour:02d}.dat"
    for attempt in range(3):
        conn = connection()
        try:
            probe_headers = {"Range": "bytes=0-0", "User-Agent": "rllm-hvappr-source-support/1.0"}
            conn.request("GET", path, headers=probe_headers)
            probe = conn.getresponse()
            probe.read()
            content_range = probe.getheader("Content-Range", "")
            if probe.status != 206 or "/" not in content_range:
                raise RuntimeError(f"HTTP {probe.status} length probe drift")
            total = int(content_range.rsplit("/", 1)[1])
            range_start = int(total * RANGE_FRACTIONS[0])
            range_end = min(total - 1, int(total * RANGE_FRACTIONS[1]))
            headers = {"Range": f"bytes={range_start}-{range_end}", "User-Agent": "rllm-hvappr-source-support/1.0"}
            conn.request("GET", path, headers=headers)
            response = conn.getresponse()
            body = response.read()
            if response.status != 206 or len(body) != range_end - range_start + 1:
                raise RuntimeError(f"HTTP {response.status} range bytes {len(body)}")
            return parse_range_body(body), hashlib.sha256(body).hexdigest()
        except Exception:
            try:
                conn.close()
            finally:
                _thread.connection = None
            if attempt == 2:
                raise
    raise AssertionError("unreachable")


def fetch_day(day: pd.Timestamp) -> dict[str, Any]:
    observations: dict[str, dict[int, float]] = {}
    hashes: list[str] = []
    expected_date = day.strftime("%m/%d/%y")
    for hour in range(24):
        try:
            rows, digest = fetch_hour(day, hour)
        except Exception as exc:
            raise RuntimeError(f"HVAPPR AirNow fetch failed for {day.date()} hour {hour:02d}") from exc
        hashes.append(digest)
        seen: set[str] = set()
        for date_text, site, value in rows:
            if date_text != expected_date or site in seen:
                raise RuntimeError(f"HVAPPR date or duplicate site-hour drift {day} {hour}")
            seen.add(site)
            observations.setdefault(site, {})[hour] = value
    monitor_aqi = [pm25_aqi(sum(hours.values()) / len(hours)) for hours in observations.values() if len(hours) >= 18]
    return {
        "source_day": day, "city_pm25_aqi": max(monitor_aqi) if monitor_aqi else np.nan,
        "eligible_monitors": len(monitor_aqi), "hourly_slice_set_sha256": canonical_hash(hashes),
    }


def download_aqi_panel() -> pd.DataFrame:
    days = list(pd.date_range(SOURCE_START, SOURCE_END, freq="1D", inclusive="left"))
    with ThreadPoolExecutor(max_workers=24) as pool:
        rows = list(pool.map(fetch_day, days))
    frame = pd.DataFrame(rows).sort_values("source_day").reset_index(drop=True)
    if frame.city_pm25_aqi.isna().any() or frame.eligible_monitors.lt(1).any():
        raise RuntimeError(f"HVAPPR missing eligible monitor-day {frame.loc[frame.city_pm25_aqi.isna()].head().to_dict('records')}")
    return frame


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env
    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_daily_variation() -> pd.DataFrame:
    from sqlalchemy import text
    start = SOURCE_START + pd.Timedelta(hours=1)
    end = SOURCE_END + pd.Timedelta(hours=1)
    engine = postgres_engine()
    try:
        frame = pd.read_sql_query(text(QUERY), engine, params={"start": start.to_pydatetime(), "end": end.to_pydatetime()})
    finally:
        engine.dispose()
    frame.decision_time = pd.to_datetime(frame.decision_time, utc=True, errors="raise")
    expected = pd.date_range(
        SOURCE_START + pd.Timedelta(days=1, hours=1),
        periods=int((SOURCE_END - SOURCE_START) / pd.Timedelta(days=1)),
        freq="1D",
    )
    if len(frame) != len(expected) or not frame.decision_time.equals(pd.Series(expected, name="decision_time")):
        raise RuntimeError("HVAPPR BTC decision grid incomplete")
    valid = frame.source_rows.eq(1440) & frame.distinct_timestamps.eq(1440) & frame.positive_prices.eq(True)
    valid &= pd.to_datetime(frame.first_ts, utc=True).eq(frame.decision_time - pd.Timedelta(days=1))
    valid &= pd.to_datetime(frame.last_ts, utc=True).eq(frame.decision_time - pd.Timedelta(minutes=1))
    frame.realized_variation = pd.to_numeric(frame.realized_variation, errors="coerce")
    valid &= np.isfinite(frame.realized_variation) & frame.realized_variation.gt(0)
    if not valid.all():
        raise RuntimeError("HVAPPR invalid BTC variation source")
    return frame


def build_features(aqi: pd.DataFrame, variation: pd.DataFrame) -> pd.DataFrame:
    frame = aqi.copy()
    frame["decision_time"] = frame.source_day + pd.Timedelta(days=1, hours=1)
    frame["aqi_change"] = frame.city_pm25_aqi.diff()
    frame["aqi_change_rank"] = strict_prior_midrank(frame.aqi_change.abs())
    frame["pollution_side"] = np.where(frame.aqi_change.gt(0), -1, np.where(frame.aqi_change.lt(0), 1, 0))
    frame = frame.merge(variation[["decision_time", "realized_variation"]], on="decision_time", how="left", validate="one_to_one")
    frame.rename(columns={"realized_variation": "btc_realized_variation"}, inplace=True)
    frame["btc_variation_rank"] = strict_prior_midrank(frame.btc_realized_variation)
    return frame


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    side = features.pollution_side.copy()
    if control == "one_day_stale_pollution": side = side.shift(1, fill_value=0)
    if control == "pollution_direction_flip": side = -side
    if control == "aqi_rise_only": side = side.where(side.eq(-1), 0)
    if control == "aqi_fall_only": side = side.where(side.eq(1), 0)
    eligible = side.ne(0) & features.aqi_change_rank.ge(0.65) & features.btc_variation_rank.ge(0.65)
    if control == "no_btc_volatility_gate": eligible = side.ne(0) & features.aqi_change_rank.ge(0.65)
    if control == "same_clock_forced_long": side = side.where(~eligible, 1)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for index in features.index[eligible]:
        decision = pd.Timestamp(features.at[index, "decision_time"])
        entry, exit_time = decision + pd.Timedelta(minutes=5), decision + pd.Timedelta(hours=24, minutes=5)
        if next_allowed is not None and entry < next_allowed: continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None: continue
        next_allowed = exit_time
        source_index = index - 1 if control == "one_day_stale_pollution" else index
        rows.append({
            "candidate": "HVAPPR-24", "control": control, "split": split,
            "decision_time": decision, "entry_time": entry, "exit_time": exit_time, "side": int(side.at[index]),
            "source_day": features.at[source_index, "source_day"], "city_pm25_aqi": int(features.at[source_index, "city_pm25_aqi"]),
            "aqi_change": float(features.at[source_index, "aqi_change"]), "aqi_change_rank": float(features.at[source_index, "aqi_change_rank"]),
            "btc_realized_variation": float(features.at[index, "btc_realized_variation"]), "btc_variation_rank": float(features.at[index, "btc_variation_rank"]),
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = clock[clock.split.eq(split)].copy()
    if subset.empty: return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    entries = pd.to_datetime(subset.entry_time, utc=True)
    longs, shorts = int(subset.side.eq(1).sum()), int(subset.side.eq(-1).sum())
    return {"events": len(subset), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(subset), "max_month_share": int(entries.dt.strftime("%Y-%m").value_counts().max()) / len(subset)}


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA: raise RuntimeError("HVAPPR preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text()); prereg.validate(registration)
    aqi = download_aqi_panel(); variation = load_daily_variation(); features = build_features(aqi, variation)
    primary = build_clock(features); controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    write_gzip_csv(aqi, AQI_PANEL); write_gzip_csv(features, FEATURE_PANEL); write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items(): write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {
        "protocol_version": "hvappr_24_sources_v1", "airnow_url_template": prereg.AIRNOW_HOURLY_URL,
        "airnow_window": [SOURCE_START.isoformat(), SOURCE_END.isoformat()], "relative_byte_range": list(RANGE_FRACTIONS),
        "btc_query": QUERY, "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
        "outputs": {"aqi": {"path": str(AQI_PANEL), "sha256": sha(AQI_PANEL), "rows": len(aqi)}, "features": {"path": str(FEATURE_PANEL), "sha256": sha(FEATURE_PANEL), "rows": len(features)}},
        "candidate_outcomes_opened": False, "no_imputation": True,
    }
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    SOURCE_MANIFEST.write_text(json.dumps(source_manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    support = {name: stats(primary, name) for name in SPLITS}; checks: dict[str, bool] = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "hvappr_24_source_support_v1", "policy_id": "HVAPPR-24",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed, "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": False, "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    report = run(); print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
