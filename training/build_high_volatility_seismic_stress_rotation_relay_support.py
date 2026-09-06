"""Materialize outcome-blind source support for frozen HVSSR-24."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from training import preregister_high_volatility_seismic_stress_rotation_relay as prereg


ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path("training/build_high_volatility_seismic_stress_rotation_relay_support.py")
PREREG_SHA = "92dc3d2086e7197311e386f4c62da3b2cdf247256d6b5ac02482415dcf0b079d"
CONTRACT = Path("results/high_volatility_seismic_stress_rotation_relay_source_contract_2026-08-12.json")
CONTRACT_SHA = "28dfbe32f2d3705cdc2a270435892564de247b59ca9ca6bbe2bdbbaa7636e8b6"
SOURCE_DIR = Path("data/high_volatility_seismic_stress_rotation_relay_sources_2023_2026")
EVENT_VERSIONS = SOURCE_DIR / "causal_candidate_event_versions.json.gz"
RAW_INDEX = SOURCE_DIR / "usgs_quakeml_response_index.json"
STRESS_PANEL = SOURCE_DIR / "daily_seismic_stress.csv.gz"
FEATURE_PANEL = SOURCE_DIR / "daily_preentry_states.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_seismic_stress_rotation_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_seismic_stress_rotation_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_seismic_stress_rotation_relay_support_2026-08-12.json")
SOURCE_START = pd.Timestamp("2023-04-30T00:00:00Z")
SOURCE_END = pd.Timestamp("2026-07-31T00:00:00Z")
QUERY_START = SOURCE_START - pd.Timedelta(days=7)
QUERY_END = SOURCE_END + pd.Timedelta(days=7)
NS = {"b": "http://quakeml.org/xmlns/bed/1.2"}
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_btc_volatility_gate", "seismic_direction_flip", "one_day_stale_seismic_stress",
    "stress_rise_only", "stress_fall_only", "same_clock_forced_long",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "decision_time", "entry_time", "exit_time", "side",
    "source_day", "daily_seismic_stress", "stress_change", "stress_change_rank",
    "causal_event_count", "btc_realized_variation", "btc_variation_rank",
)
QUERY = """
SELECT
  date_trunc('day', ts - interval '12 hours') + interval '1 day 12 hours' AS decision_time,
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
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(buffer.getvalue())


def write_gzip_json(value: Any, path: Path) -> None:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as handle:
        handle.write(raw)
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(buffer.getvalue())


def strict_prior_midrank(values: pd.Series, lookback: int = 180, minimum: int = 60) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce"); output = pd.Series(np.nan, index=numeric.index, dtype=float); history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if np.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior, dtype=float)
            output.at[index] = (np.count_nonzero(array < current) + 0.5 * np.count_nonzero(array == current)) / len(array)
        if np.isfinite(current): history.append(float(current))
    return output


def text_required(node: ET.Element, path: str) -> str:
    value = node.findtext(path, namespaces=NS)
    if value is None or not value.strip(): raise RuntimeError(f"HVSSR missing QuakeML field {path}")
    return value.strip()


def parse_event(event: ET.Element) -> dict[str, Any] | None:
    public_id = event.attrib.get("publicID")
    if not public_id: raise RuntimeError("HVSSR missing event publicID")
    magnitude_nodes = event.findall("b:magnitude", NS)
    if not magnitude_nodes:
        return None
    magnitude_values = [float(text_required(node, "b:mag/b:value")) for node in magnitude_nodes]
    if not all(math.isfinite(value) for value in magnitude_values):
        raise RuntimeError("HVSSR nonfinite magnitude")
    if max(magnitude_values) < 5.0:
        return None
    origins: dict[str, dict[str, str]] = {}
    for origin in event.findall("b:origin", NS):
        origin_id = origin.attrib.get("publicID")
        if not origin_id or origin_id in origins: raise RuntimeError("HVSSR duplicate origin id")
        origins[origin_id] = {"event_time": text_required(origin, "b:time/b:value"), "creation_time": text_required(origin, "b:creationInfo/b:creationTime")}
    magnitudes: list[dict[str, Any]] = []
    for magnitude, value in zip(magnitude_nodes, magnitude_values):
        origin_id = text_required(magnitude, "b:originID")
        if origin_id not in origins: raise RuntimeError("HVSSR unlinked magnitude")
        creation = text_required(magnitude, "b:creationInfo/b:creationTime")
        magnitudes.append({"magnitude": value, "origin_id": origin_id, "creation_time": creation})
    if not origins: raise RuntimeError("HVSSR candidate event lacks origin history")
    return {
        "event_id_sha256": hashlib.sha256(public_id.encode()).hexdigest(),
        "origins": [{"origin_id_sha256": hashlib.sha256(key.encode()).hexdigest(), **value} for key, value in sorted(origins.items())],
        "magnitudes": [{**item, "origin_id": hashlib.sha256(item["origin_id"].encode()).hexdigest()} for item in sorted(magnitudes, key=lambda x: (x["creation_time"], x["origin_id"], x["magnitude"]))],
    }


def chunk_windows() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    starts = list(pd.date_range(QUERY_START, QUERY_END, freq="15D", inclusive="left")); windows = []
    for start in starts: windows.append((start, min(start + pd.Timedelta(days=15), QUERY_END)))
    return windows


def fetch_chunk(window: tuple[pd.Timestamp, pd.Timestamp]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    start, end = window
    parameters = {"format": "quakeml", "starttime": start.isoformat(), "endtime": end.isoformat(), "eventtype": "earthquake", "includeallorigins": "true", "includeallmagnitudes": "true"}
    url = prereg.USGS_EVENT_API + "?" + urlencode(parameters)
    request = Request(url, headers={"User-Agent": "rllm-hvssr-source-support/1.0"})
    with urlopen(request, timeout=180) as response: raw = response.read()
    root = ET.fromstring(raw); nodes = root.findall(".//b:event", NS)
    if len(nodes) >= 20_000: raise RuntimeError("HVSSR USGS chunk reached catalog result limit")
    events = [parsed for node in nodes if (parsed := parse_event(node)) is not None]
    index = {"url": url, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "all_events": len(nodes), "candidate_history_events": len(events)}
    return index, events


def download_event_versions() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if EVENT_VERSIONS.exists() and RAW_INDEX.exists():
        with gzip.open(EVENT_VERSIONS, "rt", encoding="utf-8") as handle: events = json.load(handle)
        index = json.loads(RAW_INDEX.read_text())
        return events, index
    with ThreadPoolExecutor(max_workers=6) as pool: chunks = list(pool.map(fetch_chunk, chunk_windows()))
    index = [item[0] for item in chunks]; merged: dict[str, dict[str, Any]] = {}
    for _, events in chunks:
        for event in events:
            key = event["event_id_sha256"]
            if key in merged and merged[key] != event: raise RuntimeError("HVSSR duplicate event history differs across chunks")
            merged[key] = event
    values = [merged[key] for key in sorted(merged)]
    SOURCE_DIR.mkdir(parents=True, exist_ok=True); write_gzip_json(values, EVENT_VERSIONS)
    RAW_INDEX.write_text(json.dumps(index, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return values, index


def causal_stress_panel(events: list[dict[str, Any]]) -> pd.DataFrame:
    days = pd.date_range(SOURCE_START, SOURCE_END, freq="1D", inclusive="left")
    stress = {day: 0.0 for day in days}; counts = {day: 0 for day in days}
    for event in events:
        origins = {item["origin_id_sha256"]: item for item in event["origins"]}
        magnitudes = sorted(event["magnitudes"], key=lambda item: pd.Timestamp(item["creation_time"]))
        candidate_days = {pd.Timestamp(origins[item["origin_id"]]["event_time"]).floor("D") for item in magnitudes}
        for day in candidate_days:
            if day not in stress: continue
            decision = day + pd.Timedelta(days=2, hours=12)
            available = [item for item in magnitudes if pd.Timestamp(item["creation_time"]) <= decision and pd.Timestamp(origins[item["origin_id"]]["creation_time"]) <= decision]
            if not available: continue
            selected = available[-1]; origin_day = pd.Timestamp(origins[selected["origin_id"]]["event_time"]).floor("D")
            if origin_day != day or selected["magnitude"] < 5.0: continue
            stress[day] += 10.0 ** (1.5 * selected["magnitude"]); counts[day] += 1
    return pd.DataFrame({"source_day": days, "daily_seismic_stress": [stress[x] for x in days], "causal_event_count": [counts[x] for x in days]})


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env
    load_env_file(ENV_FILE); return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_daily_variation() -> pd.DataFrame:
    from sqlalchemy import text
    first_decision = SOURCE_START + pd.Timedelta(days=2, hours=12); last_decision = (SOURCE_END - pd.Timedelta(days=1)) + pd.Timedelta(days=2, hours=12)
    start, end = first_decision - pd.Timedelta(days=1), last_decision
    engine = postgres_engine()
    try: frame = pd.read_sql_query(text(QUERY), engine, params={"start": start.to_pydatetime(), "end": end.to_pydatetime()})
    finally: engine.dispose()
    frame.decision_time = pd.to_datetime(frame.decision_time, utc=True, errors="raise"); expected = pd.date_range(first_decision, last_decision, freq="1D")
    if len(frame) != len(expected) or not frame.decision_time.equals(pd.Series(expected, name="decision_time")): raise RuntimeError("HVSSR BTC decision grid incomplete")
    valid = frame.source_rows.eq(1440) & frame.distinct_timestamps.eq(1440) & frame.positive_prices.eq(True)
    valid &= pd.to_datetime(frame.first_ts, utc=True).eq(frame.decision_time - pd.Timedelta(days=1)); valid &= pd.to_datetime(frame.last_ts, utc=True).eq(frame.decision_time - pd.Timedelta(minutes=1))
    frame.realized_variation = pd.to_numeric(frame.realized_variation, errors="coerce"); valid &= np.isfinite(frame.realized_variation) & frame.realized_variation.gt(0)
    if not valid.all(): raise RuntimeError("HVSSR invalid BTC variation source")
    return frame


def build_features(stress: pd.DataFrame, variation: pd.DataFrame) -> pd.DataFrame:
    frame = stress.copy(); frame["decision_time"] = frame.source_day + pd.Timedelta(days=2, hours=12)
    frame["stress_change"] = frame.daily_seismic_stress.diff(); frame["stress_change_rank"] = strict_prior_midrank(frame.stress_change.abs())
    frame["seismic_side"] = np.where(frame.stress_change.gt(0), -1, np.where(frame.stress_change.lt(0), 1, 0))
    frame = frame.merge(variation[["decision_time", "realized_variation"]], on="decision_time", how="left", validate="one_to_one")
    frame.rename(columns={"realized_variation": "btc_realized_variation"}, inplace=True); frame["btc_variation_rank"] = strict_prior_midrank(frame.btc_realized_variation)
    return frame


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS): raise ValueError(control)
    side = features.seismic_side.copy()
    if control == "one_day_stale_seismic_stress": side = side.shift(1, fill_value=0)
    if control == "seismic_direction_flip": side = -side
    if control == "stress_rise_only": side = side.where(side.eq(-1), 0)
    if control == "stress_fall_only": side = side.where(side.eq(1), 0)
    eligible = side.ne(0) & features.stress_change_rank.ge(0.65) & features.btc_variation_rank.ge(0.65)
    if control == "no_btc_volatility_gate": eligible = side.ne(0) & features.stress_change_rank.ge(0.65)
    if control == "same_clock_forced_long": side = side.where(~eligible, 1)
    rows: list[dict[str, Any]] = []; next_allowed: pd.Timestamp | None = None
    for index in features.index[eligible]:
        decision = pd.Timestamp(features.at[index, "decision_time"]); entry = decision + pd.Timedelta(minutes=5); exit_time = entry + pd.Timedelta(hours=24)
        if next_allowed is not None and entry < next_allowed: continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None: continue
        next_allowed = exit_time; source_index = index - 1 if control == "one_day_stale_seismic_stress" else index
        rows.append({"candidate": "HVSSR-24", "control": control, "split": split, "decision_time": decision, "entry_time": entry, "exit_time": exit_time, "side": int(side.at[index]), "source_day": features.at[source_index, "source_day"], "daily_seismic_stress": float(features.at[source_index, "daily_seismic_stress"]), "stress_change": float(features.at[source_index, "stress_change"]), "stress_change_rank": float(features.at[source_index, "stress_change_rank"]), "causal_event_count": int(features.at[source_index, "causal_event_count"]), "btc_realized_variation": float(features.at[index, "btc_realized_variation"]), "btc_variation_rank": float(features.at[index, "btc_variation_rank"])})
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = clock[clock.split.eq(split)].copy()
    if subset.empty: return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    entries = pd.to_datetime(subset.entry_time, utc=True); longs, shorts = int(subset.side.eq(1).sum()), int(subset.side.eq(-1).sum())
    return {"events": len(subset), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(subset), "max_month_share": int(entries.dt.strftime("%Y-%m").value_counts().max()) / len(subset)}


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA or sha(CONTRACT) != CONTRACT_SHA: raise RuntimeError("HVSSR prerequisite hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text()); prereg.validate(registration)
    contract = json.loads(CONTRACT.read_text())
    if contract.get("source_contract_passed") is not True: raise RuntimeError("HVSSR source contract not passed")
    events, raw_index = download_event_versions(); stress = causal_stress_panel(events); variation = load_daily_variation(); features = build_features(stress, variation)
    primary = build_clock(features); controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    write_gzip_csv(stress, STRESS_PANEL); write_gzip_csv(features, FEATURE_PANEL); write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items(): write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {"protocol_version": "hvssr_24_sources_v1", "usgs_window": [QUERY_START.isoformat(), QUERY_END.isoformat()], "btc_query": QUERY, "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)}, "outputs": {"event_versions": {"path": str(EVENT_VERSIONS), "sha256": sha(EVENT_VERSIONS), "events": len(events)}, "raw_index": {"path": str(RAW_INDEX), "sha256": sha(RAW_INDEX), "responses": len(raw_index), "response_bytes": sum(item["bytes"] for item in raw_index)}, "stress": {"path": str(STRESS_PANEL), "sha256": sha(STRESS_PANEL), "rows": len(stress)}, "features": {"path": str(FEATURE_PANEL), "sha256": sha(FEATURE_PANEL), "rows": len(features)}}, "candidate_outcomes_opened": False, "no_imputation": True}
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}; SOURCE_MANIFEST.write_text(json.dumps(source_manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    support_values = {name: stats(primary, name) for name in SPLITS}; checks: dict[str, bool] = {}
    for name, values in support_values.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM_EVENTS[name]; checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20; checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {"protocol_version": "hvssr_24_source_support_v1", "policy_id": "HVSSR-24", "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]}, "source_contract": {"path": str(CONTRACT), "sha256": CONTRACT_SHA, "manifest_hash": contract["manifest_hash"]}, "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]}, "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False, "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)}, "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()}, "support": support_values, "support_checks": checks, "support_passed": passed, "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False, "decision": "pass_to_novelty" if passed else "terminal_source_support_reject"}
    result = {**core, "manifest_hash": canonical_hash(core)}; RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"); return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args(); report = run(); print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
