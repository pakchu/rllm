"""Materialize outcome-blind source support for frozen HVWODP-24."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from training import preregister_high_volatility_who_outbreak_disclosure_pressure_relay as prereg


ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path("training/build_high_volatility_who_outbreak_disclosure_pressure_relay_support.py")
PREREG_SHA = "ebbd2baf00eba25359f9a4adbfff116c2954696ad3df4b8b12537e8f13fe11dc"
SOURCE_DIR = Path("data/high_volatility_who_outbreak_disclosure_pressure_relay_sources_2023_2026")
RAW_RECORDS = SOURCE_DIR / "who_disease_outbreak_news_2022_2026.json.gz"
DAILY_PANEL = SOURCE_DIR / "daily_disclosure_pressure.csv.gz"
FEATURE_PANEL = SOURCE_DIR / "publication_day_preentry_states.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_who_outbreak_disclosure_pressure_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_who_outbreak_disclosure_pressure_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_who_outbreak_disclosure_pressure_relay_support_2026-08-12.json")
START = pd.Timestamp("2022-01-01T00:00:00Z")
END = pd.Timestamp("2026-07-30T00:00:00Z")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_btc_volatility_gate", "outbreak_direction_flip", "one_day_stale_pressure", "raw_publication_day_forced_long", "same_clock_forced_long")
CLOCK_COLUMNS = ("candidate", "control", "split", "decision_time", "entry_time", "exit_time", "side", "daily_count", "pressure", "btc_realized_variation", "btc_variation_rank")


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


def write_gzip_json(value: Any, path: Path) -> None:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as handle:
        handle.write(raw)
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


def first_url() -> str:
    params = {
        "$select": ",".join(prereg.FIELDS),
        "$filter": "PublicationDate ge 2022-01-01T00:00:00Z and PublicationDate lt 2026-07-30T00:00:00Z",
        "$orderby": "PublicationDate asc,Id asc",
        "$top": "50",
    }
    return prereg.API + "?" + urllib.parse.urlencode(params)


def fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "rllm-source-research/1.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 200:
                    raise RuntimeError(f"WHO API HTTP status {response.status}")
                return json.loads(response.read().decode("utf-8"))
        except (TimeoutError, urllib.error.URLError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise AssertionError("bounded WHO retry loop exhausted")


def normalize_record(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise RuntimeError("WHO collection item is not an object")
    missing = set(prereg.FIELDS) - set(raw)
    extras = {key for key in raw if key not in prereg.FIELDS and not key.startswith("@odata.")}
    if missing or extras:
        raise RuntimeError(f"WHO item schema drift: missing={sorted(missing)!r}, extras={sorted(extras)!r}")
    item_id = str(uuid.UUID(str(raw["Id"]))).lower()
    system_key = raw["SystemSourceKey"]
    if not isinstance(system_key, str) or not system_key.strip():
        raise RuntimeError("WHO SystemSourceKey identity is empty")
    system_key = system_key.strip()
    strings = {key: raw[key] for key in ("DonId", "UrlName", "Title")}
    if any(not isinstance(value, str) or not value.strip() for value in strings.values()):
        raise RuntimeError("WHO item string identity is empty")
    timestamps = {key: pd.Timestamp(raw[key]) for key in ("PublicationDate", "PublicationDateAndTime", "DateCreated", "LastModified")}
    if any(value.tzinfo is None for value in timestamps.values()):
        raise RuntimeError("WHO timestamp lacks timezone")
    timestamps = {key: value.tz_convert("UTC") for key, value in timestamps.items()}
    if timestamps["PublicationDate"] != timestamps["PublicationDateAndTime"]:
        raise RuntimeError("WHO publication timestamps disagree")
    publication = timestamps["PublicationDateAndTime"]
    if not START <= publication < END:
        raise RuntimeError("WHO API returned item outside frozen interval")
    return {
        "id": item_id, "system_source_key": system_key, "don_id": strings["DonId"].strip(),
        "url_name": strings["UrlName"].strip(), "title": strings["Title"].strip(),
        "publication_at": publication.isoformat(), "date_created": timestamps["DateCreated"].isoformat(),
        "last_modified": timestamps["LastModified"].isoformat(),
    }


def load_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = urllib.parse.urlparse(prereg.API)
    url: str | None = first_url()
    seen_urls: set[str] = set()
    records: list[dict[str, Any]] = []
    page_hashes: list[str] = []
    while url is not None:
        parsed = urllib.parse.urlparse(url)
        if (parsed.scheme, parsed.netloc, parsed.path) != (expected.scheme, expected.netloc, expected.path):
            raise RuntimeError("WHO pagination escaped frozen same-origin collection")
        if url in seen_urls:
            raise RuntimeError("WHO pagination cycle detected")
        seen_urls.add(url)
        payload = fetch_json(url)
        if not isinstance(payload, dict) or not isinstance(payload.get("value"), list):
            raise RuntimeError("WHO collection response schema drift")
        page_hashes.append(canonical_hash(payload))
        records.extend(normalize_record(item) for item in payload["value"])
        next_link = payload.get("@odata.nextLink")
        if next_link is not None and not isinstance(next_link, str):
            raise RuntimeError("WHO nextLink schema drift")
        url = next_link
    identities = [(row["id"], row["system_source_key"], row["don_id"], row["url_name"]) for row in records]
    if not records or len(identities) != len(set(identities)):
        raise RuntimeError("WHO collection is empty or has duplicate identities")
    records.sort(key=lambda row: (row["publication_at"], row["id"]))
    write_gzip_json(records, RAW_RECORDS)
    return records, {"pages": len(seen_urls), "rows": len(records), "page_payload_hashes": page_hashes, "normalized_records_hash": canonical_hash(records)}


def build_daily_panel(records: list[dict[str, Any]]) -> pd.DataFrame:
    publication_days = pd.Series([pd.Timestamp(row["publication_at"]).strftime("%Y-%m-%d") for row in records]).value_counts()
    frame = pd.DataFrame({"source_day": pd.date_range(START, END, inclusive="left", freq="D")})
    frame["daily_count"] = frame.source_day.dt.strftime("%Y-%m-%d").map(publication_days).fillna(0).astype(int)
    recent = frame.daily_count.rolling(28, min_periods=28).sum()
    prior = frame.daily_count.shift(28).rolling(28, min_periods=28).sum()
    frame["pressure"] = recent - prior
    frame["result_side"] = np.sign(frame.pressure).fillna(0).astype(int)
    frame["source_candidate"] = frame.daily_count.gt(0) & frame.result_side.ne(0)
    frame["decision_time"] = frame.source_day + pd.Timedelta(days=1, hours=12)
    return frame.iloc[55:].reset_index(drop=True)


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def variation_query(decisions: pd.Series) -> str:
    literals = ",".join(f"('{pd.Timestamp(item).isoformat()}'::timestamptz)" for item in decisions)
    return f"""WITH decisions(decision_time) AS (VALUES {literals})
SELECT d.decision_time, count(*) source_rows, count(DISTINCT b.ts) distinct_timestamps,
min(b.ts) first_ts, max(b.ts) last_ts, bool_and(b.open>0 AND b.close>0) positive_prices,
sqrt(sum(power(ln(b.close/b.open),2))) realized_variation
FROM decisions d JOIN bars_binance b ON b.symbol='BTCUSDT' AND b.interval='1m'
AND b.ts>=d.decision_time-interval '24 hours' AND b.ts<d.decision_time
GROUP BY d.decision_time ORDER BY d.decision_time"""


def load_features(groups: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    from sqlalchemy import text

    candidates = groups[groups.source_candidate].copy().reset_index(drop=True)
    query = variation_query(candidates.decision_time)
    engine = postgres_engine()
    try:
        variation = pd.read_sql_query(text(query), engine)
    finally:
        engine.dispose()
    variation.decision_time = pd.to_datetime(variation.decision_time, utc=True, errors="raise")
    expected = pd.to_datetime(candidates.decision_time, utc=True).reset_index(drop=True)
    if len(variation) != len(expected) or not variation.decision_time.equals(expected.rename("decision_time")):
        raise RuntimeError("HVWODP BTC decision grid incomplete")
    valid = variation.source_rows.eq(1440) & variation.distinct_timestamps.eq(1440) & variation.positive_prices.eq(True)
    valid &= pd.to_datetime(variation.first_ts, utc=True).eq(variation.decision_time - pd.Timedelta(days=1))
    valid &= pd.to_datetime(variation.last_ts, utc=True).eq(variation.decision_time - pd.Timedelta(minutes=1))
    variation.realized_variation = pd.to_numeric(variation.realized_variation, errors="coerce")
    valid &= np.isfinite(variation.realized_variation) & variation.realized_variation.gt(0)
    if not valid.all():
        raise RuntimeError("HVWODP invalid BTC variation source")
    candidates = candidates.merge(variation[["decision_time", "realized_variation"]], on="decision_time", validate="one_to_one")
    candidates.rename(columns={"realized_variation": "btc_realized_variation"}, inplace=True)
    candidates["btc_variation_rank"] = strict_prior_midrank(candidates.btc_realized_variation)
    return candidates, query


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS):
        raise ValueError(control)
    side = features.result_side.copy()
    if control == "outbreak_direction_flip":
        side = -side
    elif control == "one_day_stale_pressure":
        side = side.shift(1, fill_value=0)
    eligible = side.ne(0) & features.btc_variation_rank.ge(0.65)
    if control == "no_btc_volatility_gate":
        eligible = side.ne(0)
    if control in {"raw_publication_day_forced_long", "same_clock_forced_long"}:
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
        source_index = index - 1 if control == "one_day_stale_pressure" else index
        rows.append({"candidate": "HVWODP-24", "control": control, "split": split, "decision_time": decision, "entry_time": entry, "exit_time": exit_time, "side": int(side.at[index]), "daily_count": int(features.at[source_index, "daily_count"]), "pressure": float(features.at[source_index, "pressure"]), "btc_realized_variation": float(features.at[index, "btc_realized_variation"]), "btc_variation_rank": float(features.at[index, "btc_variation_rank"])})
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = clock[clock.split.eq(split)].copy()
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    entries = pd.to_datetime(subset.entry_time, utc=True)
    longs, shorts = int(subset.side.eq(1).sum()), int(subset.side.eq(-1).sum())
    return {"events": len(subset), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(subset), "max_month_share": int(entries.dt.strftime("%Y-%m").value_counts().max()) / len(subset)}


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVWODP preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    records, api_audit = load_records()
    groups = build_daily_panel(records)
    features, query = load_features(groups)
    primary = build_clock(features)
    controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    write_gzip_csv(groups, DAILY_PANEL)
    write_gzip_csv(features, FEATURE_PANEL)
    write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items():
        write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {"protocol_version": "hvwodp_24_sources_v1", "who_api": api_audit, "source_counts": {"records": len(records), "source_days": len(groups), "publication_day_candidates": len(features)}, "btc_query": query, "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)}, "outputs": {"raw_records": {"path": str(RAW_RECORDS), "sha256": sha(RAW_RECORDS), "rows": len(records)}, "daily_panel": {"path": str(DAILY_PANEL), "sha256": sha(DAILY_PANEL), "rows": len(groups)}, "features": {"path": str(FEATURE_PANEL), "sha256": sha(FEATURE_PANEL), "rows": len(features)}}, "candidate_outcomes_opened": False, "no_imputation": True}
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    SOURCE_MANIFEST.write_text(json.dumps(source_manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    support_values = {name: stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, values in support_values.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM_EVENTS[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {"protocol_version": "hvwodp_24_source_support_v1", "policy_id": "HVWODP-24", "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]}, "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]}, "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False, "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)}, "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()}, "support": support_values, "support_checks": checks, "support_passed": passed, "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False, "decision": "pass_to_novelty" if passed else "terminal_source_support_reject"}
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
