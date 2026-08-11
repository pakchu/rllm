"""Materialize outcome-blind source support for frozen HVDRA-24."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from training import preregister_high_volatility_doi_research_attention_relay as prereg


ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path("training/build_high_volatility_doi_research_attention_relay_support.py")
PREREG_SHA = "edab5cf88d7cdd2b90b58aa938a4e09e5c58aeae27e3dd646b42bf2c129ef3a4"
SOURCE_DIR = Path("data/high_volatility_doi_research_attention_relay_sources_2023_2026")
RAW_REPORTS = SOURCE_DIR / "crossref_title_query_records_2022_2026.json.gz"
DRAW_PANEL = SOURCE_DIR / "eligible_daily_doi_counts.csv.gz"
FEATURE_PANEL = SOURCE_DIR / "preentry_states.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_doi_research_attention_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_doi_research_attention_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_doi_research_attention_relay_support_2026-08-12.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_btc_volatility_gate", "attention_direction_flip", "one_day_stale_attention_change", "raw_day_over_day_change", "same_clock_forced_long")
CLOCK_COLUMNS = ("candidate", "control", "split", "decision_time", "entry_time", "exit_time", "side", "daily_count", "attention_change", "btc_realized_variation", "btc_variation_rank")
TITLE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:bitcoin|cryptocurrenc(?:y|ies)|crypto-?assets?|crypto assets?)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
QUERY_TERMS = ("bitcoin", "cryptocurrency", "cryptoasset")
MAILTO = "gus4734@gmail.com"


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    text = frame.to_csv(index=False, lineterminator="\n", float_format="%.12g", date_format="%Y-%m-%dT%H:%M:%SZ")
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as handle: handle.write(text.encode())
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(buffer.getvalue())


def write_gzip_json(value: Any, path: Path) -> None:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode(); buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as handle: handle.write(raw)
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(buffer.getvalue())


def strict_prior_midrank(values: pd.Series, lookback: int = 270, minimum: int = 180) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce"); output = pd.Series(np.nan, index=numeric.index, dtype=float); history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if np.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior, dtype=float); output.at[index] = (np.count_nonzero(array < current) + 0.5 * np.count_nonzero(array == current)) / len(array)
        if np.isfinite(current): history.append(float(current))
    return output


def normalize_item(item: dict[str, Any]) -> dict[str, Any]:
    titles = item.get("title")
    if not isinstance(titles, list) or not all(isinstance(value, str) for value in titles):
        raise RuntimeError("HVDRA title schema drift")
    created = item.get("created", {}).get("date-time")
    deposited = item.get("deposited", {}).get("date-time")
    doi = item.get("DOI")
    work_type = item.get("type")
    if not all(isinstance(value, str) and value for value in (doi, created, deposited, work_type)):
        raise RuntimeError("HVDRA Crossref selected-field drift")
    return {
        "doi": doi.lower(),
        "titles": [" ".join(value.split()) for value in titles],
        "type": work_type,
        "created": created,
        "deposited": deposited,
    }


def fetch_query(term: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cursor = "*"
    records: list[dict[str, Any]] = []
    pages = 0
    total_results: int | None = None
    seen_cursors: set[str] = set()
    while True:
        params = {
            "query.title": term,
            "filter": "from-created-date:2022-01-01T00:00:00,until-created-date:2026-07-30T23:59:59",
            "select": "DOI,title,type,created,deposited",
            "rows": 1000,
            "cursor": cursor,
            "mailto": MAILTO,
        }
        url = f"{prereg.CROSSREF_WORKS}?{urlencode(params)}"
        request = Request(url, headers={"User-Agent": f"rllm-research/1.0 (mailto:{MAILTO})"})
        with urlopen(request, timeout=90) as response:
            payload = json.loads(response.read())
        if payload.get("status") != "ok" or payload.get("message-type") != "work-list":
            raise RuntimeError("HVDRA Crossref response drift")
        message = payload.get("message", {})
        items = message.get("items")
        if not isinstance(items, list):
            raise RuntimeError("HVDRA Crossref items drift")
        pages += 1
        if total_results is None:
            total_results = int(message.get("total-results"))
        records.extend(normalize_item(item) for item in items)
        if not items:
            break
        if total_results is not None and len(records) >= total_results:
            break
        next_cursor = message.get("next-cursor")
        if not isinstance(next_cursor, str) or not next_cursor or next_cursor in seen_cursors:
            raise RuntimeError("HVDRA Crossref cursor drift")
        seen_cursors.add(next_cursor)
        cursor = next_cursor
    return records, {"term": term, "pages": pages, "total_results": total_results, "returned_rows": len(records)}


def download_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if RAW_REPORTS.exists():
        with gzip.open(RAW_REPORTS, "rt", encoding="utf-8") as handle:
            snapshot = json.load(handle)
    else:
        records: list[dict[str, Any]] = []
        queries: list[dict[str, Any]] = []
        for term in QUERY_TERMS:
            query_records, metadata = fetch_query(term)
            records.extend(query_records)
            queries.append(metadata)
        snapshot = {"queries": queries, "records": records}
        write_gzip_json(snapshot, RAW_REPORTS)
    if not isinstance(snapshot.get("queries"), list) or not isinstance(snapshot.get("records"), list):
        raise RuntimeError("HVDRA snapshot drift")
    if any(item["returned_rows"] != item["total_results"] for item in snapshot["queries"]):
        raise RuntimeError("HVDRA Crossref query incomplete")
    return snapshot["records"], snapshot["queries"]


def build_daily_panel(documents: list[dict[str, Any]]) -> pd.DataFrame:
    unique: dict[str, dict[str, Any]] = {}
    for item in documents:
        doi = item["doi"]
        if doi in unique and unique[doi] != item:
            raise RuntimeError("HVDRA conflicting duplicate DOI")
        unique[doi] = item
    eligible: list[dict[str, Any]] = []
    for item in unique.values():
        created = pd.Timestamp(item["created"])
        deposited = pd.Timestamp(item["deposited"])
        if created.tz is None or deposited.tz is None:
            raise RuntimeError("HVDRA non-UTC Crossref timestamp")
        created = created.tz_convert("UTC")
        deposited = deposited.tz_convert("UTC")
        if len(item["titles"]) != 1 or item["type"] not in prereg.WORK_TYPES or not TITLE_PATTERN.search(item["titles"][0]):
            continue
        if created.date() != deposited.date():
            continue
        eligible.append({"doi": item["doi"], "title": item["titles"][0], "type": item["type"], "created_time": created, "deposited_time": deposited, "source_day": created.floor("D")})
    eligible_frame = pd.DataFrame(eligible, columns=["doi", "title", "type", "created_time", "deposited_time", "source_day"])
    start = pd.Timestamp("2022-01-01T00:00:00Z")
    end = pd.Timestamp("2026-07-31T00:00:00Z")
    days = pd.DataFrame({"source_day": pd.date_range(start, end, inclusive="left", freq="D")})
    counts = eligible_frame.groupby("source_day").size().rename("daily_count") if not eligible_frame.empty else pd.Series(dtype=int, name="daily_count")
    frame = days.merge(counts, on="source_day", how="left")
    frame["daily_count"] = frame.daily_count.fillna(0).astype(int)
    frame["attention_change"] = frame.daily_count - frame.daily_count.shift(7)
    frame["raw_day_over_day_change"] = frame.daily_count - frame.daily_count.shift(1)
    frame["result_side"] = np.sign(frame.attention_change).fillna(0).astype(int)
    frame["raw_day_over_day_side"] = np.sign(frame.raw_day_over_day_change).fillna(0).astype(int)
    frame["decision_time"] = frame.source_day + pd.Timedelta(days=2, hours=12)
    frame = frame.iloc[7:].reset_index(drop=True)
    return frame


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env
    load_env_file(ENV_FILE); return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def variation_query(decisions: pd.Series) -> str:
    literals = ",".join(f"('{pd.Timestamp(item).isoformat()}'::timestamptz)" for item in decisions)
    return f"""WITH decisions(decision_time) AS (VALUES {literals})
SELECT d.decision_time, count(*) source_rows, count(DISTINCT b.ts) distinct_timestamps,
min(b.ts) first_ts, max(b.ts) last_ts, bool_and(b.open>0 AND b.close>0) positive_prices,
sqrt(sum(power(ln(b.close/b.open),2))) realized_variation
FROM decisions d JOIN bars_binance b ON b.symbol='BTCUSDT' AND b.interval='1m'
AND b.ts>=d.decision_time-interval '24 hours' AND b.ts<d.decision_time
GROUP BY d.decision_time ORDER BY d.decision_time"""


def load_variation(groups: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    from sqlalchemy import text
    query = variation_query(groups.decision_time); engine = postgres_engine()
    try: frame = pd.read_sql_query(text(query), engine)
    finally: engine.dispose()
    frame.decision_time = pd.to_datetime(frame.decision_time, utc=True, errors="raise"); expected = pd.to_datetime(groups.decision_time, utc=True).reset_index(drop=True)
    if len(frame) != len(expected) or not frame.decision_time.equals(expected.rename("decision_time")): raise RuntimeError("HVDRA BTC decision grid incomplete")
    valid = frame.source_rows.eq(1440) & frame.distinct_timestamps.eq(1440) & frame.positive_prices.eq(True)
    valid &= pd.to_datetime(frame.first_ts, utc=True).eq(frame.decision_time - pd.Timedelta(days=1)); valid &= pd.to_datetime(frame.last_ts, utc=True).eq(frame.decision_time - pd.Timedelta(minutes=1))
    frame.realized_variation = pd.to_numeric(frame.realized_variation, errors="coerce"); valid &= np.isfinite(frame.realized_variation) & frame.realized_variation.gt(0)
    if not valid.all(): raise RuntimeError("HVDRA invalid BTC variation source")
    return frame, query


def build_features(groups: pd.DataFrame, variation: pd.DataFrame) -> pd.DataFrame:
    frame = groups.merge(variation[["decision_time", "realized_variation"]], on="decision_time", how="left", validate="one_to_one")
    frame.rename(columns={"realized_variation": "btc_realized_variation"}, inplace=True); frame["btc_variation_rank"] = strict_prior_midrank(frame.btc_realized_variation); return frame


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS): raise ValueError(control)
    side = features.result_side.copy()
    if control == "one_day_stale_attention_change": side = side.shift(1, fill_value=0)
    if control == "attention_direction_flip": side = -side
    if control == "raw_day_over_day_change": side = features.raw_day_over_day_side.copy()
    eligible = side.ne(0) & features.btc_variation_rank.ge(0.65)
    if control == "no_btc_volatility_gate": eligible = side.ne(0)
    if control == "same_clock_forced_long": side = side.where(~eligible, 1)
    rows: list[dict[str, Any]] = []; next_allowed: pd.Timestamp | None = None
    for index in features.index[eligible]:
        decision = pd.Timestamp(features.at[index, "decision_time"]); entry = decision + pd.Timedelta(minutes=5); exit_time = entry + pd.Timedelta(hours=24)
        if next_allowed is not None and entry < next_allowed: continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None: continue
        next_allowed = exit_time; source_index = index - 1 if control == "one_day_stale_attention_change" else index
        rows.append({"candidate": "HVDRA-24", "control": control, "split": split, "decision_time": decision, "entry_time": entry, "exit_time": exit_time, "side": int(side.at[index]), "daily_count": float(features.at[source_index, "daily_count"]), "attention_change": float(features.at[source_index, "attention_change"]), "btc_realized_variation": float(features.at[index, "btc_realized_variation"]), "btc_variation_rank": float(features.at[index, "btc_variation_rank"])})
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = clock[clock.split.eq(split)].copy()
    if subset.empty: return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    entries = pd.to_datetime(subset.entry_time, utc=True); longs, shorts = int(subset.side.eq(1).sum()), int(subset.side.eq(-1).sum())
    return {"events": len(subset), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(subset), "max_month_share": int(entries.dt.strftime("%Y-%m").value_counts().max()) / len(subset)}


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA: raise RuntimeError("HVDRA preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text()); prereg.validate(registration)
    documents, source_metadata = download_records(); groups = build_daily_panel(documents); variation, query = load_variation(groups); features = build_features(groups, variation)
    primary = build_clock(features); controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True); write_gzip_csv(groups, DRAW_PANEL); write_gzip_csv(features, FEATURE_PANEL); write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items(): write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {"protocol_version": "hvdra_24_sources_v1", "crossref_queries": source_metadata, "source_counts": {"query_rows_with_duplicates": len(documents), "unique_doi": len({item["doi"] for item in documents}), "eligible_doi_deposits": int(groups.daily_count.sum())}, "btc_query": query, "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)}, "outputs": {"raw_records": {"path": str(RAW_REPORTS), "sha256": sha(RAW_REPORTS)}, "daily_panel": {"path": str(DRAW_PANEL), "sha256": sha(DRAW_PANEL), "rows": len(groups)}, "features": {"path": str(FEATURE_PANEL), "sha256": sha(FEATURE_PANEL), "rows": len(features)}}, "candidate_outcomes_opened": False, "no_imputation": True}
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}; SOURCE_MANIFEST.write_text(json.dumps(source_manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    support_values = {name: stats(primary, name) for name in SPLITS}; checks: dict[str, bool] = {}
    for name, values in support_values.items(): checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM_EVENTS[name]; checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20; checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values()); core = {"protocol_version": "hvdra_24_source_support_v1", "policy_id": "HVDRA-24", "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]}, "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]}, "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False, "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)}, "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()}, "support": support_values, "support_checks": checks, "support_passed": passed, "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False, "decision": "pass_to_novelty" if passed else "terminal_source_support_reject"}
    result = {**core, "manifest_hash": canonical_hash(core)}; RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"); return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args(); report = run(); print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
