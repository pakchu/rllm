"""Build source-only DDAELR-24 clocks from the official DOJ News API."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import html
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import time
from typing import Any
import unicodedata
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from training import preregister_doj_digital_asset_enforcement_lifecycle_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
API = "https://www.justice.gov/api/v1/press_releases.json"
PAGE_SIZE = 50
SCAN_START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
BTC_START = pd.Timestamp("2022-12-30T22:00:00Z")
SOURCE_DIR = Path("data/doj_digital_asset_enforcement_lifecycle_relay_sources_2023_2026")
PAGE_CACHE = SOURCE_DIR / "raw_page_cache"
RAW_ARCHIVE = SOURCE_DIR / "doj_news_api_raw_pages.jsonl.gz"
EVENTS = SOURCE_DIR / "classified_publication_days.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/doj_digital_asset_enforcement_lifecycle_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/doj_digital_asset_enforcement_lifecycle_relay_controls_2023_2026")
RESULT = Path("results/doj_digital_asset_enforcement_lifecycle_relay_support_2026-08-09.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_volatility_gate", "title_only_taxonomy", "one_event_stale_side", "direction_flip")
QUERY = """
SELECT ts,open,high,low,close
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts
"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def normalize(value: Any) -> str:
    parser = _Text()
    parser.feed(html.unescape(str(value or "")))
    text = unicodedata.normalize("NFKC", " ".join(parser.parts)).lower()
    return re.sub(r"\s+", " ", text).strip()


def contains(text: str, phrase: str) -> bool:
    escaped = re.escape(phrase).replace(r"\ ", r"\s+")
    return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text) is not None


def classify(title: str, body: str, *, title_only: bool = False) -> int:
    taxonomy = prereg.build()["taxonomy"]
    text = normalize(title) if title_only else normalize(f"{title} {body}")
    if not any(contains(text, term) for term in taxonomy["digital_asset_terms"]):
        return 0
    initiation = any(contains(text, term) for term in taxonomy["initiation_terms"])
    resolution = any(contains(text, term) for term in taxonomy["resolution_terms"])
    if initiation == resolution:
        return 0
    return -1 if initiation else 1


def page_url(page: int) -> str:
    query = urllib.parse.urlencode({
        "sort": "date", "direction": "DESC", "pagesize": PAGE_SIZE, "page": page,
        "fields": "uuid,date,title,body,url,updated",
    })
    return f"{API}?{query}"


def fetch_page(page: int) -> bytes:
    PAGE_CACHE.mkdir(parents=True, exist_ok=True)
    path = PAGE_CACHE / f"page-{page:05d}.json"
    if path.exists():
        return path.read_bytes()
    request = urllib.request.Request(page_url(page), headers={"User-Agent": "rllm-research/1.0"})
    last: Exception | None = None
    for attempt in range(5):
        try:
            raw = urllib.request.urlopen(request, timeout=60).read()
            payload = json.loads(raw)
            if payload.get("metadata", {}).get("responseInfo", {}).get("status") != 200:
                raise RuntimeError("DOJ API response status is not 200")
            path.write_bytes(raw)
            time.sleep(0.27)
            return raw
        except Exception as error:  # pragma: no cover - transport retry
            last = error
            time.sleep(2 ** attempt)
    raise RuntimeError(f"DOJ API page {page} failed") from last


def download() -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    raws: list[bytes] = []
    previous_oldest: pd.Timestamp | None = None
    page = 0
    while True:
        raw = fetch_page(page)
        payload = json.loads(raw)
        values = payload.get("results")
        if not isinstance(values, list) or not values:
            raise RuntimeError("DOJ API pagination ended before scan boundary")
        dates = pd.to_datetime([int(item["date"]) for item in values], unit="s", utc=True)
        newest, oldest = dates.max(), dates.min()
        if previous_oldest is not None and newest > previous_oldest:
            raise RuntimeError("DOJ API date ordering drift")
        previous_oldest = oldest
        raws.append(raw)
        for item, stamp in zip(values, dates):
            if stamp >= SCAN_START and stamp < END:
                rows.append({**item, "publication_date": stamp.floor("D")})
        if oldest < SCAN_START:
            break
        page += 1
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    with RAW_ARCHIVE.open("wb") as target:
        with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0, compresslevel=9) as stream:
            for raw in raws:
                stream.write(raw + b"\n")
    return rows, page + 1


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env
    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def strict_prior_midrank(values: pd.Series, lookback: int = 252, minimum: int = 126) -> pd.Series:
    output = pd.Series(np.nan, index=values.index, dtype=float)
    history: list[float] = []
    for index, current in pd.to_numeric(values, errors="coerce").items():
        prior = history[-lookback:]
        if math.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior)
            output.at[index] = (np.sum(array < current) + 0.5 * np.sum(array == current)) / len(array)
        if math.isfinite(current):
            history.append(float(current))
    return output


def publication_variation(records: list[dict[str, Any]], bars: pd.DataFrame) -> pd.DataFrame:
    dates = sorted({pd.Timestamp(item["publication_date"]) for item in records})
    frame = bars.copy()
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.drop_duplicates("ts", keep=False).set_index("ts").sort_index()
    rows = []
    for date in dates:
        decision = date + pd.Timedelta(hours=22)
        expected = pd.date_range(decision - pd.Timedelta(hours=24), decision, freq="1min", inclusive="left")
        window = frame.reindex(expected)
        valid = len(window) == 1440 and np.isfinite(window[["open", "high", "low", "close"]]).all().all() and window[["open", "high", "low", "close"]].gt(0).all().all()
        variation = float(np.log(window["close"]).diff().dropna().pow(2).sum()) if valid else np.nan
        rows.append({"publication_date": date, "decision_time": decision, "variation_24h": variation})
    result = pd.DataFrame(rows)
    result["variation_rank"] = strict_prior_midrank(result["variation_24h"])
    return result


def classified_days(records: list[dict[str, Any]], variation: pd.DataFrame, *, title_only: bool = False) -> pd.DataFrame:
    rows = []
    for item in records:
        side = classify(item.get("title", ""), item.get("body", ""), title_only=title_only)
        if side:
            rows.append({"publication_date": pd.Timestamp(item["publication_date"]), "side": side, "uuid": item["uuid"], "title": normalize(item.get("title", ""))})
    if not rows:
        return pd.DataFrame(columns=["publication_date", "side", "uuid_count", "uuid_hash", "titles"])
    frame = pd.DataFrame(rows)
    output = []
    for date, group in frame.groupby("publication_date", sort=True):
        sides = set(group["side"])
        if len(sides) != 1:
            continue
        uuids = sorted(set(group["uuid"]))
        titles = sorted(set(group["title"]))
        output.append({"publication_date": date, "side": sides.pop(), "uuid_count": len(uuids), "uuid_hash": hashlib.sha256("\n".join(uuids).encode()).hexdigest(), "titles": " | ".join(titles)})
    return pd.DataFrame(output).merge(variation, on="publication_date", how="left", validate="one_to_one")


def clock(days: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    frame = days.copy().sort_values("decision_time").reset_index(drop=True)
    side = frame["side"].copy()
    if control == "one_event_stale_side":
        side = side.shift(1)
    elif control == "direction_flip":
        side = -side
    active = np.isfinite(side) & (frame["variation_rank"].ge(0.65) if control != "no_volatility_gate" else True)
    rows = []
    reserved_until: pd.Timestamp | None = None
    for index in frame.index[active]:
        decision = pd.Timestamp(frame.at[index, "decision_time"])
        entry, exit_ = decision + pd.Timedelta(minutes=5), decision + pd.Timedelta(hours=24, minutes=5)
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_ <= end), None)
        if split is None:
            continue
        reserved_until = exit_
        rows.append({"candidate": "DDAELR-24", "control": control, "split": split, "decision_time": decision, "feature_available_time": decision, "entry_time": entry, "exit_time": exit_, "side": int(side.at[index]), "variation_24h": float(frame.at[index, "variation_24h"]), "variation_rank": float(frame.at[index, "variation_rank"]), "uuid_count": int(frame.at[index, "uuid_count"]), "uuid_hash": frame.at[index, "uuid_hash"]})
    return pd.DataFrame(rows, columns=["candidate", "control", "split", "decision_time", "feature_available_time", "entry_time", "exit_time", "side", "variation_24h", "variation_rank", "uuid_count", "uuid_hash"])


def stats(candidate: pd.DataFrame, split: str) -> dict[str, float | int]:
    subset = candidate[candidate["split"].eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(subset["side"].eq(1).sum()), int(subset["side"].eq(-1).sum())
    months = pd.to_datetime(subset["entry_time"], utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(subset), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(subset), "max_month_share": int(months.max()) / len(subset)}


def run() -> dict[str, Any]:
    from sqlalchemy import text
    records, pages = download()
    engine = postgres_engine()
    with engine.connect() as connection:
        bars = pd.read_sql_query(text(QUERY), connection, params={"start": BTC_START.to_pydatetime(), "end": END.to_pydatetime()})
    engine.dispose()
    variation = publication_variation(records, bars)
    primary_days = classified_days(records, variation)
    title_days = classified_days(records, variation, title_only=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary_days, EVENTS)
    primary = clock(primary_days)
    controls = {name: clock(title_days if name == "title_only_taxonomy" else primary_days, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, candidate in controls.items(): _write_gzip_csv(candidate, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {"protocol_version": "ddaelr_24_source_v1", "api": API, "pages": pages, "records_in_window": len(records), "raw_archive": {"path": str(RAW_ARCHIVE), "sha256": sha(RAW_ARCHIVE)}, "classified_days": {"path": str(EVENTS), "sha256": sha(EVENTS), "rows": len(primary_days)}, "postentry_outcomes_opened": False, "gross9_rows_opened": False}
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    SOURCE_MANIFEST.write_text(json.dumps(source_manifest, indent=2, allow_nan=False) + "\n")
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {"protocol_version": "ddaelr_24_source_support_v1", "policy_id": "DDAELR-24", "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": sha(prereg.DEFAULT_OUTPUT), "manifest_hash": registration["manifest_hash"]}, "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]}, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False, "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)}, "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(candidate), "promotion_authorized": False} for name, candidate in controls.items()}, "support": support, "support_checks": checks, "support_passed": passed, "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False, "decision": "pass_to_novelty" if passed else "terminal_source_support_reject"}
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    result = run()
    print(json.dumps({"passed": result["support_passed"], "support": result["support"]}, indent=2))
