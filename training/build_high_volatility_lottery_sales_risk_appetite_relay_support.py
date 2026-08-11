"""Materialize outcome-blind source support for frozen HVLSRA-24."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from training import preregister_high_volatility_lottery_sales_risk_appetite_relay as prereg


ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path("training/build_high_volatility_lottery_sales_risk_appetite_relay_support.py")
PREREG_SHA = "80b12477f0d4676da43d83abd0883216a081ce993613c4cd3f76fea8a51379f7"
SOURCE_DIR = Path("data/high_volatility_lottery_sales_risk_appetite_relay_sources_2023_2026")
RAW_REPORTS = SOURCE_DIR / "official_powerball_draw_reports_2022_2026.json.gz"
DRAW_PANEL = SOURCE_DIR / "eligible_draw_sales.csv.gz"
FEATURE_PANEL = SOURCE_DIR / "preentry_states.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_lottery_sales_risk_appetite_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_lottery_sales_risk_appetite_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_lottery_sales_risk_appetite_relay_support_2026-08-12.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_btc_volatility_gate", "sales_direction_flip", "one_draw_stale_sales_change", "weekday_balanced_change", "same_clock_forced_long")
CLOCK_COLUMNS = ("candidate", "control", "split", "decision_time", "entry_time", "exit_time", "side", "net_sales", "sales_change", "btc_realized_variation", "btc_variation_rank")
HEADER_PATTERN = re.compile(
    r"CDC:(?P<cdc>\d+)\s*/\s*(?P<draw>[A-Z][a-z]{2} [A-Z][a-z]{2}-\d{2}-\d{4}).*?"
    r"FOR DRAW\s+(?P<draw_number>\d+)\s+(?P<generated>[A-Z][a-z]{2} [A-Z][a-z]{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})"
)
NET_SALES_PATTERN = re.compile(r"^\s*NET SALES\s*:\s*\$\s*(?P<sales>[\d,]+\.\d{2})\s*$", re.MULTILINE)


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


def strict_prior_midrank(values: pd.Series, lookback: int = 180, minimum: int = 60) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce"); output = pd.Series(np.nan, index=numeric.index, dtype=float); history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if np.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior, dtype=float); output.at[index] = (np.count_nonzero(array < current) + 0.5 * np.count_nonzero(array == current)) / len(array)
        if np.isfinite(current): history.append(float(current))
    return output


def scheduled_draw_dates() -> list[date]:
    current, stop = date(2022, 1, 1), date(2026, 7, 29)
    values: list[date] = []
    while current <= stop:
        if current.weekday() in (0, 2, 5):
            values.append(current)
        current += timedelta(days=1)
    return values


def _download_report(draw_date: date) -> dict[str, Any]:
    url = prereg.REPORT_URL.format(date=draw_date.strftime("%Y%m%d"))
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=60) as response:
        raw = response.read()
    text = raw.decode("latin-1")
    return {"draw_date": draw_date.isoformat(), "url": url, "text": text, "sha256": hashlib.sha256(raw).hexdigest()}


def download_reports() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if RAW_REPORTS.exists():
        with gzip.open(RAW_REPORTS, "rt", encoding="utf-8") as handle:
            documents = json.load(handle)
    else:
        with ThreadPoolExecutor(max_workers=12) as pool:
            documents = list(pool.map(_download_report, scheduled_draw_dates()))
        write_gzip_json(documents, RAW_REPORTS)
    metadata = [
        {"draw_date": item["draw_date"], "url": item["url"], "bytes": len(item["text"].encode("latin-1")), "sha256": item["sha256"]}
        for item in documents
    ]
    return documents, metadata


def parse_report(document: dict[str, Any]) -> dict[str, Any]:
    text = document["text"]
    header = HEADER_PATTERN.search(text)
    sales_match = NET_SALES_PATTERN.search(text)
    if header is None or sales_match is None:
        raise RuntimeError(f"HVLSRA report schema drift: {document['draw_date']}")
    draw_date = datetime.strptime(header.group("draw"), "%a %b-%d-%Y").date()
    if draw_date.isoformat() != document["draw_date"]:
        raise RuntimeError("HVLSRA draw-date identity drift")
    generated_local = datetime.strptime(header.group("generated"), "%a %b-%d-%Y %H:%M:%S").replace(tzinfo=ZoneInfo("America/Chicago"))
    generated_utc = pd.Timestamp(generated_local).tz_convert("UTC")
    decision_time = pd.Timestamp(draw_date, tz="UTC") + pd.Timedelta(days=1, hours=12)
    net_sales = float(sales_match.group("sales").replace(",", ""))
    if not np.isfinite(net_sales) or net_sales <= 0:
        raise RuntimeError("HVLSRA nonpositive net sales")
    return {
        "draw_date": draw_date,
        "draw_weekday": draw_date.strftime("%A"),
        "draw_number": int(header.group("draw_number")),
        "report_generated_time": generated_utc,
        "decision_time": decision_time,
        "available_by_decision": bool(generated_utc <= decision_time),
        "net_sales": net_sales,
        "report_sha256": document["sha256"],
    }


def build_draw_panel(documents: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(parse_report(item) for item in documents).sort_values("draw_date").reset_index(drop=True)
    if frame.draw_date.duplicated().any() or frame.draw_number.duplicated().any():
        raise RuntimeError("HVLSRA duplicate draw identity")
    if frame.draw_date.tolist() != scheduled_draw_dates():
        raise RuntimeError("HVLSRA scheduled draw grid incomplete")
    frame["prior_draw_date"] = frame.draw_date.shift(1)
    frame["prior_net_sales"] = frame.net_sales.shift(1)
    frame["sales_change"] = np.log(frame.net_sales / frame.prior_net_sales)
    frame["result_side"] = np.sign(frame.sales_change).fillna(0).astype(int)
    prior_same_weekday = frame.groupby("draw_weekday", sort=False).net_sales.shift(1)
    frame["weekday_balanced_side"] = np.sign(np.log(frame.net_sales / prior_same_weekday)).fillna(0).astype(int)
    frame = frame[frame.available_by_decision & frame.result_side.ne(0)].reset_index(drop=True)
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
    if len(frame) != len(expected) or not frame.decision_time.equals(expected.rename("decision_time")): raise RuntimeError("HVLSRA BTC decision grid incomplete")
    valid = frame.source_rows.eq(1440) & frame.distinct_timestamps.eq(1440) & frame.positive_prices.eq(True)
    valid &= pd.to_datetime(frame.first_ts, utc=True).eq(frame.decision_time - pd.Timedelta(days=1)); valid &= pd.to_datetime(frame.last_ts, utc=True).eq(frame.decision_time - pd.Timedelta(minutes=1))
    frame.realized_variation = pd.to_numeric(frame.realized_variation, errors="coerce"); valid &= np.isfinite(frame.realized_variation) & frame.realized_variation.gt(0)
    if not valid.all(): raise RuntimeError("HVLSRA invalid BTC variation source")
    return frame, query


def build_features(groups: pd.DataFrame, variation: pd.DataFrame) -> pd.DataFrame:
    frame = groups.merge(variation[["decision_time", "realized_variation"]], on="decision_time", how="left", validate="one_to_one")
    frame.rename(columns={"realized_variation": "btc_realized_variation"}, inplace=True); frame["btc_variation_rank"] = strict_prior_midrank(frame.btc_realized_variation); return frame


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS): raise ValueError(control)
    side = features.result_side.copy()
    if control == "one_draw_stale_sales_change": side = side.shift(1, fill_value=0)
    if control == "sales_direction_flip": side = -side
    if control == "weekday_balanced_change": side = features.weekday_balanced_side.copy()
    eligible = side.ne(0) & features.btc_variation_rank.ge(0.65)
    if control == "no_btc_volatility_gate": eligible = side.ne(0)
    if control == "same_clock_forced_long": side = side.where(~eligible, 1)
    rows: list[dict[str, Any]] = []; next_allowed: pd.Timestamp | None = None
    for index in features.index[eligible]:
        decision = pd.Timestamp(features.at[index, "decision_time"]); entry = decision + pd.Timedelta(minutes=5); exit_time = entry + pd.Timedelta(hours=24)
        if next_allowed is not None and entry < next_allowed: continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None: continue
        next_allowed = exit_time; source_index = index - 1 if control == "one_draw_stale_sales_change" else index
        rows.append({"candidate": "HVLSRA-24", "control": control, "split": split, "decision_time": decision, "entry_time": entry, "exit_time": exit_time, "side": int(side.at[index]), "net_sales": float(features.at[source_index, "net_sales"]), "sales_change": float(features.at[source_index, "sales_change"]), "btc_realized_variation": float(features.at[index, "btc_realized_variation"]), "btc_variation_rank": float(features.at[index, "btc_variation_rank"])})
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = clock[clock.split.eq(split)].copy()
    if subset.empty: return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    entries = pd.to_datetime(subset.entry_time, utc=True); longs, shorts = int(subset.side.eq(1).sum()), int(subset.side.eq(-1).sum())
    return {"events": len(subset), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(subset), "max_month_share": int(entries.dt.strftime("%Y-%m").value_counts().max()) / len(subset)}


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA: raise RuntimeError("HVLSRA preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text()); prereg.validate(registration)
    documents, source_metadata = download_reports(); groups = build_draw_panel(documents); variation, query = load_variation(groups); features = build_features(groups, variation)
    primary = build_clock(features); controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True); write_gzip_csv(groups, DRAW_PANEL); write_gzip_csv(features, FEATURE_PANEL); write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items(): write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {"protocol_version": "hvlsra_24_sources_v1", "official_reports": source_metadata, "btc_query": query, "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)}, "outputs": {"raw_reports": {"path": str(RAW_REPORTS), "sha256": sha(RAW_REPORTS)}, "draw_panel": {"path": str(DRAW_PANEL), "sha256": sha(DRAW_PANEL), "rows": len(groups)}, "features": {"path": str(FEATURE_PANEL), "sha256": sha(FEATURE_PANEL), "rows": len(features)}}, "candidate_outcomes_opened": False, "no_imputation": True}
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}; SOURCE_MANIFEST.write_text(json.dumps(source_manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    support_values = {name: stats(primary, name) for name in SPLITS}; checks: dict[str, bool] = {}
    for name, values in support_values.items(): checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM_EVENTS[name]; checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20; checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values()); core = {"protocol_version": "hvlsra_24_source_support_v1", "policy_id": "HVLSRA-24", "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]}, "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]}, "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False, "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)}, "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()}, "support": support_values, "support_checks": checks, "support_passed": passed, "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False, "decision": "pass_to_novelty" if passed else "terminal_source_support_reject"}
    result = {**core, "manifest_hash": canonical_hash(core)}; RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"); return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args(); report = run(); print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
