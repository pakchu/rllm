"""Materialize outcome-blind source support for frozen HVFTRR-12."""
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

from training import preregister_high_volatility_fan_token_result_rotation_relay as prereg


ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path("training/build_high_volatility_fan_token_result_rotation_relay_support.py")
PREREG_SHA = "fd04741bca1211a27812d7a26ec4cabb6741b0b468f701d6886db4c6e9ed7ac1"
SOURCE_DIR = Path("data/high_volatility_fan_token_result_rotation_relay_sources_2023_2026")
RAW_MATCHES = SOURCE_DIR / "espn_laliga_2023_2026.json.gz"
MATCH_PANEL = SOURCE_DIR / "eligible_match_result_groups.csv.gz"
FEATURE_PANEL = SOURCE_DIR / "preentry_states.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_fan_token_result_rotation_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_fan_token_result_rotation_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_fan_token_result_rotation_relay_support_2026-08-12.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_btc_volatility_gate", "result_direction_flip", "one_match_stale_result", "wins_only", "losses_only", "same_clock_forced_long")
CLOCK_COLUMNS = ("candidate", "control", "split", "decision_time", "entry_time", "exit_time", "side", "match_count", "tracked_team_count", "btc_realized_variation", "btc_variation_rank")


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


def download_matches() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if RAW_MATCHES.exists():
        with gzip.open(RAW_MATCHES, "rt", encoding="utf-8") as handle: documents = json.load(handle)
    else:
        documents = []
        for year in (2023, 2024, 2025, 2026):
            url = prereg.ESPN_SCOREBOARD.format(year=year); request = Request(url, headers={"User-Agent": "rllm-hvftrr-source-support/1.0"})
            with urlopen(request, timeout=60) as response: document = json.loads(response.read())
            documents.append({"year": year, "url": url, "document": document})
        write_gzip_json(documents, RAW_MATCHES)
    metadata = [{"year": item["year"], "url": item["url"], "events": len(item["document"].get("events", [])), "document_hash": canonical_hash(item["document"])} for item in documents]
    return documents, metadata


def parse_match(event: dict[str, Any]) -> dict[str, Any] | None:
    event_id = str(event.get("id", "")); kickoff = pd.Timestamp(event.get("date")); status = event.get("status", {}).get("type", {})
    competitions = event.get("competitions")
    if not event_id or kickoff.tz is None or not isinstance(competitions, list) or len(competitions) != 1: raise RuntimeError("HVFTRR event schema drift")
    if status.get("completed") is not True: return None
    competitors = competitions[0].get("competitors")
    if not isinstance(competitors, list) or len(competitors) != 2: raise RuntimeError("HVFTRR competitor schema drift")
    ids = [str(item.get("team", {}).get("id", "")) for item in competitors]
    if len(set(ids)) != 2 or not all(ids): raise RuntimeError("HVFTRR duplicate team identity")
    tracked = [item for item in competitors if str(item["team"]["id"]) in prereg.TEAM_IDS]
    if len(tracked) != 1: return None
    scores = [int(item.get("score")) for item in competitors]
    if scores[0] == scores[1]: return None
    winner_index = int(scores[1] > scores[0]); booleans = [item.get("winner") for item in competitors]
    if booleans not in ([True, False], [False, True]) or booleans[winner_index] is not True: raise RuntimeError("HVFTRR winner-score contract drift")
    tracked_index = competitors.index(tracked[0]); side = 1 if tracked_index == winner_index else -1
    return {"event_id": event_id, "decision_time": kickoff.tz_convert("UTC") + pd.Timedelta(hours=3), "side": side, "tracked_team_id": ids[tracked_index]}


def build_match_groups(documents: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []; seen: set[str] = set()
    for item in documents:
        for event in item["document"].get("events", []):
            event_id = str(event.get("id", ""))
            if event_id in seen: raise RuntimeError("HVFTRR duplicate event across annual sources")
            seen.add(event_id); parsed = parse_match(event)
            if parsed is not None: rows.append(parsed)
    frame = pd.DataFrame(rows).sort_values(["decision_time", "event_id"]).reset_index(drop=True)
    groups: list[dict[str, Any]] = []
    for decision, subset in frame.groupby("decision_time", sort=True):
        if subset.side.nunique() != 1: continue
        groups.append({"decision_time": decision, "result_side": int(subset.side.iloc[0]), "match_count": len(subset), "tracked_team_count": subset.tracked_team_id.nunique(), "event_set_hash": canonical_hash(subset.event_id.tolist())})
    return pd.DataFrame(groups)


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
    if len(frame) != len(expected) or not frame.decision_time.equals(expected.rename("decision_time")): raise RuntimeError("HVFTRR BTC decision grid incomplete")
    valid = frame.source_rows.eq(1440) & frame.distinct_timestamps.eq(1440) & frame.positive_prices.eq(True)
    valid &= pd.to_datetime(frame.first_ts, utc=True).eq(frame.decision_time - pd.Timedelta(days=1)); valid &= pd.to_datetime(frame.last_ts, utc=True).eq(frame.decision_time - pd.Timedelta(minutes=1))
    frame.realized_variation = pd.to_numeric(frame.realized_variation, errors="coerce"); valid &= np.isfinite(frame.realized_variation) & frame.realized_variation.gt(0)
    if not valid.all(): raise RuntimeError("HVFTRR invalid BTC variation source")
    return frame, query


def build_features(groups: pd.DataFrame, variation: pd.DataFrame) -> pd.DataFrame:
    frame = groups.merge(variation[["decision_time", "realized_variation"]], on="decision_time", how="left", validate="one_to_one")
    frame.rename(columns={"realized_variation": "btc_realized_variation"}, inplace=True); frame["btc_variation_rank"] = strict_prior_midrank(frame.btc_realized_variation); return frame


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS): raise ValueError(control)
    side = features.result_side.copy()
    if control == "one_match_stale_result": side = side.shift(1, fill_value=0)
    if control == "result_direction_flip": side = -side
    if control == "wins_only": side = side.where(side.eq(1), 0)
    if control == "losses_only": side = side.where(side.eq(-1), 0)
    eligible = side.ne(0) & features.btc_variation_rank.ge(0.65)
    if control == "no_btc_volatility_gate": eligible = side.ne(0)
    if control == "same_clock_forced_long": side = side.where(~eligible, 1)
    rows: list[dict[str, Any]] = []; next_allowed: pd.Timestamp | None = None
    for index in features.index[eligible]:
        decision = pd.Timestamp(features.at[index, "decision_time"]); entry = decision + pd.Timedelta(minutes=5); exit_time = entry + pd.Timedelta(hours=12)
        if next_allowed is not None and entry < next_allowed: continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None: continue
        next_allowed = exit_time; source_index = index - 1 if control == "one_match_stale_result" else index
        rows.append({"candidate": "HVFTRR-12", "control": control, "split": split, "decision_time": decision, "entry_time": entry, "exit_time": exit_time, "side": int(side.at[index]), "match_count": int(features.at[source_index, "match_count"]), "tracked_team_count": int(features.at[source_index, "tracked_team_count"]), "btc_realized_variation": float(features.at[index, "btc_realized_variation"]), "btc_variation_rank": float(features.at[index, "btc_variation_rank"])})
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = clock[clock.split.eq(split)].copy()
    if subset.empty: return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    entries = pd.to_datetime(subset.entry_time, utc=True); longs, shorts = int(subset.side.eq(1).sum()), int(subset.side.eq(-1).sum())
    return {"events": len(subset), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(subset), "max_month_share": int(entries.dt.strftime("%Y-%m").value_counts().max()) / len(subset)}


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA: raise RuntimeError("HVFTRR preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text()); prereg.validate(registration)
    documents, source_metadata = download_matches(); groups = build_match_groups(documents); variation, query = load_variation(groups); features = build_features(groups, variation)
    primary = build_clock(features); controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True); write_gzip_csv(groups, MATCH_PANEL); write_gzip_csv(features, FEATURE_PANEL); write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items(): write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {"protocol_version": "hvftrr_12_sources_v1", "espn_sources": source_metadata, "btc_query": query, "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)}, "outputs": {"raw_matches": {"path": str(RAW_MATCHES), "sha256": sha(RAW_MATCHES)}, "match_groups": {"path": str(MATCH_PANEL), "sha256": sha(MATCH_PANEL), "rows": len(groups)}, "features": {"path": str(FEATURE_PANEL), "sha256": sha(FEATURE_PANEL), "rows": len(features)}}, "candidate_outcomes_opened": False, "no_imputation": True}
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}; SOURCE_MANIFEST.write_text(json.dumps(source_manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    support_values = {name: stats(primary, name) for name in SPLITS}; checks: dict[str, bool] = {}
    for name, values in support_values.items(): checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM_EVENTS[name]; checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20; checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values()); core = {"protocol_version": "hvftrr_12_source_support_v1", "policy_id": "HVFTRR-12", "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]}, "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]}, "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False, "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)}, "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()}, "support": support_values, "support_checks": checks, "support_passed": passed, "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False, "decision": "pass_to_novelty" if passed else "terminal_source_support_reject"}
    result = {**core, "manifest_hash": canonical_hash(core)}; RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"); return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args(); report = run(); print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
