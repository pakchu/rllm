"""Materialize outcome-blind source support for frozen HVEFPR-24."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import subprocess
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from training import preregister_high_volatility_ethereum_flaw_pressure_relay as prereg


ENV_FILE = "/home/pakchu/rllm/.env"
BUILDER = Path("training/build_high_volatility_ethereum_flaw_pressure_relay_support.py")
PREREG_SHA = "225512eb0e165b71637a4abf18c0fc0b749ad3b64d84f0cb35e8d8b8a99706c0"
SOURCE_DIR = Path("data/high_volatility_ethereum_flaw_pressure_relay_sources_2023_2026")
REPOSITORY = SOURCE_DIR / "go-ethereum.git"
RAW_REPORTS = SOURCE_DIR / "first_parent_commits_2022_2026.json.gz"
DRAW_PANEL = SOURCE_DIR / "daily_flaw_counts.csv.gz"
FEATURE_PANEL = SOURCE_DIR / "preentry_states.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_ethereum_flaw_pressure_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_ethereum_flaw_pressure_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_ethereum_flaw_pressure_relay_support_2026-08-12.json")
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM_EVENTS = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = ("no_btc_volatility_gate", "pressure_direction_flip", "one_day_stale_pressure", "raw_day_over_day_pressure", "same_clock_forced_long")
CLOCK_COLUMNS = ("candidate", "control", "split", "decision_time", "entry_time", "exit_time", "side", "daily_count", "pressure_change", "btc_realized_variation", "btc_variation_rank")
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


def run_git(*args: str) -> bytes:
    result = subprocess.run(("git", "-C", str(REPOSITORY), *args), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout


def materialize_repository() -> dict[str, Any]:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    if not REPOSITORY.exists():
        subprocess.run(("git", "clone", "--filter=blob:none", "--no-checkout", "--single-branch", "--branch", prereg.BRANCH, prereg.REMOTE, str(REPOSITORY)), check=True)
    remote = run_git("remote", "get-url", "origin").decode("utf-8").strip()
    if remote != prereg.REMOTE:
        raise RuntimeError("HVEFPR remote drift")
    object_format = run_git("rev-parse", "--show-object-format").decode("ascii").strip()
    if object_format != "sha1":
        raise RuntimeError("HVEFPR object format drift")
    run_git("cat-file", "-e", prereg.SEALED_TIP + "^{commit}")
    run_git("fsck", "--no-dangling")
    raw = run_git("log", "--first-parent", "--reverse", "--format=%H%x1f%ct%x1f%P%x1f%s%x1e", prereg.SEALED_TIP)
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise RuntimeError("HVEFPR non-UTF8 immutable subject") from error
    commits: list[dict[str, Any]] = []
    for record in text.split("\x1e"):
        record = record.strip("\n")
        if not record:
            continue
        fields = record.split("\x1f")
        if len(fields) != 4:
            raise RuntimeError("HVEFPR git log framing drift")
        commit, timestamp, parents, subject = fields
        if not re.fullmatch(r"[0-9a-f]{40}", commit) or not subject or "\n" in subject:
            raise RuntimeError("HVEFPR immutable commit schema drift")
        parent_values = parents.split() if parents else []
        if any(not re.fullmatch(r"[0-9a-f]{40}", value) for value in parent_values):
            raise RuntimeError("HVEFPR parent schema drift")
        commits.append({"commit": commit, "committer_unix": int(timestamp), "parents": parent_values, "subject": subject})
    if not commits or commits[-1]["commit"] != prereg.SEALED_TIP or len({item["commit"] for item in commits}) != len(commits):
        raise RuntimeError("HVEFPR first-parent traversal incomplete")
    write_gzip_json(commits, RAW_REPORTS)
    return {"remote": remote, "branch": prereg.BRANCH, "sealed_tip": prereg.SEALED_TIP, "object_format": object_format, "first_parent_commits": len(commits)}


def load_commits() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata = materialize_repository()
    with gzip.open(RAW_REPORTS, "rt", encoding="utf-8") as handle:
        commits = json.load(handle)
    if not isinstance(commits, list) or len(commits) != metadata["first_parent_commits"]:
        raise RuntimeError("HVEFPR frozen commit snapshot drift")
    return commits, metadata


FLAW_PATTERN = re.compile(r"(?<![A-Za-z0-9])(?:bug|bugs|fix|fixes|fixed|crash|panic|security|vulnerability|vulnerabilities)(?![A-Za-z0-9])", re.IGNORECASE)


def build_daily_panel(commits: list[dict[str, Any]]) -> pd.DataFrame:
    start = pd.Timestamp("2022-01-01T00:00:00Z")
    end = pd.Timestamp("2026-07-30T00:00:00Z")
    rows: list[dict[str, Any]] = []
    effective_day: pd.Timestamp | None = None
    for item in commits:
        timestamp = pd.Timestamp(int(item["committer_unix"]), unit="s", tz="UTC")
        day = timestamp.floor("D")
        effective_day = day if effective_day is None else max(effective_day, day)
        if start <= effective_day < end:
            rows.append({"commit": item["commit"], "effective_day": effective_day, "flaw_related": bool(FLAW_PATTERN.search(item["subject"]))})
    retained = pd.DataFrame(rows, columns=["commit", "effective_day", "flaw_related"])
    days = pd.DataFrame({"source_day": pd.date_range(start, end, inclusive="left", freq="D")})
    flaws = retained[retained.flaw_related].groupby("effective_day").size().rename("daily_count") if not retained.empty else pd.Series(dtype=int, name="daily_count")
    frame = days.merge(flaws, left_on="source_day", right_index=True, how="left")
    frame["daily_count"] = frame.daily_count.fillna(0).astype(int)
    frame["pressure_change"] = frame.daily_count - frame.daily_count.shift(7)
    frame["raw_day_over_day_change"] = frame.daily_count - frame.daily_count.shift(1)
    frame["result_side"] = -np.sign(frame.pressure_change).fillna(0).astype(int)
    frame["raw_day_over_day_side"] = -np.sign(frame.raw_day_over_day_change).fillna(0).astype(int)
    frame["decision_time"] = frame.source_day + pd.Timedelta(days=2, hours=12)
    return frame.iloc[7:].reset_index(drop=True)


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
    if len(frame) != len(expected) or not frame.decision_time.equals(expected.rename("decision_time")): raise RuntimeError("HVEFPR BTC decision grid incomplete")
    valid = frame.source_rows.eq(1440) & frame.distinct_timestamps.eq(1440) & frame.positive_prices.eq(True)
    valid &= pd.to_datetime(frame.first_ts, utc=True).eq(frame.decision_time - pd.Timedelta(days=1)); valid &= pd.to_datetime(frame.last_ts, utc=True).eq(frame.decision_time - pd.Timedelta(minutes=1))
    frame.realized_variation = pd.to_numeric(frame.realized_variation, errors="coerce"); valid &= np.isfinite(frame.realized_variation) & frame.realized_variation.gt(0)
    if not valid.all(): raise RuntimeError("HVEFPR invalid BTC variation source")
    return frame, query


def build_features(groups: pd.DataFrame, variation: pd.DataFrame) -> pd.DataFrame:
    frame = groups.merge(variation[["decision_time", "realized_variation"]], on="decision_time", how="left", validate="one_to_one")
    frame.rename(columns={"realized_variation": "btc_realized_variation"}, inplace=True); frame["btc_variation_rank"] = strict_prior_midrank(frame.btc_realized_variation); return frame


def build_clock(features: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    if control not in ("primary", *CONTROLS): raise ValueError(control)
    side = features.result_side.copy()
    if control == "one_day_stale_pressure": side = side.shift(1, fill_value=0)
    if control == "pressure_direction_flip": side = -side
    if control == "raw_day_over_day_pressure": side = features.raw_day_over_day_side.copy()
    eligible = side.ne(0) & features.btc_variation_rank.ge(0.65)
    if control == "no_btc_volatility_gate": eligible = side.ne(0)
    if control == "same_clock_forced_long": side = side.where(~eligible, 1)
    rows: list[dict[str, Any]] = []; next_allowed: pd.Timestamp | None = None
    for index in features.index[eligible]:
        decision = pd.Timestamp(features.at[index, "decision_time"]); entry = decision + pd.Timedelta(minutes=5); exit_time = entry + pd.Timedelta(hours=24)
        if next_allowed is not None and entry < next_allowed: continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None: continue
        next_allowed = exit_time; source_index = index - 1 if control == "one_day_stale_pressure" else index
        rows.append({"candidate": "HVEFPR-24", "control": control, "split": split, "decision_time": decision, "entry_time": entry, "exit_time": exit_time, "side": int(side.at[index]), "daily_count": float(features.at[source_index, "daily_count"]), "pressure_change": float(features.at[source_index, "pressure_change"]), "btc_realized_variation": float(features.at[index, "btc_realized_variation"]), "btc_variation_rank": float(features.at[index, "btc_variation_rank"])})
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = clock[clock.split.eq(split)].copy()
    if subset.empty: return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    entries = pd.to_datetime(subset.entry_time, utc=True); longs, shorts = int(subset.side.eq(1).sum()), int(subset.side.eq(-1).sum())
    return {"events": len(subset), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(subset), "max_month_share": int(entries.dt.strftime("%Y-%m").value_counts().max()) / len(subset)}


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA: raise RuntimeError("HVEFPR preregistration hash drift")
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text()); prereg.validate(registration)
    documents, source_metadata = load_commits(); groups = build_daily_panel(documents); variation, query = load_variation(groups); features = build_features(groups, variation)
    primary = build_clock(features); controls = {name: build_clock(features, name) for name in CONTROLS}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True); write_gzip_csv(groups, DRAW_PANEL); write_gzip_csv(features, FEATURE_PANEL); write_gzip_csv(primary, CLOCK)
    for name, frame in controls.items(): write_gzip_csv(frame, CONTROL_DIR / f"{name}.csv.gz")
    source_core = {"protocol_version": "hvefpr_24_sources_v1", "git": source_metadata, "source_counts": {"first_parent_commits": len(documents), "source_days": len(groups), "flaw_related_integrations": int(groups.daily_count.sum())}, "btc_query": query, "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)}, "outputs": {"raw_records": {"path": str(RAW_REPORTS), "sha256": sha(RAW_REPORTS)}, "daily_panel": {"path": str(DRAW_PANEL), "sha256": sha(DRAW_PANEL), "rows": len(groups)}, "features": {"path": str(FEATURE_PANEL), "sha256": sha(FEATURE_PANEL), "rows": len(features)}}, "candidate_outcomes_opened": False, "no_imputation": True}
    source_manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}; SOURCE_MANIFEST.write_text(json.dumps(source_manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    support_values = {name: stats(primary, name) for name in SPLITS}; checks: dict[str, bool] = {}
    for name, values in support_values.items(): checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM_EVENTS[name]; checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20; checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values()); core = {"protocol_version": "hvefpr_24_source_support_v1", "policy_id": "HVEFPR-24", "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA, "manifest_hash": registration["manifest_hash"]}, "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]}, "completed_preentry_sources_opened": True, "postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False, "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)}, "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame), "promotion_authorized": False} for name, frame in controls.items()}, "support": support_values, "support_checks": checks, "support_passed": passed, "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False, "decision": "pass_to_novelty" if passed else "terminal_source_support_reject"}
    result = {**core, "manifest_hash": canonical_hash(core)}; RESULT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"); return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args(); report = run(); print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
