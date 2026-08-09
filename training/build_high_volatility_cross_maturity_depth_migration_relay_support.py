"""Build checksum-verified source support for HVCMDM-8."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import math
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from training import preregister_high_volatility_cross_maturity_depth_migration_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA256 = "a593e05871009ef7755666371fce58802d576321239f2c49f66da3ffda9253e1"
BASE_URL = "https://data.binance.vision/data/futures/cm/daily/bookDepth"
EXPIRIES = tuple(
    pd.Timestamp(value, tz="UTC")
    for value in (
        "2023-03-31 08:00", "2023-06-30 08:00", "2023-09-29 08:00", "2023-12-29 08:00",
        "2024-03-29 08:00", "2024-06-28 08:00", "2024-09-27 08:00", "2024-12-27 08:00",
        "2025-03-28 08:00", "2025-06-27 08:00", "2025-09-26 08:00", "2025-12-26 08:00",
        "2026-03-27 08:00", "2026-06-26 08:00", "2026-09-25 08:00", "2026-12-25 08:00",
    )
)
SOURCE_DIR = Path("data/high_volatility_cross_maturity_depth_migration_relay_sources_2023_2026")
PANEL = SOURCE_DIR / "half_hour_depth_panel.csv.gz"
LEDGER = SOURCE_DIR / "archive_ledger.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_cross_maturity_depth_migration_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_cross_maturity_depth_migration_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_cross_maturity_depth_migration_relay_support_2026-08-10.json")
QUERY = """
SELECT ts,open,high,low,close
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts
"""
LEVELS = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_volatility_gate", "no_migration_gate", "near_pressure_only",
    "one_decision_stale_features", "direction_flip", "forced_long",
)
COLUMNS = (
    "candidate", "control", "split", "decision_time", "feature_available_time", "entry_time", "exit_time",
    "side", "near_contract", "far_contract", "near_pressure", "far_pressure", "term_pressure",
    "far_share", "migration", "absolute_migration_rank", "absolute_term_pressure_rank",
    "btc_variation", "btc_variation_rank",
)


@dataclass(frozen=True)
class Job:
    day: pd.Timestamp
    symbol: str


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def symbol_for_expiry(expiry: pd.Timestamp) -> str:
    return f"BTCUSD_{expiry.strftime('%y%m%d')}"


def contract_pair(decision: pd.Timestamp) -> tuple[str, str]:
    live = [expiry for expiry in EXPIRIES if expiry > decision]
    if len(live) < 2:
        raise ValueError("HVCMDM has fewer than two live quarterly maturities")
    return symbol_for_expiry(live[0]), symbol_for_expiry(live[1])


def archive_url(job: Job) -> str:
    date = job.day.strftime("%Y-%m-%d")
    stem = f"{job.symbol}-bookDepth-{date}.zip"
    return f"{BASE_URL}/{job.symbol}/{stem}"


def fetch(url: str, retries: int = 5) -> bytes:
    error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except Exception as exc:  # pragma: no cover - network retry
            error = exc
            if attempt + 1 == retries:
                break
    raise RuntimeError(f"HVCMDM download failed: {url}") from error


def expected_checksum(payload: bytes, filename: str) -> str:
    text = payload.decode("utf-8").strip()
    match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(.+)", text)
    if not match or match.group(2).strip() != filename:
        raise ValueError("HVCMDM checksum payload malformed")
    return match.group(1).lower()


def parse_archive(payload: bytes, job: Job) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if len(names) != 1 or not names[0].endswith(".csv"):
            raise ValueError("HVCMDM archive member contract failed")
        frame = pd.read_csv(archive.open(names[0]))
    if list(frame.columns) != ["timestamp", "percentage", "depth", "notional"]:
        raise ValueError("HVCMDM archive schema drift")
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="coerce")
    frame["percentage"] = pd.to_numeric(frame.percentage, errors="coerce")
    frame["depth"] = pd.to_numeric(frame.depth, errors="coerce")
    frame["notional"] = pd.to_numeric(frame.notional, errors="coerce")
    day_end = job.day + pd.Timedelta(days=1)
    if (
        frame.empty or frame.timestamp.isna().any() or frame.percentage.isna().any()
        or frame.depth.isna().any() or frame.notional.isna().any()
        or not frame.timestamp.ge(job.day).all() or not frame.timestamp.lt(day_end).all()
        or not frame.depth.gt(0).all() or not frame.notional.gt(0).all()
        or frame.duplicated(["timestamp", "percentage"]).any()
    ):
        raise ValueError("HVCMDM archive row validity failed")
    counts = frame.groupby("timestamp", sort=False).percentage.agg(["size", lambda x: tuple(sorted(x.astype(int)))])
    if not counts["size"].eq(10).all() or not counts["<lambda_0>"].map(lambda value: value == LEVELS).all():
        raise ValueError("HVCMDM snapshot level completeness failed")
    pivot = frame.pivot(index="timestamp", columns="percentage", values="depth").sort_index()
    output = pd.DataFrame(index=pivot.index)
    output["pressure"] = np.log(pivot[-1.0] / pivot[1.0])
    output["mass"] = pivot[-1.0] + pivot[1.0]
    if not np.isfinite(output).all(axis=1).all() or not output.mass.gt(0).all():
        raise ValueError("HVCMDM snapshot transform invalid")
    output["interval_start"] = output.index.floor("30min")
    grouped = output.groupby("interval_start", sort=True)
    reduced = grouped.agg(
        snapshots=("pressure", "size"), first_snapshot=("pressure", lambda x: x.index.min()),
        last_snapshot=("pressure", lambda x: x.index.max()), pressure=("pressure", "median"), mass=("mass", "median"),
    ).reset_index()
    reduced["symbol"] = job.symbol
    return reduced


def process_job(job: Job) -> tuple[pd.DataFrame, dict[str, Any]]:
    url = archive_url(job)
    filename = url.rsplit("/", 1)[-1]
    checksum_bytes = fetch(url + ".CHECKSUM")
    expected = expected_checksum(checksum_bytes, filename)
    payload = fetch(url)
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise ValueError(f"HVCMDM archive checksum mismatch: {filename}")
    reduced = parse_archive(payload, job)
    ledger = {
        "day": job.day, "symbol": job.symbol, "url": url, "sha256": observed,
        "bytes": len(payload), "rows": int(reduced.snapshots.sum()), "half_hours": len(reduced),
    }
    return reduced, ledger


def jobs() -> list[Job]:
    required: set[tuple[pd.Timestamp, str]] = set()
    decisions = pd.date_range(START + pd.Timedelta(minutes=30), END, freq="30min", inclusive="left")
    for decision in decisions:
        for symbol in contract_pair(decision):
            required.add(((decision - pd.Timedelta(minutes=30)).floor("D"), symbol))
    return [Job(day, symbol) for day, symbol in sorted(required)]


def strict_prior_midrank(values: pd.Series, lookback: int = 1440, minimum: int = 480) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, current in numeric.items():
        prior = history[-lookback:]
        if math.isfinite(current) and len(prior) >= minimum:
            array = np.asarray(prior)
            output.at[index] = (np.sum(array < current) + 0.5 * np.sum(array == current)) / len(array)
        if math.isfinite(current):
            history.append(current)
    return output


def btc_variations(bars: pd.DataFrame) -> pd.Series:
    frame = bars.copy()
    frame["ts"] = pd.to_datetime(frame.ts, utc=True)
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.drop_duplicates("ts", keep=False).set_index("ts").sort_index()
    rows: dict[pd.Timestamp, float] = {}
    for decision in pd.date_range(START + pd.Timedelta(minutes=30), END, freq="30min", inclusive="left"):
        expected = pd.date_range(decision - pd.Timedelta(minutes=30), decision, freq="1min", inclusive="left")
        window = frame.reindex(expected)
        ohlc = window[["open", "high", "low", "close"]]
        valid = bool(
            len(window) == 30 and np.isfinite(ohlc).all(axis=1).all() and ohlc.gt(0).all(axis=1).all()
            and window.high.ge(window[["open", "close"]].max(axis=1)).all()
            and window.low.le(window[["open", "close"]].min(axis=1)).all() and window.high.ge(window.low).all()
        )
        variation = float("nan")
        if valid:
            returns = np.diff(np.log(window.close.to_numpy(float)))
            variation = float(np.square(returns).sum())
            if not math.isfinite(variation) or variation <= 0:
                variation = float("nan")
        rows[decision] = variation
    return pd.Series(rows, name="btc_variation")


def feature_panel(reduced: pd.DataFrame, variations: pd.Series) -> pd.DataFrame:
    indexed = reduced.set_index(["interval_start", "symbol"]).sort_index()
    rows: list[dict[str, Any]] = []
    for decision in pd.date_range(START + pd.Timedelta(minutes=30), END, freq="30min", inclusive="left"):
        interval = decision - pd.Timedelta(minutes=30)
        near, far = contract_pair(decision)
        values: dict[str, Any] = {"decision_time": decision, "near_contract": near, "far_contract": far}
        valid = True
        selected: dict[str, pd.Series] = {}
        for label, symbol in (("near", near), ("far", far)):
            try:
                row = indexed.loc[(interval, symbol)]
            except KeyError:
                valid = False
                break
            if isinstance(row, pd.DataFrame) or int(row.snapshots) < 40:
                valid = False
                break
            first = pd.Timestamp(row.first_snapshot)
            last = pd.Timestamp(row.last_snapshot)
            if first > interval + pd.Timedelta(minutes=1) or last < decision - pd.Timedelta(minutes=1):
                valid = False
                break
            selected[label] = row
        near_pressure = far_pressure = term_pressure = far_share = float("nan")
        if valid:
            near_pressure = float(selected["near"].pressure)
            far_pressure = float(selected["far"].pressure)
            term_pressure = far_pressure - near_pressure
            denominator = float(selected["near"].mass + selected["far"].mass)
            far_share = float(selected["far"].mass / denominator) if denominator > 0 else float("nan")
            valid = all(math.isfinite(x) for x in (near_pressure, far_pressure, term_pressure, far_share)) and term_pressure != 0
        variation = float(variations.get(decision, np.nan))
        valid = valid and math.isfinite(variation)
        rows.append({
            **values, "source_valid": valid, "near_pressure": near_pressure, "far_pressure": far_pressure,
            "term_pressure": term_pressure, "far_share": far_share, "btc_variation": variation,
        })
    panel = pd.DataFrame(rows)
    prior_valid = panel.source_valid.shift(1, fill_value=False)
    panel["migration"] = (panel.far_share - panel.far_share.shift(1)).where(panel.source_valid & prior_valid)
    panel["absolute_migration_rank"] = strict_prior_midrank(panel.migration.abs())
    panel["absolute_term_pressure_rank"] = strict_prior_midrank(panel.term_pressure.abs().where(panel.source_valid))
    panel["btc_variation_rank"] = strict_prior_midrank(panel.btc_variation.where(panel.source_valid))
    return panel


def candidate_clock(panel: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    frame = panel.copy()
    migration_rank = frame.absolute_migration_rank
    pressure_rank = frame.absolute_term_pressure_rank
    variation_rank = frame.btc_variation_rank
    pressure = frame.term_pressure
    valid = frame.source_valid
    available = frame.decision_time
    if control == "one_decision_stale_features":
        migration_rank, pressure_rank, variation_rank, pressure, valid = (
            value.shift(1) for value in (migration_rank, pressure_rank, variation_rank, pressure, valid)
        )
        valid = valid.fillna(False)
        available = frame.decision_time - pd.Timedelta(minutes=30)
    volatility_gate = pd.Series(True, index=frame.index) if control == "no_volatility_gate" else variation_rank.ge(0.65)
    migration_gate = pd.Series(True, index=frame.index) if control == "no_migration_gate" else migration_rank.ge(0.85)
    pressure_gate = pressure_rank.ge(0.75)
    eligible = valid & pressure.notna() & pressure.ne(0) & volatility_gate & migration_gate & pressure_gate
    onset = eligible & ~eligible.shift(1, fill_value=False)
    side_pressure = frame.near_pressure if control == "near_pressure_only" else pressure
    side = pd.Series(np.where(side_pressure.gt(0), 1, -1), index=frame.index)
    if control == "direction_flip":
        side = -side
    elif control == "forced_long":
        side = pd.Series(1, index=frame.index)
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for index in frame.index[onset & frame.decision_time.ge(SPLITS["train"][0])]:
        decision = pd.Timestamp(frame.at[index, "decision_time"])
        entry, exit_ = decision + pd.Timedelta(minutes=5), decision + pd.Timedelta(minutes=5, hours=8)
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_ <= end), None)
        if split is None:
            continue
        reserved_until = exit_
        rows.append({
            "candidate": "HVCMDM-8", "control": control, "split": split, "decision_time": decision,
            "feature_available_time": pd.Timestamp(available.at[index]), "entry_time": entry, "exit_time": exit_,
            "side": int(side.at[index]), "near_contract": frame.at[index, "near_contract"],
            "far_contract": frame.at[index, "far_contract"], "near_pressure": float(frame.at[index, "near_pressure"]),
            "far_pressure": float(frame.at[index, "far_pressure"]), "term_pressure": float(pressure.at[index]),
            "far_share": float(frame.at[index, "far_share"]), "migration": float(frame.at[index, "migration"]),
            "absolute_migration_rank": float(migration_rank.at[index]),
            "absolute_term_pressure_rank": float(pressure_rank.at[index]),
            "btc_variation": float(frame.at[index, "btc_variation"]),
            "btc_variation_rank": float(variation_rank.at[index]),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def stats(candidate: pd.DataFrame, split: str) -> dict[str, float | int]:
    subset = candidate[candidate.split.eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs, shorts = int(subset.side.eq(1).sum()), int(subset.side.eq(-1).sum())
    months = subset.entry_time.dt.strftime("%Y-%m").value_counts()
    return {"events": len(subset), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(subset), "max_month_share": int(months.max()) / len(subset)}


def run(workers: int = 12) -> dict[str, Any]:
    from sqlalchemy import text

    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA256 or json.loads(prereg.DEFAULT_OUTPUT.read_text()) != prereg.build():
        raise RuntimeError("HVCMDM preregistration drift")
    registration = prereg.build()
    all_jobs = jobs()
    parts: list[pd.DataFrame] = []
    ledgers: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_job, job): job for job in all_jobs}
        for future in concurrent.futures.as_completed(futures):
            reduced, ledger = future.result()
            parts.append(reduced)
            ledgers.append(ledger)
    reduced = pd.concat(parts, ignore_index=True).sort_values(["interval_start", "symbol"]).reset_index(drop=True)
    ledger = pd.DataFrame(ledgers).sort_values(["day", "symbol"]).reset_index(drop=True)
    database = postgres_engine()
    with database.connect() as connection:
        bars = pd.read_sql_query(text(QUERY), connection, params={"start": START.to_pydatetime(), "end": END.to_pydatetime()})
    database.dispose()
    panel = feature_panel(reduced, btc_variations(bars))
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(panel, PANEL)
    _write_gzip_csv(ledger, LEDGER)
    source_core = {
        "protocol_version": "hvcmdm_8_source_materialization_v1",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA256, "manifest_hash": registration["manifest_hash"]},
        "official_archive_root": BASE_URL, "required_jobs": len(all_jobs), "all_archive_checksums_verified": True,
        "archive_ledger": {"path": str(LEDGER), "sha256": sha(LEDGER), "rows": len(ledger), "zip_bytes": int(ledger.bytes.sum())},
        "panel": {"path": str(PANEL), "sha256": sha(PANEL), "rows": len(panel), "valid_rows": int(panel.source_valid.sum())},
        "btc_query": QUERY, "oos_postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
    }
    source = {**source_core, "manifest_hash": canonical_hash(source_core)}
    SOURCE_MANIFEST.write_text(json.dumps(source, indent=2, allow_nan=False) + "\n")
    primary = candidate_clock(panel)
    controls = {name: candidate_clock(panel, name) for name in CONTROLS}
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, CLOCK)
    for name, candidate in controls.items():
        _write_gzip_csv(candidate, CONTROL_DIR / f"{name}.csv.gz")
    support = {name: stats(primary, name) for name in SPLITS}
    checks: dict[str, bool] = {"every_required_contract_day_checksum_verified": True}
    for name, values in support.items():
        checks[f"{name}_minimum_events"] = values["events"] >= MINIMUM[name]
        checks[f"{name}_side_balance"] = values["minority_side_share"] >= 0.20
        checks[f"{name}_month_concentration"] = values["max_month_share"] <= 0.45
    passed = all(checks.values())
    core = {
        "protocol_version": "hvcmdm_8_oos_source_support_v1", "policy_id": "HVCMDM-8",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA256, "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": source["manifest_hash"]},
        "completed_preentry_sources_opened": True, "oos_postentry_return_pnl_execution_price_opened": False, "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"), "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(candidate), "promotion_authorized": False} for name, candidate in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    result = run(args.workers)
    print(json.dumps({"passed": result["support_passed"], "support": result["support"]}, indent=2))
