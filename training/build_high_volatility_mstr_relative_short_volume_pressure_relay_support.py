"""Open source-only OOS incidence for preregistered HVMRSVP-24."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_mstr_relative_short_volume_pressure_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA256 = "4e51be73a940bffa9a4cf9534c2802490322d12e85c8f1feb3a833434db877eb"
TEMPLATE = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"
SOURCE_DIR = Path("data/high_volatility_mstr_relative_short_volume_pressure_relay_sources_2023_2026")
PAIR_PANEL = SOURCE_DIR / "mstr_qqq_short_volume_panel.csv.gz"
FEATURE_PANEL = SOURCE_DIR / "mstr_qqq_feature_panel.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_mstr_relative_short_volume_pressure_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_mstr_relative_short_volume_pressure_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_mstr_relative_short_volume_pressure_relay_support_2026-08-10.json")
QUERY = """
SELECT ts,open,high,low,close
FROM bars_binance
WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end
ORDER BY ts
"""
SPLITS = {
    "train": (pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2024-01-01T00:00:00Z")),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
CONTROLS = (
    "no_volatility_gate", "no_pressure_magnitude_gate", "mstr_share_change_only",
    "one_source_day_stale_features", "direction_flip", "forced_long",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "source_date", "feature_available_time", "decision_time",
    "entry_time", "exit_time", "side", "mstr_short_volume", "mstr_total_volume",
    "qqq_short_volume", "qqq_total_volume", "mstr_short_share", "qqq_short_share",
    "relative_pressure", "pressure_change", "mstr_share_change", "absolute_pressure_change_rank",
    "realized_variation", "realized_variation_rank",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def strict_prior_midrank(values: pd.Series, lookback: int = 252, minimum: int = 126) -> pd.Series:
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


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def parse_target_rows(raw: bytes, source_date: pd.Timestamp) -> list[dict[str, Any]]:
    text = raw.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text), delimiter="|")
    expected = ["Date", "Symbol", "ShortVolume", "ShortExemptVolume", "TotalVolume", "Market"]
    if reader.fieldnames != expected:
        raise ValueError("FINRA daily file schema drift")
    targets: list[dict[str, Any]] = []
    date_text = source_date.strftime("%Y%m%d")
    for row in reader:
        if row.get("Symbol") not in {"MSTR", "QQQ"}:
            continue
        if set(row) != set(expected) or row["Date"] != date_text:
            raise ValueError("FINRA target row identity invalid")
        short = float(row["ShortVolume"])
        total = float(row["TotalVolume"])
        exempt = float(row["ShortExemptVolume"])
        if not all(math.isfinite(value) for value in (short, total, exempt)):
            raise ValueError("FINRA target row nonfinite")
        if short < 0 or exempt < 0 or total <= 0 or short > total:
            raise ValueError("FINRA target row volume invalid")
        targets.append({
            "source_date": source_date, "symbol": row["Symbol"],
            "short_volume": short, "short_exempt_volume": exempt, "total_volume": total,
            "market": row["Market"],
        })
    if len(targets) != 2 or {row["symbol"] for row in targets} != {"MSTR", "QQQ"}:
        raise RuntimeError("FINRA source day lacks exact MSTR/QQQ pair")
    return targets


def download_date(source_date: pd.Timestamp) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    url = TEMPLATE.format(date=source_date.strftime("%Y%m%d"))
    request = urllib.request.Request(url, headers={"Accept-Encoding": "identity", "User-Agent": "rllm-source-audit/1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 200:
                raise RuntimeError(f"FINRA HTTP status {response.status}")
            raw = response.read()
            headers = response.headers
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None, {"url": url, "status": 404}
        raise
    rows = parse_target_rows(raw, source_date)
    return rows, {
        "url": url, "status": 200, "response_sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw), "etag": headers.get("ETag"), "last_modified": headers.get("Last-Modified"),
    }


def download_pair_panel() -> tuple[pd.DataFrame, dict[str, Any]]:
    dates = list(pd.date_range(START, END, freq="D", inclusive="left"))
    weekdays = [date for date in dates if date.weekday() < 5]
    with ThreadPoolExecutor(max_workers=8) as executor:
        downloaded = list(executor.map(download_date, weekdays))
    rows: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    missing: list[str] = []
    for date, (target_rows, binding) in zip(weekdays, downloaded, strict=True):
        if target_rows is None:
            missing.append(date.strftime("%Y-%m-%d"))
        else:
            rows.extend(target_rows)
            files.append(binding)
    frame = pd.DataFrame(rows)
    if frame.duplicated(["source_date", "symbol"], keep=False).any():
        raise RuntimeError("duplicate FINRA target date/symbol")
    by_symbol = {
        symbol: frame[frame.symbol.eq(symbol)].set_index("source_date").sort_index()
        for symbol in ("MSTR", "QQQ")
    }
    if not by_symbol["MSTR"].index.equals(by_symbol["QQQ"].index):
        raise RuntimeError("FINRA MSTR/QQQ source-day mismatch")
    index = by_symbol["MSTR"].index
    pair = pd.DataFrame({
        "source_date": index,
        "mstr_short_volume": by_symbol["MSTR"].short_volume.to_numpy(float),
        "mstr_total_volume": by_symbol["MSTR"].total_volume.to_numpy(float),
        "qqq_short_volume": by_symbol["QQQ"].short_volume.to_numpy(float),
        "qqq_total_volume": by_symbol["QQQ"].total_volume.to_numpy(float),
    })
    pair["feature_available_time"] = pair.source_date + pd.Timedelta(days=1)
    return pair, {"files": files, "missing_weekdays": missing, "source_days": len(pair)}


def feature_panel(pair: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    market = bars.copy()
    market["ts"] = pd.to_datetime(market.ts, utc=True)
    for column in ("open", "high", "low", "close"):
        market[column] = pd.to_numeric(market[column], errors="coerce")
    if market.duplicated("ts", keep=False).any():
        raise RuntimeError("duplicate BTCUSDT 1m bars")
    market = market.set_index("ts").sort_index()
    frame = pair.copy()
    variations: list[float] = []
    valid: list[bool] = []
    for decision in frame.feature_available_time:
        expected = pd.date_range(decision - pd.Timedelta(days=1), decision, freq="1min", inclusive="left")
        window = market.reindex(expected)
        ohlc = window[["open", "high", "low", "close"]]
        good = bool(
            np.isfinite(ohlc).all(axis=1).all() and ohlc.gt(0).all(axis=1).all()
            and window.high.ge(window[["open", "close"]].max(axis=1)).all()
            and window.low.le(window[["open", "close"]].min(axis=1)).all()
            and window.high.ge(window.low).all()
        )
        variation = float("nan")
        if good:
            variation = float(np.square(np.diff(np.log(window.close.to_numpy(float)))).sum())
            good = math.isfinite(variation) and variation > 0
        variations.append(variation)
        valid.append(good)
    frame["source_valid"] = valid
    frame["mstr_short_share"] = frame.mstr_short_volume / frame.mstr_total_volume
    frame["qqq_short_share"] = frame.qqq_short_volume / frame.qqq_total_volume
    clip = lambda values: values.clip(1e-9, 1 - 1e-9)
    frame["relative_pressure"] = np.log(clip(frame.mstr_short_share) / (1 - clip(frame.mstr_short_share))) - np.log(clip(frame.qqq_short_share) / (1 - clip(frame.qqq_short_share)))
    frame["pressure_change"] = frame.relative_pressure.diff()
    frame["mstr_share_change"] = frame.mstr_short_share.diff()
    frame["realized_variation"] = variations
    frame["absolute_pressure_change_rank"] = strict_prior_midrank(frame.pressure_change.abs().where(frame.source_valid))
    frame["realized_variation_rank"] = strict_prior_midrank(frame.realized_variation.where(frame.source_valid))
    return frame


def candidate_clock(panel: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    frame = panel.copy()
    pressure = frame.pressure_change
    pressure_rank = frame.absolute_pressure_change_rank
    variation_rank = frame.realized_variation_rank
    side_source = pressure
    valid = frame.source_valid
    source_date = frame.source_date
    if control == "one_source_day_stale_features":
        pressure = pressure.shift(1); pressure_rank = pressure_rank.shift(1)
        variation_rank = variation_rank.shift(1); valid = valid.shift(1, fill_value=False)
        side_source = pressure; source_date = source_date.shift(1)
    if control == "mstr_share_change_only":
        side_source = frame.mstr_share_change
    vol_gate = pd.Series(True, index=frame.index) if control == "no_volatility_gate" else variation_rank.ge(0.65)
    pressure_gate = pd.Series(True, index=frame.index) if control == "no_pressure_magnitude_gate" else pressure_rank.ge(0.80)
    eligible = valid & side_source.notna() & side_source.ne(0) & vol_gate & pressure_gate
    onset = eligible & ~eligible.shift(1, fill_value=False)
    side = pd.Series(np.where(side_source.gt(0), -1, 1), index=frame.index)
    if control == "direction_flip": side = -side
    elif control == "forced_long": side = pd.Series(1, index=frame.index)
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for index in frame.index[onset]:
        decision = pd.Timestamp(frame.at[index, "feature_available_time"])
        entry = decision + pd.Timedelta(minutes=5); exit_ = entry + pd.Timedelta(hours=24)
        if reserved_until is not None and entry < reserved_until: continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_ <= end), None)
        if split is None: continue
        reserved_until = exit_
        rows.append({
            "candidate": "HVMRSVP-24", "control": control, "split": split,
            "source_date": source_date.at[index], "feature_available_time": decision,
            "decision_time": decision, "entry_time": entry, "exit_time": exit_, "side": int(side.at[index]),
            "mstr_short_volume": float(frame.at[index, "mstr_short_volume"]), "mstr_total_volume": float(frame.at[index, "mstr_total_volume"]),
            "qqq_short_volume": float(frame.at[index, "qqq_short_volume"]), "qqq_total_volume": float(frame.at[index, "qqq_total_volume"]),
            "mstr_short_share": float(frame.at[index, "mstr_short_share"]), "qqq_short_share": float(frame.at[index, "qqq_short_share"]),
            "relative_pressure": float(frame.at[index, "relative_pressure"]), "pressure_change": float(pressure.at[index]),
            "mstr_share_change": float(frame.at[index, "mstr_share_change"]), "absolute_pressure_change_rank": float(pressure_rank.at[index]),
            "realized_variation": float(frame.at[index, "realized_variation"]), "realized_variation_rank": float(variation_rank.at[index]),
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stage_stats(candidate: pd.DataFrame, split: str) -> dict[str, Any]:
    subset = candidate[candidate.split.eq(split)]
    if subset.empty: return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(subset.side.eq(1).sum()); shorts = int(subset.side.eq(-1).sum())
    months = subset.entry_time.dt.strftime("%Y-%m").value_counts()
    return {"events": len(subset), "longs": longs, "shorts": shorts, "minority_side_share": min(longs, shorts) / len(subset), "max_month_share": float(months.max() / len(subset))}


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA256: raise RuntimeError("HVMRSVP preregistration drift")
    pair, transport = download_pair_panel()
    engine = postgres_engine()
    try:
        bars = pd.read_sql_query(QUERY, engine, params={"start": START, "end": END})
    finally:
        engine.dispose()
    panel = feature_panel(pair, bars); candidate = candidate_clock(panel)
    controls = {name: candidate_clock(panel, name) for name in CONTROLS}
    stats = {name: stage_stats(candidate, name) for name in SPLITS}
    checks = {name: {"minimum_events": item["events"] >= MINIMUM[name], "minority_side_share": item["minority_side_share"] >= 0.20, "max_month_share": item["max_month_share"] <= 0.45} for name, item in stats.items()}
    passed = all(all(item.values()) for item in checks.values())
    SOURCE_DIR.mkdir(parents=True, exist_ok=True); CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(pair, PAIR_PANEL); _write_gzip_csv(panel, FEATURE_PANEL); _write_gzip_csv(candidate, CLOCK)
    control_bindings = {}
    for name, clock in controls.items():
        path = CONTROL_DIR / f"{name}.csv.gz"; _write_gzip_csv(clock, path)
        control_bindings[name] = {"path": str(path), "sha256": sha(path), "rows": len(clock)}
    manifest = {"transport": transport, "pair_panel": {"path": str(PAIR_PANEL), "sha256": sha(PAIR_PANEL), "rows": len(pair)}, "feature_panel": {"path": str(FEATURE_PANEL), "sha256": sha(FEATURE_PANEL), "rows": len(panel)}, "bars_query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(), "bars_rows": len(bars)}
    manifest["manifest_hash"] = canonical_hash(manifest); SOURCE_MANIFEST.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    result = {"protocol_version": "high_volatility_mstr_relative_short_volume_pressure_relay_support_v1", "policy_id": "HVMRSVP-24", "as_of_date": "2026-08-10", "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA256}, "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": manifest["manifest_hash"]}, "candidate_clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(candidate)}, "controls": control_bindings, "stage_stats": stats, "gate_checks": checks, "decision": {"pass": passed, "status": "pass_to_novelty" if passed else "terminal_source_support_reject"}, "outcomes_opened": False, "gross9_rows_opened": False}
    result["manifest_hash"] = canonical_hash(result); RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
