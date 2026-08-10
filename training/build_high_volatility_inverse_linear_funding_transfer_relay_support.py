"""Open source-only OOS incidence for preregistered HVILFT-8."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_inverse_linear_funding_transfer_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2023-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA256 = "ebeb5b82c1bf9345a327ffe141bc3570b264880017b56a5613b9dd942a3f3f31"
VISION = "https://data.binance.vision/data/futures"
CM_REST = "https://dapi.binance.com/dapi/v1/fundingRate"
SOURCE_DIR = Path("data/high_volatility_inverse_linear_funding_transfer_relay_sources_2023_2026")
PAIR_PANEL = SOURCE_DIR / "paired_funding_panel.csv.gz"
FEATURE_PANEL = SOURCE_DIR / "paired_funding_feature_panel.csv.gz"
SOURCE_MANIFEST = SOURCE_DIR / "manifest.json"
CLOCK = Path("data/high_volatility_inverse_linear_funding_transfer_relay_clocks_2023_2026.csv.gz")
CONTROL_DIR = Path("data/high_volatility_inverse_linear_funding_transfer_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_inverse_linear_funding_transfer_relay_support_2026-08-10.json")
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
    "no_volatility_gate",
    "no_transfer_magnitude_gate",
    "coinm_funding_change_only",
    "one_settlement_stale_features",
    "direction_flip",
    "forced_long",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "observation_time", "feature_available_time",
    "decision_time", "entry_time", "exit_time", "side", "coinm_funding_rate",
    "usdm_funding_rate", "funding_differential", "funding_transfer",
    "coinm_funding_change", "absolute_transfer_rank", "realized_variation",
    "realized_variation_rank",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def strict_prior_midrank(values: pd.Series, lookback: int = 1095, minimum: int = 540) -> pd.Series:
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


def http_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "rllm-source-audit/1"})
    with urllib.request.urlopen(request, timeout=90) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP status {response.status}: {url}")
        return response.read()


def archive_url(market: str, symbol: str, month: str) -> str:
    name = f"{symbol}-fundingRate-{month}.zip"
    return f"{VISION}/{market}/monthly/fundingRate/{symbol}/{name}"


def verified_archive(url: str) -> tuple[bytes, dict[str, Any]]:
    raw = http_bytes(url)
    checksum_raw = http_bytes(url + ".CHECKSUM")
    fields = checksum_raw.decode("ascii").strip().split()
    if len(fields) < 1 or len(fields[0]) != 64:
        raise RuntimeError(f"invalid Binance checksum response: {url}")
    observed = hashlib.sha256(raw).hexdigest()
    if observed.lower() != fields[0].lower():
        raise RuntimeError(f"Binance archive checksum mismatch: {url}")
    return raw, {
        "url": url,
        "zip_sha256": observed,
        "checksum_url": url + ".CHECKSUM",
        "checksum_response_sha256": hashlib.sha256(checksum_raw).hexdigest(),
    }


def parse_archive(raw: bytes, instrument: str) -> list[dict[str, Any]]:
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        if len(names) != 1:
            raise ValueError("funding archive must contain exactly one CSV")
        body = archive.read(names[0]).decode("utf-8")
    reader = csv.DictReader(io.StringIO(body))
    if reader.fieldnames != ["calc_time", "funding_interval_hours", "last_funding_rate"]:
        raise ValueError("funding archive schema drift")
    return [parse_archive_row(row, instrument) for row in reader]


def parse_archive_row(raw: dict[str, Any], instrument: str) -> dict[str, Any]:
    if set(raw) != {"calc_time", "funding_interval_hours", "last_funding_rate"}:
        raise ValueError("funding archive row schema invalid")
    timestamp_text = raw["calc_time"]
    interval_text = raw["funding_interval_hours"]
    if not isinstance(timestamp_text, str) or not timestamp_text.isdigit():
        raise ValueError("funding calc_time is not integer milliseconds")
    if not isinstance(interval_text, str) or not interval_text.isdigit() or int(interval_text) != 8:
        raise ValueError("funding interval is not exactly eight hours")
    timestamp = pd.Timestamp(int(timestamp_text), unit="ms", tz="UTC")
    rate = float(raw["last_funding_rate"])
    if timestamp != timestamp.floor("5min") or not math.isfinite(rate):
        raise ValueError("funding archive row value invalid")
    return {"instrument": instrument, "calc_time": timestamp, "funding_rate": rate}


def parse_rest_row(raw: Any, instrument: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or not {"symbol", "fundingTime", "fundingRate"}.issubset(raw):
        raise ValueError("funding REST row schema invalid")
    if raw["symbol"] != instrument or type(raw["fundingTime"]) is not int:
        raise ValueError("funding REST identity invalid")
    timestamp = pd.Timestamp(raw["fundingTime"], unit="ms", tz="UTC")
    rate = float(raw["fundingRate"])
    if timestamp != timestamp.floor("5min") or not math.isfinite(rate):
        raise ValueError("funding REST row value invalid")
    return {"instrument": instrument, "calc_time": timestamp, "funding_rate": rate}


def download_rest_tail(start: pd.Timestamp, end: pd.Timestamp) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query = urllib.parse.urlencode({
        "symbol": "BTCUSD_PERP", "startTime": int(start.timestamp() * 1000),
        "endTime": int((end - pd.Timedelta(milliseconds=1)).timestamp() * 1000), "limit": 1000,
    })
    url = CM_REST + "?" + query
    raw = http_bytes(url)
    payload = json.loads(raw)
    if not isinstance(payload, list) or len(payload) >= 1000:
        raise RuntimeError("COIN-M REST tail invalid or potentially truncated")
    return [parse_rest_row(row, "BTCUSD_PERP") for row in payload], {
        "url": url, "response_sha256": hashlib.sha256(raw).hexdigest(), "rows": len(payload)
    }


def months(start: str, end: str) -> list[str]:
    return [str(value) for value in pd.period_range(start, end, freq="M")]


def download_pair_panel() -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    archives: list[dict[str, Any]] = []
    contracts = (
        ("cm", "BTCUSD_PERP", months("2023-01", "2026-06")),
        ("um", "BTCUSDT", months("2023-01", "2026-07")),
    )
    for market, instrument, month_values in contracts:
        for month in month_values:
            url = archive_url(market, instrument, month)
            raw, binding = verified_archive(url)
            parsed = parse_archive(raw, instrument)
            rows.extend(parsed)
            archives.append({**binding, "instrument": instrument, "month": month, "rows": len(parsed)})
    rest_rows, rest_binding = download_rest_tail(pd.Timestamp("2026-07-01T00:00:00Z"), END)
    rows.extend(rest_rows)
    frame = pd.DataFrame(rows)
    frame = frame[(frame.calc_time >= START) & (frame.calc_time < END)].copy()
    if frame.duplicated(["instrument", "calc_time"], keep=False).any():
        raise RuntimeError("duplicate funding timestamp")
    expected = pd.date_range(START, END, freq="8h", inclusive="left")
    by_instrument: dict[str, pd.DataFrame] = {}
    for instrument in ("BTCUSD_PERP", "BTCUSDT"):
        subset = frame[frame.instrument.eq(instrument)].set_index("calc_time").sort_index()
        if not subset.index.equals(expected):
            missing = expected.difference(subset.index)
            extra = subset.index.difference(expected)
            raise RuntimeError(
                f"incomplete exact {instrument} funding grid: missing={len(missing)} extra={len(extra)}"
            )
        by_instrument[instrument] = subset
    pair = pd.DataFrame({
        "calc_time": expected,
        "coinm_funding_rate": by_instrument["BTCUSD_PERP"].funding_rate.to_numpy(float),
        "usdm_funding_rate": by_instrument["BTCUSDT"].funding_rate.to_numpy(float),
    })
    return pair, {"archives": archives, "coinm_rest_tail": rest_binding, "normalized_rows": len(frame)}


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
    source_valid: list[bool] = []
    for timestamp in frame.calc_time:
        expected = pd.date_range(timestamp - pd.Timedelta(days=1), timestamp, freq="1min", inclusive="left")
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
        source_valid.append(good)
    frame["source_valid"] = source_valid
    frame["funding_differential"] = frame.coinm_funding_rate - frame.usdm_funding_rate
    frame["funding_transfer"] = frame.funding_differential.diff()
    frame["coinm_funding_change"] = frame.coinm_funding_rate.diff()
    frame["realized_variation"] = variations
    frame["absolute_transfer_rank"] = strict_prior_midrank(
        frame.funding_transfer.abs().where(frame.source_valid)
    )
    frame["realized_variation_rank"] = strict_prior_midrank(
        frame.realized_variation.where(frame.source_valid)
    )
    return frame


def candidate_clock(panel: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    frame = panel.copy()
    transfer = frame.funding_transfer
    transfer_rank = frame.absolute_transfer_rank
    variation_rank = frame.realized_variation_rank
    side_source = transfer
    valid = frame.source_valid
    observation = frame.calc_time
    if control == "one_settlement_stale_features":
        transfer = transfer.shift(1)
        transfer_rank = transfer_rank.shift(1)
        variation_rank = variation_rank.shift(1)
        valid = valid.shift(1, fill_value=False)
        side_source = transfer
        observation = frame.calc_time.shift(1)
    if control == "coinm_funding_change_only":
        side_source = frame.coinm_funding_change
    volatility_gate = pd.Series(True, index=frame.index) if control == "no_volatility_gate" else variation_rank.ge(0.65)
    magnitude_gate = pd.Series(True, index=frame.index) if control == "no_transfer_magnitude_gate" else transfer_rank.ge(0.80)
    eligible = valid & side_source.notna() & side_source.ne(0) & volatility_gate & magnitude_gate
    onset = eligible & ~eligible.shift(1, fill_value=False)
    side = pd.Series(np.where(side_source.gt(0), -1, 1), index=frame.index)
    if control == "direction_flip":
        side = -side
    elif control == "forced_long":
        side = pd.Series(1, index=frame.index)
    rows: list[dict[str, Any]] = []
    reserved_until: pd.Timestamp | None = None
    for index in frame.index[onset]:
        feature_available = pd.Timestamp(frame.at[index, "calc_time"])
        decision = feature_available + pd.Timedelta(minutes=5)
        entry = decision + pd.Timedelta(minutes=5)
        exit_ = entry + pd.Timedelta(hours=8)
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_ <= end), None)
        if split is None:
            continue
        reserved_until = exit_
        rows.append({
            "candidate": "HVILFT-8", "control": control, "split": split,
            "observation_time": observation.at[index], "feature_available_time": feature_available,
            "decision_time": decision, "entry_time": entry, "exit_time": exit_,
            "side": int(side.at[index]),
            "coinm_funding_rate": float(frame.at[index, "coinm_funding_rate"]),
            "usdm_funding_rate": float(frame.at[index, "usdm_funding_rate"]),
            "funding_differential": float(frame.at[index, "funding_differential"]),
            "funding_transfer": float(transfer.at[index]),
            "coinm_funding_change": float(frame.at[index, "coinm_funding_change"]),
            "absolute_transfer_rank": float(transfer_rank.at[index]),
            "realized_variation": float(frame.at[index, "realized_variation"]),
            "realized_variation_rank": float(variation_rank.at[index]),
        })
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(candidate: pd.DataFrame, split: str) -> dict[str, float | int]:
    subset = candidate[candidate.split.eq(split)]
    if subset.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(subset.side.eq(1).sum())
    shorts = int(subset.side.eq(-1).sum())
    months_ = subset.entry_time.dt.strftime("%Y-%m").value_counts()
    return {
        "events": len(subset), "longs": longs, "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(subset),
        "max_month_share": float(months_.max() / len(subset)),
    }


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA256:
        raise RuntimeError("HVILFT preregistration drift")
    pair, transport = download_pair_panel()
    engine = postgres_engine()
    try:
        bars = pd.read_sql_query(
            QUERY, engine,
            params={"start": START - pd.Timedelta(days=1), "end": END},
        )
    finally:
        engine.dispose()
    panel = feature_panel(pair, bars)
    candidate = candidate_clock(panel)
    controls = {name: candidate_clock(panel, name) for name in CONTROLS}
    stage_stats = {name: stats(candidate, name) for name in SPLITS}
    gate_checks = {
        name: {
            "minimum_events": item["events"] >= MINIMUM[name],
            "minority_side_share": item["minority_side_share"] >= 0.20,
            "max_month_share": item["max_month_share"] <= 0.45,
        }
        for name, item in stage_stats.items()
    }
    passed = all(all(checks.values()) for checks in gate_checks.values())
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(pair, PAIR_PANEL)
    _write_gzip_csv(panel, FEATURE_PANEL)
    _write_gzip_csv(candidate, CLOCK)
    control_bindings: dict[str, Any] = {}
    for name, clock in controls.items():
        path = CONTROL_DIR / f"{name}.csv.gz"
        _write_gzip_csv(clock, path)
        control_bindings[name] = {"path": str(path), "sha256": sha(path), "rows": len(clock)}
    source_manifest = {
        "transport": transport,
        "pair_panel": {"path": str(PAIR_PANEL), "sha256": sha(PAIR_PANEL), "rows": len(pair)},
        "feature_panel": {"path": str(FEATURE_PANEL), "sha256": sha(FEATURE_PANEL), "rows": len(panel)},
        "bars_query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(),
        "bars_rows": len(bars),
    }
    source_manifest["manifest_hash"] = canonical_hash(source_manifest)
    SOURCE_MANIFEST.write_text(json.dumps(source_manifest, indent=2, allow_nan=False) + "\n")
    result = {
        "protocol_version": "high_volatility_inverse_linear_funding_transfer_relay_support_v1",
        "policy_id": "HVILFT-8", "as_of_date": "2026-08-10",
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA256},
        "source_manifest": {"path": str(SOURCE_MANIFEST), "sha256": sha(SOURCE_MANIFEST), "manifest_hash": source_manifest["manifest_hash"]},
        "candidate_clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(candidate)},
        "controls": control_bindings, "stage_stats": stage_stats, "gate_checks": gate_checks,
        "decision": {"pass": passed, "status": "source_support_pass" if passed else "terminal_source_support_rejection"},
        "outcomes_opened": False, "gross9_rows_opened": False,
    }
    result["manifest_hash"] = canonical_hash(result)
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(run(), indent=2, allow_nan=False))
