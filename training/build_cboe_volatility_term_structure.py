"""Freeze a research-safe Cboe VIX term-structure source panel.

Only official Cboe daily index-history CSVs are read.  The downloaded responses
are validated against the research-day hashes, trimmed to 2018-2023, normalized,
and stored as deterministic gzip snapshots.  No crypto market, funding, return,
portfolio, or label data is read by this module.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any


BASE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices"
USER_AGENT = "rllm-cboe-term-structure-freeze/1.0"
SCHEMA_VERSION = 1
SOURCE_SNAPSHOT_DATE = "2026-07-17"
START_DATE = date(2018, 1, 1)
END_DATE_EXCLUSIVE = date(2024, 1, 1)
SYMBOLS = ("VIX9D", "VIX", "VIX3M")
SOURCE_COLUMNS = ("DATE", "OPEN", "HIGH", "LOW", "CLOSE")
SNAPSHOT_COLUMNS = ("date", "open", "high", "low", "close")
PANEL_COLUMNS = ("observation_date", "VIX9D_close", "VIX_close", "VIX3M_close")

FROZEN_RESPONSE_SHA256 = {
    "VIX": "fc6b8872599fde02d2ab0dc04ce1f9bd2a4cb07f1883a4fb45cf284b7cbda283",
    "VIX9D": "efd8cbe751e9fc221604324530d3365d3bccc2b72e7ef869e538318799b969e6",
    "VIX3M": "bff0015354ad4507424c2d03c0d9709737779e87e4ed840492532deedc76880a",
}
FROZEN_SOURCE_COVERAGE: dict[str, tuple[int, str, str]] = {
    "VIX": (1_521, "2018-01-02", "2023-12-29"),
    "VIX9D": (1_509, "2018-01-02", "2023-12-29"),
    "VIX3M": (1_509, "2018-01-02", "2023-12-29"),
}
# Filled from the deterministic normalized snapshots and panel.  Empty values
# are permitted only while bootstrapping a new source freeze and are rejected by
# validate_frozen_artifacts, which is used by the downstream preregistration.
FROZEN_SNAPSHOT_SHA256 = {
    "VIX": "f57273e856406ef86550c0555f330e7cf0ac18651dbfe4f3bad327ff4492b420",
    "VIX9D": "e5f15622ae9fe65bde898dfdab2c4061a0aaceb0dd7e28865f6db28bbf973d6c",
    "VIX3M": "9d8442ab8d20b87e915d666e3aab3bd0a6dc4dae8774dd554e5ce47d78e39be7",
}
FROZEN_PANEL_SHA256 = "6f1b2f7f3a5b1e4d5001d673e6ff54374791879c278248ce27b3d610e4f75dc7"
FROZEN_PANEL_COVERAGE = (1_509, "2018-01-02", "2023-12-29")


@dataclass(frozen=True)
class BuildConfig:
    output_dir: str = "data/cboe_volatility_term_structure_2018_2023"
    retries: int = 5
    timeout_seconds: int = 60
    from_snapshot: bool = False


def source_url(symbol: str) -> str:
    if symbol not in SYMBOLS:
        raise ValueError(f"unsupported Cboe index: {symbol}")
    return f"{BASE_URL}/{symbol}_History.csv"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _fetch_bytes(url: str, *, retries: int, timeout: int) -> bytes:
    error: BaseException | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={"Accept": "text/csv", "User-Agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            error = exc
        if attempt + 1 < retries:
            time.sleep(min(8.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"failed to fetch {url} after {retries} attempts") from error


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%m/%d/%Y").date()
    except ValueError as exc:
        raise ValueError(f"invalid Cboe DATE: {value!r}") from exc


def _number(value: str, *, field: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"Cboe {field} must be numeric") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"Cboe {field} must be finite and positive")
    return result


def normalize_response(payload: bytes, *, symbol: str) -> bytes:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Cboe response is not UTF-8 CSV") from exc
    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != SOURCE_COLUMNS:
        raise ValueError("Cboe source schema changed")
    rows: list[dict[str, str]] = []
    seen: set[date] = set()
    for raw in reader:
        observation = _parse_date(raw["DATE"])
        if observation < START_DATE or observation >= END_DATE_EXCLUSIVE:
            continue
        if observation in seen:
            raise ValueError(f"duplicate Cboe date for {symbol}: {observation}")
        seen.add(observation)
        opening = _number(raw["OPEN"], field="OPEN")
        high = _number(raw["HIGH"], field="HIGH")
        low = _number(raw["LOW"], field="LOW")
        close = _number(raw["CLOSE"], field="CLOSE")
        if high < max(opening, close) or low > min(opening, close) or high < low:
            raise ValueError(f"Cboe OHLC invariant failed for {symbol} {observation}")
        rows.append(
            {
                "date": observation.isoformat(),
                "open": format(opening, ".6f"),
                "high": format(high, ".6f"),
                "low": format(low, ".6f"),
                "close": format(close, ".6f"),
            }
        )
    rows.sort(key=lambda row: row["date"])
    expected_count, expected_first, expected_last = FROZEN_SOURCE_COVERAGE[symbol]
    if len(rows) != expected_count:
        raise ValueError(
            f"unexpected {symbol} source count: {len(rows)} != {expected_count}"
        )
    if rows[0]["date"] != expected_first or rows[-1]["date"] != expected_last:
        raise ValueError(f"unexpected {symbol} source coverage")
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(SNAPSHOT_COLUMNS)
    writer.writerows(tuple(row[column] for column in SNAPSHOT_COLUMNS) for row in rows)
    return output.getvalue().encode()


def parse_snapshot(payload: bytes, *, symbol: str) -> dict[str, str]:
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    if tuple(reader.fieldnames or ()) != SNAPSHOT_COLUMNS:
        raise ValueError("normalized Cboe snapshot schema changed")
    result: dict[str, str] = {}
    for raw in reader:
        observation = date.fromisoformat(raw["date"])
        if observation < START_DATE or observation >= END_DATE_EXCLUSIVE:
            raise ValueError("normalized Cboe snapshot escaped the research horizon")
        if raw["date"] in result:
            raise ValueError(f"duplicate normalized Cboe date for {symbol}")
        _number(raw["open"], field="open")
        _number(raw["high"], field="high")
        _number(raw["low"], field="low")
        close = _number(raw["close"], field="close")
        result[raw["date"]] = format(close, ".6f")
    expected_count, expected_first, expected_last = FROZEN_SOURCE_COVERAGE[symbol]
    ordered = sorted(result)
    if len(ordered) != expected_count or ordered[0] != expected_first or ordered[-1] != expected_last:
        raise ValueError(f"normalized {symbol} snapshot coverage changed")
    return result


def build_panel(snapshots: dict[str, bytes]) -> bytes:
    parsed = {symbol: parse_snapshot(payload, symbol=symbol) for symbol, payload in snapshots.items()}
    dates = sorted(set.intersection(*(set(values) for values in parsed.values())))
    expected_count, expected_first, expected_last = FROZEN_PANEL_COVERAGE
    if len(dates) != expected_count or dates[0] != expected_first or dates[-1] != expected_last:
        raise ValueError("Cboe panel intersection coverage changed")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=PANEL_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for observation in dates:
        writer.writerow(
            {
                "observation_date": observation,
                "VIX9D_close": parsed["VIX9D"][observation],
                "VIX_close": parsed["VIX"][observation],
                "VIX3M_close": parsed["VIX3M"][observation],
            }
        )
    return output.getvalue().encode()


def write_gzip(path: str | Path, payload: bytes) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as handle:
            handle.write(payload)


def read_gzip(path: str | Path) -> bytes:
    with gzip.open(path, "rb") as handle:
        return handle.read()


def artifact_paths(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    paths = {
        symbol: root / "source" / f"{symbol}_History_2018_2023.csv.gz"
        for symbol in SYMBOLS
    }
    paths["panel"] = root / "cboe_vix_term_structure_2018-01-01_2023-12-31.csv.gz"
    paths["manifest"] = root / "build_manifest.json"
    return paths


def validate_frozen_artifacts(output_dir: str | Path) -> dict[str, Any]:
    if not all(FROZEN_SNAPSHOT_SHA256.values()) or not FROZEN_PANEL_SHA256:
        raise RuntimeError("Cboe frozen artifact hashes are not initialized")
    paths = artifact_paths(output_dir)
    snapshots: dict[str, bytes] = {}
    for symbol in SYMBOLS:
        if sha256_file(paths[symbol]) != FROZEN_SNAPSHOT_SHA256[symbol]:
            raise RuntimeError(f"frozen Cboe {symbol} snapshot hash changed")
        snapshots[symbol] = read_gzip(paths[symbol])
    panel = build_panel(snapshots)
    if sha256_file(paths["panel"]) != FROZEN_PANEL_SHA256:
        raise RuntimeError("frozen Cboe panel hash changed")
    if read_gzip(paths["panel"]) != panel:
        raise RuntimeError("frozen Cboe panel no longer reproduces from snapshots")
    manifest = json.loads(paths["manifest"].read_text())
    core = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if manifest.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("Cboe source manifest hash changed")
    return manifest


def build(config: BuildConfig = BuildConfig()) -> dict[str, Any]:
    paths = artifact_paths(config.output_dir)
    snapshots: dict[str, bytes] = {}
    response_records: dict[str, Any] = {}
    for symbol in SYMBOLS:
        if config.from_snapshot:
            payload = read_gzip(paths[symbol])
            expected = FROZEN_SNAPSHOT_SHA256[symbol]
            if not expected or sha256_file(paths[symbol]) != expected:
                raise RuntimeError(f"offline Cboe {symbol} snapshot hash changed")
            snapshots[symbol] = payload
            response_records[symbol] = {
                "url": source_url(symbol),
                "full_response_sha256": FROZEN_RESPONSE_SHA256[symbol],
                "network_read": False,
            }
        else:
            response = _fetch_bytes(
                source_url(symbol),
                retries=config.retries,
                timeout=config.timeout_seconds,
            )
            actual_response = sha256_bytes(response)
            if actual_response != FROZEN_RESPONSE_SHA256[symbol]:
                raise RuntimeError(
                    f"Cboe {symbol} response changed; use the frozen snapshot or audit a new vintage"
                )
            payload = normalize_response(response, symbol=symbol)
            write_gzip(paths[symbol], payload)
            snapshots[symbol] = payload
            response_records[symbol] = {
                "url": source_url(symbol),
                "full_response_sha256": actual_response,
                "network_read": True,
            }
    panel = build_panel(snapshots)
    write_gzip(paths["panel"], panel)
    snapshot_records = {
        symbol: {
            **response_records[symbol],
            "path": str(paths[symbol]),
            "sha256": sha256_file(paths[symbol]),
            "coverage": {
                "rows": FROZEN_SOURCE_COVERAGE[symbol][0],
                "first": FROZEN_SOURCE_COVERAGE[symbol][1],
                "last": FROZEN_SOURCE_COVERAGE[symbol][2],
            },
        }
        for symbol in SYMBOLS
    }
    core: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_snapshot_date": SOURCE_SNAPSHOT_DATE,
        "builder": "training/build_cboe_volatility_term_structure.py",
        "config": asdict(config),
        "official_sources": snapshot_records,
        "source_contract": {
            "provider": "Cboe Global Indices",
            "indices": list(SYMBOLS),
            "research_horizon": [START_DATE.isoformat(), END_DATE_EXCLUSIVE.isoformat()],
            "market_or_label_rows_read": 0,
            "future_source_rows_retained": 0,
        },
        "panel": {
            "path": str(paths["panel"]),
            "sha256": sha256_file(paths["panel"]),
            "rows": FROZEN_PANEL_COVERAGE[0],
            "first": FROZEN_PANEL_COVERAGE[1],
            "last": FROZEN_PANEL_COVERAGE[2],
            "columns": list(PANEL_COLUMNS),
        },
        "created_at": "2026-07-17T15:00:00+00:00",
    }
    manifest = {**core, "manifest_hash": canonical_hash(core)}
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=BuildConfig.output_dir)
    parser.add_argument("--from-snapshot", action="store_true")
    args = parser.parse_args()
    report = build(
        BuildConfig(output_dir=args.output_dir, from_snapshot=args.from_snapshot)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
