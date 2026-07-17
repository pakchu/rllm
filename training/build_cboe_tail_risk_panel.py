"""Freeze a research-safe Cboe SKEW/VVIX/VIX source panel.

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
USER_AGENT = "rllm-cboe-tail-risk-freeze/1.0"
SCHEMA_VERSION = 1
SOURCE_SNAPSHOT_DATE = "2026-07-18"
START_DATE = date(2018, 1, 1)
END_DATE_EXCLUSIVE = date(2024, 1, 1)
SYMBOLS = ("SKEW", "VVIX", "VIX")
SOURCE_COLUMNS = {
    "SKEW": ("DATE", "SKEW"),
    "VVIX": ("DATE", "VVIX"),
    "VIX": ("DATE", "OPEN", "HIGH", "LOW", "CLOSE"),
}
VALUE_COLUMN = {"SKEW": "SKEW", "VVIX": "VVIX", "VIX": "CLOSE"}
SNAPSHOT_COLUMNS = ("date", "close")
PANEL_COLUMNS = ("observation_date", "SKEW_close", "VVIX_close", "VIX_close")

FROZEN_RESPONSE_SHA256 = {
    "SKEW": "c2434fa12ceaa749273aa8ef13f8cc192a98e92aad2d11a8df10db26135a49ac",
    "VVIX": "3b0b34f514d5bbf78759538062015250c0f0127b54e9f1005a77fba8632e9bb6",
    "VIX": "fc6b8872599fde02d2ab0dc04ce1f9bd2a4cb07f1883a4fb45cf284b7cbda283",
}
FROZEN_SOURCE_COVERAGE: dict[str, tuple[int, str, str]] = {
    "SKEW": (1_507, "2018-01-02", "2023-12-29"),
    "VVIX": (1_509, "2018-01-02", "2023-12-29"),
    "VIX": (1_521, "2018-01-02", "2023-12-29"),
}
FROZEN_SNAPSHOT_SHA256 = {
    "SKEW": "1fd9fac7b8401ee5b67eedfaa4baad0faa7f43b2cd629e07261e18bbf7685338",
    "VVIX": "fc6fea738e016baa6cbcb40bfc93d7ce0546199eeaa8153f4d84e92b7f0604e5",
    "VIX": "5015e6f5e6a9ae5e1bef3be9c972b76b8a841d1cd1aa584e7e50342039d63cbd",
}
FROZEN_PANEL_SHA256 = "cdde3f8d4bb1e23d00b192f5f9ef759aefba9087be5fd60653e9c02479dfa41a"
FROZEN_PANEL_COVERAGE = (1_507, "2018-01-02", "2023-12-29")


@dataclass(frozen=True)
class BuildConfig:
    output_dir: str = "data/cboe_tail_risk_2018_2023"
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


def _positive(value: str, *, field: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"Cboe {field} must be numeric") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"Cboe {field} must be finite and positive")
    return result


def normalize_response(payload: bytes, *, symbol: str) -> bytes:
    if symbol not in SYMBOLS:
        raise ValueError(f"unsupported Cboe index: {symbol}")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Cboe response is not UTF-8 CSV") from exc
    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != SOURCE_COLUMNS[symbol]:
        raise ValueError(f"Cboe {symbol} source schema changed")
    rows: list[tuple[str, str]] = []
    seen: set[date] = set()
    for raw in reader:
        observation = _parse_date(raw["DATE"])
        if observation < START_DATE or observation >= END_DATE_EXCLUSIVE:
            continue
        if observation in seen:
            raise ValueError(f"duplicate Cboe date for {symbol}: {observation}")
        seen.add(observation)
        if symbol == "VIX":
            opening = _positive(raw["OPEN"], field="OPEN")
            high = _positive(raw["HIGH"], field="HIGH")
            low = _positive(raw["LOW"], field="LOW")
            close = _positive(raw["CLOSE"], field="CLOSE")
            if high < max(opening, close) or low > min(opening, close) or high < low:
                raise ValueError(f"Cboe OHLC invariant failed for VIX {observation}")
        else:
            close = _positive(raw[VALUE_COLUMN[symbol]], field=VALUE_COLUMN[symbol])
        rows.append((observation.isoformat(), format(close, ".6f")))
    rows.sort()
    expected_count, expected_first, expected_last = FROZEN_SOURCE_COVERAGE[symbol]
    if len(rows) != expected_count:
        raise ValueError(
            f"unexpected {symbol} source count: {len(rows)} != {expected_count}"
        )
    if rows[0][0] != expected_first or rows[-1][0] != expected_last:
        raise ValueError(f"unexpected {symbol} source coverage")
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(SNAPSHOT_COLUMNS)
    writer.writerows(rows)
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
        close = _positive(raw["close"], field="close")
        result[raw["date"]] = format(close, ".6f")
    expected_count, expected_first, expected_last = FROZEN_SOURCE_COVERAGE[symbol]
    ordered = sorted(result)
    if (
        len(ordered) != expected_count
        or ordered[0] != expected_first
        or ordered[-1] != expected_last
    ):
        raise ValueError(f"normalized {symbol} snapshot coverage changed")
    return result


def build_panel(snapshots: dict[str, bytes]) -> bytes:
    parsed = {
        symbol: parse_snapshot(payload, symbol=symbol)
        for symbol, payload in snapshots.items()
    }
    dates = sorted(set.intersection(*(set(values) for values in parsed.values())))
    expected_count, expected_first, expected_last = FROZEN_PANEL_COVERAGE
    if (
        len(dates) != expected_count
        or dates[0] != expected_first
        or dates[-1] != expected_last
    ):
        raise ValueError("Cboe tail-risk panel intersection coverage changed")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=PANEL_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for observation in dates:
        writer.writerow(
            {
                "observation_date": observation,
                "SKEW_close": parsed["SKEW"][observation],
                "VVIX_close": parsed["VVIX"][observation],
                "VIX_close": parsed["VIX"][observation],
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
    paths["panel"] = root / "cboe_tail_risk_2018-01-01_2023-12-31.csv.gz"
    paths["manifest"] = root / "build_manifest.json"
    return paths


def validate_frozen_artifacts(output_dir: str | Path) -> dict[str, Any]:
    if not all(FROZEN_SNAPSHOT_SHA256.values()) or not FROZEN_PANEL_SHA256:
        raise RuntimeError("Cboe tail-risk frozen artifact hashes are not initialized")
    paths = artifact_paths(output_dir)
    snapshots: dict[str, bytes] = {}
    for symbol in SYMBOLS:
        if sha256_file(paths[symbol]) != FROZEN_SNAPSHOT_SHA256[symbol]:
            raise RuntimeError(f"frozen Cboe {symbol} snapshot hash changed")
        snapshots[symbol] = read_gzip(paths[symbol])
    panel = build_panel(snapshots)
    if sha256_file(paths["panel"]) != FROZEN_PANEL_SHA256:
        raise RuntimeError("frozen Cboe tail-risk panel hash changed")
    if read_gzip(paths["panel"]) != panel:
        raise RuntimeError("frozen Cboe tail-risk panel no longer reproduces")
    manifest = json.loads(paths["manifest"].read_text())
    core = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if manifest.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("Cboe tail-risk source manifest hash changed")
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
        "builder": "training/build_cboe_tail_risk_panel.py",
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
        "created_at": "2026-07-17T15:30:00+00:00",
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
