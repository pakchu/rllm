"""Freeze Cleveland Fed pre-release CPI nowcasts and first-release surprises.

The official Cleveland Fed chart file contains the daily historical nowcast
path and the first available CPI release for each reference month.  This module
extracts only the last CPI and core-CPI nowcast strictly before the official
BLS release date, then binds the release clock to the separately frozen BLS
archive panel.  No BTC, funding, return, portfolio, or label data are read.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence


SOURCE_URL = (
    "https://www.clevelandfed.org/-/media/files/webcharts/"
    "inflationnowcasting/nowcast_month.json?sc_lang=en"
)
SOURCE_PAGE = (
    "https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting"
)
BLS_PANEL = (
    "data/bls_cpi_release_breadth_2019_2023/"
    "bls_cpi_release_breadth_2019_2023.csv.gz"
)
BLS_MANIFEST = "data/bls_cpi_release_breadth_2019_2023/build_manifest.json"
BLS_PANEL_SHA256 = "d199f409952d8cb83218864d0a96573bed82b59e649067b22fc97580a06d1059"
BLS_MANIFEST_SHA256 = "fb546580e64a01a4247318c8d4dad87028686d190f51559a9162d3efa3235171"
SOURCE_SNAPSHOT_DATE = "2026-07-18"
FROZEN_RESPONSE_SHA256 = "b2e1f0fb174be417eb417488c93bf9dbcb619c4ebcaef06ed35b18b704968cd9"
FROZEN_COVERAGE = (60, "2019-01-11", "2023-12-12")
FROZEN_RAW_SHA256 = "c53ccc1a64aca61e3bcfe309d91a564f4c257f2e81a91140d64bba9dc3247709"
FROZEN_PANEL_SHA256 = "e8755bfd15ec135b2a85cedada8880bf5d4518ed07f4eef43b4b3820211d508e"
USER_AGENT = "rllm-cleveland-cpi-surprise-freeze/1.0"
SERIES = (
    "CPI Inflation",
    "Core CPI Inflation",
    "Actual CPI Inflation",
    "Actual Core CPI Inflation",
)
PANEL_COLUMNS = (
    "reference_month",
    "release_time_utc",
    "latest_nowcast_date",
    "headline_nowcast_mom_pct",
    "core_nowcast_mom_pct",
    "headline_actual_mom_pct",
    "core_actual_mom_pct",
    "headline_surprise_pct",
    "core_surprise_pct",
    "composite_surprise_pct",
    "surprise_sign_concordant",
)


@dataclass(frozen=True)
class BuildConfig:
    output_dir: str = "data/cleveland_fed_cpi_surprise_2019_2023"
    retries: int = 5
    timeout_seconds: int = 60
    from_snapshot: bool = False
    import_json: str | None = None


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


def _fetch_bytes(*, retries: int, timeout: int) -> bytes:
    error: BaseException | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                SOURCE_URL,
                headers={"Accept": "application/json", "User-Agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            error = exc
        if attempt + 1 < retries:
            time.sleep(min(8.0, 0.5 * (2**attempt)))
    raise RuntimeError("failed to fetch Cleveland Fed nowcast history") from error


def _finite(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cleveland Fed {field} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Cleveland Fed {field} must be finite")
    return parsed


def _reference_month(value: Any) -> date:
    match = re.fullmatch(r"(\d{4})-(\d{1,2})", str(value))
    if not match:
        raise ValueError(f"invalid Cleveland Fed reference month: {value!r}")
    year, month = (int(part) for part in match.groups())
    return date(year, month, 1)


def _calendar_date(label: str, reference: date) -> date:
    match = re.fullmatch(r"(\d{2})/(\d{2})", label)
    if not match:
        raise ValueError(f"invalid Cleveland Fed chart date: {label!r}")
    month, day = (int(part) for part in match.groups())
    year = reference.year + int(month < reference.month)
    return date(year, month, day)


def _series_values(dataset: Sequence[Any], name: str) -> list[float | None]:
    matches = [item for item in dataset if isinstance(item, dict) and item.get("seriesname") == name]
    if len(matches) != 1 or not isinstance(matches[0].get("data"), list):
        raise ValueError(f"Cleveland Fed series changed: {name}")
    result: list[float | None] = []
    for item in matches[0]["data"]:
        if not isinstance(item, dict):
            raise ValueError(f"Cleveland Fed {name} data item changed")
        value = item.get("value")
        result.append(None if value in (None, "") else _finite(value, field=name))
    return result


def _read_bls_panel(path: str | Path = BLS_PANEL) -> dict[date, dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    required = {"reference_month", "release_time_utc", "source_complete"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("frozen BLS CPI panel schema changed")
    result: dict[date, dict[str, str]] = {}
    for row in rows:
        reference = date.fromisoformat(row["reference_month"])
        if reference in result or row["source_complete"] != "True":
            raise ValueError("frozen BLS CPI panel is incomplete or duplicated")
        result[reference] = row
    if len(result) != FROZEN_COVERAGE[0]:
        raise ValueError("frozen BLS CPI release coverage changed")
    return result


def parse_response(
    payload: bytes,
    *,
    bls_rows: Mapping[date, Mapping[str, str]],
) -> bytes:
    try:
        charts = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Cleveland Fed response is not valid JSON") from exc
    if not isinstance(charts, list):
        raise ValueError("Cleveland Fed monthly chart root changed")
    by_reference: dict[date, Mapping[str, Any]] = {}
    for raw_chart in charts:
        if not isinstance(raw_chart, dict) or not isinstance(raw_chart.get("chart"), dict):
            raise ValueError("Cleveland Fed chart schema changed")
        reference = _reference_month(raw_chart["chart"].get("subcaption"))
        if reference in by_reference:
            raise ValueError(f"duplicate Cleveland Fed reference month: {reference}")
        by_reference[reference] = raw_chart

    output_rows: list[dict[str, str]] = []
    for reference, bls in sorted(bls_rows.items()):
        if reference not in by_reference:
            raise ValueError(f"missing Cleveland Fed reference month: {reference}")
        chart = by_reference[reference]
        categories = chart.get("categories")
        if not isinstance(categories, list) or len(categories) != 1:
            raise ValueError("Cleveland Fed category schema changed")
        category_items = categories[0].get("category")
        if not isinstance(category_items, list):
            raise ValueError("Cleveland Fed category items changed")
        labels = [
            str(item["label"])
            for item in category_items
            if isinstance(item, dict)
            and re.fullmatch(r"\d{2}/\d{2}", str(item.get("label", "")))
        ]
        dataset = chart.get("dataset")
        if not isinstance(dataset, list):
            raise ValueError("Cleveland Fed dataset schema changed")
        values = {name: _series_values(dataset, name) for name in SERIES}
        if any(len(series) != len(labels) for series in values.values()):
            raise ValueError("Cleveland Fed chart date/value alignment changed")

        actual_indices = [
            index
            for index, value in enumerate(values["Actual CPI Inflation"])
            if value is not None
        ]
        core_actual_indices = [
            index
            for index, value in enumerate(values["Actual Core CPI Inflation"])
            if value is not None
        ]
        if len(actual_indices) != 1 or actual_indices != core_actual_indices:
            raise ValueError("Cleveland Fed first-release index changed")
        actual_index = actual_indices[0]
        prior_cpi = [
            index
            for index, value in enumerate(values["CPI Inflation"][:actual_index])
            if value is not None
        ]
        prior_core = [
            index
            for index, value in enumerate(values["Core CPI Inflation"][:actual_index])
            if value is not None
        ]
        if not prior_cpi or prior_cpi[-1] != prior_core[-1]:
            raise ValueError("Cleveland Fed latest headline/core nowcast date differs")
        nowcast_index = prior_cpi[-1]
        release_date = _calendar_date(labels[actual_index], reference)
        nowcast_date = _calendar_date(labels[nowcast_index], reference)
        if bls["release_time_utc"][:10] != release_date.isoformat():
            raise ValueError("Cleveland Fed actual date disagrees with frozen BLS release")
        if nowcast_date >= release_date:
            raise ValueError("Cleveland Fed nowcast is not strictly pre-release")

        headline_nowcast = values["CPI Inflation"][nowcast_index]
        core_nowcast = values["Core CPI Inflation"][nowcast_index]
        headline_actual = values["Actual CPI Inflation"][actual_index]
        core_actual = values["Actual Core CPI Inflation"][actual_index]
        if (
            headline_nowcast is None
            or core_nowcast is None
            or headline_actual is None
            or core_actual is None
        ):
            raise ValueError("Cleveland Fed retained value is missing")
        headline_surprise = headline_actual - headline_nowcast
        core_surprise = core_actual - core_nowcast
        composite = 0.5 * (headline_surprise + core_surprise)
        concordant = headline_surprise * core_surprise > 0.0
        output_rows.append(
            {
                "reference_month": reference.isoformat(),
                "release_time_utc": bls["release_time_utc"],
                "latest_nowcast_date": nowcast_date.isoformat(),
                "headline_nowcast_mom_pct": format(headline_nowcast, ".15f"),
                "core_nowcast_mom_pct": format(core_nowcast, ".15f"),
                "headline_actual_mom_pct": format(headline_actual, ".15f"),
                "core_actual_mom_pct": format(core_actual, ".15f"),
                "headline_surprise_pct": format(headline_surprise, ".15f"),
                "core_surprise_pct": format(core_surprise, ".15f"),
                "composite_surprise_pct": format(composite, ".15f"),
                "surprise_sign_concordant": "1" if concordant else "0",
            }
        )

    expected_rows, expected_first, expected_last = FROZEN_COVERAGE
    release_dates = [row["release_time_utc"][:10] for row in output_rows]
    if (
        len(output_rows) != expected_rows
        or release_dates[0] != expected_first
        or release_dates[-1] != expected_last
    ):
        raise ValueError("Cleveland Fed CPI surprise coverage changed")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=PANEL_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in output_rows:
        writer.writerow({column: row[column] for column in PANEL_COLUMNS})
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
    return {
        "raw": root / "source" / "nowcast_month_2026-07-18.json.gz",
        "panel": root / "cleveland_fed_cpi_surprise_2019_2023.csv.gz",
        "manifest": root / "build_manifest.json",
    }


def _validate_frozen_hash(path: Path, expected: str, *, label: str) -> None:
    if expected and sha256_file(path) != expected:
        raise RuntimeError(f"frozen Cleveland Fed {label} hash changed")


def build(config: BuildConfig = BuildConfig()) -> dict[str, Any]:
    if config.from_snapshot and config.import_json is not None:
        raise ValueError("choose either from_snapshot or import_json")
    paths = artifact_paths(config.output_dir)
    if config.from_snapshot:
        _validate_frozen_hash(paths["raw"], FROZEN_RAW_SHA256, label="raw response")
        raw = read_gzip(paths["raw"])
        network_read = False
    elif config.import_json is not None:
        raw = Path(config.import_json).read_bytes()
        if sha256_bytes(raw) != FROZEN_RESPONSE_SHA256:
            raise RuntimeError("imported Cleveland Fed response differs from research vintage")
        write_gzip(paths["raw"], raw)
        network_read = False
    else:
        raw = _fetch_bytes(retries=config.retries, timeout=config.timeout_seconds)
        if sha256_bytes(raw) != FROZEN_RESPONSE_SHA256:
            raise RuntimeError("Cleveland Fed response changed; audit a new vintage")
        write_gzip(paths["raw"], raw)
        network_read = True

    if sha256_file(BLS_PANEL) != BLS_PANEL_SHA256:
        raise RuntimeError("frozen BLS CPI panel changed")
    if sha256_file(BLS_MANIFEST) != BLS_MANIFEST_SHA256:
        raise RuntimeError("frozen BLS CPI manifest changed")
    panel = parse_response(raw, bls_rows=_read_bls_panel())
    write_gzip(paths["panel"], panel)
    _validate_frozen_hash(paths["panel"], FROZEN_PANEL_SHA256, label="panel")
    with gzip.open(paths["panel"], "rt") as handle:
        rows = list(csv.DictReader(handle))
    core: dict[str, Any] = {
        "schema_version": 1,
        "source_snapshot_date": SOURCE_SNAPSHOT_DATE,
        "builder": "training/build_cleveland_fed_cpi_surprise_panel.py",
        "config": asdict(config),
        "official_sources": {
            "indicator_page": SOURCE_PAGE,
            "monthly_chart_json": SOURCE_URL,
            "bls_release_panel": BLS_PANEL,
        },
        "source_contract": {
            "provider": "Federal Reserve Bank of Cleveland",
            "chart": "Inflation Nowcasting monthly month-over-month",
            "first_available_actual_release": True,
            "latest_nowcast_strictly_precedes_release": True,
            "network_read": network_read,
            "response_sha256": sha256_bytes(raw),
            "raw_snapshot": str(paths["raw"]),
            "raw_snapshot_sha256": sha256_file(paths["raw"]),
            "bls_panel_sha256": BLS_PANEL_SHA256,
            "bls_manifest_sha256": BLS_MANIFEST_SHA256,
            "market_or_funding_rows_read": 0,
        },
        "panel": {
            "path": str(paths["panel"]),
            "sha256": sha256_file(paths["panel"]),
            "rows": len(rows),
            "first_release": rows[0]["release_time_utc"],
            "last_release": rows[-1]["release_time_utc"],
            "columns": list(PANEL_COLUMNS),
        },
    }
    manifest = {**core, "manifest_hash": canonical_hash(core)}
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=BuildConfig.output_dir)
    parser.add_argument("--retries", type=int, default=BuildConfig.retries)
    parser.add_argument("--timeout-seconds", type=int, default=BuildConfig.timeout_seconds)
    parser.add_argument("--from-snapshot", action="store_true")
    parser.add_argument("--import-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build(
        BuildConfig(
            output_dir=args.output_dir,
            retries=args.retries,
            timeout_seconds=args.timeout_seconds,
            from_snapshot=args.from_snapshot,
            import_json=args.import_json,
        )
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
