"""Export outcome-free MFIC clocks for the LVRT comparator cohort."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from training.preregister_metaorder_fragmentation_impact_curvature import (
    CANDIDATES,
    Config,
    compute_mfic,
    load_causal_frame,
    nonoverlapping_schedule,
)


POLICY_ID = "LVRT-72"
PROTOCOL_VERSION = "lvrt_mfic_pure_clock_export_v1"
IMPLEMENTATION = Path("training/export_lvrt_mfic_comparator_clocks.py")
MECHANISM_DOCUMENT = Path(
    "docs/liquidity-vacuum-replenishment-transition-mechanism-decision-2026-07-21.md"
)
MECHANISM_DOCUMENT_SHA256 = (
    "9c2400a49b77a6e93594c65ae5bc8b17f6c676743a1fdfadf367979887dd77b9"
)
MFIC_IMPLEMENTATION = Path(
    "training/preregister_metaorder_fragmentation_impact_curvature.py"
)
MFIC_IMPLEMENTATION_SHA256 = (
    "51e99dbdc5ba13e6b4ac15e3915ec5b30e36dff89c1e5b31a5f3f7f272f01a59"
)
MFIC_SUPPORT = Path(
    "results/metaorder_fragmentation_impact_curvature_support_2026-07-14.json"
)
MFIC_SUPPORT_SHA256 = (
    "03bc5b2f67f974efa04715511920701c0db875b8bb4251f2e4c734a591aa80c8"
)
FEATURE_SOURCE = Path(
    "data/binance_um_aggtrade_microstructure_btc_2020_2023/"
    "BTCUSDT_aggtrade_5m_2020-01-01_2023-12-31.csv.gz"
)
FEATURE_SOURCE_SHA256 = (
    "c2bb0e6742f8cdc4e13315e7f0a13d6ab9cd536fb40d9cb4484b7a6ba30131cf"
)
FEATURE_MANIFEST = Path(
    "data/binance_um_aggtrade_microstructure_btc_2020_2023/build_manifest.json"
)
FEATURE_MANIFEST_SHA256 = (
    "6eec40460a6146c58994e52f1af9ace4eecc0c085887d97af5ef17c30b9f7e73"
)
MARKET_SOURCE = Path(
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)
MARKET_SOURCE_SHA256 = (
    "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
)
MARKET_MANIFEST = Path(
    "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
)
MARKET_MANIFEST_SHA256 = (
    "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
)
DEFAULT_CLOCK_OUTPUT = Path("results/lvrt_mfic_pure_clocks_2026-07-21.csv.gz")
DEFAULT_MANIFEST_OUTPUT = Path(
    "results/lvrt_mfic_pure_clock_manifest_2026-07-21.json"
)
FIELDS = (
    "candidate_id",
    "split",
    "causal_origin",
    "decision_time",
    "entry_time",
    "exit_time",
    "side",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle, object_pairs_hook=reject_duplicates)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _timestamp(value: Any, *, field: str) -> datetime:
    if isinstance(value, pd.Timestamp):
        parsed = value.to_pydatetime()
    elif isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"invalid {field}: {value!r}") from exc
    else:
        raise ValueError(f"{field} must be a timestamp")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    if parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{field} must be UTC")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _verify_bindings() -> dict[str, Any]:
    expected = {
        MECHANISM_DOCUMENT: MECHANISM_DOCUMENT_SHA256,
        MFIC_IMPLEMENTATION: MFIC_IMPLEMENTATION_SHA256,
        MFIC_SUPPORT: MFIC_SUPPORT_SHA256,
        FEATURE_SOURCE: FEATURE_SOURCE_SHA256,
        FEATURE_MANIFEST: FEATURE_MANIFEST_SHA256,
        MARKET_SOURCE: MARKET_SOURCE_SHA256,
        MARKET_MANIFEST: MARKET_MANIFEST_SHA256,
    }
    for path, expected_sha in expected.items():
        if sha256_file(path) != expected_sha:
            raise ValueError(f"LVRT MFIC comparator binding changed: {path}")
    support = _load_json(MFIC_SUPPORT)
    protocol = support.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("outcomes_opened") is not False:
        raise ValueError("MFIC support opened outcomes")
    if support.get("all_candidates_pass_support") is not True:
        raise ValueError("MFIC support did not pass")
    candidates = support.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ValueError("MFIC support candidate set changed")
    return support


def _schedule_rows(
    frame: pd.DataFrame,
    support: Mapping[str, Any],
) -> list[dict[str, Any]]:
    support_by_name = {
        item["candidate"]["name"]: item
        for item in support["candidates"]
        if isinstance(item, dict) and isinstance(item.get("candidate"), dict)
    }
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        support_item = support_by_name.get(candidate.name)
        if support_item is None:
            raise ValueError(f"missing frozen MFIC candidate: {candidate.name}")
        signal = compute_mfic(frame, candidate, Config())
        schedule = nonoverlapping_schedule(signal, frame)
        expected_total = int(support_item["support"]["nonoverlap_total"])
        if len(schedule) != expected_total:
            raise ValueError(f"MFIC {candidate.name} incidence changed")
        expected_years = {
            str(year): int(count)
            for year, count in support_item["support"]["by_year"].items()
        }
        actual_years = Counter(
            str(_timestamp(value, field="entry_date").year)
            for value in schedule["entry_date"]
        )
        if dict(sorted(actual_years.items())) != dict(sorted(expected_years.items())):
            raise ValueError(f"MFIC {candidate.name} year incidence changed")
        prior_exit: datetime | None = None
        for raw in schedule.to_dict(orient="records"):
            origin = _timestamp(raw["signal_date"], field="signal_date")
            entry = _timestamp(raw["entry_date"], field="entry_date")
            exit_time = _timestamp(raw["exit_date"], field="exit_date")
            side = int(raw["side"])
            if side not in {-1, 1}:
                raise ValueError("MFIC comparator side changed")
            if entry - origin != timedelta(minutes=5):
                raise ValueError("MFIC comparator entry latency changed")
            if exit_time - entry != int(raw["hold_bars"]) * timedelta(minutes=5):
                raise ValueError("MFIC comparator hold changed")
            if prior_exit is not None and entry < prior_exit:
                raise ValueError("MFIC comparator schedule overlaps")
            prior_exit = exit_time
            rows.append(
                {
                    "candidate_id": f"mfic:{candidate.name}",
                    "split": "train" if entry.year <= 2022 else "selection",
                    "causal_origin": _iso(origin),
                    "decision_time": _iso(entry),
                    "entry_time": _iso(entry),
                    "exit_time": _iso(exit_time),
                    "side": side,
                }
            )
    return sorted(rows, key=lambda row: (row["candidate_id"], row["entry_time"]))


def _clock_bytes(rows: list[dict[str, Any]]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(FIELDS)
    for row in rows:
        writer.writerow([row[field] for field in FIELDS])
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as handle:
        handle.write(text.getvalue().encode("utf-8"))
    return output.getvalue()


def build_outputs() -> tuple[dict[str, Any], bytes]:
    support = _verify_bindings()
    frame, metadata = load_causal_frame(Config())
    rows = _schedule_rows(frame, support)
    clock_bytes = _clock_bytes(rows)
    rows_by_candidate = dict(sorted(Counter(row["candidate_id"] for row in rows).items()))
    rows_by_split = dict(sorted(Counter(row["split"] for row in rows).items()))
    core: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "as_of_date": "2026-07-21",
        "implementation_binding": {
            "path": str(IMPLEMENTATION),
            "sha256": sha256_file(IMPLEMENTATION),
        },
        "decision_binding": {
            "path": str(MECHANISM_DOCUMENT),
            "sha256": MECHANISM_DOCUMENT_SHA256,
        },
        "mfic_bindings": {
            "implementation": {
                "path": str(MFIC_IMPLEMENTATION),
                "sha256": MFIC_IMPLEMENTATION_SHA256,
            },
            "support": {
                "path": str(MFIC_SUPPORT),
                "sha256": MFIC_SUPPORT_SHA256,
            },
            "feature_source_sha256": FEATURE_SOURCE_SHA256,
            "feature_manifest_sha256": FEATURE_MANIFEST_SHA256,
            "market_source_sha256": MARKET_SOURCE_SHA256,
            "market_manifest_sha256": MARKET_MANIFEST_SHA256,
        },
        "clock": {
            "path": str(DEFAULT_CLOCK_OUTPUT),
            "sha256": hashlib.sha256(clock_bytes).hexdigest(),
            "gzip_mtime": 0,
            "schema": list(FIELDS),
            "rows": len(rows),
            "rows_by_candidate": rows_by_candidate,
            "rows_by_split": rows_by_split,
        },
        "source_audit": metadata,
        "outcome_boundary": {
            "causal_market_rows_loaded": len(frame),
            "causal_aggtrade_feature_rows_loaded": int(frame["source_available"].sum()),
            "performance_artifacts_parsed": 0,
            "return_or_pnl_fields_read": 0,
            "strict_simulation_calls": 0,
            "funding_rows_loaded": 0,
            "post_2023_rows_loaded": 0,
            "network_calls": 0,
            "economic_outcomes_computed": False,
        },
        "parameter_search_performed": False,
        "post_failure_repair_performed": False,
    }
    core["manifest_hash"] = canonical_hash(core)
    return {**core, "created_at": datetime.now(timezone.utc).isoformat()}, clock_bytes


def publish(
    manifest_path: Path,
    clock_path: Path,
    manifest: Mapping[str, Any],
    clock_bytes: bytes,
) -> None:
    if str(manifest["clock"]["path"]) != str(clock_path):
        raise ValueError("LVRT MFIC clock output path changed")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    clock_path.parent.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        clock_fd = os.open(clock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        created.append(clock_path)
        with os.fdopen(clock_fd, "wb") as handle:
            handle.write(clock_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        manifest_fd = os.open(
            manifest_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        created.append(manifest_path)
        payload = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with os.fdopen(manifest_fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    manifest, clock_bytes = build_outputs()
    publish(DEFAULT_MANIFEST_OUTPUT, DEFAULT_CLOCK_OUTPUT, manifest, clock_bytes)
    print(
        json.dumps(
            {
                "clock": str(DEFAULT_CLOCK_OUTPUT),
                "manifest": str(DEFAULT_MANIFEST_OUTPUT),
                "manifest_hash": manifest["manifest_hash"],
                "rows": manifest["clock"]["rows"],
                "rows_by_candidate": manifest["clock"]["rows_by_candidate"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
