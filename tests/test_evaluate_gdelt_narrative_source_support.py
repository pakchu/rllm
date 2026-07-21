from __future__ import annotations

import builtins
import csv
import gzip
import io
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from training import evaluate_gdelt_narrative_source_support as evaluator
from training import preregister_gdelt_narrative_rotation_clearing as prereg


UTC = timezone.utc


def _constant_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    start = date(2020, 1, 1)
    for index in range(evaluator.EXPECTED_DAILY_ROWS):
        source_date = start + timedelta(days=index)
        rows.append(
            {
                "date": source_date.isoformat(),
                "available_at": (
                    datetime.combine(source_date, datetime.min.time(), tzinfo=UTC)
                    + timedelta(hours=48, minutes=15)
                ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "global_article_count": 100_000,
                "broad_article_count": 1_000,
                "failure_article_count": 20,
                "constraint_article_count": 25,
                "adoption_article_count": 30,
            }
        )
    return rows


def _write_daily_source(path: Path, rows: list[dict[str, object]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=evaluator.source.DAILY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _rows_with_frozen_outages() -> list[dict[str, object]]:
    rows = _constant_rows()
    outage_dates = set(evaluator.source.KNOWN_GLOBAL_OUTAGE_DATES)
    for row in rows:
        if row["date"] in outage_dates:
            for column in evaluator.source.DAILY_COLUMNS[2:]:
                row[column] = 0
    return rows


def _prepare_full_source_fixture(tmp_path: Path, real_root: Path) -> dict[str, object]:
    for path in (evaluator.PREREGISTRATION, evaluator.TRANSPORT_AMENDMENT):
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((real_root / path).read_bytes())
    for path in (evaluator.EVALUATOR_SOURCE, evaluator.PROTOCOL_DOCUMENT):
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((real_root / path).read_bytes())
    daily = tmp_path / evaluator.DAILY_SOURCE
    daily.parent.mkdir(parents=True, exist_ok=True)
    _write_daily_source(daily, _rows_with_frozen_outages())
    raw = tmp_path / evaluator.RAW_SOURCE
    raw.write_bytes(b"frozen raw bundle fixture\n")
    manifest: dict[str, object] = {
        "protocol_version": evaluator.source.PROTOCOL_VERSION,
        "contract": evaluator.source.source_contract(evaluator.source.Config()),
        "contract_hash": evaluator.source.canonical_hash(
            evaluator.source.source_contract(evaluator.source.Config())
        ),
        "builder": {
            "path": str(evaluator.source.BUILDER),
            "sha256": evaluator.V2_SOURCE_SHA256,
            "v1_dependency_path": str(evaluator.source.V1_DEPENDENCY),
            "v1_dependency_sha256": evaluator.V1_SOURCE_SHA256,
        },
        "requests": {
            "count": 4,
            "response_hashes": [
                {
                    "query_id": query_id,
                    "start": evaluator.source.FROZEN_START_DATE,
                    "end_exclusive": evaluator.source.FROZEN_END_DATE_EXCLUSIVE,
                    "response_sha256": f"{index + 1:x}" * 64,
                }
                for index, (query_id, _) in enumerate(evaluator.source.QUERIES)
            ],
        },
        "source_audit": {
            "daily_rows": 1461,
            "first_date": "2020-01-01",
            "last_date": "2023-12-31",
            "first_available_at": "2020-01-03T00:15:00Z",
            "last_available_at": "2024-01-02T00:15:00Z",
            "date_resolution": "day",
            "global_norm_consistent_across_available_queries": True,
            "missing_bins_by_query": {
                "broad": 2,
                "failure": 2,
                "constraint": 2,
                "adoption": 2,
            },
            "global_outage_dates": list(evaluator.source.KNOWN_GLOBAL_OUTAGE_DATES),
            "global_outage_days": 2,
            "known_global_outage_dates_match": True,
        },
        "outputs": {
            "daily_path": str(evaluator.DAILY_SOURCE),
            "daily_sha256": evaluator.sha256_file(daily),
            "daily_columns": list(evaluator.source.DAILY_COLUMNS),
            "raw_bundle_path": str(evaluator.RAW_SOURCE),
            "raw_bundle_sha256": evaluator.sha256_file(raw),
        },
        "outcome_boundary": {
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "future_return_rows_read": 0,
            "return_or_pnl_fields_read": 0,
            "post_2023_news_rows_requested": 0,
            "economic_metrics_computed": False,
        },
    }
    manifest["manifest_hash"] = evaluator.source.canonical_hash(manifest)
    manifest_path = tmp_path / evaluator.SOURCE_MANIFEST
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    seal: dict[str, object] = {
        "protocol_version": "gdelt_gnrc_source_access_seal_v1",
        "preregistration_path": str(evaluator.PREREGISTRATION),
        "preregistration_sha256": evaluator.PREREGISTRATION_SHA256,
        "transport_amendment_path": str(evaluator.TRANSPORT_AMENDMENT),
        "transport_amendment_sha256": evaluator.TRANSPORT_AMENDMENT_SHA256,
        "source_manifest_path": str(evaluator.SOURCE_MANIFEST),
        "source_manifest_sha256": evaluator.sha256_file(manifest_path),
        "daily_source_path": str(evaluator.DAILY_SOURCE),
        "daily_source_sha256": evaluator.sha256_file(daily),
        "raw_source_path": str(evaluator.RAW_SOURCE),
        "raw_source_sha256": evaluator.sha256_file(raw),
        "evaluator_source_path": str(evaluator.EVALUATOR_SOURCE),
        "evaluator_source_sha256": evaluator.sha256_file(
            tmp_path / evaluator.EVALUATOR_SOURCE
        ),
        "protocol_document_path": str(evaluator.PROTOCOL_DOCUMENT),
        "protocol_document_sha256": evaluator.sha256_file(
            tmp_path / evaluator.PROTOCOL_DOCUMENT
        ),
        "feature_values_inspected_before_seal": False,
        "market_outcomes_opened_before_seal": False,
        "sealed_at": "2026-07-22T00:00:00Z",
    }
    seal_path = tmp_path / evaluator.SOURCE_ACCESS_SEAL
    seal_path.parent.mkdir(parents=True, exist_ok=True)
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    return seal


def test_constant_source_has_no_supported_variants_and_retires() -> None:
    result = evaluator.evaluate_rows(_constant_rows())
    assert result["decision"] == "retire_without_repair"
    assert result["family_support"]["passing_variant_count"] == 0
    assert len(result["variant_support"]) == 24
    assert tuple(result["variant_support"]) == prereg.FAMILY_VARIANT_IDS
    for variant in result["variant_support"].values():
        assert variant["train"]["admitted_events"] == 0
        assert variant["selection"]["admitted_events"] == 0
        assert tuple(variant["checks"]) == prereg.VARIANT_SUPPORT_CHECK_NAMES
        assert variant["passes"] is False


def test_source_support_rejects_truncated_or_duplicate_grid() -> None:
    rows = _constant_rows()
    with pytest.raises(ValueError, match="row count"):
        evaluator.evaluate_rows(rows[:-1])
    rows[-1] = dict(rows[-2])
    with pytest.raises(ValueError, match="duplicated"):
        evaluator.evaluate_rows(rows)


def test_evaluator_is_bound_to_frozen_preregistration() -> None:
    expected_hashes = {
        evaluator.PREREGISTRATION: evaluator.PREREGISTRATION_SHA256,
        evaluator.PREREGISTRATION_SOURCE: evaluator.PREREGISTRATION_SOURCE_SHA256,
        evaluator.PREREGISTRATION_DOCUMENT: (evaluator.PREREGISTRATION_DOCUMENT_SHA256),
        evaluator.TRANSPORT_AMENDMENT: evaluator.TRANSPORT_AMENDMENT_SHA256,
        evaluator.V1_SOURCE: evaluator.V1_SOURCE_SHA256,
        evaluator.V2_SOURCE: evaluator.V2_SOURCE_SHA256,
    }
    for path, expected_hash in expected_hashes.items():
        assert evaluator.sha256_file(path) == expected_hash
    payload = evaluator.validate_preregistration()
    assert tuple(row["variant_id"] for row in payload["variants"]) == (
        prereg.FAMILY_VARIANT_IDS
    )
    assert payload["outcome_boundary"]["outcomes_opened"] is False


@pytest.mark.parametrize("hold_days", prereg.HOLD_DAYS)
def test_split_source_grid_has_exact_entry_and_exit_boundaries(hold_days: int) -> None:
    for split_start, split_end in evaluator._split_bounds().values():
        dates = prereg.expected_split_source_dates(
            split_start=split_start,
            split_end_exclusive=split_end,
            hold_days=hold_days,
        )
        first_entry = (
            datetime.combine(dates[0], datetime.min.time(), tzinfo=UTC)
            + prereg.ENTRY_LAG
        )
        last_entry = (
            datetime.combine(dates[-1], datetime.min.time(), tzinfo=UTC)
            + prereg.ENTRY_LAG
        )
        assert first_entry >= split_start
        assert last_entry + timedelta(days=hold_days) < split_end
        assert first_entry - timedelta(days=1) < split_start
        assert last_entry + timedelta(days=hold_days + 1) >= split_end


def test_daily_loader_accepts_only_frozen_outages_and_clock(tmp_path: Path) -> None:
    rows = _rows_with_frozen_outages()
    daily = tmp_path / "daily.csv.gz"
    _write_daily_source(daily, rows)
    loaded = evaluator.load_daily_rows(daily)
    assert len(loaded) == evaluator.EXPECTED_DAILY_ROWS
    rows[100]["available_at"] = "2020-04-13T00:15:00Z"
    _write_daily_source(daily, rows)
    with pytest.raises(ValueError, match="availability clock"):
        evaluator.load_daily_rows(daily)


def test_source_access_seal_hashes_every_executable_input(
    tmp_path: Path, monkeypatch
) -> None:
    real_root = evaluator.REPOSITORY_ROOT
    monkeypatch.setattr(evaluator, "REPOSITORY_ROOT", tmp_path)
    paths = {
        "preregistration_path": evaluator.PREREGISTRATION,
        "transport_amendment_path": evaluator.TRANSPORT_AMENDMENT,
        "source_manifest_path": evaluator.SOURCE_MANIFEST,
        "daily_source_path": evaluator.DAILY_SOURCE,
        "raw_source_path": evaluator.RAW_SOURCE,
        "evaluator_source_path": evaluator.EVALUATOR_SOURCE,
        "protocol_document_path": evaluator.PROTOCOL_DOCUMENT,
    }
    for path in paths.values():
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path in {evaluator.PREREGISTRATION, evaluator.TRANSPORT_AMENDMENT}:
            destination.write_bytes((real_root / path).read_bytes())
        else:
            destination.write_text(f"fixture:{path}\n", encoding="utf-8")
    seal: dict[str, object] = {
        "protocol_version": "gdelt_gnrc_source_access_seal_v1",
        **{field: str(path) for field, path in paths.items()},
        "preregistration_sha256": evaluator.PREREGISTRATION_SHA256,
        "transport_amendment_sha256": evaluator.TRANSPORT_AMENDMENT_SHA256,
        "source_manifest_sha256": evaluator.sha256_file(evaluator.SOURCE_MANIFEST),
        "daily_source_sha256": evaluator.sha256_file(evaluator.DAILY_SOURCE),
        "raw_source_sha256": evaluator.sha256_file(evaluator.RAW_SOURCE),
        "evaluator_source_sha256": evaluator.sha256_file(evaluator.EVALUATOR_SOURCE),
        "protocol_document_sha256": evaluator.sha256_file(evaluator.PROTOCOL_DOCUMENT),
        "feature_values_inspected_before_seal": False,
        "market_outcomes_opened_before_seal": False,
        "sealed_at": "2026-07-22T00:00:00Z",
    }
    evaluator.validate_source_access_seal(seal)
    (tmp_path / evaluator.DAILY_SOURCE).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="daily_source_sha256"):
        evaluator.validate_source_access_seal(seal)


def test_build_report_reads_only_the_explicit_source_allowlist(
    tmp_path: Path, monkeypatch
) -> None:
    real_root = evaluator.REPOSITORY_ROOT
    monkeypatch.setattr(evaluator, "REPOSITORY_ROOT", tmp_path)
    _prepare_full_source_fixture(tmp_path, real_root)
    allowed = {
        (tmp_path / path).resolve()
        for path in (
            evaluator.PREREGISTRATION,
            evaluator.TRANSPORT_AMENDMENT,
            evaluator.SOURCE_ACCESS_SEAL,
            evaluator.SOURCE_MANIFEST,
            evaluator.DAILY_SOURCE,
            evaluator.RAW_SOURCE,
            evaluator.EVALUATOR_SOURCE,
            evaluator.PROTOCOL_DOCUMENT,
        )
    }
    opened: set[Path] = set()
    original_path_open = Path.open
    original_builtin_open = builtins.open
    original_io_open = io.open

    def guard(file) -> None:
        resolved = Path(file).resolve()
        if resolved not in allowed:
            raise AssertionError(f"unexpected source-support read: {resolved}")
        opened.add(resolved)

    def recording_path_open(path: Path, *args, **kwargs):
        guard(path)
        return original_path_open(path, *args, **kwargs)

    def recording_builtin_open(file, *args, **kwargs):
        guard(file)
        return original_builtin_open(file, *args, **kwargs)

    def recording_io_open(file, *args, **kwargs):
        guard(file)
        return original_io_open(file, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_path_open)
    monkeypatch.setattr(builtins, "open", recording_builtin_open)
    monkeypatch.setattr(io, "open", recording_io_open)
    report = evaluator.build_report()
    assert report["outcome_boundary"]["outcomes_opened"] is False
    assert opened <= allowed
