"""Audit a frozen Daily Treasury Statement source snapshot without outcomes.

This module opens only the source snapshot produced by
``build_daily_treasury_fiscal_flow_source``.  It verifies provenance,
literal-number round trips, accounting identities, schema transitions,
announcement coverage, and clean-rerun determinism.  It never imports or
opens market, return, label, position, PnL, CAGR, or drawdown data.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import unicodedata
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

try:
    from training import build_daily_treasury_fiscal_flow_source as source
except ModuleNotFoundError as exc:
    if __package__ or exc.name != "training":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from training import build_daily_treasury_fiscal_flow_source as source


AUDIT_SCHEMA_VERSION = 1
AUDIT_VERSION = 1
STABILITY_WINDOW_REPORTS = 5
AMOUNT_COLUMNS = (
    (
        "today_amount_usd_millions",
        "today_amount_literal",
        "today",
    ),
    (
        "month_to_date_amount_usd_millions",
        "month_to_date_amount_literal",
        "month_to_date",
    ),
    (
        "fiscal_year_to_date_amount_usd_millions",
        "fiscal_year_to_date_amount_literal",
        "fiscal_year_to_date",
    ),
)
DETERMINISTIC_SOURCE_FILES = (
    "daily_treasury_fiscal_flow_rows.csv.gz",
    "daily_treasury_operating_cash_rows.csv.gz",
    "precap_schema_announcements.csv.gz",
    "source_manifest.json",
    "source_build_report.json",
    "receipt_log.jsonl",
)
NEW_YORK = ZoneInfo("America/New_York")

ACCOUNTING_COLUMNS = (
    "record_date",
    "table_id",
    "side",
    "check_name",
    "amount_column",
    "observed_total",
    "computed_total",
    "component_count",
    "residual_usd_millions",
    "tolerance_usd_millions",
    "status",
    "notes",
)
NUMERIC_COLUMNS = (
    "record_date",
    "table_id",
    "side",
    "checked_cell_count",
    "missing_cell_count",
    "failed_cell_count",
    "status",
)
DUPLICATE_COLUMNS = (
    "record_date",
    "table_id",
    "side",
    "normalized_category_label",
    "occurrence_count",
    "raw_labels_json",
    "parent_sections_json",
    "status",
)
TRANSITION_COLUMNS = (
    "report_date",
    "source_available_date_new_york",
    "table_id",
    "side",
    "transition_type",
    "normalized_category_label",
    "before_label",
    "after_label",
    "before_parent_section",
    "after_parent_section",
    "support_before_reports",
    "support_after_reports",
    "announcement_match_count",
    "status",
)

# The 2020-04-01 PDF itself prints 75,371 for deposit MTD while printing
# 75,461 for today's deposits and -8,324 for both today's and MTD net change.
# The parser round-trips those literals exactly, so this is quarantined as an
# immutable published-source anomaly rather than silently correcting Treasury.
KNOWN_PUBLISHED_ACCOUNTING_ANOMALIES = {
    (
        "2020-04-01",
        "account_deposits_minus_withdrawals_equals_net_change",
        "month_to_date",
        -8324,
        -8415,
    )
}
ANNOUNCEMENT_COLUMNS = (
    "effective_date",
    "sheet_name",
    "source_row",
    "table_name",
    "table_section",
    "entity",
    "change",
    "scope_json",
    "matched_transition_count",
    "matched_label_count",
    "status",
)


@dataclass(frozen=True)
class AuditConfig:
    source_dir: str = "data/daily_treasury_fiscal_flow_2019_2023"
    output_dir: str = ""
    max_workers: int = 12
    verify_clean_rerun: bool = True


@dataclass(frozen=True)
class SourceSnapshot:
    source_dir: Path
    manifest: dict[str, Any]
    build_report: dict[str, Any]
    rows: tuple[dict[str, str], ...]
    table_i_rows: tuple[dict[str, str], ...]
    announcements: tuple[dict[str, str], ...]


def rounding_tolerance(component_count: int) -> float:
    if component_count < 0:
        raise ValueError("component_count must be non-negative")
    return 0.5 * (component_count + 1)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_gzip_csv(path: Path) -> tuple[dict[str, str], ...]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return tuple(dict(row) for row in csv.DictReader(handle))


def load_snapshot(source_dir: Path) -> SourceSnapshot:
    return SourceSnapshot(
        source_dir=source_dir,
        manifest=_read_json(source_dir / "source_manifest.json"),
        build_report=_read_json(source_dir / "source_build_report.json"),
        rows=_read_gzip_csv(source_dir / "daily_treasury_fiscal_flow_rows.csv.gz"),
        table_i_rows=_read_gzip_csv(
            source_dir / "daily_treasury_operating_cash_rows.csv.gz"
        ),
        announcements=_read_gzip_csv(source_dir / "precap_schema_announcements.csv.gz"),
    )


def _sha256_uncompressed_gzip(path: Path) -> str:
    with gzip.open(path, "rb") as handle:
        return source.sha256_bytes(handle.read())


def _report_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    reports = manifest.get("reports")
    if not isinstance(reports, list):
        raise ValueError("source manifest reports must be a list")
    output: dict[str, dict[str, Any]] = {}
    for raw in reports:
        if not isinstance(raw, dict) or not isinstance(raw.get("record_date"), str):
            raise ValueError("source manifest contains an invalid report entry")
        record_date = raw["record_date"]
        if record_date in output:
            raise ValueError(f"duplicate manifest report date: {record_date}")
        output[record_date] = raw
    return output


def _append_error(errors: list[str], message: str, *, limit: int = 200) -> None:
    if len(errors) < limit:
        errors.append(message)


def _active_toolchain() -> dict[str, Any]:
    parser_source = Path(source.__file__)
    return {
        "parser_source_sha256": source.sha256_file(parser_source),
        "python": platform.python_version(),
        "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
        "unicode_database": unicodedata.unidata_version,
        "america_new_york_tzfile_sha256": source._timezone_file_sha256(),
        "holiday_calendar_source_sha256": source.sha256_file(parser_source),
        "label_normalization_contract_sha256": source.sha256_bytes(
            b"NFC|dash-to-hyphen|collapse-whitespace|preserve-case-v1"
        ),
        "locale_contract": "LC_ALL=C",
    }


def _verify_toolchain_binding(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != source.SCHEMA_VERSION:
        errors.append("source schema version differs from the active parser")
    if manifest.get("parser_version") != source.PARSER_VERSION:
        errors.append("source parser version differs from the active parser")
    if manifest.get("toolchain") != _active_toolchain():
        errors.append("source toolchain binding differs from the active parser")
    return errors


def _verify_physical_inventory(
    source_dir: Path, reports: dict[str, dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    expected_top_level = {
        "raw",
        "daily_treasury_fiscal_flow_rows.csv.gz",
        "daily_treasury_operating_cash_rows.csv.gz",
        "precap_schema_announcements.csv.gz",
        "receipt_log.jsonl",
        "source_build_report.json",
        "source_manifest.json",
    }
    allowed_top_level = expected_top_level | {"audit"}
    try:
        actual_top_level = {path.name for path in source_dir.iterdir()}
    except OSError as exc:
        return [f"cannot enumerate frozen source snapshot: {exc}"]
    unexpected_top_level = sorted(actual_top_level - allowed_top_level)
    missing_top_level = sorted(expected_top_level - actual_top_level)
    if unexpected_top_level or missing_top_level:
        errors.append(
            "frozen source top-level inventory mismatch; "
            f"unexpected={unexpected_top_level}, missing={missing_top_level}"
        )
    symlinks = sorted(
        str(path.relative_to(source_dir))
        for path in source_dir.rglob("*")
        if path.is_symlink()
    )
    if symlinks:
        errors.append(f"symlinks are forbidden in the frozen source snapshot: {symlinks[:10]}")

    audit_dir = source_dir / "audit"
    allowed_audit_names = {
        "source_accounting_reconciliations.csv.gz",
        "source_announcement_reconciliation.csv.gz",
        "source_duplicate_label_audit.csv.gz",
        "source_numeric_cell_roundtrip.csv.gz",
        "source_quality_audit_manifest.json",
        "source_quality_audit_report.json",
        "source_schema_transitions.csv.gz",
    }
    if audit_dir.exists():
        if not audit_dir.is_dir():
            errors.append("audit output path is not a directory")
        else:
            unexpected_audit = sorted(
                path.name
                for path in audit_dir.iterdir()
                if path.name not in allowed_audit_names
            )
            if unexpected_audit:
                errors.append(
                    f"unexpected files in source-only audit directory: {unexpected_audit}"
                )

    raw_dir = source_dir / "raw"
    expected_raw_names = {
        "reports",
        "page-data.json",
        "page-data.json.receipt.json",
        "DailyTreasuryStatement_Announcements.xlsx",
        "DailyTreasuryStatement_Announcements.xlsx.receipt.json",
    }
    if not raw_dir.is_dir():
        return [*errors, "raw source directory is missing"]
    actual_raw_names = {path.name for path in raw_dir.iterdir()}
    if actual_raw_names != expected_raw_names:
        errors.append("raw source inventory differs from the physical-cap allowlist")

    reports_dir = raw_dir / "reports"
    expected_report_names = {
        name
        for record_date in reports
        for name in (
            f"{record_date.replace('-', '')}.pdf",
            f"{record_date.replace('-', '')}.pdf.receipt.json",
        )
    }
    if not reports_dir.is_dir():
        errors.append("raw report directory is missing")
        actual_report_names: set[str] = set()
    else:
        actual_report_names = {path.name for path in reports_dir.iterdir()}
    if actual_report_names != expected_report_names:
        unexpected = sorted(actual_report_names - expected_report_names)[:10]
        missing = sorted(expected_report_names - actual_report_names)[:10]
        errors.append(
            f"raw report inventory mismatch; unexpected={unexpected}, missing={missing}"
        )

    expected_receipt_paths = {
        source.DATASET_PAGE_DATA_URL: raw_dir / "page-data.json.receipt.json",
        source.ANNOUNCEMENTS_URL: (
            raw_dir / "DailyTreasuryStatement_Announcements.xlsx.receipt.json"
        ),
    }
    for record_date in reports:
        try:
            parsed_date = date.fromisoformat(record_date)
        except ValueError:
            continue
        expected_receipt_paths[source.report_url(parsed_date)] = (
            reports_dir / f"{parsed_date:%Y%m%d}.pdf.receipt.json"
        )

    receipt_log: dict[str, dict[str, Any]] = {}
    try:
        lines = (source_dir / "receipt_log.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        for line_number, line in enumerate(lines, start=1):
            raw_receipt = json.loads(line)
            if not isinstance(raw_receipt, dict) or not isinstance(
                raw_receipt.get("url"), str
            ):
                raise ValueError(f"invalid receipt row {line_number}")
            url = raw_receipt["url"]
            if url in receipt_log:
                raise ValueError(f"duplicate receipt URL {url}")
            receipt_log[url] = raw_receipt
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"invalid receipt log: {exc}")
    if set(receipt_log) != set(expected_receipt_paths):
        errors.append("receipt log URL inventory differs from the source allowlist")
    for url, receipt_path in expected_receipt_paths.items():
        logged = receipt_log.get(url)
        try:
            bound = _read_json(receipt_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid bound receipt {receipt_path.name}: {exc}")
            continue
        if logged != bound:
            errors.append(f"receipt log row differs from bound receipt: {url}")
        if bound.get("url") != url or bound.get("final_url") != url:
            errors.append(f"receipt URL escaped source allowlist: {url}")
    return errors


def verify_snapshot_binding(snapshot: SourceSnapshot) -> tuple[bool, list[str]]:
    errors: list[str] = []
    manifest = snapshot.manifest
    build_report = snapshot.build_report
    source_dir = snapshot.source_dir
    try:
        reports = _report_map(manifest)
    except ValueError as exc:
        return False, [str(exc)]

    errors.extend(_verify_toolchain_binding(manifest))

    horizon = manifest.get("source_horizon")
    expected_horizon = {
        "start": source.MIN_REPORT_DATE.isoformat(),
        "end": source.MAX_REPORT_DATE.isoformat(),
    }
    if horizon != expected_horizon:
        _append_error(errors, f"unexpected source horizon: {horizon!r}")

    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        _append_error(errors, "manifest metadata is missing")
        metadata = {}
    metadata_paths = (
        (
            "page_data_sha256",
            source.DATASET_PAGE_DATA_URL,
            source_dir / "raw" / "page-data.json",
        ),
        (
            "announcements_sha256",
            source.ANNOUNCEMENTS_URL,
            source_dir / "raw" / "DailyTreasuryStatement_Announcements.xlsx",
        ),
    )
    for hash_key, expected_url, path in metadata_paths:
        if not path.is_file():
            _append_error(errors, f"missing metadata bytes: {path.name}")
            continue
        if metadata.get(hash_key) != source.sha256_file(path):
            _append_error(errors, f"metadata hash mismatch: {path.name}")
        receipt_path = source._receipt_path(path)
        try:
            receipt = _read_json(receipt_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            _append_error(errors, f"invalid metadata receipt {receipt_path.name}: {exc}")
            continue
        if receipt.get("url") != expected_url or receipt.get("final_url") != expected_url:
            _append_error(errors, f"metadata receipt URL mismatch: {path.name}")
        if receipt.get("status") != 200:
            _append_error(errors, f"metadata receipt status is not 200: {path.name}")
        if receipt.get("sha256") != source.sha256_file(path):
            _append_error(errors, f"metadata receipt hash mismatch: {path.name}")
        if receipt.get("byte_length") != path.stat().st_size:
            _append_error(errors, f"metadata receipt length mismatch: {path.name}")

    raw_page_data = source_dir / "raw" / "page-data.json"
    if raw_page_data.is_file():
        try:
            indexed = source.parse_published_reports(
                raw_page_data.read_bytes(),
                start=source.MIN_REPORT_DATE,
                end=source.MAX_REPORT_DATE,
            )
            indexed_dates = {row.record_date.isoformat() for row in indexed}
            if indexed_dates != set(reports):
                _append_error(errors, "manifest report dates differ from official index")
            gaps = source._weekday_gap_audit(
                indexed,
                start=source.MIN_REPORT_DATE,
                end=source.MAX_REPORT_DATE,
            )
            if gaps:
                _append_error(errors, f"unexplained weekday gaps: {gaps[:10]}")
        except ValueError as exc:
            _append_error(errors, f"official index audit failed: {exc}")

    for error in _verify_physical_inventory(source_dir, reports):
        _append_error(errors, error)

    for record_date_iso, report in sorted(reports.items()):
        try:
            record_date = date.fromisoformat(record_date_iso)
        except ValueError:
            _append_error(errors, f"invalid manifest report date: {record_date_iso}")
            continue
        if not source.MIN_REPORT_DATE <= record_date <= source.MAX_REPORT_DATE:
            _append_error(errors, f"post-cap/out-of-range report date: {record_date_iso}")
            continue
        expected_url = source.report_url(record_date)
        if report.get("url") != expected_url:
            _append_error(errors, f"report URL mismatch: {record_date_iso}")
        raw_pdf = source_dir / "raw" / "reports" / f"{record_date:%Y%m%d}.pdf"
        if not raw_pdf.is_file():
            _append_error(errors, f"missing report PDF: {record_date_iso}")
            continue
        actual_hash = source.sha256_file(raw_pdf)
        if report.get("sha256") != actual_hash:
            _append_error(errors, f"report hash mismatch: {record_date_iso}")
        if report.get("byte_length") != raw_pdf.stat().st_size:
            _append_error(errors, f"report length mismatch: {record_date_iso}")
        if report.get("table_ids_found") != ["I", "II", "IIIA"]:
            _append_error(errors, f"required table set mismatch: {record_date_iso}")
        available, execution, stage = source.source_clock(record_date)
        if report.get("source_available_not_before_utc") != available.isoformat():
            _append_error(errors, f"availability clock mismatch: {record_date_iso}")
        if report.get("earliest_execution_time_utc") != execution.isoformat():
            _append_error(errors, f"execution clock mismatch: {record_date_iso}")
        if report.get("research_stage") != stage:
            _append_error(errors, f"research stage mismatch: {record_date_iso}")
        receipt_path = source._receipt_path(raw_pdf)
        try:
            receipt = _read_json(receipt_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            _append_error(errors, f"invalid report receipt {record_date_iso}: {exc}")
            continue
        if (
            receipt.get("url") != expected_url
            or receipt.get("final_url") != expected_url
            or receipt.get("status") != 200
            or receipt.get("sha256") != actual_hash
            or receipt.get("byte_length") != raw_pdf.stat().st_size
        ):
            _append_error(errors, f"report receipt binding mismatch: {record_date_iso}")

    normalized_specs = (
        (
            "normalized_rows",
            source_dir / "daily_treasury_fiscal_flow_rows.csv.gz",
            len(snapshot.rows),
        ),
        (
            "operating_cash_rows",
            source_dir / "daily_treasury_operating_cash_rows.csv.gz",
            len(snapshot.table_i_rows),
        ),
        (
            "schema_announcements",
            source_dir / "precap_schema_announcements.csv.gz",
            len(snapshot.announcements),
        ),
    )
    for key, path, row_count in normalized_specs:
        spec = manifest.get(key)
        if not isinstance(spec, dict):
            _append_error(errors, f"manifest lacks {key}")
            continue
        if spec.get("row_count") != row_count:
            _append_error(errors, f"row-count mismatch for {key}")
        if spec.get("file_sha256") != source.sha256_file(path):
            _append_error(errors, f"compressed hash mismatch for {key}")
        if spec.get("uncompressed_sha256") != _sha256_uncompressed_gzip(path):
            _append_error(errors, f"uncompressed hash mismatch for {key}")

    row_counts: Counter[str] = Counter()
    table_i_counts: Counter[str] = Counter()
    for row in snapshot.rows:
        record_date = row.get("record_date", "")
        report = reports.get(record_date)
        if report is None:
            _append_error(errors, f"normalized row lacks report: {record_date}")
            continue
        if row.get("source_pdf_sha256") != report.get("sha256"):
            _append_error(errors, f"normalized row PDF hash mismatch: {record_date}")
        if row.get("research_stage") != report.get("research_stage"):
            _append_error(errors, f"normalized row stage mismatch: {record_date}")
        if row.get("table_id") not in {"II", "IIIA"}:
            _append_error(errors, f"unexpected normalized table: {row.get('table_id')}")
        if (row.get("table_id"), row.get("side")) not in {
            ("II", "deposit"),
            ("II", "withdrawal"),
            ("IIIA", "issue"),
            ("IIIA", "redemption"),
        }:
            _append_error(errors, f"unexpected normalized side: {record_date}")
        row_counts[record_date] += 1
    for row in snapshot.table_i_rows:
        record_date = row.get("record_date", "")
        report = reports.get(record_date)
        if report is None:
            _append_error(errors, f"Table I row lacks report: {record_date}")
            continue
        if row.get("source_pdf_sha256") != report.get("sha256"):
            _append_error(errors, f"Table I PDF hash mismatch: {record_date}")
        table_i_counts[record_date] += 1
    for record_date, report in reports.items():
        if row_counts[record_date] != report.get("row_count"):
            _append_error(errors, f"manifest detail row count mismatch: {record_date}")
        if table_i_counts[record_date] != report.get("table_i_row_count"):
            _append_error(errors, f"manifest Table I row count mismatch: {record_date}")

    if len(snapshot.announcements) != metadata.get("precap_announcement_count"):
        _append_error(errors, "announcement count differs from manifest")
    if any(row.get("effective_date", "9999") > source.MAX_REPORT_DATE.isoformat() for row in snapshot.announcements):
        _append_error(errors, "post-cap announcement row entered normalized logic")

    expected_protocol = {
        "source_only": True,
        "post_2023_report_opened": False,
        "post_2023_api_value_row_opened": False,
        "btc_market_data_opened": False,
        "funding_opened": False,
        "future_return_opened": False,
        "labels_opened": False,
        "pnl_cagr_mdd_opened": False,
        "current_metadata_postcap_rows_used_in_logic": False,
    }
    if build_report.get("protocol") != expected_protocol:
        _append_error(errors, "source-build protocol flags are not source-only")
    if build_report.get("decision") != "SOURCE_BUILT_REQUIRES_SCHEMA_AUDIT":
        _append_error(errors, "source build incorrectly claims audit authorization")
    if build_report.get("report_count") != len(reports):
        _append_error(errors, "build report count differs from the official index")
    if build_report.get("normalized_row_count") != len(snapshot.rows):
        _append_error(errors, "build report normalized-row count mismatch")
    if build_report.get("operating_cash_row_count") != len(snapshot.table_i_rows):
        _append_error(errors, "build report Table I row count mismatch")
    if build_report.get("source_manifest_sha256") != source.sha256_file(
        source_dir / "source_manifest.json"
    ):
        _append_error(errors, "build report manifest hash mismatch")
    if not build_report.get("calendar_coverage_gate_pass"):
        _append_error(errors, "calendar coverage gate did not pass")
    return not errors, errors


def audit_numeric_roundtrip(
    snapshot: SourceSnapshot,
) -> tuple[list[dict[str, Any]], list[str], int]:
    summaries: dict[tuple[str, str, str], list[int]] = defaultdict(
        lambda: [0, 0, 0]
    )
    errors: list[str] = []
    checked = 0
    for row in snapshot.rows:
        key = (row["record_date"], row["table_id"], row["side"])
        for value_column, literal_column, _ in AMOUNT_COLUMNS:
            literal = row.get(literal_column, "")
            expected = None if row.get(value_column, "") == "" else int(row[value_column])
            recognized, parsed, _, missing, parsed_literal = source._parse_amount_cell(
                literal
            )
            checked += 1
            summaries[key][0] += 1
            if expected is None:
                summaries[key][1] += 1
            if (
                not recognized
                or parsed != expected
                or parsed_literal != literal
                or bool(missing) != (expected is None)
            ):
                summaries[key][2] += 1
                _append_error(
                    errors,
                    f"numeric literal mismatch {row['record_date']} "
                    f"{row['table_id']}/{row['side']} {literal_column}={literal!r}",
                )

    for row in snapshot.table_i_rows:
        key = (row["record_date"], "I", "operating_cash")
        try:
            values = json.loads(row["published_values_usd_millions_json"])
            literals = json.loads(row["published_value_literals_json"])
        except (json.JSONDecodeError, KeyError) as exc:
            _append_error(errors, f"invalid Table I literal JSON: {exc}")
            summaries[key][2] += 1
            continue
        if not isinstance(values, list) or not isinstance(literals, list) or len(values) != len(literals):
            _append_error(errors, f"Table I literal/value length mismatch: {row['record_date']}")
            summaries[key][2] += 1
            continue
        for expected, literal in zip(values, literals):
            recognized, parsed, _, missing, parsed_literal = source._parse_amount_cell(
                str(literal)
            )
            checked += 1
            summaries[key][0] += 1
            if expected is None:
                summaries[key][1] += 1
            if (
                not recognized
                or parsed != expected
                or parsed_literal != literal
                or bool(missing) != (expected is None)
            ):
                summaries[key][2] += 1
                _append_error(
                    errors,
                    f"Table I numeric literal mismatch {row['record_date']}: {literal!r}",
                )

    output = [
        {
            "record_date": key[0],
            "table_id": key[1],
            "side": key[2],
            "checked_cell_count": values[0],
            "missing_cell_count": values[1],
            "failed_cell_count": values[2],
            "status": "PASS" if values[2] == 0 else "FAIL",
        }
        for key, values in sorted(summaries.items())
    ]
    return output, errors, checked


def _position(row: dict[str, str]) -> tuple[int, int]:
    return int(row["page_number"]), int(row["source_order"])


def _amount(row: dict[str, str], column: str) -> int:
    value = row.get(column, "")
    if value == "":
        raise ValueError(
            f"missing amount in {row.get('record_date')} {row.get('raw_category_label')} {column}"
        )
    return int(value)


def _unique_row(
    rows: Sequence[dict[str, str]], labels: str | Iterable[str]
) -> dict[str, str]:
    accepted = {labels} if isinstance(labels, str) else set(labels)
    matches = [row for row in rows if row["raw_category_label"] in accepted]
    if len(matches) != 1:
        raise ValueError(f"expected one row for labels {sorted(accepted)}, found {len(matches)}")
    return matches[0]


def _append_reconciliation(
    output: list[dict[str, Any]],
    *,
    record_date: str,
    table_id: str,
    side: str,
    check_name: str,
    amount_column: str,
    observed: int,
    components: Sequence[int],
    signs: Sequence[int] | None = None,
    notes: str = "",
) -> None:
    applied_signs = tuple(signs) if signs is not None else (1,) * len(components)
    if len(applied_signs) != len(components):
        raise ValueError("component/sign length mismatch")
    computed = sum(sign * value for sign, value in zip(applied_signs, components))
    residual = observed - computed
    tolerance = rounding_tolerance(len(components))
    anomaly_key = (record_date, check_name, amount_column, observed, computed)
    if abs(residual) <= tolerance:
        status = "PASS"
    elif anomaly_key in KNOWN_PUBLISHED_ACCOUNTING_ANOMALIES:
        status = "PUBLISHED_SOURCE_ANOMALY"
        notes = (
            f"{notes} " if notes else ""
        ) + "Exact literal values are bound to the official PDF and quarantined."
    else:
        status = "FAIL"
    output.append(
        {
            "record_date": record_date,
            "table_id": table_id,
            "side": side,
            "check_name": check_name,
            "amount_column": amount_column,
            "observed_total": observed,
            "computed_total": computed,
            "component_count": len(components),
            "residual_usd_millions": residual,
            "tolerance_usd_millions": f"{tolerance:.1f}",
            "status": status,
            "notes": notes,
        }
    )


def _table_i_values(row: dict[str, str]) -> list[int]:
    values = json.loads(row["published_values_usd_millions_json"])
    if not isinstance(values, list) or not all(isinstance(value, int) for value in values):
        raise ValueError("Table I values are not integer JSON")
    return values


def audit_accounting(
    snapshot: SourceSnapshot,
) -> tuple[list[dict[str, Any]], list[str]]:
    by_table_side: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    table_i_by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in snapshot.rows:
        by_table_side[(row["record_date"], row["table_id"], row["side"])].append(row)
    for row in snapshot.table_i_rows:
        table_i_by_date[row["record_date"]].append(row)

    output: list[dict[str, Any]] = []
    errors: list[str] = []
    for record_date in sorted(table_i_by_date):
        try:
            table_i_rows = table_i_by_date[record_date]
            variants = {row["schema_variant"] for row in table_i_rows}
            if len(table_i_rows) != 4 or len(variants) != 1:
                raise ValueError("Table I must contain four rows of one schema variant")
            variant = next(iter(variants))
            if variant == "legacy_four_column":
                total = _unique_row(table_i_rows, "Total Operating Balance")
                components = [row for row in table_i_rows if row is not total]
                total_values = _table_i_values(total)
                component_values = [_table_i_values(row) for row in components]
                for index, observed in enumerate(total_values):
                    _append_reconciliation(
                        output,
                        record_date=record_date,
                        table_id="I",
                        side="operating_cash",
                        check_name="legacy_total_operating_balance_components",
                        amount_column=f"published_column_{index}",
                        observed=observed,
                        components=[values[index] for values in component_values],
                    )
                net = _unique_row(
                    by_table_side[(record_date, "II", "withdrawal")],
                    "Net Change in Operating Cash Balance",
                )
                for opening_index, net_column, name in (
                    (1, "today_amount_usd_millions", "today"),
                    (2, "month_to_date_amount_usd_millions", "month_to_date"),
                    (3, "fiscal_year_to_date_amount_usd_millions", "fiscal_year_to_date"),
                ):
                    _append_reconciliation(
                        output,
                        record_date=record_date,
                        table_id="I",
                        side="operating_cash",
                        check_name="legacy_opening_plus_net_equals_closing",
                        amount_column=name,
                        observed=total_values[0],
                        components=[total_values[opening_index], _amount(net, net_column)],
                    )
            elif variant == "modern_three_column":
                opening = _unique_row(
                    table_i_rows, "Treasury General Account (TGA) Opening Balance"
                )
                deposits = _unique_row(table_i_rows, "Total TGA Deposits (Table II)")
                withdrawals = _unique_row(
                    table_i_rows, "Total TGA Withdrawals (Table II) (-)"
                )
                closing = _unique_row(
                    table_i_rows, "Treasury General Account (TGA) Closing Balance"
                )
                opening_values = _table_i_values(opening)
                deposit_values = _table_i_values(deposits)
                withdrawal_values = _table_i_values(withdrawals)
                closing_values = _table_i_values(closing)
                for index, name in enumerate(
                    ("today", "month_to_date", "fiscal_year_to_date")
                ):
                    _append_reconciliation(
                        output,
                        record_date=record_date,
                        table_id="I",
                        side="operating_cash",
                        check_name="modern_opening_plus_deposits_minus_withdrawals_equals_closing",
                        amount_column=name,
                        observed=closing_values[index],
                        components=[
                            opening_values[index],
                            deposit_values[index],
                            withdrawal_values[index],
                        ],
                        signs=[1, 1, -1],
                    )
            else:
                raise ValueError(f"unknown Table I schema variant: {variant}")

            account_totals: dict[str, dict[str, str]] = {}
            for side in ("deposit", "withdrawal"):
                rows = by_table_side[(record_date, "II", side)]
                if side == "deposit":
                    account_labels = (
                        "Total Federal Reserve Account",
                        "Total TGA Deposits",
                        "Treasury General Account Total Deposits",
                    )
                    other_labels = ("Total Other Deposits",)
                    debt_labels = (
                        "Public Debt Cash Issues (Table III-B)",
                        "Public Debt Cash Issues (Table IIIB)",
                    )
                    broad_labels = (
                        "Total Deposits (excluding transfers)",
                        "Sub-Total Deposits",
                        "Total Deposits",
                    )
                else:
                    account_labels = (
                        "Total Federal Reserve Account",
                        "Total TGA Withdrawals",
                        "Treasury General Account Total Withdrawals",
                    )
                    other_labels = (
                        "Total, Other Withdrawals",
                        "Total Other Withdrawals",
                    )
                    debt_labels = (
                        "Public Debt Cash Redemp. (Table III-B)",
                        "Public Debt Cash Redemp. (Table IIIB)",
                    )
                    broad_labels = (
                        "Total Withdrawals (excluding transfers)",
                        "Sub-Total Withdrawals",
                        "Total Withdrawals",
                    )
                account = _unique_row(rows, account_labels)
                broad = _unique_row(rows, broad_labels)
                account_totals[side] = account

                other_matches = [
                    row for row in rows if row["raw_category_label"] in other_labels
                ]
                if len(other_matches) > 1:
                    raise ValueError(
                        f"expected at most one row for labels {sorted(other_labels)}, "
                        f"found {len(other_matches)}"
                    )
                if other_matches:
                    other = other_matches[0]
                    other_components = [
                        row
                        for row in rows
                        if row["row_kind"] == "detail"
                        and row["parent_section"] == other["parent_section"]
                        and _position(row) < _position(other)
                    ]
                    _append_reconciliation(
                        output,
                        record_date=record_date,
                        table_id="II",
                        side=side,
                        check_name="other_flow_subtotal_components",
                        amount_column="today",
                        observed=_amount(other, "today_amount_usd_millions"),
                        components=[
                            _amount(row, "today_amount_usd_millions")
                            for row in other_components
                        ],
                        notes=(
                            "MTD/FYTD detail rows are sparse in historical PDFs; "
                            "the displayed-component hierarchy is audited on today values."
                        ),
                    )
                elif broad["raw_category_label"] not in {
                    "Total Deposits",
                    "Total Withdrawals",
                }:
                    raise ValueError(
                        "report-defined broad subtotal requires an Other-flow subtotal: "
                        f"{sorted(other_labels)}"
                    )

                if variant == "legacy_four_column":
                    account_components = [
                        row
                        for row in rows
                        if row["row_kind"] == "detail"
                        and _position(row) < _position(account)
                    ]
                    _append_reconciliation(
                        output,
                        record_date=record_date,
                        table_id="II",
                        side=side,
                        check_name="legacy_account_total_components",
                        amount_column="today",
                        observed=_amount(account, "today_amount_usd_millions"),
                        components=[
                            _amount(row, "today_amount_usd_millions")
                            for row in account_components
                        ],
                    )
                    for value_column, _, name in AMOUNT_COLUMNS:
                        _append_reconciliation(
                            output,
                            record_date=record_date,
                            table_id="II",
                            side=side,
                            check_name="legacy_excluding_transfers_equals_account_total",
                            amount_column=name,
                            observed=_amount(broad, value_column),
                            components=[_amount(account, value_column)],
                        )
                else:
                    broad_components = [
                        row
                        for row in rows
                        if row["row_kind"] == "detail"
                        and row["raw_category_label"]
                        not in {"Sub-Total Deposits", "Sub-Total Withdrawals"}
                        and _position(row) < _position(broad)
                    ]
                    _append_reconciliation(
                        output,
                        record_date=record_date,
                        table_id="II",
                        side=side,
                        check_name="modern_subtotal_components",
                        amount_column="today",
                        observed=_amount(broad, "today_amount_usd_millions"),
                        components=[
                            _amount(row, "today_amount_usd_millions")
                            for row in broad_components
                        ],
                    )
                    debt = _unique_row(rows, debt_labels)
                    for value_column, _, name in AMOUNT_COLUMNS:
                        _append_reconciliation(
                            output,
                            record_date=record_date,
                            table_id="II",
                            side=side,
                            check_name="modern_subtotal_plus_public_debt_equals_account_total",
                            amount_column=name,
                            observed=_amount(account, value_column),
                            components=[
                                _amount(broad, value_column),
                                _amount(debt, value_column),
                            ],
                        )

            net = _unique_row(
                by_table_side[(record_date, "II", "withdrawal")],
                "Net Change in Operating Cash Balance",
            )
            for value_column, _, name in AMOUNT_COLUMNS:
                _append_reconciliation(
                    output,
                    record_date=record_date,
                    table_id="II",
                    side="both",
                    check_name="account_deposits_minus_withdrawals_equals_net_change",
                    amount_column=name,
                    observed=_amount(net, value_column),
                    components=[
                        _amount(account_totals["deposit"], value_column),
                        _amount(account_totals["withdrawal"], value_column),
                    ],
                    signs=[1, -1],
                )

            if variant == "modern_three_column":
                table_i_deposits = _unique_row(
                    table_i_rows, "Total TGA Deposits (Table II)"
                )
                table_i_withdrawals = _unique_row(
                    table_i_rows, "Total TGA Withdrawals (Table II) (-)"
                )
                for side, table_i_total in (
                    ("deposit", table_i_deposits),
                    ("withdrawal", table_i_withdrawals),
                ):
                    values = _table_i_values(table_i_total)
                    for index, (value_column, _, name) in enumerate(AMOUNT_COLUMNS):
                        _append_reconciliation(
                            output,
                            record_date=record_date,
                            table_id="I/II",
                            side=side,
                            check_name="table_i_equals_table_ii_account_total",
                            amount_column=name,
                            observed=values[index],
                            components=[_amount(account_totals[side], value_column)],
                        )

            issue_rows = by_table_side[(record_date, "IIIA", "issue")]
            redemption_rows = by_table_side[(record_date, "IIIA", "redemption")]
            total_issues = _unique_row(issue_rows, "Total Issues")
            total_redemptions = _unique_row(redemption_rows, "Total Redemptions")
            for side, rows, total in (
                ("issue", issue_rows, total_issues),
                ("redemption", redemption_rows, total_redemptions),
            ):
                components = [
                    row
                    for row in rows
                    if row["row_kind"] == "detail" and _position(row) < _position(total)
                ]
                for value_column, _, name in AMOUNT_COLUMNS:
                    _append_reconciliation(
                        output,
                        record_date=record_date,
                        table_id="IIIA",
                        side=side,
                        check_name="public_debt_total_components",
                        amount_column=name,
                        observed=_amount(total, value_column),
                        components=[_amount(row, value_column) for row in components],
                    )
            debt_net = _unique_row(
                redemption_rows, "Net Change in Public Debt Outstanding"
            )
            for value_column, _, name in AMOUNT_COLUMNS:
                _append_reconciliation(
                    output,
                    record_date=record_date,
                    table_id="IIIA",
                    side="both",
                    check_name="issues_minus_redemptions_equals_debt_net_change",
                    amount_column=name,
                    observed=_amount(debt_net, value_column),
                    components=[
                        _amount(total_issues, value_column),
                        _amount(total_redemptions, value_column),
                    ],
                    signs=[1, -1],
                )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            _append_error(errors, f"accounting audit failed for {record_date}: {exc}")

    failed = [row for row in output if row["status"] == "FAIL"]
    for row in failed[: max(0, 200 - len(errors))]:
        _append_error(
            errors,
            f"accounting residual {row['record_date']} {row['check_name']} "
            f"{row['amount_column']}={row['residual_usd_millions']}",
        )
    return output, errors


def audit_duplicate_labels(
    snapshot: SourceSnapshot,
) -> tuple[list[dict[str, Any]], list[str]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in snapshot.rows:
        grouped[
            (
                row["record_date"],
                row["table_id"],
                row["side"],
                row["normalized_category_label"],
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    errors: list[str] = []
    for key, rows in sorted(grouped.items()):
        if len(rows) < 2:
            continue
        status = "FAIL"
        output.append(
            {
                "record_date": key[0],
                "table_id": key[1],
                "side": key[2],
                "normalized_category_label": key[3],
                "occurrence_count": len(rows),
                "raw_labels_json": source.canonical_json(
                    sorted(row["raw_category_label"] for row in rows)
                ).decode("utf-8").strip(),
                "parent_sections_json": source.canonical_json(
                    sorted(row["parent_section"] for row in rows)
                ).decode("utf-8").strip(),
                "status": status,
            }
        )
        _append_error(errors, f"duplicate normalized label: {key}")
    return output, errors


def _snapshot_labels(
    snapshot: SourceSnapshot,
) -> tuple[
    list[str],
    dict[str, dict[tuple[str, str, str], tuple[str, str]]],
    dict[str, str],
]:
    report_dates = sorted(_report_map(snapshot.manifest))
    labels: dict[str, dict[tuple[str, str, str], tuple[str, str]]] = {
        record_date: {} for record_date in report_dates
    }
    variants: dict[str, str] = {}
    for row in snapshot.rows:
        key = (
            row["table_id"],
            row["side"],
            row["normalized_category_label"],
        )
        labels[row["record_date"]][key] = (
            row["raw_category_label"],
            row["parent_section"],
        )
    for row in snapshot.table_i_rows:
        key = ("I", "operating_cash", row["normalized_category_label"])
        labels[row["record_date"]][key] = (row["raw_category_label"], "")
        previous = variants.get(row["record_date"])
        if previous is not None and previous != row["schema_variant"]:
            raise ValueError(f"mixed Table I variants on {row['record_date']}")
        variants[row["record_date"]] = row["schema_variant"]
    return report_dates, labels, variants


def _schema_label_key(value: str) -> str:
    """Return a case- and punctuation-insensitive schema-label identity."""

    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _presence_transition_has_announcement_support(
    snapshot: SourceSnapshot,
    transition: dict[str, Any],
) -> bool:
    transition_text = " ".join(
        str(transition[key])
        for key in (
            "normalized_category_label",
            "before_label",
            "after_label",
        )
    )
    transition_tokens = _meaningful_tokens(transition_text)
    target = (transition["table_id"], transition["side"])
    for announcement in snapshot.announcements:
        effective = announcement.get("effective_date", "")
        direct_date = effective in {
            transition["report_date"],
            transition["source_available_date_new_york"],
        }
        if not direct_date:
            continue
        announcement_text = (
            f"{announcement.get('entity', '')} {announcement.get('change', '')}"
        )
        overlap = transition_tokens & _meaningful_tokens(announcement_text)
        scope = _announcement_scope(
            announcement.get("table_name", ""),
            announcement.get("table_section", ""),
            announcement.get("change", ""),
        )
        if target in scope and overlap:
            return True
        if len(overlap) >= 2:
            return True
    return False


def _consolidate_presence_transitions(
    snapshot: SourceSnapshot,
    candidates: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Separate structural label changes from value-driven sparse row presence.

    Historical DTS PDFs omit many zero-activity detail rows.  Five consecutive
    present/absent reports therefore do not by themselves establish a schema
    birth or death.  We retain presence changes only when an official pre-cap
    announcement supports them, or when a same-boundary birth/death pair is
    identical after case and punctuation normalization.
    """

    grouped: dict[tuple[str, str, str], list[tuple[int, dict[str, Any]]]] = (
        defaultdict(list)
    )
    for index, candidate in enumerate(candidates):
        if candidate["transition_type"] in {"stable_birth", "stable_death"}:
            grouped[
                (
                    str(candidate["report_date"]),
                    str(candidate["table_id"]),
                    str(candidate["side"]),
                )
            ].append((index, candidate))

    consumed: set[int] = set()
    replacements: list[dict[str, Any]] = []
    for entries in grouped.values():
        births = [item for item in entries if item[1]["transition_type"] == "stable_birth"]
        deaths = [item for item in entries if item[1]["transition_type"] == "stable_death"]
        for death_index, death in deaths:
            death_key = _schema_label_key(str(death["before_label"]))
            matches = [
                (birth_index, birth)
                for birth_index, birth in births
                if birth_index not in consumed
                and death_key == _schema_label_key(str(birth["after_label"]))
                and death["before_parent_section"] == birth["after_parent_section"]
            ]
            if len(matches) != 1:
                continue
            birth_index, birth = matches[0]
            consumed.update({death_index, birth_index})
            replacements.append(
                {
                    **birth,
                    "transition_type": "stable_label_change",
                    "before_label": death["before_label"],
                    "before_parent_section": death["before_parent_section"],
                }
            )

    output: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if index in consumed:
            continue
        if candidate["transition_type"] in {"stable_birth", "stable_death"}:
            if not _presence_transition_has_announcement_support(
                snapshot, candidate
            ):
                continue
        output.append(candidate)
    output.extend(replacements)
    return output


def detect_schema_transitions(
    snapshot: SourceSnapshot,
) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    try:
        dates, labels, variants = _snapshot_labels(snapshot)
    except ValueError as exc:
        return [], [str(exc)]
    reports = _report_map(snapshot.manifest)
    all_keys = sorted({key for values in labels.values() for key in values})
    output: list[dict[str, Any]] = []
    window = STABILITY_WINDOW_REPORTS

    def availability_date(record_date: str) -> str:
        stamp = datetime.fromisoformat(
            str(reports[record_date]["source_available_not_before_utc"])
        )
        return stamp.astimezone(NEW_YORK).date().isoformat()

    for key in all_keys:
        presence = [key in labels[record_date] for record_date in dates]
        for index in range(window, len(dates) - window + 1):
            previous = presence[index - window : index]
            following = presence[index : index + window]
            if len(following) < window:
                continue
            if not any(previous) and all(following):
                raw, parent = labels[dates[index]][key]
                output.append(
                    {
                        "report_date": dates[index],
                        "source_available_date_new_york": availability_date(dates[index]),
                        "table_id": key[0],
                        "side": key[1],
                        "transition_type": "stable_birth",
                        "normalized_category_label": key[2],
                        "before_label": "",
                        "after_label": raw,
                        "before_parent_section": "",
                        "after_parent_section": parent,
                        "support_before_reports": window,
                        "support_after_reports": window,
                        "announcement_match_count": 0,
                        "status": "UNRECONCILED",
                    }
                )
            elif all(previous) and not any(following):
                raw, parent = labels[dates[index - 1]][key]
                output.append(
                    {
                        "report_date": dates[index],
                        "source_available_date_new_york": availability_date(dates[index]),
                        "table_id": key[0],
                        "side": key[1],
                        "transition_type": "stable_death",
                        "normalized_category_label": key[2],
                        "before_label": raw,
                        "after_label": "",
                        "before_parent_section": parent,
                        "after_parent_section": "",
                        "support_before_reports": window,
                        "support_after_reports": window,
                        "announcement_match_count": 0,
                        "status": "UNRECONCILED",
                    }
                )
        for index in range(window, len(dates) - window):
            if not all(presence[index - window : index + window]):
                continue
            before_values = [labels[dates[pos]][key][1] for pos in range(index - window, index)]
            after_values = [labels[dates[pos]][key][1] for pos in range(index, index + window)]
            if len(set(before_values)) == 1 and len(set(after_values)) == 1 and before_values[0] != after_values[0]:
                raw_before, parent_before = labels[dates[index - 1]][key]
                raw_after, parent_after = labels[dates[index]][key]
                output.append(
                    {
                        "report_date": dates[index],
                        "source_available_date_new_york": availability_date(dates[index]),
                        "table_id": key[0],
                        "side": key[1],
                        "transition_type": "stable_parent_section_change",
                        "normalized_category_label": key[2],
                        "before_label": raw_before,
                        "after_label": raw_after,
                        "before_parent_section": parent_before,
                        "after_parent_section": parent_after,
                        "support_before_reports": window,
                        "support_after_reports": window,
                        "announcement_match_count": 0,
                        "status": "UNRECONCILED",
                    }
                )

    output = _consolidate_presence_transitions(snapshot, output)

    ordered_variants = [variants.get(record_date, "") for record_date in dates]
    variant_changes = [
        index
        for index in range(1, len(dates))
        if ordered_variants[index] != ordered_variants[index - 1]
    ]
    if ordered_variants[0] != "legacy_four_column":
        _append_error(errors, "Table I does not begin with legacy_four_column")
    if ordered_variants[-1] != "modern_three_column":
        _append_error(errors, "Table I does not end with modern_three_column")
    if len(variant_changes) != 1:
        _append_error(errors, f"Table I has {len(variant_changes)} schema transitions")
    for index in variant_changes:
        output.append(
            {
                "report_date": dates[index],
                "source_available_date_new_york": availability_date(dates[index]),
                "table_id": "I",
                "side": "operating_cash",
                "transition_type": "column_schema_change",
                "normalized_category_label": "TABLE I",
                "before_label": ordered_variants[index - 1],
                "after_label": ordered_variants[index],
                "before_parent_section": "",
                "after_parent_section": "",
                "support_before_reports": window,
                "support_after_reports": window,
                "announcement_match_count": 0,
                "status": "UNRECONCILED",
            }
        )
    output.sort(
        key=lambda row: (
            row["report_date"],
            row["table_id"],
            row["side"],
            row["transition_type"],
            row["normalized_category_label"],
        )
    )
    return output, errors


_TOKEN_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "both",
    "by",
    "change",
    "changing",
    "daily",
    "for",
    "from",
    "in",
    "into",
    "line",
    "now",
    "of",
    "on",
    "previously",
    "reported",
    "section",
    "show",
    "table",
    "the",
    "this",
    "to",
    "will",
}


def _meaningful_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) >= 2 and token not in _TOKEN_STOPWORDS
    }


def _announcement_scope(
    table_name: str, table_section: str, change: str = ""
) -> set[tuple[str, str]]:
    compact = re.sub(r"\s+", "", table_name.upper())
    tables: set[str] = set()
    change_lower = change.casefold()
    if compact == "VARIOUS" or (
        "all references" in change_lower and "throughout the dts" in change_lower
    ):
        tables.update({"I", "II", "IIIA"})
    if re.search(r"(?:^|[^0-9])1(?:$|[^0-9])", compact):
        tables.add("I")
    if re.search(r"(?:^|[^0-9])2(?:$|[^0-9])", compact):
        tables.add("II")
    if "3A" in compact:
        tables.add("IIIA")
    if not tables:
        if re.search(r"\btable\s+(?:2|ii)(?!i)", change_lower):
            tables.add("II")
        if re.search(r"\btable\s+(?:3a|iii-?a)\b", change_lower):
            tables.add("IIIA")
        if re.search(r"\btable\s+(?:1|i)(?!i)\b", change_lower):
            tables.add("I")
    if not tables:
        return set()
    section = table_section.casefold()
    output: set[tuple[str, str]] = set()
    for table in tables:
        if table == "I":
            output.add((table, "operating_cash"))
        elif table == "IIIA":
            if "issue" in section:
                output.add((table, "issue"))
            elif "redemption" in section:
                output.add((table, "redemption"))
            else:
                output.update({(table, "issue"), (table, "redemption")})
        elif "deposit" in section and "withdraw" not in section:
            output.add((table, "deposit"))
        elif "withdraw" in section and "deposit" not in section:
            output.add((table, "withdrawal"))
        else:
            output.update({(table, "deposit"), (table, "withdrawal")})
    return output


def reconcile_announcements(
    snapshot: SourceSnapshot, transitions: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    dates, labels, _ = _snapshot_labels(snapshot)
    reports = _report_map(snapshot.manifest)
    availability_to_reports: dict[str, list[str]] = defaultdict(list)
    for record_date, report in reports.items():
        available = datetime.fromisoformat(
            str(report["source_available_not_before_utc"])
        ).astimezone(NEW_YORK).date().isoformat()
        availability_to_reports[available].append(record_date)

    output: list[dict[str, Any]] = []
    errors: list[str] = []
    transition_matches: Counter[int] = Counter()
    for announcement in snapshot.announcements:
        effective = announcement["effective_date"]
        scope = _announcement_scope(
            announcement.get("table_name", ""),
            announcement.get("table_section", ""),
            announcement.get("change", ""),
        )
        scope_json = source.canonical_json(sorted([list(item) for item in scope])).decode(
            "utf-8"
        ).strip()
        if not scope:
            output.append(
                {
                    **announcement,
                    "scope_json": scope_json,
                    "matched_transition_count": 0,
                    "matched_label_count": 0,
                    "status": "OUT_OF_RETAINED_SCOPE",
                }
            )
            continue
        relevant_transitions: list[tuple[int, dict[str, Any]]] = []
        for index, transition in enumerate(transitions):
            if (transition["table_id"], transition["side"]) not in scope:
                continue
            if effective in {
                transition["report_date"],
                transition["source_available_date_new_york"],
            }:
                relevant_transitions.append((index, transition))

        candidate_dates = set(availability_to_reports.get(effective, []))
        if effective in labels:
            candidate_dates.add(effective)
        effective_date = date.fromisoformat(effective)
        first_on_or_after = next(
            (record_date for record_date in dates if date.fromisoformat(record_date) >= effective_date),
            None,
        )
        if first_on_or_after is not None:
            candidate_dates.add(first_on_or_after)
        label_texts: list[str] = []
        for record_date in sorted(candidate_dates):
            for key, (raw_label, parent) in labels[record_date].items():
                if (key[0], key[1]) in scope:
                    label_texts.append(f"{raw_label} {parent}")
        announcement_tokens = _meaningful_tokens(
            f"{announcement.get('entity', '')} {announcement.get('change', '')}"
        )
        matched_labels = sum(
            1
            for text in label_texts
            if len(announcement_tokens & _meaningful_tokens(text)) >= 2
        )
        matched_transition_indexes: list[int] = []
        for index, transition in relevant_transitions:
            transition_text = " ".join(
                str(transition[key])
                for key in (
                    "normalized_category_label",
                    "before_label",
                    "after_label",
                    "before_parent_section",
                    "after_parent_section",
                )
            )
            overlap = announcement_tokens & _meaningful_tokens(transition_text)
            if overlap or transition["transition_type"] == "column_schema_change":
                matched_transition_indexes.append(index)
        if matched_transition_indexes:
            status = "MATCHED_TRANSITION"
        elif matched_labels:
            status = "MATCHED_LABEL_EVIDENCE"
        elif relevant_transitions:
            status = "MATCHED_SCOPE_TRANSITION"
            matched_transition_indexes = [index for index, _ in relevant_transitions]
        else:
            status = "UNMATCHED"
            _append_error(
                errors,
                f"unmatched announcement {effective} {announcement.get('sheet_name')} "
                f"row {announcement.get('source_row')}",
            )
        for index in matched_transition_indexes:
            transition_matches[index] += 1
        output.append(
            {
                **announcement,
                "scope_json": scope_json,
                "matched_transition_count": len(matched_transition_indexes),
                "matched_label_count": matched_labels,
                "status": status,
            }
        )

    for index, transition in enumerate(transitions):
        matches = transition_matches[index]
        transition["announcement_match_count"] = matches
        source_equivalent = (
            transition["transition_type"] == "stable_label_change"
            and _schema_label_key(str(transition["before_label"]))
            == _schema_label_key(str(transition["after_label"]))
        )
        if matches:
            transition["status"] = "RECONCILED_ANNOUNCEMENT"
        elif source_equivalent:
            transition["status"] = "RECONCILED_SOURCE_EQUIVALENCE"
        else:
            transition["status"] = "UNRECONCILED"
            _append_error(
                errors,
                f"unannounced stable transition {transition['report_date']} "
                f"{transition['table_id']}/{transition['side']} "
                f"{transition['transition_type']} {transition['normalized_category_label']}",
            )
    output.sort(
        key=lambda row: (
            row["effective_date"],
            row["sheet_name"],
            int(row["source_row"]),
        )
    )
    return output, errors


def verify_clean_rerun(
    snapshot: SourceSnapshot,
    *,
    max_workers: int,
    source_binding_verified: bool = False,
) -> tuple[bool, list[str]]:
    if not source_binding_verified:
        return False, ["clean rerun requires a verified frozen source binding"]
    errors: list[str] = []
    horizon = snapshot.manifest["source_horizon"]
    with tempfile.TemporaryDirectory(prefix="dts-source-rerun-") as raw_temp:
        rerun_dir = Path(raw_temp) / "snapshot"
        shutil.copytree(
            snapshot.source_dir / "raw",
            rerun_dir / "raw",
            copy_function=os.link,
        )
        config = source.BuildConfig(
            start_date=str(horizon["start"]),
            end_date=str(horizon["end"]),
            output_dir=str(rerun_dir),
            max_workers=max_workers,
            request_pace_seconds=0,
        )
        source.build_source(config)
        for name in DETERMINISTIC_SOURCE_FILES:
            expected = source.sha256_file(snapshot.source_dir / name)
            actual = source.sha256_file(rerun_dir / name)
            if actual != expected:
                _append_error(errors, f"clean rerun hash mismatch: {name}")
    return not errors, errors


def _write_csv_artifact(
    path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]
) -> dict[str, Any]:
    payload = source.canonical_csv(rows, columns)
    source.write_gzip(path, payload)
    return {
        "path": path.name,
        "row_count": len(rows),
        "uncompressed_sha256": source.sha256_bytes(payload),
        "file_sha256": source.sha256_file(path),
    }


def run_audit(config: AuditConfig) -> dict[str, Any]:
    source_dir = Path(config.source_dir)
    output_dir = Path(config.output_dir) if config.output_dir else source_dir / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = load_snapshot(source_dir)

    binding_pass, binding_errors = verify_snapshot_binding(snapshot)
    numeric_rows, numeric_errors, numeric_checked = audit_numeric_roundtrip(snapshot)
    accounting_rows, accounting_errors = audit_accounting(snapshot)
    duplicate_rows, duplicate_errors = audit_duplicate_labels(snapshot)
    transition_rows, transition_errors = detect_schema_transitions(snapshot)
    announcement_rows, announcement_errors = reconcile_announcements(
        snapshot, transition_rows
    )
    if config.verify_clean_rerun and binding_pass:
        rerun_pass, rerun_errors = verify_clean_rerun(
            snapshot,
            max_workers=config.max_workers,
            source_binding_verified=True,
        )
    elif config.verify_clean_rerun:
        rerun_pass = False
        rerun_errors = ["clean rerun blocked by failed source binding"]
    else:
        rerun_pass = False
        rerun_errors = ["clean rerun was skipped"]

    artifacts = {
        "accounting_reconciliations": _write_csv_artifact(
            output_dir / "source_accounting_reconciliations.csv.gz",
            accounting_rows,
            ACCOUNTING_COLUMNS,
        ),
        "schema_transitions": _write_csv_artifact(
            output_dir / "source_schema_transitions.csv.gz",
            transition_rows,
            TRANSITION_COLUMNS,
        ),
        "announcement_reconciliation": _write_csv_artifact(
            output_dir / "source_announcement_reconciliation.csv.gz",
            announcement_rows,
            ANNOUNCEMENT_COLUMNS,
        ),
        "duplicate_label_audit": _write_csv_artifact(
            output_dir / "source_duplicate_label_audit.csv.gz",
            duplicate_rows,
            DUPLICATE_COLUMNS,
        ),
        "numeric_cell_roundtrip": _write_csv_artifact(
            output_dir / "source_numeric_cell_roundtrip.csv.gz",
            numeric_rows,
            NUMERIC_COLUMNS,
        ),
    }

    accepted_accounting_statuses = {"PASS", "PUBLISHED_SOURCE_ANOMALY"}
    accounting_pass = not accounting_errors and all(
        row["status"] in accepted_accounting_statuses for row in accounting_rows
    )
    schema_pass = not transition_errors and bool(transition_rows)
    announcement_pass = (
        not announcement_errors
        and all(row["status"] != "UNMATCHED" for row in announcement_rows)
        and all(
            str(row["status"]).startswith("RECONCILED") for row in transition_rows
        )
    )
    gates = {
        "coverage_hash_binding": {
            "pass": binding_pass,
            "failure_count": len(binding_errors),
        },
        "required_tables_and_causal_clock": {
            "pass": binding_pass,
            "failure_count": len(binding_errors),
        },
        "numeric_literal_roundtrip": {
            "pass": not numeric_errors,
            "failure_count": len(numeric_errors),
            "checked_cell_count": numeric_checked,
        },
        "table_totals_and_cash_identities": {
            "pass": accounting_pass,
            "failure_count": len(accounting_errors),
            "check_count": len(accounting_rows),
            "published_source_anomaly_count": sum(
                row["status"] == "PUBLISHED_SOURCE_ANOMALY"
                for row in accounting_rows
            ),
        },
        "duplicate_normalized_labels": {
            "pass": not duplicate_errors,
            "failure_count": len(duplicate_errors),
            "duplicate_group_count": len(duplicate_rows),
        },
        "schema_transition_detection": {
            "pass": schema_pass,
            "failure_count": len(transition_errors),
            "stable_transition_count": len(transition_rows),
        },
        "announcement_reconciliation": {
            "pass": announcement_pass,
            "failure_count": len(announcement_errors),
            "announcement_count": len(announcement_rows),
        },
        "clean_rerun_determinism": {
            "pass": rerun_pass,
            "failure_count": len(rerun_errors),
        },
        "physical_cap_and_source_only_protocol": {
            "pass": binding_pass,
            "failure_count": len(binding_errors),
        },
    }
    all_pass = all(bool(value["pass"]) for value in gates.values())
    all_errors = [
        *binding_errors,
        *numeric_errors,
        *accounting_errors,
        *duplicate_errors,
        *transition_errors,
        *announcement_errors,
        *rerun_errors,
    ]
    audit_source = Path(__file__)
    audit_manifest = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "audit_version": AUDIT_VERSION,
        "audit_source_sha256": source.sha256_file(audit_source),
        "input_source_manifest_sha256": source.sha256_file(
            source_dir / "source_manifest.json"
        ),
        "input_source_build_report_sha256": source.sha256_file(
            source_dir / "source_build_report.json"
        ),
        "stability_window_reports": STABILITY_WINDOW_REPORTS,
        "artifacts": artifacts,
    }
    audit_manifest_path = output_dir / "source_quality_audit_manifest.json"
    source._freeze_bytes(audit_manifest_path, source.canonical_json(audit_manifest))
    report = {
        "candidate_family": "DFFB",
        "decision": "SOURCE_QUALITY_PASS" if all_pass else "SOURCE_QUALITY_FAIL",
        "source_quality_gates_evaluated": True,
        "all_source_quality_gates_pass": all_pass,
        "next_stage_authorized": "SOURCE_ONLY_PREREGISTRATION" if all_pass else None,
        "failure_count": len(all_errors),
        "failures": all_errors[:200],
        "gates": gates,
        "input_source_manifest_sha256": audit_manifest[
            "input_source_manifest_sha256"
        ],
        "audit_manifest_sha256": source.sha256_file(audit_manifest_path),
        "protocol": {
            "source_only": True,
            "post_2023_report_opened": False,
            "post_2023_api_value_row_opened": False,
            "btc_market_data_opened": False,
            "funding_opened": False,
            "future_return_opened": False,
            "labels_opened": False,
            "pnl_cagr_mdd_opened": False,
        },
    }
    source._freeze_bytes(
        output_dir / "source_quality_audit_report.json", source.canonical_json(report)
    )
    return report


def _parse_args(argv: Sequence[str] | None = None) -> AuditConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", default="data/daily_treasury_fiscal_flow_2019_2023"
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--skip-clean-rerun", action="store_true")
    args = parser.parse_args(argv)
    if args.max_workers < 1:
        raise ValueError("max-workers must be positive")
    return AuditConfig(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        verify_clean_rerun=not args.skip_clean_rerun,
    )


def main(argv: Sequence[str] | None = None) -> int:
    report = run_audit(_parse_args(argv))
    print(source.canonical_json(report).decode("utf-8"), end="")
    return 0 if report["all_source_quality_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
