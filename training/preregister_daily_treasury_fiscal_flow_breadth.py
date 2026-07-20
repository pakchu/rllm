"""Freeze the outcome-blind DFFB-601 singleton mechanism.

This stage reads only source/audit metadata, hashes frozen source artifacts, and
reads artifact headers.  It deliberately does not read DTS value rows,
comparator clock rows, event incidence, market data, or strategy outcomes.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from dataclasses import asdict, dataclass, field
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Iterable


POLICY_ID = "DFFB-601"
PROTOCOL_VERSION = "daily_treasury_fiscal_flow_breadth_preregistration_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DECISION = Path(
    "docs/daily-treasury-fiscal-flow-breadth-source-axis-decision-2026-07-20.md"
)
SOURCE_DECISION_SHA256 = (
    "7ed2748506645aa4b4a9589d7c7d3dd1baa4456c78a7c7f56845358c02f18c5b"
)
SOURCE_BUILDER = Path("training/build_daily_treasury_fiscal_flow_source.py")
SOURCE_BUILDER_SHA256 = (
    "fe0f928e1578f71c3953074acb95e2ed4ec12369da13cb74b202fbbffa015e27"
)
SOURCE_AUDITOR = Path("training/audit_daily_treasury_fiscal_flow_source.py")
SOURCE_AUDITOR_SHA256 = (
    "1a80bb86884dfb32de0f5215a0c8e8d34175637a9c0b2428c85e11719e10b6fd"
)
PREREGISTRATION_SOURCE = Path(
    "training/preregister_daily_treasury_fiscal_flow_breadth.py"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/daily-treasury-fiscal-flow-breadth-preregistration-2026-07-21.md"
)
PREREGISTRATION_DOCUMENT_SHA256 = (
    "5b6fe09d9e27c01c084d1e86fbaddfcfcbadcde531623f466ac6d5796621decb"
)
DEFAULT_OUTPUT = Path(
    "results/daily_treasury_fiscal_flow_breadth_preregistration_2026-07-21.json"
)

SOURCE_ROOT = Path("data/daily_treasury_fiscal_flow_2019_2023")
SOURCE_MANIFEST = SOURCE_ROOT / "source_manifest.json"
SOURCE_MANIFEST_SHA256 = (
    "2f7273f7291cf0f2866a33a1e321f63b4bfd5ccd39fcc296f02f5f9bbc3ea307"
)
SOURCE_BUILD_REPORT = SOURCE_ROOT / "source_build_report.json"
SOURCE_BUILD_REPORT_SHA256 = (
    "80721d6c036c7ec8225b81a7cecbd80b3aa161a11151b81a3bc02fae3f6be7ef"
)
SOURCE_ROWS = SOURCE_ROOT / "daily_treasury_fiscal_flow_rows.csv.gz"
SOURCE_ROWS_SHA256 = "0d85511519b72cb5ee3d546936101750e109159955d65601a0418f4a3eb75e01"
OPERATING_CASH_ROWS = SOURCE_ROOT / "daily_treasury_operating_cash_rows.csv.gz"
OPERATING_CASH_ROWS_SHA256 = (
    "e6c3dcee34ae0a6cfe7997eca7442986f90cbbf91176676657b1d74a1488fe85"
)
SOURCE_ANNOUNCEMENTS = SOURCE_ROOT / "precap_schema_announcements.csv.gz"
SOURCE_ANNOUNCEMENTS_SHA256 = (
    "489d90299616126f3d01425c0964f7c633454413f799ae1fce1f55dbf3311a40"
)
AUDIT_ROOT = SOURCE_ROOT / "audit"
AUDIT_MANIFEST = AUDIT_ROOT / "source_quality_audit_manifest.json"
AUDIT_MANIFEST_SHA256 = (
    "465e80b82ffcc5d56186e8769febd12ee27bde0ed847484d9073577a304a49de"
)
AUDIT_REPORT = AUDIT_ROOT / "source_quality_audit_report.json"
AUDIT_REPORT_SHA256 = "d27fe30282d869cec3e223047556fd4b851c8d471cb677e3b7b342e5e9a028eb"
SCHEMA_TRANSITIONS = AUDIT_ROOT / "source_schema_transitions.csv.gz"
SCHEMA_TRANSITIONS_SHA256 = (
    "f9115b79ae41cc70349f889b5b40ff071ef1a71194e1da25114bf255fed82635"
)

FLCC_PREREGISTRATION = Path(
    "results/federal_liquidity_component_concordance_preregistration_2026-07-17.json"
)
FLCC_PREREGISTRATION_SHA256 = (
    "3410443f009e63d4068d4d44c42e148799fba515b1934b01ecc38d7208ac54ee"
)
FLCC_CLOCK = Path(
    "results/federal_liquidity_component_concordance_preregistered_clock_2026-07-17.csv.gz"
)
FLCC_CLOCK_SHA256 = "7ebb0450422d9265e46c596e0b6415b6a8816c66f5e0cbb9ccda14ca6cb4c67c"
FLCC_CLOCK_SOURCE = Path("training/federal_liquidity_component_concordance_clock.py")
FLCC_CLOCK_SOURCE_SHA256 = (
    "0fd6473135a6be712fa50c358d8b689c53bc0de1038a33955881045cfe01bf1d"
)
TADI_PREREGISTRATION = Path(
    "results/treasury_auction_demand_impulse_preregistration_2026-07-17.json"
)
TADI_PREREGISTRATION_SHA256 = (
    "b2c2260dadc4236b016fe2cf5bfc503afd4f8cd0d2c1161f4990f21f07f9568a"
)
TADI_CLOCK = Path(
    "results/treasury_auction_demand_impulse_preregistered_clock_2026-07-17.csv.gz"
)
TADI_CLOCK_SHA256 = "9bb416413a0cfee5a5ebbdb73032e5889735e88098eaa1dc264b6d224fa489f6"
TADI_CLOCK_SOURCE = Path("training/treasury_auction_demand_impulse_clock.py")
TADI_CLOCK_SOURCE_SHA256 = (
    "7d528ccb34a55f92eec34c7dfff3b3e4f0e989b033b486019a1507221c1f8329"
)

AUCTION_ROOT = Path("data/us_treasury_auction_demand_2016_2023")
AUCTION_MANIFEST = AUCTION_ROOT / "build_manifest.json"
AUCTION_MANIFEST_SHA256 = (
    "6da6a3848e89c3418efcbf0d836fda34b537a2da87a8777b74670f3912ad94f2"
)
AUCTION_PANEL = AUCTION_ROOT / "us_treasury_nominal_original_auctions_2016_2023.csv.gz"
AUCTION_PANEL_SHA256 = (
    "34a19163630c015a4f9d2671c95ca7cf7cc8a8ada024b3ef985405704fe0e4c1"
)
AUCTION_RAW_PAGE_0 = AUCTION_ROOT / "raw/auction_query_page_0.json.gz"
AUCTION_RAW_PAGE_0_SHA256 = (
    "6e609bdf4e6e859d3d957c638244070e999c343acf7793dc5e9b32988915564b"
)
AUCTION_RAW_PAGE_1 = AUCTION_ROOT / "raw/auction_query_page_1.json.gz"
AUCTION_RAW_PAGE_1_SHA256 = (
    "b20370eacc2c6f030483e49d7e6cf6db6d4dbfa89c8142a3b3e0b5540840d221"
)

SOURCE_ROWS_HEADER = [
    "record_date",
    "source_available_not_before_utc",
    "earliest_execution_time_utc",
    "research_stage",
    "table_id",
    "side",
    "parent_section",
    "raw_category_label",
    "normalized_category_label",
    "today_amount_usd_millions",
    "today_amount_literal",
    "month_to_date_amount_usd_millions",
    "month_to_date_amount_literal",
    "fiscal_year_to_date_amount_usd_millions",
    "fiscal_year_to_date_amount_literal",
    "footnote_markers",
    "missing_value_tokens",
    "row_kind",
    "page_number",
    "source_order",
    "source_pdf_sha256",
]
OPERATING_CASH_HEADER = [
    "record_date",
    "source_available_not_before_utc",
    "earliest_execution_time_utc",
    "research_stage",
    "raw_category_label",
    "normalized_category_label",
    "published_value_count",
    "published_values_usd_millions_json",
    "published_value_literals_json",
    "missing_value_tokens_json",
    "footnote_markers",
    "schema_variant",
    "page_number",
    "source_order",
    "source_pdf_sha256",
]
SCHEMA_TRANSITIONS_HEADER = [
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
]
FLCC_CLOCK_HEADER = [
    "candidate_id",
    "clock_name",
    "feature_release_date",
    "signal_release_date",
    "signal_time",
    "entry_time",
    "exit_time",
    "side",
    "horizon_releases",
    "lower_rank_numerator",
    "upper_rank_numerator",
    "prior_lookback",
    "net_rank_numerator",
    "asset_rank_numerator",
    "tga_release_rank_numerator",
    "rrp_release_rank_numerator",
    "component_breadth",
    "component_tail_breadth",
]
TADI_CLOCK_HEADER = [
    "auction_date",
    "decision_time",
    "entry_time",
    "scheduled_exit_time",
    "original_security_term",
    "cusip",
    "side",
    "clock_mode",
    "bid_to_cover_delta",
    "indirect_share_delta",
    "bid_to_cover_delta_rank",
    "indirect_share_delta_rank",
]
AUCTION_PANEL_HEADER = [
    "auction_date",
    "result_available_at_utc",
    "security_type",
    "original_security_term",
    "cusip",
    "bid_to_cover_ratio",
    "competitive_accepted_usd",
    "primary_dealer_accepted_usd",
    "direct_bidder_accepted_usd",
    "indirect_bidder_accepted_usd",
    "indirect_competitive_share",
    "closing_time_competitive_et",
    "updated_timestamp_et",
    "competitive_results_pdf_url",
    "competitive_results_xml_url",
    "source_complete",
]

FLCC_ALLOWED_COLUMNS = [
    "candidate_id",
    "clock_name",
    "signal_time",
    "entry_time",
    "exit_time",
    "side",
]
FLCC_PREREGISTRATION_TOP_LEVEL_KEYS = {
    "as_of_date",
    "candidate_class",
    "execution_contract",
    "execution_policy",
    "falsification_controls",
    "family_id",
    "feature_contract",
    "forbidden_repairs_after_outcomes",
    "manifest_hash",
    "novelty_boundary",
    "outcomes_opened",
    "protocol_version",
    "research_history_boundary",
    "sealed",
    "selection_protocol",
    "source_commit",
    "source_contract",
    "source_only_support",
    "support_gates",
    "support_passed",
}
TADI_ALLOWED_COLUMNS = [
    "auction_date",
    "decision_time",
    "entry_time",
    "scheduled_exit_time",
    "side",
    "clock_mode",
]
TADI_PREREGISTRATION_TOP_LEVEL_KEYS = {
    "as_of_date",
    "causal_feature_contract",
    "execution_contract",
    "falsification_controls",
    "manifest_hash",
    "novelty_boundary",
    "orthogonality_after_standalone_pass",
    "outcomes_opened",
    "policy",
    "protocol_version",
    "research_history_boundary",
    "selection_protocol",
    "source_contract",
    "support_gates",
}
AUCTION_PANEL_ALLOWED_COLUMNS = ["auction_date", "cusip"]
AUCTION_RAW_ALLOWED_FIELDS = [
    "auctionDate",
    "issueDate",
    "cusip",
    "securityType",
    "originalSecurityTerm",
    "reopening",
]
PROHIBITED_COMPARATOR_COLUMN_TOKENS = [
    "return",
    "pnl",
    "profit",
    "equity",
    "cagr",
    "mdd",
    "drawdown",
    "sharpe",
    "sortino",
    "price",
    "funding",
    "outcome",
]

EXPECTED_OUTCOME_BOUNDARY = {
    "source_manifest_json_read": True,
    "source_build_report_json_read": True,
    "source_audit_manifest_json_read": True,
    "source_audit_report_json_read": True,
    "source_artifact_bytes_hashed": True,
    "source_artifact_headers_read": True,
    "source_value_rows_read": 0,
    "schema_transition_artifact_bytes_hashed": True,
    "schema_transition_header_read": True,
    "schema_transition_rows_read": 0,
    "comparator_preregistration_json_read": True,
    "comparator_clock_bytes_hashed": True,
    "comparator_clock_headers_read": True,
    "comparator_clock_rows_read": 0,
    "auction_source_artifact_bytes_hashed": True,
    "auction_source_rows_read": 0,
    "source_features_derived": 0,
    "signal_incidence_rows_derived": 0,
    "market_rows_loaded": 0,
    "funding_rows_loaded": 0,
    "return_or_pnl_fields_read": 0,
    "network_calls": 0,
    "database_calls": 0,
    "subprocess_calls": 0,
}


@dataclass(frozen=True)
class Config:
    preregistration_output: str = str(DEFAULT_OUTPUT)


@dataclass
class AccessLedger:
    """Record every allowed preregistration input read deterministically."""

    events: list[dict[str, str]] = field(default_factory=list)

    def record(self, path: str | Path, operation: str) -> None:
        resolved = _repository_path(path)
        allowed = _access_allowlist().get(resolved)
        if allowed is None or operation not in allowed:
            raise RuntimeError(
                f"DFFB input access is not allowlisted: {path} ({operation})"
            )
        self.events.append(
            {"path": _display_repository_path(resolved), "operation": operation}
        )

    def validate_complete(self) -> None:
        expected = Counter(
            (_display_repository_path(path), operation)
            for path, operations in _access_allowlist().items()
            for operation in operations
        )
        observed = Counter((event["path"], event["operation"]) for event in self.events)
        if observed != expected:
            raise RuntimeError("DFFB preregistration access ledger is incomplete")

    def outcome_boundary(self) -> dict[str, Any]:
        operations = {(event["path"], event["operation"]) for event in self.events}

        def seen(path: Path, operation: str) -> bool:
            return (
                _display_repository_path(_repository_path(path)),
                operation,
            ) in operations

        boundary = {
            "source_manifest_json_read": seen(SOURCE_MANIFEST, "json_metadata"),
            "source_build_report_json_read": seen(SOURCE_BUILD_REPORT, "json_metadata"),
            "source_audit_manifest_json_read": seen(AUDIT_MANIFEST, "json_metadata"),
            "source_audit_report_json_read": seen(AUDIT_REPORT, "json_metadata"),
            "source_artifact_bytes_hashed": all(
                seen(path, "full_byte_hash")
                for path in (SOURCE_ROWS, OPERATING_CASH_ROWS, SOURCE_ANNOUNCEMENTS)
            ),
            "source_artifact_headers_read": all(
                seen(path, "header_only") for path in (SOURCE_ROWS, OPERATING_CASH_ROWS)
            ),
            "source_value_rows_read": sum(
                event["operation"] == "value_rows" for event in self.events
            ),
            "schema_transition_artifact_bytes_hashed": seen(
                SCHEMA_TRANSITIONS, "full_byte_hash"
            ),
            "schema_transition_header_read": seen(SCHEMA_TRANSITIONS, "header_only"),
            "schema_transition_rows_read": sum(
                event["operation"] == "schema_transition_rows" for event in self.events
            ),
            "comparator_preregistration_json_read": all(
                seen(path, "json_metadata")
                for path in (FLCC_PREREGISTRATION, TADI_PREREGISTRATION)
            ),
            "comparator_clock_bytes_hashed": all(
                seen(path, "full_byte_hash") for path in (FLCC_CLOCK, TADI_CLOCK)
            ),
            "comparator_clock_headers_read": all(
                seen(path, "header_only") for path in (FLCC_CLOCK, TADI_CLOCK)
            ),
            "comparator_clock_rows_read": sum(
                event["operation"] == "comparator_rows" for event in self.events
            ),
            "auction_source_artifact_bytes_hashed": all(
                seen(path, "full_byte_hash")
                for path in (AUCTION_PANEL, AUCTION_RAW_PAGE_0, AUCTION_RAW_PAGE_1)
            ),
            "auction_source_rows_read": sum(
                event["operation"] == "auction_rows" for event in self.events
            ),
            "source_features_derived": 0,
            "signal_incidence_rows_derived": 0,
            "market_rows_loaded": 0,
            "funding_rows_loaded": 0,
            "return_or_pnl_fields_read": 0,
            "network_calls": 0,
            "database_calls": 0,
            "subprocess_calls": 0,
        }
        return boundary


_ACTIVE_ACCESS_LEDGER: AccessLedger | None = None


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return candidate.resolve()


def _display_repository_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def _access_allowlist() -> dict[Path, set[str]]:
    hashes_only = (
        SOURCE_DECISION,
        SOURCE_BUILDER,
        SOURCE_AUDITOR,
        PREREGISTRATION_DOCUMENT,
        SOURCE_ANNOUNCEMENTS,
        FLCC_CLOCK_SOURCE,
        TADI_CLOCK_SOURCE,
        AUCTION_RAW_PAGE_0,
        AUCTION_RAW_PAGE_1,
    )
    json_metadata = (
        SOURCE_MANIFEST,
        SOURCE_BUILD_REPORT,
        AUDIT_MANIFEST,
        AUDIT_REPORT,
        FLCC_PREREGISTRATION,
        TADI_PREREGISTRATION,
        AUCTION_MANIFEST,
    )
    header_only = (
        SOURCE_ROWS,
        OPERATING_CASH_ROWS,
        SCHEMA_TRANSITIONS,
        FLCC_CLOCK,
        TADI_CLOCK,
        AUCTION_PANEL,
    )
    allowed = {_repository_path(path): {"full_byte_hash"} for path in hashes_only}
    allowed.update(
        {
            _repository_path(path): {"full_byte_hash", "json_metadata"}
            for path in json_metadata
        }
    )
    allowed.update(
        {
            _repository_path(path): {"full_byte_hash", "header_only"}
            for path in header_only
        }
    )
    allowed[_repository_path(PREREGISTRATION_SOURCE)] = {
        "full_byte_hash",
        "python_ast",
    }
    return allowed


def _record_access(path: str | Path, operation: str) -> None:
    if _ACTIVE_ACCESS_LEDGER is not None:
        _ACTIVE_ACCESS_LEDGER.record(path, operation)


def _require_regular_file(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    if candidate.is_symlink():
        raise RuntimeError(f"DFFB bound input is a symlink: {path}")
    info = candidate.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"DFFB bound input is not a physical regular file: {path}")
    return candidate.resolve()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    physical_path = _require_regular_file(path)
    _record_access(physical_path, "full_byte_hash")
    with physical_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unique_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"DFFB JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _read_json(path: str | Path) -> dict[str, Any]:
    physical_path = _require_regular_file(path)
    _record_access(physical_path, "json_metadata")
    payload = json.loads(
        physical_path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
    )
    if not isinstance(payload, dict):
        raise RuntimeError(f"DFFB JSON input must be an object: {path}")
    return payload


def _manifest_core(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "manifest_hash"}


def _read_gzip_header(path: str | Path) -> list[str]:
    physical_path = _require_regular_file(path)
    _record_access(physical_path, "header_only")
    with gzip.open(physical_path, "rt", encoding="utf-8", newline="") as handle:
        line = handle.readline()
    if not line:
        raise RuntimeError(f"DFFB bound CSV is empty: {path}")
    return next(csv.reader([line]))


def _require_hash(path: str | Path, expected: str, label: str) -> None:
    if sha256_file(path) != expected:
        raise RuntimeError(f"DFFB {label} SHA drift")


def _require_header(path: str | Path, expected: list[str], label: str) -> None:
    if _read_gzip_header(path) != expected:
        raise RuntimeError(f"DFFB {label} header drift")


def _require_exact(
    payload: dict[str, Any], key: str, expected: Any, label: str
) -> None:
    if payload.get(key) != expected:
        raise RuntimeError(f"DFFB {label} {key} drift")


def _validate_preregistration_import_boundary() -> None:
    physical_path = _require_regular_file(PREREGISTRATION_SOURCE)
    _record_access(physical_path, "python_ast")
    tree = ast.parse(physical_path.read_text(encoding="utf-8"))
    forbidden_import_roots = {
        "aiohttp",
        "httpx",
        "psycopg",
        "psycopg2",
        "pymysql",
        "requests",
        "socket",
        "sqlalchemy",
        "sqlite3",
        "subprocess",
        "urllib",
    }
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"__import__", "eval", "exec"}
        ):
            raise RuntimeError("DFFB preregistration contains dynamic code loading")
    prohibited = sorted(imported_roots & forbidden_import_roots)
    if prohibited:
        raise RuntimeError(
            f"DFFB preregistration imports prohibited clients: {prohibited}"
        )


def _validate_source_binding() -> dict[str, Any]:
    for path, expected, label in (
        (SOURCE_DECISION, SOURCE_DECISION_SHA256, "source decision"),
        (SOURCE_BUILDER, SOURCE_BUILDER_SHA256, "source builder"),
        (SOURCE_AUDITOR, SOURCE_AUDITOR_SHA256, "source auditor"),
        (SOURCE_MANIFEST, SOURCE_MANIFEST_SHA256, "source manifest"),
        (SOURCE_BUILD_REPORT, SOURCE_BUILD_REPORT_SHA256, "source build report"),
        (SOURCE_ROWS, SOURCE_ROWS_SHA256, "normalized source rows"),
        (OPERATING_CASH_ROWS, OPERATING_CASH_ROWS_SHA256, "operating-cash rows"),
        (SOURCE_ANNOUNCEMENTS, SOURCE_ANNOUNCEMENTS_SHA256, "announcements"),
        (AUDIT_MANIFEST, AUDIT_MANIFEST_SHA256, "audit manifest"),
        (AUDIT_REPORT, AUDIT_REPORT_SHA256, "audit report"),
        (SCHEMA_TRANSITIONS, SCHEMA_TRANSITIONS_SHA256, "schema transitions"),
    ):
        _require_hash(path, expected, label)

    _require_header(SOURCE_ROWS, SOURCE_ROWS_HEADER, "normalized source rows")
    _require_header(OPERATING_CASH_ROWS, OPERATING_CASH_HEADER, "operating-cash rows")
    _require_header(SCHEMA_TRANSITIONS, SCHEMA_TRANSITIONS_HEADER, "schema transitions")

    source_manifest = _read_json(SOURCE_MANIFEST)
    _require_exact(source_manifest, "parser_version", 3, "source manifest")
    _require_exact(
        source_manifest,
        "normalized_rows",
        {
            "file_sha256": SOURCE_ROWS_SHA256,
            "path": SOURCE_ROWS.name,
            "row_count": 205_589,
            "uncompressed_sha256": (
                "58a90f420147dcae5069c93cf6e048b3072e4852ad44a0ffe55a3b1d15e1f6a8"
            ),
        },
        "source manifest",
    )
    _require_exact(
        source_manifest,
        "operating_cash_rows",
        {
            "file_sha256": OPERATING_CASH_ROWS_SHA256,
            "path": OPERATING_CASH_ROWS.name,
            "row_count": 5_024,
            "uncompressed_sha256": (
                "c65b4715b7af2415f30481cec3a3c796d5ea4f3e0554d6b4cbe8866e354029a3"
            ),
        },
        "source manifest",
    )
    reports = source_manifest.get("reports")
    if not isinstance(reports, list) or len(reports) != 1_256:
        raise RuntimeError("DFFB source manifest report inventory drift")
    if reports[0].get("record_date") != "2019-01-02":
        raise RuntimeError("DFFB source manifest first report drift")
    if reports[-1].get("record_date") != "2023-12-29":
        raise RuntimeError("DFFB source manifest last report drift")

    build_report = _read_json(SOURCE_BUILD_REPORT)
    expected_protocol = {
        "btc_market_data_opened": False,
        "current_metadata_postcap_rows_used_in_logic": False,
        "funding_opened": False,
        "future_return_opened": False,
        "labels_opened": False,
        "pnl_cagr_mdd_opened": False,
        "post_2023_api_value_row_opened": False,
        "post_2023_report_opened": False,
        "source_only": True,
    }
    for key, expected in (
        ("candidate_family", "DFFB"),
        ("decision", "SOURCE_BUILT_REQUIRES_SCHEMA_AUDIT"),
        ("source_manifest_sha256", SOURCE_MANIFEST_SHA256),
        ("report_count", 1_256),
        ("normalized_row_count", 205_589),
        ("operating_cash_row_count", 5_024),
        (
            "stage_report_counts",
            {"boundary_quarantine": 1, "selection": 250, "train": 502, "warmup": 503},
        ),
        ("unexplained_weekday_gaps", []),
        ("protocol", expected_protocol),
    ):
        _require_exact(build_report, key, expected, "source build report")

    audit_manifest = _read_json(AUDIT_MANIFEST)
    for key, expected in (
        ("audit_schema_version", 1),
        ("audit_version", 1),
        ("audit_source_sha256", SOURCE_AUDITOR_SHA256),
        ("input_source_build_report_sha256", SOURCE_BUILD_REPORT_SHA256),
        ("input_source_manifest_sha256", SOURCE_MANIFEST_SHA256),
        ("stability_window_reports", 5),
    ):
        _require_exact(audit_manifest, key, expected, "audit manifest")
    transitions = audit_manifest.get("artifacts", {}).get("schema_transitions")
    if transitions != {
        "file_sha256": SCHEMA_TRANSITIONS_SHA256,
        "path": SCHEMA_TRANSITIONS.name,
        "row_count": 497,
        "uncompressed_sha256": (
            "ac026636e8826a57a7c644d0fecb0094e380fd216b67cad7235e5b22eff91f3c"
        ),
    }:
        raise RuntimeError("DFFB audit manifest schema_transitions drift")

    audit_report = _read_json(AUDIT_REPORT)
    for key, expected in (
        ("candidate_family", "DFFB"),
        ("decision", "SOURCE_QUALITY_PASS"),
        ("all_source_quality_gates_pass", True),
        ("source_quality_gates_evaluated", True),
        ("failure_count", 0),
        ("failures", []),
        ("audit_manifest_sha256", AUDIT_MANIFEST_SHA256),
        ("input_source_manifest_sha256", SOURCE_MANIFEST_SHA256),
        ("next_stage_authorized", "SOURCE_ONLY_PREREGISTRATION"),
    ):
        _require_exact(audit_report, key, expected, "audit report")
    expected_gates = {
        "announcement_reconciliation",
        "clean_rerun_determinism",
        "coverage_hash_binding",
        "duplicate_normalized_labels",
        "numeric_literal_roundtrip",
        "physical_cap_and_source_only_protocol",
        "required_tables_and_causal_clock",
        "schema_transition_detection",
        "table_totals_and_cash_identities",
    }
    gates = audit_report.get("gates")
    if not isinstance(gates, dict) or set(gates) != expected_gates:
        raise RuntimeError("DFFB audit report gate inventory drift")
    if not all(
        isinstance(gate, dict)
        and gate.get("pass") is True
        and gate.get("failure_count") == 0
        for gate in gates.values()
    ):
        raise RuntimeError("DFFB audit report gate drift")
    audit_protocol = audit_report.get("protocol")
    if (
        not isinstance(audit_protocol, dict)
        or audit_protocol.get("source_only") is not True
    ):
        raise RuntimeError("DFFB audit report source-only protocol drift")
    if any(
        value is not False
        for key, value in audit_protocol.items()
        if key != "source_only"
    ):
        raise RuntimeError("DFFB audit report opened a prohibited source")

    return {
        "source_decision": {
            "path": str(SOURCE_DECISION),
            "sha256": SOURCE_DECISION_SHA256,
        },
        "source_builder": {
            "path": str(SOURCE_BUILDER),
            "sha256": SOURCE_BUILDER_SHA256,
        },
        "source_auditor": {
            "path": str(SOURCE_AUDITOR),
            "sha256": SOURCE_AUDITOR_SHA256,
        },
        "source_manifest": {
            "path": str(SOURCE_MANIFEST),
            "sha256": SOURCE_MANIFEST_SHA256,
        },
        "source_build_report": {
            "path": str(SOURCE_BUILD_REPORT),
            "sha256": SOURCE_BUILD_REPORT_SHA256,
        },
        "source_rows": {
            "path": str(SOURCE_ROWS),
            "sha256": SOURCE_ROWS_SHA256,
            "header": SOURCE_ROWS_HEADER,
        },
        "operating_cash_rows": {
            "path": str(OPERATING_CASH_ROWS),
            "sha256": OPERATING_CASH_ROWS_SHA256,
            "header": OPERATING_CASH_HEADER,
        },
        "announcements": {
            "path": str(SOURCE_ANNOUNCEMENTS),
            "sha256": SOURCE_ANNOUNCEMENTS_SHA256,
        },
        "audit_manifest": {
            "path": str(AUDIT_MANIFEST),
            "sha256": AUDIT_MANIFEST_SHA256,
        },
        "audit_report": {"path": str(AUDIT_REPORT), "sha256": AUDIT_REPORT_SHA256},
        "schema_transitions": {
            "path": str(SCHEMA_TRANSITIONS),
            "sha256": SCHEMA_TRANSITIONS_SHA256,
            "header": SCHEMA_TRANSITIONS_HEADER,
            "feature_use": "audit evidence only; rows unopened in preregistration and prohibited from retrospective feature identity",
        },
    }


def _validate_preregistered_comparator(
    *,
    path: Path,
    expected_sha: str,
    expected_protocol: str,
    expected_top_level_keys: set[str],
    label: str,
) -> dict[str, Any]:
    _require_hash(path, expected_sha, f"{label} preregistration")
    payload = _read_json(path)
    if canonical_hash(_manifest_core(payload)) != payload.get("manifest_hash"):
        raise RuntimeError(f"DFFB {label} preregistration canonical hash mismatch")
    if payload.get("protocol_version") != expected_protocol:
        raise RuntimeError(f"DFFB {label} protocol drift")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError(f"DFFB {label} comparator opened outcomes")
    if set(payload) != expected_top_level_keys:
        raise RuntimeError(f"DFFB {label} preregistration schema drift")
    return {
        "path": str(path),
        "sha256": expected_sha,
        "protocol_version": expected_protocol,
        "manifest_hash": payload["manifest_hash"],
        "top_level_keys": sorted(expected_top_level_keys),
        "outcomes_opened": False,
    }


def _validate_allowlist(header: list[str], allowed: list[str], label: str) -> None:
    if not set(allowed).issubset(header):
        raise RuntimeError(f"DFFB {label} allowlisted column missing")
    normalized = [re.sub(r"[^a-z0-9]+", "", column.casefold()) for column in allowed]
    for column in normalized:
        if any(token in column for token in PROHIBITED_COMPARATOR_COLUMN_TOKENS):
            raise RuntimeError(f"DFFB {label} allowlist includes outcome column")


def _validate_comparator_binding() -> dict[str, Any]:
    flcc_prereg = _validate_preregistered_comparator(
        path=FLCC_PREREGISTRATION,
        expected_sha=FLCC_PREREGISTRATION_SHA256,
        expected_protocol="federal_liquidity_component_concordance_v1",
        expected_top_level_keys=FLCC_PREREGISTRATION_TOP_LEVEL_KEYS,
        label="FLCC",
    )
    tadi_prereg = _validate_preregistered_comparator(
        path=TADI_PREREGISTRATION,
        expected_sha=TADI_PREREGISTRATION_SHA256,
        expected_protocol="treasury_auction_demand_impulse_v1",
        expected_top_level_keys=TADI_PREREGISTRATION_TOP_LEVEL_KEYS,
        label="TADI",
    )
    for path, expected, label in (
        (FLCC_CLOCK, FLCC_CLOCK_SHA256, "FLCC clock"),
        (FLCC_CLOCK_SOURCE, FLCC_CLOCK_SOURCE_SHA256, "FLCC clock source"),
        (TADI_CLOCK, TADI_CLOCK_SHA256, "TADI clock"),
        (TADI_CLOCK_SOURCE, TADI_CLOCK_SOURCE_SHA256, "TADI clock source"),
        (AUCTION_MANIFEST, AUCTION_MANIFEST_SHA256, "auction manifest"),
        (AUCTION_PANEL, AUCTION_PANEL_SHA256, "auction panel"),
        (AUCTION_RAW_PAGE_0, AUCTION_RAW_PAGE_0_SHA256, "auction raw page 0"),
        (AUCTION_RAW_PAGE_1, AUCTION_RAW_PAGE_1_SHA256, "auction raw page 1"),
    ):
        _require_hash(path, expected, label)
    _require_header(FLCC_CLOCK, FLCC_CLOCK_HEADER, "FLCC clock")
    _require_header(TADI_CLOCK, TADI_CLOCK_HEADER, "TADI clock")
    _require_header(AUCTION_PANEL, AUCTION_PANEL_HEADER, "auction panel")
    _validate_allowlist(FLCC_CLOCK_HEADER, FLCC_ALLOWED_COLUMNS, "FLCC clock")
    _validate_allowlist(TADI_CLOCK_HEADER, TADI_ALLOWED_COLUMNS, "TADI clock")
    _validate_allowlist(
        AUCTION_PANEL_HEADER, AUCTION_PANEL_ALLOWED_COLUMNS, "auction panel"
    )

    auction_manifest = _read_json(AUCTION_MANIFEST)
    for key, expected in (
        ("schema_version", 1),
        ("output", str(AUCTION_PANEL)),
        ("output_sha256", AUCTION_PANEL_SHA256),
    ):
        _require_exact(auction_manifest, key, expected, "auction manifest")
    protocol = auction_manifest.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("outcomes_opened") is not False:
        raise RuntimeError("DFFB auction manifest outcome boundary drift")
    expected_raw = {AUCTION_RAW_PAGE_0_SHA256, AUCTION_RAW_PAGE_1_SHA256}
    actual_raw = {
        source.get("raw_gzip_sha256")
        for source in auction_manifest.get("sources", [])
        if isinstance(source, dict)
    }
    if actual_raw != expected_raw:
        raise RuntimeError("DFFB auction manifest raw-page binding drift")

    return {
        "flcc": {
            "preregistration": flcc_prereg,
            "clock": {
                "path": str(FLCC_CLOCK),
                "sha256": FLCC_CLOCK_SHA256,
                "header": FLCC_CLOCK_HEADER,
                "allowed_columns": FLCC_ALLOWED_COLUMNS,
                "materialized_columns_must_equal_allowlist": True,
                "row_filter": 'clock_name == "primary"',
            },
            "clock_source": {
                "path": str(FLCC_CLOCK_SOURCE),
                "sha256": FLCC_CLOCK_SOURCE_SHA256,
            },
        },
        "tadi": {
            "preregistration": tadi_prereg,
            "clock": {
                "path": str(TADI_CLOCK),
                "sha256": TADI_CLOCK_SHA256,
                "header": TADI_CLOCK_HEADER,
                "allowed_columns": TADI_ALLOWED_COLUMNS,
                "materialized_columns_must_equal_allowlist": True,
                "row_filter": 'clock_mode == "primary"',
            },
            "clock_source": {
                "path": str(TADI_CLOCK_SOURCE),
                "sha256": TADI_CLOCK_SOURCE_SHA256,
            },
        },
        "official_auction_settlement_calendar": {
            "manifest": {
                "path": str(AUCTION_MANIFEST),
                "sha256": AUCTION_MANIFEST_SHA256,
            },
            "normalized_panel": {
                "path": str(AUCTION_PANEL),
                "sha256": AUCTION_PANEL_SHA256,
                "header": AUCTION_PANEL_HEADER,
                "allowed_columns": AUCTION_PANEL_ALLOWED_COLUMNS,
                "materialized_columns_must_equal_allowlist": True,
            },
            "raw_pages": [
                {"path": str(AUCTION_RAW_PAGE_0), "sha256": AUCTION_RAW_PAGE_0_SHA256},
                {"path": str(AUCTION_RAW_PAGE_1), "sha256": AUCTION_RAW_PAGE_1_SHA256},
            ],
            "raw_allowed_fields": AUCTION_RAW_ALLOWED_FIELDS,
        },
        "prohibited_column_tokens": PROHIBITED_COMPARATOR_COLUMN_TOKENS,
    }


def policy() -> dict[str, Any]:
    return {
        "policy_id": POLICY_ID,
        "singleton": True,
        "hypothesis": "concordant upper-tail withdrawal/redemption breadth is a fiscal-liquidity injection (long); concordant lower-tail breadth is a drain (short)",
        "source_universe": {
            "value_field": "today_amount_usd_millions",
            "signed_value": True,
            "source_sign_semantics": "preserve the signed integer printed in the PDF; the parser applies no cash-flow sign convention",
            "side_orientation": "positive withdrawal/redemption is injection activity; positive deposit/issue is drain activity; a negative amount is a correction/reversal on its named side",
            "sign_transform": "none; no absolute value and no side-dependent multiplication",
            "allowed_table_sides": [
                ["II", "deposit"],
                ["II", "withdrawal"],
                ["IIIA", "issue"],
                ["IIIA", "redemption"],
            ],
            "required_row_kind": "detail",
            "prohibited_value_fields": [
                "month_to_date_amount_usd_millions",
                "fiscal_year_to_date_amount_usd_millions",
            ],
            "exclusion_key": "NFC; Unicode dashes to '-'; casefold; collapse whitespace; trim",
            "category_key": [
                "table_id",
                "side",
                "canonical_parent_section",
                "canonical_category_label",
            ],
            "canonical_identity": "apply exclusion_key to parent_section and normalized_category_label, then remove every character other than ASCII a-z and 0-9; raw label is not identity",
            "same_report_identity_collision": "fail support build",
            "excluded_prefixes": [
                "total ",
                "sub-total ",
                "subtotal ",
                "net change",
                "change in balance",
            ],
            "excluded_exact_labels": [
                "treasury general account total deposits",
                "treasury general account total withdrawals",
                "sub-total deposits",
                "sub-total withdrawals",
                "transfers from depositaries",
                "transfers to depositaries",
                "transfers from federal reserve account (table v)",
                "transfers to federal reserve account (table v)",
                "transfers from tga (table v)",
                "transfers to tga (table v)",
            ],
            "excluded_table_ii_bridge_prefixes": [
                "public debt cash issues",
                "public debt cash redemp",
            ],
            "exclusion_fields": ["raw_category_label", "normalized_category_label"],
            "exclusion_comparison": "apply exclusion_key to both exclusion fields before prefix or exact comparison",
            "schema_transition_feature_use": "none; punctuation/case equivalence is causal normalization, while substantive rename/parent/table/side changes reset identity",
        },
        "missingness": {
            "birth": "first causal report printing the category identity",
            "first_appearance_current": "not prior-known and not rankable",
            "absent_after_birth": 0,
            "printed_null": "non-computable; never coerce to zero",
            "death_handling": "no future-detected death; trailing print-frequency eligibility removes stale identities causally",
        },
        "category_rank": {
            "method": "strict-prior empirical midrank of signed integer amount",
            "prior_report_dates": 60,
            "prior_window": "exact preceding 60 DTS reports after causal category birth",
            "minimum_prior_non_null_prints": 12,
            "midrank_formula": "(count(prior<current)+0.5*count(prior==current))/60",
            "current_excluded": True,
            "tie_equality": "exact integer equality",
            "tail_threshold": 0.75,
            "null_in_current_or_window": "category non-computable",
        },
        "breadth": {
            "side_denominator": "all history-eligible categories on the side",
            "fail_closed": "empty denominator or any denominator member non-computable makes side non-computable",
            "side_breadth": "mean(category_rank60>=0.75)",
            "cash_impulse": "withdrawal_breadth-deposit_breadth",
            "debt_impulse": "redemption_breadth-issue_breadth",
            "total_or_amount_weighting": "prohibited",
        },
        "impulse_rank": {
            "method": "strict-prior empirical midrank",
            "prior_computable_impulses": 126,
            "midrank_formula": "(count(prior<current)+0.5*count(prior==current))/126",
            "current_excluded": True,
            "noncomputable_reports_skipped_from_history": True,
        },
        "event": {
            "long": "cash_rank126>=0.75 and debt_rank126>=0.75",
            "short": "cash_rank126<=0.25 and debt_rank126<=0.25",
            "otherwise": "none",
            "threshold_search": "forbidden",
        },
        "causal_execution": {
            "warmup_can_populate_priors": True,
            "emittable_research_stages": ["train", "selection"],
            "boundary_quarantine_feature_or_event_use": "forbidden",
            "decision_time": "source_available_not_before_utc",
            "entry_time": "earliest_execution_time_utc",
            "entry_latency_minutes": 5,
            "bar_size": "5m",
            "hold_bars": 288,
            "hold_hours": 24,
            "interval": "[entry_time,entry_time+24h)",
            "split_containment": "entry_time>=split_start and scheduled_exit_time<=split_end",
            "ordering": "(entry_time,record_date)",
            "non_overlap": "accept earliest; suppress candidate when entry_time<previous_exit_time; equality allowed; no replacement",
            "notional_leverage": 0.5,
            "base_cost_bp_per_notional_per_side": 6,
            "stress_cost_bp_per_notional_per_side": 10,
            "funding": "exact, entry-inclusive/exit-exclusive, fixed entry quantity",
        },
        "controls": {
            "cash_only": "same 0.75/0.25 tails of cash_rank126 without debt confirmation",
            "debt_only": "same 0.75/0.25 tails of debt_rank126 without cash confirmation",
            "total_net_cash": "Table-II total withdrawals minus total deposits; exact 126-prior-computable midrank and same 0.75/0.25 tails",
            "direction_flip": "accepted primary clock with every side multiplied by -1; diagnostic only",
            "one_report_delay": "accepted primary side at next in-stage report entry; 24h exit; rebuild non-overlap",
            "deterministic_random_side": "same accepted primary clock; LONG iff first byte SHA256('DFFB-601|20260721|'+entry_time_utc)<128 else SHORT",
        },
        "comparators": {
            "input_boundary": "only hash-bound source-only clocks and allowlisted columns; rows remain unopened until support stage; every preregistration-stage read is enforced by the access ledger",
            "materialized_columns": "must equal the relevant allowlist exactly",
            "prohibited_column_normalization": "casefold then remove every non-ASCII-alphanumeric character",
            "flcc": "one clock per primary candidate_id plus union decision-date comparator",
            "tadi": "clock_mode primary",
            "auction_settlement_calendar": "union official auctionDate and issueDate for normalized-panel eligible keys",
            "total_net_cash": "same-window/same-tail DTS control",
            "decision_date_timezone": "America/New_York",
            "decision_date_jaccard_maximum": 0.30,
            "dffb_within_one_us_business_day_fraction_maximum": 0.50,
            "empty_comparator": "fail",
            "signed_occupied_exposure_grid": "complete 5m UTC common span; entry-inclusive/exit-exclusive; flat 0,long +1,short -1",
            "signed_occupied_exposure_absolute_pearson_maximum": 0.40,
            "empty_overlap_or_zero_variance": "fail",
        },
        "support_gates": {
            "count_basis": "accepted primary entries after rank readiness, stage containment, and global non-overlap",
            "train_total_minimum": 24,
            "train_2021_minimum": 8,
            "train_2022_minimum": 8,
            "train_long_minimum": 6,
            "train_short_minimum": 6,
            "train_maximum_month_share": 0.25,
            "selection_total_minimum": 12,
            "selection_each_half_minimum": 4,
            "selection_long_minimum": 3,
            "selection_short_minimum": 3,
            "selection_maximum_month_share": 0.33,
            "all_novelty_and_exposure_gates": True,
            "failure_action": "reject before outcomes; no repair",
        },
        "outcome_sequence": {
            "prerequisite": "committed source-only support pass and separately committed hash-frozen strict evaluator",
            "train": "[2021-01-01T00:00:00Z,2023-01-01T00:00:00Z)",
            "selection": "[2023-01-01T00:00:00Z,2024-01-01T00:00:00Z)",
            "later_source_extension": "only after both pass, exact parser/policy, report-only and never selection or repair",
        },
        "performance_gates": {
            "absolute_return_positive_each_opened_constituent": True,
            "cagr_to_strict_mdd_minimum_each_opened_constituent": 3.0,
            "strict_mdd_maximum_each_opened_constituent": 0.15,
            "stress_absolute_return_positive_each_opened_constituent": True,
            "weekly_cluster_sign_flip_p_maximum": 0.10,
            "weekly_cluster_sign_flip_draws": 100_000,
            "weekly_cluster_sign_flip_seed": 20_260_721,
            "positive_subperiods": [
                "train_2021",
                "train_2022",
                "selection_2023H1",
                "selection_2023H2",
            ],
            "train_long_and_short_absolute_return_positive": True,
            "cagr_to_strict_mdd_margin_over_support_ready_component_controls": 0.25,
            "component_controls": ["cash_only", "debt_only", "total_net_cash"],
            "cagr_clock": "full declared wall-clock window including idle cash",
            "strict_mdd": "global and pre-entry HWM over held 5m path; adverse OHLC order, exact funding, costs and virtual adverse-exit cost",
        },
        "promotion_boundary": "research only; no live or shadow promotion",
        "stopping_rule": "stop permanently at first binding/support/novelty/exposure/train/selection failure; after incidence no sign, threshold, taxonomy, exclusion, lookback, floor, hold, latency, control, model, or inversion repair",
    }


def _validate_config(cfg: Config, *, require_new_output: bool) -> None:
    output = _repository_path(cfg.preregistration_output)
    if output.suffix != ".json":
        raise ValueError("DFFB preregistration output must be JSON")
    protected = {
        _repository_path(path)
        for path in (
            SOURCE_DECISION,
            SOURCE_BUILDER,
            SOURCE_AUDITOR,
            PREREGISTRATION_SOURCE,
            PREREGISTRATION_DOCUMENT,
            SOURCE_MANIFEST,
            SOURCE_BUILD_REPORT,
            SOURCE_ROWS,
            OPERATING_CASH_ROWS,
            SOURCE_ANNOUNCEMENTS,
            AUDIT_MANIFEST,
            AUDIT_REPORT,
            SCHEMA_TRANSITIONS,
            FLCC_PREREGISTRATION,
            FLCC_CLOCK,
            FLCC_CLOCK_SOURCE,
            TADI_PREREGISTRATION,
            TADI_CLOCK,
            TADI_CLOCK_SOURCE,
            AUCTION_MANIFEST,
            AUCTION_PANEL,
            AUCTION_RAW_PAGE_0,
            AUCTION_RAW_PAGE_1,
        )
    }
    if output in protected:
        raise ValueError("DFFB preregistration output aliases a protected source")
    if require_new_output and output.exists():
        raise FileExistsError("DFFB preregistration is immutable")


def _collect_verified_bindings() -> dict[str, Any]:
    global _ACTIVE_ACCESS_LEDGER
    if _ACTIVE_ACCESS_LEDGER is not None:
        raise RuntimeError("DFFB preregistration access ledger is already active")
    ledger = AccessLedger()
    _ACTIVE_ACCESS_LEDGER = ledger
    try:
        _validate_preregistration_import_boundary()
        if sha256_file(PREREGISTRATION_DOCUMENT) != PREREGISTRATION_DOCUMENT_SHA256:
            raise RuntimeError("DFFB preregistration document SHA drift")
        source_binding = _validate_source_binding()
        comparator_binding = _validate_comparator_binding()
        preregistration_source = {
            "path": str(PREREGISTRATION_SOURCE),
            "sha256": sha256_file(PREREGISTRATION_SOURCE),
        }
    finally:
        _ACTIVE_ACCESS_LEDGER = None
    ledger.validate_complete()
    outcome_boundary = ledger.outcome_boundary()
    if outcome_boundary != EXPECTED_OUTCOME_BOUNDARY:
        raise RuntimeError("DFFB observed outcome boundary drift")
    access_events = list(ledger.events)
    return {
        "source_binding": source_binding,
        "comparator_binding": comparator_binding,
        "preregistration_source": preregistration_source,
        "preregistration_document": {
            "path": str(PREREGISTRATION_DOCUMENT),
            "sha256": PREREGISTRATION_DOCUMENT_SHA256,
        },
        "outcome_boundary": outcome_boundary,
        "access_ledger": {
            "events": access_events,
            "event_count": len(access_events),
            "ledger_hash": canonical_hash(access_events),
        },
    }


def _artifact_core(cfg: Config) -> dict[str, Any]:
    bindings = _collect_verified_bindings()
    frozen_policy = policy()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "config": asdict(cfg),
        "source_binding": bindings["source_binding"],
        "comparator_binding": bindings["comparator_binding"],
        "policy": frozen_policy,
        "policy_hash": canonical_hash(frozen_policy),
        "outcomes_opened": False,
        "outcome_boundary": bindings["outcome_boundary"],
        "access_ledger": bindings["access_ledger"],
        "incidence_or_support_results": None,
        "research_sequence": [
            "source-only support, novelty, and occupied-exposure gates",
            "commit/hash-freeze strict evaluator",
            "train outcomes",
            "selection outcomes",
            "optional later report-only source extension",
        ],
        "preregistration_source": bindings["preregistration_source"],
        "preregistration_document": bindings["preregistration_document"],
    }


def _temporary_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    handle.close()
    return Path(handle.name)


def write_preregistration(cfg: Config) -> dict[str, Any]:
    _validate_config(cfg, require_new_output=True)
    core = _artifact_core(cfg)
    artifact = {**core, "manifest_hash": canonical_hash(core)}
    output = _repository_path(cfg.preregistration_output)
    temporary = _temporary_path(output)
    try:
        temporary.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.link(temporary, output)
        return artifact
    finally:
        temporary.unlink(missing_ok=True)


def load_preregistration(path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    artifact = _read_json(path)
    core = _manifest_core(artifact)
    if canonical_hash(core) != artifact.get("manifest_hash"):
        raise RuntimeError("DFFB preregistration canonical hash mismatch")
    if artifact.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("DFFB preregistration protocol drift")
    if artifact.get("policy_id") != POLICY_ID or artifact.get("policy") != policy():
        raise RuntimeError("DFFB preregistration policy drift")
    if artifact.get("policy_hash") != canonical_hash(policy()):
        raise RuntimeError("DFFB preregistration policy hash drift")
    if artifact.get("outcomes_opened") is not False:
        raise RuntimeError("DFFB preregistration opened outcomes")
    if artifact.get("outcome_boundary") != EXPECTED_OUTCOME_BOUNDARY:
        raise RuntimeError("DFFB preregistration outcome boundary drift")
    if artifact.get("incidence_or_support_results") is not None:
        raise RuntimeError("DFFB preregistration contains incidence or support results")
    raw_config = artifact.get("config")
    if not isinstance(raw_config, dict):
        raise RuntimeError("DFFB preregistration config missing")
    try:
        cfg = Config(**raw_config)
    except TypeError as exc:
        raise RuntimeError("DFFB preregistration config drift") from exc
    _validate_config(cfg, require_new_output=False)
    if _repository_path(path) != _repository_path(cfg.preregistration_output):
        raise RuntimeError("DFFB preregistration output-path binding drift")
    bindings = _collect_verified_bindings()
    if artifact.get("preregistration_source") != bindings["preregistration_source"]:
        raise RuntimeError("DFFB preregistration source binding drift")
    if artifact.get("preregistration_document") != bindings["preregistration_document"]:
        raise RuntimeError("DFFB preregistration document binding drift")
    if artifact.get("access_ledger") != bindings["access_ledger"]:
        raise RuntimeError("DFFB preregistration access-ledger binding drift")
    if artifact.get("outcome_boundary") != bindings["outcome_boundary"]:
        raise RuntimeError("DFFB preregistration observed outcome boundary drift")
    if artifact.get("source_binding") != bindings["source_binding"]:
        raise RuntimeError("DFFB preregistration frozen source binding drift")
    if artifact.get("comparator_binding") != bindings["comparator_binding"]:
        raise RuntimeError("DFFB preregistration comparator binding drift")
    return artifact


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preregistration-output", default=Config.preregistration_output
    )
    return Config(**vars(parser.parse_args()))


def main() -> None:
    print(
        json.dumps(
            write_preregistration(parse_args()),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
