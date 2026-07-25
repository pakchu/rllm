"""Seal the outcome-blind CEFS-D1 source-language support contract.

This module may hash frozen containers, inspect only their physical CSV
headers, and validate source-manifest metadata. It never parses a source data
row, market bar, funding mark, return, reward, model record, selected action,
trade, PnL, CAGR, or MDD.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
POLICY_ID = "CEFS-D1"
PROTOCOL_VERSION = "cboe_edge_flip_sequence_policy_preregistration_v1"
DEFAULT_OUTPUT = (
    "results/cboe_edge_flip_sequence_policy_preregistration_2026-07-25.json"
)
PRODUCER_SCRIPT = "training/preregister_cboe_edge_flip_sequence_policy.py"

AUDIT_DOCUMENT = "docs/post-lcdp-alpha-mechanism-audit-2026-07-25.md"
AUDIT_DOCUMENT_SHA256 = (
    "4bbf47feb0579e4b4766ad3c0aa9a282a082c4c60fee723065ea70c7f8d50288"
)
BOUNDARY_DOCUMENT = (
    "docs/cboe-edge-flip-sequence-policy-boundary-2026-07-25.md"
)
BOUNDARY_DOCUMENT_SHA256 = (
    "d0b522a7ac87e3526d6cd740bb81304bd73042bc327978660eb551b159c16ec3"
)
BOUNDARY_COMMIT = "b7297ccf9235da98edcbe9b35d84009245972321"

CLOCK_AUTHORITY = (
    "docs/cboe-cross-surface-pressure-grammar-boundary-2026-07-24.md"
)
CLOCK_AUTHORITY_SHA256 = (
    "0b6feb15d1e7b616b5b65bb266b15db7e3fdcf82765b5848c76d68e804cb39f2"
)

TERM_SOURCE = (
    "data/cboe_volatility_term_structure_2018_2023/"
    "cboe_vix_term_structure_2018-01-01_2023-12-31.csv.gz"
)
TERM_SOURCE_SHA256 = (
    "6f1b2f7f3a5b1e4d5001d673e6ff54374791879c278248ce27b3d610e4f75dc7"
)
TERM_HEADER = (
    "observation_date",
    "VIX9D_close",
    "VIX_close",
    "VIX3M_close",
)
TERM_HEADER_SHA256 = (
    "b2fc60cae8d080d3b47a1a55c48438b63f91530cc345f1b6ef78cee05cc57e20"
)
TERM_MANIFEST = (
    "data/cboe_volatility_term_structure_2018_2023/build_manifest.json"
)
TERM_MANIFEST_SHA256 = (
    "42b2a35ad131bd63574d2adcf684e28766dc3060fa645fc749df10dd3fb27f27"
)
TERM_MANIFEST_HASH = (
    "d10f798e26cceca76c3a998cfaf97af068d443431121ed19e80a986a08ad4ce3"
)
TERM_AUDIT = "docs/cboe-volatility-term-structure-source-audit-2026-07-17.md"
TERM_AUDIT_SHA256 = (
    "985799cd9a26217bffb678ae2d5dbfa81070c84edf955bec681de663a3b63c58"
)

TAIL_SOURCE = (
    "data/cboe_tail_risk_2018_2023/"
    "cboe_tail_risk_2018-01-01_2023-12-31.csv.gz"
)
TAIL_SOURCE_SHA256 = (
    "cdde3f8d4bb1e23d00b192f5f9ef759aefba9087be5fd60653e9c02479dfa41a"
)
TAIL_HEADER = (
    "observation_date",
    "SKEW_close",
    "VVIX_close",
    "VIX_close",
)
TAIL_HEADER_SHA256 = (
    "bdc2e42c1d356ebd815c491af9b20211d1bc8f2781c0917d92bbf04f1f0a5dc3"
)
TAIL_MANIFEST = "data/cboe_tail_risk_2018_2023/build_manifest.json"
TAIL_MANIFEST_SHA256 = (
    "9ef80ef3034c93d97c5b2a8160b2502527287d570d15f9d7166d631d9866c7bd"
)
TAIL_MANIFEST_HASH = (
    "091ddf3050035156814fe168e1edcac193e23cca9f39a3ef0140bcb5f8265d72"
)
TAIL_AUDIT = "docs/cboe-tail-risk-source-audit-2026-07-18.md"
TAIL_AUDIT_SHA256 = (
    "706c5839cc5babc7b150d71a139b659c76b2cc5a1a355de61868842000b2847b"
)

FLOW_SOURCE = (
    "data/cboe_option_flow_2020_2023/"
    "cboe_option_flow_2020-01-01_2023-12-31.csv.gz"
)
FLOW_SOURCE_SHA256 = (
    "35ef106ef01e3abadbcb4a6227187dd1d7cf2722191bd146bac06d08d1684a78"
)
FLOW_HEADER = (
    "observation_date",
    "total_pcr",
    "index_pcr",
    "equity_pcr",
    "vix_pcr",
    "spx_pcr",
    "total_call_volume",
    "total_put_volume",
    "total_volume",
    "index_call_volume",
    "index_put_volume",
    "index_volume",
    "equity_call_volume",
    "equity_put_volume",
    "equity_volume",
    "vix_call_volume",
    "vix_put_volume",
    "vix_volume",
    "spx_call_volume",
    "spx_put_volume",
    "spx_volume",
    "response_sha256",
)
FLOW_HEADER_SHA256 = (
    "a98314aa376428c5d237837121305c5cc4c4892e25ea3db3127d466b451281d7"
)
FLOW_RELATION_COLUMNS = (
    "observation_date",
    "total_pcr",
    "index_pcr",
    "equity_pcr",
    "vix_pcr",
    "spx_pcr",
    "index_volume",
    "vix_volume",
)
FLOW_INTEGRITY_COLUMNS = (
    "response_sha256",
)
FLOW_ALLOWED_COLUMNS = (*FLOW_RELATION_COLUMNS, *FLOW_INTEGRITY_COLUMNS)
FLOW_MANIFEST = "data/cboe_option_flow_2020_2023/build_manifest.json"
FLOW_MANIFEST_SHA256 = (
    "0a513b146ad5857d9ab7311e978152c308de64db8ef29c4d463eb07ea503089e"
)
FLOW_MANIFEST_HASH = (
    "07c2effecd8c67e7ddb81abf5e01620a667a52e2db02c78b742eb49b506e1bac"
)
FLOW_AUDIT = "docs/cboe-option-flow-source-audit-2026-07-18.md"
FLOW_AUDIT_SHA256 = (
    "c182ee2f9078c5bee2d2a0f3ec488105980a1c54651b8b651db2e3af96278f8f"
)

PREDECESSOR_DOCUMENTS = {
    "cspg_retirement": (
        "docs/cspg-source-support-retirement-2026-07-24.md",
        "084be289afe411d46df7bd4fea9e82528559a29d2ecdc8529bf2006d9942ac29",
    ),
    "cxrt_rejection": (
        "docs/cxrt-source-support-result-2026-07-24.md",
        "af73042b035b5cc2b1ad6cf49c0270155e19a315dde7c6da9463332896d3725d",
    ),
    "oprr_rejection": (
        "docs/oprr-source-support-result-2026-07-24.md",
        "9dfa67235b80817ceb40cf3a4fe4041300f8279ccad3a0f25dad26eeba0ef2e5",
    ),
    "cvtr_rejection": (
        "docs/cboe-volatility-term-rotation-rejection-2026-07-17.md",
        "f5101951b2ddc610e55603cbe9e96fae81cc9412ae70648eb5353e29f6d130e9",
    ),
    "cthd_rejection": (
        "docs/cboe-tail-hedge-disagreement-rejection-2026-07-18.md",
        "ab3985a9fa2175da238d8c43a8ee57dc7f0e149f81ca3ad85debfdffc955b9ef",
    ),
    "cihm_rejection": (
        "docs/cboe-institutional-hedge-migration-rejection-2026-07-18.md",
        "823bcc1613f0404937c6d405d947483057ce5b4a7f9766a5279afdb451c4cf65",
    ),
}

NEW_YORK = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
EDGE_LEVELS = ("LOWER", "EQUAL", "HIGHER")
CONTROL_LEVELS = (*EDGE_LEVELS, "MASKED")
POSITION_CONTEXTS = ("TARGET_FLAT", "TARGET_LONG", "TARGET_SHORT")
ACTION_SPACE = ("TARGET_LONG", "TARGET_FLAT", "TARGET_SHORT")
STATE_LABELS = ("EARLIEST", "EARLY", "MIDDLE", "LATE", "CURRENT")
EDGE_NAMES = (
    "TERM_FRONT_LEVEL",
    "TERM_BACK_LEVEL",
    "TERM_FRONT_CHANGE",
    "TERM_BACK_CHANGE",
    "TAIL_SKEW_CHANGE",
    "TAIL_VOLVOL_CHANGE",
    "FLOW_TOTAL_PCR_CHANGE",
    "FLOW_INDEX_PCR_CHANGE",
    "FLOW_EQUITY_PCR_CHANGE",
    "FLOW_VIX_PCR_CHANGE",
    "FLOW_SPX_PCR_CHANGE",
    "FLOW_VIX_SHARE_CHANGE",
)
CONTROL_IDS = (
    "reverse_sequence",
    "stale_current",
    "group_order_rotation",
    "within_group_value_rotation",
    "term_only",
    "tail_only",
    "flow_only",
    "current_only",
)

EDGE_FORMULAS = {
    "TERM_FRONT_LEVEL": "compare(VIX9D_close[C],VIX_close[C])",
    "TERM_BACK_LEVEL": "compare(VIX_close[C],VIX3M_close[C])",
    "TERM_FRONT_CHANGE": (
        "compare_ratio(VIX9D_close[C],VIX_close[C],"
        "VIX9D_close[P],VIX_close[P])"
    ),
    "TERM_BACK_CHANGE": (
        "compare_ratio(VIX_close[C],VIX3M_close[C],"
        "VIX_close[P],VIX3M_close[P])"
    ),
    "TAIL_SKEW_CHANGE": "compare(SKEW_close[C],SKEW_close[P])",
    "TAIL_VOLVOL_CHANGE": (
        "compare_ratio(VVIX_close[C],VIX_close[C],"
        "VVIX_close[P],VIX_close[P])"
    ),
    "FLOW_TOTAL_PCR_CHANGE": "compare(total_pcr[C],total_pcr[P])",
    "FLOW_INDEX_PCR_CHANGE": "compare(index_pcr[C],index_pcr[P])",
    "FLOW_EQUITY_PCR_CHANGE": "compare(equity_pcr[C],equity_pcr[P])",
    "FLOW_VIX_PCR_CHANGE": "compare(vix_pcr[C],vix_pcr[P])",
    "FLOW_SPX_PCR_CHANGE": "compare(spx_pcr[C],spx_pcr[P])",
    "FLOW_VIX_SHARE_CHANGE": (
        "compare_ratio(vix_volume[C],index_volume[C],"
        "vix_volume[P],index_volume[P])"
    ),
}

FORBIDDEN_SUPPORT_KEYS = frozenset(
    {
        "execution_price",
        "funding_rate",
        "future_return",
        "reward",
        "model_target",
        "model_prediction",
        "action",
        "trade",
        "pnl",
        "cagr",
        "mdd",
        "sharpe",
    }
)


@dataclass(frozen=True)
class Policy:
    source_start_inclusive: str = "2020-01-01"
    source_end_exclusive: str = "2024-01-01"
    sequence_states: int = 5
    edges_per_state: int = 12
    availability_hour_new_york: int = 9
    availability_minute: int = 30
    entry_hour_new_york: int = 9
    entry_minute: int = 35
    hold_bars: int = 288
    bar_minutes: int = 5
    minimum_total_intervals: int = 920
    minimum_year_intervals: int = 230
    minimum_quarter_intervals: int = 50
    maximum_level_edge_share: float = 0.98
    minimum_change_direction_share: float = 0.10
    maximum_change_level_share: float = 0.88
    minimum_distinct_current_signatures: int = 40
    maximum_current_signature_share: float = 0.15
    minimum_unique_sequence_share: float = 0.80
    maximum_sequence_signature_share: float = 0.02
    maximum_role_level_share_drift: float = 0.25
    minimum_reverse_difference_share: float = 0.95
    minimum_stale_semantic_difference_share: float = 0.35
    minimum_rotation_semantic_difference_share: float = 0.50


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(repository_path(path).read_bytes()).hexdigest()


def csv_header_bytes(path: str | Path) -> bytes:
    source = repository_path(path)
    if source.suffix == ".gz":
        with gzip.open(source, "rb") as handle:
            return handle.readline()
    with source.open("rb") as handle:
        return handle.readline()


def csv_header(path: str | Path) -> tuple[str, ...]:
    return tuple(csv_header_bytes(path).decode("utf-8").rstrip("\r\n").split(","))


def sha256_csv_header(path: str | Path) -> str:
    return hashlib.sha256(csv_header_bytes(path)).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def assert_committed(path: str, *, expected_commit: str | None = None) -> str:
    _git_output("ls-files", "--error-unmatch", "--", path)
    dirty = subprocess.run(
        ("git", "diff", "--quiet", "--", path),
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    staged = subprocess.run(
        ("git", "diff", "--cached", "--quiet", "--", path),
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    if dirty.returncode or staged.returncode:
        raise RuntimeError(f"frozen artifact is dirty: {path}")
    commit = _git_output("log", "-1", "--format=%H", "--", path)
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError(f"frozen artifact commit mismatch: {path}: {commit}")
    return commit


def producer_binding() -> dict[str, str]:
    commit = assert_committed(PRODUCER_SCRIPT)
    return {
        "path": PRODUCER_SCRIPT,
        "commit": commit,
        "sha256": sha256_file(PRODUCER_SCRIPT),
    }


def fixed_clock(
    observation_date: date,
    policy: Policy = Policy(),
) -> dict[str, datetime]:
    available_day = observation_date + timedelta(days=1)
    available = datetime.combine(
        available_day,
        time(policy.availability_hour_new_york, policy.availability_minute),
        tzinfo=NEW_YORK,
    )
    entry = datetime.combine(
        available_day,
        time(policy.entry_hour_new_york, policy.entry_minute),
        tzinfo=NEW_YORK,
    )
    exit_at = entry.astimezone(UTC) + timedelta(
        minutes=policy.hold_bars * policy.bar_minutes
    )
    return {
        "available_utc": available.astimezone(UTC),
        "entry_utc": entry.astimezone(UTC),
        "exit_utc": exit_at,
    }


def serialize_state(
    label: str,
    edges: Mapping[str, str],
    *,
    control: bool = False,
) -> tuple[str, ...]:
    if label not in STATE_LABELS:
        raise ValueError("CEFS state label changed")
    if tuple(edges) != EDGE_NAMES:
        raise ValueError("CEFS edge fields or order changed")
    vocabulary = CONTROL_LEVELS if control else EDGE_LEVELS
    for edge, level in edges.items():
        if level not in vocabulary:
            raise ValueError(f"invalid CEFS edge token: {edge}={level}")
    return tuple(f"{label}.{edge}={edges[edge]}" for edge in EDGE_NAMES)


def serialize_prompt(
    states: Mapping[str, Mapping[str, str]],
    position: str,
    *,
    control: bool = False,
) -> str:
    if tuple(states) != STATE_LABELS:
        raise ValueError("CEFS sequence labels or order changed")
    if position not in POSITION_CONTEXTS:
        raise ValueError("CEFS position context changed")
    lines: list[str] = []
    for label, edges in states.items():
        lines.extend(serialize_state(label, edges, control=control))
    lines.append(f"POSITION={position}")
    return "\n".join(lines) + "\n"


def _validate_policy(policy: Policy) -> None:
    if policy != Policy():
        raise ValueError("CEFS-D1 policy is frozen")
    if len(STATE_LABELS) != policy.sequence_states:
        raise RuntimeError("CEFS state count drift")
    if len(EDGE_NAMES) != policy.edges_per_state:
        raise RuntimeError("CEFS edge count drift")
    if tuple(EDGE_FORMULAS) != EDGE_NAMES:
        raise RuntimeError("CEFS formula order drift")
    probe = fixed_clock(date(2023, 3, 10), policy)
    if probe["entry_utc"] - probe["available_utc"] != timedelta(minutes=5):
        raise RuntimeError("CEFS inference buffer drift")
    if probe["exit_utc"] - probe["entry_utc"] != timedelta(hours=24):
        raise RuntimeError("CEFS exact 288-bar hold drift")


def _validate_manifest_panel(
    *,
    manifest_path: str,
    manifest_hash: str,
    panel_path: str,
    panel_sha256: str,
    expected_columns: tuple[str, ...],
) -> None:
    payload = json.loads(repository_path(manifest_path).read_text())
    if payload.get("manifest_hash") != manifest_hash:
        raise RuntimeError(f"CEFS source manifest hash mismatch: {manifest_path}")
    panel = payload.get("panel", {})
    required = {
        "path": panel_path,
        "sha256": panel_sha256,
        "columns": list(expected_columns),
    }
    for key, expected in required.items():
        if panel.get(key) != expected:
            raise RuntimeError(f"CEFS source manifest panel mismatch: {key}")
    contract = payload.get("source_contract", {})
    if contract.get("market_or_label_rows_read") != 0:
        raise RuntimeError("CEFS source manifest opened market or label rows")
    horizon = contract.get("research_horizon")
    if horizon not in (
        ["2018-01-01", "2024-01-01"],
        ["2020-01-01", "2024-01-01"],
    ):
        raise RuntimeError("CEFS source manifest horizon mismatch")


def _validate_source_anchors() -> None:
    anchors = {
        AUDIT_DOCUMENT: AUDIT_DOCUMENT_SHA256,
        BOUNDARY_DOCUMENT: BOUNDARY_DOCUMENT_SHA256,
        CLOCK_AUTHORITY: CLOCK_AUTHORITY_SHA256,
        TERM_SOURCE: TERM_SOURCE_SHA256,
        TERM_MANIFEST: TERM_MANIFEST_SHA256,
        TERM_AUDIT: TERM_AUDIT_SHA256,
        TAIL_SOURCE: TAIL_SOURCE_SHA256,
        TAIL_MANIFEST: TAIL_MANIFEST_SHA256,
        TAIL_AUDIT: TAIL_AUDIT_SHA256,
        FLOW_SOURCE: FLOW_SOURCE_SHA256,
        FLOW_MANIFEST: FLOW_MANIFEST_SHA256,
        FLOW_AUDIT: FLOW_AUDIT_SHA256,
        **{path: digest for path, digest in PREDECESSOR_DOCUMENTS.values()},
    }
    for path, expected in anchors.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"CEFS frozen dependency mismatch: {path}")
    headers = (
        (TERM_SOURCE, TERM_HEADER, TERM_HEADER_SHA256),
        (TAIL_SOURCE, TAIL_HEADER, TAIL_HEADER_SHA256),
        (FLOW_SOURCE, FLOW_HEADER, FLOW_HEADER_SHA256),
    )
    for path, expected_header, expected_hash in headers:
        if csv_header(path) != expected_header:
            raise RuntimeError(f"CEFS physical header mismatch: {path}")
        if sha256_csv_header(path) != expected_hash:
            raise RuntimeError(f"CEFS header hash mismatch: {path}")
    _validate_manifest_panel(
        manifest_path=TERM_MANIFEST,
        manifest_hash=TERM_MANIFEST_HASH,
        panel_path=TERM_SOURCE,
        panel_sha256=TERM_SOURCE_SHA256,
        expected_columns=TERM_HEADER,
    )
    _validate_manifest_panel(
        manifest_path=TAIL_MANIFEST,
        manifest_hash=TAIL_MANIFEST_HASH,
        panel_path=TAIL_SOURCE,
        panel_sha256=TAIL_SOURCE_SHA256,
        expected_columns=TAIL_HEADER,
    )
    _validate_manifest_panel(
        manifest_path=FLOW_MANIFEST,
        manifest_hash=FLOW_MANIFEST_HASH,
        panel_path=FLOW_SOURCE,
        panel_sha256=FLOW_SOURCE_SHA256,
        expected_columns=FLOW_HEADER,
    )


def _source_contracts() -> dict[str, Any]:
    return {
        "term": {
            "path": TERM_SOURCE,
            "sha256": TERM_SOURCE_SHA256,
            "physical_header": list(TERM_HEADER),
            "header_sha256": TERM_HEADER_SHA256,
            "allowed_columns": list(TERM_HEADER),
            "manifest": TERM_MANIFEST,
            "manifest_sha256": TERM_MANIFEST_SHA256,
            "manifest_hash": TERM_MANIFEST_HASH,
            "audit": TERM_AUDIT,
            "audit_sha256": TERM_AUDIT_SHA256,
        },
        "tail": {
            "path": TAIL_SOURCE,
            "sha256": TAIL_SOURCE_SHA256,
            "physical_header": list(TAIL_HEADER),
            "header_sha256": TAIL_HEADER_SHA256,
            "allowed_columns": list(TAIL_HEADER),
            "manifest": TAIL_MANIFEST,
            "manifest_sha256": TAIL_MANIFEST_SHA256,
            "manifest_hash": TAIL_MANIFEST_HASH,
            "audit": TAIL_AUDIT,
            "audit_sha256": TAIL_AUDIT_SHA256,
        },
        "flow": {
            "path": FLOW_SOURCE,
            "sha256": FLOW_SOURCE_SHA256,
            "physical_header": list(FLOW_HEADER),
            "header_sha256": FLOW_HEADER_SHA256,
            "relation_columns": list(FLOW_RELATION_COLUMNS),
            "integrity_text_columns": list(FLOW_INTEGRITY_COLUMNS),
            "integrity_text_rules": {
                "response_sha256": (
                    "exact 64 lowercase hexadecimal characters; forbidden "
                    "from relations, tokens, prompts, controls, labels, "
                    "actions, and model inputs"
                )
            },
            "forbidden_numeric_columns": [
                column
                for column in FLOW_HEADER
                if column not in FLOW_ALLOWED_COLUMNS
            ],
            "manifest": FLOW_MANIFEST,
            "manifest_sha256": FLOW_MANIFEST_SHA256,
            "manifest_hash": FLOW_MANIFEST_HASH,
            "audit": FLOW_AUDIT,
            "audit_sha256": FLOW_AUDIT_SHA256,
        },
    }


def build_manifest(
    *,
    policy: Policy = Policy(),
    producer_binding_override: Mapping[str, str] | None = None,
    validate_dependencies: bool = True,
    _skip_validation: bool = False,
) -> dict[str, Any]:
    _validate_policy(policy)
    if validate_dependencies:
        _validate_source_anchors()
        assert_committed(BOUNDARY_DOCUMENT, expected_commit=BOUNDARY_COMMIT)
    producer = (
        dict(producer_binding_override)
        if producer_binding_override is not None
        else producer_binding()
    )
    core: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "phase": "outcome_blind_source_language_support_preregistration",
        "decision": "freeze_source_language_support_only",
        "policy": asdict(policy),
        "authority": {
            "audit_document": AUDIT_DOCUMENT,
            "audit_document_sha256": AUDIT_DOCUMENT_SHA256,
            "boundary_document": BOUNDARY_DOCUMENT,
            "boundary_document_sha256": BOUNDARY_DOCUMENT_SHA256,
            "boundary_commit": BOUNDARY_COMMIT,
            "clock_authority": CLOCK_AUTHORITY,
            "clock_authority_sha256": CLOCK_AUTHORITY_SHA256,
            "producer": producer,
        },
        "sources": _source_contracts(),
        "parsing": {
            "date": "exact ten-byte YYYY-MM-DD strict Gregorian round-trip",
            "decimal": (
                "finite strictly-positive plain Decimal; no exponent, sign, "
                "whitespace, comma, underscore, float, rounding, or epsilon"
            ),
            "volume": "exact strictly-positive base-ten integer",
            "intersection": (
                "exact sorted three-panel source-date intersection; no fill, "
                "carry, interpolation, substitution, or zero"
            ),
            "vix_identity": "term VIX_close equals tail VIX_close exactly",
            "flow_hidden_fields": (
                "header and whole-file hash validation only; no numeric parse"
            ),
        },
        "relation_language": {
            "comparison_levels": list(EDGE_LEVELS),
            "comparison": "LOWER if left<right; EQUAL if left=right; else HIGHER",
            "ratio_comparison": "compare(a*d,c*b) for positive a/b and c/d",
            "ordered_edges": list(EDGE_NAMES),
            "edge_formulas": EDGE_FORMULAS,
            "aggregation_forbidden": True,
            "source_owned_side_forbidden": True,
        },
        "sequence_language": {
            "ordered_states": list(STATE_LABELS),
            "state_count": policy.sequence_states,
            "edges_per_state": policy.edges_per_state,
            "primary_edge_vocabulary": list(EDGE_LEVELS),
            "control_edge_vocabulary": list(CONTROL_LEVELS),
            "position_contexts": list(POSITION_CONTEXTS),
            "position_templates_per_schedule": 3,
            "serialization": (
                "sixty RELATIVE_STATE.EDGE=LEVEL lines in frozen order, then "
                "POSITION=TARGET_* and one final newline"
            ),
            "prompt_forbidden": [
                "raw or formatted number",
                "date, weekday, month, or year",
                "rank, quantile, score, pressure, consensus, or source side",
                "source path or hash",
                "price, return, funding, pnl, equity, or drawdown",
                "split or checkpoint identity",
                "model confidence or hidden reasoning",
            ],
            "action_space": list(ACTION_SPACE),
            "safety_target": "TARGET_FLAT without model inference",
        },
        "clock": {
            "timezone": "America/New_York via zoneinfo",
            "source_observation": "D",
            "availability": "calendar D+1 09:30 America/New_York",
            "entry": "calendar D+1 09:35 America/New_York",
            "scheduled_exit": "entry plus 288 five-minute bars",
            "weekend_and_holiday_entry": True,
            "future_row_membership_used": False,
            "interval": "[entry,scheduled_exit)",
            "reservation": (
                "ascending entry; suppress strict overlap; accept equality"
            ),
            "gap_rule": "flat after scheduled exit until a later reserved entry",
            "equality_rule": "direct target-to-target rebalance at equality",
        },
        "chronology": {
            "train_development": "[2020-01-01,2022-01-01)",
            "test_checkpoint_selection": "[2022-01-01,2023-01-01)",
            "candidate_eval": "[2023-01-01,2024-01-01)",
            "sealed_extensions": ["2024", "2025", "2026_ytd"],
            "containment": "entry and scheduled exit both inside role",
            "source_support_may_read_2020_2023_tokens": True,
            "market_access_authorized_now": False,
        },
        "controls": {
            "ordered_ids": list(CONTROL_IDS),
            "position_context_preserved": True,
            "schedule_preserved": True,
            "masked_primary_forbidden": True,
            "definitions": {
                "reverse_sequence": "reverse five state blocks",
                "stale_current": "CURRENT duplicates LATE",
                "group_order_rotation": "flow, term, tail serialization order",
                "within_group_value_rotation": (
                    "swap term pairs, swap tail pair, rotate six flow values"
                ),
                "term_only": "tail and flow MASKED",
                "tail_only": "term and flow MASKED",
                "flow_only": "term and tail MASKED",
                "current_only": "all non-current states MASKED",
            },
        },
        "support_gates": {
            "first_stop_order": [
                "authority_forbidden_access",
                "schema_chronology",
                "schedule_support",
                "primitive_edge_support",
                "state_diversity_stability",
                "source_only_controls",
                "determinism_append_replay",
            ],
            "common_date_first": "2020-01-02",
            "common_date_last": "2023-12-29",
            "common_dates_exact": 1006,
            "maximum_common_date_gap_days": 10,
            "minimum_total_intervals": policy.minimum_total_intervals,
            "minimum_year_intervals": policy.minimum_year_intervals,
            "minimum_quarter_intervals": policy.minimum_quarter_intervals,
            "maximum_level_edge_share": policy.maximum_level_edge_share,
            "minimum_change_direction_share": (
                policy.minimum_change_direction_share
            ),
            "maximum_change_level_share": policy.maximum_change_level_share,
            "minimum_distinct_current_signatures": (
                policy.minimum_distinct_current_signatures
            ),
            "maximum_current_signature_share": (
                policy.maximum_current_signature_share
            ),
            "minimum_unique_sequence_share": (
                policy.minimum_unique_sequence_share
            ),
            "maximum_sequence_signature_share": (
                policy.maximum_sequence_signature_share
            ),
            "maximum_role_level_share_drift": (
                policy.maximum_role_level_share_drift
            ),
            "minimum_reverse_difference_share": (
                policy.minimum_reverse_difference_share
            ),
            "minimum_stale_semantic_difference_share": (
                policy.minimum_stale_semantic_difference_share
            ),
            "minimum_rotation_semantic_difference_share": (
                policy.minimum_rotation_semantic_difference_share
            ),
            "prefix_end_exclusive": [
                "2021-01-01",
                "2022-01-01",
                "2023-01-01",
                "2024-01-01",
            ],
            "failure_action": "retire_cefs_d1_unchanged_before_outcomes",
            "pass_action": "authorize_economic_rllm_evaluator_freeze_only",
        },
        "forbidden_access": {
            "source_data_rows_parsed_by_preregistration": 0,
            "post_2023_source_rows_opened": 0,
            "market_rows_opened": 0,
            "funding_rows_opened": 0,
            "future_return_rows_built": 0,
            "reward_rows_built": 0,
            "model_rows_built": 0,
            "selected_action_rows_built": 0,
            "trade_rows_built": 0,
            "pnl_cagr_mdd_values_computed": 0,
        },
        "contingent_economic_chronology": {
            "authorized_now": False,
            "model_family_frozen_now": False,
            "train": "2020-2021 only after evaluator/model freeze",
            "test": "2022 once under frozen checkpoint rule",
            "eval": "2023 once; no repair",
            "sequential_transfer": ["2024", "2025", "2026_ytd"],
            "stop": "first failed source, economic, control, or transfer gate",
        },
        "research_history": {
            "globally_pristine": False,
            "candidate_specific_edge_sequence_opened": False,
            "corrected_clock_reused_without_novelty_claim": True,
            "historical_current_vintage_is_live_equivalent": False,
            "predecessor_documents": {
                key: {"path": path, "sha256": digest}
                for key, (path, digest) in PREDECESSOR_DOCUMENTS.items()
            },
        },
    }
    payload = {**core, "manifest_hash": canonical_hash(core)}
    if not _skip_validation:
        validate_manifest(payload)
    return payload


def _walk_keys(payload: Any) -> Sequence[str]:
    keys: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            keys.append(str(key).lower())
            keys.extend(_walk_keys(value))
    elif isinstance(payload, list):
        for value in payload:
            keys.extend(_walk_keys(value))
    return keys


def validate_manifest(payload: Mapping[str, Any]) -> None:
    normalized = json.loads(json.dumps(dict(payload)))
    forbidden = FORBIDDEN_SUPPORT_KEYS.intersection(_walk_keys(normalized))
    if forbidden:
        raise ValueError(
            f"CEFS preregistration contains forbidden keys: {sorted(forbidden)}"
        )
    try:
        producer = normalized["authority"]["producer"]
    except (KeyError, TypeError) as error:
        raise ValueError("CEFS preregistration producer binding missing") from error
    expected = build_manifest(
        producer_binding_override=producer,
        validate_dependencies=False,
        _skip_validation=True,
    )
    if normalized != expected:
        raise ValueError("CEFS preregistration differs from frozen code")
    counters = normalized.get("forbidden_access", {})
    if not counters or any(value != 0 for value in counters.values()):
        raise ValueError("CEFS preregistration forbidden counter is nonzero")
    language = normalized.get("sequence_language", {})
    if tuple(language.get("ordered_states", ())) != STATE_LABELS:
        raise ValueError("CEFS state order mismatch")
    if tuple(
        normalized.get("relation_language", {}).get("ordered_edges", ())
    ) != EDGE_NAMES:
        raise ValueError("CEFS edge order mismatch")
    if tuple(language.get("action_space", ())) != ACTION_SPACE:
        raise ValueError("CEFS action space mismatch")
    controls = normalized.get("controls", {}).get("ordered_ids", ())
    if tuple(controls) != CONTROL_IDS:
        raise ValueError("CEFS control identity mismatch")


def _output_path(path: str | Path) -> Path:
    candidate = Path(path)
    if (
        str(path).startswith("~")
        or candidate.is_absolute()
        or ".." in candidate.parts
        or candidate.name in {"", ".", ".."}
    ):
        raise RuntimeError("CEFS output path must be repository-relative")
    target = REPOSITORY_ROOT / candidate
    root = REPOSITORY_ROOT.resolve(strict=True)
    try:
        target.resolve(strict=False).relative_to(root)
    except ValueError as error:
        raise RuntimeError("CEFS output path escapes repository") from error
    current = REPOSITORY_ROOT
    for part in candidate.parent.parts:
        current /= part
        if current.is_symlink():
            raise RuntimeError("CEFS output parent contains a symlink")
    return target


def _assert_producer_head(payload: Mapping[str, Any]) -> None:
    producer = payload["authority"]["producer"]
    current = producer_binding()
    if current != producer:
        raise RuntimeError("CEFS producer binding differs from committed code")
    if _git_output("rev-parse", "HEAD") != current["commit"]:
        raise RuntimeError(
            "CEFS missing artifact can only be created at sealed producer HEAD"
        )


def _validate_sealed_producer(payload: Mapping[str, Any]) -> None:
    producer = payload.get("authority", {}).get("producer", {})
    if producer.get("path") != PRODUCER_SCRIPT:
        raise RuntimeError("CEFS producer path differs from frozen script")
    commit = producer.get("commit")
    digest = producer.get("sha256")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise RuntimeError("CEFS producer binding grammar is invalid")
    try:
        sealed = subprocess.run(
            ("git", "show", f"{commit}:{PRODUCER_SCRIPT}"),
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise RuntimeError("CEFS producer commit is not readable") from error
    if hashlib.sha256(sealed).hexdigest() != digest:
        raise RuntimeError("CEFS producer commit bytes do not match binding")


def write_once(
    path: str | Path,
    payload: Mapping[str, Any],
) -> str:
    validate_manifest(payload)
    _validate_sealed_producer(payload)
    output = _output_path(path)
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if output.exists():
        if output.is_symlink() or not output.is_file():
            raise RuntimeError("CEFS write-once target is not a regular file")
        if output.read_bytes() != encoded:
            raise RuntimeError(f"CEFS write-once artifact drift: {output}")
        return hashlib.sha256(encoded).hexdigest()
    _assert_producer_head(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink():
        raise RuntimeError("CEFS output parent contains a symlink")
    try:
        descriptor = os.open(
            output,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o644,
        )
    except FileExistsError:
        if output.is_symlink() or not output.is_file():
            raise RuntimeError("CEFS write-once target is not a regular file")
        if output.read_bytes() != encoded:
            raise RuntimeError(f"CEFS write-once artifact drift: {output}")
        return hashlib.sha256(encoded).hexdigest()
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return hashlib.sha256(encoded).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest()
    artifact_sha256 = write_once(args.output, manifest)
    print(
        json.dumps(
            {
                "output": args.output,
                "artifact_sha256": artifact_sha256,
                "manifest_hash": manifest["manifest_hash"],
                "source_data_rows_parsed": 0,
                "outcomes_opened": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
