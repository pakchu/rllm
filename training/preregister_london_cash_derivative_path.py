"""Seal the outcome-blind LCDP-D1 source/token-support contract.

This module may hash source containers and inspect their physical headers and
source-manifest metadata. It never parses a source data row, funding mark,
execution bar, return, reward, model record, action, trade, PnL, CAGR, or MDD.
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
POLICY_ID = "LCDP-D1"
PROTOCOL_VERSION = "london_cash_derivative_path_preregistration_v1"
DEFAULT_OUTPUT = (
    "results/london_cash_derivative_path_preregistration_2026-07-25.json"
)

PRODUCER_SCRIPT = "training/preregister_london_cash_derivative_path.py"
AUDIT_DOCUMENT = "docs/post-tracer-alpha-mechanism-audit-2026-07-25.md"
AUDIT_DOCUMENT_SHA256 = (
    "7394cd096d92b5469eb625605faaa8f53c49fc486b921269a1b2da0b08afbf9e"
)
BOUNDARY_DOCUMENT = (
    "docs/london-cash-derivative-path-boundary-2026-07-25.md"
)
BOUNDARY_DOCUMENT_SHA256 = (
    "2be38b0181d0269af4159a70070dc3a9eef340a4e277965d09cf082eea848b7e"
)
BOUNDARY_COMMIT = "0fd5e738f27b034a914853145bc8434d5d928502"

COINBASE_SOURCE = "data/coinbase_btcusd_5m_2020_2022.csv.gz"
COINBASE_SHA256 = (
    "07f7a3bddecbbc3724994645b9ac1cd0f391378e0feed421f2c8caa145aab77b"
)
COINBASE_HEADER = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source_complete",
)
COINBASE_HEADER_SHA256 = (
    "056e6938d2dea3e9ef9a9230ca192cbfcf11ea270151115f96cf4e7c94c0de17"
)

BINANCE_SOURCE = "data/coinbase_leadership_binance_5m_2020_2022.csv.gz"
BINANCE_SHA256 = (
    "1a06f1f4dbbdafaf885fb03844426eed5d5bad4aa206fa72b88db2cbd98bef94"
)
BINANCE_HEADER = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "quote_asset_volume",
)
BINANCE_HEADER_SHA256 = (
    "8b70cdf275862b56bcbdd7e10b18c7d82c9cac47dcc3dd2c5ceae78f8f102232"
)

SOURCE_MANIFEST = (
    "results/coinbase_spot_leadership_source_manifest_2026-07-16.json"
)
SOURCE_MANIFEST_SHA256 = (
    "3af321fdcafd0fe6680c4583341b6508124a979fefbf489f8d3376c7ec78a269"
)
SOURCE_MANIFEST_HASH = (
    "243ecba3b9e31548d682084dd5acc2e89c6a24423bce241dd6338a57dd6eefe9"
)

LCLR_PREREGISTRATION = (
    "docs/london-cash-lead-release-preregistration-2026-07-20.md"
)
LCLR_PREREGISTRATION_SHA256 = (
    "fd996475dba37953b1abc0ec29cfe9edbe7d33b91d61d7880f4e0c7ea9330c65"
)
LCLR_REJECTION = (
    "docs/london-cash-lead-release-support-rejection-2026-07-20.md"
)
LCLR_REJECTION_SHA256 = (
    "462a521079ae55076495885516ffe3e6e5dc870a50de7a6f3d310e3026f6d5c6"
)
TRACER_RETIREMENT = "docs/tracer4-source-support-retirement-2026-07-25.md"
TRACER_RETIREMENT_SHA256 = (
    "07f0501f826e2a722e770c616ce5c2698851e7bdcc0cc17d7958c8628bb20135"
)
LAMB_RETIREMENT = "docs/lamb21-source-support-retirement-2026-07-25.md"
LAMB_RETIREMENT_SHA256 = (
    "d73564235ed391fa0678228ef9aeca8a5bfcaa912fe75961030e78b1a68c2e60"
)

LONDON = ZoneInfo("Europe/London")
UTC = ZoneInfo("UTC")
ACTION_SPACE = ("TARGET_SHORT", "TARGET_FLAT", "TARGET_LONG")

TOKEN_SCHEMA: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("calendar_context", ("WEEKDAY", "SATURDAY", "SUNDAY")),
    (
        "daily_alignment",
        (
            "BOTH_RISE",
            "BOTH_FALL",
            "CASH_RISE_PERP_FALL",
            "CASH_FALL_PERP_RISE",
            "RETURN_MIXED_OR_FLAT",
        ),
    ),
    (
        "daily_leader",
        (
            "CASH_LEADS_RISE",
            "CASH_LEADS_FALL",
            "PERP_LEADS_RISE",
            "PERP_LEADS_FALL",
            "NO_CLEAR_LEADER",
        ),
    ),
    (
        "relative_basis_path",
        ("CASH_RICHENS", "CASH_CHEAPENS", "BASIS_ROTATES", "BASIS_FLAT"),
    ),
    (
        "arc_transfer",
        (
            "CASH_LEAD_EXTENDS",
            "CASH_LEAD_REVERSES",
            "PERP_LEAD_EXTENDS",
            "PERP_LEAD_REVERSES",
            "ARC_MIXED",
        ),
    ),
    (
        "path_efficiency",
        ("CASH_CLEANER", "PERP_CLEANER", "BOTH_CHOPPY_OR_TIE"),
    ),
    (
        "range_relation",
        ("CASH_RANGE_DOMINANT", "PERP_RANGE_DOMINANT", "RANGE_BALANCED"),
    ),
    (
        "participation_state",
        (
            "CASH_PARTICIPATION_LOW",
            "CASH_PARTICIPATION_MID",
            "CASH_PARTICIPATION_HIGH",
        ),
    ),
    (
        "participation_transition",
        (
            "CASH_SHARE_RISING",
            "CASH_SHARE_FALLING",
            "CASH_SHARE_STABLE",
            "PARTICIPATION_UNKNOWN",
        ),
    ),
    (
        "alignment_transition",
        (
            "ALIGNMENT_PERSISTS",
            "ALIGNMENT_FLIPS",
            "ALIGNMENT_DISSIPATES",
            "ALIGNMENT_MIXED",
        ),
    ),
    (
        "leader_transition",
        (
            "CASH_LEAD_PERSISTS",
            "PERP_LEAD_PERSISTS",
            "LEAD_ROTATES_TO_CASH",
            "LEAD_ROTATES_TO_PERP",
            "LEAD_MIXED",
        ),
    ),
)
TOKEN_COLUMNS = tuple(field for field, _ in TOKEN_SCHEMA)
PRIMARY_SAFETY_TOKENS = (
    "SOURCE_INVALID",
    "SOURCE_INVALID_START",
    "RANK_UNREADY",
)
CONTROL_TOKENS = (
    "CALENDAR_MASKED",
    "ABLATION_MASKED",
    "CONTROL_UNREADY",
    "CASH_ONLY_RISE",
    "CASH_ONLY_FALL",
    "CASH_ONLY_FLAT",
    "PERP_ONLY_RISE",
    "PERP_ONLY_FALL",
    "PERP_ONLY_FLAT",
)
CONTROL_IDS = (
    "cash_perp_role_swap",
    "cash_stale_one_day",
    "perp_stale_one_day",
    "lag_7_calendar_days",
    "calendar_context_mask",
    "cash_only_language",
    "perp_only_language",
)

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
    source_end_exclusive: str = "2023-01-01"
    boundary_hour_london: int = 16
    inference_minutes: int = 5
    execution_minutes: int = 10
    prior_lookback_days: int = 126
    minimum_prior_valid_days: int = 63
    sequence_lines: int = 21
    minimum_annual_source_valid_share: float = 0.97
    minimum_quarter_source_valid_share: float = 0.95
    minimum_ready_2020: int = 280
    minimum_ready_2021_2022: int = 350
    minimum_ready_post_warmup_quarter: int = 80
    minimum_category_share: float = 0.03
    maximum_category_share: float = 0.94
    minimum_control_difference_share: float = 0.05


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
        raise RuntimeError(
            f"frozen artifact commit mismatch: {path}: {commit}"
        )
    return commit


def producer_binding() -> dict[str, str]:
    commit = assert_committed(PRODUCER_SCRIPT)
    return {
        "path": PRODUCER_SCRIPT,
        "commit": commit,
        "sha256": sha256_file(PRODUCER_SCRIPT),
    }


def london_boundary(day: date, policy: Policy = Policy()) -> datetime:
    return datetime.combine(
        day,
        time(policy.boundary_hour_london),
        tzinfo=LONDON,
    )


def expected_source_slots(day: date, policy: Policy = Policy()) -> int:
    current = london_boundary(day, policy).astimezone(UTC)
    previous = london_boundary(day - timedelta(days=1), policy).astimezone(UTC)
    seconds = int((current - previous).total_seconds())
    if seconds % 300:
        raise RuntimeError("London boundary interval is not five-minute aligned")
    slots = seconds // 300
    if slots not in {276, 288, 300}:
        raise RuntimeError(f"unexpected London source slot count: {slots}")
    return slots


def calendar_slot_counts(policy: Policy = Policy()) -> dict[str, int]:
    start = date.fromisoformat(policy.source_start_inclusive)
    end = date.fromisoformat(policy.source_end_exclusive)
    counts = {"276": 0, "288": 0, "300": 0}
    cursor = start
    while cursor < end:
        counts[str(expected_source_slots(cursor, policy))] += 1
        cursor += timedelta(days=1)
    return counts


def expected_calendar_lines(policy: Policy = Policy()) -> int:
    start = date.fromisoformat(policy.source_start_inclusive)
    end = date.fromisoformat(policy.source_end_exclusive)
    return (end - start).days


def ready_vocabulary() -> dict[str, list[str]]:
    return {field: list(values) for field, values in TOKEN_SCHEMA}


def primary_vocabulary() -> dict[str, list[str]]:
    result = ready_vocabulary()
    for field in TOKEN_COLUMNS[1:]:
        result[field] = [*result[field], *PRIMARY_SAFETY_TOKENS]
    return result


def control_vocabulary() -> dict[str, list[str]]:
    result = primary_vocabulary()
    result["calendar_context"] = [
        *result["calendar_context"],
        "CALENDAR_MASKED",
    ]
    result["daily_alignment"] = [
        *result["daily_alignment"],
        "ABLATION_MASKED",
        "CONTROL_UNREADY",
        "CASH_ONLY_RISE",
        "CASH_ONLY_FALL",
        "CASH_ONLY_FLAT",
        "PERP_ONLY_RISE",
        "PERP_ONLY_FALL",
        "PERP_ONLY_FLAT",
    ]
    for field in TOKEN_COLUMNS[2:]:
        result[field] = [
            *result[field],
            "ABLATION_MASKED",
            "CONTROL_UNREADY",
        ]
    return result


def serialize_line(
    tokens: Mapping[str, str],
    *,
    control: bool = False,
    allow_source_invalid_start: bool = False,
) -> str:
    if tuple(tokens) != TOKEN_COLUMNS:
        raise ValueError("LCDP token fields or order changed")
    vocabulary = control_vocabulary() if control else primary_vocabulary()
    for field, value in tokens.items():
        if value not in vocabulary[field]:
            raise ValueError(f"invalid LCDP token: {field}={value}")
    market_values = [tokens[field] for field in TOKEN_COLUMNS[1:]]
    primary_safety = {
        value for value in market_values if value in PRIMARY_SAFETY_TOKENS
    }
    if primary_safety:
        if len(primary_safety) != 1 or len(set(market_values)) != 1:
            raise ValueError("LCDP primary safety line must be uniform")
        safety_token = market_values[0]
        if (
            safety_token == "SOURCE_INVALID_START"
            and not allow_source_invalid_start
        ):
            raise ValueError(
                "LCDP SOURCE_INVALID_START requires explicit first-line context"
            )
    if control and "CONTROL_UNREADY" in market_values:
        if set(market_values) != {"CONTROL_UNREADY"}:
            raise ValueError("LCDP CONTROL_UNREADY line must be uniform")
    return "|".join(f"{field}={tokens[field]}" for field in TOKEN_COLUMNS)


def safety_line(
    calendar_context: str,
    safety_token: str,
) -> dict[str, str]:
    if calendar_context not in ready_vocabulary()["calendar_context"]:
        raise ValueError("invalid LCDP calendar context")
    if safety_token not in PRIMARY_SAFETY_TOKENS:
        raise ValueError("invalid LCDP safety token")
    if (
        safety_token == "SOURCE_INVALID_START"
        and calendar_context != "WEEKDAY"
    ):
        raise ValueError("LCDP SOURCE_INVALID_START calendar must be WEEKDAY")
    return {
        field: calendar_context if field == "calendar_context" else safety_token
        for field in TOKEN_COLUMNS
    }


def _validate_source_anchors() -> None:
    anchors = {
        AUDIT_DOCUMENT: AUDIT_DOCUMENT_SHA256,
        BOUNDARY_DOCUMENT: BOUNDARY_DOCUMENT_SHA256,
        COINBASE_SOURCE: COINBASE_SHA256,
        BINANCE_SOURCE: BINANCE_SHA256,
        SOURCE_MANIFEST: SOURCE_MANIFEST_SHA256,
        LCLR_PREREGISTRATION: LCLR_PREREGISTRATION_SHA256,
        LCLR_REJECTION: LCLR_REJECTION_SHA256,
        TRACER_RETIREMENT: TRACER_RETIREMENT_SHA256,
        LAMB_RETIREMENT: LAMB_RETIREMENT_SHA256,
    }
    for path, expected in anchors.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"LCDP frozen dependency mismatch: {path}")
    if csv_header(COINBASE_SOURCE) != COINBASE_HEADER:
        raise RuntimeError("LCDP Coinbase physical header mismatch")
    if sha256_csv_header(COINBASE_SOURCE) != COINBASE_HEADER_SHA256:
        raise RuntimeError("LCDP Coinbase header hash mismatch")
    if csv_header(BINANCE_SOURCE) != BINANCE_HEADER:
        raise RuntimeError("LCDP Binance physical header mismatch")
    if sha256_csv_header(BINANCE_SOURCE) != BINANCE_HEADER_SHA256:
        raise RuntimeError("LCDP Binance header hash mismatch")
    manifest = json.loads(repository_path(SOURCE_MANIFEST).read_text())
    required = {
        "start_inclusive": "2020-01-01",
        "end_exclusive": "2023-01-01",
        "manifest_hash": SOURCE_MANIFEST_HASH,
        "historical_snapshot_is_point_in_time": False,
        "future_data_requested": False,
    }
    for field, expected in required.items():
        if manifest.get(field) != expected:
            raise RuntimeError(f"LCDP source manifest mismatch: {field}")
    outputs = manifest.get("outputs", {})
    if outputs.get("coinbase", {}).get("sha256") != COINBASE_SHA256:
        raise RuntimeError("LCDP Coinbase manifest output mismatch")
    if outputs.get("binance", {}).get("sha256") != BINANCE_SHA256:
        raise RuntimeError("LCDP Binance manifest output mismatch")


def _validate_policy(policy: Policy) -> None:
    if policy != Policy():
        raise ValueError("LCDP-D1 policy is frozen")
    if expected_calendar_lines(policy) != 1096:
        raise RuntimeError("LCDP calendar line count drift")
    if calendar_slot_counts(policy) != {"276": 3, "288": 1090, "300": 3}:
        raise RuntimeError("LCDP London DST slot schedule drift")


def _source_contracts() -> dict[str, Any]:
    return {
        "coinbase": {
            "path": COINBASE_SOURCE,
            "sha256": COINBASE_SHA256,
            "physical_header": list(COINBASE_HEADER),
            "header_sha256": COINBASE_HEADER_SHA256,
            "projected_columns": list(COINBASE_HEADER),
        },
        "binance": {
            "path": BINANCE_SOURCE,
            "sha256": BINANCE_SHA256,
            "physical_header": list(BINANCE_HEADER),
            "header_sha256": BINANCE_HEADER_SHA256,
            "projected_columns": list(BINANCE_HEADER),
        },
        "manifest": {
            "path": SOURCE_MANIFEST,
            "sha256": SOURCE_MANIFEST_SHA256,
            "manifest_hash": SOURCE_MANIFEST_HASH,
            "historical_snapshot_is_point_in_time": False,
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
        "phase": "outcome_blind_source_token_support_preregistration",
        "decision": "freeze_source_token_support_only",
        "policy": asdict(policy),
        "authority": {
            "audit_document": AUDIT_DOCUMENT,
            "audit_document_sha256": AUDIT_DOCUMENT_SHA256,
            "boundary_document": BOUNDARY_DOCUMENT,
            "boundary_document_sha256": BOUNDARY_DOCUMENT_SHA256,
            "boundary_commit": BOUNDARY_COMMIT,
            "producer": producer,
        },
        "sources": _source_contracts(),
        "clock": {
            "timezone": "Europe/London",
            "calendar_days": "every date including Saturday and Sunday",
            "boundary": "16:00 local",
            "source_window": "[previous local 16:00,current local 16:00)",
            "expected_five_minute_slots": [276, 288, 300],
            "calendar_slot_counts_2020_2022": calendar_slot_counts(policy),
            "state_complete": "B_D",
            "inference_deadline": "B_D+5m",
            "decision": "B_D+5m",
            "execution": "Binance five-minute open at B_D+10m",
            "target_interval": "[B_D+10m,B_(D+1)+10m)",
            "year_terminal": (
                "December 31 deterministic TARGET_FLAT at B_D+10m; "
                "no next-year outcome opened"
            ),
            "first_line": (
                "2020-01-01 emitted as SOURCE_INVALID_START and counts "
                "invalid in every denominator"
            ),
        },
        "numeric_primitives": {
            "log_path": (
                "x0=log(first open), xi=log(close_i), step_i=xi-x(i-1)"
            ),
            "return": "sum step",
            "path": "sum abs step",
            "efficiency": "abs(return)/path, zero when path is zero",
            "range": "log(max high/min low)",
            "cash_quote": "sum Coinbase volume*close",
            "perp_quote": "sum Binance quote_asset_volume",
            "cash_share": "cash_quote/(cash_quote+perp_quote)",
            "basis": "log(Coinbase price/Binance price) at start and end",
            "arcs": "first and last N/2 ordered steps",
            "prior_reference": (
                "previous 126 emitted lines, current excluded, invalid ignored, "
                "minimum 63 finite valid cash-share values, q1/3 and q2/3"
            ),
        },
        "token_language": {
            "ordered_fields": list(TOKEN_COLUMNS),
            "ready_vocabulary": ready_vocabulary(),
            "primary_vocabulary": primary_vocabulary(),
            "control_vocabulary": control_vocabulary(),
            "serialization": (
                "ordered field=value pairs joined by one ASCII |; no omissions"
            ),
            "sequence": (
                "21 emitted calendar-day lines oldest to newest plus "
                "CURRENT_TARGET"
            ),
            "prompt_forbidden": [
                "calendar date or year",
                "raw timestamp",
                "price or return",
                "volume or notional",
                "rank or quantile",
                "future path or funding",
                "reward or evaluated split statistic",
            ],
            "output_grammar": [
                '{"target":"TARGET_SHORT"}',
                '{"target":"TARGET_FLAT"}',
                '{"target":"TARGET_LONG"}',
            ],
            "action_space": list(ACTION_SPACE),
            "safety_action": "TARGET_FLAT without model invocation",
        },
        "controls": {
            "source_token_control_ids": list(CONTROL_IDS),
            "rebuild_contract": (
                "rebuild every affected primitive, prior quantile, transition, "
                "and sequence; unavailable source remains CONTROL_UNREADY"
            ),
            "later_lclr": {
                "preregistration": LCLR_PREREGISTRATION,
                "preregistration_sha256": LCLR_PREREGISTRATION_SHA256,
                "rejection": LCLR_REJECTION,
                "rejection_sha256": LCLR_REJECTION_SHA256,
                "lclr_exact_policy": (
                    "independent original 16:05 entry, 18:05 exit, 0.5x"
                ),
                "lclr_mask_daily_target": (
                    "same London date mask/side at LCDP B_D+10m; flat otherwise; "
                    "never filters primary"
                ),
            },
        },
        "support_gates": {
            "conjunctive_order": [
                "protocol_source_integrity",
                "calendar_dst_integrity",
                "source_validity",
                "readiness",
                "token_diversity",
                "control_distinctness",
                "append_replay",
                "forbidden_access",
            ],
            "calendar_lines_exact": 1096,
            "source_valid_share_min_year": (
                policy.minimum_annual_source_valid_share
            ),
            "source_valid_share_min_quarter": (
                policy.minimum_quarter_source_valid_share
            ),
            "ready_min_2020": policy.minimum_ready_2020,
            "ready_min_2021": policy.minimum_ready_2021_2022,
            "ready_min_2022": policy.minimum_ready_2021_2022,
            "ready_min_each_quarter_after_2020q1": (
                policy.minimum_ready_post_warmup_quarter
            ),
            "category_share_min": policy.minimum_category_share,
            "category_share_max": policy.maximum_category_share,
            "cash_and_perp_both_directions_each_year": True,
            "control_difference_share_min": (
                policy.minimum_control_difference_share
            ),
            "append_replay_end_exclusive": [
                "2021-01-01",
                "2022-01-01",
                "2023-01-01",
            ],
            "failure_action": "retire_lcdp_d1_unchanged_before_outcomes",
            "pass_action": "authorize_economic_rllm_evaluator_freeze_only",
        },
        "forbidden_access": {
            "source_data_rows_parsed_by_preregistration": 0,
            "funding_rows_opened": 0,
            "execution_or_post_boundary_rows_opened": 0,
            "future_return_rows_built": 0,
            "reward_rows_built": 0,
            "model_rows_built": 0,
            "action_rows_built": 0,
            "trade_rows_built": 0,
            "pnl_cagr_mdd_values_computed": 0,
            "at_or_after_2023_non_date_source_rows_parsed": 0,
        },
        "contingent_economic_chronology": {
            "authorized_now": False,
            "train": "2020 only after evaluator freeze",
            "validation": "2021 only; 2022 remains sealed",
            "test": "single calendar-2022 pass after checkpoint rule freeze",
            "sequential_transfer": ["2023", "2024", "2025", "2026_ytd"],
            "annual_refit": (
                "same frozen algorithm on all strictly prior authorized years"
            ),
            "stop": "first failed source, economic, control, or transfer gate",
        },
        "research_history": {
            "globally_pristine": False,
            "candidate_specific_joint_state_opened": False,
            "lclr_repair_forbidden": True,
            "historical_snapshot_live_equivalent": False,
            "tracer_retirement": TRACER_RETIREMENT,
            "tracer_retirement_sha256": TRACER_RETIREMENT_SHA256,
            "lamb_retirement": LAMB_RETIREMENT,
            "lamb_retirement_sha256": LAMB_RETIREMENT_SHA256,
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
            f"LCDP preregistration contains forbidden keys: {sorted(forbidden)}"
        )
    try:
        producer = normalized["authority"]["producer"]
    except (KeyError, TypeError) as error:
        raise ValueError("LCDP preregistration producer binding missing") from error
    expected = build_manifest(
        producer_binding_override=producer,
        validate_dependencies=False,
        _skip_validation=True,
    )
    if normalized != expected:
        raise ValueError("LCDP preregistration differs from frozen code")
    counters = payload.get("forbidden_access", {})
    if not counters or any(value != 0 for value in counters.values()):
        raise ValueError("LCDP preregistration forbidden counter is nonzero")
    language = payload.get("token_language", {})
    if tuple(language.get("ordered_fields", ())) != TOKEN_COLUMNS:
        raise ValueError("LCDP preregistration token order mismatch")
    if tuple(language.get("action_space", ())) != ACTION_SPACE:
        raise ValueError("LCDP preregistration action space mismatch")
    controls = payload.get("controls", {}).get("source_token_control_ids", ())
    if tuple(controls) != CONTROL_IDS:
        raise ValueError("LCDP preregistration control identity mismatch")


def _output_path(path: str | Path) -> Path:
    candidate = Path(path)
    if (
        str(path).startswith("~")
        or candidate.is_absolute()
        or ".." in candidate.parts
        or candidate.name in {"", ".", ".."}
    ):
        raise RuntimeError("LCDP output path must be repository-relative")
    target = REPOSITORY_ROOT / candidate
    root = REPOSITORY_ROOT.resolve(strict=True)
    try:
        target.resolve(strict=False).relative_to(root)
    except ValueError as error:
        raise RuntimeError("LCDP output path escapes repository") from error
    current = REPOSITORY_ROOT
    for part in candidate.parent.parts:
        current /= part
        if current.is_symlink():
            raise RuntimeError("LCDP output parent contains a symlink")
    return target


def _assert_producer_head(payload: Mapping[str, Any]) -> None:
    producer = payload["authority"]["producer"]
    current = producer_binding()
    if current != producer:
        raise RuntimeError("LCDP producer binding differs from committed code")
    if _git_output("rev-parse", "HEAD") != current["commit"]:
        raise RuntimeError(
            "LCDP missing artifact can only be created at sealed producer HEAD"
        )


def write_once(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    enforce_producer_head: bool = True,
) -> str:
    validate_manifest(payload)
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
            raise RuntimeError("LCDP write-once target is not a regular file")
        if output.read_bytes() != encoded:
            raise RuntimeError(f"LCDP write-once artifact drift: {output}")
        return hashlib.sha256(encoded).hexdigest()
    if enforce_producer_head:
        _assert_producer_head(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink():
        raise RuntimeError("LCDP output parent contains a symlink")
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
            raise RuntimeError("LCDP write-once target is not a regular file")
        if output.read_bytes() != encoded:
            raise RuntimeError(f"LCDP write-once artifact drift: {output}")
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
