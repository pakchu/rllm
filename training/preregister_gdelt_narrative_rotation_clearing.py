"""Preregister the outcome-blind GDELT narrative rotation/clearing family.

This module hashes the frozen GDELT source transport and writes policy metadata.
It must not open GDELT response/count artifacts or any market outcome source.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "gdelt_narrative_rotation_clearing_preregistration_v1"
FAMILY = "GNRC"
AS_OF_DATE = "2026-07-20"
SOURCE_TRANSPORT_COMMIT = "22af253848939855aa456b2e2a5dda02e01a84c5"
SOURCE_BUILDER = Path("training/download_gdelt_bitcoin_narrative_daily.py")
SOURCE_BUILDER_SHA256 = (
    "d756990d979e901033891ad6a8c565783dc58e8a4a9e286d6e866929dd74889e"
)
SOURCE_PROTOCOL_DOCUMENT = Path(
    "docs/gdelt-bitcoin-narrative-source-protocol-2026-07-20.md"
)
SOURCE_PROTOCOL_DOCUMENT_SHA256 = (
    "4dd39d0b0558ed5d7cbdf18a5d3235f80a53b2cdf5758cefeb0406efc7bcb2a6"
)
PREREGISTRATION_SOURCE = Path(
    "training/preregister_gdelt_narrative_rotation_clearing.py"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/gdelt-narrative-rotation-clearing-preregistration-2026-07-20.md"
)
DEFAULT_OUTPUT = Path(
    "results/gdelt_narrative_rotation_clearing_preregistration_2026-07-20.json"
)

SCORE_ARCHETYPES = ("rotation", "clearing", "rule_to_adoption")
WINDOW_PAIRS = ((7, 28), (14, 56))
THRESHOLDS = (0.5, 1.0)
HOLD_DAYS = (3, 7)
UTC = timezone.utc
ENTRY_LAG = timedelta(hours=48, minutes=25)
BAR_INTERVAL = timedelta(minutes=5)
FAMILY_VARIANT_IDS = tuple(
    f"GNRC-{score}-f{fast}s{slow}-t{int(round(threshold * 100)):03d}-h{hold}d"
    for score in SCORE_ARCHETYPES
    for fast, slow in WINDOW_PAIRS
    for threshold in THRESHOLDS
    for hold in HOLD_DAYS
)
OOS_SOURCE_COLUMNS = (
    "date",
    "available_at",
    "global_article_count",
    "broad_article_count",
    "failure_article_count",
    "constraint_article_count",
    "adoption_article_count",
)

EXPECTED_SOURCE_CONTRACT = {
    "protocol_version": "gdelt_bitcoin_narrative_daily_source_v1",
    "endpoint": "https://api.gdeltproject.org/api/v2/doc/doc",
    "mode": "timelinevolraw",
    "format": "json",
    "queries": [
        {"query_id": "broad", "query": "(bitcoin OR cryptocurrency)"},
        {
            "query_id": "failure",
            "query": (
                "(bitcoin OR cryptocurrency) AND (hack OR hacked OR exploit OR "
                "scam OR fraud OR theft OR bankruptcy OR bankrupt OR collapse OR "
                "liquidation OR liquidated)"
            ),
        },
        {
            "query_id": "constraint",
            "query": (
                "(bitcoin OR cryptocurrency) AND (ban OR banned OR regulation OR "
                "regulator OR crackdown OR lawsuit OR investigation)"
            ),
        },
        {
            "query_id": "adoption",
            "query": (
                "(bitcoin OR cryptocurrency) AND (ETF OR institutional OR adoption "
                "OR approval OR approved OR investment)"
            ),
        },
    ],
    "start_date": "2020-01-01",
    "end_date_exclusive": "2024-01-01",
    "windowing": "single_full_half_open_with_api_end_at_last_second",
    "required_date_resolution": "day",
    "availability": "source_date UTC midnight + 48h15m",
}


@dataclass(frozen=True)
class PolicyConfig:
    pseudocount: float = 0.5
    mad_consistency_scale: float = 1.4826
    mad_floor: float = 0.000001
    attention_penalty: float = 0.25
    decision_clock: str = "source_available_at"
    entry_delay_minutes: int = 10
    leverage: float = 1.0
    global_nonoverlap_per_variant: bool = True
    train_start: str = "2021-01-01T00:00:00Z"
    train_end_exclusive: str = "2023-01-01T00:00:00Z"
    selection_start: str = "2023-01-01T00:00:00Z"
    selection_end_exclusive: str = "2024-01-01T00:00:00Z"


FROZEN_CONFIG = PolicyConfig()


@dataclass(frozen=True)
class ScoreDecision:
    source_date: date
    available_at: datetime
    long_score: float
    short_score: float
    evidence_ok: bool = True


@dataclass(frozen=True)
class ScheduledEvent:
    source_date: date
    decision_time: datetime
    entry_time: datetime
    exit_time: datetime
    side: int


@dataclass(frozen=True)
class ScheduleResult:
    eligible_decisions: int
    raw_directional_triggers: int
    side_conflicts: int
    admitted_events: tuple[ScheduledEvent, ...]


@dataclass(frozen=True)
class MarketBar:
    open_time: datetime
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class FundingMark:
    timestamp: datetime
    mark_price: float
    funding_rate: float


SOURCE_EVIDENCE_GATES = {
    "minimum_broad_articles_per_slow_day": 10,
    "minimum_failure_articles_per_slow_window": 3,
    "minimum_constraint_articles_per_slow_window": 3,
    "minimum_adoption_articles_per_slow_window": 3,
    "minimum_nonzero_failure_days_per_slow_window": 2,
    "minimum_nonzero_constraint_days_per_slow_window": 2,
    "minimum_nonzero_adoption_days_per_slow_window": 2,
}

VARIANT_SUPPORT_GATES = {
    "minimum_train_events": 24,
    "minimum_selection_events": 10,
    "minimum_events_each_train_year": 8,
    "minimum_events_each_selection_half": 4,
    "minimum_train_events_each_side": 5,
    "minimum_selection_events_each_side": 2,
    "minimum_active_decision_share": 0.03,
    "maximum_active_decision_share": 0.40,
    "maximum_train_month_share": 0.20,
    "maximum_selection_month_share": 0.30,
}
VARIANT_SUPPORT_CHECK_NAMES = (
    "minimum_train_events",
    "minimum_selection_events",
    "minimum_events_each_train_year",
    "minimum_events_each_selection_half",
    "minimum_train_events_each_side",
    "minimum_selection_events_each_side",
    "train_active_decision_share",
    "selection_active_decision_share",
    "maximum_train_month_share",
    "maximum_selection_month_share",
)

FAMILY_SUPPORT_GATES = {
    "minimum_passing_variants": 8,
    "require_every_score_archetype": True,
    "require_every_window_pair": True,
    "stop_without_repair_if_failed": True,
}

OOS_SOURCE_ACCESS_SEAL = Path("results/gnrc_oos_source_access_seal_2026-07-20.json")
OOS_MARKET_ACCESS_SEAL = Path("results/gnrc_oos_market_access_seal_2026-07-20.json")
OOS_SOURCE_SEAL_FIELDS = (
    "protocol_version",
    "champion_variant_id",
    "champion_policy_hash",
    "selection_report_path",
    "selection_report_sha256",
    "oos_source_builder_path",
    "oos_source_builder_sha256",
    "oos_evaluator_path",
    "oos_evaluator_sha256",
    "oos_source_output_path",
    "source_start",
    "source_end_exclusive",
    "sealed_at",
    "no_interim_oos_access",
)
OOS_MARKET_SEAL_FIELDS = (
    *OOS_SOURCE_SEAL_FIELDS,
    "source_access_seal_path",
    "source_access_seal_sha256",
    "oos_source_output_sha256",
    "oos_source_manifest_path",
    "oos_source_manifest_sha256",
    "oos_source_rows",
    "oos_source_feature_values_inspected",
    "oos_source_outcomes_inspected",
)

ECONOMIC_PROTOCOL = {
    "market": "BTCUSDT USD-M perpetual 5m",
    "bar_grid": (
        "every UTC five-minute bar-open in [split_start, split_end_exclusive); "
        "bar interval is [open_time, open_time+5m)"
    ),
    "initial_equity": 1.0,
    "position_units": "side * equity_before_entry_cost / entry_open_price",
    "entry": "5m open exactly source available_at + 10 minutes",
    "exit": "5m open exactly H calendar days after entry",
    "base_cost": {
        "entry_bps": 2.0,
        "exit_bps": 2.0,
        "deduction": "abs(position_units) * execution_price * side_cost_bps / 10000",
    },
    "stress_cost": {
        "entry_bps": 4.0,
        "exit_bps": 4.0,
        "deduction": "abs(position_units) * execution_price * side_cost_bps / 10000",
    },
    "funding": {
        "required": True,
        "coverage": (
            "exactly one BTCUSDT funding row at every UTC 00:00, 08:00, and "
            "16:00 timestamp in the full split"
        ),
        "timing": "every Binance funding timestamp strictly after entry and at or before exit",
        "cash_change": "-position_units * funding_mark_price * funding_rate",
    },
    "marking": (
        "validate the complete split bar/funding grids; visit every bar open, all "
        "fully-held strict extremes, every close, both execution costs, and each "
        "in-position funding cash flow; flat intervals carry equity"
    ),
    "strict_intrabar_order": (
        "for every fully held bar after entry and strictly before exit, visit favorable "
        "extreme then adverse extreme before close (LONG high then low; SHORT low then "
        "high); exit occurs at the scheduled bar open before that bar's extremes"
    ),
    "absolute_return": "ending_equity / starting_equity - 1",
    "cagr": (
        "(ending_equity / starting_equity) ** (365.2425 / full_calendar_days) - 1"
    ),
    "strict_mdd": (
        "maximum (running_peak_equity - strict_path_equity) / running_peak_equity "
        "over the full calendar 5m path including flat and intratrade marks"
    ),
    "train_qualifiers": {
        "absolute_return_positive": True,
        "minimum_cagr_to_strict_mdd": 1.0,
        "maximum_strict_mdd_percent": 25.0,
        "minimum_trades": 24,
        "stress_absolute_return_positive": True,
    },
    "selection_qualifiers": {
        "absolute_return_positive": True,
        "minimum_cagr_to_strict_mdd": 1.0,
        "maximum_strict_mdd_percent": 20.0,
        "minimum_trades": 10,
    },
    "champion_selection": (
        "among train and selection qualifiers, maximize selection CAGR/strict MDD; "
        "tie by lower selection strict MDD then variant_id"
    ),
    "familywise_test": {
        "method": "one-sided Romano-Wolf max-t circular block bootstrap",
        "selection_period": "2023-01-01 through 2023-12-31 UTC daily net log equity returns",
        "null": "mean daily net log return <= 0",
        "statistic": "sqrt(n)*mean(r)/sample_std(r, ddof=1)",
        "centering": "subtract each variant's 2023 sample mean before resampling",
        "synchronization": "same circular seven-day block indices for every variant",
        "block_days": 7,
        "draws": 100000,
        "seed": 20260720,
        "family_variant_ids": list(FAMILY_VARIANT_IDS),
        "source_unsupported_train_ineligible_or_zero_variance_adjusted_p": 1.0,
        "controls": "diagnostic only and excluded from champion selection",
        "adjusted_p_maximum": 0.10,
    },
    "post_selection_oos": {
        "source_start": "2024-01-01",
        "source_end_exclusive": "2026-07-01",
        "required_source_rows": 912,
        "source_output_columns": list(OOS_SOURCE_COLUMNS),
        "source_grid_validation": (
            "parse the hashed gzip CSV and require exact sorted unique dates plus "
            "the frozen +48h15m availability clock"
        ),
        "market_start": "2024-01-01T00:00:00Z",
        "market_end_exclusive": "2026-07-01T00:00:00Z",
        "evaluation_as_of": "2026-07-20",
        "transport": "identical queries, daily grid, and +48h15m availability clock",
        "source_access_seal": {
            "path": str(OOS_SOURCE_ACCESS_SEAL),
            "required_fields": list(OOS_SOURCE_SEAL_FIELDS),
            "must_be_committed_before": "any 2024+ source request",
        },
        "market_access_seal": {
            "path": str(OOS_MARKET_ACCESS_SEAL),
            "required_fields": list(OOS_MARKET_SEAL_FIELDS),
            "must_be_committed_before": "any 2024+ BTC market or funding read",
        },
        "opening_preconditions": [
            "source-access seal validates real path hashes and is committed before 2024+ news",
            "source is downloaded without feature/outcome inspection",
            "market-access seal binds source seal, 912 rows, source output, and manifest",
            "no interim OOS market inspection",
        ],
        "required_absolute_return_positive": True,
        "minimum_cagr_to_strict_mdd": 3.0,
        "maximum_strict_mdd_percent": 15.0,
        "minimum_trades": 50,
    },
}


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("GNRC timestamp must be UTC")
    return parsed.astimezone(UTC)


def validate_count_row_clock(row: Mapping[str, Any]) -> None:
    source_date = date.fromisoformat(str(row["date"]))
    expected = datetime.combine(
        source_date, datetime.min.time(), tzinfo=UTC
    ) + timedelta(hours=48, minutes=15)
    if parse_utc(str(row["available_at"])) != expected:
        raise ValueError("GNRC source availability clock changed")


def _article_count(row: Mapping[str, Any], field: str) -> int:
    value = row[field]
    if isinstance(value, bool):
        raise ValueError("GNRC article counts must be canonical nonnegative integers")
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise ValueError("GNRC article counts must be canonical nonnegative integers")


def _base_feature(row: Mapping[str, Any], numerator: str, denominator: str) -> float:
    numerator_value = _article_count(row, numerator)
    denominator_value = _article_count(row, denominator)
    return math.log(
        (numerator_value + FROZEN_CONFIG.pseudocount)
        / (denominator_value + FROZEN_CONFIG.pseudocount)
    )


def _scaled_state(
    values: Sequence[float], fast_days: int, slow_days: int
) -> tuple[float, float]:
    if len(values) < slow_days or slow_days < 2 * fast_days:
        raise ValueError("GNRC robust feature history is incomplete")
    slow = list(values[-slow_days:])
    current_fast = list(values[-fast_days:])
    previous_fast = list(values[-2 * fast_days : -fast_days])
    center = median(slow)
    mad = median([abs(value - center) for value in slow])
    scale = max(FROZEN_CONFIG.mad_consistency_scale * mad, FROZEN_CONFIG.mad_floor)
    z_value = (mean(current_fast) - mean(slow)) / scale
    clearing = (mean(previous_fast) - mean(current_fast)) / scale
    return z_value, clearing


def compute_score_state(
    rows: Sequence[Mapping[str, Any]], fast_days: int, slow_days: int
) -> dict[str, Any]:
    if (fast_days, slow_days) not in WINDOW_PAIRS:
        raise ValueError("GNRC window pair is not frozen")
    for row in rows:
        validate_count_row_clock(row)
        global_count = _article_count(row, "global_article_count")
        broad_count = _article_count(row, "broad_article_count")
        category_counts = [
            _article_count(row, field)
            for field in (
                "failure_article_count",
                "constraint_article_count",
                "adoption_article_count",
            )
        ]
        if broad_count > global_count or any(
            count > broad_count for count in category_counts
        ):
            raise ValueError("GNRC query subset counts are inconsistent")
    ordered_dates = [date.fromisoformat(str(row["date"])) for row in rows]
    if ordered_dates != sorted(ordered_dates) or len(set(ordered_dates)) != len(rows):
        raise ValueError("GNRC source dates must be unique and sorted")
    if any(
        following - previous != timedelta(days=1)
        for previous, following in zip(ordered_dates, ordered_dates[1:], strict=False)
    ):
        raise ValueError("GNRC source grid must be daily and complete")
    features = {
        "B": [
            _base_feature(row, "broad_article_count", "global_article_count")
            for row in rows
        ],
        "F": [
            _base_feature(row, "failure_article_count", "broad_article_count")
            for row in rows
        ],
        "C": [
            _base_feature(row, "constraint_article_count", "broad_article_count")
            for row in rows
        ],
        "A": [
            _base_feature(row, "adoption_article_count", "broad_article_count")
            for row in rows
        ],
    }
    states = {
        label: _scaled_state(values, fast_days, slow_days)
        for label, values in features.items()
    }
    z_b, _ = states["B"]
    z_f, clear_f = states["F"]
    z_c, clear_c = states["C"]
    z_a, clear_a = states["A"]
    risk = 0.5 * (z_f + z_c)
    quality = z_a - risk
    attention_penalty = FROZEN_CONFIG.attention_penalty * max(z_b, 0.0)
    scalar_scores = {
        "rotation": quality - attention_penalty,
        "clearing": (
            0.5 * (clear_f + clear_c)
            - clear_a
            + FROZEN_CONFIG.attention_penalty * quality
            - attention_penalty
        ),
    }
    rule_long = min(z_a, z_c) - z_f - attention_penalty
    rule_short = z_f + 0.5 * max(z_c, 0.0) - z_a + attention_penalty
    slow_rows = rows[-slow_days:]
    evidence_ok = bool(
        sum(_article_count(row, "broad_article_count") for row in slow_rows)
        >= SOURCE_EVIDENCE_GATES["minimum_broad_articles_per_slow_day"] * slow_days
        and sum(_article_count(row, "failure_article_count") for row in slow_rows)
        >= SOURCE_EVIDENCE_GATES["minimum_failure_articles_per_slow_window"]
        and sum(_article_count(row, "constraint_article_count") for row in slow_rows)
        >= SOURCE_EVIDENCE_GATES["minimum_constraint_articles_per_slow_window"]
        and sum(_article_count(row, "adoption_article_count") for row in slow_rows)
        >= SOURCE_EVIDENCE_GATES["minimum_adoption_articles_per_slow_window"]
        and sum(_article_count(row, "failure_article_count") > 0 for row in slow_rows)
        >= SOURCE_EVIDENCE_GATES["minimum_nonzero_failure_days_per_slow_window"]
        and sum(
            _article_count(row, "constraint_article_count") > 0 for row in slow_rows
        )
        >= SOURCE_EVIDENCE_GATES["minimum_nonzero_constraint_days_per_slow_window"]
        and sum(_article_count(row, "adoption_article_count") > 0 for row in slow_rows)
        >= SOURCE_EVIDENCE_GATES["minimum_nonzero_adoption_days_per_slow_window"]
    )
    return {
        "source_date": ordered_dates[-1],
        "available_at": parse_utc(str(rows[-1]["available_at"])),
        "evidence_ok": evidence_ok,
        "rotation": {
            "long_score": scalar_scores["rotation"],
            "short_score": -scalar_scores["rotation"],
        },
        "clearing": {
            "long_score": scalar_scores["clearing"],
            "short_score": -scalar_scores["clearing"],
        },
        "rule_to_adoption": {
            "long_score": rule_long,
            "short_score": rule_short,
        },
    }


def expected_split_source_dates(
    *, split_start: datetime, split_end_exclusive: datetime, hold_days: int
) -> tuple[date, ...]:
    if hold_days not in HOLD_DAYS:
        raise ValueError("GNRC hold period is not frozen")
    if (
        split_start.tzinfo is None
        or split_end_exclusive.tzinfo is None
        or split_start.utcoffset() != timedelta(0)
        or split_end_exclusive.utcoffset() != timedelta(0)
        or split_start >= split_end_exclusive
    ):
        raise ValueError("GNRC split boundaries must be ordered UTC timestamps")
    candidate = (split_start - ENTRY_LAG).date()
    while (
        datetime.combine(candidate, datetime.min.time(), tzinfo=UTC) + ENTRY_LAG
        < split_start
    ):
        candidate += timedelta(days=1)
    expected: list[date] = []
    while True:
        entry = datetime.combine(candidate, datetime.min.time(), tzinfo=UTC) + ENTRY_LAG
        if entry + timedelta(days=hold_days) >= split_end_exclusive:
            break
        expected.append(candidate)
        candidate += timedelta(days=1)
    if not expected:
        raise ValueError("GNRC split contains no complete holding interval")
    return tuple(expected)


def schedule_events(
    decisions: Sequence[ScoreDecision],
    *,
    threshold: float,
    hold_days: int,
    split_start: datetime,
    split_end_exclusive: datetime,
) -> ScheduleResult:
    if threshold not in THRESHOLDS or hold_days not in HOLD_DAYS:
        raise ValueError("GNRC scheduler parameter is not frozen")
    expected_dates = expected_split_source_dates(
        split_start=split_start,
        split_end_exclusive=split_end_exclusive,
        hold_days=hold_days,
    )
    ordered = sorted(decisions, key=lambda row: (row.available_at, row.source_date))
    if list(decisions) != ordered or len({row.source_date for row in ordered}) != len(
        ordered
    ):
        raise ValueError("GNRC score decisions must be unique and causally sorted")
    if tuple(row.source_date for row in ordered) != expected_dates:
        raise ValueError(
            "GNRC score decisions must equal the complete split source grid"
        )
    eligible = 0
    raw_triggers = 0
    conflicts = 0
    events: list[ScheduledEvent] = []
    last_exit: datetime | None = None
    for decision in ordered:
        expected_available = datetime.combine(
            decision.source_date, datetime.min.time(), tzinfo=UTC
        ) + timedelta(hours=48, minutes=15)
        if (
            decision.available_at.tzinfo is None
            or decision.available_at.utcoffset() != timedelta(0)
            or decision.available_at != expected_available
        ):
            raise ValueError("GNRC score decision availability changed")
        entry = decision.available_at + timedelta(
            minutes=FROZEN_CONFIG.entry_delay_minutes
        )
        exit_time = entry + timedelta(days=hold_days)
        if (
            not decision.evidence_ok
            or entry < split_start
            or exit_time >= split_end_exclusive
        ):
            continue
        eligible += 1
        long_active = decision.long_score >= threshold
        short_active = decision.short_score >= threshold
        if not long_active and not short_active:
            continue
        raw_triggers += 1
        if long_active and short_active:
            conflicts += 1
            continue
        if last_exit is not None and entry <= last_exit:
            continue
        events.append(
            ScheduledEvent(
                source_date=decision.source_date,
                decision_time=decision.available_at.astimezone(UTC),
                entry_time=entry.astimezone(UTC),
                exit_time=exit_time.astimezone(UTC),
                side=1 if long_active else -1,
            )
        )
        last_exit = exit_time
    return ScheduleResult(eligible, raw_triggers, conflicts, tuple(events))


def support_rates(result: ScheduleResult) -> dict[str, Any]:
    events = result.admitted_events
    month_counts: dict[str, int] = {}
    year_counts: dict[str, int] = {}
    half_counts: dict[str, int] = {}
    for event in events:
        month = event.entry_time.strftime("%Y-%m")
        month_counts[month] = month_counts.get(month, 0) + 1
        year = event.entry_time.strftime("%Y")
        year_counts[year] = year_counts.get(year, 0) + 1
        half = f"{year}-H{1 if event.entry_time.month <= 6 else 2}"
        half_counts[half] = half_counts.get(half, 0) + 1
    return {
        "eligible_decisions": result.eligible_decisions,
        "raw_directional_triggers": result.raw_directional_triggers,
        "side_conflicts": result.side_conflicts,
        "admitted_events": len(events),
        "long_events": sum(event.side == 1 for event in events),
        "short_events": sum(event.side == -1 for event in events),
        "entry_year_counts": dict(sorted(year_counts.items())),
        "entry_half_counts": dict(sorted(half_counts.items())),
        "entry_month_counts": dict(sorted(month_counts.items())),
        "active_decision_share": (
            len(events) / result.eligible_decisions
            if result.eligible_decisions
            else None
        ),
        "maximum_month_share": (
            max(month_counts.values()) / len(events) if events else None
        ),
    }


def evaluate_variant_support(
    train: Mapping[str, Any], selection: Mapping[str, Any]
) -> dict[str, bool]:
    gates = VARIANT_SUPPORT_GATES
    train_share = train["active_decision_share"]
    selection_share = selection["active_decision_share"]
    checks = {
        "minimum_train_events": train["admitted_events"]
        >= gates["minimum_train_events"],
        "minimum_selection_events": selection["admitted_events"]
        >= gates["minimum_selection_events"],
        "minimum_events_each_train_year": min(
            train["entry_year_counts"].get("2021", 0),
            train["entry_year_counts"].get("2022", 0),
        )
        >= gates["minimum_events_each_train_year"],
        "minimum_events_each_selection_half": min(
            selection["entry_half_counts"].get("2023-H1", 0),
            selection["entry_half_counts"].get("2023-H2", 0),
        )
        >= gates["minimum_events_each_selection_half"],
        "minimum_train_events_each_side": min(
            train["long_events"], train["short_events"]
        )
        >= gates["minimum_train_events_each_side"],
        "minimum_selection_events_each_side": min(
            selection["long_events"], selection["short_events"]
        )
        >= gates["minimum_selection_events_each_side"],
        "train_active_decision_share": bool(
            train_share is not None
            and gates["minimum_active_decision_share"]
            <= train_share
            <= gates["maximum_active_decision_share"]
        ),
        "selection_active_decision_share": bool(
            selection_share is not None
            and gates["minimum_active_decision_share"]
            <= selection_share
            <= gates["maximum_active_decision_share"]
        ),
        "maximum_train_month_share": bool(
            train["maximum_month_share"] is not None
            and train["maximum_month_share"] <= gates["maximum_train_month_share"]
        ),
        "maximum_selection_month_share": bool(
            selection["maximum_month_share"] is not None
            and selection["maximum_month_share"]
            <= gates["maximum_selection_month_share"]
        ),
    }
    if tuple(checks) != VARIANT_SUPPORT_CHECK_NAMES:
        raise AssertionError("GNRC variant support check set changed")
    return checks


def evaluate_family_support(
    support_by_variant: Mapping[str, Mapping[str, bool]],
) -> dict[str, Any]:
    if set(support_by_variant) != set(FAMILY_VARIANT_IDS):
        raise ValueError("GNRC family support must contain exactly 24 frozen variants")
    if any(
        tuple(gates) != VARIANT_SUPPORT_CHECK_NAMES
        or any(type(value) is not bool for value in gates.values())
        for gates in support_by_variant.values()
    ):
        raise ValueError("GNRC family support gates are incomplete or non-boolean")
    passing_ids = tuple(
        variant_id
        for variant_id in FAMILY_VARIANT_IDS
        if all(support_by_variant[variant_id].values())
    )
    config_by_id = {row["variant_id"]: row for row in variants()}
    passing_scores = {config_by_id[variant_id]["score"] for variant_id in passing_ids}
    passing_windows = {
        (
            config_by_id[variant_id]["fast_days"],
            config_by_id[variant_id]["slow_days"],
        )
        for variant_id in passing_ids
    }
    checks = {
        "minimum_passing_variants": len(passing_ids)
        >= FAMILY_SUPPORT_GATES["minimum_passing_variants"],
        "every_score_archetype": passing_scores == set(SCORE_ARCHETYPES),
        "every_window_pair": passing_windows == set(WINDOW_PAIRS),
    }
    advances = all(checks.values())
    return {
        "passing_variant_ids": list(passing_ids),
        "passing_variant_count": len(passing_ids),
        "checks": checks,
        "family_advances": advances,
        "decision": "advance_to_market" if advances else "retire_without_repair",
    }


def execution_cost(position_units: float, price: float, side_cost_bps: float) -> float:
    if not all(
        math.isfinite(value) for value in (position_units, price, side_cost_bps)
    ):
        raise ValueError("GNRC execution cost inputs must be finite")
    if price <= 0 or side_cost_bps < 0:
        raise ValueError("GNRC execution cost inputs are invalid")
    return abs(position_units) * price * side_cost_bps / 10_000.0


def funding_cash_change(
    position_units: float, mark_price: float, funding_rate: float
) -> float:
    if not all(
        math.isfinite(value) for value in (position_units, mark_price, funding_rate)
    ):
        raise ValueError("GNRC funding inputs must be finite")
    if mark_price <= 0:
        raise ValueError("GNRC funding mark price must be positive")
    return -position_units * mark_price * funding_rate


def strict_held_bar_prices(
    side: int, high: float, low: float, close: float
) -> tuple[float, ...]:
    if side not in {-1, 1} or not all(
        math.isfinite(value) and value > 0 for value in (high, low, close)
    ):
        raise ValueError("GNRC strict-bar inputs are invalid")
    if low > high or not low <= close <= high:
        raise ValueError("GNRC OHLC ordering is invalid")
    return (high, low, close) if side == 1 else (low, high, close)


def _complete_bar_grid(
    split_start: datetime, split_end_exclusive: datetime
) -> tuple[datetime, ...]:
    if (
        split_start.tzinfo is None
        or split_end_exclusive.tzinfo is None
        or split_start.utcoffset() != timedelta(0)
        or split_end_exclusive.utcoffset() != timedelta(0)
        or split_start >= split_end_exclusive
        or split_start.second
        or split_start.microsecond
        or split_start.minute % 5
        or split_end_exclusive.second
        or split_end_exclusive.microsecond
        or split_end_exclusive.minute % 5
    ):
        raise ValueError("GNRC market split must be an ordered UTC 5m grid")
    count, remainder = divmod(
        int((split_end_exclusive - split_start).total_seconds()),
        int(BAR_INTERVAL.total_seconds()),
    )
    if remainder:
        raise ValueError("GNRC market split is not divisible by five minutes")
    return tuple(split_start + index * BAR_INTERVAL for index in range(count))


def _complete_funding_grid(
    split_start: datetime, split_end_exclusive: datetime
) -> tuple[datetime, ...]:
    _complete_bar_grid(split_start, split_end_exclusive)
    cursor = split_start.replace(minute=0, second=0, microsecond=0)
    while cursor.hour % 8:
        cursor += timedelta(hours=1)
    if cursor < split_start:
        cursor += timedelta(hours=8)
    rows: list[datetime] = []
    while cursor < split_end_exclusive:
        rows.append(cursor)
        cursor += timedelta(hours=8)
    return tuple(rows)


def performance_metrics(
    strict_equity_values: Sequence[float],
    *,
    split_start: datetime,
    split_end_exclusive: datetime,
) -> dict[str, float]:
    _complete_bar_grid(split_start, split_end_exclusive)
    if len(strict_equity_values) < 2:
        raise ValueError("GNRC equity path requires at least two marks")
    if any(not math.isfinite(value) or value <= 0 for value in strict_equity_values):
        raise ValueError("GNRC equity must remain finite and positive")
    calendar_days = (split_end_exclusive - split_start).total_seconds() / 86_400.0
    peak = strict_equity_values[0]
    strict_mdd = 0.0
    for value in strict_equity_values:
        peak = max(peak, value)
        strict_mdd = max(strict_mdd, (peak - value) / peak)
    absolute_return = strict_equity_values[-1] / strict_equity_values[0] - 1.0
    cagr = (strict_equity_values[-1] / strict_equity_values[0]) ** (
        365.2425 / calendar_days
    ) - 1.0
    return {
        "absolute_return": absolute_return,
        "cagr": cagr,
        "strict_mdd": strict_mdd,
        "cagr_to_strict_mdd": (
            cagr / strict_mdd if strict_mdd > 0 else math.inf if cagr > 0 else 0.0
        ),
        "full_calendar_days": calendar_days,
    }


def evaluate_market_path(
    schedule: ScheduleResult,
    bars: Sequence[MarketBar],
    funding_marks: Sequence[FundingMark],
    *,
    split_start: datetime,
    split_end_exclusive: datetime,
    side_cost_bps: float,
) -> dict[str, float | int]:
    expected_bars = _complete_bar_grid(split_start, split_end_exclusive)
    if tuple(bar.open_time for bar in bars) != expected_bars:
        raise ValueError("GNRC market bars must equal the complete UTC 5m grid")
    for bar in bars:
        if (
            not all(
                math.isfinite(value) and value > 0
                for value in (bar.open, bar.high, bar.low, bar.close)
            )
            or not bar.low
            <= min(bar.open, bar.close)
            <= max(bar.open, bar.close)
            <= bar.high
        ):
            raise ValueError("GNRC market bar OHLC is invalid")
    expected_funding = _complete_funding_grid(split_start, split_end_exclusive)
    if tuple(mark.timestamp for mark in funding_marks) != expected_funding:
        raise ValueError("GNRC funding must equal the complete UTC eight-hour grid")
    for mark in funding_marks:
        if not math.isfinite(mark.mark_price) or mark.mark_price <= 0:
            raise ValueError("GNRC funding mark price is invalid")
        if not math.isfinite(mark.funding_rate):
            raise ValueError("GNRC funding rate is invalid")
    if side_cost_bps not in {
        ECONOMIC_PROTOCOL["base_cost"]["entry_bps"],
        ECONOMIC_PROTOCOL["stress_cost"]["entry_bps"],
    }:
        raise ValueError("GNRC side cost must be the frozen base or stress cost")

    events = schedule.admitted_events
    if tuple(events) != tuple(sorted(events, key=lambda event: event.entry_time)):
        raise ValueError("GNRC admitted events must be causally sorted")
    for index, event in enumerate(events):
        expected_decision = datetime.combine(
            event.source_date, datetime.min.time(), tzinfo=UTC
        ) + timedelta(hours=48, minutes=15)
        if (
            event.side not in {-1, 1}
            or event.decision_time != expected_decision
            or event.entry_time != expected_decision + timedelta(minutes=10)
            or event.exit_time - event.entry_time
            not in {timedelta(days=days) for days in HOLD_DAYS}
            or event.entry_time not in expected_bars
            or event.exit_time not in expected_bars
            or (index and event.entry_time <= events[index - 1].exit_time)
        ):
            raise ValueError("GNRC admitted event timing is invalid")

    entries = {event.entry_time: event for event in events}
    exits = {event.exit_time: event for event in events}
    funding_by_time = {mark.timestamp: mark for mark in funding_marks}
    cash = float(ECONOMIC_PROTOCOL["initial_equity"])
    units = 0.0
    active: ScheduledEvent | None = None
    strict_values = [cash]

    def append_equity(price: float) -> None:
        equity = cash + units * price
        if not math.isfinite(equity) or equity <= 0:
            raise ValueError("GNRC strict path reached nonpositive equity")
        strict_values.append(equity)

    for bar in bars:
        append_equity(bar.open)
        funding = funding_by_time.get(bar.open_time)
        if (
            funding is not None
            and active is not None
            and active.entry_time < funding.timestamp <= active.exit_time
        ):
            cash += funding_cash_change(units, funding.mark_price, funding.funding_rate)
            append_equity(bar.open)
        exiting = exits.get(bar.open_time)
        if exiting is not None:
            if active != exiting:
                raise ValueError("GNRC exit does not match the active event")
            cash += units * bar.open
            cash -= execution_cost(units, bar.open, side_cost_bps)
            units = 0.0
            active = None
            append_equity(bar.open)
        entering = entries.get(bar.open_time)
        if entering is not None:
            if active is not None:
                raise ValueError("GNRC entry overlaps the active event")
            equity_before_cost = cash
            units = entering.side * equity_before_cost / bar.open
            cash -= units * bar.open
            cash -= execution_cost(units, bar.open, side_cost_bps)
            active = entering
            append_equity(bar.open)
        if active is None:
            append_equity(bar.close)
        else:
            for price in strict_held_bar_prices(
                active.side, bar.high, bar.low, bar.close
            ):
                append_equity(price)
    if active is not None or units != 0.0:
        raise ValueError("GNRC position remained open at the split boundary")
    metrics = performance_metrics(
        strict_values,
        split_start=split_start,
        split_end_exclusive=split_end_exclusive,
    )
    return {**metrics, "trade_count": len(events)}


def _canonical_seal_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"GNRC OOS seal {field} is not a canonical path")
    candidate = Path(value)
    if (
        candidate.is_absolute()
        or candidate.as_posix() != value
        or ".." in candidate.parts
    ):
        raise ValueError(f"GNRC OOS seal {field} is not a canonical path")
    resolved = repository_path(candidate).resolve()
    if not resolved.is_relative_to(REPOSITORY_ROOT.resolve()):
        raise ValueError(f"GNRC OOS seal {field} escaped the repository")
    return resolved


def _verify_seal_file_hash(
    payload: Mapping[str, Any], path_field: str, hash_field: str
) -> Path:
    path = _canonical_seal_path(payload[path_field], path_field)
    if not path.is_file() or sha256_file(path) != payload[hash_field]:
        raise ValueError(f"GNRC OOS seal {path_field} hash does not match")
    return path


def _validate_oos_source_artifact(path: Path) -> int:
    expected_start = date(2024, 1, 1)
    expected_end = date(2026, 7, 1)
    expected_dates = tuple(
        expected_start + timedelta(days=index)
        for index in range((expected_end - expected_start).days)
    )
    observed_dates: list[date] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != OOS_SOURCE_COLUMNS:
                raise ValueError("GNRC OOS source columns changed")
            for row in reader:
                raw_date = row.get("date")
                raw_available = row.get("available_at")
                if raw_date is None or raw_available is None or None in row:
                    raise ValueError("GNRC OOS source row shape is invalid")
                source_date = date.fromisoformat(raw_date)
                if source_date.isoformat() != raw_date:
                    raise ValueError("GNRC OOS source date is not canonical")
                expected_available = datetime.combine(
                    source_date, datetime.min.time(), tzinfo=UTC
                ) + timedelta(hours=48, minutes=15)
                if parse_utc(raw_available) != expected_available:
                    raise ValueError("GNRC OOS source availability clock changed")
                observed_dates.append(source_date)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise ValueError("GNRC OOS source artifact is unreadable") from error
    if tuple(observed_dates) != expected_dates:
        raise ValueError("GNRC OOS source dates must equal the complete 912-day grid")
    return len(observed_dates)


def validate_oos_seal(
    payload: Mapping[str, Any],
    *,
    stage: str,
    _allow_existing_source_output: bool = False,
) -> None:
    if stage == "source_access":
        required = OOS_SOURCE_SEAL_FIELDS
    elif stage == "market_access":
        required = OOS_MARKET_SEAL_FIELDS
    else:
        raise ValueError("GNRC OOS seal stage is invalid")
    if set(payload) != set(required):
        raise ValueError("GNRC OOS seal fields are incomplete or expanded")
    if payload["protocol_version"] != "gnrc_oos_access_seal_v1":
        raise ValueError("GNRC OOS seal protocol changed")
    if payload["champion_variant_id"] not in set(FAMILY_VARIANT_IDS):
        raise ValueError("GNRC OOS champion is outside the frozen family")
    hash_fields = [
        field
        for field in required
        if field.endswith("_sha256") or field.endswith("_hash")
    ]
    if any(
        not isinstance(payload[field], str)
        or len(payload[field]) != 64
        or any(character not in "0123456789abcdef" for character in payload[field])
        for field in hash_fields
    ):
        raise ValueError("GNRC OOS seal hash is malformed")
    if (
        payload["source_start"] != "2024-01-01"
        or payload["source_end_exclusive"] != "2026-07-01"
    ):
        raise ValueError("GNRC OOS source interval changed")
    parse_utc(str(payload["sealed_at"]))
    if payload["no_interim_oos_access"] is not True:
        raise ValueError("GNRC OOS seal permits interim access")

    selection_report = _verify_seal_file_hash(
        payload, "selection_report_path", "selection_report_sha256"
    )
    _verify_seal_file_hash(
        payload, "oos_source_builder_path", "oos_source_builder_sha256"
    )
    _verify_seal_file_hash(payload, "oos_evaluator_path", "oos_evaluator_sha256")
    try:
        selection_payload = json.loads(selection_report.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("GNRC OOS selection report is not valid JSON") from error
    if (
        not isinstance(selection_payload, Mapping)
        or selection_payload.get("champion_variant_id")
        != payload["champion_variant_id"]
        or selection_payload.get("champion_policy_hash")
        != payload["champion_policy_hash"]
    ):
        raise ValueError("GNRC OOS champion does not match the selection report")
    source_output = _canonical_seal_path(
        payload["oos_source_output_path"], "oos_source_output_path"
    )
    if (
        stage == "source_access"
        and source_output.exists()
        and not _allow_existing_source_output
    ):
        raise ValueError("GNRC OOS source output predates its access seal")

    if stage == "market_access":
        if payload["oos_source_feature_values_inspected"] is not False:
            raise ValueError("GNRC OOS source feature values were inspected")
        if payload["oos_source_outcomes_inspected"] is not False:
            raise ValueError("GNRC OOS source outcomes were inspected")
        if (
            not source_output.is_file()
            or sha256_file(source_output) != payload["oos_source_output_sha256"]
        ):
            raise ValueError("GNRC OOS source output hash does not match")
        observed_source_rows = _validate_oos_source_artifact(source_output)
        if payload["oos_source_rows"] != observed_source_rows:
            raise ValueError("GNRC OOS source row count is invalid")
        _verify_seal_file_hash(
            payload, "oos_source_manifest_path", "oos_source_manifest_sha256"
        )
        if payload["source_access_seal_path"] != str(OOS_SOURCE_ACCESS_SEAL):
            raise ValueError("GNRC OOS source-access seal path changed")
        source_seal_path = _verify_seal_file_hash(
            payload, "source_access_seal_path", "source_access_seal_sha256"
        )
        try:
            source_seal = json.loads(source_seal_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("GNRC OOS source-access seal is not valid JSON") from error
        if not isinstance(source_seal, Mapping):
            raise ValueError("GNRC OOS source-access seal is not an object")
        validate_oos_seal(
            source_seal,
            stage="source_access",
            _allow_existing_source_output=True,
        )
        shared = set(OOS_SOURCE_SEAL_FIELDS) - {"sealed_at"}
        if any(source_seal[field] != payload[field] for field in shared):
            raise ValueError("GNRC OOS market seal diverges from its source seal")
        if parse_utc(str(payload["sealed_at"])) < parse_utc(
            str(source_seal["sealed_at"])
        ):
            raise ValueError("GNRC OOS market seal predates its source seal")


def validate_source_contract() -> dict[str, Any]:
    if sha256_file(SOURCE_BUILDER) != SOURCE_BUILDER_SHA256:
        raise ValueError("GNRC frozen source builder changed")
    if sha256_file(SOURCE_PROTOCOL_DOCUMENT) != SOURCE_PROTOCOL_DOCUMENT_SHA256:
        raise ValueError("GNRC frozen source protocol document changed")
    return json.loads(json.dumps(EXPECTED_SOURCE_CONTRACT, sort_keys=True))


def variant_id(
    score: str, fast_days: int, slow_days: int, threshold: float, hold_days: int
) -> str:
    threshold_code = f"{int(round(threshold * 100)):03d}"
    return f"GNRC-{score}-f{fast_days}s{slow_days}-t{threshold_code}-h{hold_days}d"


def variants() -> list[dict[str, Any]]:
    rows = [
        {
            "variant_id": variant_id(score, fast, slow, threshold, hold),
            "score": score,
            "fast_days": fast,
            "slow_days": slow,
            "threshold": threshold,
            "hold_days": hold,
            "long_rule": "long_score >= threshold",
            "short_rule": "short_score >= threshold",
            "conflict_rule": "flat if both sides trigger",
        }
        for score in SCORE_ARCHETYPES
        for fast, slow in WINDOW_PAIRS
        for threshold in THRESHOLDS
        for hold in HOLD_DAYS
    ]
    if tuple(row["variant_id"] for row in rows) != FAMILY_VARIANT_IDS:
        raise ValueError("GNRC candidate lattice changed")
    return rows


def build_payload() -> dict[str, Any]:
    source_contract = validate_source_contract()
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "family": FAMILY,
        "as_of_date": AS_OF_DATE,
        "decision": "freeze_source_features_and_lattice_before_source_counts",
        "source_transport": {
            "commit": SOURCE_TRANSPORT_COMMIT,
            "builder": str(SOURCE_BUILDER),
            "builder_sha256": SOURCE_BUILDER_SHA256,
            "protocol_document": str(SOURCE_PROTOCOL_DOCUMENT),
            "protocol_document_sha256": SOURCE_PROTOCOL_DOCUMENT_SHA256,
            "contract": source_contract,
            "daily_artifact_opened": False,
            "raw_bundle_opened": False,
        },
        "policy": {
            "config": asdict(FROZEN_CONFIG),
            "base_features": {
                "xB": "log((broad + 0.5) / (global + 0.5))",
                "xF": "log((failure + 0.5) / (broad + 0.5))",
                "xC": "log((constraint + 0.5) / (broad + 0.5))",
                "xA": "log((adoption + 0.5) / (broad + 0.5))",
            },
            "robust_features": {
                "index_sets": {
                    "current_fast": "[t-f+1, t] inclusive",
                    "previous_fast": "[t-2f+1, t-f] inclusive",
                    "slow": "[t-s+1, t] inclusive",
                    "full_history_required": "at least s consecutive daily rows; s >= 2f",
                },
                "mean": "ordinary arithmetic mean",
                "median": "middle observation; average the two middle observations when even",
                "MAD": "median(abs(x - median(slow))) on the same slow index set",
                "Zx": (
                    "(mean_fast(x, including t) - mean_slow(x, including t)) / "
                    "max(1.4826*MAD_slow(x, including t), 1e-6)"
                ),
                "CLRx": (
                    "(mean(x[t-2f+1:t-f]) - mean(x[t-f+1:t])) / "
                    "max(1.4826*MAD_slow(x, including t), 1e-6)"
                ),
                "Risk": "0.5 * (ZF + ZC)",
                "Quality": "ZA - Risk",
                "ClearRisk": "0.5 * (CLRF + CLRC)",
                "ClearAdoption": "CLRA",
            },
            "scores": {
                "rotation": {
                    "long": "Quality - 0.25*max(ZB, 0)",
                    "short": "-(Quality - 0.25*max(ZB, 0))",
                },
                "clearing": {
                    "long": (
                        "ClearRisk - ClearAdoption + 0.25*Quality - 0.25*max(ZB, 0)"
                    ),
                    "short": (
                        "-(ClearRisk - ClearAdoption + 0.25*Quality - 0.25*max(ZB, 0))"
                    ),
                },
                "rule_to_adoption": {
                    "long": "min(ZA, ZC) - ZF - 0.25*max(ZB, 0)",
                    "short": "ZF + 0.5*max(ZC, 0) - ZA + 0.25*max(ZB, 0)",
                },
            },
            "source_clock_assertion": (
                "every row date is consecutive and unique; available_at must equal "
                "source date UTC midnight + 48h15m"
            ),
            "scheduler": {
                "ordering": "ascending (available_at, source_date), unique source_date",
                "required_source_grid": (
                    "exactly every source date whose source_date+48h25m entry and "
                    "full hold satisfy split_start <= entry and exit < split_end"
                ),
                "decision": "source available_at",
                "entry": "decision + 10 minutes",
                "exit": "entry + hold calendar days",
                "split_containment": (
                    "split_start <= entry and exit < split_end_exclusive"
                ),
                "state": "reset flat independently at each split",
                "conflict": "flat when long and short both reach threshold",
                "nonoverlap": (
                    "admit only when entry > previous admitted exit; same-time roll prohibited"
                ),
            },
            "support_denominators": {
                "eligible_decision": (
                    "full feature history, source evidence gate true, and split-contained "
                    "entry/exit, before score and nonoverlap"
                ),
                "raw_directional_trigger": (
                    "eligible decision with either side at threshold before conflict/nonoverlap"
                ),
                "admitted_event": "directional trigger after conflict and nonoverlap",
                "active_decision_share": "admitted events / eligible decisions",
                "calendar_attribution": "UTC entry timestamp",
                "maximum_month_share": (
                    "maximum admitted events in one UTC entry month / admitted events"
                ),
            },
        },
        "variants": variants(),
        "source_evidence_gates": SOURCE_EVIDENCE_GATES,
        "variant_support_gates": VARIANT_SUPPORT_GATES,
        "family_support_gates": FAMILY_SUPPORT_GATES,
        "controls": [
            "simple_quality_score",
            "broad_attention_only",
            "no_attention_penalty",
            "stale_one_source_day",
            "direction_flip",
            "deterministic_random_side_seed_20260720",
        ],
        "control_role": "diagnostic only; excluded from the 24-way champion family",
        "economic_protocol": ECONOMIC_PROTOCOL,
        "rllm_boundary": {
            "part_of_gnrc_primary_claim": False,
            "reason": "abstention and sizing alter realized entries/exposure",
            "required_before_any_rllm_result": [
                "separate overlay preregistration",
                "model prompt parser checkpoint and training data hashes",
                "overlay multiplicity correction",
                "holdout not previously opened by deterministic GNRC evaluation",
            ],
            "gnrc_primary_oos_may_be_reused_for_rllm_claim": False,
            "may_retime_reverse_or_create_events": False,
            "status": "exploratory_only_until_separate_preregistration",
        },
        "outcome_boundary": {
            "gdelt_daily_rows_read": 0,
            "gdelt_raw_responses_read": 0,
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "future_return_rows_read": 0,
            "return_or_pnl_fields_read": 0,
            "outcomes_opened": False,
        },
        "failure_action": "retire_family_without_threshold_or_sign_repair",
    }
    payload["preregistration_source"] = str(PREREGISTRATION_SOURCE)
    payload["preregistration_source_sha256"] = sha256_file(PREREGISTRATION_SOURCE)
    payload["preregistration_document"] = str(PREREGISTRATION_DOCUMENT)
    payload["preregistration_document_sha256"] = sha256_file(PREREGISTRATION_DOCUMENT)
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def write_once(path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    destination = repository_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    try:
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except FileExistsError as error:
        raise FileExistsError(
            f"GNRC preregistration is write-once: {destination}"
        ) from error
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(
        json.dumps(write_once(args.output), indent=2, sort_keys=True, allow_nan=False)
    )


if __name__ == "__main__":
    main()
