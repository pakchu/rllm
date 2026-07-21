"""Evaluate DEWH-144 source support and pure-clock novelty without outcomes."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

import numpy as np
import pandas as pd

from training.evaluate_trollbox_semantic_disagreement_resolution_novelty import (
    ClockRow,
    exact_entry_jaccard,
    exposure_metrics,
    maximum_tolerant_matches,
)
from training.freeze_dewh_comparator_cohort import (
    DEHR_CLOCK,
    LIVE_CLOCK,
    MFIC_CLOCK,
    REQUIRED_MEMBERS,
    _load_afcs,
    _load_bafr,
    _load_json,
    _load_pure_clock,
)


POLICY_ID = "DEWH-144"
PROTOCOL_VERSION = "deribit_expiry_wall_handoff_source_gate_v1"
BAR = timedelta(minutes=5)
LOOKBACK_DAYS = 365
MINIMUM_PRIOR_EXPIRIES = 180
HOLD_BARS = 144
IMPLEMENTATION = Path("training/evaluate_deribit_expiry_wall_handoff_support.py")
MECHANISM_DOCUMENT = Path(
    "docs/deribit-expiry-wall-handoff-mechanism-decision-2026-07-21.md"
)
MECHANISM_DOCUMENT_SHA256 = (
    "f5b378b75e3d32c18e32b245f62674a7a7b25f90ec7761d865ddb6c627a93ce8"
)
SOURCE = Path(
    "data/deribit_btc_expiry_wall_2019_2023/"
    "BTC_deribit_expiry_wall_2019-01-01_2023-12-31.csv.gz"
)
SOURCE_SHA256 = "53e8c829d8dd49eb669218067409a1b5175900c88fd75652c0ad420f6b6167f5"
SOURCE_MANIFEST = Path("data/deribit_btc_expiry_wall_2019_2023/build_manifest.json")
SOURCE_MANIFEST_SHA256 = (
    "dde10a20d6efc3026be253daefe88bff0bae4ba379deaa90b4d08431dd741c36"
)
SOURCE_MANIFEST_HASH = (
    "dbecf89849c356e4b5900600e2727ad5f972c9a62487b8d897407a0e22da104e"
)
COMPARATOR_FREEZE = Path("results/dewh_comparator_cohort_freeze_2026-07-21.json")
COMPARATOR_FREEZE_SHA256 = (
    "f404e44c376cb39f145f54a09def8cf5d516ce7bbc624c30ebfb26c140a75e6e"
)
COMPARATOR_FREEZE_MANIFEST_HASH = (
    "8f599ca5dfdf0615b15b932d6d13247e5e62564506d3c2acfbfe7fcc07752aab"
)
DEFAULT_REPORT = Path("results/deribit_expiry_wall_handoff_source_gate_2026-07-21.json")
DEFAULT_CLOCK = Path("results/dewh_pure_clocks_2026-07-21.csv.gz")
TRAIN_START = datetime(2020, 7, 1, tzinfo=timezone.utc)
TRAIN_END = datetime(2023, 1, 1, tzinfo=timezone.utc)
SELECTION_END = datetime(2024, 1, 1, tzinfo=timezone.utc)
SPLITS = (
    ("train", TRAIN_START, TRAIN_END),
    ("selection", TRAIN_END, SELECTION_END),
)
SOURCE_COLUMNS = (
    "expiry_time",
    "delivery_event_time",
    "source_observation_earliest",
    "index_price",
    "distinct_strike_count",
    "total_position",
    "dominant_strike",
    "dominant_strike_position",
    "wall_share",
    "strike_position_hhi",
    "largest_individual_instrument_share",
    "local_log_spacing",
    "signed_normalized_wall_distance",
    "wall_tie_count",
    "delivery_delay_seconds",
    "maximum_event_row_span_seconds",
)
CLOCK_FIELDS = (
    "candidate_id",
    "split",
    "causal_origin",
    "decision_time",
    "availability_time",
    "entry_time",
    "exit_time",
    "side",
)


@dataclass(frozen=True)
class Candidate:
    causal_origin: datetime
    delivery_time: datetime
    decision_time: datetime
    availability_time: datetime
    entry_time: datetime
    exit_time: datetime
    side: int
    split: str | None = None


@dataclass(frozen=True)
class ScheduleAudit:
    raw_candidates: int
    split_contained_candidates: int
    split_boundary_drops: int
    overlap_suppressions: int
    accepted_candidates: int


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


def normalized_entry(observation: datetime) -> datetime:
    if observation.tzinfo is None or observation.utcoffset() != timedelta(0):
        raise ValueError("DEWH observation must be UTC")
    timestamp = cast(pd.Timestamp, pd.Timestamp(observation))
    boundary = cast(pd.Timestamp, timestamp.ceil("5min"))
    return cast(pd.Timestamp, boundary + pd.Timedelta(minutes=5)).to_pydatetime()


def strict_prior_calendar_midrank(
    timestamps: Sequence[datetime],
    values: Sequence[float],
    *,
    lookback_days: int = LOOKBACK_DAYS,
    minimum: int = MINIMUM_PRIOR_EXPIRIES,
) -> np.ndarray:
    if len(timestamps) != len(values):
        raise ValueError("DEWH rank timestamps and values differ in length")
    if lookback_days < 1 or minimum < 1:
        raise ValueError("DEWH rank window and minimum must be positive")
    output = np.full(len(values), np.nan, dtype=float)
    left = 0
    for index, (timestamp, value) in enumerate(zip(timestamps, values)):
        if index and timestamp <= timestamps[index - 1]:
            raise ValueError("DEWH source clock is not strictly chronological")
        boundary = timestamp - timedelta(days=lookback_days)
        while left < index and timestamps[left] < boundary:
            left += 1
        prior = np.asarray(values[left:index], dtype=float)
        if len(prior) < minimum:
            continue
        output[index] = float(
            ((prior < value).sum() + 0.5 * (prior == value).sum()) / len(prior)
        )
    return output


def load_source() -> tuple[pd.DataFrame, dict[str, Any]]:
    bindings = {
        MECHANISM_DOCUMENT: MECHANISM_DOCUMENT_SHA256,
        SOURCE: SOURCE_SHA256,
        SOURCE_MANIFEST: SOURCE_MANIFEST_SHA256,
        COMPARATOR_FREEZE: COMPARATOR_FREEZE_SHA256,
    }
    for path, expected in bindings.items():
        if sha256_file(path) != expected:
            raise ValueError(f"DEWH source-gate binding changed: {path}")
    manifest = _load_json(SOURCE_MANIFEST)
    if manifest.get("manifest_hash") != SOURCE_MANIFEST_HASH:
        raise ValueError("DEWH source manifest hash changed")
    if manifest.get("candidate_incidence_computed") is not False:
        raise ValueError("DEWH source manifest opened candidate incidence")
    if manifest.get("aggregate", {}).get("sha256") != SOURCE_SHA256:
        raise ValueError("DEWH source aggregate binding changed")
    if manifest.get("aggregate", {}).get("columns") != list(SOURCE_COLUMNS):
        raise ValueError("DEWH source manifest schema changed")
    if (
        manifest.get("outcome_boundary", {}).get("economic_outcomes_computed")
        is not False
    ):
        raise ValueError("DEWH source manifest computed outcomes")

    frame = pd.read_csv(
        SOURCE,
        compression="gzip",
        usecols=cast(Any, list(SOURCE_COLUMNS)),
    )
    if list(frame.columns) != list(SOURCE_COLUMNS) or len(frame) != 1484:
        raise ValueError("DEWH source frame identity changed")
    for column in (
        "expiry_time",
        "delivery_event_time",
        "source_observation_earliest",
    ):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    if (
        frame["expiry_time"].duplicated().any()
        or not frame["expiry_time"].is_monotonic_increasing
    ):
        raise ValueError("DEWH source expiry clock is invalid")
    timestamp_columns = {
        "expiry_time",
        "delivery_event_time",
        "source_observation_earliest",
    }
    numeric = frame[
        [column for column in SOURCE_COLUMNS if column not in timestamp_columns]
    ].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("DEWH source contains non-finite values")
    if bool(frame["distinct_strike_count"].lt(3).any()):
        raise ValueError("DEWH source retained an invalid strike set")
    if bool(frame["wall_tie_count"].ne(1).any()):
        raise ValueError("DEWH source retained a tied wall")
    if bool(frame["local_log_spacing"].le(0.0).any()):
        raise ValueError("DEWH source retained invalid strike spacing")
    if bool(frame["wall_share"].le(0.0).any()) or bool(
        frame["wall_share"].gt(1.0).any()
    ):
        raise ValueError("DEWH wall share escaped (0, 1]")
    expected_observation = frame["delivery_event_time"] + pd.Timedelta(minutes=65)
    if not frame["source_observation_earliest"].equals(expected_observation):
        raise ValueError("DEWH source observation clock changed")

    dates = [
        cast(pd.Timestamp, value).to_pydatetime()
        for value in cast(pd.Series, frame["expiry_time"])
    ]
    frame["rank_total_position"] = strict_prior_calendar_midrank(
        dates,
        [float(value) for value in cast(pd.Series, frame["total_position"])],
    )
    frame["rank_wall_share"] = strict_prior_calendar_midrank(
        dates,
        [float(value) for value in cast(pd.Series, frame["wall_share"])],
    )
    frame["rank_largest_instrument_share"] = strict_prior_calendar_midrank(
        dates,
        [
            float(value)
            for value in cast(pd.Series, frame["largest_individual_instrument_share"])
        ],
    )
    frame["rank_ready"] = (
        frame[["rank_total_position", "rank_wall_share"]].notna().all(axis=1)
    )
    return frame, {
        "rows": len(frame),
        "first_expiry": dates[0].isoformat(),
        "last_expiry": dates[-1].isoformat(),
        "rank_ready_rows": int(frame["rank_ready"].sum()),
        "source_columns_read": list(SOURCE_COLUMNS),
        "forbidden_source_columns_read": 0,
    }


def _primary_mask(frame: pd.DataFrame) -> np.ndarray:
    distance = cast(pd.Series, frame["signed_normalized_wall_distance"]).abs()
    return (
        frame["rank_ready"]
        & frame["rank_total_position"].ge(0.50)
        & frame["rank_wall_share"].ge(0.70)
        & distance.ge(0.25)
        & distance.le(1.00)
        & distance.ne(0.0)
    ).to_numpy(bool)


def _split(candidate: Candidate) -> str | None:
    for name, start, end in SPLITS:
        if (
            candidate.causal_origin >= start
            and candidate.delivery_time >= start
            and candidate.decision_time >= start
            and candidate.availability_time >= start
            and candidate.entry_time >= start
            and candidate.exit_time <= end
        ):
            return name
    return None


def build_candidates(
    frame: pd.DataFrame,
    mask: np.ndarray,
    sides: np.ndarray,
    *,
    extra_entry_bars: int = 0,
) -> list[Candidate]:
    if extra_entry_bars < 0:
        raise ValueError("DEWH extra entry delay cannot be negative")
    rows: list[Candidate] = []
    for index in np.flatnonzero(mask):
        side = int(sides[index])
        if side not in {-1, 1}:
            continue
        expiry = cast(pd.Timestamp, frame.iloc[index]["expiry_time"]).to_pydatetime()
        delivery = cast(
            pd.Timestamp, frame.iloc[index]["delivery_event_time"]
        ).to_pydatetime()
        observation = cast(
            pd.Timestamp, frame.iloc[index]["source_observation_earliest"]
        ).to_pydatetime()
        entry = normalized_entry(observation) + extra_entry_bars * BAR
        rows.append(
            Candidate(
                causal_origin=expiry,
                delivery_time=delivery,
                decision_time=observation,
                availability_time=observation,
                entry_time=entry,
                exit_time=entry + HOLD_BARS * BAR,
                side=side,
            )
        )
    return rows


def schedule_candidates(
    candidates: Iterable[Candidate],
) -> tuple[list[Candidate], ScheduleAudit]:
    ordered = sorted(candidates, key=lambda row: (row.entry_time, row.side))
    contained = [replace(row, split=split) for row in ordered if (split := _split(row))]
    accepted: list[Candidate] = []
    prior_exit = datetime.min.replace(tzinfo=timezone.utc)
    overlaps = 0
    for row in contained:
        if row.side not in {-1, 1}:
            raise ValueError("DEWH candidate side changed")
        if (
            row.entry_time.second
            or row.entry_time.microsecond
            or row.entry_time.minute % 5
        ):
            raise ValueError("DEWH entry left the five-minute grid")
        if row.exit_time - row.entry_time != HOLD_BARS * BAR:
            raise ValueError("DEWH hold changed")
        if row.entry_time < prior_exit:
            overlaps += 1
            continue
        accepted.append(row)
        prior_exit = row.exit_time
    return accepted, ScheduleAudit(
        raw_candidates=len(ordered),
        split_contained_candidates=len(contained),
        split_boundary_drops=len(ordered) - len(contained),
        overlap_suppressions=overlaps,
        accepted_candidates=len(accepted),
    )


def _summary(candidates: Sequence[Candidate], split: str) -> dict[str, Any]:
    rows = [row for row in candidates if row.split == split]
    total = len(rows)
    months = Counter(
        f"{row.entry_time.year:04d}-{row.entry_time.month:02d}" for row in rows
    )
    weekdays = Counter(str(row.entry_time.weekday()) for row in rows)
    return {
        "accepted_events": total,
        "side_counts": {
            "LONG": sum(row.side == 1 for row in rows),
            "SHORT": sum(row.side == -1 for row in rows),
        },
        "year_counts": dict(
            sorted(Counter(str(row.entry_time.year) for row in rows).items())
        ),
        "half_counts": dict(
            sorted(
                Counter(
                    f"{row.entry_time.year}-H{1 if row.entry_time.month <= 6 else 2}"
                    for row in rows
                ).items()
            )
        ),
        "quarter_counts": dict(
            sorted(
                Counter(
                    f"{row.entry_time.year}-Q{(row.entry_time.month - 1) // 3 + 1}"
                    for row in rows
                ).items()
            )
        ),
        "active_months": len(months),
        "maximum_calendar_month_share": max(months.values()) / total if total else 0.0,
        "maximum_utc_entry_weekday_share": max(weekdays.values()) / total
        if total
        else 0.0,
    }


def _clock_rows(candidates: Sequence[Candidate]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": "dewh:primary",
            "split": row.split,
            "causal_origin": row.causal_origin.isoformat(),
            "decision_time": row.decision_time.isoformat(),
            "availability_time": row.availability_time.isoformat(),
            "entry_time": row.entry_time.isoformat(),
            "exit_time": row.exit_time.isoformat(),
            "side": row.side,
        }
        for row in candidates
    ]


def _control_report(
    accepted: Sequence[Candidate], audit: ScheduleAudit | None
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "clock_hash": canonical_hash(_clock_rows(accepted)),
        "splits": {name: _summary(accepted, name) for name, _, _ in SPLITS},
    }
    if audit is not None:
        report["schedule_audit"] = asdict(audit)
    return report


def support_gate(primary: Mapping[str, Any]) -> dict[str, Any]:
    train = primary["splits"]["train"]
    selection = primary["splits"]["selection"]
    checks = {
        "train_total_between_60_and_240": 60 <= train["accepted_events"] <= 240,
        "train_2020h2_at_least_8": train["half_counts"].get("2020-H2", 0) >= 8,
        "train_2021_at_least_18": train["year_counts"].get("2021", 0) >= 18,
        "train_2022_at_least_18": train["year_counts"].get("2022", 0) >= 18,
        "train_each_half_at_least_8": all(
            train["half_counts"].get(label, 0) >= 8
            for label in ("2020-H2", "2021-H1", "2021-H2", "2022-H1", "2022-H2")
        ),
        "train_long_at_least_20": train["side_counts"]["LONG"] >= 20,
        "train_short_at_least_20": train["side_counts"]["SHORT"] >= 20,
        "train_active_months_at_least_24": train["active_months"] >= 24,
        "train_month_share_at_most_0_15": train["maximum_calendar_month_share"] <= 0.15,
        "train_weekday_share_at_most_0_30": train["maximum_utc_entry_weekday_share"]
        <= 0.30,
        "selection_total_between_20_and_100": 20 <= selection["accepted_events"] <= 100,
        "selection_each_half_at_least_8": all(
            selection["half_counts"].get(f"2023-H{half}", 0) >= 8 for half in (1, 2)
        ),
        "selection_each_quarter_at_least_3": all(
            selection["quarter_counts"].get(f"2023-Q{quarter}", 0) >= 3
            for quarter in range(1, 5)
        ),
        "selection_long_at_least_6": selection["side_counts"]["LONG"] >= 6,
        "selection_short_at_least_6": selection["side_counts"]["SHORT"] >= 6,
        "selection_active_months_at_least_8": selection["active_months"] >= 8,
        "selection_month_share_at_most_0_25": selection["maximum_calendar_month_share"]
        <= 0.25,
        "selection_weekday_share_at_most_0_35": selection[
            "maximum_utc_entry_weekday_share"
        ]
        <= 0.35,
    }
    return {"checks": checks, "passed": all(checks.values())}


def _deterministic_side(entry: datetime) -> int:
    digest = hashlib.sha256(
        f"DEWH-144-random-side-20260721|{entry.isoformat()}".encode("ascii")
    ).digest()
    return 1 if digest[0] < 128 else -1


def _load_comparators() -> dict[str, list[ClockRow]]:
    freeze = _load_json(COMPARATOR_FREEZE)
    if freeze.get("manifest_hash") != COMPARATOR_FREEZE_MANIFEST_HASH:
        raise ValueError("DEWH comparator freeze manifest changed")
    if freeze.get("required_members") != list(REQUIRED_MEMBERS):
        raise ValueError("DEWH comparator freeze membership changed")
    mfic_schema = [
        "candidate_id",
        "split",
        "causal_origin",
        "decision_time",
        "entry_time",
        "exit_time",
        "side",
    ]
    live_schema = [
        "candidate_id",
        "split",
        "decision_time",
        "entry_time",
        "exit_time",
        "side",
    ]
    dehr_schema = [
        "candidate_id",
        "causal_origin",
        "decision_time",
        "original_entry_time",
        "entry_time",
        "exit_time",
        "side",
    ]
    tuples = [
        *_load_afcs(),
        *_load_bafr(),
        *_load_pure_clock(MFIC_CLOCK, expected_schema=mfic_schema),
        *_load_pure_clock(LIVE_CLOCK, expected_schema=live_schema),
        *_load_pure_clock(DEHR_CLOCK, expected_schema=dehr_schema),
    ]
    grouped: dict[str, list[ClockRow]] = {}
    for candidate_id, entry, exit_time, side in tuples:
        grouped.setdefault(candidate_id, []).append(
            ClockRow(
                candidate_id=candidate_id,
                split="comparator",
                causal_origin=entry,
                decision_time=entry,
                entry_time=entry,
                exit_time=exit_time,
                side=side,
            )
        )
    if set(grouped) != set(REQUIRED_MEMBERS):
        raise ValueError("DEWH comparator member set changed")
    return grouped


def novelty_gate(
    primary: Sequence[Candidate],
) -> tuple[dict[str, Any], dict[str, Any], int]:
    clocks = [
        ClockRow(
            candidate_id="dewh:primary",
            split=cast(str, row.split),
            causal_origin=row.causal_origin,
            decision_time=row.decision_time,
            entry_time=row.entry_time,
            exit_time=row.exit_time,
            side=row.side,
        )
        for row in primary
    ]
    metrics: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    comparator_rows = 0
    for candidate_id, comparator in sorted(_load_comparators().items()):
        eligible = [row for row in comparator if row.entry_time < SELECTION_END]
        comparator_rows += len(eligible)
        start = max(TRAIN_START, min(row.entry_time for row in eligible))
        end = min(SELECTION_END, max(row.exit_time for row in eligible))
        left = [
            row
            for row in clocks
            if start <= row.entry_time < end and row.exit_time <= end
        ]
        right = [
            row
            for row in eligible
            if start <= row.entry_time < end and row.exit_time <= end
        ]
        if not left or not right:
            raise ValueError(f"empty DEWH novelty common coverage: {candidate_id}")
        jaccard, exact_matches = exact_entry_jaccard(left, right)
        tolerant_matches = maximum_tolerant_matches(left, right)
        correlation, position_jaccard = exposure_metrics(
            left, right, start=start, end=end
        )
        item = {
            "common_start": start.isoformat(),
            "common_end_exclusive": end.isoformat(),
            "dewh_events": len(left),
            "comparator_events": len(right),
            "exact_entry_matches": exact_matches,
            "exact_entry_jaccard": jaccard,
            "exact_entry_dewh_coverage": exact_matches / len(left),
            "maximum_one_to_one_matches_within_6h": tolerant_matches,
            "dewh_tolerant_match_coverage": tolerant_matches / len(left),
            "signed_occupied_exposure_correlation": correlation,
            "position_bar_jaccard": position_jaccard,
        }
        metrics[candidate_id] = item
        prefix = candidate_id.replace(":", "_")
        if candidate_id == "dehr:dehr_72_normalized":
            checks[f"{prefix}_exact_coverage_at_most_0_50"] = (
                item["exact_entry_dewh_coverage"] <= 0.50
            )
            checks[f"{prefix}_absolute_exposure_correlation_at_most_0_50"] = (
                abs(correlation) <= 0.50
            )
        else:
            checks[f"{prefix}_exact_jaccard_at_most_0_20"] = jaccard <= 0.20
            checks[f"{prefix}_tolerant_coverage_at_most_0_35"] = (
                item["dewh_tolerant_match_coverage"] <= 0.35
            )
            checks[f"{prefix}_absolute_exposure_correlation_at_most_0_40"] = (
                abs(correlation) <= 0.40
            )
    return metrics, {"checks": checks, "passed": all(checks.values())}, comparator_rows


def _clock_bytes(rows: Sequence[Candidate]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(CLOCK_FIELDS)
    for row in rows:
        writer.writerow(
            [
                "dewh:primary",
                row.split,
                row.causal_origin.isoformat(),
                row.decision_time.isoformat(),
                row.availability_time.isoformat(),
                row.entry_time.isoformat(),
                row.exit_time.isoformat(),
                row.side,
            ]
        )
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as handle:
        handle.write(text.getvalue().encode("utf-8"))
    return output.getvalue()


def build_outputs() -> tuple[dict[str, Any], bytes | None]:
    frame, source_audit = load_source()
    primary_mask = _primary_mask(frame)
    wall_side = np.sign(
        cast(pd.Series, frame["signed_normalized_wall_distance"]).to_numpy(float)
    ).astype(np.int8)
    raw_primary = build_candidates(frame, primary_mask, wall_side)
    primary, primary_audit = schedule_candidates(raw_primary)
    primary_report = _control_report(primary, primary_audit)
    support = support_gate(primary_report)

    distance = cast(pd.Series, frame["signed_normalized_wall_distance"]).abs()
    rank_ready = cast(pd.Series, frame["rank_ready"])
    total_gate = cast(pd.Series, frame["rank_total_position"]).ge(0.50)
    wall_gate = cast(pd.Series, frame["rank_wall_share"]).ge(0.70)
    instrument_gate = cast(pd.Series, frame["rank_largest_instrument_share"]).ge(0.70)
    distance_gate = distance.ge(0.25) & distance.le(1.00) & distance.ne(0.0)

    control_specs = {
        "wall_concentration_gate_ablation": rank_ready & total_gate & distance_gate,
        "total_position_gate_ablation": rank_ready & wall_gate & distance_gate,
        "normalized_distance_band_ablation": (
            rank_ready & total_gate & wall_gate & distance.ne(0.0)
        ),
        "largest_instrument_concentration": (
            rank_ready & total_gate & instrument_gate & distance_gate
        ),
        "expiry_time_only": pd.Series(True, index=frame.index),
    }
    controls: dict[str, Any] = {}
    for name, mask in control_specs.items():
        raw = build_candidates(frame, mask.to_numpy(bool), wall_side)
        accepted, audit = schedule_candidates(raw)
        controls[name] = _control_report(accepted, audit)

    direction_flip = [replace(row, side=-row.side) for row in primary]
    random_side = [
        replace(row, side=_deterministic_side(row.entry_time)) for row in primary
    ]
    alternating_side = [
        replace(row, side=1 if index % 2 == 0 else -1)
        for index, row in enumerate(primary)
    ]
    delayed_raw = build_candidates(frame, primary_mask, wall_side, extra_entry_bars=1)
    delayed, delayed_audit = schedule_candidates(delayed_raw)
    controls.update(
        {
            "direction_flip": _control_report(direction_flip, None),
            "deterministic_random_side": _control_report(random_side, None),
            "fixed_alternating_side": _control_report(alternating_side, None),
            "one_bar_execution_delay": _control_report(delayed, delayed_audit),
        }
    )

    if support["passed"]:
        novelty_metrics, novelty, comparator_rows = novelty_gate(primary)
        clock_bytes: bytes | None = _clock_bytes(primary)
    else:
        novelty_metrics = {}
        novelty = {
            "checks": {},
            "passed": False,
            "skipped_reason": "source support failed",
        }
        comparator_rows = 0
        clock_bytes = None
    passed = bool(support["passed"] and novelty["passed"])
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
        "source_binding": {
            "path": str(SOURCE),
            "sha256": SOURCE_SHA256,
            "manifest": str(SOURCE_MANIFEST),
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
            "manifest_hash": SOURCE_MANIFEST_HASH,
        },
        "comparator_freeze_binding": {
            "path": str(COMPARATOR_FREEZE),
            "sha256": COMPARATOR_FREEZE_SHA256,
            "manifest_hash": COMPARATOR_FREEZE_MANIFEST_HASH,
        },
        "configuration": {
            "lookback_days": LOOKBACK_DAYS,
            "minimum_prior_expiries": MINIMUM_PRIOR_EXPIRIES,
            "total_position_rank_minimum": 0.50,
            "wall_share_rank_minimum": 0.70,
            "absolute_normalized_distance_minimum": 0.25,
            "absolute_normalized_distance_maximum": 1.00,
            "hold_bars": HOLD_BARS,
            "exposure": 0.5,
            "train_start": TRAIN_START.isoformat(),
            "train_end_exclusive": TRAIN_END.isoformat(),
            "selection_end_exclusive": SELECTION_END.isoformat(),
        },
        "source_audit": source_audit,
        "primary": primary_report,
        "controls": controls,
        "support_gate": support,
        "novelty_metrics": novelty_metrics,
        "novelty_gate": novelty,
        "combined_gate_passed": passed,
        "pure_clock": (
            {
                "path": str(DEFAULT_CLOCK),
                "sha256": hashlib.sha256(clock_bytes).hexdigest(),
                "gzip_mtime": 0,
                "schema": list(CLOCK_FIELDS),
                "rows": len(primary),
            }
            if clock_bytes is not None
            else None
        ),
        "outcome_boundary": {
            "dewh_source_rows_read": source_audit["rows"],
            "comparator_clock_rows_read": comparator_rows,
            "market_rows_loaded": 0,
            "funding_rows_loaded": 0,
            "performance_artifacts_parsed": 0,
            "return_or_pnl_fields_read": 0,
            "strict_simulation_calls": 0,
            "post_2023_source_rows_loaded": 0,
            "network_calls": 0,
            "economic_outcomes_computed": False,
        },
        "parameter_search_performed": False,
        "post_failure_repair_performed": False,
        "failure_action": None if passed else "retire_before_economic_evaluation",
        "next_action": (
            "freeze strict train evaluator"
            if passed
            else "retire DEWH-144 without threshold, side, timing, or hold repair"
        ),
    }
    core["result_hash"] = canonical_hash(core)
    return {**core, "created_at": datetime.now(timezone.utc).isoformat()}, clock_bytes


def publish(
    report_path: Path,
    clock_path: Path,
    report: Mapping[str, Any],
    clock_bytes: bytes | None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    clock_path.parent.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        if clock_bytes is not None:
            clock_fd = os.open(clock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            created.append(clock_path)
            with os.fdopen(clock_fd, "wb") as handle:
                handle.write(clock_bytes)
                handle.flush()
                os.fsync(handle.fileno())
        report_fd = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        created.append(report_path)
        payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        with os.fdopen(report_fd, "w", encoding="utf-8") as handle:
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
    report, clock_bytes = build_outputs()
    publish(DEFAULT_REPORT, DEFAULT_CLOCK, report, clock_bytes)
    print(
        json.dumps(
            {
                "report": str(DEFAULT_REPORT),
                "clock": str(DEFAULT_CLOCK) if clock_bytes is not None else None,
                "result_hash": report["result_hash"],
                "passed": report["combined_gate_passed"],
                "support_passed": report["support_gate"]["passed"],
                "novelty_passed": report["novelty_gate"]["passed"],
                "train_events": report["primary"]["splits"]["train"]["accepted_events"],
                "selection_events": report["primary"]["splits"]["selection"][
                    "accepted_events"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
