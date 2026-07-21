"""Build the outcome-blind source-support report for TSDR-72."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


POLICY_ID = "TSDR-72"
PROTOCOL_VERSION = "trollbox_semantic_disagreement_resolution_support_v1"
BAR = timedelta(minutes=5)
DEADLINE = timedelta(hours=6)
HOLD = timedelta(hours=6)
ONE_HOUR = timedelta(hours=1)

MECHANISM_DOCUMENT = Path(
    "docs/trollbox-semantic-disagreement-resolution-relay-mechanism-decision-"
    "2026-07-21.md"
)
MECHANISM_DOCUMENT_SHA256 = (
    "61ac8501410abafea694926ee04e996146ec69702fb4be5e24512390a264949b"
)
SEMANTIC_CLOCK = Path("results/bitmex_trollbox_semantic_clock_2026-07-20.json")
SEMANTIC_CLOCK_SHA256 = (
    "af8687564614ec5a1cbd7a1438c908f687af7bd99ceede9539016e5c1b111bd4"
)
SEMANTIC_CLOCK_MANIFEST_HASH = (
    "fdcd9c7c376b18df2799acf24af04a421ca679e27009e6a539888defc7438aa8"
)
SEMANTIC_SUPPORT = Path("results/bitmex_trollbox_semantic_support_2026-07-20.json")
SEMANTIC_SUPPORT_SHA256 = (
    "2b89f710d59a5c0708d400541defb43d5e292f6d9bdedbe66d6bdcf614d09e94"
)
SEMANTIC_SUPPORT_RESULT_HASH = (
    "5996b7d7497d6bf5e96343f7ceca766363d58aa34280aea0fdb7b8653a8b1725"
)
ATTENTION_CLOCK = Path("results/bitmex_trollbox_attention_clock_2026-07-20.json")
ATTENTION_CLOCK_SHA256 = (
    "5b60016a3d612f8cd29ea4548241daea76b6a6b60759837ab7bfcd60b8727f73"
)
ATTENTION_CLOCK_MANIFEST_HASH = (
    "8d1eebc60906942f5900454f956c41f8e1ccb2f00d8e97ad426669e983abdb7e"
)
DEFAULT_OUTPUT = Path(
    "results/trollbox_semantic_disagreement_resolution_support_2026-07-21.json"
)

TRAIN_START = datetime(2020, 7, 1, tzinfo=timezone.utc)
TRAIN_END = datetime(2022, 1, 1, tzinfo=timezone.utc)
SELECTION_END = datetime(2023, 1, 1, tzinfo=timezone.utc)
SPLITS = (
    ("train", TRAIN_START, TRAIN_END),
    ("selection", TRAIN_END, SELECTION_END),
)

EVENT_FIELDS = {
    "observation_start",
    "observation_end",
    "entry_earliest",
    "exit_time",
    "crowd_label",
    "contrarian_side",
    "bullish_participants",
    "bearish_participants",
    "unclear_participants",
    "selected_participants",
    "selected_messages",
    "meta_instruction_guarded_messages",
}
LABELS = {"BULLISH", "BEARISH", "UNCLEAR"}


@dataclass(frozen=True)
class SemanticEvent:
    observation_start: datetime
    observation_end: datetime
    entry_earliest: datetime
    legacy_exit_time: datetime
    crowd_label: str
    bullish_participants: int
    bearish_participants: int
    unclear_participants: int
    selected_participants: int
    selected_messages: int
    meta_instruction_guarded_messages: int

    @property
    def is_clear(self) -> bool:
        return self.crowd_label in {"BULLISH", "BEARISH"}

    @property
    def is_strong_disagreement(self) -> bool:
        return (
            self.crowd_label == "UNCLEAR"
            and self.bullish_participants >= 2
            and self.bearish_participants >= 2
        )

    @property
    def clear_side(self) -> int:
        if self.crowd_label == "BULLISH":
            return 1
        if self.crowd_label == "BEARISH":
            return -1
        raise ValueError("an UNCLEAR event has no clear side")


@dataclass(frozen=True)
class Candidate:
    onset_end: datetime
    resolution_end: datetime
    entry: datetime
    exit: datetime
    side: int
    onset_bullish: int
    onset_bearish: int
    split: str | None = None


@dataclass(frozen=True)
class BuildAudit:
    raw_candidates: int
    resolved_episodes: int
    expired_episodes: int
    unresolved_end_of_source: int


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


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle, object_pairs_hook=_reject_duplicate_pairs)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _timestamp(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a timestamp string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc
    if parsed.tzinfo is not None:
        if parsed.utcoffset() != timedelta(0):
            raise ValueError(f"{field} must be UTC")
        return parsed.astimezone(timezone.utc)
    return parsed.replace(tzinfo=timezone.utc)


def _strict_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value < 0:
        raise ValueError(f"{field} must be nonnegative")
    return value


def _consensus_label(*, bullish: int, bearish: int) -> str:
    if bullish >= 2 and bullish >= 2.0 * max(1, bearish):
        return "BULLISH"
    if bearish >= 2 and bearish >= 2.0 * max(1, bullish):
        return "BEARISH"
    return "UNCLEAR"


def _parse_event(raw: Any) -> SemanticEvent:
    if not isinstance(raw, dict) or set(raw) != EVENT_FIELDS:
        raise ValueError("semantic event schema changed")

    start = _timestamp(raw["observation_start"], field="observation_start")
    end = _timestamp(raw["observation_end"], field="observation_end")
    entry = _timestamp(raw["entry_earliest"], field="entry_earliest")
    legacy_exit = _timestamp(raw["exit_time"], field="exit_time")
    if end - start != BAR:
        raise ValueError("semantic observation must be exactly five minutes")
    if entry - end != BAR:
        raise ValueError("semantic entry latency must be exactly five minutes")
    if legacy_exit - entry != timedelta(hours=2):
        raise ValueError("legacy semantic exit clock changed")

    label = raw["crowd_label"]
    if not isinstance(label, str) or label not in LABELS:
        raise ValueError("semantic crowd label changed")
    contrarian = raw["contrarian_side"]
    if isinstance(contrarian, bool) or not isinstance(contrarian, int):
        raise ValueError("contrarian_side schema changed")
    if contrarian not in {-1, 0, 1}:
        raise ValueError("contrarian_side domain changed")

    bullish = _strict_int(raw["bullish_participants"], field="bullish_participants")
    bearish = _strict_int(raw["bearish_participants"], field="bearish_participants")
    unclear = _strict_int(raw["unclear_participants"], field="unclear_participants")
    selected = _strict_int(raw["selected_participants"], field="selected_participants")
    messages = _strict_int(raw["selected_messages"], field="selected_messages")
    guarded = _strict_int(
        raw["meta_instruction_guarded_messages"],
        field="meta_instruction_guarded_messages",
    )
    if bullish + bearish + unclear != selected:
        raise ValueError("participant label counts do not sum to selected participants")
    if selected < 1 or selected > 8:
        raise ValueError("selected participant count left the frozen range")
    if messages < selected or messages > 2 * selected:
        raise ValueError("selected message count left the frozen range")
    if guarded > messages:
        raise ValueError("guarded message count exceeds selected messages")
    if _consensus_label(bullish=bullish, bearish=bearish) != label:
        raise ValueError("semantic majority contract changed")

    return SemanticEvent(
        observation_start=start,
        observation_end=end,
        entry_earliest=entry,
        legacy_exit_time=legacy_exit,
        crowd_label=label,
        bullish_participants=bullish,
        bearish_participants=bearish,
        unclear_participants=unclear,
        selected_participants=selected,
        selected_messages=messages,
        meta_instruction_guarded_messages=guarded,
    )


def _verify_source_bindings() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    expected_hashes = {
        MECHANISM_DOCUMENT: MECHANISM_DOCUMENT_SHA256,
        SEMANTIC_CLOCK: SEMANTIC_CLOCK_SHA256,
        SEMANTIC_SUPPORT: SEMANTIC_SUPPORT_SHA256,
        ATTENTION_CLOCK: ATTENTION_CLOCK_SHA256,
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise ValueError(f"TSDR-72 frozen input changed: {path}")

    semantic = load_json(SEMANTIC_CLOCK)
    support = load_json(SEMANTIC_SUPPORT)
    attention = load_json(ATTENTION_CLOCK)

    if semantic.get("protocol_version") != "bitmex_trollbox_semantic_clock_v1":
        raise ValueError("semantic clock protocol changed")
    if semantic.get("policy_id") != "TBASR-24":
        raise ValueError("semantic clock identity changed")
    if semantic.get("manifest_hash") != SEMANTIC_CLOCK_MANIFEST_HASH:
        raise ValueError("semantic clock manifest changed")
    if semantic.get("support_result_hash") != SEMANTIC_SUPPORT_RESULT_HASH:
        raise ValueError("semantic clock support binding changed")
    if semantic.get("market_or_outcomes_opened") is not False:
        raise ValueError("semantic clock opened market outcomes")
    if semantic.get("private_text_committed") is not False:
        raise ValueError("semantic clock committed private text")

    if support.get("protocol_version") != "bitmex_trollbox_semantic_support_v1":
        raise ValueError("semantic support protocol changed")
    if support.get("result_hash") != SEMANTIC_SUPPORT_RESULT_HASH:
        raise ValueError("semantic support result changed")
    if support.get("market_or_outcomes_opened") is not False:
        raise ValueError("semantic support opened market outcomes")
    if support.get("private_text_committed") is not False:
        raise ValueError("semantic support committed private text")
    if support.get("semantic_clock_written") is not True:
        raise ValueError("semantic support did not write its frozen clock")
    support_gate = support.get("support_gate")
    if not isinstance(support_gate, dict) or support_gate.get("passed") is not True:
        raise ValueError("semantic support did not pass")
    protocol = support.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("semantic support protocol is malformed")
    for field in ("market_rows_loaded", "funding_rows_loaded", "outcome_rows_loaded"):
        if protocol.get(field) != 0:
            raise ValueError(f"semantic support loaded forbidden {field}")

    if attention.get("protocol_version") != "bitmex_trollbox_attention_clock_v1":
        raise ValueError("attention clock protocol changed")
    if attention.get("manifest_hash") != ATTENTION_CLOCK_MANIFEST_HASH:
        raise ValueError("attention clock manifest changed")
    if attention.get("outcomes_opened") is not False:
        raise ValueError("attention clock opened outcomes")
    if attention.get("message_semantics_opened") is not False:
        raise ValueError("attention clock unexpectedly opened semantics")
    return semantic, support, attention


def load_semantic_events() -> tuple[list[SemanticEvent], dict[str, Any]]:
    semantic, support, attention = _verify_source_bindings()
    raw_semantic = semantic.get("events")
    raw_attention = attention.get("events")
    if not isinstance(raw_semantic, list) or not isinstance(raw_attention, list):
        raise ValueError("frozen clocks must contain event lists")
    if len(raw_semantic) != 5417 or len(raw_attention) != len(raw_semantic):
        raise ValueError("frozen attention event count changed")

    events = [_parse_event(row) for row in raw_semantic]
    prior_end: datetime | None = None
    for index, (event, attention_row) in enumerate(zip(events, raw_attention)):
        if prior_end is not None and event.observation_end <= prior_end:
            raise ValueError("semantic events are not strictly chronological")
        prior_end = event.observation_end
        if not isinstance(attention_row, dict):
            raise ValueError("attention event schema changed")
        expected_times = {
            "observation_start": event.observation_start,
            "observation_end": event.observation_end,
            "entry_earliest": event.entry_earliest,
            "exit_time": event.legacy_exit_time,
        }
        if set(attention_row) != set(expected_times):
            raise ValueError("attention event schema changed")
        for field, expected in expected_times.items():
            if _timestamp(attention_row[field], field=f"attention[{index}].{field}") != expected:
                raise ValueError("semantic and attention clocks diverged")

    audit = {
        "semantic_support_result_hash": support["result_hash"],
        "attention_event_count": len(raw_attention),
        "semantic_event_count": len(events),
        "source_start": events[0].observation_start.isoformat(),
        "source_end": events[-1].observation_end.isoformat(),
        "strong_disagreement_windows": sum(e.is_strong_disagreement for e in events),
        "clear_windows": sum(e.is_clear for e in events),
    }
    return events, audit


def _candidate(onset: SemanticEvent, resolution: SemanticEvent) -> Candidate:
    return Candidate(
        onset_end=onset.observation_end,
        resolution_end=resolution.observation_end,
        entry=resolution.entry_earliest,
        exit=resolution.entry_earliest + HOLD,
        side=resolution.clear_side,
        onset_bullish=onset.bullish_participants,
        onset_bearish=onset.bearish_participants,
    )


def build_primary_candidates(
    events: Sequence[SemanticEvent],
) -> tuple[list[Candidate], BuildAudit]:
    active: SemanticEvent | None = None
    candidates: list[Candidate] = []
    expired = 0
    for event in events:
        if active is not None and event.observation_end > active.observation_end + DEADLINE:
            expired += 1
            active = None
        if active is not None:
            if event.is_clear:
                candidates.append(_candidate(active, event))
                active = None
            continue
        if event.is_strong_disagreement:
            active = event
    return candidates, BuildAudit(
        raw_candidates=len(candidates),
        resolved_episodes=len(candidates),
        expired_episodes=expired,
        unresolved_end_of_source=int(active is not None),
    )


def build_clear_after_clear_candidates(
    events: Sequence[SemanticEvent],
) -> tuple[list[Candidate], BuildAudit]:
    active: SemanticEvent | None = None
    candidates: list[Candidate] = []
    expired = 0
    for event in events:
        if active is not None and event.observation_end > active.observation_end + DEADLINE:
            expired += 1
            active = None
        if active is not None:
            if event.is_clear:
                candidates.append(_candidate(active, event))
                active = None
            continue
        if event.is_clear:
            active = event
    return candidates, BuildAudit(
        raw_candidates=len(candidates),
        resolved_episodes=len(candidates),
        expired_episodes=expired,
        unresolved_end_of_source=int(active is not None),
    )


def _deterministic_side(tag: str, entry: datetime) -> int:
    key = f"{tag}|{entry.isoformat()}".encode("utf-8")
    return 1 if hashlib.sha256(key).digest()[0] < 128 else -1


def build_unresolved_candidates(
    events: Sequence[SemanticEvent],
) -> tuple[list[Candidate], BuildAudit]:
    active: SemanticEvent | None = None
    candidates: list[Candidate] = []
    resolved = 0

    def emit(onset: SemanticEvent) -> None:
        resolution = onset.observation_end + DEADLINE
        entry = resolution + BAR
        candidates.append(
            Candidate(
                onset_end=onset.observation_end,
                resolution_end=resolution,
                entry=entry,
                exit=entry + HOLD,
                side=_deterministic_side("TSDR-72-unresolved-20260721", entry),
                onset_bullish=onset.bullish_participants,
                onset_bearish=onset.bearish_participants,
            )
        )

    for event in events:
        if active is not None and event.observation_end > active.observation_end + DEADLINE:
            emit(active)
            active = None
        if active is not None:
            if event.is_clear:
                resolved += 1
                active = None
            continue
        if event.is_strong_disagreement:
            active = event

    source_coverage_end = SELECTION_END
    if active is not None and active.observation_end + DEADLINE <= source_coverage_end:
        emit(active)
        active = None
    return candidates, BuildAudit(
        raw_candidates=len(candidates),
        resolved_episodes=resolved,
        expired_episodes=len(candidates),
        unresolved_end_of_source=int(active is not None),
    )


def _candidate_split(candidate: Candidate) -> str | None:
    for name, start, end in SPLITS:
        if (
            candidate.onset_end >= start
            and candidate.resolution_end >= start
            and candidate.entry >= start
            and candidate.exit <= end
        ):
            return name
    return None


def schedule_candidates(
    candidates: Iterable[Candidate],
) -> tuple[list[Candidate], ScheduleAudit]:
    ordered = sorted(
        candidates,
        key=lambda row: (row.entry, row.onset_end, row.resolution_end, row.side),
    )
    if any(row.side not in {-1, 1} for row in ordered):
        raise ValueError("candidate side must be -1 or +1")
    if any(row.exit - row.entry != HOLD for row in ordered):
        raise ValueError("candidate hold changed")

    contained: list[Candidate] = []
    for row in ordered:
        split = _candidate_split(row)
        if split is not None:
            contained.append(replace(row, split=split))

    accepted: list[Candidate] = []
    next_free = datetime.min.replace(tzinfo=timezone.utc)
    overlap_suppressions = 0
    for row in contained:
        if row.entry < next_free:
            overlap_suppressions += 1
            continue
        accepted.append(row)
        next_free = row.exit
    return accepted, ScheduleAudit(
        raw_candidates=len(ordered),
        split_contained_candidates=len(contained),
        split_boundary_drops=len(ordered) - len(contained),
        overlap_suppressions=overlap_suppressions,
        accepted_candidates=len(accepted),
    )


def _clock_rows(candidates: Sequence[Candidate]) -> list[dict[str, Any]]:
    return [
        {
            "split": row.split,
            "onset_end": row.onset_end.isoformat(),
            "resolution_end": row.resolution_end.isoformat(),
            "entry": row.entry.isoformat(),
            "exit": row.exit.isoformat(),
            "side": row.side,
        }
        for row in candidates
    ]


def _period_counts(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _summary(candidates: Sequence[Candidate], split: str) -> dict[str, Any]:
    rows = [row for row in candidates if row.split == split]
    months = Counter(f"{row.entry.year:04d}-{row.entry.month:02d}" for row in rows)
    weekdays = Counter(str(row.entry.weekday()) for row in rows)
    total = len(rows)
    return {
        "accepted_events": total,
        "side_counts": {
            "LONG": sum(row.side == 1 for row in rows),
            "SHORT": sum(row.side == -1 for row in rows),
        },
        "year_counts": _period_counts(str(row.entry.year) for row in rows),
        "half_counts": _period_counts(
            f"{row.entry.year}-H{1 if row.entry.month <= 6 else 2}" for row in rows
        ),
        "quarter_counts": _period_counts(
            f"{row.entry.year}-Q{(row.entry.month - 1) // 3 + 1}" for row in rows
        ),
        "active_utc_weeks": len({row.entry.isocalendar()[:2] for row in rows}),
        "maximum_calendar_month_share": (
            max(months.values()) / total if total else math.nan
        ),
        "maximum_utc_entry_weekday_share": (
            max(weekdays.values()) / total if total else math.nan
        ),
    }


def _gate(primary: Mapping[str, Any]) -> dict[str, Any]:
    train = primary["splits"]["train"]
    selection = primary["splits"]["selection"]
    train_quarters = train["quarter_counts"]
    selection_quarters = selection["quarter_counts"]
    checks = {
        "train_total_at_least_150": train["accepted_events"] >= 150,
        "train_partial_2020_at_least_45": train["year_counts"].get("2020", 0) >= 45,
        "train_2021_at_least_90": train["year_counts"].get("2021", 0) >= 90,
        "train_each_quarter_at_least_15": all(
            train_quarters.get(period, 0) >= 15
            for period in (
                "2020-Q3",
                "2020-Q4",
                "2021-Q1",
                "2021-Q2",
                "2021-Q3",
                "2021-Q4",
            )
        ),
        "train_long_at_least_60": train["side_counts"]["LONG"] >= 60,
        "train_short_at_least_60": train["side_counts"]["SHORT"] >= 60,
        "train_active_weeks_at_least_60": train["active_utc_weeks"] >= 60,
        "train_month_share_at_most_0_15": (
            train["maximum_calendar_month_share"] <= 0.15
        ),
        "train_weekday_share_at_most_0_22": (
            train["maximum_utc_entry_weekday_share"] <= 0.22
        ),
        "selection_total_at_least_60": selection["accepted_events"] >= 60,
        "selection_each_half_at_least_25": all(
            selection["half_counts"].get(period, 0) >= 25
            for period in ("2022-H1", "2022-H2")
        ),
        "selection_each_quarter_at_least_12": all(
            selection_quarters.get(f"2022-Q{quarter}", 0) >= 12
            for quarter in range(1, 5)
        ),
        "selection_long_at_least_25": selection["side_counts"]["LONG"] >= 25,
        "selection_short_at_least_25": selection["side_counts"]["SHORT"] >= 25,
        "selection_active_weeks_at_least_30": selection["active_utc_weeks"] >= 30,
        "selection_month_share_at_most_0_18": (
            selection["maximum_calendar_month_share"] <= 0.18
        ),
        "selection_weekday_share_at_most_0_25": (
            selection["maximum_utc_entry_weekday_share"] <= 0.25
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


def _exact_clock_control(
    primary: Sequence[Candidate],
    *,
    mode: str,
) -> list[Candidate]:
    result: list[Candidate] = []
    for row in primary:
        if mode == "initial_plurality":
            delta = row.onset_bullish - row.onset_bearish
            side = (
                1
                if delta > 0
                else -1
                if delta < 0
                else _deterministic_side("TSDR-72-initial-tie-20260721", row.entry)
            )
        elif mode == "direction_flip":
            side = -row.side
        elif mode == "random_side":
            side = _deterministic_side("TSDR-72-random-side-20260721", row.entry)
        else:
            raise ValueError(f"unknown exact-clock control: {mode}")
        result.append(replace(row, side=side))
    return result


def _control_report(
    candidates: Sequence[Candidate],
    audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "clock_hash": canonical_hash(_clock_rows(candidates)),
        "splits": {
            name: _summary(candidates, name) for name, _, _ in SPLITS
        },
    }
    if audit is not None:
        report["audit"] = dict(audit)
    return report


def build_report() -> dict[str, Any]:
    events, source_audit = load_semantic_events()
    raw_primary, primary_build = build_primary_candidates(events)
    primary, primary_schedule = schedule_candidates(raw_primary)

    initial_plurality = _exact_clock_control(primary, mode="initial_plurality")
    direction_flip = _exact_clock_control(primary, mode="direction_flip")
    random_side = _exact_clock_control(primary, mode="random_side")

    clear_raw, clear_build = build_clear_after_clear_candidates(events)
    clear_control, clear_schedule = schedule_candidates(clear_raw)
    unresolved_raw, unresolved_build = build_unresolved_candidates(events)
    unresolved_control, unresolved_schedule = schedule_candidates(unresolved_raw)

    delayed_raw = [
        replace(row, entry=row.entry + ONE_HOUR, exit=row.exit + ONE_HOUR, split=None)
        for row in raw_primary
    ]
    delayed_control, delayed_schedule = schedule_candidates(delayed_raw)

    primary_report = {
        "clock_hash": canonical_hash(_clock_rows(primary)),
        "build_audit": asdict(primary_build),
        "schedule_audit": asdict(primary_schedule),
        "splits": {name: _summary(primary, name) for name, _, _ in SPLITS},
    }
    support_gate = _gate(primary_report)
    controls = {
        "initial_plurality": _control_report(initial_plurality),
        "direction_flip": _control_report(direction_flip),
        "deterministic_random_side": _control_report(random_side),
        "clear_after_clear": _control_report(
            clear_control,
            {"build": asdict(clear_build), "schedule": asdict(clear_schedule)},
        ),
        "unresolved_disagreement": _control_report(
            unresolved_control,
            {
                "build": asdict(unresolved_build),
                "schedule": asdict(unresolved_schedule),
            },
        ),
        "one_hour_execution_delay": _control_report(
            delayed_control,
            {"schedule": asdict(delayed_schedule)},
        ),
    }

    core: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "as_of_date": "2026-07-21",
        "decision_binding": {
            "path": str(MECHANISM_DOCUMENT),
            "sha256": MECHANISM_DOCUMENT_SHA256,
        },
        "source_binding": {
            "semantic_clock": str(SEMANTIC_CLOCK),
            "semantic_clock_sha256": SEMANTIC_CLOCK_SHA256,
            "semantic_clock_manifest_hash": SEMANTIC_CLOCK_MANIFEST_HASH,
            "semantic_support": str(SEMANTIC_SUPPORT),
            "semantic_support_sha256": SEMANTIC_SUPPORT_SHA256,
            "semantic_support_result_hash": SEMANTIC_SUPPORT_RESULT_HASH,
            "attention_clock": str(ATTENTION_CLOCK),
            "attention_clock_sha256": ATTENTION_CLOCK_SHA256,
            "attention_clock_manifest_hash": ATTENTION_CLOCK_MANIFEST_HASH,
        },
        "configuration": {
            "deadline_bars": 72,
            "hold_bars": 72,
            "entry_latency_bars": 1,
            "minimum_bullish_disagreement_participants": 2,
            "minimum_bearish_disagreement_participants": 2,
            "exposure": 0.5,
            "train_start": TRAIN_START.isoformat(),
            "train_end_exclusive": TRAIN_END.isoformat(),
            "selection_end_exclusive": SELECTION_END.isoformat(),
        },
        "outcome_boundary": {
            "market_rows_loaded": 0,
            "funding_rows_loaded": 0,
            "outcome_rows_loaded": 0,
            "return_or_pnl_fields_read": 0,
            "raw_private_text_opened": False,
            "raw_private_text_committed": False,
            "post_2022_semantic_rows_loaded": 0,
            "network_calls": 0,
            "outcomes_opened": False,
        },
        "source_audit": source_audit,
        "primary": primary_report,
        "controls": controls,
        "support_gate": support_gate,
        "parameter_search_performed": False,
        "post_failure_repair_performed": False,
        "failure_action": None if support_gate["passed"] else "retire_before_market_access",
        "next_action": (
            "freeze pure clocks and evaluate source-only novelty"
            if support_gate["passed"]
            else "retire TSDR-72 without opening market outcomes"
        ),
    }
    core["result_hash"] = canonical_hash(core)
    return {
        **core,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report()
    write_report(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "result_hash": report["result_hash"],
                "passed": report["support_gate"]["passed"],
                "train_events": report["primary"]["splits"]["train"][
                    "accepted_events"
                ],
                "selection_events": report["primary"]["splits"]["selection"][
                    "accepted_events"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
