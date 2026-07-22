"""Build source-only BIRB-120 support, control, and novelty clocks."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from training import preregister_sec_bitcoin_issuer_reactivation_breadth as prereg


PROTOCOL_VERSION = "sec_bitcoin_issuer_reactivation_breadth_support_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path("training/build_sec_bitcoin_issuer_reactivation_breadth_support.py")
PREREGISTRATION = Path(
    "results/sec_bitcoin_issuer_reactivation_breadth_"
    "preregistration_2026-07-23.json"
)
PREREGISTRATION_SHA256 = (
    "acdd79007901427657a360acc15613fdacaf34c2238033eb9c77bda694527023"
)
PREREGISTRATION_MANIFEST_HASH = (
    "2e0c7dcf55963a5e5f9ea87b3e3b6f551b34b4016cb1a292603e31b41e0be5a4"
)
PREREGISTRATION_POLICY_HASH = (
    "424a904a8512635275ad276666ee66155ffa251721c24a08c4a9a057fe8b15a4"
)
DEFAULT_CLOCK = Path(
    "data/sec_bitcoin_issuer_reactivation_breadth_2020_2023/"
    "birb120_support_clocks_2020_2023.csv.gz"
)
DEFAULT_REPORT = Path(
    "results/sec_bitcoin_issuer_reactivation_breadth_support_2026-07-23.json"
)

UTC = timezone.utc
READY_DELAY = timedelta(minutes=60)
REACTIVATION_GAP = timedelta(days=365)
REPEAT_GAP = timedelta(days=90)
BREADTH_WINDOW = timedelta(days=7)
STALE_SHIFT = timedelta(days=30)
HOLD = timedelta(hours=120)
NEAR_WINDOW = timedelta(hours=12)
SOURCE_END = datetime(2024, 1, 1, tzinfo=UTC)
TRAIN_START = datetime(2020, 1, 1, tzinfo=UTC)
TRAIN_END = datetime(2023, 1, 1, tzinfo=UTC)
SELECTION_END = SOURCE_END

CLOCK_COLUMNS = (
    "candidate",
    "control",
    "signal_id",
    "split",
    "signal_ready",
    "entry_time",
    "exit_time",
    "side",
    "threshold",
    "trigger_accessions",
    "breadth_accessions",
    "breadth_issuers",
    "breadth_count",
)


@dataclass(frozen=True, order=True)
class IssuerEvent:
    ready: datetime
    issuer: str
    accessions: tuple[str, ...]
    forms: tuple[str, ...] = ()
    prior_ready: datetime | None = None
    gap_seconds: float | None = None


@dataclass(frozen=True)
class BreadthSignal:
    control: str
    signal_ready: datetime
    threshold: int
    trigger_accessions: tuple[str, ...]
    breadth_accessions: tuple[str, ...]
    breadth_issuers: tuple[str, ...]


@dataclass(frozen=True)
class ScheduledSignal:
    control: str
    signal_id: str
    split: str
    signal_ready: datetime
    entry_time: datetime
    exit_time: datetime
    threshold: int
    trigger_accessions: tuple[str, ...]
    breadth_accessions: tuple[str, ...]
    breadth_issuers: tuple[str, ...]


@dataclass(frozen=True)
class Config:
    source: str = str(prereg.SOURCE_ARTIFACT)
    preregistration: str = str(PREREGISTRATION)
    prior_microstructure_bundle: str = str(prereg.COMPARATOR_SPECS[0]["path"])
    trollbox_clock: str = str(prereg.COMPARATOR_SPECS[1]["path"])
    live_clock: str = str(prereg.COMPARATOR_SPECS[2]["path"])
    clock_output: str = str(DEFAULT_CLOCK)
    report_output: str = str(DEFAULT_REPORT)


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        raise RuntimeError("path must be repository-relative")
    resolved = (REPOSITORY_ROOT / candidate).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("path must remain repository-relative") from exc
    return resolved


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def parse_time(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def format_time(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def ceil_5m(value: datetime) -> datetime:
    value = value.astimezone(UTC)
    epoch_us = int(value.timestamp() * 1_000_000)
    width_us = 300 * 1_000_000
    rounded = ((epoch_us + width_us - 1) // width_us) * width_us
    return datetime.fromtimestamp(rounded / 1_000_000, tz=UTC)


def _issuer_key(ciks: Sequence[Any]) -> str:
    numeric = sorted(int(str(value)) for value in ciks)
    if not numeric:
        raise RuntimeError("SEC accession has no CIK")
    return f"{numeric[0]:010d}"


def source_events_from_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[list[IssuerEvent], dict[str, Any]]:
    by_accession: dict[str, dict[str, Any]] = {}
    input_rows = 0
    amendments = 0
    for row in rows:
        input_rows += 1
        accession = str(row["accession"])
        form = str(row["form"])
        amendment = bool(row["amendment"])
        ready = parse_time(str(row["acceptance_datetime"])) + READY_DELAY
        if ready >= SOURCE_END + READY_DELAY:
            raise RuntimeError("post-2023 SEC value row is forbidden")
        canonical = {
            "accession": accession,
            "form": form,
            "amendment": amendment,
            "ready": ready,
            "issuer": _issuer_key(row["ciks"]),
        }
        previous = by_accession.get(accession)
        if previous is not None and previous != canonical:
            raise RuntimeError(f"conflicting duplicate SEC accession: {accession}")
        by_accession[accession] = canonical
        amendments += int(amendment)

    eligible = [
        row
        for row in by_accession.values()
        if not row["amendment"] and row["form"] in {"6-K", "8-K"}
    ]
    grouped: dict[tuple[datetime, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        grouped[(row["ready"], row["issuer"])].append(row)

    events: list[IssuerEvent] = []
    for (ready, issuer), group in sorted(grouped.items()):
        events.append(
            IssuerEvent(
                ready=ready,
                issuer=issuer,
                accessions=tuple(sorted(row["accession"] for row in group)),
                forms=tuple(sorted({row["form"] for row in group})),
            )
        )
    return events, {
        "input_document_rows": input_rows,
        "unique_accessions": len(by_accession),
        "amendment_document_rows": amendments,
        "eligible_accessions": len(eligible),
        "issuer_ready_events": len(events),
        "same_issuer_ready_collapsed_accessions": len(eligible) - len(events),
        "distinct_issuers": len({event.issuer for event in events}),
    }


def load_source_events(path: str | Path) -> tuple[list[IssuerEvent], dict[str, Any]]:
    def rows() -> Iterable[Mapping[str, Any]]:
        with gzip.open(_repository_path(path), "rt", encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)

    return source_events_from_rows(rows())


def classify_events(events: Sequence[IssuerEvent]) -> dict[str, list[IssuerEvent]]:
    previous: dict[str, datetime] = {}
    classified: dict[str, list[IssuerEvent]] = {
        "reactivation": [],
        "birth": [],
        "any": [],
        "repeat": [],
    }
    grouped: dict[datetime, list[IssuerEvent]] = defaultdict(list)
    for event in events:
        grouped[event.ready].append(event)
    for ready in sorted(grouped):
        batch = sorted(grouped[ready], key=lambda event: (event.issuer, event.accessions))
        for event in batch:
            prior = previous.get(event.issuer)
            gap = None if prior is None else (ready - prior).total_seconds()
            enriched = IssuerEvent(
                ready=event.ready,
                issuer=event.issuer,
                accessions=event.accessions,
                forms=event.forms,
                prior_ready=prior,
                gap_seconds=gap,
            )
            classified["any"].append(enriched)
            if prior is None:
                classified["birth"].append(enriched)
            else:
                elapsed = ready - prior
                if elapsed >= REACTIVATION_GAP:
                    classified["reactivation"].append(enriched)
                if elapsed < REPEAT_GAP:
                    classified["repeat"].append(enriched)
        for event in batch:
            previous[event.issuer] = ready
    return classified


def _permuted_events(events: Sequence[IssuerEvent]) -> list[IssuerEvent]:
    by_year: dict[int, set[str]] = defaultdict(set)
    for event in events:
        by_year[event.ready.year].add(event.issuer)
    mappings: dict[int, dict[str, str]] = {}
    for year, issuers in by_year.items():
        source = sorted(issuers)
        destination = sorted(
            issuers,
            key=lambda issuer: hashlib.sha256(
                f"BIRB-120|year-cik-permutation|{year}|{issuer}".encode()
            ).hexdigest(),
        )
        mappings[year] = dict(zip(source, destination, strict=True))
    return [
        IssuerEvent(
            ready=event.ready,
            issuer=mappings[event.ready.year][event.issuer],
            accessions=event.accessions,
            forms=event.forms,
        )
        for event in events
    ]


def _shift_events(events: Sequence[IssuerEvent], offset: timedelta) -> list[IssuerEvent]:
    shifted: list[IssuerEvent] = []
    for event in events:
        ready = event.ready + offset
        if ready >= SOURCE_END:
            continue
        shifted.append(
            IssuerEvent(
                ready=ready,
                issuer=event.issuer,
                accessions=event.accessions,
                forms=event.forms,
                prior_ready=event.prior_ready,
                gap_seconds=event.gap_seconds,
            )
        )
    return shifted


def breadth_signals(
    events: Sequence[IssuerEvent], *, control: str, threshold: int
) -> list[BreadthSignal]:
    grouped: dict[datetime, list[IssuerEvent]] = defaultdict(list)
    for event in events:
        grouped[event.ready].append(event)
    active: dict[str, IssuerEvent] = {}
    signals: list[BreadthSignal] = []
    for ready in sorted(grouped):
        cutoff = ready - BREADTH_WINDOW
        active = {
            issuer: event for issuer, event in active.items() if event.ready > cutoff
        }
        before = len(active)
        new_events: list[IssuerEvent] = []
        for event in sorted(grouped[ready], key=lambda item: (item.issuer, item.accessions)):
            if event.issuer in active:
                continue
            active[event.issuer] = event
            new_events.append(event)
        after = len(active)
        if before < threshold <= after and new_events:
            contributors = [active[issuer] for issuer in sorted(active)]
            signals.append(
                BreadthSignal(
                    control=control,
                    signal_ready=ready,
                    threshold=threshold,
                    trigger_accessions=tuple(
                        sorted(accession for event in new_events for accession in event.accessions)
                    ),
                    breadth_accessions=tuple(
                        sorted(
                            accession
                            for event in contributors
                            for accession in event.accessions
                        )
                    ),
                    breadth_issuers=tuple(sorted(active)),
                )
            )
    return signals


def single_signals(events: Sequence[IssuerEvent]) -> list[BreadthSignal]:
    return [
        BreadthSignal(
            control="single_reactivation",
            signal_ready=event.ready,
            threshold=1,
            trigger_accessions=event.accessions,
            breadth_accessions=event.accessions,
            breadth_issuers=(event.issuer,),
        )
        for event in events
    ]


def _split_for(entry: datetime, exit_time: datetime) -> str | None:
    if TRAIN_START <= entry and exit_time <= TRAIN_END:
        return "train"
    if TRAIN_END <= entry and exit_time <= SELECTION_END:
        return "selection"
    return None


def _signal_id(signal: BreadthSignal) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "control": signal.control,
                "signal_ready": format_time(signal.signal_ready),
                "threshold": signal.threshold,
                "trigger_accessions": signal.trigger_accessions,
                "breadth_accessions": signal.breadth_accessions,
                "breadth_issuers": signal.breadth_issuers,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def schedule_nonoverlap(signals: Sequence[BreadthSignal]) -> list[ScheduledSignal]:
    accepted: list[ScheduledSignal] = []
    prior_exit: datetime | None = None
    used_accessions: set[str] = set()
    for signal in sorted(
        signals,
        key=lambda item: (
            item.signal_ready,
            item.trigger_accessions,
            item.breadth_accessions,
        ),
    ):
        entry = ceil_5m(signal.signal_ready) + timedelta(minutes=5)
        exit_time = entry + HOLD
        split = _split_for(entry, exit_time)
        if split is None:
            continue
        if prior_exit is not None and entry < prior_exit:
            continue
        identities = set(signal.breadth_accessions)
        if identities & used_accessions:
            continue
        accepted.append(
            ScheduledSignal(
                control=signal.control,
                signal_id=_signal_id(signal),
                split=split,
                signal_ready=signal.signal_ready,
                entry_time=entry,
                exit_time=exit_time,
                threshold=signal.threshold,
                trigger_accessions=signal.trigger_accessions,
                breadth_accessions=signal.breadth_accessions,
                breadth_issuers=signal.breadth_issuers,
            )
        )
        prior_exit = exit_time
        used_accessions.update(identities)
    return accepted


def build_control_schedules(events: Sequence[IssuerEvent]) -> dict[str, list[ScheduledSignal]]:
    classified = classify_events(events)
    permuted = classify_events(_permuted_events(events))["reactivation"]
    raw: dict[str, list[BreadthSignal]] = {
        "primary": breadth_signals(
            classified["reactivation"], control="primary", threshold=3
        ),
        "first_ever_birth_breadth": breadth_signals(
            classified["birth"], control="first_ever_birth_breadth", threshold=3
        ),
        "any_mention_breadth": breadth_signals(
            classified["any"], control="any_mention_breadth", threshold=3
        ),
        "repeat_filer_breadth": breadth_signals(
            classified["repeat"], control="repeat_filer_breadth", threshold=3
        ),
        "single_reactivation": single_signals(classified["reactivation"]),
        "stale_30d": breadth_signals(
            _shift_events(classified["reactivation"], STALE_SHIFT),
            control="stale_30d",
            threshold=3,
        ),
        "year_cik_permutation": breadth_signals(
            permuted, control="year_cik_permutation", threshold=3
        ),
        "threshold_two": breadth_signals(
            classified["reactivation"], control="threshold_two", threshold=2
        ),
        "threshold_four": breadth_signals(
            classified["reactivation"], control="threshold_four", threshold=4
        ),
    }
    return {name: schedule_nonoverlap(rows) for name, rows in raw.items()}


def _period_counts(rows: Sequence[ScheduledSignal]) -> dict[str, Any]:
    years = Counter(str(row.entry_time.year) for row in rows)
    months = Counter(row.entry_time.strftime("%Y-%m") for row in rows)
    quarters = Counter(
        f"{row.entry_time.year}Q{(row.entry_time.month - 1) // 3 + 1}" for row in rows
    )
    halves = Counter(
        f"{row.entry_time.year}H{1 if row.entry_time.month <= 6 else 2}" for row in rows
    )
    ordered = sorted(row.entry_time for row in rows)
    gaps = [
        (right - left).total_seconds() / 86400 for left, right in zip(ordered, ordered[1:])
    ]
    return {
        "total": len(rows),
        "by_year": dict(sorted(years.items())),
        "by_month": dict(sorted(months.items())),
        "by_quarter": dict(sorted(quarters.items())),
        "by_half": dict(sorted(halves.items())),
        "active_quarters": len(quarters),
        "maximum_month_share": max(months.values(), default=0) / len(rows) if rows else 0.0,
        "maximum_quarter_share": max(quarters.values(), default=0) / len(rows) if rows else 0.0,
        "maximum_calendar_gap_days": max(gaps, default=0.0),
        "distinct_breadth_issuers": len(
            {issuer for row in rows for issuer in row.breadth_issuers}
        ),
    }


def schedule_stats(rows: Sequence[ScheduledSignal]) -> dict[str, Any]:
    train = [row for row in rows if row.split == "train"]
    selection = [row for row in rows if row.split == "selection"]
    return {
        "total": len(rows),
        "train": _period_counts(train),
        "selection": _period_counts(selection),
        "first_entry": format_time(min(row.entry_time for row in rows)) if rows else None,
        "last_entry": format_time(max(row.entry_time for row in rows)) if rows else None,
    }


def _near_containment(
    primary: Sequence[datetime], comparator: Sequence[datetime], window: timedelta
) -> float:
    if not primary:
        return 0.0
    comparator = sorted(comparator)
    matched = 0
    j = 0
    for current in sorted(primary):
        while j < len(comparator) and comparator[j] < current - window:
            j += 1
        if j < len(comparator) and comparator[j] <= current + window:
            matched += 1
    return matched / len(primary)


def overlap_metrics(
    primary: Sequence[ScheduledSignal], comparator: Sequence[ScheduledSignal | datetime]
) -> dict[str, Any]:
    p = {row.entry_time for row in primary}
    c = {
        row.entry_time if isinstance(row, ScheduledSignal) else row
        for row in comparator
        if TRAIN_START
        <= (row.entry_time if isinstance(row, ScheduledSignal) else row)
        < SOURCE_END
    }
    intersection = p & c
    union = p | c
    return {
        "primary_entries": len(p),
        "comparator_entries": len(c),
        "exact_intersection": len(intersection),
        "exact_entry_jaccard": len(intersection) / len(union) if union else 0.0,
        "primary_to_comparator_near_containment": _near_containment(
            sorted(p), sorted(c), NEAR_WINDOW
        ),
        "comparator_to_primary_near_containment": _near_containment(
            sorted(c), sorted(p), NEAR_WINDOW
        ),
    }


def _load_comparator_times(cfg: Config) -> dict[str, list[datetime]]:
    groups: dict[str, list[datetime]] = defaultdict(list)

    bundle = json.loads(
        _repository_path(cfg.prior_microstructure_bundle).read_text(encoding="utf-8")
    )
    for name, comparator in bundle["comparators"].items():
        for event in comparator.get("events", []):
            groups[f"micro:{name}"].append(parse_time(str(event["signal_date"])))

    trollbox = json.loads(_repository_path(cfg.trollbox_clock).read_text(encoding="utf-8"))
    for event in trollbox["events"]:
        if int(event.get("contrarian_side", 0)) != 0:
            groups["semantic:trollbox"].append(parse_time(str(event["entry_earliest"])))

    with gzip.open(_repository_path(cfg.live_clock), "rt", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            groups[f"live:{row['candidate_id']}"] .append(parse_time(row["entry_time"]))
    return dict(groups)


def _source_gate_results(primary: Sequence[ScheduledSignal]) -> dict[str, Any]:
    stats = schedule_stats(primary)
    train = stats["train"]
    selection = stats["selection"]
    checks = {
        "train_total": train["total"] >= 24,
        "selection_total": selection["total"] >= 8,
        "each_train_year": all(train["by_year"].get(str(year), 0) >= 6 for year in (2020, 2021, 2022)),
        "each_selection_half": all(selection["by_half"].get(f"2023H{half}", 0) >= 3 for half in (1, 2)),
        "train_distinct_issuers": train["distinct_breadth_issuers"] >= 18,
        "selection_distinct_issuers": selection["distinct_breadth_issuers"] >= 6,
        "train_active_quarters": train["active_quarters"] >= 8,
        "train_month_share": train["maximum_month_share"] <= 0.20,
        "selection_month_share": selection["maximum_month_share"] <= 0.20,
        "train_quarter_share": train["maximum_quarter_share"] <= 0.40,
        "selection_quarter_share": selection["maximum_quarter_share"] <= 0.40,
        "train_gap": train["maximum_calendar_gap_days"] <= 150,
        "selection_gap": selection["maximum_calendar_gap_days"] <= 150,
        "duplicate_signal_ids": len({row.signal_id for row in primary}) == len(primary),
        "duplicate_accepted_accessions": len(
            [accession for row in primary for accession in row.breadth_accessions]
        )
        == len({accession for row in primary for accession in row.breadth_accessions}),
    }
    return {"passed": all(checks.values()), "checks": checks, "stats": stats}


def _specificity_results(
    schedules: Mapping[str, Sequence[ScheduledSignal]]
) -> dict[str, Any]:
    primary = schedules["primary"]
    metrics = {
        name: overlap_metrics(primary, rows)
        for name, rows in schedules.items()
        if name != "primary"
    }
    caps = {
        "first_ever_birth_breadth": 0.50,
        "any_mention_breadth": 0.60,
        "repeat_filer_breadth": 0.50,
        "stale_30d": 0.50,
        "year_cik_permutation": 0.50,
    }
    checks = {
        name: metrics[name]["primary_to_comparator_near_containment"] <= cap
        for name, cap in caps.items()
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "caps": caps,
        "metrics": metrics,
        "single_reactivation_proximity_is_report_only": True,
        "threshold_controls_are_report_only": True,
    }


def _novelty_results(
    primary: Sequence[ScheduledSignal], comparator_times: Mapping[str, Sequence[datetime]]
) -> dict[str, Any]:
    metrics = {
        name: overlap_metrics(primary, list(times))
        for name, times in sorted(comparator_times.items())
    }
    checks: dict[str, bool] = {}
    for name, row in metrics.items():
        if row["comparator_entries"] < 10:
            continue
        checks[name] = (
            row["exact_entry_jaccard"] <= 0.10
            and row["primary_to_comparator_near_containment"] <= 0.35
        )
    return {
        "passed": bool(checks) and all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "minimum_comparator_entries": 10,
        "maximum_exact_entry_jaccard": 0.10,
        "maximum_primary_near_containment": 0.35,
    }


def _clock_rows(schedules: Mapping[str, Sequence[ScheduledSignal]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for control, signals in schedules.items():
        for signal in signals:
            rows.append(
                {
                    "candidate": prereg.POLICY_ID,
                    "control": control,
                    "signal_id": signal.signal_id,
                    "split": signal.split,
                    "signal_ready": format_time(signal.signal_ready),
                    "entry_time": format_time(signal.entry_time),
                    "exit_time": format_time(signal.exit_time),
                    "side": 1,
                    "threshold": signal.threshold,
                    "trigger_accessions": "|".join(signal.trigger_accessions),
                    "breadth_accessions": "|".join(signal.breadth_accessions),
                    "breadth_issuers": "|".join(signal.breadth_issuers),
                    "breadth_count": len(signal.breadth_issuers),
                }
            )
    return sorted(rows, key=lambda row: (row["control"], row["entry_time"], row["signal_id"]))


def _write_gzip_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as zipped:
                with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                    writer = csv.DictWriter(text, fieldnames=CLOCK_COLUMNS, lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _validate_frozen_inputs(cfg: Config) -> dict[str, Any]:
    if cfg != Config(
        clock_output=cfg.clock_output,
        report_output=cfg.report_output,
    ):
        raise RuntimeError("BIRB support input configuration is frozen")
    prereg_sha = sha256_file(cfg.preregistration)
    if prereg_sha != PREREGISTRATION_SHA256:
        raise RuntimeError("BIRB preregistration file hash mismatch")
    payload = json.loads(_repository_path(cfg.preregistration).read_text(encoding="utf-8"))
    prereg.validate_preregistration(payload)
    if payload["manifest_hash"] != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("BIRB preregistration manifest mismatch")
    if payload["policy_hash"] != PREREGISTRATION_POLICY_HASH:
        raise RuntimeError("BIRB preregistration policy mismatch")
    if sha256_file(cfg.source) != prereg.SOURCE_ARTIFACT_SHA256:
        raise RuntimeError("BIRB SEC source hash mismatch")
    for spec, configured in zip(
        prereg.COMPARATOR_SPECS,
        (cfg.prior_microstructure_bundle, cfg.trollbox_clock, cfg.live_clock),
        strict=True,
    ):
        if str(spec["path"]) != configured or sha256_file(configured) != spec["sha256"]:
            raise RuntimeError(f"BIRB comparator binding mismatch: {spec['name']}")
    return payload


def build_support(cfg: Config = Config()) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prereg_payload = _validate_frozen_inputs(cfg)
    events, source_stats = load_source_events(cfg.source)
    schedules = build_control_schedules(events)
    source_gates = _source_gate_results(schedules["primary"])
    specificity = _specificity_results(schedules)
    comparator_times = _load_comparator_times(cfg)
    novelty = _novelty_results(schedules["primary"], comparator_times)
    decision = (
        "PASS_SOURCE"
        if source_gates["passed"] and specificity["passed"] and novelty["passed"]
        else "REJECT_SOURCE"
    )
    rows = _clock_rows(schedules)
    report: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": prereg.POLICY_ID,
        "decision": decision,
        "next_action": (
            "freeze strict pre-2024 evaluator before opening BTC rows"
            if decision == "PASS_SOURCE"
            else "retire BIRB-120 without BTC outcomes or parameter repair"
        ),
        "config": asdict(cfg),
        "anchors": {
            "preregistration": str(PREREGISTRATION),
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "preregistration_manifest_hash": prereg_payload["manifest_hash"],
            "preregistration_policy_hash": prereg_payload["policy_hash"],
            "source_sha256": prereg.SOURCE_ARTIFACT_SHA256,
            "builder_source": str(SCRIPT_PATH),
            "builder_sha256": sha256_file(SCRIPT_PATH),
        },
        "source_stats": source_stats,
        "reactivation_stats": {
            name: len(rows) for name, rows in classify_events(events).items()
        },
        "schedule_stats": {
            name: schedule_stats(rows) for name, rows in schedules.items()
        },
        "source_gates": source_gates,
        "mechanism_specificity": specificity,
        "novelty": novelty,
        "clock": {
            "path": cfg.clock_output,
            "rows": len(rows),
            "columns": list(CLOCK_COLUMNS),
            "contains_market_or_outcome_fields": False,
        },
        "outcome_boundary": {
            "sec_source_value_rows_read": source_stats["input_document_rows"],
            "sec_filing_body_rows_read": 0,
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "future_return_rows_read": 0,
            "pnl_cagr_mdd_opened": False,
            "post_2023_source_value_rows_read": 0,
            "network_calls": 0,
            "subprocess_calls": 0,
        },
    }
    report["manifest_hash"] = canonical_hash(report)
    return report, rows


def run(cfg: Config = Config()) -> dict[str, Any]:
    report, rows = build_support(cfg)
    clock_path = _repository_path(cfg.clock_output)
    report_path = _repository_path(cfg.report_output)
    _write_gzip_csv(clock_path, rows)
    report["clock"]["sha256"] = sha256_file(cfg.clock_output)
    report["manifest_hash"] = canonical_hash(
        {key: value for key, value in report.items() if key != "manifest_hash"}
    )
    _write_json(report_path, report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock-output", default=str(DEFAULT_CLOCK))
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(
        Config(clock_output=args.clock_output, report_output=args.report_output)
    )
    print(
        json.dumps(
            {
                "candidate": report["candidate"],
                "decision": report["decision"],
                "primary": report["source_gates"]["stats"],
                "clock": report["clock"],
                "manifest_hash": report["manifest_hash"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
