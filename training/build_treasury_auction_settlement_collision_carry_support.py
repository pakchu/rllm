"""Build source-only TASCC-72 support, control, and novelty clocks."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from training import preregister_treasury_auction_settlement_collision_carry as prereg


PROTOCOL_VERSION = "treasury_auction_settlement_collision_carry_support_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path("training/build_treasury_auction_settlement_collision_carry_support.py")
PREREGISTRATION = Path(
    "results/treasury_auction_settlement_collision_carry_"
    "preregistration_2026-07-23.json"
)
PREREGISTRATION_SHA256 = (
    "090cc8e07a76ae033db413d0d0a9356f2a43c38fcb865a06c20f06b5f04cad67"
)
PREREGISTRATION_MANIFEST_HASH = (
    "56be616879365bca531de961c868b3c39ec406d0436094aa0cf27f5440881f4e"
)
PREREGISTRATION_POLICY_HASH = (
    "07fa97ad5ce1720d4296cac7c768f03ba5e153795ba046bbc110b535593d85d9"
)
DEFAULT_CLOCK = Path(
    "data/treasury_auction_settlement_collision_carry_2020_2023/"
    "tascc72_support_clocks_2020_2023.csv.gz"
)
DEFAULT_REPORT = Path(
    "results/treasury_auction_settlement_collision_carry_support_2026-07-23.json"
)

UTC = timezone.utc
HOLD = timedelta(hours=72)
NEAR_WINDOW = timedelta(hours=12)
TRAIN_START = datetime(2020, 1, 1, tzinfo=UTC)
TRAIN_END = datetime(2023, 1, 1, tzinfo=UTC)
SOURCE_END = datetime(2024, 1, 1, tzinfo=UTC)
BELLY_TERMS = frozenset({"5-Year", "7-Year"})
LONG_TERMS = frozenset({"10-Year", "20-Year", "30-Year"})
ALL_TERMS = frozenset(
    {"2-Year", "3-Year", "5-Year", "7-Year", "10-Year", "20-Year", "30-Year"}
)

CLOCK_COLUMNS = (
    "candidate",
    "control",
    "signal_id",
    "split",
    "auction_or_issue_date",
    "latest_result_available",
    "signal_time",
    "entry_time",
    "exit_time",
    "side",
    "terms",
    "cusips",
    "belly_terms",
    "long_terms",
)


@dataclass(frozen=True, order=True)
class Auction:
    auction_date: date
    issue_date: date
    result_available: datetime
    term: str
    cusip: str


@dataclass(frozen=True)
class BasketSignal:
    control: str
    calendar_date: date
    latest_result_available: datetime
    signal_time: datetime
    terms: tuple[str, ...]
    cusips: tuple[str, ...]


@dataclass(frozen=True)
class ScheduledSignal:
    control: str
    signal_id: str
    split: str
    calendar_date: date
    latest_result_available: datetime
    signal_time: datetime
    entry_time: datetime
    exit_time: datetime
    terms: tuple[str, ...]
    cusips: tuple[str, ...]


@dataclass(frozen=True)
class Config:
    panel: str = str(prereg.AUCTION_PANEL)
    manifest: str = str(prereg.AUCTION_MANIFEST)
    raw_page_0: str = str(prereg.RAW_PAGES[0]["path"])
    raw_page_1: str = str(prereg.RAW_PAGES[1]["path"])
    tadi_clock: str = str(prereg.COMPARATOR_SPECS[0]["path"])
    dffb_clock: str = str(prereg.COMPARATOR_SPECS[1]["path"])
    flcc_clock: str = str(prereg.COMPARATOR_SPECS[2]["path"])
    rrp_clock: str = str(prereg.COMPARATOR_SPECS[3]["path"])
    live_clock: str = str(prereg.COMPARATOR_SPECS[4]["path"])
    preregistration: str = str(PREREGISTRATION)
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
    epoch_us = int(value.astimezone(UTC).timestamp() * 1_000_000)
    width_us = 300 * 1_000_000
    rounded = ((epoch_us + width_us - 1) // width_us) * width_us
    return datetime.fromtimestamp(rounded / 1_000_000, tz=UTC)


def _bool(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise RuntimeError(f"invalid boolean literal: {value!r}")


def join_source_rows(
    panel_rows: Iterable[Mapping[str, Any]],
    raw_rows: Iterable[Mapping[str, Any]],
) -> tuple[list[Auction], dict[str, Any]]:
    panel: dict[tuple[str, str], dict[str, Any]] = {}
    panel_rows_read = 0
    for row in panel_rows:
        panel_rows_read += 1
        key = (str(row["auction_date"])[:10], str(row["cusip"]))
        if key in panel:
            raise RuntimeError(f"duplicate normalized auction key: {key}")
        panel[key] = {
            "result_available": parse_time(str(row["result_available_at_utc"])),
            "term": str(row["original_security_term"]),
            "source_complete": _bool(row["source_complete"]),
        }

    matched: dict[tuple[str, str], dict[str, Any]] = {}
    raw_rows_parsed = 0
    raw_rows_outside_panel = 0
    raw_post_2023_transport_rows = 0
    for row in raw_rows:
        raw_rows_parsed += 1
        auction_date_text = str(row["auctionDate"])[:10]
        if auction_date_text >= "2024-01-01":
            raw_post_2023_transport_rows += 1
        key = (auction_date_text, str(row["cusip"]))
        if key not in panel:
            raw_rows_outside_panel += 1
            continue
        if key in matched:
            raise RuntimeError(f"duplicate raw auction key: {key}")
        matched[key] = {
            "issue_date": str(row["issueDate"])[:10],
            "security_type": str(row["securityType"]),
            "term": str(row["originalSecurityTerm"]),
            "reopening": str(row["reopening"]),
        }
    if set(matched) != set(panel):
        missing = sorted(set(panel) - set(matched))[:5]
        raise RuntimeError(f"raw auction join incomplete: {missing}")

    auctions: list[Auction] = []
    incomplete = 0
    post_2023_issue_dates_rejected = 0
    for key in sorted(panel):
        normalized = panel[key]
        raw = matched[key]
        if raw["reopening"] != "No":
            raise RuntimeError(f"normalized original issue became reopening: {key}")
        if normalized["term"] != raw["term"]:
            raise RuntimeError(f"auction term join mismatch: {key}")
        if raw["term"] not in ALL_TERMS or raw["security_type"] not in {"Note", "Bond"}:
            raise RuntimeError(f"auction universe drift: {key}")
        if not normalized["source_complete"]:
            incomplete += 1
            continue
        issue_date = date.fromisoformat(raw["issue_date"])
        if issue_date >= SOURCE_END.date():
            post_2023_issue_dates_rejected += 1
            continue
        auctions.append(
            Auction(
                auction_date=date.fromisoformat(key[0]),
                issue_date=issue_date,
                result_available=normalized["result_available"],
                term=raw["term"],
                cusip=key[1],
            )
        )
    return auctions, {
        "panel_rows_read": panel_rows_read,
        "raw_transport_rows_parsed": raw_rows_parsed,
        "raw_transport_rows_outside_pre2024_panel": raw_rows_outside_panel,
        "raw_post_2023_transport_rows_parsed_for_key_filter": raw_post_2023_transport_rows,
        "raw_rows_joined_to_pre2024_panel": len(matched),
        "source_incomplete_rows_excluded": incomplete,
        "post_2023_issue_dates_rejected": post_2023_issue_dates_rejected,
        "post_2023_rows_materialized_into_tascc": 0,
        "eligible_auction_rows": len(auctions),
    }


def _iter_panel(path: str | Path) -> Iterable[Mapping[str, Any]]:
    with gzip.open(_repository_path(path), "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(prereg.PANEL_ALLOWED_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"auction panel missing fields: {sorted(missing)}")
        for row in reader:
            yield {field: row[field] for field in prereg.PANEL_ALLOWED_COLUMNS}


def _iter_raw(paths: Sequence[str | Path]) -> Iterable[Mapping[str, Any]]:
    for path in paths:
        with gzip.open(_repository_path(path), "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload.get("securityList") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise RuntimeError("Treasury raw page schema drift")
        for row in rows:
            if not isinstance(row, dict) or "auctionDate" not in row or "cusip" not in row:
                raise RuntimeError("Treasury raw row key schema drift")
            yield row


def load_auctions(cfg: Config) -> tuple[list[Auction], dict[str, Any]]:
    return join_source_rows(
        _iter_panel(cfg.panel), _iter_raw((cfg.raw_page_0, cfg.raw_page_1))
    )


def _signal(
    control: str,
    calendar_date: date,
    group: Sequence[Auction],
    *,
    signal_time: datetime,
) -> BasketSignal:
    return BasketSignal(
        control=control,
        calendar_date=calendar_date,
        latest_result_available=max(row.result_available for row in group),
        signal_time=signal_time,
        terms=tuple(sorted({row.term for row in group})),
        cusips=tuple(sorted(row.cusip for row in group)),
    )


def settlement_signals(
    auctions: Sequence[Auction], *, control: str, mode: str
) -> list[BasketSignal]:
    grouped: dict[date, list[Auction]] = defaultdict(list)
    date_field = "auction_date" if mode == "auction_collision" else "issue_date"
    for auction in auctions:
        grouped[getattr(auction, date_field)].append(auction)
    signals: list[BasketSignal] = []
    for calendar_date, group in sorted(grouped.items()):
        terms = {row.term for row in group}
        belly = terms & BELLY_TERMS
        long = terms & LONG_TERMS
        if mode in {"primary", "result_time", "auction_collision", "delayed"}:
            eligible = bool(belly and long)
        elif mode == "belly":
            eligible = bool(belly)
        elif mode == "long":
            eligible = bool(long)
        elif mode == "multitenor":
            eligible = len(terms) >= 2
        elif mode == "single":
            eligible = len(terms) == 1
        else:
            raise RuntimeError(f"unknown TASCC signal mode: {mode}")
        if not eligible:
            continue
        marker = datetime.combine(calendar_date, datetime.min.time(), tzinfo=UTC)
        latest = max(row.result_available for row in group)
        if mode not in {"auction_collision", "result_time"} and latest > marker:
            continue
        if mode in {"auction_collision", "result_time"}:
            signal_time = latest
        elif mode == "delayed":
            signal_time = marker + timedelta(days=7)
        else:
            signal_time = marker
        if signal_time >= SOURCE_END:
            continue
        signals.append(_signal(control, calendar_date, group, signal_time=signal_time))
    return signals


def _permuted_auctions(auctions: Sequence[Auction]) -> list[Auction]:
    by_year: dict[int, set[str]] = defaultdict(set)
    for auction in auctions:
        by_year[auction.auction_date.year].add(auction.term)
    mappings: dict[int, dict[str, str]] = {}
    for year, terms in by_year.items():
        source = sorted(terms)
        target = sorted(
            terms,
            key=lambda term: hashlib.sha256(
                f"TASCC-72|term-year-permutation|{year}|{term}".encode()
            ).hexdigest(),
        )
        mappings[year] = dict(zip(source, target, strict=True))
    return [
        Auction(
            auction_date=row.auction_date,
            issue_date=row.issue_date,
            result_available=row.result_available,
            term=mappings[row.auction_date.year][row.term],
            cusip=row.cusip,
        )
        for row in auctions
    ]


def _split_for(entry: datetime, exit_time: datetime) -> str | None:
    if TRAIN_START <= entry and exit_time <= TRAIN_END:
        return "train"
    if TRAIN_END <= entry and exit_time <= SOURCE_END:
        return "selection"
    return None


def _signal_id(signal: BasketSignal) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "control": signal.control,
                "calendar_date": signal.calendar_date.isoformat(),
                "latest_result_available": format_time(signal.latest_result_available),
                "signal_time": format_time(signal.signal_time),
                "terms": signal.terms,
                "cusips": signal.cusips,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def schedule_nonoverlap(signals: Sequence[BasketSignal]) -> list[ScheduledSignal]:
    accepted: list[ScheduledSignal] = []
    prior_exit: datetime | None = None
    used_cusips: set[str] = set()
    used_dates: set[date] = set()
    for signal in sorted(signals, key=lambda row: (row.signal_time, row.cusips)):
        entry = ceil_5m(signal.signal_time) + timedelta(minutes=5)
        exit_time = entry + HOLD
        split = _split_for(entry, exit_time)
        if split is None:
            continue
        if prior_exit is not None and entry < prior_exit:
            continue
        if signal.calendar_date in used_dates or set(signal.cusips) & used_cusips:
            continue
        accepted.append(
            ScheduledSignal(
                control=signal.control,
                signal_id=_signal_id(signal),
                split=split,
                calendar_date=signal.calendar_date,
                latest_result_available=signal.latest_result_available,
                signal_time=signal.signal_time,
                entry_time=entry,
                exit_time=exit_time,
                terms=signal.terms,
                cusips=signal.cusips,
            )
        )
        prior_exit = exit_time
        used_dates.add(signal.calendar_date)
        used_cusips.update(signal.cusips)
    return accepted


def build_control_schedules(auctions: Sequence[Auction]) -> dict[str, list[ScheduledSignal]]:
    specifications = {
        "primary": (auctions, "primary"),
        "belly_settlement_calendar": (auctions, "belly"),
        "long_settlement_calendar": (auctions, "long"),
        "any_multitenor_settlement": (auctions, "multitenor"),
        "single_tenor_settlement": (auctions, "single"),
        "auction_date_collision": (auctions, "auction_collision"),
        "term_year_permutation": (_permuted_auctions(auctions), "primary"),
        "result_time_clock": (auctions, "result_time"),
        "settlement_plus_7d": (auctions, "delayed"),
    }
    return {
        control: schedule_nonoverlap(
            settlement_signals(rows, control=control, mode=mode)
        )
        for control, (rows, mode) in specifications.items()
    }


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
        (right - left).total_seconds() / 86400
        for left, right in zip(ordered, ordered[1:])
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
        "distinct_terms": sorted({term for row in rows for term in row.terms}),
        "distinct_cusips": len({cusip for row in rows for cusip in row.cusips}),
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
    index = 0
    for current in sorted(primary):
        while index < len(comparator) and comparator[index] < current - window:
            index += 1
        if index < len(comparator) and comparator[index] <= current + window:
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


def _read_csv_times(
    path: str | Path,
    field: str,
    *,
    filters: Mapping[str, str] | None = None,
    group_field: str | None = None,
) -> dict[str, list[datetime]]:
    groups: dict[str, list[datetime]] = defaultdict(list)
    with gzip.open(_repository_path(path), "rt", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if filters and any(row.get(key) != value for key, value in filters.items()):
                continue
            group = row[group_field] if group_field else "primary"
            groups[group].append(parse_time(row[field]))
    return dict(groups)


def load_comparator_times(cfg: Config) -> dict[str, list[datetime]]:
    groups: dict[str, list[datetime]] = {}
    for key, values in _read_csv_times(
        cfg.tadi_clock, "entry_time", filters={"clock_mode": "primary"}
    ).items():
        groups[f"tadi:{key}"] = values
    for key, values in _read_csv_times(
        cfg.dffb_clock, "entry_time_utc", filters={"clock": "primary"}
    ).items():
        groups[f"dffb:{key}"] = values
    for key, values in _read_csv_times(
        cfg.flcc_clock,
        "entry_time",
        filters={"clock_name": "component_concordance_only"},
        group_field="candidate_id",
    ).items():
        groups[f"flcc:{key}"] = values
    for key, values in _read_csv_times(
        cfg.rrp_clock, "entry_time", filters={"clock_mode": "primary"}
    ).items():
        groups[f"rrp:{key}"] = values
    for key, values in _read_csv_times(
        cfg.live_clock, "entry_time", group_field="candidate_id"
    ).items():
        groups[f"live:{key}"] = values
    return groups


def _source_gates(primary: Sequence[ScheduledSignal]) -> dict[str, Any]:
    stats = schedule_stats(primary)
    train = stats["train"]
    selection = stats["selection"]
    checks = {
        "train_total": train["total"] >= 18,
        "selection_total": selection["total"] >= 8,
        "each_train_year": all(train["by_year"].get(str(year), 0) >= 6 for year in (2020, 2021, 2022)),
        "each_selection_half": all(selection["by_half"].get(f"2023H{half}", 0) >= 3 for half in (1, 2)),
        "train_active_quarters": train["active_quarters"] >= 8,
        "train_month_share": train["maximum_month_share"] <= 0.20,
        "selection_month_share": selection["maximum_month_share"] <= 0.20,
        "train_quarter_share": train["maximum_quarter_share"] <= 0.40,
        "selection_quarter_share": selection["maximum_quarter_share"] <= 0.40,
        "train_gap": train["maximum_calendar_gap_days"] <= 90,
        "selection_gap": selection["maximum_calendar_gap_days"] <= 90,
        "both_maturity_groups": all(
            set(row.terms) & BELLY_TERMS and set(row.terms) & LONG_TERMS
            for row in primary
        ),
        "all_results_known_by_signal": all(
            row.latest_result_available <= row.signal_time for row in primary
        ),
        "duplicate_issue_dates": len({row.calendar_date for row in primary}) == len(primary),
        "duplicate_cusips": len([cusip for row in primary for cusip in row.cusips])
        == len({cusip for row in primary for cusip in row.cusips}),
    }
    return {"passed": all(checks.values()), "checks": checks, "stats": stats}


def _specificity(schedules: Mapping[str, Sequence[ScheduledSignal]]) -> dict[str, Any]:
    primary = schedules["primary"]
    metrics = {
        name: overlap_metrics(primary, rows)
        for name, rows in schedules.items()
        if name != "primary"
    }
    caps = {"auction_date_collision": 0.50, "term_year_permutation": 0.50}
    checks = {
        name: metrics[name]["primary_to_comparator_near_containment"] <= cap
        for name, cap in caps.items()
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "caps": caps,
        "metrics": metrics,
        "component_and_superset_controls_report_only": [
            "belly_settlement_calendar",
            "long_settlement_calendar",
            "any_multitenor_settlement",
            "result_time_clock",
        ],
    }


def _novelty(
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
            terms = set(signal.terms)
            rows.append(
                {
                    "candidate": prereg.POLICY_ID,
                    "control": control,
                    "signal_id": signal.signal_id,
                    "split": signal.split,
                    "auction_or_issue_date": signal.calendar_date.isoformat(),
                    "latest_result_available": format_time(signal.latest_result_available),
                    "signal_time": format_time(signal.signal_time),
                    "entry_time": format_time(signal.entry_time),
                    "exit_time": format_time(signal.exit_time),
                    "side": -1,
                    "terms": "|".join(signal.terms),
                    "cusips": "|".join(signal.cusips),
                    "belly_terms": "|".join(sorted(terms & BELLY_TERMS)),
                    "long_terms": "|".join(sorted(terms & LONG_TERMS)),
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


def _validate_inputs(cfg: Config) -> dict[str, Any]:
    if cfg != Config(clock_output=cfg.clock_output, report_output=cfg.report_output):
        raise RuntimeError("TASCC support input configuration is frozen")
    if sha256_file(cfg.preregistration) != PREREGISTRATION_SHA256:
        raise RuntimeError("TASCC preregistration file hash mismatch")
    payload = json.loads(_repository_path(cfg.preregistration).read_text(encoding="utf-8"))
    prereg.validate_preregistration(payload)
    if payload["manifest_hash"] != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("TASCC preregistration manifest mismatch")
    if payload["policy_hash"] != PREREGISTRATION_POLICY_HASH:
        raise RuntimeError("TASCC preregistration policy mismatch")
    source_specs = (
        (cfg.panel, prereg.AUCTION_PANEL_SHA256),
        (cfg.manifest, prereg.AUCTION_MANIFEST_SHA256),
        (cfg.raw_page_0, prereg.RAW_PAGES[0]["sha256"]),
        (cfg.raw_page_1, prereg.RAW_PAGES[1]["sha256"]),
    )
    if any(sha256_file(path) != expected for path, expected in source_specs):
        raise RuntimeError("TASCC source binding mismatch")
    configured = (
        cfg.tadi_clock,
        cfg.dffb_clock,
        cfg.flcc_clock,
        cfg.rrp_clock,
        cfg.live_clock,
    )
    for spec, path in zip(prereg.COMPARATOR_SPECS, configured, strict=True):
        if str(spec["path"]) != path or sha256_file(path) != spec["sha256"]:
            raise RuntimeError(f"TASCC comparator binding mismatch: {spec['name']}")
    return payload


def build_support(cfg: Config = Config()) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prereg_payload = _validate_inputs(cfg)
    auctions, source_stats = load_auctions(cfg)
    schedules = build_control_schedules(auctions)
    source_gates = _source_gates(schedules["primary"])
    specificity = _specificity(schedules)
    novelty = _novelty(schedules["primary"], load_comparator_times(cfg))
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
            "freeze strict pre-2024 evaluator before opening TASCC BTC rows"
            if decision == "PASS_SOURCE"
            else "retire TASCC-72 without TASCC BTC outcomes or parameter repair"
        ),
        "config": asdict(cfg),
        "anchors": {
            "preregistration": str(PREREGISTRATION),
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "preregistration_manifest_hash": prereg_payload["manifest_hash"],
            "preregistration_policy_hash": prereg_payload["policy_hash"],
            "panel_sha256": prereg.AUCTION_PANEL_SHA256,
            "manifest_sha256": prereg.AUCTION_MANIFEST_SHA256,
            "raw_page_sha256": [spec["sha256"] for spec in prereg.RAW_PAGES],
            "builder_source": str(SCRIPT_PATH),
            "builder_sha256": sha256_file(SCRIPT_PATH),
        },
        "source_stats": source_stats,
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
            "panel_value_rows_read": source_stats["panel_rows_read"],
            "raw_transport_rows_parsed": source_stats["raw_transport_rows_parsed"],
            "post_2023_raw_transport_rows_parsed_for_key_filter": source_stats[
                "raw_post_2023_transport_rows_parsed_for_key_filter"
            ],
            "post_2023_rows_materialized_into_tascc": 0,
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "future_return_rows_read": 0,
            "pnl_cagr_mdd_opened": False,
            "network_calls": 0,
            "subprocess_calls": 0,
        },
    }
    report["manifest_hash"] = canonical_hash(report)
    return report, rows


def run(cfg: Config = Config()) -> dict[str, Any]:
    report, rows = build_support(cfg)
    _write_gzip_csv(_repository_path(cfg.clock_output), rows)
    report["clock"]["sha256"] = sha256_file(cfg.clock_output)
    report["manifest_hash"] = canonical_hash(
        {key: value for key, value in report.items() if key != "manifest_hash"}
    )
    _write_json(_repository_path(cfg.report_output), report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock-output", default=str(DEFAULT_CLOCK))
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(Config(clock_output=args.clock_output, report_output=args.report_output))
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
