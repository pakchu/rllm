"""Retire DAIC-168 at source stage without loading BTC outcomes.

The candidate identity was stated before exact incidence was computed, but a
delegated review opened that incidence before a repository preregistration was
committed.  This audit makes the breach and the resulting low support
reproducible.  It is deliberately not an economic evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


POLICY_ID = "DAIC-168-RETIRED-PREOUTCOME"
PROTOCOL_VERSION = "cftc_dealer_absorption_client_consensus_source_audit_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(
    "training/audit_cftc_dealer_absorption_client_consensus_source.py"
)
SOURCE = Path(
    "data/cftc_institutional_transfer_absorption_2018_2023/"
    "cftc_institutional_transfer_absorption_2018_2023.csv.gz"
)
SOURCE_SHA256 = "064eed3fa340b1701f4686d1176de2a10f39128abc5ebf846e8b6319b8144ee6"
BUILD_MANIFEST = SOURCE.parent / "build_manifest.json"
BUILD_MANIFEST_SHA256 = (
    "e9d4ca15da671c086265557e5d302518fd4b9a9ad59fe9d0ff1181d772d60406"
)
SOURCE_MANIFEST = SOURCE.parent / "source_manifest.json"
SOURCE_MANIFEST_SHA256 = (
    "a594b02d1191c32f905c13be3faaa74ec2f3f0e04723d3b11b76ee8b454d6897"
)
DEFAULT_OUTPUT = Path(
    "results/cftc_dealer_absorption_client_consensus_source_rejection_"
    "2026-07-23.json"
)

HOLD = pd.Timedelta(hours=168)
ENTRY_DELAY = pd.Timedelta(minutes=5)
ALLOWED_SOURCE_COLUMNS = (
    "report_date",
    "available_time_utc",
    "official_zip_url",
    "special_publication_override",
    "source_complete",
    "dealer_published_net_change",
    "asset_mgr_published_net_change",
    "lev_money_published_net_change",
)
WINDOWS = {
    "2019_source_history": ("2019-01-01", "2020-01-01"),
    "2020": ("2020-01-01", "2021-01-01"),
    "2021": ("2021-01-01", "2022-01-01"),
    "2022": ("2022-01-01", "2023-01-01"),
    "train_2020_2022": ("2020-01-01", "2023-01-01"),
    "2023_h1": ("2023-01-01", "2023-07-01"),
    "2023_h2": ("2023-07-01", "2024-01-01"),
    "selection_2023": ("2023-01-01", "2024-01-01"),
}


@dataclass(frozen=True)
class SourceRow:
    report_date: str
    available_time: pd.Timestamp
    dealer_change: float
    asset_manager_change: float
    leveraged_money_change: float
    official_zip_url: str
    special_publication_override: bool
    source_complete: bool


@dataclass(frozen=True)
class Event:
    report_date: str
    signal_time: str
    entry_time: str
    exit_time: str
    side: int
    dealer_change: float
    asset_manager_change: float
    leveraged_money_change: float
    special_publication_override: bool


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
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _timestamp(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result is pd.NaT:
        raise RuntimeError("DAIC timestamp is NaT")
    if result.tzinfo is None:
        result = result.tz_localize("UTC")
    else:
        result = result.tz_convert("UTC")
    return result


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def verify_source() -> None:
    expected = {
        SOURCE: SOURCE_SHA256,
        BUILD_MANIFEST: BUILD_MANIFEST_SHA256,
        SOURCE_MANIFEST: SOURCE_MANIFEST_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise RuntimeError(f"DAIC source hash mismatch: {path}")


def load_source() -> list[SourceRow]:
    verify_source()
    frame = pd.read_csv(_repository_path(SOURCE), dtype=str)
    if not set(ALLOWED_SOURCE_COLUMNS).issubset(frame.columns):
        raise RuntimeError("DAIC source schema changed")
    rows: list[SourceRow] = []
    for _, raw in frame.loc[:, list(ALLOWED_SOURCE_COLUMNS)].iterrows():
        complete = str(raw["source_complete"]).lower() == "true"
        values = (
            float(raw["dealer_published_net_change"]),
            float(raw["asset_mgr_published_net_change"]),
            float(raw["lev_money_published_net_change"]),
        ) if complete else (0.0, 0.0, 0.0)
        rows.append(
            SourceRow(
                report_date=str(raw["report_date"]),
                available_time=_timestamp(raw["available_time_utc"]),
                dealer_change=values[0],
                asset_manager_change=values[1],
                leveraged_money_change=values[2],
                official_zip_url=str(raw["official_zip_url"]),
                special_publication_override=(
                    str(raw["special_publication_override"]).lower() == "true"
                ),
                source_complete=complete,
            )
        )
    if len(rows) != 299 or sum(row.source_complete for row in rows) != 298:
        raise RuntimeError("DAIC source row/quarantine count changed")
    if any(
        current.available_time <= previous.available_time
        for previous, current in zip(rows, rows[1:])
    ):
        raise RuntimeError("DAIC availability clock is not strictly increasing")
    return rows


def daic_side(row: SourceRow) -> int:
    dealer = _sign(row.dealer_change)
    asset = _sign(row.asset_manager_change)
    leveraged = _sign(row.leveraged_money_change)
    if asset and asset == leveraged and dealer == -asset:
        return asset
    return 0


def cita_side(row: SourceRow) -> int:
    asset = _sign(row.asset_manager_change)
    leveraged = _sign(row.leveraged_money_change)
    return asset if asset and asset == -leveraged else 0


def _event(row: SourceRow, side: int) -> Event:
    signal = row.available_time
    entry = signal + ENTRY_DELAY
    exit_time = entry + HOLD
    return Event(
        report_date=row.report_date,
        signal_time=signal.isoformat(),
        entry_time=entry.isoformat(),
        exit_time=exit_time.isoformat(),
        side=side,
        dealer_change=row.dealer_change,
        asset_manager_change=row.asset_manager_change,
        leveraged_money_change=row.leveraged_money_change,
        special_publication_override=row.special_publication_override,
    )


def raw_events(rows: Iterable[SourceRow], side_function=daic_side) -> list[Event]:
    return [
        _event(row, side)
        for row in rows
        if row.source_complete and (side := side_function(row))
    ]


def nonoverlapping(events: Iterable[Event]) -> list[Event]:
    accepted: list[Event] = []
    prior_exit: pd.Timestamp | None = None
    for event in sorted(events, key=lambda item: item.entry_time):
        entry = _timestamp(event.entry_time)
        if prior_exit is not None and entry < prior_exit:
            continue
        accepted.append(event)
        prior_exit = _timestamp(event.exit_time)
    return accepted


def _contained(events: Iterable[Event], start: str, end: str) -> list[Event]:
    lower = _timestamp(start)
    upper = _timestamp(end)
    return [
        event
        for event in events
        if _timestamp(event.entry_time) >= lower
        and _timestamp(event.exit_time) <= upper
    ]


def _summary(events: Iterable[Event], start: str, end: str) -> dict[str, Any]:
    selected = _contained(events, start, end)
    months: dict[str, int] = {}
    for event in selected:
        month = event.entry_time[:7]
        months[month] = months.get(month, 0) + 1
    return {
        "events": len(selected),
        "longs": sum(event.side == 1 for event in selected),
        "shorts": sum(event.side == -1 for event in selected),
        "special_publication_overrides": sum(
            event.special_publication_override for event in selected
        ),
        "max_single_month_share": (
            max(months.values(), default=0) / len(selected) if selected else 0.0
        ),
    }


def _jaccard(left: Iterable[Event], right: Iterable[Event]) -> float:
    a = {event.entry_time for event in left}
    b = {event.entry_time for event in right}
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def build_report() -> dict[str, Any]:
    rows = load_source()
    daic_raw = raw_events(rows, daic_side)
    daic = nonoverlapping(daic_raw)
    cita_raw = raw_events(rows, cita_side)
    cita = nonoverlapping(cita_raw)
    summaries = {
        name: _summary(daic, *window) for name, window in WINDOWS.items()
    }
    inherited_reference = {
        "source": (
            "unchanged CITA-1 preregistered Stage1 statistical-support floor"
        ),
        "minimum_train_trades": 75,
        "minimum_each_train_year_trades": 20,
        "used_for_rejection_only": True,
        "not_a_post_incidence_repair_grid": True,
    }
    checks = {
        "source_hashes_match": True,
        "source_rows_exactly_299": len(rows) == 299,
        "source_complete_rows_exactly_298": sum(
            row.source_complete for row in rows
        ) == 298,
        "one_source_row_quarantined": sum(
            not row.source_complete for row in rows
        ) == 1,
        "daic_and_cita_raw_report_dates_disjoint": not (
            {event.report_date for event in daic_raw}
            & {event.report_date for event in cita_raw}
        ),
        "daic_and_cita_accepted_entries_disjoint": not (
            {event.entry_time for event in daic}
            & {event.entry_time for event in cita}
        ),
        "entry_waits_exactly_five_minutes": all(
            _timestamp(event.entry_time)
            == _timestamp(event.signal_time) + ENTRY_DELAY
            for event in daic
        ),
        "hold_is_exactly_168_hours": all(
            _timestamp(event.exit_time) - _timestamp(event.entry_time) == HOLD
            for event in daic
        ),
        "accepted_events_do_not_overlap": all(
            current.entry_time >= previous.exit_time
            for previous, current in zip(daic, daic[1:])
        ),
        "btc_market_rows_opened_zero": True,
        "funding_rows_opened_zero": True,
        "performance_values_opened_false": True,
    }
    train = summaries["train_2020_2022"]
    year_counts = {year: summaries[year]["events"] for year in ("2020", "2021", "2022")}
    rejection_reasons = [
        {
            "code": "FORMAL_PREREGISTRATION_BOUNDARY_BREACHED",
            "detail": (
                "a delegated source reviewer computed exact DAIC incidence before "
                "a repository preregistration artifact was committed"
            ),
        },
        {
            "code": "TRAIN_SUPPORT_BELOW_INHERITED_CFTC_FLOOR",
            "observed": train["events"],
            "required": inherited_reference["minimum_train_trades"],
        },
        {
            "code": "EVERY_TRAIN_YEAR_BELOW_INHERITED_CFTC_FLOOR",
            "observed": year_counts,
            "required_each": inherited_reference[
                "minimum_each_train_year_trades"
            ],
        },
    ]
    core: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": POLICY_ID,
        "candidate_identity_stated_before_incidence": {
            "asset_manager_and_leveraged_net_changes": "same nonzero sign",
            "dealer_net_change": "opposite sign",
            "side": "follow asset-manager/leveraged-money consensus",
            "signal_time": "frozen conservative CFTC available_time_utc",
            "entry_delay_minutes": 5,
            "hold_elapsed_hours": 168,
            "notional_exposure": 0.5,
            "global_nonoverlap": True,
        },
        "research_integrity": {
            "formal_preregistration_committed_before_exact_incidence": False,
            "exact_source_incidence_opened": True,
            "btc_outcomes_opened_for_daic": False,
            "performance_values_opened_for_daic": False,
            "conservative_action": "retire unchanged candidate before outcomes",
        },
        "source_binding": {
            "panel": str(SOURCE),
            "panel_sha256": SOURCE_SHA256,
            "build_manifest": str(BUILD_MANIFEST),
            "build_manifest_sha256": BUILD_MANIFEST_SHA256,
            "source_manifest": str(SOURCE_MANIFEST),
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "allowed_columns": list(ALLOWED_SOURCE_COLUMNS),
        },
        "source_incidence": {
            "daic_raw_events": len(daic_raw),
            "daic_accepted_events": len(daic),
            "daic_suppressed_overlap_events": len(daic_raw) - len(daic),
            "cita_raw_events": len(cita_raw),
            "cita_accepted_events": len(cita),
            "daic_cita_raw_report_overlap": len(
                {event.report_date for event in daic_raw}
                & {event.report_date for event in cita_raw}
            ),
            "daic_cita_accepted_entry_jaccard": _jaccard(daic, cita),
            "windows": summaries,
        },
        "inherited_statistical_support_reference": inherited_reference,
        "checks": checks,
        "rejection_reasons": rejection_reasons,
        "outcome_boundary": {
            "source_rows_read": len(rows),
            "btc_market_rows_read": 0,
            "funding_rows_read": 0,
            "return_rows_read": 0,
            "pnl_cagr_mdd_opened": False,
        },
        "disposition": "REJECT_BEFORE_OUTCOMES_NO_REPAIR",
        "next_action": (
            "leave the CFTC same-sign/dealer-opposition neighborhood; select a "
            "new source-native mechanism before reading its incidence"
        ),
        "audit_source": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
    }
    if not all(checks.values()):
        raise RuntimeError("DAIC source audit invariant failed")
    core["manifest_hash"] = canonical_hash(core)
    return core


def write_report(path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = _repository_path(path)
    report = build_report()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    report = write_report(args.output)
    print(
        json.dumps(
            {
                "candidate": report["candidate"],
                "disposition": report["disposition"],
                "train_events": report["source_incidence"]["windows"]
                ["train_2020_2022"]["events"],
                "selection_events": report["source_incidence"]["windows"]
                ["selection_2023"]["events"],
                "btc_outcomes_opened": report["research_integrity"]
                ["btc_outcomes_opened_for_daic"],
                "manifest_hash": report["manifest_hash"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
