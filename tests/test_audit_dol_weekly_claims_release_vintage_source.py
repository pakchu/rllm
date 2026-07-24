from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest

from training import audit_dol_weekly_claims_release_vintage_source as audit


FIXED_NOW = datetime(2026, 7, 24, 0, 0, tzinfo=timezone.utc)
FUTURE_CANARY = "FUTURE_2026_BODY_MUST_REMAIN_OPAQUE_9713"


def _modern_teaser(
    *,
    week: str = "December 27",
    current: int = 199_000,
    comparison: int = 215_000,
    current_average: int = 218_750,
    average_comparison: int = 217_000,
) -> str:
    difference = current - comparison
    average_difference = current_average - average_comparison

    def phrase(value: int) -> str:
        if value > 0:
            return f"an increase of {value:,}"
        if value < 0:
            return f"a decrease of {-value:,}"
        return "unchanged"

    return (
        f"In the week ending {week}, the advance figure for seasonally "
        f"adjusted initial claims was {current:,}, {phrase(difference)} "
        "from the previous week's revised level. The previous week's level "
        f"was revised up by 1,000 from {comparison - 1_000:,} to "
        f"{comparison:,}. The 4-week moving average was "
        f"{current_average:,}, {phrase(average_difference)} from the "
        "previous week's revised average. The previous week's average was "
        f"revised up by 250 from {average_comparison - 250:,} to "
        f"{average_comparison:,}."
    )


def _legacy_teaser() -> str:
    return (
        "SEASONALLY ADJUSTED DATA In the week ending Jan. 7, the advance "
        "figure for seasonally adjusted initial claims was 399,000, a "
        "decrease of 19,000 from the previous week's revised figure of "
        "418,000. The 4-week moving average was 381,750, an increase of "
        "3,750 from the previous week's revised average of 378,000."
    )


def _teaser_block(
    *,
    node_id: int,
    release_date: str,
    path_date: str,
    title: str = "Unemployment Insurance Weekly Claims Report",
    body: str,
    suffix: str = "",
) -> bytes:
    return (
        f'<div data-history-node-id="{node_id}" '
        f'about="/newsroom/releases/eta/eta{path_date}{suffix}" '
        'class="left-teaser-text">'
        f'<p class="dol-date-text">{release_date}</p>'
        f'<a href="/newsroom/releases/eta/eta{path_date}{suffix}">'
        f"<h3><span>{title}</span></h3></a>"
        '<div class="field field--name-field-press-body '
        'field--type-text-with-summary field--label-hidden clearfix">'
        f"<p>{body}</p></div></div>"
    ).encode("utf-8")


def _newsroom_page(
    records: list[bytes],
    *,
    current_page: int,
    next_page: int | None,
) -> bytes:
    next_link = ""
    if next_page is not None:
        next_link = (
            f'<a class="page-link" href="?page={next_page}" '
            'title="Go to next page" rel="next">next</a>'
        )
    return (
        b"<!DOCTYPE html><html><body>"
        + b"".join(records)
        + f'<nav id="pag1">{next_link}</nav></body></html>'.encode()
    )


def _archive_inventory(year: int, links: list[str]) -> bytes:
    cells = "".join(f'<a href="{link}">x</a>' for link in links)
    return (
        f"<!DOCTYPE html><html><body><h1>{year}</h1>{cells}</body></html>"
    ).encode()


def _saturdays(year: int, count: int) -> list[date]:
    value = date(year, 1, 1)
    while value.weekday() != 5:
        value += timedelta(days=1)
    rows: list[date] = []
    while value.year == year and len(rows) < count:
        rows.append(value)
        value += timedelta(days=7)
    assert len(rows) == count
    return rows


def _state_inventory_fixture(year: int) -> bytes:
    links = [
        f"/unemploy/page8/{year}/{value:%m%d%y}.html"
        for value in _saturdays(year, audit.EXPECTED_STATE_TABLE_COUNTS[year])
    ]
    links.extend(audit.FROZEN_MALFORMED_PATHS.get(year, ()))
    return _archive_inventory(year, links)


def _state_table_html(
    initial_week: date,
    *,
    legacy: bool = False,
    row_mutator: Callable[
        [str, list[str]], tuple[str, list[str]]
    ]
    | None = None,
    total_mutator: Callable[[list[str]], list[str]] | None = None,
) -> bytes:
    insured_week = initial_week - timedelta(days=7)
    jurisdiction_rows: list[tuple[str, list[str]]] = []
    totals = [0] * 12
    for index, name in enumerate(audit.JURISDICTIONS, start=1):
        raw_values = [
            str(100 + index),
            str(index - 27),
            str(27 - index),
            str(index % 5),
            str(index % 3),
            str(1_000 + index),
            f"{1 + (index % 25) / 10:.1f}",
            str(index - 20),
            str(20 - index),
            str(index % 7),
            str(index % 4),
            str(1_000 + index + index % 7 + index % 4),
        ]
        if row_mutator is not None:
            name, raw_values = row_mutator(name, list(raw_values))
        jurisdiction_rows.append((name, raw_values))
        for column, value in enumerate(raw_values):
            if column == audit.RATE_COLUMN:
                continue
            totals[column] += int(value.replace(",", ""))
    total_values = [str(value) for value in totals]
    total_values[audit.RATE_COLUMN] = "2.0"
    if total_mutator is not None:
        total_values = total_mutator(total_values)

    def row(name: str, values: list[str], *, total: bool = False) -> str:
        spacer_at = 5
        cells = []
        for index, value in enumerate(values):
            if index == spacer_at:
                cells.append("<td>&nbsp;</td>")
            cells.append(f"<td>{value}</td>")
        row_id = "totals" if total else name.replace(" ", "_")
        return f'<tr><th id="{row_id}">{name}</th>{"".join(cells)}</tr>'

    rows = "".join(row(name, values) for name, values in jurisdiction_rows)
    rows += row("Totals", total_values, total=True)
    if legacy:
        initial_heading = (
            "Initial Claims Filed During Week Ended "
            f"{initial_week:%B} {initial_week.day}, {initial_week.year}"
        )
        insured_heading = (
            "Insured Unemployment For Week Ended "
            f"{insured_week:%B} {insured_week.day}, {insured_week.year}"
        )
        wrapper = "<FONT SIZE='-2'>"
    else:
        initial_heading = (
            "INITIAL CLAIMS FILED DURING WEEK ENDED "
            f"{initial_week:%B} {initial_week.day}"
        )
        insured_heading = (
            "INSURED UNEMPLOYMENT FOR WEEK ENDED "
            f"{insured_week:%B} {insured_week.day}"
        )
        wrapper = ""
    return (
        "<!DOCTYPE html><html><body><table>"
        f"<tr><th>{initial_heading}</th><th>{insured_heading}</th></tr>"
        f"{wrapper}{rows}</table>"
        "<p>Figures Appearing In Cols. Showing Over-The-Week Changes "
        "Reflect All Revisions In Data For Prior Week Submitted By State "
        "Agencies</p></body></html>"
    ).encode()


def _payload(
    intent: audit.RequestIntent,
    *,
    status: int = 200,
    raw: bytes = b"<html></html>",
    content_type: str = "text/html; charset=UTF-8",
    extra_headers: tuple[tuple[str, str], ...] = (),
) -> audit.HttpPayload:
    return audit.HttpPayload(
        intent=intent,
        status=status,
        headers=(
            ("Content-Type", content_type),
            ("Content-Length", str(len(raw))),
            *extra_headers,
        ),
        raw=raw,
        request_started_at_utc=FIXED_NOW,
        response_completed_at_utc=FIXED_NOW,
    )


def _national(
    release_date: date,
    *,
    week_ending: date | None = None,
    digest_seed: str = "n",
) -> audit.NationalRelease:
    week_ending = week_ending or audit.previous_saturday(release_date)
    return audit.NationalRelease(
        node_id=int(release_date.strftime("%Y%m%d")),
        path=f"/newsroom/releases/eta/eta{release_date:%Y%m%d}",
        release_date=release_date,
        availability_at_utc=audit.historical_availability(release_date),
        week_ending=week_ending,
        current_initial_claims=200_000,
        signed_change=-1_000,
        prior_status="revised",
        prior_before=200_000,
        prior_after=201_000,
        current_four_week_average=205_000,
        signed_average_change=-250,
        prior_average_before=205_000,
        prior_average_after=205_250,
        teaser_sha256=audit.sha256_bytes(
            f"{digest_seed}-{release_date}".encode()
        ),
        source_page_sha256=audit.sha256_bytes(
            f"page-{release_date}".encode()
        ),
        grammar_variant="synthetic",
    )


def _state_summary(week_ending: date) -> audit.StateTableSummary:
    digest = audit.sha256_bytes(f"state-{week_ending}".encode())
    return audit.StateTableSummary(
        week_ending=week_ending,
        insured_week_ending=week_ending - timedelta(days=7),
        path=f"/unemploy/page8/{week_ending.year}/{week_ending:%m%d%y}.html",
        raw_sha256=digest,
        structural_sha256=audit.sha256_bytes(
            f"struct-{week_ending}".encode()
        ),
        jurisdiction_count=53,
        arithmetic_columns_reconciled=11,
        grammar_variant="synthetic",
    )


def _production_calendar() -> list[audit.NationalRelease]:
    rows: list[audit.NationalRelease] = []
    for year in range(2012, 2025):
        value = date(year, 1, 1)
        while value.year == year:
            if value.weekday() == 3:
                rows.append(_national(value))
            value += timedelta(days=1)

    value = date(2025, 1, 1)
    dates: list[date] = []
    while value.year == 2025:
        if value.weekday() == 3 and not (
            date(2025, 10, 2) <= value <= date(2025, 11, 13)
        ):
            dates.append(value)
        value += timedelta(days=1)
    dates.remove(date(2025, 11, 27))
    dates.append(date(2025, 11, 26))
    dates.remove(date(2025, 12, 25))
    dates.extend((date(2025, 12, 24), date(2025, 12, 31)))
    assert len(dates) == 46
    rows.extend(_national(value) for value in sorted(dates))
    return sorted(rows, key=lambda row: row.release_date)


def _production_state_summaries(
    national: list[audit.NationalRelease],
) -> list[audit.StateTableSummary]:
    by_year: dict[int, list[date]] = {year: [] for year in range(2012, 2026)}
    for row in national:
        if row.week_ending.year in by_year:
            by_year[row.week_ending.year].append(row.week_ending)
    rows: list[audit.StateTableSummary] = []
    for year, count in audit.EXPECTED_STATE_TABLE_COUNTS.items():
        unique = sorted(set(by_year[year]))
        assert len(unique) >= count
        rows.extend(_state_summary(value) for value in unique[:count])
    return rows


def test_boundary_constants_and_source_identity_are_frozen() -> None:
    assert audit.sha256_file(audit.BOUNDARY_PATH) == audit.BOUNDARY_SHA256
    assert audit.SOURCE_START == date(2012, 1, 1)
    assert audit.SOURCE_END == date(2026, 1, 1)
    assert audit.ALLOWED_HOSTS == frozenset(
        {"oui.doleta.gov", "www.dol.gov"}
    )
    assert audit.REQUEST_HEADERS == (
        (
            "User-Agent",
            "rllm-dol-wcrv-source-audit/1.0 "
            "(https://github.com/pakchu/rllm)",
        ),
        ("Accept", "text/html"),
        ("Accept-Encoding", "identity"),
    )
    assert sum(audit.EXPECTED_STATE_TABLE_COUNTS.values()) == 692


def test_schedule_page_requires_exact_publication_statement() -> None:
    raw = (
        b"<html><body>The UI Weekly Claims News Release is published each "
        b"week on Thursday morning at 8:30am EST. Exceptions to this "
        b"schedule occur when a Thursday falls on a Federal Holiday.</body></html>"
    )
    result = audit.parse_schedule_page(raw)
    assert result["normal_release_time"] == "08:30 America/New_York"
    with pytest.raises(audit.SourceContractError):
        audit.parse_schedule_page(raw.replace(b"8:30am", b"9:30am"))


def test_newsroom_scanner_keeps_2026_body_opaque_and_parses_2025() -> None:
    raw = _newsroom_page(
        [
            _teaser_block(
                node_id=1,
                release_date="January 8, 2026",
                path_date="20260108",
                body=FUTURE_CANARY
                + " In the week ending January 3, claims were 999,999.",
            ),
            _teaser_block(
                node_id=2,
                release_date="December 31, 2025",
                path_date="20251231",
                body=_modern_teaser(),
            ),
        ],
        current_page=0,
        next_page=1,
    )
    page = audit.parse_newsroom_page(raw, expected_page=0)
    assert page.next_page == 1
    assert page.opaque_postwindow_count == 1
    assert len(page.releases) == 1
    assert page.releases[0].week_ending == date(2025, 12, 27)
    rendered = audit.canonical_json_bytes(page.aggregate_dict())
    assert FUTURE_CANARY.encode() not in rendered
    assert b"999999" not in rendered


@pytest.mark.parametrize("bad", [b"\xff", b"\x00", b"\xef\xbf\xbd"])
def test_newsroom_byte_validator_rejects_invalid_page_encoding(
    bad: bytes,
) -> None:
    raw = _newsroom_page(
        [
            _teaser_block(
                node_id=1,
                release_date="January 8, 2026",
                path_date="20260108",
                body=FUTURE_CANARY,
            )
        ],
        current_page=0,
        next_page=None,
    ).replace(FUTURE_CANARY.encode(), bad)
    with pytest.raises(audit.SourceContractError):
        audit.parse_newsroom_page(raw, expected_page=0)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda raw: raw.replace(b'?page=1"', b'?published_at=2025&page=1"'),
        lambda raw: raw.replace(b'rel="next"', b'rel="next previous"'),
        lambda raw: raw.replace(b"?page=1", b"?page=2"),
        lambda raw: raw.replace(
            b'<nav id="pag1">',
            b'<nav id="pag1"><a href="?page=1" rel="next">dup</a>',
        ),
    ],
)
def test_newsroom_pagination_rejects_noncanonical_next_chain(
    mutation: Callable[[bytes], bytes],
) -> None:
    raw = _newsroom_page(
        [
            _teaser_block(
                node_id=2,
                release_date="December 31, 2025",
                path_date="20251231",
                body=_modern_teaser(),
            )
        ],
        current_page=0,
        next_page=1,
    )
    with pytest.raises(audit.NewsroomError):
        audit.parse_newsroom_page(mutation(raw), expected_page=0)


def test_newsroom_chain_requires_contiguous_nonoverlapping_crossing() -> None:
    first = audit.parse_newsroom_page(
        _newsroom_page(
            [
                _teaser_block(
                    node_id=1,
                    release_date="December 31, 2025",
                    path_date="20251231",
                    body=_modern_teaser(),
                )
            ],
            current_page=0,
            next_page=1,
        ),
        expected_page=0,
    )
    last = audit.parse_newsroom_page(
        _newsroom_page(
            [
                _teaser_block(
                    node_id=2,
                    release_date="December 29, 2011",
                    path_date="20111229",
                    title="Other ETA release",
                    body="opaque old body",
                )
            ],
            current_page=1,
            next_page=2,
        ),
        expected_page=1,
    )
    audit.validate_newsroom_chain([first, last])
    with pytest.raises(audit.NewsroomError):
        audit.validate_newsroom_chain([replace(first, next_page=2), last])
    with pytest.raises(audit.NewsroomError):
        audit.validate_newsroom_chain(
            [first, replace(last, maximum_displayed_date=date(2026, 1, 1))]
        )
    with pytest.raises(audit.NewsroomError):
        audit.validate_newsroom_chain([first])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"release_date": "December 30, 2025"},
        {"path_date": "20251230"},
        {"suffix": "-1"},
        {"title": "Unemployment Claims Weekly Report"},
    ],
)
def test_newsroom_record_identity_is_exact(kwargs: dict[str, str]) -> None:
    values = {
        "node_id": 2,
        "release_date": "December 31, 2025",
        "path_date": "20251231",
        "body": _modern_teaser(),
    }
    values.update(kwargs)
    raw = _newsroom_page(
        [_teaser_block(**values)],
        current_page=0,
        next_page=None,
    )
    if kwargs == {"suffix": "-1"}:
        assert len(audit.parse_newsroom_page(raw, expected_page=0).releases) == 1
    elif "title" in kwargs:
        assert audit.parse_newsroom_page(raw, expected_page=0).releases == ()
    else:
        with pytest.raises(audit.NewsroomError):
            audit.parse_newsroom_page(raw, expected_page=0)


def test_modern_and_legacy_teasers_parse_without_later_revision_imputation() -> None:
    modern = audit.parse_national_teaser(
        _modern_teaser().encode(),
        release_date=date(2025, 12, 31),
        node_id=1,
        path="/newsroom/releases/eta/eta20251231",
        source_page_sha256="a" * 64,
    )
    assert modern.week_ending == date(2025, 12, 27)
    assert modern.current_initial_claims == 199_000
    assert modern.signed_change == -16_000
    assert modern.prior_before == 214_000
    assert modern.prior_after == 215_000

    legacy = audit.parse_national_teaser(
        _legacy_teaser().encode(),
        release_date=date(2012, 1, 12),
        node_id=2,
        path="/newsroom/releases/eta/eta20120112",
        source_page_sha256="b" * 64,
    )
    assert legacy.week_ending == date(2012, 1, 7)
    assert legacy.prior_status == "revised"
    assert legacy.prior_before is None
    assert legacy.prior_after == 418_000
    assert legacy.prior_average_before is None
    assert legacy.prior_average_after == 378_000


def test_teaser_arithmetic_mismatch_rejects() -> None:
    raw = _modern_teaser().replace(
        "a decrease of 16,000", "a decrease of 15,000"
    )
    with pytest.raises(audit.ArithmeticContractError):
        audit.parse_national_teaser(
            raw.encode(),
            release_date=date(2025, 12, 31),
            node_id=1,
            path="/newsroom/releases/eta/eta20251231",
            source_page_sha256="a" * 64,
        )


def test_state_inventory_freezes_counts_and_excludes_only_known_2023_error() -> None:
    for year in range(2012, 2026):
        result = audit.parse_state_inventory(
            _state_inventory_fixture(year),
            requested_year=year,
        )
        assert len(result.artifacts) == audit.EXPECTED_STATE_TABLE_COUNTS[year]
        assert result.malformed_link_count == len(
            audit.FROZEN_MALFORMED_PATHS.get(year, ())
        )
        assert all(
            row.path not in {
                value
                for paths in audit.FROZEN_MALFORMED_PATHS.values()
                for value in paths
            }
            for row in result.artifacts
        )


@pytest.mark.parametrize(
    "bad_link",
    [
        "/unemploy/page8/2023/023023.html",
        "/unemploy/page8/2023/020223.html",
        "/unemploy/page8/2023/020423.xml",
        "/unemploy/page8/2024/020423.html",
        "/unemploy/page8/2023/020423.html?x=1",
        "https://oui.doleta.gov/unemploy/page8/2023/020423.html",
        "//oui.doleta.gov/unemploy/page8/2023/020423.html",
        "/UNEMPLOY/PAGE8/2023/020423.html",
        "/unemploy/%70age8/2023/020423.html",
    ],
)
def test_state_inventory_rejects_any_other_malformed_link(
    bad_link: str,
) -> None:
    raw = _state_inventory_fixture(2023).replace(
        audit.FROZEN_MALFORMED_2023_PATH.encode(),
        bad_link.encode(),
    )
    with pytest.raises(audit.StateInventoryError):
        audit.parse_state_inventory(raw, requested_year=2023)


@pytest.mark.parametrize("legacy", [False, True])
def test_state_table_parses_53_by_12_and_reconciles_totals(
    legacy: bool,
) -> None:
    week = date(2025, 1, 4)
    raw = _state_table_html(week, legacy=legacy)
    result = audit.parse_state_table(
        raw,
        path="/unemploy/page8/2025/010425.html",
    )
    assert result.week_ending == week
    assert result.insured_week_ending == date(2024, 12, 28)
    assert result.jurisdiction_count == 53
    assert result.arithmetic_columns_reconciled == 11


@pytest.mark.parametrize(
    "row_mutator,total_mutator",
    [
        (
            lambda name, values: (
                ("Alabama" if name == "Alaska" else name),
                values,
            ),
            None,
        ),
        (
            lambda name, values: (
                name,
                values[:-1] if name == "Alabama" else values,
            ),
            None,
        ),
        (
            lambda name, values: (
                name,
                [("-1" if index == 0 else value) for index, value in enumerate(values)]
                if name == "Alabama"
                else values,
            ),
            None,
        ),
        (
            lambda name, values: (
                name,
                [("100.1" if index == audit.RATE_COLUMN else value)
                 for index, value in enumerate(values)]
                if name == "Alabama"
                else values,
            ),
            None,
        ),
        (
            None,
            lambda values: [
                str(int(value) + 1) if index == 0 else value
                for index, value in enumerate(values)
            ],
        ),
    ],
)
def test_state_table_rejects_structure_numeric_and_total_errors(
    row_mutator: Callable[[str, list[str]], tuple[str, list[str]]] | None,
    total_mutator: Callable[[list[str]], list[str]] | None,
) -> None:
    raw = _state_table_html(
        date(2025, 1, 4),
        row_mutator=row_mutator,
        total_mutator=total_mutator,
    )
    with pytest.raises(audit.StateTableError):
        audit.parse_state_table(
            raw,
            path="/unemploy/page8/2025/010425.html",
        )


def test_historical_clock_is_exact_across_dst_and_holiday_dates() -> None:
    assert audit.historical_availability(date(2025, 1, 2)).isoformat() == (
        "2025-01-02T14:00:00+00:00"
    )
    assert audit.historical_availability(date(2025, 7, 3)).isoformat() == (
        "2025-07-03T13:00:00+00:00"
    )
    assert audit.historical_availability(date(2025, 11, 26)).isoformat() == (
        "2025-11-26T14:00:00+00:00"
    )


def test_full_support_keeps_missing_state_tables_without_deleting_national() -> None:
    national = _production_calendar()
    tables = _production_state_summaries(national)
    support = audit.evaluate_support(
        national,
        tables,
        inventory_counts=audit.EXPECTED_STATE_TABLE_COUNTS,
        malformed_link_counts={
            year: len(audit.FROZEN_MALFORMED_PATHS.get(year, ()))
            for year in range(2012, 2026)
        },
    )
    assert support["decision"] == "SOURCE_SUPPORT_PASS"
    assert support["national_release_count"] == len(national)
    assert support["state_table_count"] == 692
    assert support["national_release_count"] > support["state_table_count"]
    assert support["state_missing_event_count"] > 0

    failed = audit.evaluate_support(
        national,
        [
            row
            for row in tables
            if row.week_ending.year != 2019
        ],
        inventory_counts=audit.EXPECTED_STATE_TABLE_COUNTS,
        malformed_link_counts={
            year: len(audit.FROZEN_MALFORMED_PATHS.get(year, ()))
            for year in range(2012, 2026)
        },
    )
    assert failed["decision"] == "TERMINAL_REJECT"


def test_support_rejects_duplicate_teaser_and_state_artifact_hashes() -> None:
    national = _production_calendar()
    tables = _production_state_summaries(national)
    malformed = {
        year: len(audit.FROZEN_MALFORMED_PATHS.get(year, ()))
        for year in range(2012, 2026)
    }
    duplicate_teaser = list(national)
    duplicate_teaser[1] = replace(
        duplicate_teaser[1],
        teaser_sha256=duplicate_teaser[0].teaser_sha256,
    )
    result = audit.evaluate_support(
        duplicate_teaser,
        tables,
        inventory_counts=audit.EXPECTED_STATE_TABLE_COUNTS,
        malformed_link_counts=malformed,
    )
    assert result["decision"] == "TERMINAL_REJECT"
    assert result["checks"]["national_unique_identity"] is False

    duplicate_table = list(tables)
    duplicate_table[1] = replace(
        duplicate_table[1],
        raw_sha256=duplicate_table[0].raw_sha256,
    )
    result = audit.evaluate_support(
        national,
        duplicate_table,
        inventory_counts=audit.EXPECTED_STATE_TABLE_COUNTS,
        malformed_link_counts=malformed,
    )
    assert result["decision"] == "TERMINAL_REJECT"
    assert result["checks"]["state_table_unique_artifacts"] is False


class _RetryTransport:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.intents: list[audit.RequestIntent] = []

    def request(self, intent: audit.RequestIntent) -> audit.HttpPayload:
        self.intents.append(intent)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        assert isinstance(outcome, audit.HttpPayload)
        return replace(outcome, intent=intent)


def test_retry_scope_identity_and_fixed_delays() -> None:
    intent = audit.make_get_intent(audit.SCHEDULE_URL)
    success = _payload(intent, raw=b"<html>ok</html>")
    transport = _RetryTransport(
        [
            audit.RetryableTransportError("timeout"),
            _payload(intent, status=503, raw=b""),
            success,
        ]
    )
    sleeps: list[float] = []
    result = audit.request_with_retries(
        transport,
        intent,
        body_cap=1_000,
        sleep=sleeps.append,
    )
    assert result.status == 200
    assert sleeps == [1.0, 2.0]
    assert transport.intents == [intent, intent, intent]

    forbidden = _RetryTransport([_payload(intent, status=403, raw=b"deny")])
    with pytest.raises(audit.TransportError):
        audit.request_with_retries(
            forbidden,
            intent,
            body_cap=1_000,
            sleep=lambda _: None,
        )
    assert len(forbidden.intents) == 1


@pytest.mark.parametrize(
    "payload",
    [
        lambda intent: _payload(
            intent,
            raw=b"<html/>",
            extra_headers=(("Content-Length", "7"),),
        ),
        lambda intent: _payload(
            intent,
            raw=b"<html/>",
            extra_headers=(("Content-Encoding", "gzip"),),
        ),
        lambda intent: _payload(
            intent,
            raw=b"<html/>",
            extra_headers=(("Content-Range", "bytes 0-6/7"),),
        ),
        lambda intent: _payload(
            intent,
            raw=b"<html/>",
            content_type="application/json",
        ),
    ],
)
def test_http_validation_rejects_ambiguous_or_encoded_payloads(
    payload: Callable[[audit.RequestIntent], audit.HttpPayload],
) -> None:
    intent = audit.make_get_intent(audit.SCHEDULE_URL)
    transport = _RetryTransport([payload(intent)])
    with pytest.raises(audit.TransportError):
        audit.request_with_retries(
            transport,
            intent,
            body_cap=100,
            sleep=lambda _: None,
        )


@pytest.mark.parametrize(
    "url",
    [
        "http://oui.doleta.gov/unemploy/claims_arch.asp",
        "https://user@oui.doleta.gov/unemploy/claims_arch.asp",
        "https://oui.doleta.gov:443/unemploy/claims_arch.asp",
        "https://www.dol.gov/newsroom/releases/eta?published_at=2025",
        "https://oui.doleta.gov/unemploy/page8/2025/010425.html?x=1",
        "https://example.com/unemploy/page8/2025/010425.html",
    ],
)
def test_get_intent_rejects_authority_and_query_variants(url: str) -> None:
    with pytest.raises(audit.TransportError):
        audit.make_get_intent(url)


def test_report_is_deterministic_aggregate_only_and_canary_free() -> None:
    national = _production_calendar()
    tables = _production_state_summaries(national)
    support = audit.evaluate_support(
        national,
        tables,
        inventory_counts=audit.EXPECTED_STATE_TABLE_COUNTS,
        malformed_link_counts={
            year: len(audit.FROZEN_MALFORMED_PATHS.get(year, ()))
            for year in range(2012, 2026)
        },
    )
    first = audit.build_report(
        execution_authority="offline_fixture",
        source_audit_authoritative=False,
        verifier_commit="a" * 40,
        runner_blob="b" * 40,
        manifest_sha256="c" * 64,
        sentinel_sha256="d" * 64,
        schedule_sha256="e" * 64,
        newsroom_pages_sha256="f" * 64,
        inventories_sha256="1" * 64,
        state_tables_sha256="2" * 64,
        support=support,
    )
    second = audit.build_report(
        execution_authority="offline_fixture",
        source_audit_authoritative=False,
        verifier_commit="a" * 40,
        runner_blob="b" * 40,
        manifest_sha256="c" * 64,
        sentinel_sha256="d" * 64,
        schedule_sha256="e" * 64,
        newsroom_pages_sha256="f" * 64,
        inventories_sha256="1" * 64,
        state_tables_sha256="2" * 64,
        support=support,
    )
    first_raw = audit.canonical_json_bytes(first)
    assert first_raw == audit.canonical_json_bytes(second)
    assert FUTURE_CANARY.encode() not in first_raw
    assert b"current_initial_claims" not in first_raw
    assert b"week_ending" not in first_raw
    assert b"release_date" not in first_raw


def test_module_import_surface_is_standard_library_only() -> None:
    source = inspect.getsource(audit)
    forbidden = (
        "import pandas",
        "import numpy",
        "import torch",
        "import transformers",
        "sqlalchemy",
        "psycopg",
        "ccxt",
        "requests",
    )
    assert all(value not in source for value in forbidden)
    assert '_strict_utf8(raw, label="newsroom page")' not in source
    assert '"-I",' in source
    assert '"-S",' in source
    assert "ui/data.pdf" not in source
    assert "wkclaims/report.asp" not in source
    assert "DataDownloads.asp" not in source


def test_fixture_paths_reject_production_alias_and_existing_attempt(
    tmp_path: Path,
) -> None:
    paths = audit.AuditPaths(
        sentinel=tmp_path / "attempt.started",
        manifest=tmp_path / "manifest.ndjson",
        raw_dir=tmp_path / "raw",
        report=tmp_path / "report.json",
    )
    validated = audit.validate_fixture_paths(paths)
    assert validated == paths
    paths.sentinel.write_text("used")
    with pytest.raises(audit.ProtocolError):
        audit.reserve_attempt(
            paths=paths,
            verifier_commit="a" * 40,
            runner_blob="b" * 40,
        )

    with pytest.raises(audit.ProtocolError):
        audit.validate_fixture_paths(audit.PRODUCTION_PATHS)


def test_report_publication_is_atomic_no_clobber(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    audit._atomic_publish(report, b'{"ok":true}\n')
    assert report.read_bytes() == b'{"ok":true}\n'
    with pytest.raises(audit.PublicationError):
        audit._atomic_publish(report, b'{"ok":false}\n')
    assert report.read_bytes() == b'{"ok":true}\n'


def test_disk_guard_uses_filesystem_reported_used_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    used = audit.DISK_USED_LIMIT - 1
    free = audit.DISK_FREE_FLOOR + 1
    monkeypatch.setattr(
        audit.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(
            total=used + free + 64 * 1024**3,
            used=used,
            free=free,
        ),
    )

    assert audit._disk_guard() == (used, free)


def test_manifest_chain_replay_binds_sequence_hashes_and_raw_bytes(
    tmp_path: Path,
) -> None:
    paths = audit.AuditPaths(
        sentinel=tmp_path / "attempt.started",
        manifest=tmp_path / "manifest.ndjson",
        raw_dir=tmp_path / "raw",
        report=tmp_path / "report.json",
    )
    guard = audit.reserve_attempt(
        paths=paths,
        verifier_commit="a" * 40,
        runner_blob="b" * 40,
    )
    guard.append("request_intent", {"key": "fixture"})
    raw = b"<html>fixture</html>"
    artifact = paths.raw_dir / "00001-fixture.html"
    artifact.write_bytes(raw)
    guard.append(
        "response_committed",
        {
            "body_bytes": len(raw),
            "body_sha256": audit.sha256_bytes(raw),
            "raw_artifact": artifact.name,
        },
    )
    guard.append("source_support_complete", {"decision": "SOURCE_SUPPORT_PASS"})
    result = audit.validate_manifest_chain(
        paths.manifest,
        raw_dir=paths.raw_dir,
        require_complete=True,
    )
    assert result["final_record_hash"] == guard.previous_hash
    assert result["record_count"] == guard.sequence
    assert result["raw_artifact_count"] == 1

    artifact.write_bytes(b"tampered")
    with pytest.raises(audit.ProtocolError):
        audit.validate_manifest_chain(
            paths.manifest,
            raw_dir=paths.raw_dir,
            require_complete=True,
        )


def test_canonical_report_is_valid_json_without_nonfinite_values() -> None:
    payload = audit.canonical_json_bytes({"b": 2, "a": 1})
    assert payload == b'{"a":1,"b":2}\n'
    assert json.loads(payload) == {"a": 1, "b": 2}
