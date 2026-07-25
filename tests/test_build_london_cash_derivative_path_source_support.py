from __future__ import annotations

import csv
import gzip
import io
import json
import math
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from training import build_london_cash_derivative_path_source_support as s


def metric(
    total_return: float,
    *,
    first_return: float | None = None,
    second_return: float | None = None,
    efficiency: float = 0.5,
    log_range: float = 0.1,
    quote_notional: float = 100.0,
    start_price: float = 100.0,
    end_price: float | None = None,
) -> s.VenueMetrics:
    first = total_return / 2 if first_return is None else first_return
    second = total_return - first if second_return is None else second_return
    end = (
        start_price * math.exp(total_return)
        if end_price is None
        else end_price
    )
    return s.VenueMetrics(
        start_price=start_price,
        end_price=end,
        total_return=total_return,
        first_return=first,
        second_return=second,
        efficiency=efficiency,
        log_range=log_range,
        quote_notional=quote_notional,
    )


def pair(
    day: date,
    cash_return: float = 0.02,
    perp_return: float = 0.01,
    *,
    cash_quote: float = 100.0,
    perp_quote: float = 100.0,
    cash_kwargs: dict | None = None,
    perp_kwargs: dict | None = None,
) -> s.DayPair:
    cash_options = dict(cash_kwargs or {})
    perp_options = dict(perp_kwargs or {})
    cash_options.setdefault("quote_notional", cash_quote)
    perp_options.setdefault("quote_notional", perp_quote)
    return s.DayPair(
        london_date=day,
        cash=metric(cash_return, **cash_options),
        perp=metric(perp_return, **perp_options),
        cash_reason="valid",
        perp_reason="valid",
    )


def invalid_pair(day: date, reason: str = "missing") -> s.DayPair:
    return s.DayPair(day, None, None, reason, reason)


def empty_inputs() -> s.SourceInputs:
    audit = s.ParseAudit(
        path="synthetic",
        physical_date_rows=0,
        selected_non_date_rows=0,
        post_cutoff_date_only_rows=0,
        at_or_after_2023_date_only_rows=0,
        post_cutoff_non_date_rows=0,
        at_or_after_2023_non_date_rows=0,
        first_timestamp="synthetic",
        last_timestamp="synthetic",
    )
    return s.SourceInputs(
        end_exclusive="2023-01-01",
        dates=[],
        pairs=[],
        coinbase_audit=audit,
        binance_audit=audit,
    )


def ready_line(
    day: date,
    *,
    alignment: str = "BOTH_RISE",
    leader: str = "CASH_LEADS_RISE",
    participation: str = "CASH_PARTICIPATION_MID",
) -> s.StreamLine:
    tokens = {
        field: values[0] for field, values in s.prereg.TOKEN_SCHEMA
    }
    tokens["calendar_context"] = s._calendar_context(day)
    tokens["daily_alignment"] = alignment
    tokens["daily_leader"] = leader
    tokens["participation_state"] = participation
    return s.StreamLine(
        london_date=day,
        state=s.READY,
        tokens=tokens,
        serialized=s.prereg.serialize_line(tokens),
        cash_sign=1,
        perp_sign=1,
    )


def bars_for_day(
    day: date,
    *,
    price: float = 100.0,
    source_complete: bool = True,
) -> tuple[list[s.Bar], list[datetime]]:
    expected = s.expected_timestamps(day)
    bars = [
        s.Bar(
            timestamp=timestamp,
            open=price,
            high=price * 1.01,
            low=price * 0.99,
            close=price,
            quote_notional=10.0,
            source_complete=source_complete,
        )
        for timestamp in expected
    ]
    return bars, expected


def write_source(
    path: Path,
    *,
    header: tuple[str, ...],
    rows: list[list[str]],
) -> None:
    handle = (
        gzip.open(path, "wt", encoding="utf-8", newline="")
        if path.suffix == ".gz"
        else path.open("wt", encoding="utf-8", newline="")
    )
    with handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def test_frozen_policy_rejects_mutation() -> None:
    with pytest.raises(ValueError, match="policy is frozen"):
        s._validate_policy(replace(s.FrozenPolicy(), sequence_lines=20))


@pytest.mark.parametrize(
    ("day", "slots"),
    [
        (date(2020, 2, 1), 288),
        (date(2020, 3, 29), 276),
        (date(2020, 10, 25), 300),
    ],
)
def test_expected_timestamps_respect_london_dst(day: date, slots: int) -> None:
    timestamps = s.expected_timestamps(day)
    assert len(timestamps) == slots
    assert timestamps[-1] + timedelta(minutes=5) == s._boundary(
        day
    ).astimezone(s.prereg.UTC)


def test_london_assignment_uses_next_1600_boundary() -> None:
    before = datetime(2020, 1, 2, 15, 55, tzinfo=s.prereg.LONDON)
    at = datetime(2020, 1, 2, 16, 0, tzinfo=s.prereg.LONDON)
    assert s._assigned_london_day(before) == date(2020, 1, 2)
    assert s._assigned_london_day(at) == date(2020, 1, 3)


@pytest.mark.parametrize(
    "token",
    [
        "2020-01-01",
        "2020-01-01T00:00:00Z",
        "2020-01-01T00:00:00+01:00",
        "2020-01-01 00:00:00+00:00",
        "2020-1-01 00:00:00",
        "2020-01-1 00:00:00",
        "2020-01-01 0:00:00",
        "2020-01-01 00:0:00",
        "2020-01-01 00:00:0",
        "2020-01-01 00:00:00 ",
        " 2020-01-01 00:00:00",
    ],
)
def test_timestamp_parser_rejects_nonphysical_utc_grammar(token: str) -> None:
    with pytest.raises(ValueError, match="exact naive-UTC"):
        s._parse_timestamp(token)
    assert s._parse_timestamp("2020-01-01 00:00:00") == datetime(
        2020,
        1,
        1,
        tzinfo=s.prereg.UTC,
    )


def test_venue_metrics_accept_complete_ordinary_and_dst_days() -> None:
    for day in (date(2020, 2, 1), date(2020, 3, 29), date(2020, 10, 25)):
        bars, expected = bars_for_day(day)
        result, reason = s.build_venue_metrics(bars, expected)
        assert reason == "valid"
        assert result is not None
        assert result.total_return == pytest.approx(0.0)
        assert result.quote_notional == pytest.approx(10.0 * len(expected))


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("missing", "timestamp_count"),
        ("duplicate", "timestamp_grid"),
        ("incomplete", "source_incomplete"),
        ("nonfinite", "nonfinite"),
        ("nonpositive", "nonpositive_price"),
        ("negative_notional", "negative_notional"),
        ("bad_ohlc", "ohlc_order"),
    ],
)
def test_venue_metrics_fail_closed(mutation: str, reason: str) -> None:
    bars, expected = bars_for_day(date(2020, 2, 1))
    if mutation == "missing":
        bars = bars[:-1]
    elif mutation == "duplicate":
        bars[-1] = replace(bars[-1], timestamp=bars[-2].timestamp)
    elif mutation == "incomplete":
        bars[0] = replace(bars[0], source_complete=False)
    elif mutation == "nonfinite":
        bars[0] = replace(bars[0], close=float("nan"))
    elif mutation == "nonpositive":
        bars[0] = replace(bars[0], open=0.0)
    elif mutation == "negative_notional":
        bars[0] = replace(bars[0], quote_notional=-1.0)
    elif mutation == "bad_ohlc":
        bars[0] = replace(bars[0], low=101.0, high=99.0)
    result, actual_reason = s.build_venue_metrics(bars, expected)
    assert result is None
    assert actual_reason == reason


def test_first_day_is_always_invalid_start() -> None:
    days = [date(2020, 1, 1), date(2020, 1, 2)]
    groups: dict[date, list[s.Bar]] = {}
    for day in days:
        groups[day] = bars_for_day(day)[0]
    pairs = s.build_day_pairs(days, groups, groups)
    assert not pairs[0].source_valid
    assert pairs[0].cash_reason == s.SOURCE_INVALID_START
    assert pairs[1].source_valid


def test_parser_reads_date_only_after_prefix_cutoff(tmp_path: Path) -> None:
    path = tmp_path / "cash.csv.gz"
    before = "2020-12-31 15:55:00"
    after = "2020-12-31 16:00:00"
    after_2023 = "2023-01-01 00:00:00"
    write_source(
        path,
        header=s.prereg.COINBASE_HEADER,
        rows=[
            [before, "100", "101", "99", "100", "1", "1"],
            [after, "NOT_PARSED", "x", "x", "x", "x", "x"],
            [after_2023, "NOT_PARSED", "x", "x", "x", "x", "x"],
        ],
    )
    groups, audit = s.read_venue_source(
        path,
        expected_header=s.prereg.COINBASE_HEADER,
        end_exclusive="2021-01-01",
        venue="cash",
    )
    assert sum(map(len, groups.values())) == 1
    assert audit.selected_non_date_rows == 1
    assert audit.post_cutoff_date_only_rows == 2
    assert audit.at_or_after_2023_date_only_rows == 1
    assert audit.post_cutoff_non_date_rows == 0
    assert audit.at_or_after_2023_non_date_rows == 0


def test_parser_rejects_duplicate_physical_timestamp(tmp_path: Path) -> None:
    path = tmp_path / "perp.csv"
    row = ["2020-01-01 00:00:00", "100", "101", "99", "100", "10"]
    write_source(path, header=s.prereg.BINANCE_HEADER, rows=[row, row])
    with pytest.raises(ValueError, match="duplicate or nonchronological"):
        s.read_venue_source(
            path,
            expected_header=s.prereg.BINANCE_HEADER,
            end_exclusive="2021-01-01",
            venue="perp",
        )


def test_parser_rejects_malformed_selected_numeric(tmp_path: Path) -> None:
    path = tmp_path / "perp.csv"
    write_source(
        path,
        header=s.prereg.BINANCE_HEADER,
        rows=[
            [
                "2020-01-01 00:00:00",
                "NOT_A_NUMBER",
                "101",
                "99",
                "100",
                "10",
            ]
        ],
    )
    with pytest.raises(ValueError):
        s.read_venue_source(
            path,
            expected_header=s.prereg.BINANCE_HEADER,
            end_exclusive="2021-01-01",
            venue="perp",
        )


def test_exact_quantile_uses_linear_interpolation() -> None:
    assert s.exact_quantile([0.0, 10.0], 1 / 3) == pytest.approx(10 / 3)
    assert s.exact_quantile([0.0, 3.0, 6.0, 9.0], 2 / 3) == pytest.approx(6.0)


@pytest.mark.parametrize(
    ("cash_sign", "perp_sign", "expected"),
    [
        (1, 1, "BOTH_RISE"),
        (-1, -1, "BOTH_FALL"),
        (1, -1, "CASH_RISE_PERP_FALL"),
        (-1, 1, "CASH_FALL_PERP_RISE"),
        (0, 1, "RETURN_MIXED_OR_FLAT"),
    ],
)
def test_daily_alignment_branches(
    cash_sign: int,
    perp_sign: int,
    expected: str,
) -> None:
    assert s._daily_alignment(cash_sign, perp_sign) == expected


@pytest.mark.parametrize(
    ("cash_return", "perp_return", "expected"),
    [
        (0.02, 0.01, "CASH_LEADS_RISE"),
        (-0.02, -0.01, "CASH_LEADS_FALL"),
        (0.01, 0.02, "PERP_LEADS_RISE"),
        (-0.01, -0.02, "PERP_LEADS_FALL"),
        (0.01, -0.02, "NO_CLEAR_LEADER"),
        (0.01, 0.01, "NO_CLEAR_LEADER"),
    ],
)
def test_daily_leader_branches(
    cash_return: float,
    perp_return: float,
    expected: str,
) -> None:
    assert s._daily_leader(
        metric(cash_return),
        metric(perp_return),
    ) == expected


@pytest.mark.parametrize(
    ("cash_start", "cash_end", "expected"),
    [
        (101.0, 102.0, "CASH_RICHENS"),
        (102.0, 101.0, "CASH_CHEAPENS"),
        (99.0, 101.0, "BASIS_ROTATES"),
        (100.0, 100.0, "BASIS_FLAT"),
    ],
)
def test_basis_path_branches(
    cash_start: float,
    cash_end: float,
    expected: str,
) -> None:
    cash = metric(
        0.0,
        start_price=cash_start,
        end_price=cash_end,
    )
    perp = metric(0.0, start_price=100.0, end_price=100.0)
    assert s._basis_path(cash, perp) == expected


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (0.01, 0.01, "CASH_LEAD_EXTENDS"),
        (0.01, -0.01, "CASH_LEAD_REVERSES"),
        (-0.01, -0.01, "PERP_LEAD_EXTENDS"),
        (-0.01, 0.01, "PERP_LEAD_REVERSES"),
        (0.0, 0.01, "ARC_MIXED"),
    ],
)
def test_arc_transfer_branches(
    first: float,
    second: float,
    expected: str,
) -> None:
    cash = metric(
        first + second,
        first_return=first,
        second_return=second,
    )
    perp = metric(0.0, first_return=0.0, second_return=0.0)
    assert s._arc_transfer(cash, perp) == expected


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (2.0, 1.0, "LEFT"),
        (1.0, 2.0, "RIGHT"),
        (1.0, 1.0, "TIE"),
    ],
)
def test_exact_relation_compare_branches(
    left: float,
    right: float,
    expected: str,
) -> None:
    assert s._compare(left, right, "LEFT", "RIGHT", "TIE") == expected


@pytest.mark.parametrize(
    ("share", "expected"),
    [
        (0.2, "CASH_PARTICIPATION_LOW"),
        (0.5, "CASH_PARTICIPATION_MID"),
        (0.8, "CASH_PARTICIPATION_HIGH"),
        (0.3, "CASH_PARTICIPATION_MID"),
        (0.7, "CASH_PARTICIPATION_MID"),
    ],
)
def test_participation_state_has_exact_tie_rules(
    share: float,
    expected: str,
) -> None:
    assert s._participation_state(share, 0.3, 0.7) == expected


def test_ready_tokens_use_immediately_previous_calendar_line() -> None:
    current = pair(date(2020, 4, 2))
    previous = ready_line(
        date(2020, 4, 1),
        alignment="BOTH_FALL",
        leader="PERP_LEADS_FALL",
        participation="CASH_PARTICIPATION_LOW",
    )
    tokens, _, _ = s.ready_tokens(
        current,
        low_quantile=0.4,
        high_quantile=0.6,
        previous=previous,
    )
    assert tokens["participation_transition"] == "CASH_SHARE_RISING"
    assert tokens["alignment_transition"] == "ALIGNMENT_FLIPS"
    assert tokens["leader_transition"] == "LEAD_ROTATES_TO_CASH"
    safety_tokens = s._uniform_tokens(date(2020, 4, 1), s.SOURCE_INVALID)
    safety = s.StreamLine(
        date(2020, 4, 1),
        s.SOURCE_INVALID,
        safety_tokens,
        s.prereg.serialize_line(safety_tokens),
        0,
        0,
    )
    tokens, _, _ = s.ready_tokens(
        current,
        low_quantile=0.4,
        high_quantile=0.6,
        previous=safety,
    )
    assert tokens["participation_transition"] == "PARTICIPATION_UNKNOWN"
    assert tokens["alignment_transition"] == "ALIGNMENT_MIXED"
    assert tokens["leader_transition"] == "LEAD_MIXED"


def test_rank_is_strictly_prior_and_invalid_days_are_not_compressed() -> None:
    start = date(2020, 1, 1)
    pairs = [invalid_pair(start, s.SOURCE_INVALID_START)]
    for offset in range(1, 64):
        share = offset / 100
        pairs.append(
            pair(
                start + timedelta(days=offset),
                cash_quote=share,
                perp_quote=1 - share,
            )
        )
    current = pair(
        start + timedelta(days=64),
        cash_quote=0.99,
        perp_quote=0.01,
    )
    pairs.append(current)
    stream = s.build_relational_stream(
        pairs,
        control=False,
        invalid_token=s.SOURCE_INVALID,
        first_token=s.SOURCE_INVALID_START,
    )
    assert stream[63].state == s.RANK_UNREADY
    assert stream[64].state == s.READY
    assert (
        stream[64].tokens["participation_state"]
        == "CASH_PARTICIPATION_HIGH"
    )
    pairs.insert(40, invalid_pair(start + timedelta(days=40)))
    for index, item in enumerate(pairs):
        pairs[index] = replace(item, london_date=start + timedelta(days=index))
    stream = s.build_relational_stream(
        pairs,
        control=False,
        invalid_token=s.SOURCE_INVALID,
        first_token=s.SOURCE_INVALID_START,
    )
    assert stream[64].state == s.RANK_UNREADY
    assert stream[65].state == s.READY


def test_rank_quantile_excludes_current_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = date(2020, 1, 1)
    pairs = [invalid_pair(start, s.SOURCE_INVALID_START)]
    pairs.extend(
        pair(
            start + timedelta(days=index),
            cash_quote=float(index),
            perp_quote=100.0 - index,
        )
        for index in range(1, 64)
    )
    pairs.append(
        pair(
            start + timedelta(days=64),
            cash_quote=99.0,
            perp_quote=1.0,
        )
    )
    real_quantile = s.exact_quantile
    seen: list[list[float]] = []

    def capture(values: list[float], quantile: float) -> float:
        seen.append(list(values))
        return real_quantile(values, quantile)

    monkeypatch.setattr(s, "exact_quantile", capture)
    stream = s.build_relational_stream(
        pairs,
        control=False,
        invalid_token=s.SOURCE_INVALID,
        first_token=s.SOURCE_INVALID_START,
    )
    assert stream[-1].state == s.READY
    assert seen
    assert all(len(values) == 63 for values in seen[-2:])
    assert all(0.99 not in values for values in seen[-2:])


def test_transformed_control_prefixes_are_unready() -> None:
    start = date(2020, 1, 1)
    pairs = [pair(start + timedelta(days=index)) for index in range(10)]
    transformed = s.transformed_pairs(pairs)
    assert not transformed["cash_stale_one_day"][0].source_valid
    assert not transformed["perp_stale_one_day"][0].source_valid
    assert all(
        not transformed["lag_7_calendar_days"][index].source_valid
        for index in range(7)
    )
    assert transformed["lag_7_calendar_days"][7].source_valid


def test_derived_controls_preserve_safety_and_mask_ready() -> None:
    day0 = date(2020, 1, 1)
    safety_tokens = s._uniform_tokens(day0, s.SOURCE_INVALID_START)
    safety = s.StreamLine(
        day0,
        s.SOURCE_INVALID_START,
        safety_tokens,
        s.prereg.serialize_line(
            safety_tokens,
            allow_source_invalid_start=True,
        ),
        0,
        0,
    )
    day1 = date(2020, 4, 1)
    ready = ready_line(day1)
    pairs = [invalid_pair(day0), pair(day1)]
    controls = s.build_derived_controls([safety, ready], pairs)
    assert controls["calendar_context_mask"][0] == safety.serialized
    assert "calendar_context=CALENDAR_MASKED" in (
        controls["calendar_context_mask"][1]
    )
    assert "daily_alignment=CASH_ONLY_RISE" in (
        controls["cash_only_language"][1]
    )
    assert "daily_alignment=PERP_ONLY_RISE" in (
        controls["perp_only_language"][1]
    )
    assert "ABLATION_MASKED" in controls["cash_only_language"][1]


def test_sequence_hash_and_december_terminal_exclusion() -> None:
    start = date(2022, 12, 10)
    pairs = [pair(start + timedelta(days=index)) for index in range(22)]
    primary = [ready_line(item.london_date) for item in pairs]
    controls = {
        key: primary for key in (
            "cash_perp_role_swap",
            "cash_stale_one_day",
            "perp_stale_one_day",
            "lag_7_calendar_days",
        )
    }
    derived = {
        key: [line.serialized for line in primary]
        for key in (
            "calendar_context_mask",
            "cash_only_language",
            "perp_only_language",
        )
    }
    rows = s.assemble_rows(
        [item.london_date for item in pairs],
        pairs,
        primary,
        controls,
        derived,
    )
    assert rows[19]["primary_sequence_hash"] == ""
    assert len(rows[20]["primary_sequence_hash"]) == 64
    dec31 = next(row for row in rows if row["london_date"] == "2022-12-31")
    assert dec31["model_eligible"] is False


def test_compare_prefix_detects_deliberate_corruption() -> None:
    full = [
        {"london_date": "2020-01-01", "value": "a"},
        {"london_date": "2020-01-02", "value": "b"},
    ]
    assert s.compare_prefix_records(full, full)["passed"] is True
    corrupt = [dict(full[0]), dict(full[1])]
    corrupt[1]["value"] = "c"
    result = s.compare_prefix_records(full, corrupt)
    assert result["passed"] is False
    assert result["first_mismatches"] == ["2020-01-02"]


def test_deterministic_gzip_bytes_and_exact_header() -> None:
    row = {column: "" for column in s.OUTPUT_COLUMNS}
    row.update(
        {
            "london_date": "2020-01-01",
            "boundary_utc": "2020-01-01T16:00:00Z",
            "expected_slots": 288,
            "source_state": s.SOURCE_INVALID_START,
            "rank_ready": False,
            "model_eligible": False,
        }
    )
    first = s.deterministic_token_gzip([row])
    second = s.deterministic_token_gzip([row])
    assert first == second
    with gzip.GzipFile(fileobj=io.BytesIO(first), mode="rb") as handle:
        text = handle.read().decode()
    assert text.splitlines()[0] == ",".join(s.OUTPUT_COLUMNS)
    assert ",false,false," in text.splitlines()[1]


def test_output_schema_contains_no_forbidden_numeric_primitive() -> None:
    forbidden = {
        "price",
        "return",
        "volume",
        "notional",
        "funding",
        "reward",
        "action",
        "trade",
        "pnl",
        "cagr",
        "mdd",
    }
    assert not forbidden.intersection(name.lower() for name in s.OUTPUT_COLUMNS)


def test_gate_record_is_conjunctive() -> None:
    assert s._gate_record(1, "x", {"a": True, "b": True})["passed"] is True
    assert s._gate_record(1, "x", {"a": True, "b": False})["passed"] is False


def test_mask_controls_require_every_ready_line_and_exact_fields() -> None:
    policy = s.FrozenPolicy()
    controls = {
        "cash_perp_role_swap": {
            "jointly_ready": 100,
            "different": 5,
            "difference_share": 0.05,
        },
        "calendar_context_mask": {
            "jointly_ready": 100,
            "different": 100,
            "required_field_correct": 100,
            "difference_share": 1.0,
        },
    }
    checks = s.control_gate_checks(controls, policy)
    assert checks == {
        "cash_perp_role_swap": True,
        "calendar_context_mask": True,
    }
    controls["calendar_context_mask"]["different"] = 99
    assert s.control_gate_checks(controls, policy)[
        "calendar_context_mask"
    ] is False
    controls["calendar_context_mask"]["different"] = 100
    controls["calendar_context_mask"]["required_field_correct"] = 99
    assert s.control_gate_checks(controls, policy)[
        "calendar_context_mask"
    ] is False


def test_run_official_reports_authority_failure_as_gate_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, dict] = {}
    monkeypatch.setattr(
        s,
        "validate_execution_seal",
        lambda: (_ for _ in ()).throw(RuntimeError("synthetic seal failure")),
    )
    monkeypatch.setattr(
        s,
        "validate_frozen_authority",
        lambda: (_ for _ in ()).throw(
            AssertionError("must not reach authority")
        ),
    )
    monkeypatch.setattr(
        s,
        "write_once_json",
        lambda path, payload: captured.setdefault(str(path), dict(payload))
        and "hash",
    )
    report = s.run_official()
    assert report["decision"] == "fail"
    assert len(report["gates"]) == 1
    assert report["gates"][0]["name"] == "protocol_source_integrity"
    assert report["error"]["type"] == "RuntimeError"
    assert s.REJECTION_REPORT in captured
    assert s.PASS_REPORT not in captured


def test_run_official_stops_at_first_source_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, dict] = {}
    monkeypatch.setattr(s, "validate_execution_seal", lambda: {
        "manifest_hash": "seal",
        "runner": {},
        "tests": {},
        "contract": {},
        "boundary": {},
        "preregistration": {},
    })
    monkeypatch.setattr(s, "validate_frozen_authority", lambda: {
        "anchors": {},
        "source_manifest_hash": "source",
    })
    monkeypatch.setattr(s, "_authority_report", lambda *_: {})
    monkeypatch.setattr(
        s,
        "load_source_inputs",
        lambda **_: (_ for _ in ()).throw(ValueError("synthetic source fail")),
    )
    monkeypatch.setattr(
        s,
        "write_once_json",
        lambda path, payload: captured.setdefault(str(path), dict(payload))
        and "hash",
    )
    report = s.run_official()
    assert report["decision"] == "fail"
    assert report["failure_action"] == (
        "retire_lcdp_d1_unchanged_before_outcomes"
    )
    assert len(report["gates"]) == 1
    assert report["gates"][0]["name"] == "protocol_source_integrity"
    assert s.REJECTION_REPORT in captured
    assert s.PASS_REPORT not in captured


def test_run_official_stops_before_later_metrics_on_readiness_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, dict] = {}
    dummy_inputs = empty_inputs()
    dummy_bundle = type("Bundle", (), {"rows": [{"london_date": "x"}]})()
    monkeypatch.setattr(s, "validate_execution_seal", lambda: {})
    monkeypatch.setattr(s, "validate_frozen_authority", lambda: {})
    monkeypatch.setattr(s, "_authority_report", lambda *_: {})
    monkeypatch.setattr(s, "load_source_inputs", lambda **_: dummy_inputs)
    monkeypatch.setattr(
        s,
        "parser_integrity_checks",
        lambda _: {"parser": True},
    )
    monkeypatch.setattr(
        s,
        "calendar_integrity_checks",
        lambda _: {"calendar": True},
    )
    monkeypatch.setattr(s, "source_validity_metrics", lambda _: {"x": 1})
    monkeypatch.setattr(
        s,
        "validity_gate_checks",
        lambda *_: {"validity": True},
    )
    monkeypatch.setattr(s, "finish_bundle", lambda *_args, **_kwargs: dummy_bundle)
    monkeypatch.setattr(s, "readiness_metrics", lambda _: {"x": 1})
    monkeypatch.setattr(
        s,
        "readiness_gate_checks",
        lambda *_: {"readiness": False},
    )
    monkeypatch.setattr(
        s,
        "diversity_metrics",
        lambda _: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    monkeypatch.setattr(s, "forbidden_counters", lambda _: {"forbidden": 0})
    monkeypatch.setattr(
        s,
        "write_once_json",
        lambda path, payload: captured.setdefault(str(path), dict(payload))
        and "hash",
    )
    report = s.run_official()
    assert report["decision"] == "fail"
    assert [gate["name"] for gate in report["gates"]] == [
        "protocol_source_integrity",
        "calendar_dst_integrity",
        "source_validity",
        "readiness",
    ]
    assert "diversity" not in report["details"]


def test_run_official_pass_writes_only_token_and_pass_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    json_outputs: dict[str, dict] = {}
    byte_outputs: dict[str, bytes] = {}
    dummy_inputs = empty_inputs()
    dummy_bundle = type(
        "Bundle",
        (),
        {"rows": [{"london_date": "2020-01-01", "x": 1}]},
    )()
    monkeypatch.setattr(s, "validate_execution_seal", lambda: {})
    monkeypatch.setattr(s, "validate_frozen_authority", lambda: {})
    monkeypatch.setattr(s, "_authority_report", lambda *_: {})
    monkeypatch.setattr(s, "load_source_inputs", lambda **_: dummy_inputs)
    monkeypatch.setattr(
        s,
        "parser_integrity_checks",
        lambda _: {"parser": True},
    )
    monkeypatch.setattr(
        s,
        "calendar_integrity_checks",
        lambda _: {"calendar": True},
    )
    monkeypatch.setattr(s, "source_validity_metrics", lambda _: {"x": 1})
    monkeypatch.setattr(
        s,
        "validity_gate_checks",
        lambda *_: {"validity": True},
    )
    monkeypatch.setattr(s, "finish_bundle", lambda *_args, **_kwargs: dummy_bundle)
    monkeypatch.setattr(s, "readiness_metrics", lambda _: {"x": 1})
    monkeypatch.setattr(
        s,
        "readiness_gate_checks",
        lambda *_: {"readiness": True},
    )
    monkeypatch.setattr(s, "diversity_metrics", lambda _: {"x": 1})
    monkeypatch.setattr(
        s,
        "diversity_gate_checks",
        lambda *_: {"diversity": True},
    )
    monkeypatch.setattr(s, "control_metrics", lambda _: {"x": 1})
    monkeypatch.setattr(
        s,
        "control_gate_checks",
        lambda *_: {"controls": True},
    )
    monkeypatch.setattr(
        s,
        "run_append_replay",
        lambda *_args, **_kwargs: {"passed": True},
    )
    monkeypatch.setattr(s, "forbidden_counters", lambda _: {"forbidden": 0})
    monkeypatch.setattr(s, "deterministic_token_gzip", lambda _: b"tokens")
    monkeypatch.setattr(
        s,
        "write_once_bytes",
        lambda path, content: byte_outputs.setdefault(str(path), content)
        and "token-sha",
    )
    monkeypatch.setattr(
        s,
        "write_once_json",
        lambda path, payload: json_outputs.setdefault(str(path), dict(payload))
        and "report-sha",
    )
    report = s.run_official()
    assert report["decision"] == "pass"
    assert report["pass_action"] == (
        "authorize_economic_rllm_evaluator_freeze_only"
    )
    assert s.TOKEN_OUTPUT in byte_outputs
    assert s.PASS_REPORT in json_outputs
    assert s.REJECTION_REPORT not in json_outputs
    assert len(report["gates"]) == 8


def test_manifest_and_report_hashes_are_self_consistent() -> None:
    core = {"a": 1, "b": [2, 3]}
    assert s.canonical_hash(core) == hashlib_sha256(s.canonical_bytes(core))


def hashlib_sha256(content: bytes) -> str:
    import hashlib

    return hashlib.sha256(content).hexdigest()
