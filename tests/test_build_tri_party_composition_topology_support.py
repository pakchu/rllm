from __future__ import annotations

import csv
from collections import OrderedDict
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction
import gzip
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import pytest

from training import build_tri_party_composition_topology_support as support
from training import preregister_tri_party_composition_topology as prereg


UTC = timezone.utc


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


SEMANTICS = {
    mnemonic: (f"segment-{index}", f"measure-{index}", f"subset-{index}", f"series-{index}")
    for index, mnemonic in enumerate(prereg.SOURCE_ALLOWLIST)
}


def _source_row(
    mnemonic: str,
    day: date,
    *,
    value: str = "1",
    disclosure_edit: str = "0",
    available_at: datetime | None = None,
    semantics: Mapping[str, tuple[str, str, str, str]] = SEMANTICS,
) -> list[str]:
    segment, measure, subset, series = semantics.get(
        mnemonic,
        ("unselected-segment", "unselected-measure", "unselected-subset", "unselected-series"),
    )
    return [
        mnemonic,
        day.isoformat(),
        support.canonical_utc(available_at or support.expected_availability(day)),
        value,
        disclosure_edit,
        segment,
        measure,
        subset,
        series,
    ]


def _csv_text(rows: Sequence[Sequence[str]]) -> str:
    raw = io.StringIO(newline="")
    writer = csv.writer(raw, lineterminator="\n")
    writer.writerow(prereg.SOURCE_COLUMNS)
    writer.writerows(rows)
    return raw.getvalue()


def _parse_rows(rows: Sequence[Sequence[str]], *, expected_rows: int | None = None):
    return support.parse_source_stream(
        io.StringIO(_csv_text(rows)),
        expected_rows=expected_rows,
        semantics=SEMANTICS,
    )


def _complete_source_rows(day: date, *, value: str = "1") -> list[list[str]]:
    rows: list[list[str]] = []
    for index, mnemonic in enumerate(prereg.SOURCE_ALLOWLIST):
        # Keep all synthetic transaction volumes positive and rates exact.
        rows.append(_source_row(mnemonic, day, value=str(index + 1) if value == "vary" else value))
    return rows


def _primitive_values(seed: int) -> OrderedDict[str, Fraction]:
    return OrderedDict(
        (primitive, Fraction(seed * 101 + index, 10_000))
        for index, primitive in enumerate(prereg.PRIMITIVES)
    )


def _vector(
    seed: int,
    available_at: datetime,
    *,
    observation_date: date | None = None,
    valid: bool = True,
    reasons: tuple[str, ...] = (),
) -> support.SourceVector:
    return support.SourceVector(
        observation_date=observation_date or available_at.date(),
        available_at=available_at,
        valid=valid,
        invalid_reasons=reasons,
        primitives=_primitive_values(seed) if valid else None,
    )


def _tokens(**overrides: str) -> OrderedDict[str, str]:
    values = OrderedDict(prereg.SERIALIZATION_SPECIMEN)
    values.update(overrides)
    return OrderedDict((name, values[name]) for name in prereg.TOKEN_COLUMNS)


def _opportunity(
    entry: datetime,
    *,
    split: str | None = "train",
    reserved: bool = True,
    tokens: OrderedDict[str, str] | None = None,
) -> support.Opportunity:
    return support.Opportunity(
        observation_date=entry.date() - timedelta(days=8),
        available_at=entry - timedelta(minutes=5),
        signal_available=entry - timedelta(minutes=5),
        entry=entry,
        exit=entry + timedelta(hours=120),
        tokens=tokens or _tokens(),
        reserved=reserved,
        split=split,
    )


def _history(count: int, *, start: datetime = dt("2020-09-10T00:00:00Z")) -> list[support.SourceVector]:
    return [
        _vector(index, start + timedelta(days=index), observation_date=(start + timedelta(days=index)).date())
        for index in range(count)
    ]


def test_sealed_rows_skip_before_mnemonic_and_value_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fail_if_value_decoded(value: str) -> Fraction:
        calls.append(value)
        raise AssertionError("sealed value was decoded")

    monkeypatch.setattr(prereg, "parse_exact_decimal", fail_if_value_decoded)
    sealed_day = date(2022, 12, 24)
    vectors, audit = _parse_rows(
        [
            _source_row(
                "TPCT-SEALED-UNSELECTED",
                sealed_day,
                value="not-a-decimal-that-must-remain-sealed",
                disclosure_edit="unexpected",
            )
        ],
        expected_rows=1,
    )
    assert vectors == []
    assert calls == []
    assert audit.physical_rows_read == 1
    assert audit.eligible_selected_rows_seen == 0
    assert audit.eligible_selected_values_converted == 0
    assert audit.sealed_values_converted == 0
    assert audit.sealed_candidate_statistics == 0


def test_selected_late_2022_and_2023_rows_never_enter_tpct_parsing_or_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoded_values: list[str] = []
    selected_parser_calls: list[str] = []

    def fail_if_value_decoded(value: str) -> Fraction:
        decoded_values.append(value)
        raise AssertionError("sealed selected value was decoded")

    def fail_if_selected_row_parsed(*args: Any, **kwargs: Any) -> Any:
        selected_parser_calls.append("called")
        raise AssertionError("sealed selected row entered TPCT parsing")

    monkeypatch.setattr(prereg, "parse_exact_decimal", fail_if_value_decoded)
    monkeypatch.setattr(
        support,
        "_selected_row_from_fields",
        fail_if_selected_row_parsed,
    )
    rows: list[list[str]] = []
    for sealed_day in (date(2022, 12, 24), date(2023, 6, 1)):
        for mnemonic in prereg.SOURCE_ALLOWLIST:
            rows.append(
                _source_row(
                    mnemonic,
                    sealed_day,
                    value="SEALED_SELECTED_VALUE",
                    disclosure_edit="unexpected",
                )
            )

    vectors, audit = _parse_rows(rows, expected_rows=len(rows))

    assert vectors == []
    assert decoded_values == []
    assert selected_parser_calls == []
    assert audit.eligible_selected_rows_seen == 0
    assert audit.eligible_selected_values_converted == 0
    assert audit.eligible_source_dates == 0
    assert audit.complete_vectors == 0
    assert audit.invalid_source_dates == 0
    assert audit.sealed_values_converted == 0
    assert audit.sealed_candidate_statistics == 0


def test_parser_rejects_header_drift_before_rows_are_interpreted() -> None:
    with pytest.raises(RuntimeError, match="columns changed"):
        support.parse_source_stream(
            io.StringIO("mnemonic,observation_date\n"),
            expected_rows=None,
            semantics=SEMANTICS,
        )


def test_parser_rejects_row_width_drift() -> None:
    with pytest.raises(RuntimeError, match="row width changed"):
        _parse_rows([["too", "short"]])


def test_parser_rejects_duplicate_selected_rows() -> None:
    day = date(2020, 9, 2)
    mnemonic = prereg.SOURCE_ALLOWLIST[0]
    with pytest.raises(RuntimeError, match="duplicated"):
        _parse_rows([_source_row(mnemonic, day), _source_row(mnemonic, day)])


def test_parser_rejects_selected_semantic_drift() -> None:
    day = date(2020, 9, 2)
    row = _source_row(prereg.SOURCE_ALLOWLIST[0], day)
    row[8] = "different-series"
    with pytest.raises(RuntimeError, match="semantics changed"):
        _parse_rows([row])


def test_vector_marks_disclosure_edited_value_invalid_without_imputation() -> None:
    day = date(2020, 9, 2)
    rows = _complete_source_rows(day)
    rows[3][4] = "1"
    vectors, audit = _parse_rows(rows)
    assert len(vectors) == 1
    assert vectors[0].valid is False
    assert vectors[0].invalid_reasons == ("invalid_or_disclosure_edited_value",)
    assert vectors[0].primitives is None
    assert audit.invalid_source_dates == 1


def test_vector_marks_noncanonical_decimal_invalid() -> None:
    day = date(2020, 9, 2)
    rows = _complete_source_rows(day)
    rows[0][3] = "01.0"
    vectors, audit = _parse_rows(rows)
    assert vectors[0].valid is False
    assert vectors[0].invalid_reasons == ("invalid_or_disclosure_edited_value",)
    assert audit.eligible_selected_values_converted == len(prereg.SOURCE_ALLOWLIST) - 1


def test_vector_rejects_nonpositive_transaction_volume() -> None:
    day = date(2020, 9, 2)
    rows = _complete_source_rows(day)
    tv_index = prereg.SOURCE_ALLOWLIST.index("REPO-TRIV1_TV_OO-P")
    rows[tv_index][3] = "0"
    vectors, audit = _parse_rows(rows)
    assert vectors[0].valid is False
    assert vectors[0].invalid_reasons == ("nonpositive_transaction_volume",)
    assert audit.complete_vectors == 0


def test_equal_availability_batch_excludes_current_rows_and_uses_greatest_date_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior_lengths: list[int] = []
    token_calls: list[tuple[Mapping[str, Fraction], Mapping[str, Fraction]]] = []

    def strict_prior_midrank(current: Fraction, prior: Iterable[Fraction]) -> Fraction:
        prior_tuple = tuple(prior)
        prior_lengths.append(len(prior_tuple))
        return current

    def build_tokens(current: Mapping[str, Fraction], previous: Mapping[str, Fraction]) -> OrderedDict[str, str]:
        token_calls.append((dict(current), dict(previous)))
        return _tokens()

    monkeypatch.setattr(prereg, "strict_prior_midrank", strict_prior_midrank)
    monkeypatch.setattr(prereg, "build_tokens", build_tokens)

    same_available = dt("2021-06-01T00:00:00Z")
    late_same_batch = _vector(900, same_available, observation_date=date(2020, 12, 31))
    early_same_batch = _vector(100, same_available, observation_date=date(2020, 12, 30))
    later = _vector(901, same_available + timedelta(days=1), observation_date=date(2021, 1, 1))
    opportunities, audit = support.build_opportunities(
        [*_history(252), late_same_batch, early_same_batch, later]
    )
    assert opportunities
    assert audit.rank_complete_decisions == 2
    assert set(prior_lengths) == {252}
    assert len(prior_lengths) == 3 * len(prereg.PRIMITIVES)
    assert token_calls[0][1] == dict(_primitive_values(900))


def test_rank_decisions_require_exactly_252_prior_vectors_without_fallback() -> None:
    vectors = [*_history(251), _vector(999, dt("2021-06-01T00:00:00Z"))]
    opportunities, audit = support.build_opportunities(vectors)
    assert opportunities == []
    assert audit.rank_complete_decisions == 0
    assert audit.predecessor_only_decisions == 0
    assert audit.token_states == 0


def test_invalid_batch_breaks_predecessor_continuity_without_removing_prior_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prereg, "build_tokens", lambda current, previous: _tokens())
    vectors = [
        *_history(252),
        _vector(300, dt("2021-06-01T00:00:00Z")),
        _vector(301, dt("2021-06-02T00:00:00Z"), valid=False, reasons=("missing",)),
        _vector(302, dt("2021-06-03T00:00:00Z")),
        _vector(303, dt("2021-06-04T00:00:00Z")),
    ]
    opportunities, audit = support.build_opportunities(vectors)
    assert len(opportunities) == 1
    assert opportunities[0].observation_date == date(2021, 6, 4)
    assert audit.predecessor_only_decisions == 2
    assert audit.token_states == 1


def test_reservation_is_action_independent_rejects_overlaps_and_accepts_touching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prereg, "build_tokens", lambda current, previous: _tokens())
    first = dt("2021-06-01T00:00:00Z")
    vectors = [
        *_history(252),
        _vector(400, first),
        _vector(401, first + timedelta(days=1)),
        _vector(402, first + timedelta(days=2)),
        _vector(403, first + timedelta(days=6)),
    ]
    opportunities, audit = support.build_opportunities(vectors)
    assert [(row.reserved, row.split) for row in opportunities] == [
        (True, "train"),
        (False, None),
        (True, "train"),
    ]
    assert audit.reservation_suppressed == 1


def test_reserved_boundary_crossing_is_split_rejected_after_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prereg, "build_tokens", lambda current, previous: _tokens())
    crossing = dt("2021-12-27T00:00:00Z")
    vectors = [*_history(252, start=dt("2021-04-01T00:00:00Z")), _vector(500, crossing - timedelta(days=6)), _vector(501, crossing)]
    opportunities, audit = support.build_opportunities(vectors)
    assert len(opportunities) == 1
    assert opportunities[0].reserved is True
    assert opportunities[0].split is None
    assert audit.split_rejected_after_reservation == 1


def test_build_rejects_naive_availability_and_sealed_vectors() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        support.build_opportunities([_vector(1, datetime(2021, 1, 1))])
    with pytest.raises(RuntimeError, match="sealed vector"):
        support.build_opportunities([_vector(1, dt("2023-01-01T00:00:00Z"))])


def test_source_support_calendar_and_seal_gates_are_explicit() -> None:
    train_rows = [_opportunity(dt("2021-01-01T00:05:00Z") + timedelta(days=index), split="train") for index in range(2)]
    selection_rows = [_opportunity(dt("2022-01-01T00:05:00Z"), split="selection")]
    result = support.source_support(
        [*train_rows, *selection_rows],
        source_audit=support.SourceAudit(
            physical_rows_read=3,
            eligible_selected_rows_seen=3,
            eligible_selected_values_converted=3,
            eligible_source_dates=3,
            complete_vectors=3,
            invalid_source_dates=0,
            sealed_values_converted=1,
            sealed_candidate_statistics=0,
        ),
    )
    checks = result["checks"]
    assert result["passed"] is False
    assert checks["train_total_min_75"] is False
    assert checks["selection_2022_min_55"] is False
    assert checks["sealed_values_converted_zero"] is False
    assert checks["sealed_candidate_statistics_zero"] is True
    assert "cross_boundary_blackout_days" in result


def test_token_support_gates_reject_sparse_and_unseen_selection_vocabulary() -> None:
    train = [
        _opportunity(dt("2021-01-01T00:05:00Z") + timedelta(days=index), tokens=_tokens(maturity_wings="OVERNIGHT_LEADS"))
        for index in range(4)
    ]
    selection = [
        _opportunity(dt("2022-01-01T00:05:00Z"), split="selection", tokens=_tokens(maturity_wings="LONG_TERM_LEADS"))
    ]
    checks, summaries = support.token_support_checks(train, selection)
    assert checks["train.maturity_wings.each_count_min_3"] is False
    assert checks["train.maturity_wings.max_share_0_85"] is False
    assert checks["selection_token_values_seen_in_train"] is False
    assert summaries["train"]["distributions"]["maturity_wings"]["OVERNIGHT_LEADS"] == 4


def test_gzip_clock_is_deterministic_schema_bound_and_excludes_forbidden_columns() -> None:
    rows = [
        _opportunity(dt("2021-01-01T00:05:00Z"), split="train"),
        _opportunity(dt("2022-01-01T00:05:00Z"), split="selection"),
        _opportunity(dt("2022-02-01T00:05:00Z"), split=None),
        _opportunity(dt("2022-03-01T00:05:00Z"), split="selection", reserved=False),
    ]
    first = support.gzip_clock(rows)
    second = support.gzip_clock(rows)
    assert first == second
    assert first[:2] == b"\x1f\x8b"
    assert first[4:8] == b"\x00\x00\x00\x00"
    with gzip.GzipFile(fileobj=io.BytesIO(first), mode="rb") as zipped:
        decoded = zipped.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(decoded))
    assert tuple(reader.fieldnames or ()) == support.CLOCK_COLUMNS
    emitted = list(reader)
    assert len(emitted) == 2
    assert not support.FORBIDDEN_CLOCK_COLUMNS.intersection(reader.fieldnames or [])
    assert "outcome" in support.FORBIDDEN_CLOCK_COLUMNS
    expected_signature = support.canonical_hash({name: _tokens()[name] for name in prereg.TOKEN_COLUMNS})
    assert {row["token_signature_sha256"] for row in emitted} == {expected_signature}


def test_fraction_payload_never_projects_exact_values_through_binary_float() -> None:
    payload = support.fraction_payload(Fraction(1, 3))
    assert payload == {"numerator": 1, "denominator": 3}
    assert all(not isinstance(value, float) for value in payload.values())


def test_clock_row_rejects_unemitted_opportunities() -> None:
    with pytest.raises(ValueError, match="not emitted"):
        support._clock_row(_opportunity(dt("2022-01-01T00:05:00Z"), split=None))
    with pytest.raises(ValueError, match="not emitted"):
        support._clock_row(_opportunity(dt("2022-01-01T00:05:00Z"), reserved=False))


def test_build_report_preserves_outcome_boundary_and_decision_boundary() -> None:
    opportunities = [_opportunity(dt("2021-01-01T00:05:00Z"), split="train")]
    audit = support.SourceAudit(1, 1, 1, 1, 1, 0)
    build_audit = support.BuildAudit(1, 1, 0, 1, 0, 0)
    failing = {"passed": False, "checks": {"synthetic_gate": False}}
    report = support.build_report(
        protocol_commit="abc123",
        source_audit=audit,
        build_audit=build_audit,
        opportunities=opportunities,
        support=failing,
        clock_sha256="0" * 64,
    )
    assert report["decision"] == "retire_TPCT_120_unchanged_before_comparators_or_outcomes"
    assert all(value == 0 for key, value in report["outcome_boundary"].items() if key.endswith("rows_read") or key in {"model_labels_created", "model_training_runs", "sealed_values_converted", "network_calls"})
    assert report["clock_artifact"]["contains_raw_values"] is False
    assert report["clock_artifact"]["contains_market_funding_or_outcomes"] is False
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == support.canonical_hash(core)


def test_assert_protocol_committed_requires_tracked_clean_worktree_and_clean_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_git(*args: str) -> SimpleNamespace:
        calls.append(args)
        if args[:1] == ("ls-files",):
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if args[:2] == ("diff", "--quiet"):
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if args[:3] == ("diff", "--cached", "--quiet"):
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if args[:2] == ("status", "--porcelain"):
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if args == ("rev-parse", "HEAD"):
            return SimpleNamespace(returncode=0, stdout="abc123\n", stderr="")
        raise AssertionError(args)

    monkeypatch.setattr(support, "_git", fake_git)
    assert support.assert_protocol_committed() == "abc123"
    assert ("rev-parse", "HEAD") in calls

    def dirty_git(*args: str) -> SimpleNamespace:
        if args[:2] == ("status", "--porcelain"):
            return SimpleNamespace(returncode=0, stdout="?? tests/test_build_tri_party_composition_topology_support.py\n", stderr="")
        return fake_git(*args)

    monkeypatch.setattr(support, "_git", dirty_git)
    with pytest.raises(RuntimeError, match="worktree is not HEAD-clean"):
        support.assert_protocol_committed()


def test_load_registration_accepts_only_source_support_authorized_preregistration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "candidate": prereg.POLICY_ID,
        "manifest_hash": support.PREREGISTRATION_MANIFEST_HASH,
        "contract_hash": support.PREREGISTRATION_CONTRACT_HASH,
        "decision": {
            "source_support_authorized": True,
            "market_outcomes_authorized": False,
            "model_training_authorized": False,
            "sealed_eval_authorized": False,
        },
    }
    registration = tmp_path / "registration.json"
    registration.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(support, "repository_path", lambda path: registration)
    monkeypatch.setattr(support, "sha256_file", lambda path: support.PREREGISTRATION_SHA256)
    monkeypatch.setattr(prereg, "validate_manifest", lambda manifest, revalidate_files=False: None)
    assert support.load_registration() == payload

    payload["decision"]["market_outcomes_authorized"] = True
    registration.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="opened a later stage"):
        support.load_registration()
