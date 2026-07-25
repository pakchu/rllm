from __future__ import annotations

import csv
import gzip
import hashlib
import json
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from training import build_cboe_edge_flip_sequence_policy_support as s
from training import preregister_cboe_edge_flip_sequence_policy as p


def common_row(day: date, index: int) -> s.CommonRow:
    toggle = Decimal(index % 5)
    return s.CommonRow(
        observation_date=day,
        vix9d=Decimal("12") + toggle,
        vix=Decimal("14") + Decimal(index % 2),
        vix3m=Decimal("16") - Decimal(index % 3),
        skew=Decimal("110") + Decimal((index * 3) % 11),
        vvix=Decimal("80") + Decimal((index * 7) % 13),
        total_pcr=Decimal("0.8") + Decimal(index % 4) / Decimal("10"),
        index_pcr=Decimal("1.0") + Decimal(index % 3) / Decimal("10"),
        equity_pcr=Decimal("0.6") + Decimal((index + 1) % 4) / Decimal("10"),
        vix_pcr=Decimal("0.7") + Decimal((index + 2) % 5) / Decimal("10"),
        spx_pcr=Decimal("0.9") + Decimal((index + 3) % 4) / Decimal("10"),
        index_volume=1000 + index * 13,
        vix_volume=100 + (index % 7) * 11,
    )


def common_rows(start: date, count: int) -> list[s.CommonRow]:
    return [common_row(start + timedelta(days=index), index) for index in range(count)]


def panel_inputs(rows: list[s.CommonRow]) -> s.SourceInputs:
    term = {
        row.observation_date: s.TermRow(
            row.observation_date,
            row.vix9d,
            row.vix,
            row.vix3m,
        )
        for row in rows
    }
    tail = {
        row.observation_date: s.TailRow(
            row.observation_date,
            row.skew,
            row.vvix,
            row.vix,
        )
        for row in rows
    }
    flow = {
        row.observation_date: s.FlowRow(
            row.observation_date,
            row.total_pcr,
            row.index_pcr,
            row.equity_pcr,
            row.vix_pcr,
            row.spx_pcr,
            row.index_volume,
            row.vix_volume,
        )
        for row in rows
    }

    def panel(values: dict[date, object]) -> s.PanelResult:
        snapshots = {
            cutoff.isoformat(): {
                day: value for day, value in values.items() if day < cutoff
            }
            for cutoff in s.PREFIX_CUTOFFS
        }
        audit = s.ParseAudit(
            physical_rows=len(values),
            date_only_rows=0,
            value_rows=len(values),
            first_date=min(values).isoformat(),
            last_date=max(values).isoformat(),
            prefix_rows={key: len(value) for key, value in snapshots.items()},
        )
        return s.PanelResult(values, snapshots, audit)

    return s.SourceInputs(panel(term), panel(tail), panel(flow))


def write_gzip_csv(path: Path, header: tuple[str, ...], rows: list[list[str]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def fake_terminal_seal() -> dict[str, object]:
    commit = "a" * 40
    core = {
        "protocol_version": s.SEAL_PROTOCOL,
        "policy_id": p.POLICY_ID,
        "contract": {
            "path": s.CONTRACT_PATH,
            "commit": s.CONTRACT_COMMIT,
            "sha256": s.CONTRACT_SHA256,
        },
        "preregistration": {
            "path": s.PREREGISTRATION_PATH,
            "commit": s.PREREGISTRATION_COMMIT,
            "sha256": s.PREREGISTRATION_SHA256,
            "manifest_hash": s.PREREGISTRATION_MANIFEST_HASH,
        },
        "preregistration_producer": {
            "path": p.PRODUCER_SCRIPT,
            "commit": s.PREREGISTRATION_PRODUCER_COMMIT,
            "sha256": s.PREREGISTRATION_PRODUCER_SHA256,
        },
        "runner": {
            "path": s.RUNNER_PATH,
            "commit": commit,
            "sha256": "b" * 64,
        },
        "tests": {
            "path": s.TEST_PATH,
            "commit": commit,
            "sha256": "c" * 64,
        },
        "source_values_opened": False,
        "outcomes_opened": False,
    }
    return {**core, "manifest_hash": s.canonical_hash(core)}


def write_terminal_fixture(root: Path, decision: str) -> dict[str, object]:
    source_output = None
    control_output = None
    source_row_hash = None
    control_row_hash = None
    authority: dict[str, object] = {}
    gates = [
        s._gate_record(
            1,
            s.GATE_NAMES[0],
            (
                {
                    "authority_valid": True,
                    "worktree_clean": True,
                    **{
                        name: True
                        for name in s.FORBIDDEN_COUNTER_NAMES
                    },
                }
                if decision == "pass"
                else {"authority_valid": False}
            ),
        )
    ]
    details: dict[str, object] = {}
    if decision == "pass":
        seal = fake_terminal_seal()
        authority = s._expected_terminal_authority(seal)
        for index, name in enumerate(s.GATE_NAMES[1:], start=2):
            gates.append(s._gate_record(index, name, {"fixture": True}))
        schedules = s.build_schedules(
            common_rows(date(2020, 1, 1), 6)
        )
        controls = s.build_controls(schedules)
        schedule_detail = s.schedule_metrics(schedules)
        schedule_detail["schedule_replay"] = {}
        details = {
            "parser": {},
            "schedule": schedule_detail,
            "edge_support": s.edge_metrics(schedules),
            "diversity_stability": s.diversity_metrics(schedules),
            "controls": s.control_metrics(schedules, controls),
            "determinism_append_replay": {},
        }
        source = root / s.SOURCE_OUTPUT
        control = root / s.CONTROL_OUTPUT
        source.parent.mkdir(parents=True, exist_ok=True)
        control.parent.mkdir(parents=True, exist_ok=True)
        source_records = [s.schedule_record(row) for row in schedules]
        control_records = [s.control_record(row) for row in controls]
        source_bytes = s.deterministic_csv_gzip(
            source_records,
            s.SOURCE_OUTPUT_COLUMNS,
        )
        control_bytes = s.deterministic_csv_gzip(
            control_records,
            s.CONTROL_OUTPUT_COLUMNS,
        )
        source.write_bytes(source_bytes)
        control.write_bytes(control_bytes)
        source_row_hash = hashlib.sha256(
            s._canonical_records(source_records)
        ).hexdigest()
        control_row_hash = hashlib.sha256(
            s._canonical_records(control_records)
        ).hexdigest()
        source_output = {
            "path": s.SOURCE_OUTPUT,
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "rows": len(source_records),
        }
        control_output = {
            "path": s.CONTROL_OUTPUT,
            "sha256": hashlib.sha256(control_bytes).hexdigest(),
            "rows": len(control_records),
        }
    report = s._result_report(
        decision=decision,
        failure_action=(
            "retire_cefs_d1_unchanged_before_outcomes"
            if decision == "fail"
            else None
        ),
        pass_action=(
            "authorize_economic_rllm_evaluator_freeze_only"
            if decision == "pass"
            else None
        ),
        authority=authority,
        gates=gates,
        details=details,
        counters=s.forbidden_counters(),
        source_hash=source_row_hash,
        control_hash=control_row_hash,
        source_output=source_output,
        control_output=control_output,
        error=(
            RuntimeError("fixture authority failure")
            if decision == "fail"
            else None
        ),
    )
    path = root / (
        s.PASS_REPORT if decision == "pass" else s.REJECTION_REPORT
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(s._json_bytes(report))
    return report


def synthetic_terminal_details() -> dict[str, object]:
    inputs = panel_inputs(common_rows(date(2020, 1, 1), 40))
    common = s.join_common_rows(inputs)
    schedules = s.build_schedules(common)
    controls = s.build_controls(schedules)
    schedule = s.schedule_metrics(schedules)
    schedule["schedule_replay"] = s.schedule_replay_metrics(
        inputs,
        common,
        schedules,
    )
    return {
        "parser": s.parser_metrics(inputs, common),
        "schedule": schedule,
        "edge_support": s.edge_metrics(schedules),
        "diversity_stability": s.diversity_metrics(schedules),
        "controls": s.control_metrics(schedules, controls),
        "determinism_append_replay": s.append_replay_metrics(
            inputs,
            common,
            schedules,
            controls,
        ),
    }


def test_exact_comparison_and_cross_multiplication() -> None:
    assert s.compare(Decimal("1"), Decimal("2")) == "LOWER"
    assert s.compare(Decimal("2"), Decimal("2.0")) == "EQUAL"
    assert s.compare_ratio(
        Decimal("2"),
        Decimal("4"),
        Decimal("1"),
        Decimal("2"),
    ) == "EQUAL"
    assert s.compare_ratio(3, 4, 1, 2) == "HIGHER"


def test_edge_state_uses_frozen_order() -> None:
    previous, current = common_rows(date(2020, 1, 1), 2)
    state = s.edge_state(previous, current)
    assert state.observation_date == current.observation_date
    assert len(state.levels) == len(p.EDGE_NAMES) == 12
    assert set(state.levels) <= set(p.EDGE_LEVELS)


def test_sequence_requires_six_common_rows() -> None:
    assert s.build_schedules(common_rows(date(2020, 1, 1), 5)) == []
    schedules = s.build_schedules(common_rows(date(2020, 1, 1), 6))
    assert len(schedules) == 1
    assert schedules[0].observation_date == date(2020, 1, 6)
    assert all(prompt.count("\n") == 61 for prompt in schedules[0].prompts)


def test_dst_overlap_is_suppressed_action_independently() -> None:
    schedules = s.build_schedules(common_rows(date(2023, 3, 5), 12))
    accepted = [row for row in schedules if row.reservation_state == "ACCEPTED"]
    suppressed = [
        row for row in schedules if row.reservation_state == "SUPPRESSED_OVERLAP"
    ]
    assert suppressed
    assert all(
        right.entry_utc >= left.exit_utc
        for left, right in zip(accepted, accepted[1:])
    )
    assert all(
        row.exit_utc - row.entry_utc == timedelta(hours=24) for row in accepted
    )


def test_fall_back_clock_and_exact_boundary_acceptance() -> None:
    normal = s.build_schedules(common_rows(date(2023, 1, 1), 9))
    accepted = [row for row in normal if row.reservation_state == "ACCEPTED"]
    assert len(accepted) >= 2
    assert accepted[1].entry_utc == accepted[0].exit_utc

    fall_back = s.build_schedules(common_rows(date(2023, 11, 1), 12))
    accepted = [
        row for row in fall_back if row.reservation_state == "ACCEPTED"
    ]
    assert all(
        row.exit_utc - row.entry_utc == timedelta(minutes=288 * 5)
        for row in accepted
    )
    assert all(
        right.entry_utc >= left.exit_utc
        for left, right in zip(accepted, accepted[1:])
    )


def test_role_crossing_is_audit_only_and_not_model_eligible() -> None:
    schedules = s.build_schedules(common_rows(date(2021, 12, 24), 8))
    crossing = [row for row in schedules if row.role == "ROLE_CROSSING"]
    assert crossing
    assert all(not row.model_eligible for row in crossing)
    metrics = s.schedule_metrics(schedules)
    assert metrics["complete_primary_role_crossing"] == 0


def test_controls_cover_every_eligible_position_and_identity() -> None:
    schedules = s.build_schedules(common_rows(date(2020, 2, 1), 10))
    eligible = [row for row in schedules if row.model_eligible]
    controls = s.build_controls(schedules)
    assert len(controls) == (
        len(eligible) * len(p.POSITION_CONTEXTS) * len(p.CONTROL_IDS)
    )
    assert [row.control_id for row in controls[:8]] == list(p.CONTROL_IDS)
    group = next(row for row in controls if row.control_id == "group_order_rotation")
    assert group.semantic_difference is False


def test_suppressed_and_role_crossing_rows_produce_no_controls() -> None:
    eligible = s.build_schedules(common_rows(date(2020, 2, 1), 6))[0]
    suppressed = replace(
        eligible,
        reservation_state="SUPPRESSED_OVERLAP",
        role="SUPPRESSED",
        model_eligible=False,
    )
    crossing = replace(
        eligible,
        role="ROLE_CROSSING",
        model_eligible=False,
    )
    controls = s.build_controls((eligible, suppressed, crossing))
    assert len(controls) == (
        len(p.POSITION_CONTEXTS) * len(p.CONTROL_IDS)
    )
    assert {row.observation_date for row in controls} == {
        eligible.observation_date
    }


def test_masked_controls_never_enter_primary_prompt() -> None:
    schedule = s.build_schedules(common_rows(date(2020, 2, 1), 6))[0]
    controls = s.build_controls([schedule])
    assert "MASKED" not in schedule.prompts[0]
    assert any("MASKED" in row.prompt for row in controls)


def test_flow_parser_does_not_numeric_parse_hidden_columns(tmp_path: Path) -> None:
    path = tmp_path / "flow.csv.gz"
    row = ["not-decoded"] * len(p.FLOW_HEADER)
    index = {column: offset for offset, column in enumerate(p.FLOW_HEADER)}
    row[index["observation_date"]] = "2020-01-02"
    for column in (
        "total_pcr",
        "index_pcr",
        "equity_pcr",
        "vix_pcr",
        "spx_pcr",
    ):
        row[index[column]] = "1.0"
    row[index["index_volume"]] = "1000"
    row[index["vix_volume"]] = "100"
    row[index["response_sha256"]] = "a" * 64
    write_gzip_csv(path, p.FLOW_HEADER, [row])
    result = s._read_panel(path, expected_header=p.FLOW_HEADER, parser=s._parse_flow)
    assert result.audit.value_rows == 1
    assert result.rows[date(2020, 1, 2)].index_volume == 1000


def test_pre2020_rows_are_date_only(tmp_path: Path) -> None:
    path = tmp_path / "term.csv.gz"
    rows = [
        ["2019-12-31", "bad", "bad", "bad"],
        ["2020-01-02", "12", "13", "14"],
    ]
    write_gzip_csv(path, p.TERM_HEADER, rows)
    result = s._read_panel(path, expected_header=p.TERM_HEADER, parser=s._parse_term)
    assert result.audit.date_only_rows == 1
    assert result.audit.value_rows == 1


def test_post2023_row_rejects_before_non_date_parse(tmp_path: Path) -> None:
    path = tmp_path / "term.csv.gz"
    write_gzip_csv(
        path,
        p.TERM_HEADER,
        [["2024-01-02", "bad", "bad", "bad"]],
    )
    with pytest.raises(ValueError, match="post-2023"):
        s._read_panel(path, expected_header=p.TERM_HEADER, parser=s._parse_term)


def test_prefix_snapshot_is_sealed_before_date_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "term.csv.gz"
    write_gzip_csv(
        path,
        p.TERM_HEADER,
        [
            ["2020-12-31", "12", "13", "14"],
            ["2021-01-01", "12", "13", "14"],
        ],
    )
    events: list[str] = []
    original_seal = s._seal_due_prefixes
    original_parse = s._parse_date

    def seal(*args: object, **kwargs: object) -> None:
        if args[0] == "2021-01-01":
            events.append("snapshot")
        original_seal(*args, **kwargs)

    def parse(token: str) -> date:
        if token == "2021-01-01":
            events.append("date_parser")
        return original_parse(token)

    monkeypatch.setattr(s, "_seal_due_prefixes", seal)
    monkeypatch.setattr(s, "_parse_date", parse)
    result = s._read_panel(
        path,
        expected_header=p.TERM_HEADER,
        parser=s._parse_term,
    )
    assert events == ["snapshot", "date_parser"]
    assert tuple(result.prefix_rows["2021-01-01"]) == (
        date(2020, 12, 31),
    )


@pytest.mark.parametrize(
    "token",
    ["+1", "1e3", " 1", "1 ", "-1", "0", "١", "１２"],
)
def test_decimal_grammar_is_fail_closed(token: str) -> None:
    with pytest.raises(ValueError):
        s._decimal(token, field="probe")


@pytest.mark.parametrize("token", ["١", "１２", "1.0", "+1", "0"])
def test_integer_grammar_is_ascii_base_ten_only(token: str) -> None:
    with pytest.raises(ValueError):
        s._integer(token, field="probe")


@pytest.mark.parametrize(
    "token",
    ["２０２０-01-01", "٢٠٢٠-01-01", "2020-١١-01"],
)
def test_date_grammar_is_ascii_only(token: str) -> None:
    with pytest.raises(ValueError):
        s._parse_date(token)


def test_common_join_requires_exact_vix_identity() -> None:
    inputs = panel_inputs(common_rows(date(2020, 1, 1), 8))
    tail = dict(inputs.tail.rows)
    first = min(tail)
    tail[first] = replace(tail[first], vix=Decimal("999"))
    altered = replace(inputs, tail=replace(inputs.tail, rows=tail))
    with pytest.raises(ValueError, match="VIX identity"):
        s.join_common_rows(altered)


def test_schedule_and_control_records_have_frozen_column_order() -> None:
    schedule = s.build_schedules(common_rows(date(2020, 2, 1), 6))[0]
    assert tuple(s.schedule_record(schedule)) == s.SOURCE_OUTPUT_COLUMNS
    control = s.build_controls([schedule])[0]
    assert tuple(s.control_record(control)) == s.CONTROL_OUTPUT_COLUMNS


def test_deterministic_gzip_has_fixed_bytes() -> None:
    record = {column: "x" for column in s.SOURCE_OUTPUT_COLUMNS}
    first = s.deterministic_csv_gzip([record], s.SOURCE_OUTPUT_COLUMNS)
    second = s.deterministic_csv_gzip([record], s.SOURCE_OUTPUT_COLUMNS)
    assert first == second
    assert gzip.decompress(first).startswith(b"observation_date,")


def test_prefix_and_synthetic_append_leave_prior_records_identical() -> None:
    rows = common_rows(date(2020, 1, 1), 40)
    inputs = panel_inputs(rows)
    common = s.join_common_rows(inputs)
    schedules = s.build_schedules(common)
    controls = s.build_controls(schedules)
    schedule_replay = s.schedule_replay_metrics(inputs, common, schedules)
    assert schedule_replay["passed"] is True
    replay = s.append_replay_metrics(inputs, common, schedules, controls)
    assert replay["passed"] is True
    assert replay["synthetic_append_prior_identical"] is True
    assert replay["synthetic_append_prior_schedule_identical"] is True
    assert replay["synthetic_append_prior_control_identical"] is True


def test_prefix_replay_rejects_an_omitted_prior_record() -> None:
    rows = common_rows(date(2020, 1, 1), 40)
    inputs = panel_inputs(rows)
    cutoff = "2024-01-01"
    damaged_term_prefix = dict(inputs.term.prefix_rows[cutoff])
    damaged_term_prefix.pop(rows[8].observation_date)
    damaged_term = replace(
        inputs.term,
        prefix_rows={
            **inputs.term.prefix_rows,
            cutoff: damaged_term_prefix,
        },
    )
    damaged = replace(inputs, term=damaged_term)
    common = s.join_common_rows(inputs)
    schedules = s.build_schedules(common)
    controls = s.build_controls(schedules)
    assert s.schedule_replay_metrics(
        damaged,
        common,
        schedules,
    )["passed"] is False
    replay = s.append_replay_metrics(
        damaged,
        common,
        schedules,
        controls,
    )
    assert replay["prefixes"][cutoff]["passed"] is False


def test_control_metrics_distinguish_order_from_semantics() -> None:
    schedules = s.build_schedules(common_rows(date(2020, 2, 1), 15))
    controls = s.build_controls(schedules)
    metrics = s.control_metrics(schedules, controls)
    group = metrics["by_control"]["group_order_rotation"]
    assert group["byte_difference_share"] == 1.0
    assert group["semantic_difference_share"] == 0.0


def test_result_hash_detects_mutation() -> None:
    core = {"decision": "fail", "gates": []}
    payload = {**core, "result_hash": s.canonical_hash(core)}
    mutated = dict(payload)
    mutated["decision"] = "pass"
    recomputed = {
        key: value for key, value in mutated.items() if key != "result_hash"
    }
    assert mutated["result_hash"] != s.canonical_hash(recomputed)


def test_terminal_detail_schema_accepts_exact_producer_metrics() -> None:
    s._validate_terminal_detail_schema(synthetic_terminal_details())


def test_terminal_detail_schema_rejects_nested_extra_and_missing_prefix() -> None:
    with_extra = synthetic_terminal_details()
    with_extra["schedule"]["schedule_replay"]["forbidden"] = True
    with pytest.raises(RuntimeError, match="schedule replay fields"):
        s._validate_terminal_detail_schema(with_extra)

    missing_prefix = synthetic_terminal_details()
    missing_prefix["determinism_append_replay"]["prefixes"].pop(
        "2024-01-01"
    )
    with pytest.raises(RuntimeError, match="append replay prefixes"):
        s._validate_terminal_detail_schema(missing_prefix)


def test_publication_transaction_rolls_back_every_final_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(s, "REPOSITORY_ROOT", tmp_path)
    real_link = s.os.link
    calls = 0

    def failing_link(
        source: str | Path,
        target: str | Path,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected report publication failure")
        real_link(source, target, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(s.os, "link", failing_link)
    artifacts = (
        ("data/source.gz", b"source"),
        ("data/control.gz", b"control"),
        ("results/pass.json", b"report"),
    )
    with pytest.raises(OSError, match="injected"):
        s.publish_write_once_transaction(artifacts)
    assert all(not (tmp_path / path).exists() for path, _ in artifacts)
    assert not list(tmp_path.rglob("*.cefs-stage-*"))


def test_partial_terminal_state_aborts_without_new_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(s, "REPOSITORY_ROOT", tmp_path)
    source = tmp_path / s.SOURCE_OUTPUT
    source.parent.mkdir(parents=True)
    source.write_bytes(b"partial")
    with pytest.raises(RuntimeError, match="partial terminal"):
        s.pre_run_terminal_state()
    assert not (tmp_path / s.REJECTION_REPORT).exists()


def test_conflicting_terminal_reports_abort_without_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(s, "REPOSITORY_ROOT", tmp_path)
    write_terminal_fixture(tmp_path, "pass")
    rejection = tmp_path / s.REJECTION_REPORT
    rejection.parent.mkdir(parents=True, exist_ok=True)
    rejection.write_text("{}")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(RuntimeError, match="conflicting terminal"):
        s.pre_run_terminal_state()
    after = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before


def test_rejection_with_pass_output_aborts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(s, "REPOSITORY_ROOT", tmp_path)
    write_terminal_fixture(tmp_path, "fail")
    source = tmp_path / s.SOURCE_OUTPUT
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"conflict")
    with pytest.raises(RuntimeError, match="rejection conflicts"):
        s.pre_run_terminal_state()


def test_hash_drifted_pass_output_aborts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(s, "REPOSITORY_ROOT", tmp_path)
    write_terminal_fixture(tmp_path, "pass")
    monkeypatch.setattr(
        s,
        "_terminal_expected_gate_checks",
        lambda index, details: {"fixture": True},
    )
    monkeypatch.setattr(
        s,
        "_validate_terminal_detail_schema",
        lambda details: None,
    )
    (tmp_path / s.SOURCE_OUTPUT).write_bytes(b"drift")
    with pytest.raises(RuntimeError, match="output hash mismatch"):
        s.pre_run_terminal_state()


def test_terminal_pass_recomputes_canonical_output_row_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(s, "REPOSITORY_ROOT", tmp_path)
    report = write_terminal_fixture(tmp_path, "pass")
    report["source_row_hash"] = "0" * 64
    core = {
        key: value for key, value in report.items() if key != "result_hash"
    }
    report["result_hash"] = s.canonical_hash(core)
    (tmp_path / s.PASS_REPORT).write_bytes(s._json_bytes(report))
    monkeypatch.setattr(
        s,
        "_terminal_expected_gate_checks",
        lambda index, details: {"fixture": True},
    )
    monkeypatch.setattr(
        s,
        "_validate_terminal_detail_schema",
        lambda details: None,
    )
    with pytest.raises(RuntimeError, match="canonical row hash mismatch"):
        s.pre_run_terminal_state()


def test_terminal_pass_rejects_self_hashed_invalid_source_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(s, "REPOSITORY_ROOT", tmp_path)
    report = write_terminal_fixture(tmp_path, "pass")
    records = s._terminal_output_records(
        s.SOURCE_OUTPUT,
        s.SOURCE_OUTPUT_COLUMNS,
    )
    records[0]["role"] = "EVAL"
    encoded = s.deterministic_csv_gzip(records, s.SOURCE_OUTPUT_COLUMNS)
    (tmp_path / s.SOURCE_OUTPUT).write_bytes(encoded)
    report["source_output"]["sha256"] = hashlib.sha256(encoded).hexdigest()
    report["source_row_hash"] = hashlib.sha256(
        s._canonical_records(records)
    ).hexdigest()
    core = {
        key: value for key, value in report.items() if key != "result_hash"
    }
    report["result_hash"] = s.canonical_hash(core)
    (tmp_path / s.PASS_REPORT).write_bytes(s._json_bytes(report))
    monkeypatch.setattr(
        s,
        "_terminal_expected_gate_checks",
        lambda index, details: {"fixture": True},
    )
    monkeypatch.setattr(
        s,
        "_validate_terminal_detail_schema",
        lambda details: None,
    )
    with pytest.raises(RuntimeError, match="role mismatch"):
        s.pre_run_terminal_state()


def test_terminal_pass_rejects_self_hashed_invalid_control_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(s, "REPOSITORY_ROOT", tmp_path)
    report = write_terminal_fixture(tmp_path, "pass")
    records = s._terminal_output_records(
        s.CONTROL_OUTPUT,
        s.CONTROL_OUTPUT_COLUMNS,
    )
    records[0]["semantic_difference"] = "false"
    encoded = s.deterministic_csv_gzip(records, s.CONTROL_OUTPUT_COLUMNS)
    (tmp_path / s.CONTROL_OUTPUT).write_bytes(encoded)
    report["control_output"]["sha256"] = hashlib.sha256(encoded).hexdigest()
    report["control_row_hash"] = hashlib.sha256(
        s._canonical_records(records)
    ).hexdigest()
    core = {
        key: value for key, value in report.items() if key != "result_hash"
    }
    report["result_hash"] = s.canonical_hash(core)
    (tmp_path / s.PASS_REPORT).write_bytes(s._json_bytes(report))
    monkeypatch.setattr(
        s,
        "_terminal_expected_gate_checks",
        lambda index, details: {"fixture": True},
    )
    monkeypatch.setattr(
        s,
        "_validate_terminal_detail_schema",
        lambda details: None,
    )
    with pytest.raises(RuntimeError, match="control rows do not replay"):
        s.pre_run_terminal_state()


def test_invalid_terminal_result_hash_aborts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(s, "REPOSITORY_ROOT", tmp_path)
    write_terminal_fixture(tmp_path, "fail")
    path = tmp_path / s.REJECTION_REPORT
    payload = json.loads(path.read_text())
    payload["result_hash"] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="result hash mismatch"):
        s.pre_run_terminal_state()


def test_valid_terminal_pass_returns_idempotently_before_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(s, "REPOSITORY_ROOT", tmp_path)
    report = write_terminal_fixture(tmp_path, "pass")
    monkeypatch.setattr(
        s,
        "_validate_terminal_execution_seal",
        fake_terminal_seal,
    )
    monkeypatch.setattr(
        s,
        "_terminal_expected_gate_checks",
        lambda index, details: {"fixture": True},
    )
    monkeypatch.setattr(
        s,
        "_validate_terminal_detail_schema",
        lambda details: None,
    )

    def forbidden_authority_call() -> None:
        raise AssertionError("terminal return must precede authority evaluation")

    monkeypatch.setattr(s, "validate_execution_seal", forbidden_authority_call)
    assert s.run_official() == report


def test_incomplete_terminal_pass_is_not_idempotently_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(s, "REPOSITORY_ROOT", tmp_path)
    report = write_terminal_fixture(tmp_path, "pass")
    path = tmp_path / s.PASS_REPORT
    report["authority"] = {}
    report["gates"] = []
    core = {
        key: value for key, value in report.items() if key != "result_hash"
    }
    report["result_hash"] = s.canonical_hash(core)
    path.write_bytes(s._json_bytes(report))
    with pytest.raises(RuntimeError, match="gate count mismatch"):
        s.pre_run_terminal_state()


def test_terminal_pass_with_empty_authority_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(s, "REPOSITORY_ROOT", tmp_path)
    report = write_terminal_fixture(tmp_path, "pass")
    report["authority"] = {}
    core = {
        key: value for key, value in report.items() if key != "result_hash"
    }
    report["result_hash"] = s.canonical_hash(core)
    (tmp_path / s.PASS_REPORT).write_bytes(s._json_bytes(report))
    monkeypatch.setattr(
        s,
        "_validate_terminal_execution_seal",
        fake_terminal_seal,
    )
    monkeypatch.setattr(
        s,
        "_terminal_expected_gate_checks",
        lambda index, details: {"fixture": True},
    )
    monkeypatch.setattr(
        s,
        "_validate_terminal_detail_schema",
        lambda details: None,
    )
    with pytest.raises(RuntimeError, match="pass authority mismatch"):
        s.pre_run_terminal_state()


def test_valid_gate_one_rejection_returns_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(s, "REPOSITORY_ROOT", tmp_path)
    report = write_terminal_fixture(tmp_path, "fail")

    def forbidden_authority_call() -> None:
        raise AssertionError("terminal return must precede authority evaluation")

    monkeypatch.setattr(s, "validate_execution_seal", forbidden_authority_call)
    assert s.run_official() == report


def test_current_preregistration_producer_must_match_sealed_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.loads(
        s.repository_path(s.PREREGISTRATION_PATH).read_text()
    )
    sealed = dict(payload["authority"]["producer"])
    monkeypatch.setattr(
        p,
        "producer_binding",
        lambda: {**sealed, "sha256": "0" * 64},
    )
    with pytest.raises(RuntimeError, match="producer differs"):
        s._validate_current_preregistration_producer(payload)


def test_forbidden_counters_are_all_zero() -> None:
    counters = s.forbidden_counters()
    assert tuple(counters) == s.FORBIDDEN_COUNTER_NAMES
    assert all(value == 0 for value in counters.values())


def test_contract_and_preregistration_authority_validate_without_rows() -> None:
    authority = s.validate_frozen_authority()
    assert authority["contract"]["sha256"] == s.CONTRACT_SHA256
    assert authority["preregistration"]["manifest_hash"] == (
        s.PREREGISTRATION_MANIFEST_HASH
    )
