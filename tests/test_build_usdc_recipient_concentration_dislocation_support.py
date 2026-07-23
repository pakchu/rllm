from __future__ import annotations

import hashlib
import gzip
import json
import subprocess
from collections import Counter
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from training import build_usdc_recipient_concentration_dislocation_support as urcd


UTC = timezone.utc


def dt(value: str) -> datetime:
    return urcd.parse_time(value)


def _event(
    index: int,
    available_at: datetime,
    *,
    amount: int = 10,
    recipient: int | None = None,
) -> urcd.Event:
    recipient_id = index if recipient is None else recipient
    return urcd.Event(
        amount_raw=amount,
        recipient=f"0x{recipient_id + 1:040x}",
        available_at=available_at,
        block_timestamp=available_at - timedelta(minutes=20),
        block_number=10_000 + index,
        transaction_index=index,
        log_index=index,
        block_hash=f"0x{index + 1:064x}",
        transaction_hash=f"0x{index + 10_000:064x}",
    )


def _metric(
    endpoint: datetime,
    *,
    hhi: Fraction = Fraction(1, 2),
    amount: int = 100,
    valid: bool = True,
    identity_seed: int = 1,
) -> urcd.WindowMetric:
    return urcd.WindowMetric(
        endpoint=endpoint,
        valid=valid,
        event_count=4 if valid else 0,
        recipient_count=3 if valid else 0,
        total_amount_raw=amount if valid else 0,
        amount_hhi=hhi if valid else Fraction(0),
        event_count_hhi=hhi if valid else Fraction(0),
        row_identities=(
            (
                f"0x{identity_seed:064x}",
                f"0x{identity_seed + 1:064x}",
                identity_seed,
            ),
        )
        if valid
        else (),
    )


def _snapshot(endpoint: datetime, side: int = 1) -> urcd.Snapshot:
    return urcd.Snapshot(
        metric=_metric(endpoint, hhi=Fraction(1, 10)),
        statistic="amount_hhi",
        current_stat=Fraction(1, 10),
        q20=Fraction(1, 5),
        q80=Fraction(4, 5),
        q50_amount_raw=50,
        valid_prior_windows=180,
        materiality_applied=True,
        state=side,
    )


def _candidate(
    entry: datetime,
    side: int,
    split: str,
    *,
    control: str = "primary",
    seed: int = 1,
) -> urcd.Candidate:
    decision = entry - urcd.ENTRY_DELAY
    snapshot = _snapshot(decision, side)
    return urcd.Candidate(
        control=control,
        split=split,
        decision_time=decision,
        entry_time=entry,
        exit_time=entry + urcd.HOLD,
        side=side,
        previous_state=0,
        snapshot=snapshot,
        signal_id=f"{seed:064x}",
    )


def test_window_is_exactly_left_open_right_closed_and_hhi_is_exact() -> None:
    endpoint = dt("2021-01-02T00:00:00Z")
    lower = endpoint - timedelta(hours=24)
    events = [
        _event(0, lower, recipient=0),
        _event(1, lower + timedelta(seconds=1), recipient=0),
        _event(2, endpoint - timedelta(hours=3), recipient=0),
        _event(3, endpoint - timedelta(hours=2), recipient=1),
        _event(4, endpoint, recipient=2),
    ]
    metrics = urcd.build_window_metrics(
        events,
        anchor_start=endpoint,
        end_exclusive=endpoint + urcd.SIX_HOURS,
        coverage_start=lower,
    )
    metric = metrics[endpoint]
    assert metric.valid is True
    assert metric.event_count == 4
    assert metric.recipient_count == 3
    assert metric.total_amount_raw == 40
    assert metric.amount_hhi == Fraction(3, 8)
    assert metric.event_count_hhi == Fraction(3, 8)
    assert events[0].identity not in metric.row_identities
    assert events[-1].identity in metric.row_identities


def test_snapshot_uses_strict_prior_daily_panel_and_independent_amount_sort() -> None:
    endpoint = dt("2022-01-01T00:00:00Z")
    metrics = {endpoint: _metric(endpoint, hhi=Fraction(1, 100), amount=10_000)}
    for day in range(1, 181):
        prior = endpoint - day * timedelta(days=1)
        metrics[prior] = _metric(
            prior,
            hhi=Fraction(day, 200),
            amount=181 - day,
            identity_seed=day + 10,
        )
    snapshot = urcd.snapshot_at(
        metrics, endpoint, statistic="amount_hhi", apply_materiality=True
    )
    assert snapshot.valid_prior_windows == 180
    assert snapshot.q20 == Fraction(36, 200)
    assert snapshot.q80 == Fraction(144, 200)
    assert snapshot.q50_amount_raw == 90
    assert snapshot.state == 1
    assert snapshot.metric.endpoint not in [
        metrics[endpoint - day * timedelta(days=1)].endpoint
        for day in range(1, 181)
    ]


def test_equal_reference_thresholds_are_neutral() -> None:
    endpoint = dt("2022-01-01T06:00:00Z")
    metrics = {endpoint: _metric(endpoint, hhi=Fraction(1, 2), amount=100)}
    for day in range(1, 181):
        prior = endpoint - day * timedelta(days=1)
        metrics[prior] = _metric(prior, hhi=Fraction(1, 2), amount=100)
    assert urcd.snapshot_at(
        metrics, endpoint, statistic="amount_hhi", apply_materiality=True
    ).state == 0


def test_signal_id_matches_frozen_canonical_json() -> None:
    decision = dt("2023-01-02T06:00:00Z")
    identities = [
        (f"0x{2:064x}", f"0x{3:064x}", 2),
        (f"0x{1:064x}", f"0x{4:064x}", 1),
    ]
    payload = {
        "candidate": "URCD-72",
        "control": "primary",
        "decision_time": "2023-01-02T06:00:00Z",
        "row_identities": [list(item) for item in sorted(identities)],
        "side": "LONG",
    }
    expected = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    assert urcd.signal_id("primary", decision, 1, identities) == expected


def test_permutations_are_deterministic_and_preserve_exact_marginals() -> None:
    start = dt("2021-01-01T00:00:00Z")
    events = [
        _event(
            index,
            start + index * timedelta(hours=6),
            amount=100 + index,
            recipient=index % 4,
        )
        for index in range(20)
    ]
    recipient_a = urcd.permute_field(events, "recipient_year_permutation")
    recipient_b = urcd.permute_field(events, "recipient_year_permutation")
    amount = urcd.permute_field(events, "amount_year_permutation")
    assert recipient_a == recipient_b
    assert Counter(row.recipient for row in recipient_a) == Counter(
        row.recipient for row in events
    )
    assert [row.amount_raw for row in recipient_a] == [row.amount_raw for row in events]
    assert Counter(row.amount_raw for row in amount) == Counter(
        row.amount_raw for row in events
    )
    assert [row.recipient for row in amount] == [row.recipient for row in events]
    assert [row.identity for row in recipient_a] == [row.identity for row in events]
    assert any(left.recipient != right.recipient for left, right in zip(events, recipient_a))
    assert any(left.amount_raw != right.amount_raw for left, right in zip(events, amount))


def test_tail_transition_resets_after_invalid_previous_anchor() -> None:
    metrics: dict[datetime, urcd.WindowMetric] = {}
    for index, endpoint in enumerate(urcd.iter_anchors(urcd.SOURCE_START, urcd.SEALED_FROM)):
        hhi = Fraction(index % 9 + 1, 10)
        metrics[endpoint] = _metric(
            endpoint, hhi=hhi, amount=100, identity_seed=index + 1
        )
    target = dt("2021-05-01T00:00:00Z")
    metrics[target - urcd.SIX_HOURS] = _metric(
        target - urcd.SIX_HOURS, hhi=Fraction(1, 2), identity_seed=90_000
    )
    metrics[target] = _metric(target, hhi=Fraction(1, 100), identity_seed=90_001)
    metrics[target + urcd.SIX_HOURS] = _metric(
        target + urcd.SIX_HOURS, hhi=Fraction(1, 100), identity_seed=90_002
    )
    metrics[target + 2 * urcd.SIX_HOURS] = _metric(
        target + 2 * urcd.SIX_HOURS, valid=False
    )
    metrics[target + 3 * urcd.SIX_HOURS] = _metric(
        target + 3 * urcd.SIX_HOURS, hhi=Fraction(1, 100), identity_seed=90_003
    )
    raw = urcd.build_raw_candidates("primary", metrics)
    by_time = {row.decision_time: row for row in raw}
    assert by_time[target].side == 1
    assert target + urcd.SIX_HOURS not in by_time
    assert by_time[target + 3 * urcd.SIX_HOURS].side == 1
    assert by_time[target + 3 * urcd.SIX_HOURS].previous_state == 0


def test_split_crossing_is_removed_before_independent_reservation() -> None:
    snapshot = _snapshot(dt("2021-01-01T00:00:00Z"))

    def raw(decision: datetime, seed: int) -> urcd.RawCandidate:
        return urcd.RawCandidate(
            control="primary",
            decision_time=decision,
            side=1 if seed % 2 else -1,
            previous_state=0,
            snapshot=replace_snapshot(snapshot, decision),
            signal_id=f"{seed:064x}",
        )

    candidates = [
        raw(dt("2020-12-30T00:00:00Z"), 1),
        raw(dt("2021-01-01T00:00:00Z"), 2),
        raw(dt("2022-12-30T00:00:00Z"), 3),
        raw(dt("2023-01-01T00:00:00Z"), 4),
    ]
    scheduled = urcd.schedule_candidates(candidates, "primary")
    assert [(row.split, row.decision_time) for row in scheduled] == [
        ("train", dt("2021-01-01T00:00:00Z")),
        ("selection", dt("2023-01-01T00:00:00Z")),
    ]


def replace_snapshot(snapshot: urcd.Snapshot, endpoint: datetime) -> urcd.Snapshot:
    return urcd.Snapshot(
        metric=_metric(endpoint, hhi=snapshot.metric.amount_hhi),
        statistic=snapshot.statistic,
        current_stat=snapshot.current_stat,
        q20=snapshot.q20,
        q80=snapshot.q80,
        q50_amount_raw=snapshot.q50_amount_raw,
        valid_prior_windows=snapshot.valid_prior_windows,
        materiality_applied=snapshot.materiality_applied,
        state=snapshot.state,
    )


def test_direction_flip_preserves_exact_clock_and_reverses_only_side() -> None:
    primary = [
        _candidate(dt("2022-01-01T00:10:00Z"), 1, "train", seed=1),
        _candidate(dt("2023-03-01T00:10:00Z"), -1, "selection", seed=2),
    ]
    flipped = urcd.direction_flip(primary)
    assert [row.entry_time for row in flipped] == [row.entry_time for row in primary]
    assert [row.exit_time for row in flipped] == [row.exit_time for row in primary]
    assert [row.side for row in flipped] == [-row.side for row in primary]
    assert all(row.control == "direction_flip" for row in flipped)


def _well_distributed_primary() -> list[urcd.Candidate]:
    rows: list[urcd.Candidate] = []
    seed = 1
    for year, count, split in ((2021, 40, "train"), (2022, 40, "train"), (2023, 36, "selection")):
        start = datetime(year, 1, 2, 0, 10, tzinfo=UTC)
        for index in range(count):
            entry = start + index * timedelta(days=9)
            rows.append(_candidate(entry, 1 if index % 2 == 0 else -1, split, seed=seed))
            seed += 1
    return rows


def test_support_gates_pass_distributed_balanced_clock_and_fail_identical_placebo() -> None:
    primary = _well_distributed_primary()
    controls: dict[str, list[urcd.Candidate]] = {
        control: [replace_control(row, control) for row in primary]
        for control in urcd.CONTROL_ORDER
    }
    for control in urcd.PERMUTATION_CONTROLS:
        controls[control] = [
            shift_candidate(row, control, timedelta(days=1)) for row in primary
        ]
    _, checks = urcd.source_support_checks(controls)
    assert all(checks.values())

    controls["recipient_year_permutation"] = [
        replace_control(row, "recipient_year_permutation") for row in primary
    ]
    _, checks = urcd.source_support_checks(controls)
    assert checks["recipient_year_permutation_train_exact_entry_jaccard"] is False
    assert checks["recipient_year_permutation_selection_same_side_reproduction"] is False


def replace_control(row: urcd.Candidate, control: str) -> urcd.Candidate:
    return urcd.Candidate(
        control=control,
        split=row.split,
        decision_time=row.decision_time,
        entry_time=row.entry_time,
        exit_time=row.exit_time,
        side=row.side,
        previous_state=row.previous_state,
        snapshot=row.snapshot,
        signal_id=hashlib.sha256(f"{control}|{row.signal_id}".encode()).hexdigest(),
    )


def shift_candidate(
    row: urcd.Candidate, control: str, delay: timedelta
) -> urcd.Candidate:
    shifted = replace_control(row, control)
    return urcd.Candidate(
        control=control,
        split=row.split,
        decision_time=shifted.decision_time + delay,
        entry_time=shifted.entry_time + delay,
        exit_time=shifted.exit_time + delay,
        side=shifted.side,
        previous_state=shifted.previous_state,
        snapshot=shifted.snapshot,
        signal_id=shifted.signal_id,
    )


def test_novelty_metrics_are_exact_and_bidirectional() -> None:
    start = dt("2023-01-01T00:00:00Z")
    left = [start, start + timedelta(hours=12), start + timedelta(hours=24)]
    right = [start, start + timedelta(hours=17), start + timedelta(hours=100)]
    result = urcd.novelty_metrics(left, right, timedelta(hours=6))
    assert result["exact_entry_jaccard"] == "1/5"
    assert result["left_near_share"] == "2/3"
    assert result["right_near_share"] == "2/3"
    assert result["maximum_bidirectional_containment"] == "2/3"


def test_comparator_loader_semantically_decodes_only_frozen_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(urcd, "REPOSITORY_ROOT", tmp_path)
    path = tmp_path / "comparator.csv.gz"
    header = "candidate,control,entry_time,side,feature\n"
    rows = [
        "SYNTH,primary,2023-10-01T00:00:00Z,LONG,a\n",
        "SYNTH,primary,2024-01-01T00:00:00Z,INVALID,b\n",
        "SYNTH,unknown_future_control,2024-01-01T00:00:00Z,INVALID,c\n",
    ]
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write(header)
        handle.writelines(rows)
    spec = {
        "candidate": "SYNTH",
        "controls": ("primary",),
        "known_controls": ("primary",),
        "path": Path("comparator.csv.gz"),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "header_line_sha256": hashlib.sha256(header.encode()).hexdigest(),
        "comparison_start": "2023-09-01T00:00:00Z",
        "comparison_end_exclusive": "2024-01-01T00:00:00Z",
    }
    views, audit = urcd.load_comparator_entries((spec,))
    assert views == {"SYNTH:primary": (dt("2023-10-01T00:00:00Z"),)}
    assert audit["relevant_four_field_rows_decoded"] == 1
    assert audit["out_of_overlap_timestamp_sentinels_scanned"] == 2
    assert audit["files"]["SYNTH"][
        "out_of_overlap_timestamp_sentinels_scanned"
    ] == 2


def _write_comparator_fixture(path: Path, header: str, rows: list[str]) -> dict[str, str]:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write(header)
        handle.writelines(rows)
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "header_line_sha256": hashlib.sha256(header.encode()).hexdigest(),
    }


def test_all_comparator_bindings_pass_before_first_data_row_is_decoded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(urcd, "REPOSITORY_ROOT", tmp_path)
    header = "candidate,control,entry_time,side\n"
    good = tmp_path / "good.csv.gz"
    bad = tmp_path / "bad.csv.gz"
    good_hashes = _write_comparator_fixture(
        good,
        header,
        ["GOOD,primary,2023-10-01T00:00:00Z,INVALID\n"],
    )
    bad_hashes = _write_comparator_fixture(
        bad,
        header,
        ["BAD,primary,2023-10-01T00:00:00Z,LONG\n"],
    )
    common = {
        "controls": ("primary",),
        "known_controls": ("primary",),
        "comparison_start": "2023-09-01T00:00:00Z",
        "comparison_end_exclusive": "2024-01-01T00:00:00Z",
    }
    specs = (
        {
            **common,
            "candidate": "GOOD",
            "path": Path("good.csv.gz"),
            **good_hashes,
        },
        {
            **common,
            "candidate": "BAD",
            "path": Path("bad.csv.gz"),
            **bad_hashes,
            "sha256": "0" * 64,
        },
    )
    with pytest.raises(RuntimeError, match="comparator hash drift: BAD"):
        urcd.load_comparator_entries(specs)


def test_comparator_unknown_control_and_forbidden_header_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(urcd, "REPOSITORY_ROOT", tmp_path)
    path = tmp_path / "unknown.csv.gz"
    header = "candidate,control,entry_time,side\n"
    hashes = _write_comparator_fixture(
        path,
        header,
        ["SYNTH,mystery,2023-10-01T00:00:00Z,LONG\n"],
    )
    spec = {
        "candidate": "SYNTH",
        "controls": ("primary",),
        "known_controls": ("primary", "secondary"),
        "path": Path("unknown.csv.gz"),
        **hashes,
        "comparison_start": "2023-09-01T00:00:00Z",
        "comparison_end_exclusive": "2024-01-01T00:00:00Z",
    }
    with pytest.raises(RuntimeError, match="row control is unknown"):
        urcd.load_comparator_entries((spec,))

    forbidden = tmp_path / "forbidden.csv.gz"
    forbidden_header = "candidate,control,entry_time,side,future_return\n"
    forbidden_hashes = _write_comparator_fixture(forbidden, forbidden_header, [])
    forbidden_spec = {
        **spec,
        "path": Path("forbidden.csv.gz"),
        **forbidden_hashes,
    }
    with pytest.raises(RuntimeError, match="outcome field forbidden"):
        urcd.load_comparator_entries((forbidden_spec,))


def test_failed_source_support_short_circuits_comparator_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty = {control: [] for control in urcd.CONTROL_ORDER}
    monkeypatch.setattr(urcd, "build_controls", lambda _events: empty)

    def forbidden() -> Any:
        raise AssertionError("comparator loader was opened")

    payload, _ = urcd._build_core(
        [],
        {"eligible_mint_value_rows_decoded": 0},
        artifact_eligible=True,
        comparator_loader=forbidden,
        clock_output=urcd.DEFAULT_CLOCK_OUTPUT,
    )
    assert payload["source_support_passed"] is False
    assert payload["comparator_audit"]["relevant_four_field_rows_decoded"] == 0
    assert payload["outcome_boundary"]["btc_market_rows_decoded"] == 0
    assert payload["decision"].endswith("before_comparators_and_outcomes")


def test_injected_support_pass_cannot_authorize_novelty_or_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty = {control: [] for control in urcd.CONTROL_ORDER}
    monkeypatch.setattr(urcd, "build_controls", lambda _events: empty)
    monkeypatch.setattr(
        urcd, "source_support_checks", lambda _controls: ({}, {"synthetic": True})
    )
    payload, _ = urcd.build_support_from_events([])
    assert payload["source_support_passed"] is True
    assert payload["artifact_eligible"] is False
    assert payload["novelty_status"] == "forbidden_for_injected_or_synthetic_build"
    assert payload["advance_to_strict_outcome_evaluator_freeze"] is False
    assert payload["outcome_boundary"]["pnl_cagr_mdd_values_decoded"] == 0


def test_deterministic_gzip_is_byte_identical_and_sorted() -> None:
    rows = [
        urcd.candidate_row(_candidate(dt("2023-02-01T00:10:00Z"), 1, "selection", seed=2)),
        urcd.candidate_row(_candidate(dt("2022-02-01T00:10:00Z"), -1, "train", seed=1)),
    ]
    first = urcd.deterministic_gzip_csv(rows)
    second = urcd.deterministic_gzip_csv(reversed(rows))
    assert first == second
    with gzip_bytes(first) as text:
        lines = text.splitlines()
    assert lines[1].split(",")[5] < lines[2].split(",")[5]


class gzip_bytes:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self) -> str:
        import gzip

        return gzip.decompress(self.payload).decode("utf-8")

    def __exit__(self, *_args: object) -> None:
        return None


def test_preregistration_binding_reproduces_without_source_value_access() -> None:
    payload = urcd.validate_preregistration()
    assert payload["candidate"] == "URCD-72"
    assert payload["artifact_eligible"] is True
    assert payload["source_values_or_incidence_opened"] is False
    assert payload["comparator_rows_opened_during_preregistration"] is False


def test_mechanism_binding_drift_stops_before_source_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(urcd, "_assert_protocol_committed", lambda: None)
    monkeypatch.setattr(urcd, "validate_preregistration", lambda: {})

    def binding_hash(path: str | Path) -> str:
        candidate = Path(path)
        if candidate == urcd.prereg.BOUNDARY_DOCUMENT:
            return urcd.prereg.BOUNDARY_DOCUMENT_SHA256
        if candidate == urcd.prereg.MECHANISM_DOCUMENT:
            return "0" * 64
        raise AssertionError(f"unexpected later binding read: {candidate}")

    monkeypatch.setattr(urcd, "sha256_file", binding_hash)
    monkeypatch.setattr(
        urcd,
        "load_source_events",
        lambda: (_ for _ in ()).throw(AssertionError("source values opened")),
    )
    with pytest.raises(RuntimeError, match="mechanism document hash drift"):
        urcd.build_real_support_payload()


def test_protocol_commit_guard_rejects_untracked_and_dirty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(urcd, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(urcd, "SCRIPT_PATH", Path("builder.py"))
    monkeypatch.setattr(urcd, "TEST_PATH", Path("test_builder.py"))
    monkeypatch.setattr(urcd, "IMPLEMENTATION_CONTRACT", Path("contract.md"))
    for name in ("builder.py", "test_builder.py", "contract.md"):
        (tmp_path / name).write_text(name)
    calls: list[tuple[str, ...]] = []

    def untracked(*args: str) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 1, "", "untracked")

    monkeypatch.setattr(urcd, "_git_check", untracked)
    with pytest.raises(RuntimeError, match="not committed"):
        urcd._assert_protocol_committed()
    assert calls[0][:3] == ("ls-files", "--error-unmatch", "--")

    results = iter(
        [
            subprocess.CompletedProcess((), 0, "", ""),
            subprocess.CompletedProcess((), 1, "", "dirty"),
        ]
    )
    monkeypatch.setattr(urcd, "_git_check", lambda *_args: next(results))
    with pytest.raises(RuntimeError, match="differs from HEAD"):
        urcd._assert_protocol_committed()


@pytest.mark.parametrize("path", ["../escape", "/tmp/escape"])
def test_repository_path_escape_fails_closed(path: str) -> None:
    with pytest.raises(RuntimeError, match="repository-relative"):
        urcd._path(path)
