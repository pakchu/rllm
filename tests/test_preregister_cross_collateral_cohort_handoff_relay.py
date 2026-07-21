from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from training import preregister_cross_collateral_cohort_handoff_relay as prereg


UTC = timezone.utc


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    return prereg.build_manifest()


def _observation(
    time: datetime,
    *,
    crowd_rank: float = 0.50,
    handoff_rank: float = 0.50,
    handoff_value: float = 0.10,
    combined_valid: bool = True,
    rank_ready: bool = True,
) -> prereg.AnchorObservation:
    return prereg.AnchorObservation(
        time=time,
        combined_valid=combined_valid,
        rank_ready=rank_ready,
        crowd_rank=crowd_rank if rank_ready else None,
        handoff_rank=handoff_rank if rank_ready else None,
        handoff_value=handoff_value if rank_ready else None,
    )


def _armed_long_machine(start: datetime) -> prereg.CCHRStateMachine:
    machine = prereg.CCHRStateMachine()
    assert machine.process(_observation(start, crowd_rank=0.50)) is None
    assert machine.armed is True
    assert (
        machine.process(
            _observation(
                start + timedelta(hours=1),
                crowd_rank=0.95,
                handoff_rank=0.60,
                handoff_value=0.10,
            )
        )
        is None
    )
    assert machine.active is True
    return machine


def test_manifest_is_deterministic_singleton_and_outcome_blind(
    manifest: dict[str, Any],
) -> None:
    assert manifest == prereg.build_manifest()
    assert manifest["protocol_version"] == prereg.PROTOCOL_VERSION
    assert manifest["policy_id"] == "CCHR-288"
    assert manifest["policy"]["singleton"] is True
    assert manifest["policy"]["features"]["parameter_grid"] == []
    assert manifest["outcomes_opened"] is False
    assert manifest["outcome_boundary"] == prereg.OUTCOME_BOUNDARY
    core = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    assert manifest["manifest_hash"] == prereg.canonical_hash(core)
    assert manifest["policy_hash"] == prereg.canonical_hash(manifest["policy"])
    for field in (
        "source_csv_values_read",
        "comparator_rows_read",
        "outcome_bearing_provenance_json_parsed",
        "cchr_feature_rows_derived",
        "signal_incidence_rows_derived",
        "market_rows_loaded",
        "funding_rows_loaded",
        "return_or_pnl_fields",
        "post_2023_rows_loaded",
        "network_calls",
        "subprocess_calls",
    ):
        assert manifest["outcome_boundary"][field] == 0


def test_returned_manifest_cannot_mutate_frozen_module_contracts() -> None:
    artifact = prereg.build_manifest()
    artifact["policy"]["controls"]["no_age"]["difference"] = "drift"
    artifact["pure_clock_requirements"]["pdlh"]["required_member_count"] = 1
    artifact["outcome_boundary"]["market_rows_loaded"] = 1

    assert prereg.CONTROL_DEFINITIONS["no_age"]["difference"] == (
        "primary minimum_age_hours=0"
    )
    assert prereg.PURE_CLOCK_REQUIREMENTS["pdlh"]["required_member_count"] == 16
    assert prereg.OUTCOME_BOUNDARY["market_rows_loaded"] == 0


def test_source_contract_is_oi_independent_and_hash_bound(
    manifest: dict[str, Any],
) -> None:
    source = manifest["source_binding"]
    assert source["path"] == str(prereg.SOURCE_PATH)
    assert source["sha256"] == prereg.SOURCE_SHA256
    assert source["manifest_sha256"] == prereg.SOURCE_MANIFEST_SHA256
    assert source["columns"] == list(prereg.SOURCE_COLUMNS)
    assert tuple(source["columns"]) == (
        "date",
        "um_count_long_short_ratio",
        "um_sum_taker_long_short_vol_ratio",
        "cm_sum_taker_long_short_vol_ratio",
    )
    policy_source = manifest["policy"]["source"]
    assert policy_source["existing_source_complete_excluded"] is True
    assert "source_complete" in policy_source["forbidden_columns"]
    assert all("open_interest" not in column for column in source["columns"])
    assert any(
        "open_interest" in column for column in policy_source["forbidden_columns"]
    )
    assert "full 168" in policy_source["post_gap_quarantine"]


def test_preregistration_hashes_but_never_parses_source_or_comparator_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guarded = {
        prereg._repository_path(prereg.SOURCE_PATH),
        *(
            prereg._repository_path(binding["path"])
            for binding in prereg.COMPARATOR_PROVENANCE_BINDINGS.values()
        ),
    }
    original_open = Path.open
    read_sizes: dict[Path, list[int]] = {path: [] for path in guarded}

    class HashOnlyReader:
        def __init__(self, path: Path, handle: Any) -> None:
            self.path = path
            self.handle = handle

        def __enter__(self) -> HashOnlyReader:
            self.handle.__enter__()
            return self

        def __exit__(self, *args: Any) -> Any:
            return self.handle.__exit__(*args)

        def read(self, size: int = -1) -> bytes:
            assert size == 1024 * 1024
            read_sizes[self.path].append(size)
            return self.handle.read(size)

    def guarded_open(
        path: Path,
        mode: str = "r",
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        resolved = path.resolve()
        handle = original_open(path, mode, *args, **kwargs)
        if resolved in guarded:
            assert mode == "rb"
            return HashOnlyReader(resolved, handle)
        return handle

    monkeypatch.setattr(Path, "open", guarded_open)
    artifact = prereg.build_manifest()
    assert artifact["outcome_boundary"]["source_csv_values_read"] == 0
    assert artifact["outcome_boundary"]["comparator_rows_read"] == 0
    assert all(calls for calls in read_sizes.values())


def test_midrank_requires_exact_strict_prior_168_and_exact_ties() -> None:
    prior = [0.0] * 84 + [1.0] * 84
    assert prereg.empirical_midrank(0.0, prior) == 0.25
    assert prereg.empirical_midrank(1.0, prior) == 0.75
    with pytest.raises(ValueError, match="exactly 168"):
        prereg.empirical_midrank(0.0, prior[:-1])
    with pytest.raises(ValueError, match="finite"):
        prereg.empirical_midrank(float("nan"), prior)


def test_combined_anchor_uses_only_exact_12_row_local_projection() -> None:
    anchor = datetime(2022, 1, 1, 11, 55, tzinfo=UTC)
    rows = [
        prereg.SourceRow(
            time=anchor - timedelta(minutes=5 * offset),
            um_global_ratio=None if offset else 1.1,
            um_taker_ratio=1.0,
            cm_taker_ratio=1.0,
        )
        for offset in range(11, -1, -1)
    ]
    assert prereg.combined_anchor_valid(rows, anchor) is True
    assert prereg.combined_anchor_valid(rows[:-1], anchor) is False

    duplicate = list(rows)
    duplicate[5] = replace(duplicate[5], time=duplicate[4].time)
    assert prereg.combined_anchor_valid(duplicate, anchor) is False

    bad_taker = list(rows)
    bad_taker[3] = replace(bad_taker[3], cm_taker_ratio=0.0)
    assert prereg.combined_anchor_valid(bad_taker, anchor) is False

    bad_global = list(rows)
    bad_global[-1] = replace(bad_global[-1], um_global_ratio=float("nan"))
    assert prereg.combined_anchor_valid(bad_global, anchor) is False
    assert prereg.combined_anchor_valid(rows, anchor - timedelta(minutes=5)) is False


def test_neutral_arm_setup_and_first_age12_handoff_are_exact() -> None:
    start = datetime(2022, 1, 1, 0, 55, tzinfo=UTC)
    machine = _armed_long_machine(start)
    setup = start + timedelta(hours=1)
    for age in range(1, 12):
        assert (
            machine.process(
                _observation(
                    setup + timedelta(hours=age),
                    crowd_rank=0.95,
                    handoff_rank=0.60,
                    handoff_value=0.10,
                )
            )
            is None
        )
    handoff = setup + timedelta(hours=12)
    candidate = machine.process(
        _observation(
            handoff,
            crowd_rank=0.75,
            handoff_rank=0.75,
            handoff_value=-0.10,
        )
    )
    assert candidate == prereg.Candidate(
        episode_start=setup,
        handoff_anchor=handoff,
        decision_time=handoff + timedelta(minutes=5),
        entry_time=handoff + timedelta(minutes=10),
        exit_time=handoff + timedelta(hours=24, minutes=10),
        side=-1,
    )
    assert machine.active is False
    assert machine.armed is False


def test_gap_and_termination_anchor_reset_without_same_anchor_rearm() -> None:
    start = datetime(2022, 2, 1, 0, 55, tzinfo=UTC)
    machine = _armed_long_machine(start)
    cancellation = start + timedelta(hours=2)
    assert machine.process(_observation(cancellation, crowd_rank=0.50)) is None
    assert machine.active is False
    assert machine.armed is False
    assert (
        machine.process(
            _observation(cancellation + timedelta(hours=1), crowd_rank=0.50)
        )
        is None
    )
    assert machine.armed is True

    gap = cancellation + timedelta(hours=2)
    assert (
        machine.process(_observation(gap, combined_valid=False, rank_ready=False))
        is None
    )
    assert machine.active is False
    assert machine.armed is False
    assert (
        machine.process(
            _observation(
                gap + timedelta(hours=1),
                crowd_rank=0.95,
                handoff_rank=0.90,
                handoff_value=0.10,
            )
        )
        is None
    )
    assert machine.active is False
    assert machine.armed is False


def test_age72_handoff_precedes_expiry_and_prior_opposite_blocks_retry() -> None:
    start = datetime(2022, 3, 1, 0, 55, tzinfo=UTC)
    machine = _armed_long_machine(start)
    setup = start + timedelta(hours=1)
    for age in range(1, 72):
        assert (
            machine.process(
                _observation(
                    setup + timedelta(hours=age),
                    crowd_rank=0.95,
                    handoff_rank=0.60,
                    handoff_value=0.10,
                )
            )
            is None
        )
    candidate = machine.process(
        _observation(
            setup + timedelta(hours=72),
            crowd_rank=0.75,
            handoff_rank=0.80,
            handoff_value=-0.10,
        )
    )
    assert candidate is not None

    blocked = _armed_long_machine(start + timedelta(days=10))
    blocked_setup = start + timedelta(days=10, hours=1)
    for age in range(1, 11):
        blocked.process(
            _observation(
                blocked_setup + timedelta(hours=age),
                crowd_rank=0.95,
                handoff_rank=0.60,
                handoff_value=0.10,
            )
        )
    blocked.process(
        _observation(
            blocked_setup + timedelta(hours=11),
            crowd_rank=0.95,
            handoff_rank=0.80,
            handoff_value=-0.10,
        )
    )
    assert (
        blocked.process(
            _observation(
                blocked_setup + timedelta(hours=12),
                crowd_rank=0.75,
                handoff_rank=0.80,
                handoff_value=-0.10,
            )
        )
        is None
    )
    assert blocked.active is True


def test_split_containment_precedes_global_nonoverlap() -> None:
    split = prereg.Split(
        "synthetic",
        datetime(2023, 1, 1, tzinfo=UTC),
        datetime(2023, 1, 5, tzinfo=UTC),
    )
    crossing = prereg.Candidate(
        episode_start=datetime(2022, 12, 31, 23, 55, tzinfo=UTC),
        handoff_anchor=datetime(2023, 1, 1, 11, 55, tzinfo=UTC),
        decision_time=datetime(2023, 1, 1, 12, 0, tzinfo=UTC),
        entry_time=datetime(2023, 1, 1, 12, 5, tzinfo=UTC),
        exit_time=datetime(2023, 1, 2, 12, 5, tzinfo=UTC),
        side=1,
    )
    valid = prereg.Candidate(
        episode_start=datetime(2023, 1, 1, 0, 55, tzinfo=UTC),
        handoff_anchor=datetime(2023, 1, 1, 12, 55, tzinfo=UTC),
        decision_time=datetime(2023, 1, 1, 13, 0, tzinfo=UTC),
        entry_time=datetime(2023, 1, 1, 13, 5, tzinfo=UTC),
        exit_time=datetime(2023, 1, 2, 13, 5, tzinfo=UTC),
        side=-1,
    )
    assert prereg.candidate_split(crossing, [split]) is None
    assert prereg.candidate_split(valid, [split]) == "synthetic"
    end_boundary = prereg.Candidate(
        episode_start=datetime(2023, 1, 2, 0, 55, tzinfo=UTC),
        handoff_anchor=datetime(2023, 1, 2, 12, 55, tzinfo=UTC),
        decision_time=datetime(2023, 1, 2, 13, 0, tzinfo=UTC),
        entry_time=datetime(2023, 1, 2, 13, 5, tzinfo=UTC),
        exit_time=datetime(2023, 1, 3, 13, 5, tzinfo=UTC),
        side=1,
    )
    assert (
        prereg.candidate_split(
            end_boundary,
            [replace(split, end=end_boundary.exit_time)],
        )
        is None
    )
    contained = [
        item
        for item in (crossing, valid)
        if prereg.candidate_split(item, [split]) is not None
    ]
    assert prereg.schedule_nonoverlap(contained) == [valid]

    overlap = prereg.Candidate(
        episode_start=datetime(2023, 1, 1, 1, 55, tzinfo=UTC),
        handoff_anchor=datetime(2023, 1, 1, 13, 55, tzinfo=UTC),
        decision_time=datetime(2023, 1, 1, 14, 0, tzinfo=UTC),
        entry_time=datetime(2023, 1, 1, 14, 5, tzinfo=UTC),
        exit_time=datetime(2023, 1, 2, 14, 5, tzinfo=UTC),
        side=1,
    )
    boundary = end_boundary
    assert prereg.schedule_nonoverlap([overlap, boundary, valid]) == [valid, boundary]
    with pytest.raises(ValueError, match="side"):
        prereg.schedule_nonoverlap([replace(valid, side=0)])
    with pytest.raises(ValueError, match="24 hours"):
        prereg.candidate_split(
            replace(valid, exit_time=valid.entry_time + timedelta(hours=1)),
            [split],
        )
    with pytest.raises(ValueError, match="five-minute aligned"):
        prereg.schedule_nonoverlap(
            [replace(valid, entry_time=valid.entry_time + timedelta(minutes=3))]
        )


def test_random_side_uses_exact_ascii_utc_seconds() -> None:
    entry = datetime(2023, 4, 5, 6, 10, tzinfo=UTC)
    stamp = entry.strftime("%Y-%m-%dT%H:%M:%SZ")
    first = hashlib.sha256(f"CCHR-288|20260721|{stamp}".encode("ascii")).digest()[0]
    assert prereg.random_side(entry) == (1 if first % 2 == 0 else -1)
    assert prereg.random_side(entry.astimezone(timezone(timedelta(hours=9)))) == (
        1 if first % 2 == 0 else -1
    )
    with pytest.raises(ValueError, match="minute-aligned"):
        prereg.random_side(entry + timedelta(microseconds=1))
    with pytest.raises(ValueError, match="timezone-aware"):
        prereg.random_side(entry.replace(tzinfo=None))


def test_state_machine_rejects_non_55_hourly_anchor() -> None:
    machine = prereg.CCHRStateMachine()
    with pytest.raises(ValueError, match="exactly 55"):
        machine.process(_observation(datetime(2023, 1, 1, 0, 50, tzinfo=UTC)))


def test_controls_candidate_map_and_pending_pure_clocks_are_exact(
    manifest: dict[str, Any],
) -> None:
    assert set(prereg.CONTROL_DEFINITIONS) == {
        "crowd_resolution_only",
        "handoff_only",
        "no_age",
        "um_taker_only",
        "cm_stale_1h",
        "one_hour_execution_delay",
        "direction_flip",
        "deterministic_random_side",
    }
    members = manifest["comparator_candidate_map"]
    assert len(members) == 62
    family_counts: dict[str, int] = {}
    for value in members.values():
        family = value["family"]
        family_counts[family] = family_counts.get(family, 0) + 1
    assert family_counts == {
        "ccpr": 6,
        "dlpd": 1,
        "dtv": 24,
        "far": 12,
        "live": 3,
        "pdlh": 16,
    }
    assert list(members) == sorted(members)
    assert manifest["comparator_candidate_map_hash"] == prereg.canonical_hash(members)
    assert set(manifest["pure_clock_requirements"]) == {"pdlh", "dtv", "far", "live"}
    assert manifest["authorization"] == {
        "real_incidence_authorized_by_this_artifact": False,
        "reason": "mandatory pure-clock comparator freeze is a separate prerequisite",
        "outcome_evaluator_authorized": False,
    }
    assert manifest["policy"]["comparator_contract"]["clock_schema"] == list(
        prereg.CLOCK_SCHEMA
    )
    assert (
        manifest["policy"]["comparator_contract"]["legacy_search_execution_forbidden"]
        is True
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(policy_id="OTHER"), "policy ID drift"),
        (
            lambda value: value["policy"]["features"].update(
                rank_lookback_hourly_anchors=1
            ),
            "policy drift",
        ),
        (
            lambda value: value["outcome_boundary"].update(market_rows_loaded=1),
            "outcome boundary drift",
        ),
        (
            lambda value: value["comparator_candidate_map"].pop("dlpd:DLPD-12:primary"),
            "candidate map drift",
        ),
        (
            lambda value: value["authorization"].update(
                real_incidence_authorized_by_this_artifact=True
            ),
            "authorization drift",
        ),
        (
            lambda value: value["mechanism_decision"].update(sha256="0" * 64),
            "mechanism decision binding drift",
        ),
        (
            lambda value: value["source_binding"].update(columns=["date"]),
            "source binding drift",
        ),
        (
            lambda value: value["comparator_provenance_bindings"].pop(
                "ccpr_source_clock"
            ),
            "comparator provenance binding drift",
        ),
    ],
)
def test_rehashed_semantic_drift_fails_closed(
    manifest: dict[str, Any],
    mutation: Any,
    message: str,
) -> None:
    drift = deepcopy(manifest)
    mutation(drift)
    core = {key: value for key, value in drift.items() if key != "manifest_hash"}
    drift["manifest_hash"] = prereg.canonical_hash(core)
    with pytest.raises(RuntimeError, match=message):
        prereg.validate_manifest(drift, verify_sources=False)


def test_write_is_atomic_immutable_and_replayable(tmp_path: Path) -> None:
    output = tmp_path / "cchr-prereg.json"
    cfg = prereg.Config(preregistration_output=str(output))
    artifact = prereg.write_preregistration(cfg)
    assert artifact == json.loads(output.read_text(encoding="utf-8"))
    assert prereg.load_preregistration(output) == artifact
    with pytest.raises(FileExistsError, match="immutable"):
        prereg.write_preregistration(cfg)
    with pytest.raises(ValueError, match="protected"):
        prereg.write_preregistration(
            prereg.Config(preregistration_output=str(prereg.SOURCE_MANIFEST))
        )
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(output)
    with pytest.raises(ValueError, match="symlink"):
        prereg.write_preregistration(prereg.Config(preregistration_output=str(symlink)))


def test_canonical_json_is_order_independent_compact_and_rejects_nan() -> None:
    assert prereg.canonical_json({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    assert prereg.canonical_hash({"b": 2, "a": 1}) == prereg.canonical_hash(
        {"a": 1, "b": 2}
    )
    with pytest.raises(ValueError, match="compliant"):
        prereg.canonical_json({"bad": float("nan")})
