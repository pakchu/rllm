from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from training import build_gross9_async_three_way_consensus_train_clocks as builder


def _clock(component: str, rows: list[tuple[str, int, str | None, str | None]]) -> pd.DataFrame:
    values = []
    for entry_raw, side, decision_raw, available_raw in rows:
        entry = pd.Timestamp(entry_raw)
        values.append(
            {
                "candidate": component,
                "control": "primary",
                "split": "train",
                "decision_time": pd.Timestamp(decision_raw) if decision_raw else entry - pd.Timedelta("5m"),
                "feature_available_time": pd.Timestamp(available_raw) if available_raw else entry - pd.Timedelta("3m"),
                "entry_time": entry,
                "exit_time": entry + pd.Timedelta("8h"),
                "side": side,
            }
        )
    return pd.DataFrame(values, columns=builder.COMMON_CLOCK_FIELDS)


def _write_clock(path: Path, component: str, entries: list[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=builder.COMMON_CLOCK_FIELDS)
        writer.writeheader()
        for entry_raw, side in entries:
            entry = pd.Timestamp(entry_raw)
            writer.writerow(
                {
                    "candidate": component,
                    "control": "primary",
                    "split": "train" if entry < builder.TRAIN_END else "test",
                    "decision_time": str(entry - pd.Timedelta("5m")),
                    "feature_available_time": str(entry - pd.Timedelta("3m")),
                    "entry_time": str(entry),
                    "exit_time": str(entry + pd.Timedelta("8h")),
                    "side": side,
                }
            )


def _write_prior(path: Path, rows: list[tuple[str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["entry_time", "exit_time", "side"])
        writer.writeheader()
        for entry, exit_time, side in rows:
            writer.writerow({"entry_time": entry, "exit_time": exit_time, "side": side})


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_candidate_family_is_exact_84_triples_from_nine_components() -> None:
    family = builder.candidate_family()
    assert len(builder.COMPONENT_ORDER) == 9
    assert len(family) == 84
    assert len(set(family)) == 84
    assert family[0] == builder.triple_id(builder.COMPONENT_ORDER[0], builder.COMPONENT_ORDER[1], builder.COMPONENT_ORDER[2])
    assert "__ASYNC_SAME_SIDE_3WAY_6H__" in family[0]


def test_three_way_uses_latest_each_component_inclusive_window_and_canonical_trigger() -> None:
    left = _clock("A", [("2023-07-01T00:00:00Z", 1, "2023-06-30T23:50:00Z", "2023-06-30T23:55:00Z")])
    middle = _clock("B", [("2023-07-01T05:59:00Z", 1, "2023-07-01T05:40:00Z", "2023-07-01T05:45:00Z")])
    right = _clock("C", [("2023-07-01T06:00:00Z", 1, "2023-07-01T05:30:00Z", "2023-07-01T05:35:00Z")])
    result, diag = builder.build_async_three_way_consensus_clock("A", "B", "C", left, middle, right)
    assert result["entry_time"].tolist() == [pd.Timestamp("2023-07-01T06:00:00Z")]
    row = result.iloc[0]
    assert row["side"] == 1
    assert row["trigger_component_id"] == "C"
    assert row["left_selected_entry_time"] == pd.Timestamp("2023-07-01T00:00:00Z")
    assert row["middle_selected_entry_time"] == pd.Timestamp("2023-07-01T05:59:00Z")
    assert row["decision_time"] == pd.Timestamp("2023-07-01T05:40:00Z")
    assert row["feature_available_time"] == pd.Timestamp("2023-07-01T05:45:00Z")
    assert diag["post_reservation_rows"] == 1


def test_lower_boundary_at_exact_six_hours_is_inclusive() -> None:
    left = _clock("A", [("2023-07-01T00:00:00Z", 1, None, None)])
    middle = _clock("B", [("2023-07-01T00:01:00Z", 1, None, None)])
    right = _clock("C", [("2023-07-01T06:00:00Z", 1, None, None)])
    result, _ = builder.build_async_three_way_consensus_clock("A", "B", "C", left, middle, right)
    assert result["entry_time"].tolist() == [pd.Timestamp("2023-07-01T06:00:00Z")]


def test_two_or_three_simultaneous_triggers_allowed_with_earliest_component_order() -> None:
    left = _clock("A", [("2023-07-01T00:00:00Z", 1, None, None)])
    middle = _clock("B", [("2023-07-01T00:00:00Z", 1, None, None)])
    right = _clock("C", [("2023-07-01T00:00:00Z", 1, None, None)])
    result, _ = builder.build_async_three_way_consensus_clock("A", "B", "C", left, middle, right)
    assert len(result) == 1
    assert result.loc[0, "trigger_component_id"] == "A"


def test_if_long_and_short_both_qualify_same_timestamp_drop_both() -> None:
    left = _clock("A", [("2023-07-01T00:00:00Z", 1, None, None), ("2023-07-01T05:00:00Z", 1, None, None), ("2023-07-01T05:00:00Z", -1, None, None)])
    middle = _clock("B", [("2023-07-01T00:00:00Z", 1, None, None), ("2023-07-01T05:00:00Z", -1, None, None)])
    right = _clock("C", [("2023-07-01T00:00:00Z", 1, None, None), ("2023-07-01T05:00:00Z", -1, None, None)])
    result, diag = builder.build_async_three_way_consensus_clock("A", "B", "C", left, middle, right)
    assert pd.Timestamp("2023-07-01T05:00:00Z") not in set(result["entry_time"])
    assert diag["ambiguous_same_timestamp_both_sides_rows_dropped"] == 2


def test_half_open_reservation_drops_overlaps_and_allows_abutment() -> None:
    left = _clock("A", [("2023-07-01T00:00:00Z", 1, None, None), ("2023-07-01T09:00:00Z", -1, None, None)])
    middle = _clock("B", [("2023-07-01T01:00:00Z", 1, None, None), ("2023-07-01T02:00:00Z", 1, None, None), ("2023-07-01T09:00:00Z", -1, None, None)])
    right = _clock("C", [("2023-07-01T01:00:00Z", 1, None, None), ("2023-07-01T02:00:00Z", 1, None, None), ("2023-07-01T09:00:00Z", -1, None, None)])
    result, diag = builder.build_async_three_way_consensus_clock("A", "B", "C", left, middle, right)
    assert result["entry_time"].tolist() == [pd.Timestamp("2023-07-01T01:00:00Z"), pd.Timestamp("2023-07-01T09:00:00Z")]
    assert diag["reservation_dropped_rows"] == 1


def test_support_stats_and_gates_require_weeks_and_both_halves() -> None:
    entries = pd.to_datetime(
        [
            "2023-07-03T00:00:00Z", "2023-07-10T00:00:00Z", "2023-07-17T00:00:00Z", "2023-07-24T00:00:00Z", "2023-08-07T00:00:00Z",
            "2023-10-02T00:00:00Z", "2023-10-09T00:00:00Z", "2023-10-16T00:00:00Z", "2023-10-23T00:00:00Z", "2023-11-06T00:00:00Z",
        ],
        utc=True,
    )
    frame = pd.DataFrame({"entry_time": entries, "side": [1, -1] * 5})
    stats = builder.support_stats(frame)
    assert stats["events"] == 10
    assert stats["minority_side_share"] == 0.5
    assert stats["distinct_iso_weeks"] == 10
    assert builder.support_checks(stats) == {
        "minimum_events": True,
        "side_balance": True,
        "month_concentration": True,
        "distinct_iso_weeks": True,
        "both_train_halves": True,
    }


def test_load_train_prefix_clock_reuses_hash_validated_train_only_boundary(tmp_path: Path) -> None:
    path = tmp_path / "A.csv.gz"
    _write_clock(path, "A", [("2023-07-01T00:00:00Z", 1), ("2024-01-01T00:00:00Z", -1)])
    frame = builder.load_train_prefix_clock("A", {"A": {"clock": {"path": str(path), "sha256": _sha(path)}}})
    assert frame["entry_time"].tolist() == [pd.Timestamp("2023-07-01T00:00:00Z")]
    assert frame.attrs["stopped_at_train_end"] is True


def _unit_prior_artifacts(components: tuple[str, ...], same_dir: Path, handoff_dir: Path) -> dict[str, object]:
    pairs: dict[str, dict[str, object]] = {"same_side": {}, "handoff": {}}
    for i, left in enumerate(components):
        for right in components[i + 1 :]:
            same_id = builder.same_side.pair_id(left, right)
            handoff_id = builder.handoff.pair_id(left, right)
            for family, candidate, directory in (("same_side", same_id, same_dir), ("handoff", handoff_id, handoff_dir)):
                path = directory / f"{candidate}.csv.gz"
                if not path.exists():
                    _write_prior(path, [])
                pairs[family][candidate] = {"clock": {"path": str(path), "sha256": _sha(path), "rows": 0 if path.stat().st_size and not pd.read_csv(path).shape[0] else len(pd.read_csv(path, usecols=["entry_time"]))}}
    return {family: {"pairs": value} for family, value in pairs.items()}


def test_prior_disclosure_reports_constituent_unions_handoff_overlap_and_duplicate(tmp_path: Path) -> None:
    same_dir = tmp_path / "same"
    handoff_dir = tmp_path / "handoff"
    components = ("A", "B", "C")
    same_id = builder.same_side.pair_id("A", "B")
    handoff_id = builder.handoff.pair_id("A", "B")
    rows = [("2023-07-01 00:00:00+00:00", "2023-07-01 08:00:00+00:00", 1)]
    _write_prior(same_dir / f"{same_id}.csv.gz", rows)
    _write_prior(handoff_dir / f"{handoff_id}.csv.gz", rows)
    prior = builder.load_prior_clock_schedules(components, same_dir, handoff_dir, _unit_prior_artifacts(components, same_dir, handoff_dir))
    candidate_clock = pd.DataFrame({"entry_time": pd.to_datetime(["2023-07-01T00:00:00Z"], utc=True), "exit_time": pd.to_datetime(["2023-07-01T08:00:00Z"], utc=True), "side": [1]})
    disclosure = builder.prior_clock_disclosure(candidate_clock, components, prior)
    assert disclosure["constituent_pair_intersections"][same_id]["exact_intersection"] == 1
    assert disclosure["prior_family_unions"]["same_side"]["exact_intersection"] == 1
    assert disclosure["incidental_handoff_exact_overlap"] == 1
    assert set(disclosure["exact_full_schedule_duplicates"]) == {same_id, handoff_id}
    assert disclosure["exact_duplicate_reject"] is True


def test_load_prior_clock_schedules_default_path_loads_validated_preregistration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration = {"prior_clock_source_support_artifacts": []}
    authenticated = {"same_side": {"pairs": {}}, "handoff": {"pairs": {}}}
    monkeypatch.setattr(
        builder,
        "load_validated_preregistration",
        lambda: (types.SimpleNamespace(), registration),
    )
    monkeypatch.setattr(
        builder,
        "load_validated_prior_source_artifacts",
        lambda value: authenticated if value is registration else pytest.fail("wrong registration"),
    )

    assert builder.load_prior_clock_schedules((), prior_artifacts=None) == {}


def test_load_validated_preregistration_is_dynamic_and_checks_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "training.preregister_gross9_async_three_way_consensus_search"
    output = tmp_path / "prereg.json"
    registration = {
        "policy_id": builder.POLICY_ID,
        "component_order": list(builder.COMPONENT_ORDER),
        "candidate_family": list(builder.candidate_family()),
        "candidate_family_size": 84,
        "manifest_hash": "unit",
    }
    output.write_text(json.dumps(registration) + "\n")
    module = types.SimpleNamespace(
        DEFAULT_OUTPUT=output,
        validate=lambda value: None,
        build=lambda: registration,
    )
    monkeypatch.setitem(sys.modules, module_name, module)
    _, loaded = builder.load_validated_preregistration()
    assert loaded == registration


def test_run_materializes_all_84_with_no_outcome_leakage_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    components = tuple(f"C{i}-8" for i in range(9))
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text(json.dumps({"manifest_hash": "unit"}) + "\n")
    monkeypatch.setattr(builder, "COMPONENT_ORDER", components)
    monkeypatch.setattr(
        builder,
        "load_validated_preregistration",
        lambda: (types.SimpleNamespace(DEFAULT_OUTPUT=prereg_path), {"manifest_hash": "unit"}),
    )
    monkeypatch.setattr(builder, "verify_bound_component_artifacts", lambda: {component: {"verified": True} for component in components})
    clocks = {
        component: _clock(component, [("2023-07-01T00:00:00Z", 1, None, None)])
        for component in components
    }
    monkeypatch.setattr(builder, "load_train_prefix_clock", lambda component: clocks[component])
    monkeypatch.setattr(builder, "load_validated_prior_source_artifacts", lambda registration: {"same_side": {"path": "same.json", "sha256": "s", "manifest_hash": "m", "policy_id": "G9ASYNCPAIR-8", "candidate_family_size": 36, "binding": {"policy_id": "G9ASYNCPAIR-8"}, "pairs": {}}, "handoff": {"path": "handoff.json", "sha256": "h", "manifest_hash": "m", "policy_id": "G9ASYNCHANDOFF-8", "candidate_family_size": 36, "binding": {"policy_id": "G9ASYNCHANDOFF-8"}, "pairs": {}}})
    def fake_prior(component_order, same_side_clock_dir, handoff_clock_dir, prior_artifacts=None):
        return {f"prior_{i}": {"family": "same_side" if i < 36 else "handoff", "components": list(components[:2]), "path": "", "sha256": "", "rows": 0, "keys": set(), "signature": tuple()} for i in range(72)}
    monkeypatch.setattr(builder, "load_prior_clock_schedules", fake_prior)
    result = builder.run(tmp_path / "out_clocks", tmp_path / "result.json", tmp_path / "same", tmp_path / "handoff")
    assert result["candidate_family_size"] == 84
    assert len(result["triples"]) == 84
    assert len(list((tmp_path / "out_clocks").glob("*.csv.gz"))) == 84
    assert result["prior_clock_schedules_opened"]["total_prior_schedules"] == 72
    boundary = result["evidence_boundary"]
    assert boundary["market_rows_opened"] is False
    assert boundary["funding_opened"] is False
    assert boundary["combination_returns_or_pnl_opened"] is False
    assert boundary["combination_economic_outcomes_opened"] is False
    assert boundary["prior_clock_schedules_opened_for_disclosure_only"] is True


def test_trigger_component_ids_records_exact_simultaneous_trigger_set() -> None:
    left = _clock("A", [("2023-07-01T00:00:00Z", 1, None, None)])
    middle = _clock("B", [("2023-07-01T00:00:00Z", 1, None, None)])
    right = _clock("C", [("2023-07-01T00:00:00Z", 1, None, None)])
    result, _ = builder.build_async_three_way_consensus_clock("A", "B", "C", left, middle, right)
    assert result.loc[0, "trigger_component_id"] == "A"
    assert result.loc[0, "trigger_component_ids"] == "A|B|C"


def test_pre_reservation_disclosure_uses_pre_keys_not_post_reserved_clock() -> None:
    left = _clock("A", [("2023-07-01T00:00:00Z", 1, None, None)])
    middle = _clock("B", [("2023-07-01T01:00:00Z", 1, None, None), ("2023-07-01T02:00:00Z", 1, None, None)])
    right = _clock("C", [("2023-07-01T01:00:00Z", 1, None, None), ("2023-07-01T02:00:00Z", 1, None, None)])
    clocks = {"A": left, "B": middle, "C": right}
    pre, _ = builder.build_async_three_way_consensus_pre_clock("A", "B", "C", left, middle, right)
    post = builder.reserve_half_open(pre)
    disclosure = builder.triple_pre_reservation_key_disclosure(pre, ("A", "B", "C"), clocks)
    assert len(pre) == 2
    assert len(post) == 1
    assert disclosure["triple_pre_reservation_keys"] == 2
    assert disclosure["constituent_same_side_pair_pre_reservation_union"]["triple_key_coverage_share"] == 1.0


def test_load_prior_clock_schedules_errors_on_missing_or_sha_drift(tmp_path: Path) -> None:
    components = ("A", "B")
    same_dir = tmp_path / "same"
    handoff_dir = tmp_path / "handoff"
    same_id = builder.same_side.pair_id("A", "B")
    handoff_id = builder.handoff.pair_id("A", "B")
    _write_prior(same_dir / f"{same_id}.csv.gz", [])
    _write_prior(handoff_dir / f"{handoff_id}.csv.gz", [])
    artifacts = _unit_prior_artifacts(components, same_dir, handoff_dir)
    artifacts["same_side"]["pairs"][same_id]["clock"]["sha256"] = "bad"
    with pytest.raises(RuntimeError, match="SHA drift"):
        builder.load_prior_clock_schedules(components, same_dir, handoff_dir, artifacts)
    artifacts = _unit_prior_artifacts(components, same_dir, handoff_dir)
    (handoff_dir / f"{handoff_id}.csv.gz").unlink()
    with pytest.raises(RuntimeError, match="missing prior clock schedule"):
        builder.load_prior_clock_schedules(components, same_dir, handoff_dir, artifacts)


def test_load_prior_clock_requires_exit_inside_train(tmp_path: Path) -> None:
    path = tmp_path / "prior.csv.gz"
    _write_prior(path, [("2023-12-31 20:00:00+00:00", "2024-01-01 04:00:00+00:00", 1)])
    with pytest.raises(RuntimeError, match="escaped train containment"):
        builder._load_prior_clock(path, expected_sha256=_sha(path), expected_rows=1)


def _prior_registration(same_binding: dict[str, object], handoff_binding: dict[str, object] | None = None) -> dict[str, object]:
    if handoff_binding is None:
        handoff_binding = {
            "policy_id": builder.handoff.POLICY_ID,
            "path": str(builder.handoff.RESULT),
            "sha256": builder.sha256_file(builder.handoff.RESULT),
            "manifest_hash": json.loads(Path(builder.handoff.RESULT).read_text())["manifest_hash"],
        }
    return {"prior_clock_source_support_artifacts": [same_binding, handoff_binding]}


def test_load_validated_prior_source_artifacts_requires_prereg_bound_path_sha_manifest() -> None:
    same_binding = {
        "policy_id": builder.same_side.POLICY_ID,
        "path": str(builder.same_side.RESULT),
        "sha256": builder.sha256_file(builder.same_side.RESULT),
        "manifest_hash": json.loads(Path(builder.same_side.RESULT).read_text())["manifest_hash"],
        "schedule_scope": "unit",
    }
    loaded = builder.load_validated_prior_source_artifacts(_prior_registration(same_binding))
    assert loaded["same_side"]["binding"] == same_binding
    assert loaded["same_side"]["path"] == same_binding["path"]
    assert loaded["same_side"]["sha256"] == same_binding["sha256"]
    assert loaded["same_side"]["manifest_hash"] == same_binding["manifest_hash"]


def test_load_validated_prior_source_artifacts_errors_on_missing_or_binding_drift(tmp_path: Path) -> None:
    same_binding = {
        "policy_id": builder.same_side.POLICY_ID,
        "path": str(tmp_path / "missing.json"),
        "sha256": "missing",
        "manifest_hash": "missing",
    }
    with pytest.raises(RuntimeError, match="missing prior source-support artifact"):
        builder.load_validated_prior_source_artifacts(_prior_registration(same_binding))

    same_binding = {
        "policy_id": builder.same_side.POLICY_ID,
        "path": str(builder.same_side.RESULT),
        "sha256": "bad",
        "manifest_hash": json.loads(Path(builder.same_side.RESULT).read_text())["manifest_hash"],
    }
    with pytest.raises(RuntimeError, match="artifact SHA drift"):
        builder.load_validated_prior_source_artifacts(_prior_registration(same_binding))

    same_binding = {
        "policy_id": builder.same_side.POLICY_ID,
        "path": str(builder.same_side.RESULT),
        "sha256": builder.sha256_file(builder.same_side.RESULT),
        "manifest_hash": "bad",
    }
    with pytest.raises(RuntimeError, match="manifest binding drift"):
        builder.load_validated_prior_source_artifacts(_prior_registration(same_binding))
