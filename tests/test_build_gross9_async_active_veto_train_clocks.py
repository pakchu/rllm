from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from training import build_gross9_async_active_veto_train_clocks as builder
from training import preregister_gross9_async_active_veto_search as prereg


def _clock(component: str, rows: list[tuple[str, int]]) -> pd.DataFrame:
    values = []
    for raw, side in rows:
        entry = pd.Timestamp(raw)
        values.append(
            {
                "candidate": component,
                "control": "primary",
                "split": "train",
                "decision_time": entry - pd.Timedelta("10m"),
                "feature_available_time": entry - pd.Timedelta("5m"),
                "entry_time": entry,
                "exit_time": entry + pd.Timedelta("8h"),
                "side": side,
            }
        )
    return pd.DataFrame(values, columns=builder.COMMON_CLOCK_FIELDS)


def _write_actual_prereg_to_temp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, builder_sha: str | None = None) -> dict[str, object]:
    if builder_sha is None:
        builder_sha = builder.sha256_file(builder.__file__)
    monkeypatch.setattr(prereg, "BUILDER_SHA256", builder_sha)
    payload = prereg.build()
    prereg.validate(payload)
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(prereg, "DEFAULT_OUTPUT", prereg_path)
    monkeypatch.setattr(builder.importlib, "import_module", lambda name: prereg)
    return payload


def test_active_veto_window_is_strict_lower_inclusive_upper_and_latest_supersedes() -> None:
    base = _clock("A", [("2023-07-01T06:00:00Z", 1), ("2023-07-01T14:00:00Z", 1)])
    veto = _clock(
        "B",
        [
            ("2023-07-01T00:00:00Z", -1),  # exactly t-6h for first base: excluded
            ("2023-07-01T05:00:00Z", -1),  # would suppress first if latest
            ("2023-07-01T06:00:00Z", 1),  # same-time inclusive and latest: keep
            ("2023-07-01T13:59:00Z", 1),
            ("2023-07-01T14:00:00Z", -1),  # same-time opposite latest: suppress
        ],
    )
    result, diag = builder.build_active_veto_clock("A", "B", base, veto)
    assert result["entry_time"].tolist() == [pd.Timestamp("2023-07-01T06:00:00Z")]
    row = result.iloc[0]
    assert row["veto_entry_time"] == pd.Timestamp("2023-07-01T06:00:00Z")
    assert row["veto_side"] == 1
    assert row["veto_relation"] == "same_side_latest_keep"
    assert diag["opposite_latest_veto_suppressions"] == 1
    assert diag["same_side_latest_veto_keeps"] == 1


def test_no_veto_keeps_and_opposite_suppresses_without_reverse() -> None:
    base = _clock("A", [("2023-07-01T01:00:00Z", -1), ("2023-07-01T10:00:00Z", 1)])
    veto = _clock("B", [("2023-07-01T09:00:00Z", -1)])
    result, diag = builder.build_active_veto_clock("A", "B", base, veto)
    assert result["entry_time"].tolist() == [pd.Timestamp("2023-07-01T01:00:00Z")]
    assert result.iloc[0]["side"] == -1
    assert diag["no_veto_window_keeps"] == 1
    assert diag["opposite_latest_veto_suppressions"] == 1
    assert set(result["side"]) == {-1}


def test_candidate_local_half_open_reservation_after_veto_materialization() -> None:
    base = _clock("A", [("2023-07-01T00:00:00Z", 1), ("2023-07-01T01:00:00Z", 1), ("2023-07-01T08:00:00Z", 1)])
    veto = _clock("B", [("2023-06-30T23:59:00Z", 1)])
    result, diag = builder.build_active_veto_clock("A", "B", base, veto)
    assert result["entry_time"].tolist() == [pd.Timestamp("2023-07-01T00:00:00Z"), pd.Timestamp("2023-07-01T08:00:00Z")]
    assert diag["reservation_dropped_rows"] == 1


def test_support_gates_include_weeks_halves_and_opposite_suppression() -> None:
    entries = pd.to_datetime(
        [
            "2023-07-01T00:00:00Z",
            "2023-07-08T00:00:00Z",
            "2023-07-15T00:00:00Z",
            "2023-07-22T00:00:00Z",
            "2023-08-05T00:00:00Z",
            "2023-10-07T00:00:00Z",
            "2023-10-14T00:00:00Z",
            "2023-11-04T00:00:00Z",
            "2023-11-11T00:00:00Z",
            "2023-12-02T00:00:00Z",
        ],
        utc=True,
    )
    frame = pd.DataFrame({"entry_time": entries, "side": [1, -1] * 5})
    stats = builder.support_stats(frame, opposite_suppressions=1)
    assert stats["events"] == 10
    assert stats["distinct_iso_weeks"] == 10
    assert stats["first_half_events"] == 5
    assert stats["second_half_events"] == 5
    assert all(builder.support_checks(stats).values())
    assert builder.support_checks(builder.support_stats(frame, opposite_suppressions=0))["opposite_suppression_presence"] is False


def test_duplicate_group_all_rejects_base_current_and_prior_nonempty_only() -> None:
    clock = pd.DataFrame({"entry_time": pd.to_datetime(["2023-07-01T00:00:00Z"], utc=True), "exit_time": pd.to_datetime(["2023-07-01T08:00:00Z"], utc=True), "side": [1]})
    empty = pd.DataFrame(columns=["entry_time", "exit_time", "side"])
    candidates = {"c1": clock.copy(), "c2": clock.copy(), "empty": empty}
    bases = {"base": clock.copy()}
    priors = {"prior": {"family": "same_side", "keys": builder._clock_key_set(clock), "signature": builder._schedule_signature(clock)}}
    report = builder.duplicate_gate_report(candidates, bases, priors)
    assert set(report["rejected_candidates"]) == {"c1", "c2"}
    assert any(reason.startswith("current:") for reason in report["candidate_reject_reasons"]["c1"])
    assert any(reason.startswith("base:") for reason in report["candidate_reject_reasons"]["c1"])
    assert any(reason.startswith("prior:") for reason in report["candidate_reject_reasons"]["c1"])
    assert "empty" not in report["candidate_reject_reasons"]


def test_candidate_family_is_72_ordered_pairs() -> None:
    family = builder.candidate_family()
    assert len(builder.COMPONENT_ORDER) == 9
    assert len(family) == 72
    assert len(set(family)) == 72
    assert family[0] == builder.candidate_id(builder.COMPONENT_ORDER[0], builder.COMPONENT_ORDER[1])
    assert family[1] == builder.candidate_id(builder.COMPONENT_ORDER[0], builder.COMPONENT_ORDER[2])


def test_preregistration_hard_fails_when_module_or_artifact_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def missing_module(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(builder.importlib, "import_module", missing_module)
    with pytest.raises(RuntimeError, match="missing preregistration module"):
        builder.load_validated_preregistration()

    monkeypatch.setattr(prereg, "DEFAULT_OUTPUT", tmp_path / "missing.json")
    monkeypatch.setattr(builder.importlib, "import_module", lambda name: prereg)
    with pytest.raises(RuntimeError, match="missing committed preregistration artifact"):
        builder.load_validated_preregistration()


def test_dynamic_preregistration_validation_uses_actual_prereg_build_validate_and_transparency(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_actual_prereg_to_temp(monkeypatch, tmp_path)
    result = builder.load_validated_preregistration()
    assert result["available"] is True
    assert result["status"] == "validated_against_committed_preregistration"
    assert result["prior_source_support_artifacts_cross_checked"] is True
    assert result["research_boundary_disclosure_cross_checked"] is True


def test_preregistration_rejects_placeholder_or_mismatched_builder_hash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_actual_prereg_to_temp(monkeypatch, tmp_path, builder_sha="PENDING_G9ASYNCACTIVEVETO_BUILDER_FOLLOWUP")
    with pytest.raises(RuntimeError, match="builder hash is missing or placeholder"):
        builder.load_validated_preregistration()

    _write_actual_prereg_to_temp(monkeypatch, tmp_path, builder_sha="0" * 64)
    with pytest.raises(RuntimeError, match="builder hash mismatch"):
        builder.load_validated_preregistration()


def test_preregistration_rejects_prior_binding_or_research_boundary_drift(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(prereg, "BUILDER_SHA256", builder.sha256_file(builder.__file__))
    monkeypatch.setattr(prereg, "PRIOR_SOURCE_SUPPORT_ARTIFACTS", [])
    payload = prereg.build()
    prereg_path = tmp_path / "prior_drift.json"
    prereg_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(prereg, "DEFAULT_OUTPUT", prereg_path)
    monkeypatch.setattr(builder.importlib, "import_module", lambda name: prereg)
    with pytest.raises(RuntimeError, match="prior source-support binding drift"):
        builder.load_validated_preregistration()

    monkeypatch.undo()
    _write_actual_prereg_to_temp(monkeypatch, tmp_path)
    payload = prereg.build()
    payload["research_boundary"]["family_operator_gate_threshold_or_order_changed_after_preliminary_source_materialization"] = True
    payload["manifest_hash"] = prereg.canonical_hash({k: v for k, v in payload.items() if k != "manifest_hash"})
    prereg_path = tmp_path / "boundary_drift.json"
    prereg_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(prereg, "DEFAULT_OUTPUT", prereg_path)
    monkeypatch.setattr(prereg, "build", lambda: payload)
    monkeypatch.setattr(builder.importlib, "import_module", lambda name: prereg)
    with pytest.raises(RuntimeError, match="retune boundary drift"):
        builder.load_validated_preregistration()


def test_preregistration_rejects_preliminary_receipt_hash_drift(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_actual_prereg_to_temp(monkeypatch, tmp_path)
    payload = prereg.build()
    payload["preliminary_source_materialization_receipt"]["sha256"] = "0" * 64
    payload["manifest_hash"] = prereg.canonical_hash({k: v for k, v in payload.items() if k != "manifest_hash"})
    prereg_path = tmp_path / "receipt_drift.json"
    prereg_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(prereg, "DEFAULT_OUTPUT", prereg_path)
    monkeypatch.setattr(prereg, "build", lambda: payload)
    monkeypatch.setattr(builder.importlib, "import_module", lambda name: prereg)
    with pytest.raises(RuntimeError, match="preliminary source materialization binding drift"):
        builder.load_validated_preregistration()


def test_run_writes_all_72_without_leakage_and_authenticates_prior(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    component_order = tuple(f"C{i}" for i in range(9))
    clocks = {component: _clock(component, [("2023-07-01T00:00:00Z", 1), ("2023-07-01T12:00:00Z", -1)]) for component in component_order}
    monkeypatch.setattr(builder, "COMPONENT_ORDER", component_order)
    monkeypatch.setattr(builder, "verify_bound_component_artifacts", lambda: {component: {"verified": True} for component in component_order})
    monkeypatch.setattr(builder, "load_train_prefix_clock", lambda component: clocks[component])
    monkeypatch.setattr(builder, "load_validated_preregistration", lambda: {"available": True, "status": "validated_against_committed_preregistration"})
    auth_called = {"value": False}

    def fake_prior_artifacts() -> dict[str, object]:
        auth_called["value"] = True
        return {"same_side": {"members": {}}, "handoff": {"members": {}}, "three_way": {"members": {}}}

    monkeypatch.setattr(builder, "load_validated_prior_source_artifacts", fake_prior_artifacts)
    monkeypatch.setattr(builder, "load_prior_clock_schedules", lambda artifacts: {})
    result = builder.run(tmp_path / "clocks", tmp_path / "result.json", tmp_path / "controls")
    assert auth_called["value"] is True
    assert result["candidate_family_size"] == 72
    assert len(result["candidates"]) == 72
    assert len(list((tmp_path / "clocks").glob("*.csv.gz"))) == 72
    assert len(list((tmp_path / "controls").glob("*.csv.gz"))) == 9
    boundary = result["evidence_boundary"]
    assert boundary["market_rows_opened"] is False
    assert boundary["funding_opened"] is False
    assert boundary["returns_or_pnl_opened"] is False
    assert boundary["economic_outcomes_opened"] is False
    assert boundary["preliminary_source_materialization_commit"] == "1bfddd3c"
    assert boundary["source_incidence_and_support_counts_opened_before_committed_preregistration"] is True
    assert boundary["family_operator_gate_threshold_or_order_changed_after_preliminary_source_materialization"] is False
    assert boundary["preliminary_14_source_passes_used_to_retune"] is False
    research = result["research_boundary"]
    assert research["source_incidence_and_support_counts_opened_before_committed_preregistration"] is True
    assert research["preliminary_14_source_passes_used_to_retune"] is False
    assert result["preliminary_source_materialization_receipt"]["commit"] == "1bfddd3c"
    assert (tmp_path / "result.json").is_file()
    with gzip.open(next((tmp_path / "clocks").glob("*.csv.gz")), "rt", encoding="utf-8") as handle:
        assert "pnl" not in handle.readline().lower()
