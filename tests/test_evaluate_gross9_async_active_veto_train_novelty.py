from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from training import evaluate_gross9_async_active_veto_train_novelty as n
from training import preregister_gross9_async_active_veto_search as prereg


def _df(rows: list[tuple[str, str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": split,
            "entry_time": pd.Timestamp(entry),
            "exit_time": pd.Timestamp(exit_time),
            "side": side,
        }
        for split, entry, exit_time, side in rows
    )


def _write_clock(path: Path, rows: list[tuple[str, str, str, int]]) -> None:
    frame = pd.DataFrame(rows, columns=["split", "entry_time", "exit_time", "side"])
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)


def _write_manifested_json(path: Path, payload: dict[str, object]) -> str:
    core = copy.deepcopy(payload)
    core.pop("manifest_hash", None)
    payload["manifest_hash"] = n.canonical_hash(core)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return n.sha256_file(path)


def _support_with_passed(passed: list[str]) -> dict[str, object]:
    candidates: dict[str, object] = {}
    duplicate_rejected: list[str] = []
    for idx, candidate in enumerate(prereg.CANDIDATE_FAMILY):
        source_pass = candidate in passed
        duplicate_fail = idx == 0 and not source_pass
        if duplicate_fail:
            duplicate_rejected.append(candidate)
        candidates[candidate] = {
            "components": {"base": "A", "veto": "B"},
            "operator": "ordered_async_active_opposite_veto_latest_veto_in_strict_lower_inclusive_upper_6h",
            "clock": {"path": f"{candidate}.csv.gz", "sha256": "a" * 64, "rows": 12 if source_pass else 0},
            "construction_diagnostics": {
                "base_events_seen": 13 if source_pass else 0,
                "no_veto_window_keeps": 10 if source_pass else 0,
                "same_side_latest_veto_keeps": 1 if source_pass else 0,
                "opposite_latest_veto_suppressions": 1 if source_pass else 0,
                "duplicate_rows_dropped": 0,
                "pre_reservation_rows": 12 if source_pass else 0,
                "post_reservation_rows": 12 if source_pass else 0,
                "reservation_dropped_rows": 0,
            },
            "support": {
                "events": 12 if source_pass else 0,
                "longs": 6 if source_pass else 0,
                "shorts": 6 if source_pass else 0,
                "minority_side_share": 0.5 if source_pass else 0.0,
                "max_month_share": 0.25 if source_pass else 0.0,
                "distinct_iso_weeks": 10 if source_pass else 0,
            },
            "support_checks": {
                "minimum_events": source_pass,
                "side_balance": source_pass,
                "month_concentration": source_pass,
                "distinct_iso_weeks": source_pass,
                "both_train_halves": source_pass,
                "opposite_suppression_presence": source_pass,
            },
            "duplicate_gate": {"rejected": duplicate_fail, "reasons": ["current:other"] if duplicate_fail else []},
            "support_passed": source_pass,
            "advance_to_gross9_novelty": source_pass,
            "advance_to_economic_outcomes": False,
            "decision": "pass_to_gross9_novelty" if source_pass else "terminal_source_support_reject",
        }
    return {
        "protocol_version": n.source_support.PROTOCOL_VERSION,
        "policy_id": n.POLICY_ID,
        "preregistration": {
            "sha256": n.PREREGISTRATION_SHA256,
            "manifest_hash": n.PREREGISTRATION_MANIFEST_HASH,
            "status": "validated_against_committed_preregistration",
            "prior_source_support_artifacts_cross_checked": True,
            "research_boundary_disclosure_cross_checked": True,
        },
        "candidate_family": list(prereg.CANDIDATE_FAMILY),
        "candidate_family_size": prereg.FAMILY_SIZE,
        "decision": "pass_supported_active_veto_candidates_to_gross9_novelty",
        "passed_candidates": passed,
        "preliminary_source_materialization_receipt": {
            "commit": prereg.PRELIMINARY_SOURCE_MATERIALIZATION_RECEIPT["commit"],
            "path": prereg.PRELIMINARY_SOURCE_MATERIALIZATION_RECEIPT["path"],
            "sha256": prereg.PRELIMINARY_SOURCE_MATERIALIZATION_RECEIPT["sha256"],
            "manifest_hash": prereg.PRELIMINARY_SOURCE_MATERIALIZATION_RECEIPT["manifest_hash"],
            "builder": copy.deepcopy(prereg.PRELIMINARY_SOURCE_MATERIALIZATION_RECEIPT["builder"]),
            "placeholder_preregistration_sha256": prereg.PRELIMINARY_SOURCE_MATERIALIZATION_RECEIPT["preregistration_artifact_with_placeholder_builder_binding"]["sha256"],
            "placeholder_preregistration_manifest_hash": prereg.PRELIMINARY_SOURCE_MATERIALIZATION_RECEIPT["preregistration_artifact_with_placeholder_builder_binding"]["manifest_hash"],
            "placeholder_builder_value": prereg.PRELIMINARY_SOURCE_MATERIALIZATION_RECEIPT["placeholder_builder_value"],
            "passed_candidates": prereg.PRELIMINARY_SOURCE_MATERIALIZATION_RECEIPT["support_count_disclosure"]["passed_candidates"],
        },
        "duplicate_gate_summary": {"rejected_candidates": duplicate_rejected},
        "candidates": candidates,
        "evidence_boundary": {
            "component_clock_rows_materialized_train_prefix_only": True,
            "gross9_rows_opened": False,
            "market_rows_opened": False,
            "entry_exit_prices_opened": False,
            "funding_opened": False,
            "returns_or_pnl_opened": False,
            "economic_outcomes_opened": False,
            "base_control_economic_outcomes_opened": False,
            "oos_component_rows_materialized": 0,
        },
    }


def test_constants_bind_active_veto_prereg_source_support_and_gross9_hashes() -> None:
    assert n.POLICY_ID == "G9ASYNCACTIVEVETO-8"
    assert n.PREREGISTRATION_SHA256 == "5bb0abae46a5716451b07268a268cdd112a78829786772c4aeec8bc43f383f25"
    assert n.PREREGISTRATION_MANIFEST_HASH == "871c7fb8c8825cb30c0967cab46a2a8cc7342f46f37c673372b45d2501d6aa6e"
    assert n.SOURCE_SUPPORT_SHA256 == "ee966e59e219886b561a23e605cf225f44d393128f210a360048addfeba42f20"
    assert n.SOURCE_SUPPORT_MANIFEST_HASH == "ec32caa65a0945fc73b6d863cb1b3fa810f4c58ffd8aed68408fff949e4d6f32"
    assert n.GROSS9_MANIFEST_SHA256 == "5433812da786a959cda1cfcf4825bc2e4a228ea8152a4b8cce1e867f29adf073"
    assert len(prereg.CANDIDATE_FAMILY) == 72
    assert n.LIMITS == n.pair_novelty.LIMITS


def test_train_split_and_candidate_clock_window_are_reused_strictly(tmp_path: Path) -> None:
    path = tmp_path / "candidate.csv.gz"
    _write_clock(
        path,
        [
            ("train", "2023-07-01T00:00:00Z", "2023-07-01T08:00:00Z", 1),
            ("test", "2023-07-02T00:00:00Z", "2023-07-02T08:00:00Z", -1),
        ],
    )
    with pytest.raises(RuntimeError, match="row/window drift"):
        n.load_candidate_clock({"path": str(path), "sha256": n.sha256_file(path), "rows": 2}, "C")

    train_only_path = tmp_path / "candidate_train_only.csv.gz"
    _write_clock(train_only_path, [("train", "2023-07-01T00:00:00Z", "2023-07-01T08:00:00Z", 1)])
    common = n.load_candidate_clock({"path": str(train_only_path), "sha256": n.sha256_file(train_only_path), "rows": 1}, "C")
    assert len(common) == 1
    assert common.iloc[0]["split"] == "train"


def test_assert_source_boundary_rejects_opened_outcomes() -> None:
    boundary = {
        "component_clock_rows_materialized_train_prefix_only": True,
        "gross9_rows_opened": False,
        "market_rows_opened": False,
        "entry_exit_prices_opened": False,
        "funding_opened": False,
        "returns_or_pnl_opened": True,
        "economic_outcomes_opened": False,
        "base_control_economic_outcomes_opened": False,
        "oos_component_rows_materialized": 0,
    }
    with pytest.raises(RuntimeError, match="returns_or_pnl_opened"):
        n._assert_source_boundary_closed(boundary)


def test_load_frozen_controls_rejects_duplicate_gate_projection_drift(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    prereg_path = tmp_path / "prereg.json"
    registration = prereg.build()
    prereg_sha = _write_manifested_json(prereg_path, registration)
    monkeypatch.setattr(n, "PREREGISTRATION", prereg_path)
    monkeypatch.setattr(n, "PREREGISTRATION_SHA256", prereg_sha)
    monkeypatch.setattr(n, "PREREGISTRATION_MANIFEST_HASH", registration["manifest_hash"])

    support_path = tmp_path / "support.json"
    passed = list(prereg.CANDIDATE_FAMILY[: n.EXPECTED_SUPPORTED_COUNT])
    support = _support_with_passed(passed)
    support["preregistration"]["sha256"] = prereg_sha  # type: ignore[index]
    support["preregistration"]["manifest_hash"] = registration["manifest_hash"]  # type: ignore[index]
    support["candidates"][passed[0]]["duplicate_gate"] = {"rejected": True, "reasons": ["base:A"]}  # type: ignore[index]
    support_sha = _write_manifested_json(support_path, support)
    monkeypatch.setattr(n, "SOURCE_SUPPORT", support_path)
    monkeypatch.setattr(n, "SOURCE_SUPPORT_SHA256", support_sha)
    monkeypatch.setattr(n, "SOURCE_SUPPORT_MANIFEST_HASH", support["manifest_hash"])

    with pytest.raises(RuntimeError, match="duplicate-gate projection drift"):
        n.load_frozen_controls()


def test_load_frozen_controls_rejects_wrong_supported_count(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    prereg_path = tmp_path / "prereg.json"
    registration = prereg.build()
    prereg_sha = _write_manifested_json(prereg_path, registration)
    monkeypatch.setattr(n, "PREREGISTRATION", prereg_path)
    monkeypatch.setattr(n, "PREREGISTRATION_SHA256", prereg_sha)
    monkeypatch.setattr(n, "PREREGISTRATION_MANIFEST_HASH", registration["manifest_hash"])

    support_path = tmp_path / "support.json"
    passed = list(prereg.CANDIDATE_FAMILY[: n.EXPECTED_SUPPORTED_COUNT - 1])
    support = _support_with_passed(passed)
    support["preregistration"]["sha256"] = prereg_sha  # type: ignore[index]
    support["preregistration"]["manifest_hash"] = registration["manifest_hash"]  # type: ignore[index]
    support_sha = _write_manifested_json(support_path, support)
    monkeypatch.setattr(n, "SOURCE_SUPPORT", support_path)
    monkeypatch.setattr(n, "SOURCE_SUPPORT_SHA256", support_sha)
    monkeypatch.setattr(n, "SOURCE_SUPPORT_MANIFEST_HASH", support["manifest_hash"])

    with pytest.raises(RuntimeError, match="source-supported roster drift"):
        n.load_frozen_controls()


def test_run_counts_all_72_but_evaluates_only_source_and_duplicate_supported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    passed = list(prereg.CANDIDATE_FAMILY[: n.EXPECTED_SUPPORTED_COUNT])
    support = _support_with_passed(passed)
    support["manifest_hash"] = "supporthash"
    manifest = {
        "manifest_hash": "grosshash",
        "clocks": {
            sleeve: {"rows": 1, "path": f"{sleeve}.csv.gz", "sha256": "b" * 64, "counts": {"train": 1}}
            for sleeve in n.gross9.EXPECTED_WEIGHTS
        },
    }
    monkeypatch.setattr(n, "load_frozen_controls", lambda: ({"manifest_hash": "prehash"}, support, manifest))
    monkeypatch.setattr(
        n,
        "load_gross9_train_clock",
        lambda sleeve, record: (_df([("train", "2023-07-20T00:00:00Z", "2023-07-20T08:00:00Z", 1)]), 1, 1),
    )
    calls: list[str] = []

    def fake_eval(candidate: str, clock_record: dict[str, object], gross9_clocks: dict[str, pd.DataFrame]) -> dict[str, object]:
        calls.append(candidate)
        passed_novelty = candidate == passed[-1]
        return {
            "candidate_clock_rows_opened": int(clock_record["rows"]),
            "gross9_sleeves": {},
            "gross9_pass": passed_novelty,
            "gross9_novelty_status": "passed" if passed_novelty else "failed",
            "advance_to_train_economics": passed_novelty,
            "decision": "pass_to_train_economics" if passed_novelty else "terminal_gross9_novelty_reject",
        }

    monkeypatch.setattr(n, "evaluate_candidate", fake_eval)

    result = n.run(tmp_path / "novelty.json")

    assert list(result["candidates"]) == list(prereg.CANDIDATE_FAMILY)
    assert len(result["candidates"]) == 72
    assert calls == passed
    assert result["gross9_novelty_evaluated_candidate_count"] == n.EXPECTED_SUPPORTED_COUNT
    assert result["gross9_novelty_passed_candidates"] == [passed[-1]]
    assert result["advance_to_economic_outcomes"] is True
    assert result["candidates"][prereg.CANDIDATE_FAMILY[-1]]["gross9_novelty_status"] == "not_evaluated_source_or_exact_duplicate_failed"
    boundary = result["evidence_boundary"]
    assert boundary["candidate_family_rows_counted"] == 72
    assert boundary["source_supported_candidate_clock_rows_opened"] == n.EXPECTED_SUPPORTED_COUNT * 12
    assert boundary["source_supported_candidate_clock_rows_evaluated"] == n.EXPECTED_SUPPORTED_COUNT * 12
    assert boundary["unsupported_candidate_clock_rows_opened_for_novelty"] == 0
    assert boundary["exact_duplicate_gate_projected_for_all_72"] is True
    assert boundary["btc_price_or_return_rows_opened"] == 0
    assert boundary["funding_rows_opened"] == 0
    assert boundary["economic_outcome_rows_opened"] == 0
    written = json.loads((tmp_path / "novelty.json").read_text())
    core = dict(written)
    manifest_hash = core.pop("manifest_hash")
    assert manifest_hash == n.canonical_hash(core)
