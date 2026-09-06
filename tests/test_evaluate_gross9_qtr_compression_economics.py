from __future__ import annotations

import copy
import gzip
import json
import types
from pathlib import Path

import pytest

from training import evaluate_gross9_qtr_compression_economics as qtr


def _write_json_with_manifest(path: Path, payload: dict) -> dict:
    payload = copy.deepcopy(payload)
    payload["manifest_hash"] = qtr.canonical_hash(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _write_gzip_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        if not rows:
            handle.write("x\n")
            return
        columns = list(rows[0])
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(",".join(str(row[column]) for column in columns) + "\n")


def _valid_prereg(
    tmp_path: Path,
    source_preregistration: dict,
    source_clock_package: dict,
    terminal_novelty: dict,
    near_failures=None,
) -> dict:
    near_failures = near_failures or {"cand_rex_veto_7": 0.44, "markov_transition_long": 0.42}
    return _write_json_with_manifest(tmp_path / "prereg.json", {
        "protocol_version": qtr.prereg.PROTOCOL_VERSION,
        "policy_id": qtr.POLICY_ID,
        "source_policy_id": qtr.SOURCE_POLICY_ID,
        "hypothesis_count": 1,
        "source_clock_reuse": {
            "source_policy_id": qtr.SOURCE_POLICY_ID,
            "identical_sleeves_weights_and_rule": True,
            "source_preregistration": source_preregistration,
            "source_clock_package": source_clock_package,
            "component_sleeves": list(qtr.prereg.distill.DISTILLED_SLEEVES),
            "sleeve_weights": qtr.prereg.distill.SLEEVE_WEIGHTS,
            "gross_exposure_sum": 0.5,
        },
        "terminal_additive_novelty_binding": {
            **terminal_novelty,
            "terminal_additive_decision": "terminal_gross9_novelty_reject",
            "terminal_additive_gross9_pass": False,
            "near_6h_overlap_is_disclosure_not_authorization_gate": True,
            "all_exact_entry_occupied_and_abs_pearson_passed": True,
            "near_6h_failures": {"cand_rex_veto_7": 0.44, "markov_transition_long": 0.44},
            "additive_gross9_alpha_authorized": False,
            "standalone_replacement_compression_authorized_to_test": True,
        },
        "preliminary_train_diagnostic_binding": {
            "legacy_active_veto_familywise_p_non_authorizing": True,
            "preliminary_train_receipt": qtr.prereg.distill.PRELIMINARY_SEQUENCING_RECEIPT,
        },
        "oos_gate_rule": {
            "sequence": ["test2024", "eval2025", "final2026"],
            "repair_authorized_after_failure": False,
            "source_min_nonzero_signed_episodes": {"test": 12, "eval": 12, "final": 8},
        },
        "evidence_boundary": {"oos_outcomes_opened_by_this_preregistration": False},
    })


def _valid_artifacts(monkeypatch, tmp_path: Path):
    source_prereg_payload = _write_json_with_manifest(
        tmp_path / "source_prereg.json", {"policy_id": qtr.SOURCE_POLICY_ID}
    )
    source_preregistration = {
        "path": str(tmp_path / "source_prereg.json"),
        "sha256": qtr.sha256_file(tmp_path / "source_prereg.json"),
        "manifest_hash": source_prereg_payload["manifest_hash"],
    }

    builder = tmp_path / "builder.py"; builder.write_text("# builder\n", encoding="utf-8")
    sleeves = {}
    for base in qtr.prereg.distill.DISTILLED_BASES:
        clock = tmp_path / f"{base}.csv.gz"
        _write_gzip_csv(clock, [{"split": "train", "entry_time": "2023-07-01T00:00:00Z", "exit_time": "2023-07-01T08:00:00Z", "side": 1}])
        candidate = f"{base}__{qtr.prereg.distill.ACTIVE_VETO_OPERATOR}__{qtr.prereg.distill.DISTILLATION_VETO}"
        sleeves[base] = {
            "sleeve_id": base,
            "weight": qtr.prereg.distill.SLEEVE_WEIGHTS[candidate],
            "clock": {"path": str(clock), "sha256": qtr.sha256_file(clock), "rows": 1},
        }
    schedules = {}
    for name in ["transitions", "segments", "signed_episodes"]:
        schedule = tmp_path / f"{name}.csv.gz"
        _write_gzip_csv(schedule, [{"x": 1}, {"x": 2}])
        schedules[name] = {"path": str(schedule), "sha256": qtr.sha256_file(schedule), "rows": 2}
    package = _write_json_with_manifest(tmp_path / "clock_package.json", {
        "policy_id": qtr.SOURCE_POLICY_ID,
        "decision": "materialized_shadow_distilled_clock_package",
        "preregistration": {**source_preregistration, "status": "validated_against_committed_preregistration"},
        "implementation": {"builder": {"path": str(builder), "sha256": qtr.sha256_file(builder)}},
        "components": {"base_order": list(qtr.prereg.distill.DISTILLED_BASES)},
        "sleeves": sleeves,
        "portfolio_schedules": schedules,
        "portfolio_source_stats": {"splits": {"train": {"signed_episodes": 4}, "test": {"signed_episodes": 12}, "eval": {"signed_episodes": 13}, "final": {"signed_episodes": 8}}},
    })
    sleeve_row = lambda near_pass: {
        "checks": {"exact_entry_jaccard": True, "one_to_one_6h_max_matched_share": near_pass, "occupied_5m_bar_jaccard": True, "absolute_signed_exposure_pearson": True},
        "metrics": {"one_to_one_6h_max_matched_share": 0.44 if not near_pass else 0.0},
    }
    novelty = _write_json_with_manifest(tmp_path / "novelty.json", {
        "policy_id": qtr.SOURCE_POLICY_ID,
        "decision": "terminal_gross9_novelty_reject",
        "gross9_pass": False,
        "advance_to_economic_outcomes": False,
        "preregistration": source_preregistration,
        "source_package": {"path": str(tmp_path / "clock_package.json"), "sha256": qtr.sha256_file(tmp_path / "clock_package.json"), "manifest_hash": package["manifest_hash"], "predecessor_mutated": False},
        "gross9_sleeves": {"cand_rex_veto_7": sleeve_row(False), "markov_transition_long": sleeve_row(False), "fresh_kimchi_fx": sleeve_row(True)},
    })
    package_receipt = {
        "path": str(tmp_path / "clock_package.json"),
        "sha256": qtr.sha256_file(tmp_path / "clock_package.json"),
        "manifest_hash": package["manifest_hash"],
    }
    novelty_receipt = {
        "path": str(tmp_path / "novelty.json"),
        "sha256": qtr.sha256_file(tmp_path / "novelty.json"),
        "manifest_hash": novelty["manifest_hash"],
    }
    prereg_payload = _valid_prereg(
        tmp_path,
        source_preregistration,
        package_receipt,
        novelty_receipt,
    )
    monkeypatch.setattr(qtr.prereg, "build", lambda: prereg_payload)
    monkeypatch.setattr(qtr.prereg, "validate", lambda value: None)
    monkeypatch.setattr(qtr, "TERMINAL_NOVELTY", tmp_path / "novelty.json")
    return prereg_payload, package, novelty


def test_frozen_authorization_accepts_terminal_additive_reject_as_replacement_disclosure(monkeypatch, tmp_path):
    prereg_payload, package, novelty = _valid_artifacts(monkeypatch, tmp_path)
    auth = qtr.load_frozen_authorization(tmp_path / "prereg.json", tmp_path / "clock_package.json")
    assert [s.name for s in auth.sleeves] == list(qtr.prereg.distill.DISTILLED_BASES)
    assert auth.source_clock_package["manifest_hash"] == package["manifest_hash"]
    assert auth.terminal_additive_novelty["manifest_hash"] == novelty["manifest_hash"]
    assert auth.overlap_disclosure["near_6h_overlap_is_disclosure_not_authorization_gate"] is True
    assert set(auth.overlap_disclosure["near_6h_failures"]) == {"cand_rex_veto_7", "markov_transition_long"}
    assert auth.source_signed_episodes_by_split["final"] == 8


def test_authorization_rejects_non_near6h_overlap_failure(monkeypatch, tmp_path):
    _valid_artifacts(monkeypatch, tmp_path)
    bad = json.loads((tmp_path / "novelty.json").read_text(encoding="utf-8"))
    bad["gross9_sleeves"]["fresh_kimchi_fx"]["checks"]["occupied_5m_bar_jaccard"] = False
    bad["manifest_hash"] = qtr.canonical_hash({k: v for k, v in bad.items() if k != "manifest_hash"})
    (tmp_path / "novelty.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-near6h"):
        qtr._validate_terminal_overlap(bad)


def test_stage_checks_keep_oos_p_and_signed_episode_gates():
    primary = {
        "base": {"absolute_return_pct": 1, "cagr_to_strict_mdd": 3, "strict_mdd_pct": 10, "mean_exposure_weighted_gross_edge_bp": 21},
        "stress": {"absolute_return_pct": 1, "cagr_to_strict_mdd": 2.5},
        "calendar_halves": {"first": {"absolute_return_pct": 1}, "second": {"absolute_return_pct": 1}},
        "cluster_signflip": {"pvalue": 0.10},
    }
    assert "oos_cluster_signflip_p_max_0_1" not in qtr.stage_checks("train", primary)
    checks = qtr.stage_checks("test", primary, source_signed_episodes=12)
    assert checks["oos_cluster_signflip_p_max_0_1"] is True
    assert checks["source_min_nonzero_signed_episodes"] is True
    assert qtr.stage_checks("test", primary, source_signed_episodes=11)["source_min_nonzero_signed_episodes"] is False
    primary["cluster_signflip"] = {"pvalue": 0.10001}
    assert qtr.stage_checks("test", primary, source_signed_episodes=12)["oos_cluster_signflip_p_max_0_1"] is False


def test_predecessor_gate_blocks_before_any_loader_opening(monkeypatch, tmp_path):
    monkeypatch.setattr(qtr, "load_frozen_authorization", lambda: types.SimpleNamespace(sleeves=[], source_signed_episodes_by_split={}))
    monkeypatch.setattr(qtr, "load_portfolio_clock", lambda *a, **k: (_ for _ in ()).throw(AssertionError("loader opened")))
    outputs = {stage: tmp_path / f"{stage}.json" for stage in qtr.STAGES}
    with pytest.raises(RuntimeError, match="missing predecessor train"):
        qtr.run("test", output=tmp_path / "test.json", sleeves=[], outputs=outputs)

    core = {"protocol_version": qtr.PROTOCOL_VERSION, "policy_id": qtr.POLICY_ID, "stage": "train", "passed": False, "advance_to_next_stage": False}
    outputs["train"].write_text(json.dumps({**core, "manifest_hash": qtr.canonical_hash(core)}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="predecessor did not pass"):
        qtr.run("test", output=tmp_path / "test.json", sleeves=[], outputs=outputs)
