from __future__ import annotations

import copy
import json

import pytest

import training.audit_gross9_ajr_medium_frequency as module


def test_same_gross_weights_use_frozen_grid_and_identical_gross() -> None:
    for weight in module.WEIGHTS:
        combined, comparator = module.same_gross_weights(weight)
        assert sum(combined.values()) == pytest.approx(9.0 + weight)
        assert sum(comparator.values()) == pytest.approx(9.0 + weight)
        assert combined[module.CANDIDATE] == weight
        assert module.CANDIDATE not in comparator
        scale = (9.0 + weight) / 9.0
        for sleeve, base_weight in module.BASELINE_WEIGHTS.items():
            assert comparator[sleeve] == pytest.approx(base_weight * scale)


def test_frequency_uses_calendar_window_and_reports_sleeves() -> None:
    metric = {
        "trades": 156,
        "trades_by_sleeve": {"base": 120, module.CANDIDATE: 36},
    }
    report = module.frequency_report(
        metric, start="2025-01-01", end="2026-01-01"
    )
    assert report["completed_trades"] == 156
    assert report["completed_trades_by_sleeve"] == metric["trades_by_sleeve"]
    assert report["combined_completed_trades_per_week"] == pytest.approx(
        156 / (365 / 7)
    )


def test_result_hash_is_deterministic_and_detects_mutation() -> None:
    first = module.finalize_payload({"phase": "selection", "weights": [0.1]})
    second = module.finalize_payload({"weights": [0.1], "phase": "selection"})
    assert first["result_hash"] == second["result_hash"]
    module.verify_result_hash(first)
    changed = copy.deepcopy(first)
    changed["weights"] = [0.25]
    with pytest.raises(RuntimeError, match="result hash drifted"):
        module.verify_result_hash(changed)


def test_selection_artifact_seals_pre2024_train_and_no_rerank() -> None:
    row = {"candidate_weight": 0.25, "passes": True}
    payload = module.finalize_payload({
        "phase": "selection",
        "selection_windows": ["train"],
        "candidate_source_end_exclusive": "2024-01-01",
        "weight_grid": list(module.WEIGHTS),
        "future_opened": False,
        "future_can_rerank": False,
        "frozen_top1": row,
    })
    assert module.verify_selection_artifact(payload) == row
    exposed = copy.deepcopy(payload)
    exposed["selection_windows"] = ["train", "test2024"]
    exposed = module.finalize_payload(exposed)
    with pytest.raises(RuntimeError, match="non-train"):
        module.verify_selection_artifact(exposed)


def test_eval_consumes_frozen_top_without_selection_replay(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = module.finalize_payload({
        "phase": "selection",
        "selection_windows": ["train"],
        "candidate_source_end_exclusive": "2024-01-01",
        "weight_grid": list(module.WEIGHTS),
        "future_opened": False,
        "future_can_rerank": False,
        "candidate_freeze_hash": "freeze",
        "candidate_selection_schedule_hashes": {"fit": "schedule"},
        "frozen_top1": {"candidate_weight": 0.25, "passes": True},
    })
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection), encoding="utf-8")
    monkeypatch.setattr(
        module, "selection_payload",
        lambda _cfg: (_ for _ in ()).throw(AssertionError("eval reranked")),
    )
    monkeypatch.setattr(
        module, "validate_candidate_freeze",
        lambda _cfg: ({"freeze_hash": "different"}, {}),
    )
    with pytest.raises(RuntimeError, match="different AJR freeze"):
        module.eval_payload(module.Config(selection_output=str(path)))


def test_committed_ajr_freeze_and_schedule_hashes_are_bound() -> None:
    frozen, oos = module.validate_candidate_freeze(module.Config())
    assert frozen["freeze_hash"] == oos["freeze_hash"]
    assert frozen["selection_schedule_hashes"] == oos["selection_schedule_hashes"]
    assert frozen["selection_schedule_hashes"]["select_2023"]
