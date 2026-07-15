import json

import pytest

from training.freeze_stable_ensemble_conditional_pullback_alpha import (
    AUDIT_COMMIT,
    FUTURE_WINDOWS,
    MANIFEST_FIELDS,
    OOS_GATE,
    SELECTION_COMMIT,
    _write_once,
    manifest_hash,
    validate_manifest,
)


def _manifest() -> dict:
    core = {
        "schema_version": 1,
        "phase": "pre_oos_frozen",
        "oos_opened": False,
        "selection_end_exclusive": "2024-01-01",
        "source_commits": {"selection": SELECTION_COMMIT, "audit": AUDIT_COMMIT},
        "source_artifacts": {},
        "source_prefix_hashes": {},
        "feature_prefix_hash": "x",
        "model_spec": {},
        "candidate_spec": {"quantiles": {"funding_q": 0.3, "premium_q": 0.5, "low_width_q": 0.2, "pullback_q": 0.4}},
        "execution_spec": {},
        "selected_activation_hash": "x",
        "selected_schedule_hashes": {},
        "selected_stats": {},
        "stability_summary": {},
        "future_windows": {key: list(value) for key, value in FUTURE_WINDOWS.items()},
        "oos_gate": OOS_GATE,
    }
    assert tuple(core) == MANIFEST_FIELDS
    output = {"created_at": "now", **core}
    output["manifest_hash"] = manifest_hash(output)
    return output


def test_manifest_is_sealed_hashed_and_has_preregistered_windows():
    manifest = _manifest()
    validate_manifest(manifest)
    assert manifest["oos_opened"] is False
    assert OOS_GATE["test_2024"]["min_ratio"] == 3.0
    assert OOS_GATE["eval_2025"]["min_trades"] == 12
    assert OOS_GATE["holdout_2026h1"]["min_trades"] == 6


def test_manifest_write_once_refuses_different_payload(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = _manifest()
    assert _write_once(path, manifest) == manifest
    assert _write_once(path, {**manifest, "created_at": "later"}) == manifest
    changed = json.loads(json.dumps(manifest))
    changed["selected_activation_hash"] = "different"
    changed["manifest_hash"] = manifest_hash(changed)
    with pytest.raises(RuntimeError, match="refusing"):
        _write_once(path, changed)


def test_manifest_rejects_opened_oos_and_changed_gate():
    opened = {**_manifest(), "oos_opened": True}
    opened["manifest_hash"] = manifest_hash(opened)
    with pytest.raises(RuntimeError, match="sealed"):
        validate_manifest(opened)
    changed = json.loads(json.dumps(_manifest()))
    changed["oos_gate"]["test_2024"]["min_ratio"] = 2.9
    changed["manifest_hash"] = manifest_hash(changed)
    with pytest.raises(RuntimeError, match="gate"):
        validate_manifest(changed)
