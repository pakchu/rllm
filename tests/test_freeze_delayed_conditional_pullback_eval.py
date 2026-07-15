import json

import pytest

from training.freeze_delayed_conditional_pullback_eval import (
    EVAL_GATE,
    EVAL_WINDOWS,
    MANIFEST_FIELDS,
    _write_once,
    manifest_hash,
    validate_manifest,
)


def _manifest() -> dict:
    core = {
        "schema_version": 1,
        "phase": "test_2024_frozen_for_eval",
        "test_opened": True,
        "eval_opened": False,
        "test_end_exclusive": "2025-01-01",
        "source_commit": "x",
        "source_artifact": {},
        "source_hashes_through_2024": {},
        "feature_hash_through_2024": "x",
        "model_spec": {},
        "candidate_spec": {"quantiles": {"funding_q": 0.3, "premium_q": 0.5, "low_width_q": 0.2, "pullback_q": 0.4}},
        "execution_spec": {},
        "selected_activation_hash": "x",
        "selected_schedule_hashes": {},
        "selected_stats": {},
        "stability_summary": {},
        "eval_windows": {key: list(value) for key, value in EVAL_WINDOWS.items()},
        "eval_gate": EVAL_GATE,
    }
    assert tuple(core) == MANIFEST_FIELDS
    manifest = {"created_at": "now", **core}
    manifest["manifest_hash"] = manifest_hash(manifest)
    return manifest


def test_eval_manifest_is_sealed_and_preregistered():
    manifest = _manifest()
    validate_manifest(manifest)
    assert manifest["eval_opened"] is False
    assert EVAL_GATE["eval_2025"]["min_ratio"] == 3.0
    assert EVAL_GATE["holdout_2026h1"]["min_trades"] == 6


def test_eval_manifest_is_write_once(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = _manifest()
    assert _write_once(path, manifest) == manifest
    changed = json.loads(json.dumps(manifest))
    changed["selected_activation_hash"] = "changed"
    changed["manifest_hash"] = manifest_hash(changed)
    with pytest.raises(RuntimeError, match="refusing"):
        _write_once(path, changed)


def test_eval_manifest_rejects_opened_or_changed_gate():
    opened = {**_manifest(), "eval_opened": True}
    opened["manifest_hash"] = manifest_hash(opened)
    with pytest.raises(RuntimeError, match="sealed"):
        validate_manifest(opened)
    changed = json.loads(json.dumps(_manifest()))
    changed["eval_gate"]["eval_2025"]["min_ratio"] = 2.9
    changed["manifest_hash"] = manifest_hash(changed)
    with pytest.raises(RuntimeError, match="gate"):
        validate_manifest(changed)
