from __future__ import annotations

import pytest

import training.preregister_causal_residual_expert_switcher_2026 as prereg


def test_protocol_freezes_exact_online_policy_and_outcome_boundary() -> None:
    protocol = prereg.protocol()
    assert protocol["evidence_boundary"]["confirmation_2026_post_entry_returns_opened"] is False
    assert protocol["evidence_boundary"]["confirmation_window"] == [
        "2026-01-01T00:00:00Z",
        "2026-07-01T00:00:00Z",
    ]
    assert protocol["online_policy"]["ridge_alpha"] == 300.0
    assert protocol["online_policy"]["minimum_training_rows"] == 52
    assert protocol["online_policy"]["maximum_recent_training_rows"] == 104
    assert len(protocol["online_policy"]["features"]) == 13
    assert protocol["risk_policy"]["gross_scale"] == "clip(reference/current, 0.25, 1.0)"


def test_run_rejects_development_artifact_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prereg, "sha256_file", lambda path: "drift")
    with pytest.raises(RuntimeError, match="development result changed"):
        prereg.run(output="unused.json", docs_output="unused.md")


def test_protocol_hash_is_deterministic() -> None:
    assert prereg.canonical_hash(prereg.protocol()) == prereg.canonical_hash(prereg.protocol())
