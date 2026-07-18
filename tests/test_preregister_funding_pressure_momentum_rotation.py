from __future__ import annotations

import json

from training import preregister_funding_pressure_momentum_rotation as prereg


def test_protocol_is_singleton_and_keeps_outcomes_sealed() -> None:
    policy = prereg.protocol()
    assert policy["feature_formula"]["score"] == (
        "residual_level_z + residual_rotation - funding_pressure_change_z"
    )
    assert policy["clock"]["entry_time"] == "Monday 00:10 UTC open"
    assert policy["selection_2023"]["singleton_no_parameter_ranking"] is True
    boundary = policy["evidence_boundary"]
    assert boundary["exact_fpmr_score_or_post_entry_return_opened"] is False
    assert boundary["selection_2023_outcomes_opened"] is False
    assert boundary["test_2024_outcomes_opened"] is False


def test_run_writes_hash_bound_artifacts(tmp_path) -> None:
    output = tmp_path / "registration.json"
    docs = tmp_path / "registration.md"
    payload = prereg.run(output, docs)
    loaded = json.loads(output.read_text())
    assert loaded == payload
    assert loaded["protocol_hash"] == prereg.canonical_hash(loaded["protocol"])
    assert loaded["protocol_hash"] in docs.read_text()


def test_protocol_contains_no_mutable_grid() -> None:
    encoded = json.dumps(prereg.protocol(), sort_keys=True)
    assert "candidate_grid" not in encoded
    assert "ranked_candidates" not in encoded
    assert prereg.protocol()["sequential_oos"]["no_sign_weight_lookback_hold_or_score_repair"]
