import json

from training import preregister_high_volatility_directional_trade_breadth_asymmetry_relay as prereg


def test_manifest_is_deterministic_and_self_bound() -> None:
    first = prereg.build()
    second = prereg.build()
    assert first == second
    prereg.validate(first)
    assert first["manifest_hash"] == prereg.canonical_hash(
        {key: value for key, value in first.items() if key != "manifest_hash"}
    )


def test_frozen_contract_preserves_sequential_evidence_boundary() -> None:
    result = prereg.build()
    assert result["policy_id"] == "HVDTBA-6"
    assert result["source_incidence_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert result["policy"]["history_hours"] == 2160
    assert result["policy"]["minimum_history_hours"] == 1440
    assert result["novelty_gates"]["must_pass_before_economics"] is True
    assert result["economic_gates"]["stop_on_first_failure"] is True
    assert result["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False
    assert result["research_boundary"]["candidate_count"] == 1
    assert result["research_boundary"]["grid"] is False


def test_written_artifact_matches_builder(tmp_path) -> None:
    payload = prereg.build()
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    assert json.loads(path.read_text()) == payload
