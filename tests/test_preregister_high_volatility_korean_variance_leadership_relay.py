import json

from training import preregister_high_volatility_korean_variance_leadership_relay as prereg


def test_manifest_is_deterministic_and_self_bound() -> None:
    payload = prereg.build()
    assert payload == prereg.build()
    prereg.validate(payload)
    assert payload["manifest_hash"] == prereg.canonical_hash(
        {key: value for key, value in payload.items() if key != "manifest_hash"}
    )


def test_frozen_regional_variance_contract() -> None:
    payload = prereg.build()
    assert payload["policy_id"] == "HVKVLR-6"
    assert payload["source_incidence_opened"] is False
    assert payload["policy"]["window_hours"] == 6
    assert payload["policy"]["history_hours"] == 2160
    assert payload["policy"]["minimum_history_hours"] == 1440
    assert payload["clock"]["gross_exposure"] == 0.5
    assert payload["novelty_gates"]["must_pass_before_economics"] is True
    assert payload["research_boundary"]["candidate_count"] == 1
    assert payload["research_boundary"]["grid"] is False
    assert payload["diagnostic_controls"]["cannot_be_promoted"] is True


def test_serialized_payload_round_trip(tmp_path) -> None:
    payload = prereg.build()
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    assert json.loads(path.read_text()) == payload
