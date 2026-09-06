import json

from training import preregister_high_volatility_alt_basket_leadership_relay as prereg


def test_manifest_is_deterministic_and_self_bound() -> None:
    payload = prereg.build()
    assert payload == prereg.build()
    prereg.validate(payload)
    assert payload["manifest_hash"] == prereg.canonical_hash(
        {key: value for key, value in payload.items() if key != "manifest_hash"}
    )


def test_frozen_alt_basket_leadership_contract() -> None:
    payload = prereg.build()
    assert payload["policy_id"] == "HVABLR-6"
    assert payload["features"]["universe"] == prereg.SYMBOLS
    assert payload["policy"]["leadership_rank_min"] == 0.80
    assert payload["policy"]["btc_variation_rank_min"] == 0.65
    assert payload["policy"]["minimum_history_hours"] == 1440
    assert payload["source_incidence_opened"] is False
    assert payload["novelty_gates"]["must_pass_before_economics"] is True
    assert payload["research_boundary"]["candidate_count"] == 1
    assert payload["research_boundary"]["grid"] is False


def test_serialized_payload_round_trip(tmp_path) -> None:
    payload = prereg.build()
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    assert json.loads(path.read_text()) == payload
