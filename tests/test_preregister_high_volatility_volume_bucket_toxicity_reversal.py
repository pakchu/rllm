import json

from training import preregister_high_volatility_volume_bucket_toxicity_reversal as prereg


def test_manifest_is_deterministic_and_self_bound() -> None:
    payload = prereg.build()
    assert payload == prereg.build()
    prereg.validate(payload)
    assert payload["manifest_hash"] == prereg.canonical_hash(
        {key: value for key, value in payload.items() if key != "manifest_hash"}
    )


def test_frozen_intrinsic_clock_and_evidence_boundary() -> None:
    payload = prereg.build()
    assert payload["policy_id"] == "HVVBTR-6"
    assert payload["source_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert payload["policy"]["target_buckets_per_day"] == 24
    assert payload["policy"]["toxicity_buckets"] == 24
    assert payload["policy"]["history_buckets"] == 720
    assert payload["policy"]["minimum_history_buckets"] == 480
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
