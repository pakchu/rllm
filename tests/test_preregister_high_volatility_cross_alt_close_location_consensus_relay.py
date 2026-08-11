import json
from training import preregister_high_volatility_cross_alt_close_location_consensus_relay as p


def test_manifest_hash_and_frozen_contract():
    payload = p.build(); core = dict(payload); manifest_hash = core.pop("manifest_hash")
    assert manifest_hash == p.canonical_hash(core)
    assert payload["policy_id"] == "HVCACLR-8"
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["clock"]["hold"] == "8 elapsed hours"
    assert payload["policy"]["minimum_consensus_breadth"] == 5
    assert payload["policy"]["strength_rank_min"] == 0.75
    assert payload["policy"]["variation_rank_min"] == 0.65
    assert payload["post_stage_volatility_audit"]["rv20_q90_entry_filter"] is False


def test_default_output_matches_builder(tmp_path):
    payload = p.build()
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(payload, ensure_ascii=False))
    assert json.loads(path.read_text()) == payload
