import hashlib
import json

from training import preregister_high_volatility_cross_alt_extreme_traversal_consensus_relay as prereg


def test_hvcatcr_preregistration_is_outcome_blind_and_hash_bound(tmp_path):
    report = prereg.build()
    assert report["policy_id"] == "HVCATCR-8"
    assert report["outcomes_opened"] is False
    assert report["source_incidence_opened"] is False
    assert report["gross9_rows_opened"] is False
    assert report["policy"]["consensus_min"] == 5
    assert report["policy"]["variation_rank_min"] == 0.65
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == prereg.canonical_hash(core)
    encoded = (json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    path = tmp_path / "prereg.json"
    path.write_bytes(encoded)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == hashlib.sha256(encoded).hexdigest()
