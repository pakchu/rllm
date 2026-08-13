import hashlib
import json
from pathlib import Path

from training import evaluate_high_volatility_bds_nonlinear_dependence_relay_gross9_novelty as n


RESULT = Path("results/high_volatility_bds_nonlinear_dependence_relay_gross9_novelty_2026-08-13.json")
EXPECTED = "d70cd898edb50c571ae45ec361e83ef911ce39abf1acfe5d9696f0a6a6bc4bfc"


def test_gross9_pass_is_immutable_and_economics_remain_sealed():
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED
    result = json.loads(RESULT.read_text())
    manifest_hash = result.pop("manifest_hash")
    assert n.canonical_hash(result) == manifest_hash
    assert result["policy_id"] == "HVBDS-8"
    assert result["every_gross9_sleeve_passed"]
    assert result["gross9_novelty_status"] == "passed"
    assert result["advance_to_economic_outcomes"]
    assert not result["evidence_boundary"]["outcomes_opened"]
    assert result["evidence_boundary"]["economic_outcome_rows_opened"] == 0
    assert max(
        sleeve["metrics"]["one_to_one_6h_max_matched_share"]
        for sleeve in result["gross9_sleeves"].values()
    ) <= 0.35
    assert max(
        sleeve["metrics"]["absolute_signed_exposure_pearson"]
        for sleeve in result["gross9_sleeves"].values()
    ) <= 0.35
