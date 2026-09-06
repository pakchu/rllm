import hashlib
import json

from training import evaluate_high_volatility_cross_alt_barrier_rejection_reversal_gross9_novelty as n


EXPECTED = "bea85ae1bbd2fb8e511236112f7fa090504e4d251039949da58366224d92d5f9"


def test_novelty_pass_is_immutable_and_authorizes_economics():
    assert hashlib.sha256(n.OUTPUT.read_bytes()).hexdigest() == EXPECTED
    result = json.loads(n.OUTPUT.read_text())
    manifest_hash = result.pop("manifest_hash")
    assert n.canonical_hash(result) == manifest_hash
    assert result["source_support_passed"]
    assert result["every_gross9_sleeve_passed"]
    assert result["gross9_novelty_status"] == "passed"
    assert result["advance_to_economic_outcomes"]
    evidence = result["evidence_boundary"]
    assert evidence["btc_execution_rows_opened"] == 0
    assert evidence["btc_price_or_return_rows_opened"] == 0
    assert evidence["funding_rows_opened"] == 0
    assert evidence["economic_outcome_rows_opened"] == 0
    assert not evidence["outcomes_opened"]
    assert max(
        sleeve["metrics"]["one_to_one_6h_max_matched_share"]
        for sleeve in result["gross9_sleeves"].values()
    ) <= 0.35
