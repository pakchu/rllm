import hashlib, json
from pathlib import Path

from training import evaluate_high_volatility_cross_alt_realized_quarticity_concentration_spillover_relay_gross9_novelty as n


RESULT = Path("results/high_volatility_cross_alt_realized_quarticity_concentration_spillover_relay_gross9_novelty_2026-08-13.json")
EXPECTED = "99e6e3cde2368f3abe82e7c4c82dc7fbbf37104f47e29b1cb4ab73ba8f6d6617"


def test_novelty_pass_is_immutable_and_economics_remained_sealed():
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED
    payload = json.loads(RESULT.read_text())
    manifest_hash = payload.pop("manifest_hash")
    assert n.canonical_hash(payload) == manifest_hash
    assert payload["policy_id"] == "HVCARQ-6"
    assert payload["every_gross9_sleeve_passed"]
    assert payload["advance_to_economic_outcomes"]
    evidence = payload["evidence_boundary"]
    assert evidence["btc_execution_rows_opened"] == 0
    assert evidence["btc_price_or_return_rows_opened"] == 0
    assert evidence["funding_rows_opened"] == 0
    assert evidence["economic_outcome_rows_opened"] == 0
    assert not evidence["outcomes_opened"]
    assert max(v["metrics"]["exact_entry_jaccard"] for v in payload["gross9_sleeves"].values()) == 0
    assert max(v["metrics"]["one_to_one_6h_max_matched_share"] for v in payload["gross9_sleeves"].values()) == 0.14
    assert max(v["metrics"]["occupied_5m_bar_jaccard"] for v in payload["gross9_sleeves"].values()) < 0.044
    assert max(v["metrics"]["absolute_signed_exposure_pearson"] for v in payload["gross9_sleeves"].values()) < 0.060
