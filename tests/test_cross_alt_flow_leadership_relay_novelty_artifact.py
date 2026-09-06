import hashlib
import json

from training import evaluate_cross_alt_flow_leadership_relay_gross9_novelty as novelty


def test_caflr_novelty_failure_is_outcome_sealed():
    assert hashlib.sha256(novelty.OUTPUT.read_bytes()).hexdigest() == "07b63622f77356db1730c23b3e9f77f2fa77ded9fe61af54ea21d69ae5cd95be"
    result = json.loads(novelty.OUTPUT.read_text())
    assert result["policy_id"] == "CAFLR-6"
    assert result["every_gross9_sleeve_passed"] is False
    assert result["gross9_novelty_status"] == "failed"
    assert result["advance_to_economic_outcomes"] is False
    boundary = result["evidence_boundary"]
    assert boundary["btc_execution_rows_opened"] == 0
    assert boundary["funding_rows_opened"] == 0
    assert boundary["economic_outcome_rows_opened"] == 0
    assert boundary["outcomes_opened"] is False


def test_caflr_failure_is_only_near_six_hour_overlap():
    result = json.loads(novelty.OUTPUT.read_text())
    for sleeve in result["gross9_sleeves"].values():
        assert sleeve["checks"]["one_to_one_6h_max_matched_share"] is False
        assert sleeve["metrics"]["one_to_one_6h_max_matched_share"] > 0.35
        assert sleeve["checks"]["exact_entry_jaccard"] is True
        assert sleeve["checks"]["occupied_5m_bar_jaccard"] is True
        assert sleeve["checks"]["absolute_signed_exposure_pearson"] is True
