import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_taker_imbalance_concentration_continuation_gross9_novelty_2026-08-13.json")


def test_novelty_pass_is_reproducible_and_blind() -> None:
    report = json.loads(RESULT.read_text())
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == "7adeb91d03fa1b3a08bac75866ef1d184dd7b0f431e257c03789ce9980bede59"
    assert report["policy_id"] == "HVTICC-8"
    assert report["source_support_passed"] and report["every_gross9_sleeve_passed"]
    assert report["advance_to_economic_outcomes"]
    assert not report["evidence_boundary"]["outcomes_opened"]
    assert report["evidence_boundary"]["economic_outcome_rows_opened"] == 0


def test_every_sleeve_passes_limits() -> None:
    report = json.loads(RESULT.read_text())
    assert all(value["passed"] and all(value["checks"].values()) for value in report["gross9_sleeves"].values())
    metrics = [value["metrics"] for value in report["gross9_sleeves"].values()]
    assert max(value["exact_entry_jaccard"] for value in metrics) == 0
    assert max(value["one_to_one_6h_max_matched_share"] for value in metrics) < 0.207
    assert max(value["occupied_5m_bar_jaccard"] for value in metrics) < 0.060
    assert max(value["absolute_signed_exposure_pearson"] for value in metrics) < 0.079
