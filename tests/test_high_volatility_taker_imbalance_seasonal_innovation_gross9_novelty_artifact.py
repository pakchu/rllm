import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_taker_imbalance_seasonal_innovation_gross9_novelty_2026-08-13.json")


def test_novelty_pass_is_reproducible_and_blind() -> None:
    report = json.loads(RESULT.read_text())
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == "a6c18816810617151913826628d36d51c4f8056ce5169b2496bd585a66619063"
    assert report["policy_id"] == "HVTISI-8"
    assert report["source_support_passed"]
    assert report["every_gross9_sleeve_passed"]
    assert report["advance_to_economic_outcomes"]
    assert not report["evidence_boundary"]["outcomes_opened"]
    assert report["evidence_boundary"]["economic_outcome_rows_opened"] == 0


def test_every_sleeve_passes_limits() -> None:
    report = json.loads(RESULT.read_text())
    assert all(value["passed"] and all(value["checks"].values()) for value in report["gross9_sleeves"].values())
    metrics = [value["metrics"] for value in report["gross9_sleeves"].values()]
    assert max(value["exact_entry_jaccard"] for value in metrics) == 0
    assert max(value["one_to_one_6h_max_matched_share"] for value in metrics) < 0.265
    assert max(value["occupied_5m_bar_jaccard"] for value in metrics) < 0.069
    assert max(value["absolute_signed_exposure_pearson"] for value in metrics) < 0.103
