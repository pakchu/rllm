import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_hikkake_pattern_relay_gross9_novelty_2026-08-11.json")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hikkake_novelty_pass_is_frozen_and_outcome_blind() -> None:
    report = json.loads(RESULT.read_text())
    assert report["every_gross9_sleeve_passed"] is True
    assert report["advance_to_economic_outcomes"] is True
    assert report["evidence_boundary"]["outcomes_opened"] is False
    metrics = [value["metrics"] for value in report["gross9_sleeves"].values()]
    assert max(value["exact_entry_jaccard"] for value in metrics) <= 0.1
    assert max(value["one_to_one_6h_max_matched_share"] for value in metrics) <= 0.35
    assert max(value["occupied_5m_bar_jaccard"] for value in metrics) <= 0.25
    assert max(value["absolute_signed_exposure_pearson"] for value in metrics) <= 0.35
    assert sha(RESULT) == "02438d6701810b19e9cc0cb2f682d6317c09e07258686fd90671587d786d5720"
