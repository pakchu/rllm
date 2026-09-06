import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/high_volatility_price_level_occupation_relay_gross9_novelty_2026-08-10.json"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hvplo_novelty_pass_is_frozen_blind():
    report = json.loads(RESULT.read_text())
    assert report["every_gross9_sleeve_passed"] is True
    assert report["advance_to_economic_outcomes"] is True
    assert report["evidence_boundary"]["outcomes_opened"] is False
    metrics = [value["metrics"] for value in report["gross9_sleeves"].values()]
    assert max(value["one_to_one_6h_max_matched_share"] for value in metrics) <= 0.35
    assert max(value["occupied_5m_bar_jaccard"] for value in metrics) <= 0.25
    assert max(value["absolute_signed_exposure_pearson"] for value in metrics) <= 0.35
    assert sha256(RESULT) == "76032f9d2be4cc7c044faf4cbb78550d98079ff28c8f17fa532f5afd31956946"
