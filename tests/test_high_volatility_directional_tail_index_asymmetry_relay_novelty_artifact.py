import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/high_volatility_directional_tail_index_asymmetry_relay_gross9_novelty_2026-08-10.json"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hvdtiar_novelty_pass_is_frozen_blind():
    report = json.loads(RESULT.read_text())
    assert report["every_gross9_sleeve_passed"] is True
    assert report["advance_to_economic_outcomes"] is True
    assert report["evidence_boundary"]["outcomes_opened"] is False
    metrics = [value["metrics"] for value in report["gross9_sleeves"].values()]
    assert max(value["one_to_one_6h_max_matched_share"] for value in metrics) <= 0.35
    assert max(value["occupied_5m_bar_jaccard"] for value in metrics) <= 0.25
    assert max(value["absolute_signed_exposure_pearson"] for value in metrics) <= 0.35
    assert sha256(RESULT) == "b7ce8732dcf24f37998f5843cfa48fc742179007dbcf63ef4db65ddb829fbad5"
