import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/high_volatility_directional_variation_timing_migration_relay_gross9_novelty_2026-08-10.json"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hvdvtm_novelty_pass_is_frozen_blind():
    report = json.loads(RESULT.read_text())
    assert report["every_gross9_sleeve_passed"] is True
    assert report["advance_to_economic_outcomes"] is True
    assert report["evidence_boundary"]["outcomes_opened"] is False
    metrics = [value["metrics"] for value in report["gross9_sleeves"].values()]
    assert max(value["one_to_one_6h_max_matched_share"] for value in metrics) <= 0.35
    assert max(value["occupied_5m_bar_jaccard"] for value in metrics) <= 0.25
    assert max(value["absolute_signed_exposure_pearson"] for value in metrics) <= 0.35
    assert sha256(RESULT) == "1dc5ef0b65a88353b39dda9a26ee6b6064fcea31441fc5463003e49828584bf7"
