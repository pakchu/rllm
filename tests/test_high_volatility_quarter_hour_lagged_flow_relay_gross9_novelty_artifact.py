import json

from training import evaluate_high_volatility_quarter_hour_lagged_flow_relay_gross9_novelty as novelty


EXPECTED_SHA = "cc369a980f8c4d5fdc4f713bc235b93791ff8977996f8039d336119038099e2f"


def test_reproduced_gross9_pass_is_sealed() -> None:
    value = json.loads(novelty.OUTPUT.read_text())
    assert novelty.sha(novelty.OUTPUT) == EXPECTED_SHA
    assert value["every_gross9_sleeve_passed"] is True
    assert value["gross9_novelty_status"] == "passed"
    assert value["advance_to_economic_outcomes"] is True
    assert value["evidence_boundary"]["outcomes_opened"] is False
    metrics = [row["metrics"] for row in value["gross9_sleeves"].values()]
    assert max(row["exact_entry_jaccard"] for row in metrics) == 0.0
    assert max(row["one_to_one_6h_max_matched_share"] for row in metrics) < 0.18
    assert max(row["occupied_5m_bar_jaccard"] for row in metrics) < 0.068
    assert max(row["absolute_signed_exposure_pearson"] for row in metrics) < 0.026
