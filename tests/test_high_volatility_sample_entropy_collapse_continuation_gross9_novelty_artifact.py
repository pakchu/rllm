import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_sample_entropy_collapse_continuation_gross9_novelty_2026-08-13.json")


def test_gross9_novelty_pass_is_reproducible_and_outcome_blind() -> None:
    payload = json.loads(RESULT.read_text())
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == (
        "58f9737e6b00f8eb778a2274258cb9219e66b844d849f57331a3e71150fa53bb"
    )
    assert payload["policy_id"] == "HVSENC-8"
    assert payload["source_support_passed"] is True
    assert payload["every_gross9_sleeve_passed"] is True
    assert payload["gross9_novelty_status"] == "passed"
    assert payload["advance_to_economic_outcomes"] is True
    assert payload["evidence_boundary"]["outcomes_opened"] is False
    assert payload["evidence_boundary"]["economic_outcome_rows_opened"] == 0
    assert payload["evidence_boundary"]["btc_execution_rows_opened"] == 0


def test_every_sleeve_passes_every_frozen_limit() -> None:
    payload = json.loads(RESULT.read_text())
    for sleeve in payload["gross9_sleeves"].values():
        assert sleeve["passed"] is True
        assert all(sleeve["checks"].values())
    metrics = [item["metrics"] for item in payload["gross9_sleeves"].values()]
    assert max(item["exact_entry_jaccard"] for item in metrics) == 0.0
    assert max(item["one_to_one_6h_max_matched_share"] for item in metrics) == 5 / 29
    assert max(item["occupied_5m_bar_jaccard"] for item in metrics) < 0.051
    assert max(item["absolute_signed_exposure_pearson"] for item in metrics) < 0.091
