import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/high_volatility_cross_alt_return_tail_asymmetry_relay_gross9_novelty_2026-08-11.json"
)


def test_terminal_novelty_rejection() -> None:
    result = json.loads(RESULT.read_text())
    assert result["gross9_novelty_status"] == "failed"
    assert result["every_gross9_sleeve_passed"] is False
    assert result["advance_to_economic_outcomes"] is False
    frozen = result["gross9_sleeves"]["frozen_annual_rank7"]
    assert frozen["checks"]["one_to_one_6h_max_matched_share"] is False
    assert frozen["metrics"]["one_to_one_6h_max_matched_share"] == 0.3793103448275862
    assert all(
        item["passed"]
        for sleeve, item in result["gross9_sleeves"].items()
        if sleeve != "frozen_annual_rank7"
    )
    assert result["evidence_boundary"]["btc_execution_rows_opened"] == 0
    assert result["evidence_boundary"]["economic_outcome_rows_opened"] == 0
    assert result["evidence_boundary"]["outcomes_opened"] is False
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == (
        "0d03afe0b982ae781d3ea175f02d8915bfc2b72610058766d28aa360d6b260b5"
    )


def test_economics_remains_sealed() -> None:
    for stage in ("train", "test", "eval", "final"):
        assert not Path(
            f"results/high_volatility_cross_alt_return_tail_asymmetry_relay_{stage}_economics_2026-08-11.json"
        ).exists()
