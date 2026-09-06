import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_intraday_hour_reversal_gross9_novelty_2026-08-11.json")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hvihr_novelty_passes_with_outcomes_sealed():
    artifact = json.loads(RESULT.read_text())
    assert sha(RESULT) == "036e1a2b2548ee1a156d3b029752cd4e660a68ce56fba6ceff46f479d4494261"
    assert artifact["policy_id"] == "HVIHR-1"
    assert artifact["every_gross9_sleeve_passed"] is True
    assert artifact["advance_to_economic_outcomes"] is True
    assert artifact["gross9_novelty_status"] == "passed"
    assert artifact["evidence_boundary"]["outcomes_opened"] is False
    assert artifact["evidence_boundary"]["btc_execution_rows_opened"] == 0
    assert artifact["evidence_boundary"]["funding_rows_opened"] == 0
    assert all(result["passed"] for result in artifact["gross9_sleeves"].values())
    assert max(result["metrics"]["exact_entry_jaccard"] for result in artifact["gross9_sleeves"].values()) == 0.0
    assert max(result["metrics"]["one_to_one_6h_max_matched_share"] for result in artifact["gross9_sleeves"].values()) < 0.14
    assert max(result["metrics"]["occupied_5m_bar_jaccard"] for result in artifact["gross9_sleeves"].values()) < 0.01
    assert max(result["metrics"]["absolute_signed_exposure_pearson"] for result in artifact["gross9_sleeves"].values()) < 0.011
