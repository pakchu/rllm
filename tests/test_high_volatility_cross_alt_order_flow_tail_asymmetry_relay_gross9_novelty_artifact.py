import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/high_volatility_cross_alt_order_flow_tail_asymmetry_relay_gross9_novelty_2026-08-11.json"
)


def test_terminal_novelty_rejection() -> None:
    result = json.loads(RESULT.read_text())
    assert result["gross9_novelty_status"] == "failed"
    assert result["advance_to_economic_outcomes"] is False
    rex = result["gross9_sleeves"]["cand_rex_veto_7"]
    assert rex["checks"]["one_to_one_6h_max_matched_share"] is False
    assert rex["metrics"]["one_to_one_6h_max_matched_share"] == 0.37
    assert all(
        item["passed"]
        for sleeve, item in result["gross9_sleeves"].items()
        if sleeve != "cand_rex_veto_7"
    )
    assert result["evidence_boundary"]["btc_execution_rows_opened"] == 0
    assert result["evidence_boundary"]["economic_outcome_rows_opened"] == 0
    assert result["evidence_boundary"]["outcomes_opened"] is False
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == (
        "f93e72f7a1e37b54d9e9eaf4eee0dfb1d7ed02878b96c70bcf06ff20ea990bb8"
    )


def test_economics_remains_sealed() -> None:
    for stage in ("train", "test", "eval", "final"):
        assert not Path(
            f"results/high_volatility_cross_alt_order_flow_tail_asymmetry_relay_{stage}_economics_2026-08-11.json"
        ).exists()
