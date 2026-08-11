import hashlib, json
from pathlib import Path

RESULT = Path("results/high_volatility_cross_alt_close_location_consensus_relay_gross9_novelty_2026-08-11.json")


def test_terminal_novelty_rejection_artifact():
    data = json.loads(RESULT.read_text())
    assert data["gross9_novelty_status"] == "failed"
    assert data["every_gross9_sleeve_passed"] is False
    assert data["advance_to_economic_outcomes"] is False
    assert data["evidence_boundary"]["economic_outcome_rows_opened"] == 0
    assert data["evidence_boundary"]["btc_price_or_return_rows_opened"] == 0
    assert all(
        sleeve["metrics"]["one_to_one_6h_max_matched_share"] > 0.35
        for sleeve in data["gross9_sleeves"].values()
    )
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == "11a4366c45bf13289ee8ae0bd00d522a42c3f2bb45082ee4d51a28d1df9325ba"


def test_economics_never_opened():
    for stage in ("train", "test", "eval", "final"):
        assert not Path(f"results/high_volatility_cross_alt_close_location_consensus_relay_{stage}_economics_2026-08-11.json").exists()
