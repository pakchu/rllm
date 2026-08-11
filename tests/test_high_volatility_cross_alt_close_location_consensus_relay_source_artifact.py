import hashlib, json
from pathlib import Path

RESULT = Path("results/high_volatility_cross_alt_close_location_consensus_relay_support_2026-08-11.json")


def test_source_support_pass_artifact():
    data = json.loads(RESULT.read_text())
    assert data["support_passed"] is True
    assert data["decision"] == "pass_to_novelty"
    assert data["advance_to_gross9_novelty"] is True
    assert data["advance_to_economic_outcomes"] is False
    assert {k: v["events"] for k, v in data["support"].items()} == {
        "train": 144, "test": 298, "eval": 281, "final": 147,
    }
    assert all(data["support_checks"].values())
    assert data["clock"]["rows"] == 870
    assert data["postentry_return_pnl_execution_price_opened"] is False
    assert data["funding_values_opened"] is False
    assert data["gross9_rows_opened"] is False
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == "a55abb4491fe1f21f4d6f146213adcc2949cb64867bf56387662619bcaafab3f"


def test_economics_remains_sealed():
    for stage in ("train", "test", "eval", "final"):
        assert not Path(f"results/high_volatility_cross_alt_close_location_consensus_relay_{stage}_economics_2026-08-11.json").exists()
