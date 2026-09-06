import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/high_volatility_cross_alt_order_flow_tail_asymmetry_relay_support_2026-08-11.json"
)


def test_source_support_pass() -> None:
    result = json.loads(RESULT.read_text())
    assert result["support_passed"] is True
    assert result["decision"] == "pass_to_novelty"
    assert result["advance_to_gross9_novelty"] is True
    assert result["advance_to_economic_outcomes"] is False
    assert {key: value["events"] for key, value in result["support"].items()} == {
        "train": 127,
        "test": 261,
        "eval": 276,
        "final": 161,
    }
    assert all(result["support_checks"].values())
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["funding_values_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == (
        "21a51a673453c40f7d55ec178a3a619012d5586ddbb8fa709e723299be322bd9"
    )


def test_downstream_still_sealed() -> None:
    assert not Path(
        "results/high_volatility_cross_alt_order_flow_tail_asymmetry_relay_gross9_novelty_2026-08-11.json"
    ).exists()
    for stage in ("train", "test", "eval", "final"):
        assert not Path(
            f"results/high_volatility_cross_alt_order_flow_tail_asymmetry_relay_{stage}_economics_2026-08-11.json"
        ).exists()
