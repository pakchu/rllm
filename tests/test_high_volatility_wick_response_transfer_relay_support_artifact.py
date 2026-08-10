import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_wick_response_transfer_relay_support_2026-08-10.json")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hvwrtr_support_pass_is_frozen_blind_reproducible():
    report = json.loads(RESULT.read_text())
    assert report["support_passed"] is True
    assert report["advance_to_gross9_novelty"] is True
    assert report["advance_to_economic_outcomes"] is False
    assert report["postentry_return_pnl_execution_price_opened"] is False
    assert report["funding_values_opened"] is False
    assert report["gross9_rows_opened"] is False
    assert {key: value["events"] for key, value in report["support"].items()} == {
        "train": 95, "test": 202, "eval": 176, "final": 92,
    }
    assert report["clock"]["rows"] == 565
    assert sha256(Path(report["clock"]["path"])) == report["clock"]["sha256"]
    assert report["clock"]["sha256"] == "b9a5195f5b3704a75e9e79face964d2d972f18dceee269b9c0a57ddcc5c09b6f"
    assert sha256(RESULT) == "bcf684242b610b78a33baad65516a6893ba34eae182e2ff6a8b0f7240d3909fd"
