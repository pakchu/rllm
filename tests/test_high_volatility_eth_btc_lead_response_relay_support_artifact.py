import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/high_volatility_eth_btc_lead_response_relay_support_2026-08-10.json"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hveblr_support_pass_is_frozen_blind_reproducible():
    report = json.loads(RESULT.read_text())
    assert report["support_passed"] is True
    assert report["advance_to_gross9_novelty"] is True
    assert report["advance_to_economic_outcomes"] is False
    assert report["postentry_return_pnl_execution_price_opened"] is False
    assert report["funding_values_opened"] is False
    assert report["gross9_rows_opened"] is False
    assert {key: value["events"] for key, value in report["support"].items()} == {
        "train": 43,
        "test": 89,
        "eval": 85,
        "final": 45,
    }
    assert report["clock"]["rows"] == 262
    assert sha256(Path(report["clock"]["path"])) == report["clock"]["sha256"]
    assert report["clock"]["sha256"] == "824b2f21070adf509ba40c3163d687777e908a98b36a2607041dbc9e4aa4228f"
    assert sha256(RESULT) == "c2df9df969983f54621284c280ca5e6499674f39db4c3cb0c29a7bbea73d10ee"
