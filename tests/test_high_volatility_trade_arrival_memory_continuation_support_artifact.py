import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_trade_arrival_memory_continuation_support_2026-08-10.json")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_support_pass_is_frozen_blind_and_reproducible():
    artifact = json.loads(RESULT.read_text())
    assert artifact["support_passed"] is True
    assert artifact["advance_to_gross9_novelty"] is True
    assert artifact["advance_to_economic_outcomes"] is False
    assert artifact["postentry_return_pnl_execution_price_opened"] is False
    assert artifact["funding_values_opened"] is False
    assert artifact["gross9_rows_opened"] is False
    assert {key: value["events"] for key, value in artifact["support"].items()} == {
        "train": 65, "test": 140, "eval": 117, "final": 49,
    }
    assert artifact["clock"]["rows"] == 371
    assert sha(Path(artifact["clock"]["path"])) == artifact["clock"]["sha256"] == "c09ee3a0a38a5044c915e3a8fa304de724e9c8dcd8aa6961c32de21e7d519440"
    assert sha(RESULT) == "2f588ed548f4083ac834193163acb60f9ee71f9ce74acadbb744f5d87ccf3c05"
