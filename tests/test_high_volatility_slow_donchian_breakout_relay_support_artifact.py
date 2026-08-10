import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/high_volatility_slow_donchian_breakout_relay_support_2026-08-10.json"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hvsdbr_support_pass_is_frozen_blind_reproducible():
    report = json.loads(RESULT.read_text())
    assert report["support_passed"] is True
    assert report["advance_to_gross9_novelty"] is True
    assert report["advance_to_economic_outcomes"] is False
    assert report["postentry_return_pnl_execution_price_opened"] is False
    assert report["funding_values_opened"] is False
    assert report["gross9_rows_opened"] is False
    assert {key: value["events"] for key, value in report["support"].items()} == {
        "train": 14,
        "test": 48,
        "eval": 41,
        "final": 17,
    }
    assert report["clock"]["rows"] == 120
    assert sha256(Path(report["clock"]["path"])) == report["clock"]["sha256"]
    assert report["clock"]["sha256"] == "4202a594e478d3e5ebaa03f5a38af502890ffc8e0f965bdcbc463702d8b7e389"
    assert sha256(RESULT) == "f2fcf52885d20524150a6e86358502a062193c52ebdc39aaf9d38185d7250412"
