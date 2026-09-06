import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/high_volatility_range_turnover_coupling_relay_support_2026-08-10.json"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hvrtcr_support_pass_is_frozen_blind_reproducible():
    report = json.loads(RESULT.read_text())
    assert report["support_passed"] is True
    assert report["advance_to_gross9_novelty"] is True
    assert report["advance_to_economic_outcomes"] is False
    assert report["postentry_return_pnl_execution_price_opened"] is False
    assert report["funding_values_opened"] is False
    assert report["gross9_rows_opened"] is False
    assert {key: value["events"] for key, value in report["support"].items()} == {
        "train": 42,
        "test": 89,
        "eval": 104,
        "final": 45,
    }
    assert report["clock"]["rows"] == 280
    assert sha256(Path(report["clock"]["path"])) == report["clock"]["sha256"]
    assert report["clock"]["sha256"] == "29de599c631e03a84155b84ea50915704533b42549e7031989ef3bc35c65b69c"
    assert sha256(RESULT) == "50d804e9be2095d341a4104dc5ef1dda57a0d72907cb320def111b949444ed07"
