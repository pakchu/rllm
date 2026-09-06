import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_taker_imbalance_concentration_continuation_support_2026-08-13.json")
CLOCK = Path("data/high_volatility_taker_imbalance_concentration_continuation_clocks_2023_2026.csv.gz")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_pass_is_sealed_and_outcome_blind() -> None:
    report = json.loads(RESULT.read_text())
    assert sha(RESULT) == "7503ee375d978fc11a2c21343553d535d25adedeedcc61bab7fa953295223b20"
    assert report["policy_id"] == "HVTICC-8"
    assert report["support_passed"] and report["decision"] == "pass_to_novelty"
    assert report["advance_to_gross9_novelty"] and not report["advance_to_economic_outcomes"]
    assert not report["postentry_return_pnl_execution_price_opened"]
    assert not report["gross9_rows_opened"]


def test_all_source_gates_and_hashes_pass() -> None:
    report = json.loads(RESULT.read_text())
    assert all(report["support_checks"].values())
    assert {name: values["events"] for name, values in report["support"].items()} == {
        "train": 62,
        "test": 95,
        "eval": 89,
        "final": 51,
    }
    assert report["clock"] == {
        "path": str(CLOCK),
        "sha256": "55e1cc00aee7f8bdb30726ccd101be668278fb3ba4b63b15f71a384eb20e9719",
        "rows": 297,
    }
    assert report["source_manifest"]["sha256"] == "5d08b9de176d949862caa480e85ab80888e933a13c54319f83a75676a4dfb22c"
