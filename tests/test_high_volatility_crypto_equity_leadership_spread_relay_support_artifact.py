import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_crypto_equity_leadership_spread_relay_support_2026-08-13.json")
CLOCK = Path("data/high_volatility_crypto_equity_leadership_spread_relay_clocks_2023_2026.csv.gz")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_terminal_source_rejection_is_sealed_and_outcome_blind() -> None:
    report = json.loads(RESULT.read_text())
    assert sha(RESULT) == "42bcfab019e7f9e232ed8fa7e562ed085638e3cf24360ceec377b705267df72b"
    assert report["policy_id"] == "HVCELSR-24"
    assert not report["support_passed"]
    assert report["decision"] == "terminal_source_support_reject"
    assert not report["advance_to_gross9_novelty"]
    assert not report["advance_to_economic_outcomes"]
    assert not report["postentry_return_pnl_execution_price_opened"]
    assert not report["gross9_rows_opened"]


def test_only_frozen_train_minimum_fails_and_hashes_are_bound() -> None:
    report = json.loads(RESULT.read_text())
    assert {name: values["events"] for name, values in report["support"].items()} == {
        "train": 6, "test": 27, "eval": 14, "final": 11,
    }
    failed = {name for name, passed in report["support_checks"].items() if not passed}
    assert failed == {"train_minimum_events"}
    assert report["clock"] == {
        "path": str(CLOCK),
        "sha256": "f5d0cdcaa1db86aa284dd5dfee6c8d03fad2e43f171d9df53344af2bfcecfda3",
        "rows": 58,
    }
    assert report["source_manifest"]["sha256"] == "b41949625ce0e62904a8833791e49b668549afead8939b94059734e0d1782cac"
    assert not Path(
        "results/high_volatility_crypto_equity_leadership_spread_relay_gross9_novelty_2026-08-13.json"
    ).exists()
