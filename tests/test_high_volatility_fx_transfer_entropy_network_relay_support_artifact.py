import hashlib
import json
from pathlib import Path


RESULT = Path("results/high_volatility_fx_transfer_entropy_network_relay_support_2026-08-13.json")
CLOCK = Path("data/high_volatility_fx_transfer_entropy_network_relay_clocks_2023_2026.csv.gz")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_terminal_source_rejection_is_sealed_and_outcome_blind() -> None:
    report = json.loads(RESULT.read_text())
    assert sha(RESULT) == "aed6aadaa24d197c40741d954b07d72bffcc52fdd7cb61bc12847b8ca0682e0f"
    assert report["policy_id"] == "HVFXTE-12"
    assert not report["support_passed"]
    assert report["decision"] == "terminal_source_support_reject"
    assert not report["advance_to_gross9_novelty"]
    assert not report["advance_to_economic_outcomes"]
    assert not report["postentry_return_pnl_execution_price_opened"]
    assert not report["gross9_rows_opened"]


def test_frozen_failures_and_hashes_are_bound() -> None:
    report = json.loads(RESULT.read_text())
    assert {name: values["events"] for name, values in report["support"].items()} == {
        "train": 3,
        "test": 13,
        "eval": 16,
        "final": 6,
    }
    failed = {name for name, passed in report["support_checks"].items() if not passed}
    assert failed == {"train_minimum_events", "train_month_concentration", "final_minimum_events"}
    assert report["clock"] == {
        "path": str(CLOCK),
        "sha256": "841916e40fde4c4e293e1e21b7d50f48510993d5e633910c02574a5388ebd6e5",
        "rows": 38,
    }
    assert report["source_manifests"]["transfer_entropy_fx"]["sha256"] == (
        "299dff6651de912aae5678c18a9c881061eaaed25ee29b8619568270015f27c0"
    )
    assert not Path(
        "results/high_volatility_fx_transfer_entropy_network_relay_gross9_novelty_2026-08-13.json"
    ).exists()
