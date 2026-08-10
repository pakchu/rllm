import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/high_volatility_ticket_elasticity_sponsorship_relay_support_2026-08-10.json"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hvter_source_rejection_is_frozen_blind_and_terminal():
    report = json.loads(RESULT.read_text())
    assert report["support_passed"] is False
    assert report["decision"] == "terminal_source_support_reject"
    assert report["advance_to_gross9_novelty"] is False
    assert report["advance_to_economic_outcomes"] is False
    assert report["postentry_return_pnl_execution_price_opened"] is False
    assert report["funding_values_opened"] is False
    assert report["gross9_rows_opened"] is False
    assert {key: value["events"] for key, value in report["support"].items()} == {
        "train": 16,
        "test": 47,
        "eval": 34,
        "final": 43,
    }
    assert report["support"]["train"]["max_month_share"] == 0.5625
    assert report["support_checks"]["train_month_concentration"] is False
    assert all(
        value for key, value in report["support_checks"].items()
        if key != "train_month_concentration"
    )
    assert sha256(RESULT) == "72cee05ccfdc29d84ca1420a280784b25671e589209aafd7de9b136ed150c06e"


def test_hvter_downstream_stages_remain_sealed():
    assert not Path(
        "results/high_volatility_ticket_elasticity_sponsorship_relay_gross9_novelty_2026-08-10.json"
    ).exists()
    for stage in ("train", "test", "eval", "final"):
        assert not Path(
            f"results/high_volatility_ticket_elasticity_sponsorship_relay_{stage}_economics_2026-08-10.json"
        ).exists()
