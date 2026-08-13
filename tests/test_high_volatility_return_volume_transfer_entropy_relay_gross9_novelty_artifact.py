import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/high_volatility_return_volume_transfer_entropy_relay_gross9_novelty_2026-08-13.json"
)


def test_gross9_pass_is_sealed_and_economically_blind() -> None:
    report = json.loads(RESULT.read_text())
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == (
        "8c1e2c0151c40b9290a6062b4891d9e5314448e016d7b737c9e090b44f1ad3c7"
    )
    assert report["policy_id"] == "HVRVTE-8"
    assert report["every_gross9_sleeve_passed"]
    assert report["gross9_novelty_status"] == "passed"
    assert report["advance_to_economic_outcomes"]
    assert all(value["passed"] for value in report["gross9_sleeves"].values())
    boundary = report["evidence_boundary"]
    assert boundary["candidate_clock_rows_opened"] == 213
    assert boundary["gross9_structural_clock_rows_opened"] == 1064
    assert boundary["btc_price_or_return_rows_opened"] == 0
    assert boundary["funding_rows_opened"] == 0
    assert boundary["economic_outcome_rows_opened"] == 0
    assert not boundary["outcomes_opened"]


def test_worst_metrics_remain_inside_frozen_limits() -> None:
    report = json.loads(RESULT.read_text())
    metrics = [value["metrics"] for value in report["gross9_sleeves"].values()]
    assert max(item["exact_entry_jaccard"] for item in metrics) == 0.007407407407407408
    assert max(item["one_to_one_6h_max_matched_share"] for item in metrics) == 0.14583333333333334
    assert max(item["occupied_5m_bar_jaccard"] for item in metrics) == 0.04581757040773434
    assert max(abs(item["absolute_signed_exposure_pearson"]) for item in metrics) == 0.03794657229784656
