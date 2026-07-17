from __future__ import annotations

import json
from pathlib import Path

import training.preregister_funding_adjusted_delivery_carry as fadc


def test_protocol_is_canonical_and_market_neutral() -> None:
    body = fadc.protocol()
    assert body["candidate_id"] == "FADC-21"
    assert body["signal"]["direction"] == {
        "carry_gap_positive": "long perpetual, short current-quarter",
        "carry_gap_negative": "short perpetual, long current-quarter",
    }
    assert body["ledger"]["initial_gross"] == 1.0
    assert body["ledger"]["funding_interval"] == (
        "entry_time <= funding_time < exit_time"
    )
    assert body["signal"]["entry_requirements"][
        "minimum_expected_edge_fraction"
    ] == 0.003
    assert fadc.canonical_hash(body) == fadc.canonical_hash(fadc.protocol())


def test_preregistration_writes_no_outcome(tmp_path: Path) -> None:
    output = tmp_path / "prereg.json"
    docs = tmp_path / "prereg.md"
    artifact = fadc.run(output=str(output), docs_output=str(docs))
    loaded = json.loads(output.read_text())
    assert loaded == artifact
    assert artifact["outcome_columns_loaded"] == []
    assert artifact["pnl_opened"] is False
    assert artifact["oos_2024_plus_opened"] is False
    assert artifact["protocol_hash"] == fadc.canonical_hash(artifact["protocol"])
    assert "absolute return is mandatory" in docs.read_text()


def test_support_and_sequential_gates_are_locked() -> None:
    body = fadc.protocol()
    support = body["support_gates_before_pnl"]
    assert support["minimum_entries_total"] == 24
    assert support["minimum_entries_each_year"] == 10
    assert support["minimum_entries_each_half"] == 5
    assert support["maximum_entry_month_share"] == 0.15
    stages = body["sequential_outcome_gates"]
    assert stages["stage_1_2021_2022"]["combined_cagr_to_strict_mdd_minimum"] == 2.0
    assert stages["stage_2_2023_holdout"]["cagr_to_strict_mdd_minimum"] == 3.0
    assert stages["stage_2_2023_holdout"]["minimum_trades"] == 8
    assert "2024" in stages["future_boundary"]
