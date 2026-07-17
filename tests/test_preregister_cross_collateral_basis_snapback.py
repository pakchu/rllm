from __future__ import annotations

import json

from training.preregister_cross_collateral_basis_snapback import (
    SOURCE_MANIFEST_CONTENT_HASH,
    SOURCE_MANIFEST_FILE_SHA256,
    canonical_hash,
    protocol,
    run,
)


def test_protocol_freezes_outcome_blind_largest_supported_threshold() -> None:
    p = protocol()
    assert p["feature_formula"]["threshold_support_grid"] == [1.5, 2.0, 2.5, 3.0]
    assert p["feature_formula"]["lookback_bars"] == 4032
    assert p["feature_formula"]["minimum_prior_bars"] == 3226
    assert p["support_only_selection"]["threshold_rule"].startswith("select the largest")
    assert p["support_only_selection"]["no_return_or_ohlc_path_statistic_may_break_ties"]
    assert p["support_only_selection"]["selection_period"] == ["2021-01-01", "2023-01-01"]
    assert p["evidence_boundary"]["2023_label"].startswith("development")


def test_protocol_freezes_causal_clock_roll_and_reservation() -> None:
    p = protocol()
    clock = p["signal_and_execution_clock"]
    assert clock["signal_available"].startswith("t+5m")
    assert clock["entry"].startswith("t+10m")
    assert clock["maximum_hold_bars"] == 144
    assert "strictly before delivery_time" in clock["roll_rule"]
    assert "full reserved clock" in clock["overlap"]


def test_protocol_is_relative_value_but_blocks_live_without_collateral_ledger() -> None:
    p = protocol()
    assert p["economic_object"]["initial_leg_gross"] == {"um": 0.5, "cm": 0.5}
    assert p["frozen_policy"]["no_directional_or_regime_gate"] is True
    assert "BTC collateral" in p["collateral_accounting_boundary"]["research_ledger_omits"]
    assert "cannot be promoted" in p["collateral_accounting_boundary"]["consequence"]


def test_protocol_binds_source_and_strict_risk_contract() -> None:
    p = protocol()
    assert p["source_contract"]["manifest_content_hash"] == SOURCE_MANIFEST_CONTENT_HASH
    assert p["source_contract"]["manifest_file_sha256"] == SOURCE_MANIFEST_FILE_SHA256
    assert "excluding manifest_hash" in p["source_contract"]["manifest_content_hash_algorithm"]
    assert p["source_contract"]["no_post_2023_source_before_2023_gate"] is True
    assert p["strict_risk_contract"]["absolute_return_always_reported"] is True
    assert "global/pre-entry" in p["strict_risk_contract"]["hwm"]
    assert p["derivative_ledger"]["base_gross_one_round_trip_cost_bp"] == 12.0
    assert p["strict_risk_contract"]["missing_held_path"].startswith("fail the evaluation")


def test_protocol_requires_hard_2023_gate_without_repairs() -> None:
    p = protocol()
    gate = p["development_2023_gate"]
    assert gate["cagr_to_strict_mdd_at_least"] == 3.0
    assert gate["strict_mdd_at_most_pct"] == 15.0
    assert gate["h1_and_h2_absolute_return_positive"] is True
    assert gate["both_z_sign_branches_absolute_return_positive"] is True
    assert "2^m" in gate["monthly_cluster_signflip"]
    assert "rejection artifact" in p["post_2023_sequence"]["terminal_failure"]
    assert "Do not repair" in p["stop_rule"]


def test_protocol_binds_live_anchor_and_defines_overlap_denominators() -> None:
    gate = protocol()["orthogonality_gate"]
    assert gate["live_anchor"]["weights"] == {
        "oi_upbit_ratio288_low": 0.65,
        "new_long_minimal_funding_premium": 1.75,
        "cand_rex_veto_7": 1.45,
    }
    assert "divided by all CCBS entries" in gate["exact_5m_entry_overlap_share"]
    assert gate["entry_day_jaccard"] == "intersection over union of UTC entry-date sets"
    assert "<year>" in gate["frozen_anchor_entry_clock_artifact"]
    assert "every evaluated" in gate["entry_overlap_scope"]
    assert gate["marginal_portfolio_improvement"]["ccbs_weight"] == 0.25
    assert gate["btc_return_source"]["sha256"]


def test_protocol_freezes_fractional_inverse_research_ledger() -> None:
    ledger = protocol()["derivative_ledger"]
    assert ledger["cm_contract_multiplier_usd"] == 100.0
    assert ledger["cm_contract_rounding"].startswith("none")
    assert ledger["um_quantity_rounding"].startswith("none")
    assert "same contemporaneous cm_mark" in ledger["cm_usd_derivative_pnl"]
    assert "entry fees do not resize" in ledger["entry_sequence"]


def test_hash_is_stable_and_run_writes_matching_artifacts(tmp_path) -> None:
    out, docs = tmp_path / "pre.json", tmp_path / "pre.md"
    payload = run(str(out), str(docs))
    loaded = json.loads(out.read_text())
    assert loaded["protocol_hash"] == canonical_hash(loaded["protocol"])
    assert loaded["protocol_hash"] == payload["protocol_hash"]
    assert loaded["protocol_hash"] in docs.read_text()
