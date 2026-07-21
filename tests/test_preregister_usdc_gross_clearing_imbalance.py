from __future__ import annotations

from dataclasses import replace

import pytest

from training import preregister_usdc_gross_clearing_imbalance as ugci


def test_frozen_policy_has_exact_causal_clock_and_execution() -> None:
    cfg = ugci.FROZEN_CONFIG
    assert cfg.event_clock == "available_at"
    assert cfg.packet_hours == 6
    assert cfg.include_zero_event_packets is True
    assert cfg.lookback_days == 180
    assert cfg.minimum_history_packets == 360
    assert cfg.gross_tail_quantile == 0.95
    assert cfg.minimum_imbalance_ratio == 0.60
    assert cfg.entry_delay_minutes == 10
    assert cfg.hold_bars * cfg.bar_minutes == 24 * 60
    assert cfg.global_nonoverlap is True


def test_config_repair_is_rejected() -> None:
    with pytest.raises(ValueError, match="configuration is frozen"):
        ugci._validate_config(replace(ugci.FROZEN_CONFIG, minimum_imbalance_ratio=0.50))


def test_preregistration_hashes_only_and_opens_no_outcome() -> None:
    payload = ugci.preregistration_payload()
    assert payload["candidate"] == "UGCI-288"
    assert payload["source"]["rows_parsed_during_preregistration"] == 0
    assert payload["source"]["numeric_fields_parsed_during_preregistration"] == 0
    assert payload["outcome_boundary"] == {
        "outcomes_opened": False,
        "outcome_sources_opened": False,
        "post_2023_source_rows_opened": False,
        "btc_market_rows_read": 0,
        "funding_rows_read": 0,
        "future_return_rows_read": 0,
        "return_or_pnl_fields_read": 0,
        "comparator_rows_read": 0,
        "network_calls": 0,
        "subprocess_calls": 0,
    }
    assert payload["manifest_hash"] == ugci.canonical_hash(
        {key: value for key, value in payload.items() if key != "manifest_hash"}
    )


def test_comparator_set_and_stop_rule_are_frozen() -> None:
    payload = ugci.preregistration_payload()
    members = {
        (item["candidate"], control)
        for item in payload["novelty_comparators"]
        for control in item["controls"]
    }
    assert members == {
        ("AMTR-48", "primary"),
        ("AMTR-48", "cross_minter"),
        ("SQFD-6", "primary"),
        ("SQFD-6", "no_usdt_lag"),
        ("SQFD-6", "no_participation"),
        ("SDDR-12", "primary"),
        ("UCBR-12", "primary"),
    }
    assert payload["support_gate"]["stop_if_failed"] is True
    assert payload["support_gate"]["minimum_common_candidate_events"] == 10
    assert payload["support_gate"]["minimum_common_comparator_events"] == 5
    for comparator in payload["novelty_comparators"]:
        assert comparator["comparison_end_exclusive"] == "2024-01-01T00:00:00Z"
    assert payload["one_way_sequence"]["failure_action"] == (
        "retire UGCI-288 without repair"
    )


def test_rllm_cannot_repair_or_retime_the_base_clock() -> None:
    payload = ugci.preregistration_payload()
    boundary = payload["rllm_boundary"]
    assert boundary["allowed_only_after_deterministic_economics_pass"] is True
    assert boundary["may_abstain_or_size_only_under_separate_freeze"] is True
    assert boundary["may_create_retime_reverse_or_repair_events"] is False


def test_preregistration_artifact_is_write_once(tmp_path) -> None:
    destination = tmp_path / "freeze.json"
    written = ugci.write_payload(destination)
    assert written["candidate"] == "UGCI-288"
    with pytest.raises(FileExistsError, match="write-once"):
        ugci.write_payload(destination)
