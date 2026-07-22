from __future__ import annotations

import json
from pathlib import Path

import pytest
from Crypto.Hash import keccak

from training import preregister_wbtc_custody_bridge_flow_source as protocol


RESULT_PATH = Path(
    "results/wbtc_custody_bridge_flow_source_protocol_2026-07-23.json"
)


def _keccak_topic(signature: str) -> str:
    digest = keccak.new(digest_bits=256)
    digest.update(signature.encode("ascii"))
    return "0x" + digest.hexdigest()


def test_frozen_contract_topics_and_envelope_are_exact() -> None:
    assert protocol.WBTC_ADDRESS == "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599"
    assert protocol.WBTC_DECIMALS == 8
    assert protocol.MINT_TOPIC == _keccak_topic("Mint(address,uint256)")
    assert protocol.BURN_TOPIC == _keccak_topic("Burn(address,uint256)")
    assert protocol.TRANSFER_TOPIC == _keccak_topic(
        "Transfer(address,address,uint256)"
    )
    assert protocol.ZERO_TOPIC == "0x" + "00" * 32
    assert protocol.START_BOUNDARY_BLOCK == 9_193_266
    assert protocol.END_BOUNDARY_BLOCK == 18_908_895
    assert protocol.LAST_SOURCE_BLOCK == 18_908_894
    assert protocol.CONFIRMATION_BLOCKS == 64
    assert protocol.MAX_BLOCK_RANGE == 10_000


def test_protocol_is_source_only_and_forbids_direction_selection() -> None:
    payload = protocol.build_protocol()
    assert payload["outcome_boundary"] == {
        "source_only": True,
        "full_source_incidence_opened": False,
        "mechanism_features_opened": False,
        "btc_market_rows_read": 0,
        "funding_rows_read": 0,
        "future_return_rows_read": 0,
        "labels_opened": False,
        "pnl_cagr_mdd_opened": False,
        "post_2023_contract_event_rows_read": 0,
        "post_2023_confirmation_headers_may_be_read": True,
    }
    assert payload["later_mechanism_boundary"] == {
        "standalone_mint_long_burn_short_authorized": False,
        "standalone_reversed_direction_authorized": False,
        "weak_factor_combinations_must_be_preregistered": True,
        "multiplicity_control_required": True,
        "llm_direction_selection_authorized": False,
    }
    assert payload["availability_contract"]["effective_output_field"] == (
        "available_at"
    )


def test_integrity_contract_requires_dual_replay_and_receipt_pairing() -> None:
    payload = protocol.build_protocol()
    integrity = payload["integrity_contract"]
    assert integrity["independent_log_replays"] == 2
    assert integrity["semantic_event_zero_transfer_pair_required"] is True
    assert integrity["semantic_event_and_transfer_double_count_allowed"] is False
    assert integrity["receipt_success_required"] is True
    assert integrity["removed_logs_allowed"] is False
    assert payload["source_promotion_gates"]["any_failure_effect"].startswith(
        "REJECT_NO_REPAIR"
    )
    source_text = json.dumps(payload["source_contract"], sort_keys=True)
    assert "rpc_url" not in source_text
    assert "eth.drpc" not in source_text
    assert "mevblocker" not in source_text


def test_bounded_probe_is_dual_transport_and_outcome_blind() -> None:
    payload = protocol.build_protocol()
    probe = payload["bounded_probe"]
    assert probe["transports"] == 2
    assert probe["receipt_zero_transfer_pairs_verified"] == 4
    assert probe["btc_market_or_outcomes_opened"] is False
    assert [row["rows"] for row in probe["ranges"]] == [1, 1, 1, 1]
    assert len({row["canonical_sha256"] for row in probe["ranges"]}) == 4


def test_decision_binding_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(protocol, "DECISION_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="decision hash changed"):
        protocol.build_protocol()


def test_generated_protocol_matches_committed_artifact(tmp_path: Path) -> None:
    generated = protocol.run(tmp_path / "protocol.json")
    committed = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert generated == committed
    core = {key: value for key, value in committed.items() if key != "manifest_hash"}
    assert committed["manifest_hash"] == protocol.canonical_hash(core)
    assert committed["decision_binding"] == {
        "path": str(protocol.DECISION_PATH),
        "sha256": protocol.DECISION_SHA256,
    }
    assert committed["implementation_binding"]["sha256"] == protocol.sha256_file(
        protocol.SCRIPT_PATH
    )
