from __future__ import annotations

import json

import pytest

from training import audit_usdc_role_topology as topology


A = "0x" + "11" * 20
B = "0x" + "22" * 20
C = "0x" + "33" * 20
D = "0x" + "44" * 20


def _event(
    event: str,
    actor: str,
    *,
    recipient: str = "",
    amount: int = 100,
    year: str = "2023",
) -> topology.RoleEvent:
    return topology.RoleEvent(event, actor, recipient, amount, year)


def test_directed_topology_is_aggregate_and_address_free() -> None:
    events = [
        _event("mint", A, recipient=B, amount=100, year="2021"),
        _event("mint", C, recipient=B, amount=300, year="2022"),
        _event("mint", A, recipient=D, amount=100, year="2023"),
        _event("burn", B, amount=200, year="2022"),
        _event("burn", B, amount=200, year="2023"),
        _event("burn", C, amount=100, year="2023"),
    ]

    audit = topology.audit_events(events)
    graph = audit["directed_recipient_burner_topology"]
    assert audit["role_overlap"] == {
        "mint_caller_and_burn_caller": 1,
        "mint_recipient_and_burn_caller": 1,
        "mint_recipient_and_mint_caller": 0,
        "all_three": 0,
    }
    assert graph["recipient_burner_roles"] == 1
    assert graph["distinct_mint_callers_into_roles"] == 2
    assert graph["distinct_minter_recipient_edges"] == 2
    assert graph["mint_leg_events"] == 2
    assert graph["mint_leg_event_share"] == pytest.approx(2 / 3)
    assert graph["mint_leg_amount_share"] == pytest.approx(0.8)
    assert graph["burn_leg_events"] == 2
    assert graph["burn_leg_event_share"] == pytest.approx(2 / 3)
    assert graph["burn_leg_amount_share"] == pytest.approx(0.8)
    encoded = json.dumps(audit, sort_keys=True)
    assert all(address not in encoded for address in (A, B, C, D))


def test_full_period_role_membership_is_never_authorized_as_feature() -> None:
    audit = topology.audit_events([_event("mint", A, recipient=B), _event("burn", B)])
    interpretation = audit["interpretation"]
    assert interpretation["directed_graph_exists"] is True
    assert interpretation["full_period_membership_is_descriptive_only"] is True
    assert interpretation["future_membership_authorized_for_features"] is False
    assert interpretation["pair_incidence_calculated"] is False
    assert interpretation["economic_support_established"] is False


def test_unknown_event_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="unexpected role event"):
        topology.audit_events([_event("transfer", A, recipient=B)])


def test_source_contract_requires_replay_headers_n64_and_source_only() -> None:
    valid = {
        "protocol_version": "ethereum_stablecoin_issuance_redemption_source_v1",
        "output": {"sha256": topology.SOURCE_CSV_SHA256, "rows": 266_362},
        "dual_replay": {"canonical_replay_equal": True},
        "header_materialization": {"event_block_hash_cross_checked": True},
        "source_contract": {"confirmation_blocks": 64},
        "source_audit": {
            "finalized_coverage": {"observed_finalized_block_at_least_required": True}
        },
        "outcome_boundary": {"source_only": True},
    }
    topology._validate_source_manifest(valid)
    invalid = json.loads(json.dumps(valid))
    invalid["source_contract"]["confirmation_blocks"] = 63
    with pytest.raises(RuntimeError, match=r"N\+64"):
        topology._validate_source_manifest(invalid)
