from __future__ import annotations

import json

from training.preregister_alt_funding_carry_harvest import canonical_hash, protocol, run


def test_protocol_freezes_single_market_neutral_carry_policy() -> None:
    p = protocol()
    assert p["frozen_policy"]["policy_id"] == "AFCH01"
    assert p["frozen_policy"]["sleeve_gross"] == 0.25
    assert p["frozen_policy"]["hold_days"] == 28
    assert p["feature_formula"]["minimum_projected_28d_carry"] == 0.0018
    assert p["feature_formula"]["normalized_weights"]["target_factor_beta"] == 0.0


def test_protocol_requires_funding_to_pay_costs_and_forward_shadow() -> None:
    p = protocol()
    assert p["selection_2023_2024"]["realized_funding_cash_at_least_transaction_cost"] is True
    assert p["eval_2025"]["realized_funding_cash_at_least_transaction_cost"] is True
    assert p["final_2026_and_forward"]["minimum_forward_shadow_days_for_promotion"] == 90
    assert p["evidence_boundary"]["forward_shadow_required_for_promotion"] is True


def test_protocol_forbids_price_and_regime_repairs() -> None:
    p = protocol()
    assert p["frozen_policy"]["no_price_direction_or_regime_gate"] is True
    assert "No funding lookback, carry hurdle, hold" in p["stop_rule"]
    assert p["source_contract"]["no_2026_source_before_2023_2025_pass"] is True


def test_hash_is_stable_and_run_writes_matching_artifacts(tmp_path) -> None:
    out, docs = tmp_path / "pre.json", tmp_path / "pre.md"
    payload = run(str(out), str(docs))
    loaded = json.loads(out.read_text())
    assert loaded["protocol_hash"] == canonical_hash(loaded["protocol"])
    assert loaded["protocol_hash"] == payload["protocol_hash"]
    assert loaded["protocol_hash"] in docs.read_text()
