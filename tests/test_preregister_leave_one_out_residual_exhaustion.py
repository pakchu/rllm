from __future__ import annotations

import json

from training.preregister_leave_one_out_residual_exhaustion import (
    canonical_hash,
    protocol,
    run,
)


def test_protocol_freezes_four_policies_and_sequential_holdouts() -> None:
    p = protocol()
    assert [x["policy_id"] for x in p["policies"]] == ["L01", "L02", "L03", "L04"]
    assert p["selection"]["multiple_testing_hypotheses"] == 4
    assert p["holdout_2025"]["opened_only_after_policy_commit"] is True
    assert p["final_2026"]["opened_only_after_2025_pass"] is True
    assert p["evidence_boundary"]["post_entry_returns_opened"] is False


def test_protocol_is_beta_neutral_market_pair_not_btc_gate() -> None:
    p = protocol()
    assert p["signal"]["weight_formula"]["target_factor_beta"] == 0.0
    assert p["signal"]["weight_formula"]["gross"] == 1.0
    assert p["universe"]["trading_target"] == "two-leg alt-perpetual pair; no BTC position"
    assert len(p["universe"]["symbols"]) == 6


def test_hash_is_stable_and_run_writes_matching_artifacts(tmp_path) -> None:
    out = tmp_path / "pre.json"
    doc = tmp_path / "pre.md"
    payload = run(str(out), str(doc))
    loaded = json.loads(out.read_text())
    assert loaded["protocol_hash"] == canonical_hash(loaded["protocol"])
    assert loaded["protocol_hash"] == payload["protocol_hash"]
    assert loaded["protocol_hash"] in doc.read_text()
