from __future__ import annotations

import json

from training.preregister_leave_one_out_residual_continuation import (
    canonical_hash,
    protocol,
    run,
)


def test_protocol_freezes_one_policy_before_2025() -> None:
    p = protocol()
    assert p["frozen_policy"]["policy_id"] == "LORC01"
    assert p["frozen_policy"]["residual_horizon_hours"] == 12
    assert p["frozen_policy"]["hold_hours"] == 12
    assert p["evidence_boundary"]["holdout_2025_post_entry_returns_opened"] is False
    assert p["holdout_2025"]["single_confirmatory_hypothesis"] is True


def test_policy_is_factor_neutral_continuation_without_btc_position() -> None:
    p = protocol()
    policy = p["frozen_policy"]
    assert policy["long_leg"] == "winner"
    assert policy["short_leg"] == "loser"
    assert policy["weight_formula"]["target_factor_beta"] == 0.0
    assert policy["weight_formula"]["gross"] == 1.0
    assert p["universe"]["trading_target"] == "two-leg alt-perpetual pair; no BTC position"


def test_protocol_discloses_contaminated_research_and_strict_stop_rule() -> None:
    p = protocol()
    assert p["evidence_boundary"]["research_2023_2024_contaminated_for_confirmation"] is True
    assert p["hypothesis_origin"]["no_additional_2023_2024_threshold_or_pair_search"] is True
    assert p["holdout_2025"]["cagr_to_strict_mdd_at_least"] == 3.0
    assert p["holdout_2025"]["strict_mdd_at_most_pct"] == 15.0
    assert "No threshold, direction, hold" in p["stop_rule"]


def test_hash_is_stable_and_run_writes_matching_artifacts(tmp_path) -> None:
    out = tmp_path / "pre.json"
    doc = tmp_path / "pre.md"
    payload = run(str(out), str(doc))
    loaded = json.loads(out.read_text())
    assert loaded["protocol_hash"] == canonical_hash(loaded["protocol"])
    assert loaded["protocol_hash"] == payload["protocol_hash"]
    assert loaded["protocol_hash"] in doc.read_text()
