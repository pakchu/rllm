from __future__ import annotations

import json
from pathlib import Path

from training import preregister_premium_compression_breakout_relay as prereg


def test_policy_is_single_fixed_causal_candidate() -> None:
    policy = prereg.Policy()
    assert policy.policy_id == "PCBR-12"
    assert policy.context_bars_5m == 24
    assert policy.trigger_bars_5m == 2
    assert policy.prior_nonoverlap_shift_bars_5m == 26
    assert policy.entry_delay_bars_5m == 2
    assert policy.hold_bars_5m == 12
    assert policy.leverage == 0.5


def test_manifest_keeps_execution_outcomes_closed() -> None:
    report = prereg.build_manifest()
    prereg.validate_manifest(report, verify_feature_source=False)
    history = report["research_history_boundary"]
    assert history["exact_pcbr_post_entry_outcomes_opened"] is False
    assert history["candidate_count"] == 1
    assert history["threshold_grid"] is False
    assert history["direction_search"] is False
    assert history["hold_search"] is False
    source = report["source_contract"]
    assert source["execution_sources_may_not_be_opened_by_support_builder"] is True
    assert "BTCUSDT_price_or_return" in report["causal_feature_contract"][
        "forbidden_features"
    ]


def test_manifest_hash_and_write_are_deterministic(tmp_path: Path) -> None:
    first = prereg.build_manifest()
    second = prereg.build_manifest()
    assert first == second
    assert first["manifest_hash"] == prereg.canonical_hash(
        {key: value for key, value in first.items() if key != "manifest_hash"}
    )
    output = tmp_path / "prereg.json"
    written = prereg.write_manifest(output)
    assert json.loads(output.read_text(encoding="utf-8")) == written


def test_support_and_outcome_boundaries_are_frozen() -> None:
    report = prereg.build_manifest()
    support = report["support_gate"]
    assert support["minimum_events"] == {"train": 180, "test": 60, "eval": 120}
    assert support["minimum_each_side_share"] == 0.25
    outcome = report["outcome_gate"]
    assert outcome["cagr_to_strict_mdd_min"] == 3.0
    assert outcome["strict_mdd_max_pct"] == 15.0
    assert outcome["stress_cagr_to_strict_mdd_min"] == 2.5
    assert outcome["sequential_opening"] == "train_then_test_then_eval_stop_on_first_failure"
