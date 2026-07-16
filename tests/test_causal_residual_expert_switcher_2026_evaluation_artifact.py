from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import evaluate_causal_residual_expert_switcher_2026 as evaluator


RESULT = Path("results/causal_residual_expert_switcher_2026_evaluation_2026-07-17.json")
EXPECTED_FILE_SHA256 = "986f78317ee99ec629b32a14379d68d6d5ea216f459488b1c4860aeee1c98f4e"


def test_cres_2026_one_shot_result_is_locked_and_rejected() -> None:
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == EXPECTED_FILE_SHA256
    payload = json.loads(RESULT.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert evaluator.canonical_hash(core) == payload["manifest_hash"]
    assert payload["outcomes_opened"] is True
    assert payload["evaluation_source_sha256"] == "3811b72a64ca03d0e9285dfa593bc621bb1490fe105a48267622e16b1945cc22"
    assert payload["clock_sha256"] == evaluator.CLOCK_SHA256
    assert payload["seed_sha256"] == evaluator.SEED_SHA256
    assert payload["strategy_gate"]["passes"] is False
    assert payload["disposition"] == "retire_cres1_no_2026_repair"


def test_cres_2026_decisions_are_causal_and_metrics_cover_full_h1() -> None:
    payload = json.loads(RESULT.read_text())
    trace = payload["decision_trace"]
    assert len(trace) == payload["decision_counts"]["base_events"] == 68
    assert all(row["decision_materialized_before_outcome"] is True for row in trace)
    assert payload["decision_counts"] == {
        "base_events": 68,
        "executed": 11,
        "continuation": 7,
        "reversion": 4,
        "flat": 57,
    }
    primary = payload["primary"]
    assert primary["h1"]["calendar_start"] == "2026-01-01"
    assert primary["h1"]["calendar_end_exclusive"] == "2026-07-01"
    assert primary["h1"]["absolute_return_pct"] < 0.0
    assert primary["q1"]["absolute_return_pct"] < 0.0
    assert primary["q2"]["absolute_return_pct"] > 0.0
    assert payload["stress_10bp"]["h1"]["absolute_return_pct"] < 0.0
    assert payload["weekly_cluster_signflip"]["raw_p_value"] > 0.05
