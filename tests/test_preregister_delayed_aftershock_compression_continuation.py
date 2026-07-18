from __future__ import annotations

import json

import pytest

from training import preregister_delayed_aftershock_compression_continuation as dacc


def test_manifest_is_singleton_and_outcome_blind() -> None:
    payload = dacc.build_manifest()
    dacc.validate_manifest(payload)
    assert payload["outcomes_opened"] is False
    assert payload["selection_protocol"]["candidate_count"] == 1
    assert payload["selection_protocol"]["sealed"] == [
        "2024",
        "2025",
        "2026_ytd",
    ]
    def keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield key
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    frozen_keys = set(keys(payload))
    for forbidden in (
        "absolute_return_pct",
        "cagr_pct",
        "strict_mdd_pct",
        "pnl",
        "trades",
        "win_rate",
    ):
        assert forbidden not in frozen_keys


def test_causal_offsets_leave_a_complete_compression_and_trigger() -> None:
    policy = dacc.Policy()
    assert policy.compression_bars == 6
    assert policy.reacceleration_bars == 3
    assert policy.trigger_offset_bars == (
        policy.compression_bars + policy.reacceleration_bars
    )
    assert policy.entry_offset_bars == policy.trigger_offset_bars + 1
    assert policy.delayed_entry_offset_bars == policy.entry_offset_bars + 1
    assert policy.hold_bars == 48
    assert policy.reference_min_periods < policy.reference_bars


def test_write_once_refuses_policy_mutation(tmp_path) -> None:
    output = tmp_path / "freeze.json"
    payload = dacc.build_manifest()
    assert dacc.write_once(output, payload) == "created"
    assert dacc.write_once(output, payload) == "verified_existing"
    changed = json.loads(output.read_text())
    changed["policy"]["hold_bars"] = 25
    output.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="hash mismatch"):
        dacc.write_once(output, payload)
