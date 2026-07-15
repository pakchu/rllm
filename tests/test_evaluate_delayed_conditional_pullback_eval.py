import inspect
import json
from dataclasses import replace

from training.evaluate_delayed_conditional_pullback_eval import (
    EVAL_END,
    EXPECTED_MANIFEST_HASH,
    Config,
    passes_eval_gate,
    run,
    validate_frozen_manifest,
)
from training.freeze_delayed_conditional_pullback_eval import EVAL_GATE


def _stats() -> dict:
    return {
        name: {
            "absolute_return_pct": 1.0,
            "cagr_to_strict_mdd": 3.1,
            "strict_mdd_pct": 10.0,
            "trades": gate["min_trades"],
        }
        for name, gate in EVAL_GATE.items()
    }


def test_committed_eval_manifest_matches_pin_and_is_sealed():
    manifest = validate_frozen_manifest(Config())
    assert manifest["manifest_hash"] == EXPECTED_MANIFEST_HASH
    assert manifest["eval_opened"] is False


def test_eval_horizon_is_fail_closed():
    for cutoff in ("2026-05-01", "2027-01-01"):
        try:
            validate_frozen_manifest(replace(Config(), exclude_from=cutoff))
        except RuntimeError as error:
            assert "eval horizon" in str(error)
        else:
            raise AssertionError("non-frozen eval cutoff was accepted")
    assert Config.exclude_from == EVAL_END


def test_eval_gate_requires_every_window():
    stats = _stats()
    assert passes_eval_gate(stats)
    for name in stats:
        failed = json.loads(json.dumps(stats))
        failed[name]["cagr_to_strict_mdd"] = 2.99
        assert not passes_eval_gate(failed)


def test_run_replays_2024_prefix_before_opening_eval():
    source = inspect.getsource(run)
    assert source.index("validate_frozen_manifest") < source.index("_replay_through_2024")
    assert source.index("_replay_through_2024") < source.index("build_full_design(cfg)")
    assert "first call permitted to open 2025+" in source
