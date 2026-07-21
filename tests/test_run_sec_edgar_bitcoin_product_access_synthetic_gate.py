from __future__ import annotations

from dataclasses import replace

import pytest

import training.run_sec_edgar_bitcoin_product_access_synthetic_gate as runner
from training.run_sec_edgar_bitcoin_product_access_synthetic_gate import (
    Config,
    _validate_config,
    evaluate_records,
    frozen_cases,
    validate_preregistration,
)


def _record(case: dict[str, object]) -> dict[str, object]:
    guarded = bool(case["guarded"])
    return {
        "name": case["name"],
        "case_hash": case["case_hash"],
        "guarded": guarded,
        "parsed_ok": True,
        "expected_match": True,
        "quote_valid": True,
        "guard_correct": True,
        "equivalence_group": case.get("equivalence_group"),
        "redacted_excerpt": case["redacted_excerpt"],
        "actual_class": ("UNSUPPORTED" if guarded else str(case["expected_class"])),
    }


def _passing_records() -> list[dict[str, object]]:
    return [_record(case) for case in frozen_cases()]


def _runtime(
    *, allocated: int = 6 * 1024**3, reserved: int = 7 * 1024**3
) -> dict[str, int]:
    return {
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
    }


def test_preregistration_anchor_authorizes_only_synthetic_gate() -> None:
    preregistration = validate_preregistration()
    decision = preregistration["decision"]
    assert decision["synthetic_model_gate_authorized"] is True
    assert decision["filing_body_transport_authorized"] is False
    assert decision["historical_semantic_execution_authorized"] is False
    assert decision["economic_evaluation_authorized"] is False


def test_frozen_cases_have_exact_guard_and_equivalence_contract() -> None:
    cases = frozen_cases()
    assert len(cases) == 24
    guarded = [row for row in cases if row["guarded"]]
    assert len(guarded) == 2
    assert all(row["guard_match"] for row in guarded)
    assert not any(row["guard_match"] for row in cases if not row["guarded"])
    equivalents = [
        row["redacted_excerpt"]
        for row in cases
        if row.get("equivalence_group") == "entity_product_date_amount_swap"
    ]
    assert len(equivalents) == 2
    assert equivalents[0] == equivalents[1]


def test_preregistration_rejects_live_synthetic_constant_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "SYNTHETIC_CASES", ({"name": "tampered"},))
    with pytest.raises(ValueError, match="live synthetic constants drifted"):
        runner.validate_preregistration()


def test_evaluation_passes_only_complete_exact_semantic_and_memory_gate() -> None:
    records = _passing_records()
    result = evaluate_records(records, _runtime())
    assert result["passed"] is True
    assert result["counts"]["cases"] == 24
    assert result["counts"]["model_calls"] == 22
    assert all(result["checks"].values())

    wrong_class = [dict(row) for row in records]
    wrong_class[0]["expected_match"] = False
    assert evaluate_records(wrong_class, _runtime())["passed"] is False

    wrong_manifest = [dict(row) for row in records]
    wrong_manifest[0]["case_hash"] = "0" * 64
    with pytest.raises(ValueError, match="manifest differs"):
        evaluate_records(wrong_manifest, _runtime())

    over_allocated = _runtime(allocated=7 * 1024**3 + 1)
    assert evaluate_records(records, over_allocated)["passed"] is False

    over_reserved = _runtime(reserved=int(7.25 * 1024**3) + 1)
    assert evaluate_records(records, over_reserved)["passed"] is False


def test_synthetic_config_is_frozen() -> None:
    cfg = Config()
    _validate_config(cfg)
    with pytest.raises(ValueError, match="configuration is frozen"):
        _validate_config(replace(cfg, maximum_new_tokens=32))
