from __future__ import annotations

from dataclasses import replace

import pytest

from training.run_sec_edgar_bitcoin_constraint_synthetic_gate import (
    Config,
    _validate_config,
    evaluate_records,
    frozen_cases,
    validate_preregistration,
)


def _record(
    name: str,
    *,
    guarded: bool = False,
    label: str = "BTC_CONSTRAINT_DRAW",
    role: str = "BTC_SALE",
    group: str | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "guarded": guarded,
        "parsed_ok": True,
        "expected_match": True,
        "quote_valid": True,
        "guard_correct": True,
        "equivalence_group": group,
        "redacted_excerpt": "same" if group else name,
        "actual_label": "UNSUPPORTED" if guarded else label,
        "actual_role": "NONE" if guarded else role,
    }


def test_preregistration_anchor_authorizes_only_synthetic_gate() -> None:
    prereg = validate_preregistration()
    decision = prereg["decision"]
    assert decision["synthetic_model_gate_authorized"] is True
    assert decision["filing_body_transport_authorized"] is False
    assert decision["economic_evaluation_authorized"] is False


def test_frozen_cases_have_exact_guard_and_equivalence_contract() -> None:
    cases = frozen_cases()
    assert len(cases) == 17
    guarded = [row for row in cases if row["guarded"]]
    assert len(guarded) == 2
    assert all(row["guard_match"] for row in guarded)
    assert not any(row["guard_match"] for row in cases if not row["guarded"])
    equivalents = [
        row["redacted_excerpt"]
        for row in cases
        if row.get("equivalence_group") == "entity_date_amount_swap"
    ]
    assert len(equivalents) == 2
    assert equivalents[0] == equivalents[1]


def test_evaluate_records_passes_only_complete_exact_battery() -> None:
    records = [_record(f"case_{index}") for index in range(13)]
    records.extend(
        [
            _record("guard_a", guarded=True),
            _record("guard_b", guarded=True),
            _record("entity_a", label="BTC_CONSTRAINT_BUFFER", role="BTC_RETENTION", group="swap"),
            _record("entity_b", label="BTC_CONSTRAINT_BUFFER", role="BTC_RETENTION", group="swap"),
        ]
    )
    result = evaluate_records(records)
    assert result["passed"] is True
    assert result["counts"]["cases"] == 17
    assert result["counts"]["model_calls"] == 15
    failed = [dict(row) for row in records]
    failed[0]["expected_match"] = False
    assert evaluate_records(failed)["passed"] is False


def test_synthetic_config_is_frozen() -> None:
    cfg = Config()
    _validate_config(cfg)
    with pytest.raises(ValueError, match="configuration is frozen"):
        _validate_config(replace(cfg, maximum_new_tokens=32))
