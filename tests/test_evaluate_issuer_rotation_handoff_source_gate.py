from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from training import evaluate_issuer_rotation_handoff_source_gate as gate


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    return gate.build_report()


def test_source_gate_retires_irh_at_impossible_short_tail(
    report: dict[str, Any],
) -> None:
    assert report["earliest_failure"] == {
        "required_event": "usdt_eth:redeem",
        "strictly_prior_rows_required": 32,
        "minimum_total_rows_for_any_tail_event": 33,
        "observed_total_rows": 3,
        "maximum_possible_short_tail_events": 0,
        "tail_validity_pass": False,
        "short_template_possible": False,
        "side_balance_possible": False,
    }
    assert report["decision"]["status"] == "retired_before_pair_incidence"
    assert report["decision"]["failed_gates"] == [
        "tail_validity",
        "side_balance",
    ]
    assert report["decision"]["repair_authorized"] is False


def test_source_gate_opens_no_pair_comparator_or_outcome_rows(
    report: dict[str, Any],
) -> None:
    boundary = report["outcome_boundary"]
    assert boundary["source_csv_rows_read"] == 0
    assert boundary["pair_incidence_rows_derived"] == 0
    assert boundary["comparator_clock_rows_read"] == 0
    assert boundary["btc_market_rows_read"] == 0
    assert boundary["funding_rows_read"] == 0
    assert boundary["future_return_rows_read"] == 0
    assert report["outcomes_opened"] is False
    assert report["authorization"]["outcome_evaluator"] is False


def test_support_math_requires_33rd_redeem_for_first_tail_event() -> None:
    assert (
        gate._evaluate_support({"usdt_eth:redeem": 32})["tail_validity_pass"] is False
    )
    supported = gate._evaluate_support({"usdt_eth:redeem": 33})
    assert supported["tail_validity_pass"] is True
    assert supported["maximum_possible_short_tail_events"] == 1


def test_validation_rejects_repair_authorization(report: dict[str, Any]) -> None:
    tampered = deepcopy(report)
    tampered["decision"]["repair_authorized"] = True
    core = {key: value for key, value in tampered.items() if key != "manifest_hash"}
    tampered["manifest_hash"] = gate.canonical_hash(core)
    with pytest.raises(RuntimeError, match="decision drift"):
        gate.validate_report(tampered, verify_files=False)


def test_validation_rejects_fabricated_redeem_support(report: dict[str, Any]) -> None:
    tampered = deepcopy(report)
    tampered["source"]["event_counts"]["usdt_eth:redeem"] = 33
    tampered["earliest_failure"] = gate._evaluate_support(
        tampered["source"]["event_counts"]
    )
    core = {key: value for key, value in tampered.items() if key != "manifest_hash"}
    tampered["manifest_hash"] = gate.canonical_hash(core)
    with pytest.raises(RuntimeError, match="decision drift|failed gates drift"):
        gate.validate_report(tampered, verify_files=False)


def test_run_is_byte_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "irh_source_gate.json"
    first = gate.run(output)
    first_bytes = output.read_bytes()
    second = gate.run(output)
    assert first == second
    assert output.read_bytes() == first_bytes
