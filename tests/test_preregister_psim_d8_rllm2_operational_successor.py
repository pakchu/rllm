from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training import preregister_psim_d8_rllm2_operational_successor as prereg


RESULT = prereg.REPO_ROOT / prereg.DEFAULT_OUTPUT
RESULT_SHA256 = (
    "85ede8a56393b11f4f1ced7e304adb3c2639132c1f0b008ed973aae92af9ef54"
)
MANIFEST_HASH = (
    "c9b8a7527d90e8de3b1aeadac834c4b9d7a97bc3358c08256f79fa24fc18266c"
)
SCIENTIFIC_CONTRACT_HASH = (
    "59a7c1dd03155d8552614e4886087ca1dd08db4cc8c8953257c2f6f68d28af23"
)


def test_committed_preregistration_is_exact_and_canonical() -> None:
    payload = prereg.build_preregistration()
    assert json.loads(RESULT.read_text(encoding="utf-8")) == payload
    assert RESULT.read_bytes() == prereg.canonical_json_bytes(
        payload,
        pretty=True,
    )
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == RESULT_SHA256
    core = {
        key: value
        for key, value in payload.items()
        if key != "manifest_hash"
    }
    assert payload["manifest_hash"] == prereg.canonical_hash(core)
    assert payload["manifest_hash"] == MANIFEST_HASH


def test_preregistration_reads_only_three_predecessor_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    original = Path.read_bytes

    def audited_read_bytes(path: Path) -> bytes:
        try:
            relative = path.resolve().relative_to(
                prereg.REPO_ROOT
            ).as_posix()
        except ValueError:
            relative = path.as_posix()
        observed.append(relative)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", audited_read_bytes)
    payload = prereg.build_preregistration()
    assert set(observed) == set(payload["access_boundary"]["files_read"])
    assert set(observed) == {
        prereg.RLLM1_PREREGISTRATION.as_posix(),
        prereg.RLLM1_ATTEMPT.as_posix(),
        prereg.RLLM1_FAILURE.as_posix(),
    }
    boundary = payload["access_boundary"]
    assert boundary["market_or_funding_paths_read"] == []
    assert boundary["market_rows_parsed"] == 0
    assert boundary["funding_rows_parsed"] == 0
    assert boundary["market_or_funding_payload_bytes_hashed"] is False
    assert boundary["model_loaded"] is False
    assert boundary["model_outputs_created"] == 0
    assert boundary["rewards_created"] == 0
    assert boundary["economic_metrics_computed"] == 0
    assert boundary["test_outcomes_opened"] is False
    assert boundary["eval_outcomes_opened"] is False


def test_predecessor_is_terminal_before_any_forward_or_market() -> None:
    payload = prereg.build_preregistration()
    predecessor = payload["predecessor_terminal_evidence"]
    assert predecessor["execution_commit"] == (
        "ce9ba77782ff0cc34411d60dc1ba7def5bea707f"
    )
    assert predecessor["terminal_record_commit"] == (
        "8ec8d4711900f405a206b1980a51fdcd582a1415"
    )
    assert predecessor["runner_sha256"] == prereg.RLLM1_RUNNER_SHA256
    assert predecessor["attempt"]["sha256"] == (
        prereg.RLLM1_ATTEMPT_SHA256
    )
    assert predecessor["attempt"]["attempt_hash"] == (
        prereg.RLLM1_ATTEMPT_HASH
    )
    assert predecessor["failure"]["sha256"] == (
        prereg.RLLM1_FAILURE_SHA256
    )
    assert predecessor["failure"]["result_hash"] == (
        prereg.RLLM1_FAILURE_RESULT_HASH
    )
    assert predecessor["model_forwards_started"] == 0
    assert predecessor["challenge_predictions_created"] == 0
    assert predecessor["market_or_economic_payload_opened"] is False
    assert predecessor["predecessor_rerun_authorized"] is False


def test_scientific_contract_is_inherited_without_resample_or_threshold_change() -> None:
    payload = prereg.build_preregistration()
    inherited = payload["inherited_scientific_contract"]
    assert inherited["contract_hash"] == SCIENTIFIC_CONTRACT_HASH
    assert inherited["case_roster_hash"] == prereg.CASE_ROSTER_HASH
    assert inherited["rllm1_preregistration"] == {
        "path": prereg.RLLM1_PREREGISTRATION.as_posix(),
        "sha256": prereg.RLLM1_PREREGISTRATION_SHA256,
        "manifest_hash": prereg.RLLM1_PREREGISTRATION_MANIFEST_HASH,
    }
    base = json.loads(
        (prereg.REPO_ROOT / prereg.RLLM1_PREREGISTRATION).read_text(
            encoding="utf-8"
        )
    )
    inherited_payload = inherited["payload"]
    for key, value in inherited_payload.items():
        assert value == base[key]
    challenge = inherited_payload["memorization_contract"]
    assert challenge["capacity"]["combined_selected_challenge_events"] == 128
    assert challenge["exact_one_sided_binomial_chance"] == 0.125
    assert challenge["bonferroni_reject_p_below"] == pytest.approx(
        0.01 / 3.0
    )
    assert inherited_payload["model_contract"]["id"] == (
        "google/gemma-4-E4B-it"
    )
    delta = payload["sole_operational_delta"]
    assert delta["model_or_data_change"] is False
    assert delta["challenge_or_threshold_change"] is False
    assert delta["source_resample"] is False
    assert delta["prompt_or_candidate_change"] is False
    assert delta["market_or_outcome_information_used"] is False


def test_rllm2_has_new_fixed_one_shot_paths_and_no_output_override() -> None:
    payload = prereg.build_preregistration()
    one_shot = payload["one_shot_execution"]
    assert one_shot == {
        "attempt_path": (
            "results/psim_d8_rllm2_base_memorization_gate_"
            "attempt_2026-07-27.json"
        ),
        "result_path": (
            "results/psim_d8_rllm2_base_memorization_gate_2026-07-27.json"
        ),
        "clean_head_equals_origin_main_required": True,
        "attempt_created_before_weight_load": True,
        "attempt_consumed_on_any_post_sentinel_failure": True,
        "output_override_allowed": False,
        "validate_only_loads_weights": False,
    }
    assert one_shot["attempt_path"] != (
        prereg.RLLM1_ATTEMPT.as_posix()
    )
    assert payload["terminal_actions"]["operational_or_memorization_failure"] == (
        "REJECT_PSIM_D8_RLLM2_BEFORE_NEXT_MARKET_STAGE_NO_REPAIR_"
        "RESAMPLE_MODEL_SWAP_OR_RERUN"
    )
