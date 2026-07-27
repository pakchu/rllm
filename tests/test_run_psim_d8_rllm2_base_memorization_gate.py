from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from training import preregister_psim_d8_rllm1 as rllm1_prereg
from training import preregister_psim_d8_rllm2_operational_successor as prereg
from training import run_psim_d8_rllm1_base_memorization_gate as base
from training import run_psim_d8_rllm2_base_memorization_gate as gate


@pytest.fixture(scope="module")
def cases() -> list[dict[str, Any]]:
    events = base._load_frozen_gzip_jsonl(
        rllm1_prereg.D8_EVENTS,
        expected_sha256=rllm1_prereg.D8_EVENTS_SHA256,
        expected_decompressed_sha256=(
            rllm1_prereg.D8_EVENTS_ROWS_SHA256
        ),
    )
    return base.build_challenge_cases(events)


def _fake_prepared(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "successor_preregistration": {
            "predecessor_terminal_evidence": {"terminal": True},
            "access_boundary": {
                "files_read": [
                    prereg.RLLM1_PREREGISTRATION.as_posix(),
                    prereg.RLLM1_ATTEMPT.as_posix(),
                    prereg.RLLM1_FAILURE.as_posix(),
                ]
            },
        },
        "preregistration": {
            "source_authority": {"frozen": True},
            "access_boundary": {
                "source_files_read": [
                    rllm1_prereg.D8_EVENTS.as_posix(),
                    rllm1_prereg.D8_CARDS.as_posix(),
                ]
            },
        },
        "runtime": {"validated": True},
        "processor": object(),
        "cases": cases,
        "prompt_capacity": {"validated": True},
    }


def test_successor_preregistration_is_exact() -> None:
    payload = gate.validate_preregistration()
    assert payload["candidate"]["id"] == "PSIM-D8-RLLM2"
    assert payload["manifest_hash"] == gate.PREREGISTRATION_MANIFEST_HASH
    assert payload["inherited_scientific_contract"]["contract_hash"] == (
        gate.SCIENTIFIC_CONTRACT_HASH
    )
    assert payload["inherited_scientific_contract"]["case_roster_hash"] == (
        gate.CASE_ROSTER_HASH
    )
    assert payload["sole_operational_delta"][
        "challenge_or_threshold_change"
    ] is False


def test_direct_script_entrypoint_imports_without_pythonpath() -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(Path(gate.__file__)), "--help"],
        cwd=gate.REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--validate-only" in completed.stdout
    assert "--output" not in completed.stdout


class _FakeCuda:
    @staticmethod
    def memory_allocated(_: int) -> int:
        return 1


class _FakeTorch:
    cuda = _FakeCuda()

    @staticmethod
    def device(value: Any) -> str:
        text = str(value)
        if text in {"0", "cuda"}:
            return "cuda:0"
        return text


class _FakeParameter:
    device = "cuda:0"


def _fake_model(advisory: dict[str, Any]) -> Any:
    model_type = type(
        "Gemma4ForConditionalGeneration",
        (),
        {
            "is_quantized": True,
            "device": "cuda:0",
            "hf_device_map": advisory,
            "parameters": lambda self: iter([_FakeParameter()]),
        },
    )
    return model_type()


def test_device_placement_accepts_empty_advisory_but_rejects_offload() -> None:
    empty = gate.validate_loaded_model_placement(
        _fake_model({}),
        _FakeTorch,
    )
    assert empty["model_device"] == "cuda:0"
    assert empty["first_parameter_device"] == "cuda:0"
    assert empty["empty_hf_device_map_accepted"] is True
    explicit = gate.validate_loaded_model_placement(
        _fake_model({"": 0, "model": "cuda:0"}),
        _FakeTorch,
    )
    assert explicit["empty_hf_device_map_accepted"] is False
    assert set(explicit["hf_device_map_normalized"].values()) == {
        "cuda:0"
    }
    for unsafe in ({"": "cpu"}, {"": "disk"}, {"": "meta"}, {"": 1}):
        with pytest.raises(RuntimeError, match="advisory device map"):
            gate.validate_loaded_model_placement(
                _fake_model(unsafe),
                _FakeTorch,
            )


def test_attempt_guard_and_clean_head_precede_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "result.json"
    attempt = tmp_path / "attempt.json"
    monkeypatch.setattr(gate, "DEFAULT_OUTPUT", output)
    monkeypatch.setattr(gate, "DEFAULT_ATTEMPT", attempt)
    attempt.write_text("consumed", encoding="utf-8")
    monkeypatch.setattr(
        gate,
        "_clean_committed_head",
        lambda: pytest.fail("clean check ran after consumed attempt"),
    )
    with pytest.raises(RuntimeError, match="already attempted"):
        gate.run_gate()
    attempt.unlink()

    def fail_clean() -> str:
        raise RuntimeError("clean-first")

    monkeypatch.setattr(gate, "_clean_committed_head", fail_clean)
    monkeypatch.setattr(
        gate,
        "prepare_source_only_gate",
        lambda: pytest.fail("preflight ran before clean check"),
    )
    with pytest.raises(RuntimeError, match="clean-first"):
        gate.run_gate()


def test_success_path_consumes_attempt_before_scorer_and_writes_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cases: list[dict[str, Any]],
) -> None:
    output = tmp_path / "result.json"
    attempt = tmp_path / "attempt.json"
    monkeypatch.setattr(gate, "DEFAULT_OUTPUT", output)
    monkeypatch.setattr(gate, "DEFAULT_ATTEMPT", attempt)
    monkeypatch.setattr(gate, "_clean_committed_head", lambda: "b" * 40)
    monkeypatch.setattr(
        gate,
        "prepare_source_only_gate",
        lambda: _fake_prepared(cases),
    )

    class FakeScorer:
        def __init__(self, _: Any) -> None:
            assert attempt.is_file()
            self.forward_calls_started = 0

        def score(self, prompt: str) -> dict[str, Any]:
            assert prompt.endswith("ANSWER=")
            self.forward_calls_started += 1
            return {
                "predicted_code": "A",
                "code_logits": {
                    code: float(-index)
                    for index, code in enumerate(
                        rllm1_prereg.MEMORIZATION_CHALLENGE_CODES
                    )
                },
                "input_tokens": 1,
                "inference_seconds": 0.0,
            }

        def metrics(self) -> dict[str, Any]:
            return {"fake": True}

    payload = gate.run_gate(scorer_factory=FakeScorer)
    assert attempt.is_file()
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert payload["policy_id"] == "PSIM-D8-RLLM2"
    assert payload["challenge"]["decision"] == "pass"
    assert payload["challenge"]["terminal_action"] == gate.PASS_ACTION
    assert payload["challenge"]["source_feature_construction_authorized"] is True
    assert payload["challenge"]["market_access_authorized"] is False
    assert len(payload["challenge"]["predictions"]) == 128
    with pytest.raises(RuntimeError, match="already attempted"):
        gate.run_gate(scorer_factory=FakeScorer)


def test_post_sentinel_failure_is_written_terminally_without_rerun(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cases: list[dict[str, Any]],
) -> None:
    output = tmp_path / "result.json"
    attempt = tmp_path / "attempt.json"
    monkeypatch.setattr(gate, "DEFAULT_OUTPUT", output)
    monkeypatch.setattr(gate, "DEFAULT_ATTEMPT", attempt)
    monkeypatch.setattr(gate, "_clean_committed_head", lambda: "c" * 40)
    monkeypatch.setattr(
        gate,
        "prepare_source_only_gate",
        lambda: _fake_prepared(cases),
    )

    class FailingScorer:
        def __init__(self, _: Any) -> None:
            assert attempt.is_file()
            raise RuntimeError("synthetic load failure")

    with pytest.raises(RuntimeError, match="synthetic load failure"):
        gate.run_gate(scorer_factory=FailingScorer)
    failure = json.loads(output.read_text(encoding="utf-8"))
    assert failure["decision"] == "reject"
    assert failure["terminal_action"] == gate.FAILURE_ACTION
    assert failure["failure"]["stage"] == "MODEL_CONSTRUCTION"
    assert failure["observations"]["model_forwards_started"] == 0
    assert failure["observations"]["challenge_predictions_created"] == 0
    assert failure["rerun_authorized"] is False
    assert failure["access_boundary"]["market_or_funding_paths_read"] == []
    with pytest.raises(RuntimeError, match="already attempted"):
        gate.run_gate(scorer_factory=FailingScorer)


def test_runner_has_no_market_or_economic_import_surface() -> None:
    source_path = Path(gate.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert not imports.intersection(
        {
            "backtest",
            "database",
            "numpy",
            "pandas",
            "psycopg",
            "sqlalchemy",
        }
    )
    assert gate.DEFAULT_OUTPUT == prereg.RLLM2_RESULT
    assert gate.DEFAULT_ATTEMPT == prereg.RLLM2_ATTEMPT
    assert gate.DEFAULT_OUTPUT != gate.DEFAULT_ATTEMPT
    payload = prereg.build_preregistration()
    assert gate.FAILURE_ACTION == payload["terminal_actions"][
        "operational_or_memorization_failure"
    ]
    assert gate.PASS_ACTION == payload["terminal_actions"]["pass"]
