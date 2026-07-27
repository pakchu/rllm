from __future__ import annotations

import ast
import gzip
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from training import (
    preregister_psim_d8_rllm2_s2_chunked_source_feature as prereg,
)
from training import (
    run_psim_d8_rllm2_s2_chunked_source_feature_seal as runner,
)

s1 = prereg.s1


def _row(
    index: int,
    *,
    relation_forwarded: bool,
) -> dict[str, Any]:
    code_to_label = {
        code: label
        for code, label in zip(
            s1.RELATION_CODE_ORDER,
            s1.rllm1.RELATION_LABELS,
        )
    }
    core = {
        "schema_version": s1.SOURCE_ROW_SCHEMA_VERSION,
        "row_index": index,
        "schedule": "ARCHIVE_D90",
        "decision_at": f"2020-01-{index + 1:02d}T12:05:00Z",
        "split_year": 2020,
        "split": "train",
        "card_hash": f"card-{index}",
        "prior_card_hash": f"prior-{index}",
        "selector_digest": f"selector-{index}",
        "selected_subcard_ordinal": 0,
        "selected_subcard_hash": f"subcard-{index}",
        "selected_subcard_payload_sha256": f"payload-{index}",
        "selected_relation_unit_count": int(relation_forwarded),
        "eligible_relation_unit_count": int(relation_forwarded),
        "forced_no_eligible": not relation_forwarded,
        "source_payload": {"row": chr(ord("A") + index)},
        "source_payload_sha256": f"source-{index}",
        "policy_prompt": f"POLICY-{index}",
        "policy_prompt_sha256": f"policy-{index}",
        "relation_teacher_code_to_label": code_to_label,
        "relation_teacher_prompt": f"RELATION-{index}",
        "relation_teacher_prompt_sha256": f"relation-{index}",
        "embedding_forward_required": True,
        "relation_teacher_forward_required": relation_forwarded,
        "forced_relation_when_teacher_skipped": (
            None if relation_forwarded else "INSUFFICIENT_EVIDENCE"
        ),
    }
    return {**core, "row_hash": runner.canonical_hash(core)}


def _prepared(rows: list[dict[str, Any]]) -> dict[str, Any]:
    relation_forwards = sum(
        bool(row["relation_teacher_forward_required"]) for row in rows
    )
    equivalence_case = {
        "row_index": 0,
        "row_hash": rows[0]["row_hash"],
        "policy_tokens": 5,
        "relation_tokens": 7,
        "selection_reasons": ["unit"],
    }
    capacity_case = {
        "row_index": 0,
        "row_hash": rows[0]["row_hash"],
        "policy_tokens": 5,
        "relation_tokens": 7,
    }
    return {
        "preregistration": {
            "pre_market_equivalence_gate": {
                "equivalence_case_roster_hash": "equivalence-roster",
                "capacity_case_hash": "capacity-case",
                "embedding_thresholds": {
                    "minimum_cosine_similarity": 0.99999,
                    "maximum_rms_absolute_delta": 0.01,
                    "maximum_absolute_delta": 0.05,
                },
                "relation_thresholds": {
                    "maximum_mean_absolute_delta": 0.01,
                    "maximum_absolute_delta": 0.03,
                },
            }
        },
        "runtime": {"validated": True},
        "processor": object(),
        "rows": rows,
        "roster": {
            "row_count": len(rows),
            "source_row_roster_hash": runner.SOURCE_ROW_ROSTER_HASH,
            "embedding_forward_count": len(rows),
            "relation_teacher_forward_count": relation_forwards,
        },
        "prompt_capacity": {
            "maximum_input_tokens": prereg.MAXIMUM_INPUT_TOKENS,
            "truncation": False,
            "policy": {
                "count": len(rows),
                "minimum_tokens": 5,
                "maximum_tokens": 5,
                "mean_tokens": 5.0,
                "maximum_identity": {"row_index": 0, "tokens": 5},
                "counts": [5] * len(rows),
            },
            "relation_teacher": {
                "count": len(rows),
                "minimum_tokens": 7,
                "maximum_tokens": 7,
                "mean_tokens": 7.0,
                "maximum_identity": {"row_index": 0, "tokens": 7},
                "counts": [7] * len(rows),
            },
        },
        "equivalence_cases": [equivalence_case],
        "capacity_case": capacity_case,
    }


def _patch_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runner, "DEFAULT_ATTEMPT", tmp_path / "attempt.json")
    monkeypatch.setattr(
        runner,
        "DEFAULT_EQUIVALENCE_RESULT",
        tmp_path / "equivalence.json",
    )
    monkeypatch.setattr(runner, "DEFAULT_OUTPUT", tmp_path / "result.json")
    monkeypatch.setattr(
        runner,
        "DEFAULT_SOURCE_ROWS",
        tmp_path / "source.jsonl.gz",
    )
    monkeypatch.setattr(
        runner,
        "DEFAULT_EMBEDDINGS",
        tmp_path / "embeddings.npz",
    )
    monkeypatch.setattr(
        runner,
        "DEFAULT_RELATION_LOGITS",
        tmp_path / "relation_logits.npz",
    )
    monkeypatch.setattr(
        runner,
        "DEFAULT_RELATION_ROWS",
        tmp_path / "relation_rows.jsonl.gz",
    )
    monkeypatch.setattr(
        runner,
        "DEFAULT_CHECKPOINT_DIRECTORY",
        tmp_path / "checkpoint",
    )


def _patch_gate_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    def pass_gate(
        _scorer: Any,
        *,
        execution_commit: str,
        attempt_path: Path,
        attempt: dict[str, Any],
        prepared: dict[str, Any],
    ) -> dict[str, Any]:
        case = prepared["equivalence_cases"][0]
        payload = runner._gate_result_payload(
            execution_commit=execution_commit,
            attempt_path=attempt_path,
            attempt=attempt,
            prepared=prepared,
            decision="pass",
            case_results=[_valid_case_result(case)],
            capacity_result=_valid_capacity_result(
                prepared["capacity_case"]
            ),
        )
        runner._write_gate_result(payload)
        return payload

    monkeypatch.setattr(runner, "run_pre_market_gates", pass_gate)


def _scan_evidence(tokens: int) -> dict[str, Any]:
    return {
        "input_tokens": tokens,
        "chunk_size_tokens": prereg.CHUNK_SIZE,
        "chunk_count": (
            tokens + prereg.CHUNK_SIZE - 1
        )
        // prereg.CHUNK_SIZE,
        "token_reconstruction_exact": True,
    }


def _valid_case_result(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_index": case["row_index"],
        "row_hash": case["row_hash"],
        "selection_reasons": case["selection_reasons"],
        "policy_tokens": case["policy_tokens"],
        "relation_tokens": case["relation_tokens"],
        "policy_scan": _scan_evidence(case["policy_tokens"]),
        "relation_scan": _scan_evidence(case["relation_tokens"]),
        "token_contract_pass": True,
        "embedding_comparison": {
            "all_values_finite": True,
            "cosine_similarity": 1.0,
            "rms_absolute_delta": 0.0,
            "maximum_absolute_delta": 0.0,
            "pass": True,
        },
        "relation_comparison": {
            "reference_finite": True,
            "candidate_finite": True,
            "same_canonical_nonfinite_semantics": True,
            "reference_code": "F",
            "candidate_code": "F",
            "mean_absolute_delta": 0.0,
            "maximum_absolute_delta": 0.0,
            "pass": True,
        },
        "pass": True,
    }


def _valid_capacity_result(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_index": case["row_index"],
        "row_hash": case["row_hash"],
        "policy_tokens": case["policy_tokens"],
        "relation_tokens": case["relation_tokens"],
        "policy_scan": _scan_evidence(case["policy_tokens"]),
        "relation_scan": _scan_evidence(case["relation_tokens"]),
        "embedding_all_values_finite": True,
        "relation_logits_finite_or_canonical_nan": True,
        "predicted_relation_code": "F",
        "peak_allocated_bytes": 1024,
        "maximum_peak_allocated_bytes": (
            prereg.MAXIMUM_PEAK_ALLOCATED_BYTES
        ),
        "outputs_reused_by_full_extraction": False,
        "pass": True,
    }


def test_preregistration_is_exact() -> None:
    payload = runner.validate_preregistration()
    assert payload["candidate"]["id"] == prereg.STAGE_ID
    assert payload["manifest_hash"] == runner.PREREGISTRATION_MANIFEST_HASH
    assert (
        payload["unchanged_scientific_contract"][
            "source_row_roster_hash"
        ]
        == runner.SOURCE_ROW_ROSTER_HASH
    )
    assert payload["access_boundary"]["market_or_funding_paths_read"] == []
    assert (
        payload["access_boundary"][
            "s1_checkpoint_or_partial_model_outputs_read"
        ]
        is False
    )


def test_direct_script_entrypoint_imports_without_pythonpath() -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(Path(runner.__file__)), "--help"],
        cwd=runner.REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--validate-only" in completed.stdout
    assert "--resume" in completed.stdout


def test_deterministic_artifact_encodings() -> None:
    arrays = {
        "z": np.asarray([[1.0, 2.0]], dtype=np.float32),
        "a": np.asarray([1, 2], dtype=np.int32),
    }
    first = runner._deterministic_npz_bytes(arrays)
    second = runner._deterministic_npz_bytes(arrays)
    assert first == second
    rows = [{"b": 2, "a": 1}]
    first_gzip, first_rows_sha = (
        runner._deterministic_jsonl_gzip_bytes(rows)
    )
    second_gzip, second_rows_sha = (
        runner._deterministic_jsonl_gzip_bytes(rows)
    )
    assert first_gzip == second_gzip
    assert first_rows_sha == second_rows_sha
    assert gzip.decompress(first_gzip) == b'{"a":1,"b":2}\n'
    assert int.from_bytes(first_gzip[4:8], "little") == 0
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        assert archive.namelist() == ["a.npy", "z.npy"]
        assert all(
            info.date_time == (1980, 1, 1, 0, 0, 0)
            for info in archive.infolist()
        )


def test_fixed_chunk_scan_reconstructs_tokens_and_applies_softcap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(prereg, "CHUNK_SIZE", 2)

    class Inputs(dict[str, Any]):
        def to(self, _: Any) -> Inputs:
            return self

    inputs = Inputs(
        input_ids=torch.asarray([[1, 2, 3, 4, 5]], dtype=torch.long),
        attention_mask=torch.ones((1, 5), dtype=torch.long),
    )
    monkeypatch.setattr(runner, "_prompt_inputs", lambda _p, _s: inputs)

    class Cache:
        def __init__(self, tokens: list[int]) -> None:
            self.tokens = tokens

    class InnerModel:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def __call__(self, **kwargs: Any) -> Any:
            prior = kwargs["past_key_values"]
            prior_tokens = [] if prior is None else prior.tokens
            current = kwargs["input_ids"][0].tolist()
            full = prior_tokens + current
            self.calls.append(
                {
                    "current": current,
                    "prior": prior_tokens,
                    "position_ids": kwargs["position_ids"][0].tolist(),
                    "attention_mask_length": int(
                        kwargs["attention_mask"].shape[-1]
                    ),
                    "use_cache": kwargs["use_cache"],
                }
            )
            hidden = torch.zeros(
                (1, len(current), s1.EMBEDDING_WIDTH),
                dtype=torch.float32,
            )
            hidden[0, -1, 0] = float(sum(full))
            hidden[0, -1, 1] = float(len(full))
            return type(
                "Output",
                (),
                {
                    "last_hidden_state": hidden,
                    "past_key_values": Cache(full),
                },
            )()

    class OuterModel:
        device = "cpu"

        def __init__(self) -> None:
            self.model = InnerModel()

        def lm_head(self, hidden: Any) -> Any:
            return torch.arange(6, dtype=torch.float32) + hidden[0]

    scorer = runner.Gemma4SourceFeatureScorer.__new__(
        runner.Gemma4SourceFeatureScorer
    )
    scorer.torch = torch
    scorer.processor = object()
    scorer.model = OuterModel()
    scorer.code_ids = {
        code: index
        for index, code in enumerate(s1.RELATION_CODE_ORDER)
    }
    scorer.final_logit_softcapping = 2.0
    scorer.forward_calls_started = 0
    scorer.chunk_forwards_started = 0
    scorer.reference_forwards_started = 0
    scorer.embedding_forwards_started = 0
    scorer.relation_forwards_started = 0
    scorer.embedding_inference_seconds = 0.0
    scorer.relation_inference_seconds = 0.0
    scorer._empty_prompt_cache = lambda: None

    vector, tokens, evidence = scorer.chunked_embedding_with_evidence(
        "policy"
    )
    assert tokens == 5
    assert evidence == {
        "input_tokens": 5,
        "chunk_size_tokens": 2,
        "chunk_count": 3,
        "token_reconstruction_exact": True,
    }
    assert vector[0] == 15.0
    assert vector[1] == 5.0
    assert scorer.model.model.calls == [
        {
            "current": [1, 2],
            "prior": [],
            "position_ids": [0, 1],
            "attention_mask_length": 2,
            "use_cache": True,
        },
        {
            "current": [3, 4],
            "prior": [1, 2],
            "position_ids": [2, 3],
            "attention_mask_length": 4,
            "use_cache": True,
        },
        {
            "current": [5],
            "prior": [1, 2, 3, 4],
            "position_ids": [4],
            "attention_mask_length": 5,
            "use_cache": True,
        },
    ]

    logits, code, relation_tokens, relation_evidence = (
        scorer.chunked_relation_with_evidence("relation")
    )
    raw = np.arange(6, dtype=np.float32) + 15.0
    expected = np.tanh(raw / 2.0) * 2.0
    np.testing.assert_allclose(logits, expected, rtol=0.0, atol=1e-6)
    assert code == runner._predicted_code(logits)
    assert relation_tokens == 5
    assert relation_evidence["token_reconstruction_exact"] is True
    assert scorer.chunk_forwards_started == 6


def test_gate_pass_is_bound_and_capacity_outputs_are_not_reused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = [_row(0, relation_forwarded=True)]
    prepared = _prepared(rows)
    _patch_paths(monkeypatch, tmp_path)
    attempt_path = Path(runner.DEFAULT_ATTEMPT)
    attempt_path.write_text("{}", encoding="utf-8")
    attempt = {
        "attempt_hash": "attempt",
        "execution_commit": "a" * 40,
        "runner_sha256": runner.sha256_file(Path(runner.__file__)),
    }
    attempt_path.write_bytes(
        runner.canonical_json_bytes(attempt, pretty=True)
    )

    class GateScorer:
        def reference_embedding(self, _: str) -> tuple[np.ndarray, int]:
            return np.ones(s1.EMBEDDING_WIDTH, dtype=np.float32), 5

        def chunked_embedding_with_evidence(
            self,
            _: str,
        ) -> tuple[np.ndarray, int, dict[str, Any]]:
            return (
                np.ones(s1.EMBEDDING_WIDTH, dtype=np.float32),
                5,
                _scan_evidence(5),
            )

        def reference_relation(
            self,
            _: str,
        ) -> tuple[np.ndarray, str, int]:
            return np.arange(6, dtype=np.float32), "F", 7

        def chunked_relation_with_evidence(
            self,
            _: str,
        ) -> tuple[np.ndarray, str, int, dict[str, Any]]:
            return (
                np.arange(6, dtype=np.float32),
                "F",
                7,
                _scan_evidence(7),
            )

        def reset_peak_memory_stats(self) -> None:
            return None

        def peak_allocated_bytes(self) -> int:
            return 1024

    payload = runner.run_pre_market_gates(
        GateScorer(),
        execution_commit="a" * 40,
        attempt_path=attempt_path,
        attempt=attempt,
        prepared=prepared,
    )
    assert payload["decision"] == "pass"
    assert payload["full_source_extraction_authorized"] is True
    assert payload["capacity_result"][
        "outputs_reused_by_full_extraction"
    ] is False
    validated = runner.validate_equivalence_result(
        attempt_path=attempt_path,
        attempt=attempt,
        prepared=prepared,
    )
    assert validated == payload
    mutations = {
        "skeletal case": lambda value: value.update(
            {"case_results": [{"pass": True}]}
        ),
        "wrong row hash": lambda value: value["case_results"][0].update(
            {"row_hash": "tampered"}
        ),
        "reused capacity output": lambda value: value[
            "capacity_result"
        ].update({"outputs_reused_by_full_extraction": True}),
        "excess peak VRAM": lambda value: value[
            "capacity_result"
        ].update(
            {
                "peak_allocated_bytes": (
                    prereg.MAXIMUM_PEAK_ALLOCATED_BYTES + 1
                )
            }
        ),
        "opened market path": lambda value: value[
            "access_boundary"
        ].update({"market_or_funding_paths_read": ["market.csv"]}),
    }
    for mutate in mutations.values():
        tampered = json.loads(json.dumps(payload))
        mutate(tampered)
        tampered_core = {
            key: value
            for key, value in tampered.items()
            if key != "result_hash"
        }
        tampered["result_hash"] = runner.canonical_hash(tampered_core)
        Path(runner.DEFAULT_EQUIVALENCE_RESULT).write_bytes(
            runner.canonical_json_bytes(tampered, pretty=True)
        )
        with pytest.raises(RuntimeError, match="RLLM2-S2"):
            runner.validate_equivalence_result(
                attempt_path=attempt_path,
                attempt=attempt,
                prepared=prepared,
            )


def test_gate_threshold_failure_is_terminal_before_extraction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = [_row(0, relation_forwarded=True)]
    prepared = _prepared(rows)
    _patch_paths(monkeypatch, tmp_path)
    attempt_path = Path(runner.DEFAULT_ATTEMPT)
    attempt = {
        "attempt_hash": "attempt",
        "execution_commit": "a" * 40,
        "runner_sha256": runner.sha256_file(Path(runner.__file__)),
    }
    attempt_path.write_bytes(
        runner.canonical_json_bytes(attempt, pretty=True)
    )

    class FailingGateScorer:
        def reference_embedding(self, _: str) -> tuple[np.ndarray, int]:
            return np.ones(s1.EMBEDDING_WIDTH, dtype=np.float32), 5

        def chunked_embedding_with_evidence(
            self,
            _: str,
        ) -> tuple[np.ndarray, int, dict[str, Any]]:
            return (
                np.zeros(s1.EMBEDDING_WIDTH, dtype=np.float32),
                5,
                {"token_reconstruction_exact": True},
            )

        def reference_relation(
            self,
            _: str,
        ) -> tuple[np.ndarray, str, int]:
            return np.arange(6, dtype=np.float32), "F", 7

        def chunked_relation_with_evidence(
            self,
            _: str,
        ) -> tuple[np.ndarray, str, int, dict[str, Any]]:
            return (
                np.arange(6, dtype=np.float32),
                "F",
                7,
                {"token_reconstruction_exact": True},
            )

    with pytest.raises(RuntimeError, match="equivalence threshold"):
        runner.run_pre_market_gates(
            FailingGateScorer(),
            execution_commit="a" * 40,
            attempt_path=attempt_path,
            attempt=attempt,
            prepared=prepared,
        )
    payload = json.loads(
        Path(runner.DEFAULT_EQUIVALENCE_RESULT).read_text()
    )
    assert payload["decision"] == "reject"
    assert payload["full_source_extraction_authorized"] is False
    assert not Path(runner.DEFAULT_CHECKPOINT_DIRECTORY).exists()


def test_success_path_writes_and_verifies_all_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = [
        _row(0, relation_forwarded=True),
        _row(1, relation_forwarded=False),
    ]
    prepared = _prepared(rows)
    _patch_paths(monkeypatch, tmp_path)
    _patch_gate_pass(monkeypatch)
    monkeypatch.setattr(runner, "_clean_committed_head", lambda: "a" * 40)
    monkeypatch.setattr(
        runner,
        "prepare_source_only_stage",
        lambda: prepared,
    )

    class FakeScorer:
        def __init__(self, _: Any) -> None:
            assert Path(runner.DEFAULT_ATTEMPT).is_file()
            self.forward_calls_started = 0
            self.chunk_forwards_started = 0
            self.reference_forwards_started = 0
            self.embedding_forwards_started = 0
            self.relation_forwards_started = 0

        def embed(self, _: str) -> tuple[np.ndarray, int]:
            self.forward_calls_started += 1
            self.embedding_forwards_started += 1
            return (
                np.full(
                    s1.EMBEDDING_WIDTH,
                    self.embedding_forwards_started,
                    dtype=np.float32,
                ),
                5,
            )

        def score_relation(
            self,
            _: str,
        ) -> tuple[np.ndarray, str, int]:
            self.forward_calls_started += 1
            self.relation_forwards_started += 1
            return np.arange(6, dtype=np.float32), "F", 7

        def metrics(self) -> dict[str, Any]:
            return {
                "model_forwards_started": self.forward_calls_started,
                "embedding_forwards_started": (
                    self.embedding_forwards_started
                ),
                "relation_forwards_started": (
                    self.relation_forwards_started
                ),
            }

    payload = runner.run_stage(scorer_factory=FakeScorer)
    assert payload["decision"] == "pass"
    assert payload["terminal_action"] == runner.PASS_ACTION
    assert payload["open_2020_train_outcomes_authorized"] is True
    assert payload["equivalence_gate"]["decision"] == "pass"
    assert not Path(runner.DEFAULT_CHECKPOINT_DIRECTORY).exists()
    for path in (
        runner.DEFAULT_ATTEMPT,
        runner.DEFAULT_EQUIVALENCE_RESULT,
        runner.DEFAULT_OUTPUT,
        runner.DEFAULT_SOURCE_ROWS,
        runner.DEFAULT_EMBEDDINGS,
        runner.DEFAULT_RELATION_LOGITS,
        runner.DEFAULT_RELATION_ROWS,
    ):
        assert Path(path).is_file()
    relation_rows = [
        json.loads(line)
        for line in gzip.decompress(
            Path(runner.DEFAULT_RELATION_ROWS).read_bytes()
        )
        .decode("utf-8")
        .splitlines()
    ]
    assert relation_rows[0]["predicted_code"] == "F"
    assert (
        relation_rows[1]["predicted_relation"]
        == "INSUFFICIENT_EVIDENCE"
    )


def test_committed_row_resumes_without_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = [
        _row(0, relation_forwarded=True),
        _row(1, relation_forwarded=False),
    ]
    prepared = _prepared(rows)
    _patch_paths(monkeypatch, tmp_path)
    _patch_gate_pass(monkeypatch)
    monkeypatch.setattr(runner, "_clean_committed_head", lambda: "c" * 40)
    monkeypatch.setattr(
        runner,
        "prepare_source_only_stage",
        lambda: prepared,
    )
    instances: list[Any] = []

    class FakeScorer:
        def __init__(self, _: Any) -> None:
            self.forward_calls_started = 0
            self.chunk_forwards_started = 0
            self.reference_forwards_started = 0
            self.embedding_forwards_started = 0
            self.relation_forwards_started = 0
            self.embedding_prompts: list[str] = []
            self.relation_prompts: list[str] = []
            instances.append(self)

        def embed(self, prompt: str) -> tuple[np.ndarray, int]:
            self.forward_calls_started += 1
            self.embedding_forwards_started += 1
            self.embedding_prompts.append(prompt)
            return np.ones(s1.EMBEDDING_WIDTH, dtype=np.float32), 5

        def score_relation(
            self,
            prompt: str,
        ) -> tuple[np.ndarray, str, int]:
            self.forward_calls_started += 1
            self.relation_forwards_started += 1
            self.relation_prompts.append(prompt)
            return np.arange(6, dtype=np.float32), "F", 7

        def metrics(self) -> dict[str, Any]:
            return {
                "model_forwards_started": self.forward_calls_started,
                "embedding_forwards_started": (
                    self.embedding_forwards_started
                ),
                "relation_forwards_started": (
                    self.relation_forwards_started
                ),
            }

    original_clear = runner._clear_inflight_row
    monkeypatch.setattr(
        runner,
        "_clear_inflight_row",
        lambda _: (_ for _ in ()).throw(
            KeyboardInterrupt("committed-row interruption")
        ),
    )
    with pytest.raises(KeyboardInterrupt):
        runner.run_stage(scorer_factory=FakeScorer)
    assert not Path(runner.DEFAULT_OUTPUT).exists()
    assert Path(runner.DEFAULT_EQUIVALENCE_RESULT).is_file()

    monkeypatch.setattr(runner, "_clear_inflight_row", original_clear)
    payload = runner.run_stage(scorer_factory=FakeScorer, resume=True)
    assert payload["checkpoint_evidence"]["resumed"] is True
    assert payload["checkpoint_evidence"]["resumed_from_completed_rows"] == 1
    assert instances[0].embedding_prompts == ["POLICY-0"]
    assert instances[1].embedding_prompts == ["POLICY-1"]
    assert instances[1].relation_prompts == []


def test_ambiguous_inflight_row_rejects_without_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = [_row(0, relation_forwarded=True)]
    prepared = _prepared(rows)
    _patch_paths(monkeypatch, tmp_path)
    _patch_gate_pass(monkeypatch)
    monkeypatch.setattr(runner, "_clean_committed_head", lambda: "d" * 40)
    monkeypatch.setattr(
        runner,
        "prepare_source_only_stage",
        lambda: prepared,
    )

    class InterruptedScorer:
        def __init__(self, _: Any) -> None:
            self.forward_calls_started = 0
            self.chunk_forwards_started = 0
            self.reference_forwards_started = 0
            self.embedding_forwards_started = 0
            self.relation_forwards_started = 0

        def embed(self, _: str) -> tuple[np.ndarray, int]:
            self.embedding_forwards_started += 1
            raise KeyboardInterrupt("ambiguous row")

    with pytest.raises(KeyboardInterrupt):
        runner.run_stage(scorer_factory=InterruptedScorer)
    assert not Path(runner.DEFAULT_OUTPUT).exists()

    def no_replay(_: Any) -> Any:
        pytest.fail("ambiguous row was replayed")

    with pytest.raises(RuntimeError, match="ambiguous started forwards"):
        runner.run_stage(scorer_factory=no_replay, resume=True)
    failure = json.loads(Path(runner.DEFAULT_OUTPUT).read_text())
    assert failure["decision"] == "reject"
    assert failure["observations"]["completed_source_rows"] == 0
    assert failure["rerun_authorized"] is False


def test_pre_gate_interruption_is_terminal_and_not_resumable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = [_row(0, relation_forwarded=True)]
    prepared = _prepared(rows)
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "_clean_committed_head", lambda: "e" * 40)
    monkeypatch.setattr(
        runner,
        "prepare_source_only_stage",
        lambda: prepared,
    )

    class InterruptedGateScorer:
        def __init__(self, _: Any) -> None:
            self.forward_calls_started = 0
            self.chunk_forwards_started = 0
            self.reference_forwards_started = 0
            self.embedding_forwards_started = 0
            self.relation_forwards_started = 0

        def reference_embedding(self, _: str) -> Any:
            raise KeyboardInterrupt("pre-gate interruption")

    with pytest.raises(KeyboardInterrupt):
        runner.run_stage(scorer_factory=InterruptedGateScorer)
    failure = json.loads(Path(runner.DEFAULT_OUTPUT).read_text())
    gate = json.loads(
        Path(runner.DEFAULT_EQUIVALENCE_RESULT).read_text()
    )
    assert failure["decision"] == "reject"
    assert failure["failure"]["stage"] == (
        "PRE_MARKET_EQUIVALENCE_AND_CAPACITY"
    )
    assert gate["decision"] == "reject"
    assert not Path(runner.DEFAULT_CHECKPOINT_DIRECTORY).exists()
    with pytest.raises(RuntimeError, match="resume state is unavailable"):
        runner.run_stage(scorer_factory=InterruptedGateScorer, resume=True)


def test_all_forced_no_eligible_rows_skip_relation_forward() -> None:
    rows = [
        _row(0, relation_forwarded=False),
        _row(1, relation_forwarded=False),
    ]

    class EmbedOnlyScorer:
        def __init__(self) -> None:
            self.relation_calls = 0

        def embed(self, _: str) -> tuple[np.ndarray, int]:
            return np.ones(s1.EMBEDDING_WIDTH, dtype=np.float32), 5

        def score_relation(self, _: str) -> Any:
            self.relation_calls += 1
            pytest.fail("forced-no-eligible row called relation scorer")

    scorer = EmbedOnlyScorer()
    shards = [
        runner._process_shard(
            scorer,
            rows=rows,
            policy_counts=[5, 5],
            relation_counts=[7, 7],
            start=index,
            stop=index + 1,
        )
        for index in range(2)
    ]
    merged = runner._merge_shards(shards, rows=rows)
    assert scorer.relation_calls == 0
    assert np.all(np.isnan(merged["relation_logits"]))


def test_runner_has_no_market_or_s1_checkpoint_import_surface() -> None:
    source_path = Path(runner.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
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
            "pandas",
            "psycopg",
            "sqlalchemy",
        }
    )
    assert "psim_d8_rllm2_source_feature_2026-07-27" not in source
    assert runner.DEFAULT_OUTPUT == prereg.RESULT_PATH
    assert runner.DEFAULT_ATTEMPT == prereg.ATTEMPT_PATH
    assert runner.DEFAULT_EQUIVALENCE_RESULT == (
        prereg.EQUIVALENCE_RESULT_PATH
    )
    assert runner.PASS_ACTION == prereg.build_preregistration()[
        "terminal_actions"
    ]["success"]
    assert runner.FAILURE_ACTION == prereg.build_preregistration()[
        "terminal_actions"
    ]["failure"]
