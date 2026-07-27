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
    preregister_psim_d8_rllm2_s3_chunked_native_source_feature as prereg,
)
from training import (
    run_psim_d8_rllm2_s3_chunked_native_source_feature_seal as runner,
)

s1 = prereg.s2.s1


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
            strict=True,
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
    repeatability_case = {
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
            "pre_market_repeatability_and_capacity_gate": {
                "repeat_count": prereg.REPEAT_COUNT,
                "repeat_model_load_count": prereg.REPEAT_MODEL_LOAD_COUNT,
                "repeatability_case_roster_hash": "repeatability-roster",
                "capacity_case_hash": "capacity-case",
                "capacity_repeat_count": prereg.REPEAT_COUNT,
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
        "repeatability_cases": [repeatability_case],
        "capacity_case": capacity_case,
    }


def _patch_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runner, "DEFAULT_ATTEMPT", tmp_path / "attempt.json")
    monkeypatch.setattr(
        runner,
        "DEFAULT_REPEATABILITY_RESULT",
        tmp_path / "repeatability.json",
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


def _write_gate_attempt(path: Path) -> dict[str, Any]:
    attempt = {
        "attempt_hash": "attempt",
        "execution_commit": "a" * 40,
        "runner_sha256": runner.sha256_file(Path(runner.__file__)),
    }
    path.write_bytes(runner.canonical_json_bytes(attempt, pretty=True))
    return attempt


class _FakeScorer:
    def __init__(
        self,
        load_index: int,
        events: list[str],
        *,
        embedding_mismatch: bool = False,
        relation_code_mismatch: bool = False,
        capacity_oom: bool = False,
        interrupt_gate: bool = False,
        interrupt_extraction: bool = False,
    ) -> None:
        self.load_index = load_index
        self.events = events
        self.embedding_mismatch = embedding_mismatch
        self.relation_code_mismatch = relation_code_mismatch
        self.capacity_oom = capacity_oom
        self.interrupt_gate = interrupt_gate
        self.interrupt_extraction = interrupt_extraction
        self.closed = False
        self.forward_calls_started = 0
        self.chunk_forwards_started = 0
        self.embedding_forwards_started = 0
        self.relation_forwards_started = 0
        self.embed_prompts: list[str] = []
        self.relation_prompts: list[str] = []

    def _embedding(self) -> np.ndarray:
        values = np.ones(s1.EMBEDDING_WIDTH, dtype=np.float32)
        if self.embedding_mismatch and self.load_index == 1:
            values[0] = 2.0
        return values

    def chunked_embedding_with_evidence(
        self,
        _: str,
    ) -> tuple[np.ndarray, int, dict[str, Any]]:
        if self.interrupt_gate:
            raise KeyboardInterrupt("gate interruption")
        self.forward_calls_started += 1
        self.chunk_forwards_started += 1
        self.embedding_forwards_started += 1
        return self._embedding(), 5, _scan_evidence(5)

    def chunked_relation_with_evidence(
        self,
        _: str,
    ) -> tuple[np.ndarray, str, int, dict[str, Any]]:
        self.forward_calls_started += 1
        self.chunk_forwards_started += 1
        self.relation_forwards_started += 1
        logits = np.arange(6, dtype=np.float32)
        code = "E" if self.relation_code_mismatch else "F"
        return logits, code, 7, _scan_evidence(7)

    def reset_peak_memory_stats(self) -> None:
        return None

    def peak_allocated_bytes(self) -> int:
        if self.capacity_oom:
            return prereg.MAXIMUM_PEAK_ALLOCATED_BYTES + 1
        return 1024

    def embed(self, prompt: str) -> tuple[np.ndarray, int]:
        self.embed_prompts.append(prompt)
        self.forward_calls_started += 1
        self.embedding_forwards_started += 1
        if self.interrupt_extraction:
            raise KeyboardInterrupt("ambiguous extraction row")
        return self._embedding(), 5

    def score_relation(
        self,
        prompt: str,
    ) -> tuple[np.ndarray, str, int]:
        self.relation_prompts.append(prompt)
        self.forward_calls_started += 1
        self.relation_forwards_started += 1
        return np.arange(6, dtype=np.float32), "F", 7

    def metrics(self) -> dict[str, Any]:
        return {
            "model_forwards_started": self.forward_calls_started,
            "chunk_forwards_started": self.chunk_forwards_started,
            "embedding_forwards_started": self.embedding_forwards_started,
            "relation_forwards_started": self.relation_forwards_started,
            "peak_allocated_bytes": 1024,
        }

    def close(self) -> None:
        if not self.closed:
            self.events.append(f"close-{self.load_index}")
            self.closed = True


class _FakeFactory:
    def __init__(
        self,
        *,
        embedding_mismatch: bool = False,
        relation_code_mismatch: bool = False,
        capacity_oom: bool = False,
        interrupt_gate: bool = False,
        interrupt_extraction_index: int | None = None,
    ) -> None:
        self.embedding_mismatch = embedding_mismatch
        self.relation_code_mismatch = relation_code_mismatch
        self.capacity_oom = capacity_oom
        self.interrupt_gate = interrupt_gate
        self.interrupt_extraction_index = interrupt_extraction_index
        self.events: list[str] = []
        self.instances: list[_FakeScorer] = []

    def __call__(self, _: Any) -> _FakeScorer:
        assert Path(runner.DEFAULT_ATTEMPT).is_file()
        load_index = len(self.instances)
        self.events.append(f"construct-{load_index}")
        scorer = _FakeScorer(
            load_index,
            self.events,
            embedding_mismatch=self.embedding_mismatch,
            relation_code_mismatch=(
                self.relation_code_mismatch and load_index == 1
            ),
            capacity_oom=self.capacity_oom,
            interrupt_gate=self.interrupt_gate and load_index == 0,
            interrupt_extraction=(
                load_index == self.interrupt_extraction_index
            ),
        )
        self.instances.append(scorer)
        return scorer


def _run_gate(
    factory: _FakeFactory,
    *,
    prepared: dict[str, Any],
    attempt_path: Path,
    attempt: dict[str, Any],
) -> dict[str, Any]:
    return runner.run_pre_market_gates(
        factory,
        prepared["processor"],
        execution_commit="a" * 40,
        attempt_path=attempt_path,
        attempt=attempt,
        prepared=prepared,
    )


def _publish_fixture(
    checkpoint_directory: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, bytes],
]:
    rows = [_row(0, relation_forwarded=True)]
    arrays = {
        "row_index": np.asarray([0], dtype=np.int32),
        "embedding": np.ones(
            (1, s1.EMBEDDING_WIDTH),
            dtype=np.float32,
        ),
        "relation_logits": np.arange(
            len(s1.RELATION_CODE_ORDER),
            dtype=np.float32,
        ).reshape(1, -1),
        "relation_forwarded": np.asarray([1], dtype=np.uint8),
        "policy_input_tokens": np.asarray([5], dtype=np.int32),
        "relation_input_tokens": np.asarray([7], dtype=np.int32),
    }
    records, raw = runner._build_final_artifacts(
        rows=rows,
        arrays=arrays,
    )
    core = {
        "protocol_version": runner.PROTOCOL_VERSION,
        "stage_id": prereg.STAGE_ID,
        "decision": "pass",
        "terminal_action": runner.PASS_ACTION,
        "artifacts": records,
    }
    checkpoint_directory.mkdir()
    return {**core, "result_hash": runner.canonical_hash(core)}, records, raw


def test_preregistration_is_exact() -> None:
    payload = runner.validate_preregistration()
    assert payload["candidate"]["id"] == prereg.STAGE_ID
    assert payload["manifest_hash"] == runner.PREREGISTRATION_MANIFEST_HASH
    assert payload["scientific_contract"]["source_row_roster_hash"] == (
        runner.SOURCE_ROW_ROSTER_HASH
    )
    gate = payload["pre_market_repeatability_and_capacity_gate"]
    assert gate["repeat_model_load_count"] == 2
    assert gate["full_extraction_uses_fresh_third_model_load"] is True
    assert payload["access_boundary"][
        "s1_or_s2_checkpoint_or_model_outputs_read"
    ] is False


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
    assert first == runner._deterministic_npz_bytes(arrays)
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
    scorer.embedding_forwards_started = 0
    scorer.relation_forwards_started = 0
    scorer.embedding_inference_seconds = 0.0
    scorer.relation_inference_seconds = 0.0
    scorer._empty_prompt_cache = lambda: None

    vector, tokens, evidence = scorer.chunked_embedding_with_evidence(
        "policy"
    )
    assert tokens == 5
    assert evidence == _scan_evidence(5)
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
    np.testing.assert_allclose(
        logits,
        np.tanh(raw / 2.0) * 2.0,
        rtol=0.0,
        atol=1e-6,
    )
    assert code == runner._predicted_code(logits)
    assert relation_tokens == 5
    assert relation_evidence["token_reconstruction_exact"] is True


def test_gate_uses_two_independent_closed_model_loads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepared([_row(0, relation_forwarded=True)])
    _patch_paths(monkeypatch, tmp_path)
    attempt_path = Path(runner.DEFAULT_ATTEMPT)
    attempt = _write_gate_attempt(attempt_path)
    factory = _FakeFactory()
    payload = _run_gate(
        factory,
        prepared=prepared,
        attempt_path=attempt_path,
        attempt=attempt,
    )
    assert payload["decision"] == "pass"
    assert factory.events == [
        "construct-0",
        "close-0",
        "construct-1",
        "close-1",
    ]
    assert len(payload["load_results"]) == 2
    assert payload["cross_load_comparison"]["pass"] is True
    assert payload["gate_outputs_reused_by_full_extraction"] is False
    assert payload["full_extraction_uses_fresh_third_model_load"] is True
    assert runner.validate_repeatability_result(
        attempt_path=attempt_path,
        attempt=attempt,
        prepared=prepared,
    ) == payload


@pytest.mark.parametrize(
    ("factory", "mismatch_key"),
    [
        (_FakeFactory(embedding_mismatch=True), "embedding_hashes_identical"),
        (_FakeFactory(relation_code_mismatch=True), "relation_codes_identical"),
    ],
)
def test_cross_load_mismatch_rejects_before_extraction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    factory: _FakeFactory,
    mismatch_key: str,
) -> None:
    prepared = _prepared([_row(0, relation_forwarded=True)])
    _patch_paths(monkeypatch, tmp_path)
    attempt_path = Path(runner.DEFAULT_ATTEMPT)
    attempt = _write_gate_attempt(attempt_path)
    with pytest.raises(RuntimeError, match="repeatability or capacity"):
        _run_gate(
            factory,
            prepared=prepared,
            attempt_path=attempt_path,
            attempt=attempt,
        )
    payload = json.loads(
        Path(runner.DEFAULT_REPEATABILITY_RESULT).read_text()
    )
    assert payload["decision"] == "reject"
    assert payload["cross_load_comparison"][mismatch_key] is False
    assert payload["full_source_extraction_authorized"] is False
    assert not Path(runner.DEFAULT_CHECKPOINT_DIRECTORY).exists()


def test_capacity_failure_is_terminal_before_extraction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepared([_row(0, relation_forwarded=True)])
    _patch_paths(monkeypatch, tmp_path)
    attempt_path = Path(runner.DEFAULT_ATTEMPT)
    attempt = _write_gate_attempt(attempt_path)
    factory = _FakeFactory(capacity_oom=True)
    with pytest.raises(RuntimeError, match="repeatability or capacity"):
        _run_gate(
            factory,
            prepared=prepared,
            attempt_path=attempt_path,
            attempt=attempt,
        )
    payload = json.loads(
        Path(runner.DEFAULT_REPEATABILITY_RESULT).read_text()
    )
    assert payload["decision"] == "reject"
    assert payload["load_results"][0]["capacity_result"]["pass"] is False
    assert payload["full_source_extraction_authorized"] is False


def test_pass_gate_deep_validation_rejects_rehashed_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepared([_row(0, relation_forwarded=True)])
    _patch_paths(monkeypatch, tmp_path)
    attempt_path = Path(runner.DEFAULT_ATTEMPT)
    attempt = _write_gate_attempt(attempt_path)
    payload = _run_gate(
        _FakeFactory(),
        prepared=prepared,
        attempt_path=attempt_path,
        attempt=attempt,
    )
    mutations = [
        lambda value: value["load_results"][0].update(
            {"case_results": [{"pass": True}]}
        ),
        lambda value: value["load_results"][0]["case_results"][0].update(
            {"row_hash": "tampered"}
        ),
        lambda value: value["load_results"][0]["case_results"][0].update(
            {"embedding_float32_bytes_sha256": "0" * 64}
        ),
        lambda value: value["load_results"][0]["case_results"][0].update(
            {"policy_tokens": 6}
        ),
        lambda value: value["load_results"][0]["case_results"][0][
            "policy_scan"
        ].update({"chunk_count": 99}),
        lambda value: value["load_results"][0][
            "capacity_result"
        ].update({"outputs_reused_by_full_extraction": True}),
        lambda value: value["load_results"][0][
            "capacity_result"
        ].update(
            {
                "peak_allocated_bytes": (
                    prereg.MAXIMUM_PEAK_ALLOCATED_BYTES + 1
                )
            }
        ),
        lambda value: value.update(
            {"gate_outputs_reused_by_full_extraction": True}
        ),
        lambda value: value.update(
            {"full_extraction_uses_fresh_third_model_load": False}
        ),
        lambda value: value["access_boundary"].update(
            {"market_or_funding_paths_read": ["market.csv"]}
        ),
        lambda value: value["load_results"][0].update(
            {"scorer_closed_before_next_load": False}
        ),
    ]
    for mutate in mutations:
        tampered = json.loads(json.dumps(payload))
        mutate(tampered)
        core = {
            key: value
            for key, value in tampered.items()
            if key != "result_hash"
        }
        tampered["result_hash"] = runner.canonical_hash(core)
        Path(runner.DEFAULT_REPEATABILITY_RESULT).write_bytes(
            runner.canonical_json_bytes(tampered, pretty=True)
        )
        with pytest.raises(RuntimeError, match="RLLM2-S3"):
            runner.validate_repeatability_result(
                attempt_path=attempt_path,
                attempt=attempt,
                prepared=prepared,
            )


def test_success_uses_fresh_third_model_and_publishes_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = [
        _row(0, relation_forwarded=True),
        _row(1, relation_forwarded=False),
    ]
    prepared = _prepared(rows)
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "_clean_committed_head", lambda: "a" * 40)
    monkeypatch.setattr(
        runner,
        "prepare_source_only_stage",
        lambda: prepared,
    )
    factory = _FakeFactory()
    payload = runner.run_stage(scorer_factory=factory)
    assert payload["decision"] == "pass"
    assert payload["terminal_action"] == runner.PASS_ACTION
    assert payload["open_2020_train_outcomes_authorized"] is True
    assert payload["repeatability_and_capacity_gate"]["decision"] == "pass"
    assert payload["model_load_evidence"] == {
        "gate_model_loads": 2,
        "full_extraction_model_load_ordinal": 3,
        "fresh_from_gate_models": True,
    }
    assert factory.events[:5] == [
        "construct-0",
        "close-0",
        "construct-1",
        "close-1",
        "construct-2",
    ]
    assert factory.events[-1] == "close-2"
    assert factory.instances[0].embed_prompts == []
    assert factory.instances[1].embed_prompts == []
    assert factory.instances[2].embed_prompts == ["POLICY-0", "POLICY-1"]
    assert not Path(runner.DEFAULT_CHECKPOINT_DIRECTORY).exists()
    for path in (
        runner.DEFAULT_ATTEMPT,
        runner.DEFAULT_REPEATABILITY_RESULT,
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
    assert relation_rows[1]["predicted_relation"] == (
        "INSUFFICIENT_EVIDENCE"
    )


@pytest.mark.parametrize(
    "fault_label",
    [
        "artifact_promoted:source_rows",
        "artifact_promoted:embeddings",
        "artifact_promoted:relation_logits",
        "artifact_promoted:relation_rows",
    ],
)
def test_publish_interruption_rolls_back_visible_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fault_label: str,
) -> None:
    _patch_paths(monkeypatch, tmp_path)
    checkpoint = Path(runner.DEFAULT_CHECKPOINT_DIRECTORY)
    output = Path(runner.DEFAULT_OUTPUT)
    payload, records, raw = _publish_fixture(checkpoint)

    def interrupt(label: str) -> None:
        if label == fault_label:
            raise KeyboardInterrupt(fault_label)

    monkeypatch.setattr(runner, "_publish_progress", interrupt)
    with pytest.raises(KeyboardInterrupt, match="artifact_promoted"):
        runner._publish_success_atomically(
            checkpoint_directory=checkpoint,
            output_path=output,
            payload=payload,
            artifact_records=records,
            artifact_bytes=raw,
        )
    assert checkpoint.is_dir()
    assert not output.exists()
    assert not runner._publish_journal_path(output).exists()
    assert not runner._publish_result_staging_path(output).exists()
    assert all(
        not Path(str(record["path"])).exists()
        for record in records.values()
    )


def test_publish_commit_point_completes_result_after_interruption(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_paths(monkeypatch, tmp_path)
    checkpoint = Path(runner.DEFAULT_CHECKPOINT_DIRECTORY)
    output = Path(runner.DEFAULT_OUTPUT)
    payload, records, raw = _publish_fixture(checkpoint)

    def interrupt(label: str) -> None:
        if label == "checkpoint_removed_before_result":
            raise KeyboardInterrupt(label)

    monkeypatch.setattr(runner, "_publish_progress", interrupt)
    runner._publish_success_atomically(
        checkpoint_directory=checkpoint,
        output_path=output,
        payload=payload,
        artifact_records=records,
        artifact_bytes=raw,
    )
    assert not checkpoint.exists()
    assert json.loads(output.read_text()) == payload
    assert not runner._publish_journal_path(output).exists()
    assert not runner._publish_result_staging_path(output).exists()
    assert all(
        Path(str(record["path"])).is_file()
        for record in records.values()
    )


def test_publish_recovery_completes_verified_partial_transaction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_paths(monkeypatch, tmp_path)
    checkpoint = Path(runner.DEFAULT_CHECKPOINT_DIRECTORY)
    output = Path(runner.DEFAULT_OUTPUT)
    payload, records, raw = _publish_fixture(checkpoint)
    staging = checkpoint / "final_staging"
    staging.mkdir()
    for name, record in records.items():
        runner._write_new_bytes(
            staging / Path(str(record["path"])).name,
            raw[name],
        )
    result_staging = runner._publish_result_staging_path(output)
    runner._write_new_bytes(
        result_staging,
        runner.canonical_json_bytes(payload, pretty=True),
    )
    journal = runner._publish_journal_payload(
        checkpoint_directory=checkpoint,
        output_path=output,
        payload=payload,
        artifact_records=records,
        result_staging=result_staging,
    )
    runner._write_new_json(runner._publish_journal_path(output), journal)
    for name in ("source_rows", "embeddings"):
        record = records[name]
        target = Path(str(record["path"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        runner._promote_staged_file(
            staging / target.name,
            target,
        )
    recovered = runner._recover_publish_transaction(
        checkpoint_directory=checkpoint,
        output_path=output,
    )
    assert recovered == payload
    assert json.loads(output.read_text()) == payload
    assert not checkpoint.exists()
    assert not runner._publish_journal_path(output).exists()
    assert all(
        Path(str(record["path"])).is_file()
        for record in records.values()
    )


def test_checkpoint_cleanup_failure_cannot_publish_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_paths(monkeypatch, tmp_path)
    checkpoint = Path(runner.DEFAULT_CHECKPOINT_DIRECTORY)
    output = Path(runner.DEFAULT_OUTPUT)
    payload, records, raw = _publish_fixture(checkpoint)
    monkeypatch.setattr(
        runner.shutil,
        "rmtree",
        lambda _: (_ for _ in ()).throw(OSError("cleanup failed")),
    )
    with pytest.raises(OSError, match="cleanup failed"):
        runner._publish_success_atomically(
            checkpoint_directory=checkpoint,
            output_path=output,
            payload=payload,
            artifact_records=records,
            artifact_bytes=raw,
        )
    assert checkpoint.is_dir()
    assert not output.exists()
    assert not runner._publish_journal_path(output).exists()
    assert all(
        not Path(str(record["path"])).exists()
        for record in records.values()
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
    monkeypatch.setattr(runner, "_clean_committed_head", lambda: "c" * 40)
    monkeypatch.setattr(
        runner,
        "prepare_source_only_stage",
        lambda: prepared,
    )
    factory = _FakeFactory()
    original_clear = runner._clear_inflight_row
    monkeypatch.setattr(
        runner,
        "_clear_inflight_row",
        lambda _: (_ for _ in ()).throw(
            KeyboardInterrupt("committed-row interruption")
        ),
    )
    with pytest.raises(KeyboardInterrupt):
        runner.run_stage(scorer_factory=factory)
    assert not Path(runner.DEFAULT_OUTPUT).exists()
    assert Path(runner.DEFAULT_REPEATABILITY_RESULT).is_file()
    monkeypatch.setattr(runner, "_clear_inflight_row", original_clear)
    payload = runner.run_stage(scorer_factory=factory, resume=True)
    assert payload["checkpoint_evidence"]["resumed"] is True
    assert payload["checkpoint_evidence"][
        "resumed_from_completed_rows"
    ] == 1
    assert factory.instances[2].embed_prompts == ["POLICY-0"]
    assert factory.instances[3].embed_prompts == ["POLICY-1"]
    assert factory.instances[3].relation_prompts == []


def test_ambiguous_inflight_row_rejects_without_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepared([_row(0, relation_forwarded=True)])
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "_clean_committed_head", lambda: "d" * 40)
    monkeypatch.setattr(
        runner,
        "prepare_source_only_stage",
        lambda: prepared,
    )
    factory = _FakeFactory(interrupt_extraction_index=2)
    with pytest.raises(KeyboardInterrupt):
        runner.run_stage(scorer_factory=factory)
    assert not Path(runner.DEFAULT_OUTPUT).exists()
    assert len(factory.instances) == 3
    with pytest.raises(RuntimeError, match="ambiguous started forwards"):
        runner.run_stage(scorer_factory=factory, resume=True)
    assert len(factory.instances) == 3
    failure = json.loads(Path(runner.DEFAULT_OUTPUT).read_text())
    assert failure["decision"] == "reject"
    assert failure["observations"]["completed_source_rows"] == 0
    assert failure["rerun_authorized"] is False


def test_pre_gate_interruption_is_terminal_and_not_resumable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepared([_row(0, relation_forwarded=True)])
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "_clean_committed_head", lambda: "e" * 40)
    monkeypatch.setattr(
        runner,
        "prepare_source_only_stage",
        lambda: prepared,
    )
    factory = _FakeFactory(interrupt_gate=True)
    with pytest.raises(KeyboardInterrupt):
        runner.run_stage(scorer_factory=factory)
    failure = json.loads(Path(runner.DEFAULT_OUTPUT).read_text())
    gate = json.loads(
        Path(runner.DEFAULT_REPEATABILITY_RESULT).read_text()
    )
    assert failure["decision"] == "reject"
    assert failure["failure"]["stage"] == (
        "PRE_MARKET_REPEATABILITY_AND_CAPACITY"
    )
    assert gate["decision"] == "reject"
    assert factory.events == ["construct-0", "close-0"]
    with pytest.raises(RuntimeError, match="resume state is unavailable"):
        runner.run_stage(scorer_factory=factory, resume=True)


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


def test_runner_has_no_market_or_predecessor_output_import_surface() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
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
    assert "pre_market_equivalence_gate" not in source
    assert "reference_embedding" not in source
    assert "reference_relation" not in source
    for predecessor in (prereg.s2.s1, prereg.s2):
        for attribute in (
            "CHECKPOINT_DIRECTORY",
            "SOURCE_ROWS_PATH",
            "EMBEDDINGS_PATH",
            "RELATION_LOGITS_PATH",
            "RELATION_ROWS_PATH",
        ):
            assert getattr(predecessor, attribute).as_posix() not in source
    assert runner.DEFAULT_OUTPUT == prereg.RESULT_PATH
    assert runner.DEFAULT_ATTEMPT == prereg.ATTEMPT_PATH
    assert runner.DEFAULT_REPEATABILITY_RESULT == (
        prereg.REPEATABILITY_RESULT_PATH
    )
    contract = prereg.build_preregistration()
    assert runner.PASS_ACTION == contract["terminal_actions"]["success"]
    assert runner.FAILURE_ACTION == contract["terminal_actions"]["failure"]
