from __future__ import annotations

import ast
import gzip
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import zipfile

import numpy as np
import pytest

from training import preregister_psim_d8_rllm2_source_feature_seal as prereg
from training import run_psim_d8_rllm2_source_feature_seal as runner


def _row(
    index: int,
    *,
    relation_forwarded: bool,
) -> dict[str, Any]:
    code_to_label = {
        code: label
        for code, label in zip(
            prereg.RELATION_CODE_ORDER,
            prereg.rllm1.RELATION_LABELS,
        )
    }
    core = {
        "schema_version": prereg.SOURCE_ROW_SCHEMA_VERSION,
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
    return {
        "preregistration": {"validated": True},
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
    }


def _patch_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runner, "DEFAULT_ATTEMPT", tmp_path / "attempt.json")
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


def test_preregistration_is_exact() -> None:
    payload = runner.validate_preregistration()
    assert payload["candidate"]["id"] == prereg.STAGE_ID
    assert payload["manifest_hash"] == runner.PREREGISTRATION_MANIFEST_HASH
    assert (
        payload["source_row_contract"]["roster"][
            "source_row_roster_hash"
        ]
        == runner.SOURCE_ROW_ROSTER_HASH
    )
    assert payload["access_boundary"]["market_or_funding_paths_read"] == []


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
        assert all(
            info.external_attr == 0o600 << 16
            for info in archive.infolist()
        )


def test_checkpoint_chain_round_trip_and_tamper_rejection(
    tmp_path: Path,
) -> None:
    rows = [_row(0, relation_forwarded=True), _row(1, relation_forwarded=False)]
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "binding.json").write_text("{}", encoding="utf-8")
    first_arrays = {
        "row_index": np.asarray([0], dtype=np.int32),
        "embedding": np.ones(
            (1, prereg.EMBEDDING_WIDTH),
            dtype=np.float32,
        ),
        "relation_logits": np.asarray(
            [np.arange(6, dtype=np.float32)]
        ),
        "relation_forwarded": np.asarray([1], dtype=np.uint8),
        "policy_input_tokens": np.asarray([5], dtype=np.int32),
        "relation_input_tokens": np.asarray([7], dtype=np.int32),
    }
    prior = runner.canonical_hash(
        {
            "state": "PSIM_D8_RLLM2_S1_CHECKPOINT_CHAIN_START",
            "source_row_roster_hash": runner.SOURCE_ROW_ROSTER_HASH,
        }
    )
    first_metadata = runner._write_checkpoint_shard(
        checkpoint,
        shard_index=0,
        start=0,
        stop=1,
        prior_shard_hash=prior,
        rows=rows,
        arrays=first_arrays,
    )
    second_arrays = {
        "row_index": np.asarray([1], dtype=np.int32),
        "embedding": np.full(
            (1, prereg.EMBEDDING_WIDTH),
            2.0,
            dtype=np.float32,
        ),
        "relation_logits": np.asarray(
            [runner._canonical_nan_row(6)]
        ),
        "relation_forwarded": np.asarray([0], dtype=np.uint8),
        "policy_input_tokens": np.asarray([5], dtype=np.int32),
        "relation_input_tokens": np.asarray([7], dtype=np.int32),
    }
    second_metadata = runner._write_checkpoint_shard(
        checkpoint,
        shard_index=1,
        start=1,
        stop=2,
        prior_shard_hash=first_metadata["shard_hash"],
        rows=rows,
        arrays=second_arrays,
    )
    loaded, terminal, completed = runner._load_checkpoint_prefix(
        checkpoint,
        rows=rows,
    )
    assert completed == 2
    assert terminal == second_metadata["shard_hash"]
    np.testing.assert_array_equal(
        loaded[0]["embedding"],
        first_arrays["embedding"],
    )

    npz_path = checkpoint / "shard_0001.npz"
    original_npz = npz_path.read_bytes()
    npz_path.write_bytes(original_npz[:-1] + bytes([original_npz[-1] ^ 1]))
    with pytest.raises(RuntimeError, match="shard binding"):
        runner._load_checkpoint_prefix(checkpoint, rows=rows)
    npz_path.write_bytes(original_npz)

    metadata_path = checkpoint / "shard_0001.json"
    tampered = json.loads(metadata_path.read_text(encoding="utf-8"))
    tampered["stop"] = 3
    metadata_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="shard binding"):
        runner._load_checkpoint_prefix(checkpoint, rows=rows)


def test_success_path_writes_and_verifies_all_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = [_row(0, relation_forwarded=True), _row(1, relation_forwarded=False)]
    prepared = _prepared(rows)
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runner,
        "_clean_committed_head",
        lambda: "a" * 40,
    )
    monkeypatch.setattr(
        runner,
        "prepare_source_only_stage",
        lambda: prepared,
    )

    class FakeScorer:
        def __init__(self, _: Any) -> None:
            assert Path(runner.DEFAULT_ATTEMPT).is_file()
            self.forward_calls_started = 0
            self.embedding_forwards_started = 0
            self.relation_forwards_started = 0

        def embed(self, prompt: str) -> tuple[np.ndarray, int]:
            assert prompt.startswith("POLICY-")
            self.forward_calls_started += 1
            self.embedding_forwards_started += 1
            return (
                np.full(
                    prereg.EMBEDDING_WIDTH,
                    self.embedding_forwards_started,
                    dtype=np.float32,
                ),
                5,
            )

        def score_relation(
            self,
            prompt: str,
        ) -> tuple[np.ndarray, str, int]:
            assert prompt.startswith("RELATION-")
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
    assert payload["open_2021_or_later_outcomes_authorized"] is False
    assert payload["model_metrics"][
        "total_embedding_forwards_across_attempt"
    ] == 2
    assert payload["model_metrics"][
        "total_relation_forwards_across_attempt"
    ] == 1
    assert not Path(runner.DEFAULT_CHECKPOINT_DIRECTORY).exists()
    for path in (
        runner.DEFAULT_ATTEMPT,
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
    with pytest.raises(RuntimeError, match="already attempted"):
        runner.run_stage(scorer_factory=FakeScorer)


def test_interrupted_safe_boundary_resumes_without_replaying_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = [_row(0, relation_forwarded=True), _row(1, relation_forwarded=False)]
    prepared = _prepared(rows)
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runner,
        "_clean_committed_head",
        lambda: "c" * 40,
    )
    monkeypatch.setattr(
        runner,
        "prepare_source_only_stage",
        lambda: prepared,
    )
    instances: list[Any] = []

    class FakeScorer:
        def __init__(self, _: Any) -> None:
            self.forward_calls_started = 0
            self.embedding_forwards_started = 0
            self.relation_forwards_started = 0
            self.embedding_prompts: list[str] = []
            self.relation_prompts: list[str] = []
            instances.append(self)

        def embed(self, prompt: str) -> tuple[np.ndarray, int]:
            self.forward_calls_started += 1
            self.embedding_forwards_started += 1
            self.embedding_prompts.append(prompt)
            return np.ones(prereg.EMBEDDING_WIDTH, dtype=np.float32), 5

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

    def interrupt_after_first_row_commit(
        checkpoint_directory: Path,
    ) -> None:
        raise KeyboardInterrupt("synthetic committed-row interruption")

    monkeypatch.setattr(
        runner,
        "_clear_inflight_row",
        interrupt_after_first_row_commit,
    )
    with pytest.raises(KeyboardInterrupt):
        runner.run_stage(scorer_factory=FakeScorer)
    assert not Path(runner.DEFAULT_OUTPUT).exists()
    assert (Path(runner.DEFAULT_CHECKPOINT_DIRECTORY) / "shard_0000.json").is_file()
    assert (Path(runner.DEFAULT_CHECKPOINT_DIRECTORY) / "inflight.json").is_file()
    Path(runner.DEFAULT_SOURCE_ROWS).write_bytes(b"stale-partial-final")

    monkeypatch.setattr(
        runner,
        "_clear_inflight_row",
        original_clear,
    )
    payload = runner.run_stage(scorer_factory=FakeScorer, resume=True)
    assert payload["checkpoint_evidence"]["resumed"] is True
    assert payload["checkpoint_evidence"]["resumed_from_completed_rows"] == 1
    assert len(instances) == 2
    assert instances[0].embedding_prompts == ["POLICY-0"]
    assert instances[1].embedding_prompts == ["POLICY-1"]
    assert instances[1].relation_prompts == []
    assert Path(runner.DEFAULT_SOURCE_ROWS).read_bytes() != (
        b"stale-partial-final"
    )


def test_ambiguous_inflight_row_terminally_rejects_without_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = [_row(0, relation_forwarded=True)]
    prepared = _prepared(rows)
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runner,
        "_clean_committed_head",
        lambda: "d" * 40,
    )
    monkeypatch.setattr(
        runner,
        "prepare_source_only_stage",
        lambda: prepared,
    )

    class InterruptedScorer:
        def __init__(self, _: Any) -> None:
            self.forward_calls_started = 0
            self.embedding_forwards_started = 0
            self.relation_forwards_started = 0

        def embed(self, _: str) -> tuple[np.ndarray, int]:
            self.forward_calls_started += 1
            self.embedding_forwards_started += 1
            raise KeyboardInterrupt("synthetic ambiguous interruption")

    with pytest.raises(KeyboardInterrupt):
        runner.run_stage(scorer_factory=InterruptedScorer)
    inflight = Path(runner.DEFAULT_CHECKPOINT_DIRECTORY) / "inflight.json"
    assert inflight.is_file()

    def no_replay(_: Any) -> Any:
        pytest.fail("ambiguous row was replayed")

    with pytest.raises(RuntimeError, match="ambiguous started forwards"):
        runner.run_stage(scorer_factory=no_replay, resume=True)
    failure = json.loads(Path(runner.DEFAULT_OUTPUT).read_text())
    assert failure["decision"] == "reject"
    assert failure["failure"]["stage"] == "CHECKPOINT_BOOTSTRAP"
    assert failure["observations"]["completed_source_rows"] == 0
    assert failure["rerun_authorized"] is False


def test_all_forced_no_eligible_rows_never_call_relation_forward() -> None:
    rows = [_row(0, relation_forwarded=False), _row(1, relation_forwarded=False)]

    class EmbedOnlyScorer:
        def __init__(self) -> None:
            self.relation_calls = 0

        def embed(self, _: str) -> tuple[np.ndarray, int]:
            return np.ones(prereg.EMBEDDING_WIDTH, dtype=np.float32), 5

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
    relation_rows = runner._relation_rows(rows, merged)
    assert {
        row["predicted_relation"] for row in relation_rows
    } == {"INSUFFICIENT_EVIDENCE"}


def test_real_scorer_methods_use_frozen_model_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")

    class Inputs(dict[str, Any]):
        def to(self, _: Any) -> "Inputs":
            return self

    inputs = Inputs(
        input_ids=torch.asarray([[1, 2, 3]], dtype=torch.long),
        attention_mask=torch.asarray([[1, 1, 1]], dtype=torch.long),
    )
    monkeypatch.setattr(runner, "_prompt_inputs", lambda _p, _s: inputs)

    class InnerModel:
        def __init__(self) -> None:
            self.kwargs: dict[str, Any] = {}

        def __call__(self, **kwargs: Any) -> Any:
            self.kwargs = kwargs
            hidden = torch.arange(
                3 * prereg.EMBEDDING_WIDTH,
                dtype=torch.float32,
            ).reshape(1, 3, prereg.EMBEDDING_WIDTH)
            return type("Output", (), {"last_hidden_state": hidden})()

    class OuterModel:
        device = "cpu"

        def __init__(self) -> None:
            self.model = InnerModel()
            self.kwargs: dict[str, Any] = {}

        def __call__(self, **kwargs: Any) -> Any:
            self.kwargs = kwargs
            logits = torch.zeros((1, 1, 6), dtype=torch.float32)
            logits[0, 0, 5] = 3.0
            return type("Output", (), {"logits": logits})()

    scorer = runner.Gemma4SourceFeatureScorer.__new__(
        runner.Gemma4SourceFeatureScorer
    )
    scorer.torch = torch
    scorer.processor = object()
    scorer.model = OuterModel()
    scorer.code_ids = {
        code: index
        for index, code in enumerate(prereg.RELATION_CODE_ORDER)
    }
    scorer.forward_calls_started = 0
    scorer.embedding_forwards_started = 0
    scorer.relation_forwards_started = 0
    scorer.embedding_inference_seconds = 0.0
    scorer.relation_inference_seconds = 0.0

    vector, embedding_tokens = scorer.embed("policy")
    logits, code, relation_tokens = scorer.score_relation("relation")
    assert vector.shape == (prereg.EMBEDDING_WIDTH,)
    assert embedding_tokens == relation_tokens == 3
    assert code == "F"
    np.testing.assert_array_equal(logits, np.asarray([0, 0, 0, 0, 0, 3]))
    assert scorer.model.model.kwargs["use_cache"] is False
    assert scorer.model.model.kwargs["return_dict"] is True
    assert scorer.model.kwargs["use_cache"] is False
    assert scorer.model.kwargs["logits_to_keep"] == 1
    assert scorer.model.kwargs["return_dict"] is True


def test_caught_failure_is_terminal_and_not_resumable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = [_row(0, relation_forwarded=True)]
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runner,
        "_clean_committed_head",
        lambda: "b" * 40,
    )
    monkeypatch.setattr(
        runner,
        "prepare_source_only_stage",
        lambda: _prepared(rows),
    )

    class FailingScorer:
        def __init__(self, _: Any) -> None:
            assert Path(runner.DEFAULT_ATTEMPT).is_file()
            raise RuntimeError("synthetic model load failure")

    with pytest.raises(RuntimeError, match="synthetic model load failure"):
        runner.run_stage(scorer_factory=FailingScorer)
    failure = json.loads(Path(runner.DEFAULT_OUTPUT).read_text())
    assert failure["decision"] == "reject"
    assert failure["terminal_action"] == runner.FAILURE_ACTION
    assert failure["failure"]["stage"] == "MODEL_CONSTRUCTION"
    assert failure["resume_authorized"] is False
    assert failure["rerun_authorized"] is False
    assert failure["access_boundary"]["market_or_funding_paths_read"] == []
    with pytest.raises(RuntimeError, match="resume state is unavailable"):
        runner.run_stage(scorer_factory=FailingScorer, resume=True)


def test_runner_has_no_market_or_economic_import_surface() -> None:
    source_path = Path(runner.__file__)
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
            "pandas",
            "psycopg",
            "sqlalchemy",
        }
    )
    assert runner.DEFAULT_OUTPUT == prereg.RESULT_PATH
    assert runner.DEFAULT_ATTEMPT == prereg.ATTEMPT_PATH
    assert runner.PASS_ACTION == prereg.build_preregistration()[
        "terminal_actions"
    ]["success"]
    assert runner.FAILURE_ACTION == prereg.build_preregistration()[
        "terminal_actions"
    ]["failure"]
