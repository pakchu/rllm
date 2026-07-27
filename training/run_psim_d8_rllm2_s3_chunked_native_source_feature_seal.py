#!/usr/bin/env python3
"""Run the PSIM-D8-RLLM2-S3 chunked-native source-feature seal."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import io
import json
import shutil
import sys
import time
import zipfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_BOOTSTRAP_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_REPO_ROOT))

from training import (
    preregister_psim_d8_rllm2_s3_chunked_native_source_feature as prereg,
)
from training import run_psim_d8_rllm1_base_memorization_gate as base
from training import run_psim_d8_rllm2_base_memorization_gate as rllm2_gate
from utils import disable_transformers_allocator_warmup

s2 = prereg.s2
s1 = s2.s1

REPO_ROOT = _BOOTSTRAP_REPO_ROOT
PROTOCOL_VERSION = (
    "psim_d8_rllm2_s3_chunked_native_source_feature_seal_v1"
)
PREREGISTRATION = prereg.DEFAULT_OUTPUT
PREREGISTRATION_SHA256 = (
    "7edb7eeef115e0579099f9aa990648f1db1cde28053867c0ae1edb6c60deb196"
)
PREREGISTRATION_MANIFEST_HASH = (
    "263f0476ce887cad7b5e9c0174e02b6f714a8ce54029ef5788a6179ad1c396f3"
)
SOURCE_ROW_ROSTER_HASH = (
    "033df68d9067a88cb14eb83f92b7638f0addc2372ef08184ef75e9fe3f7ba47c"
)
PASS_ACTION = (
    "ACCEPT_PSIM_D8_RLLM2_S3_CHUNKED_NATIVE_SOURCE_"
    "FEATURE_SEAL_OPEN_2020_TRAIN_OUTCOMES_ONLY"
)
FAILURE_ACTION = (
    "REJECT_PSIM_D8_RLLM2_S3_NO_REPAIR_RERUN_MODEL_SWAP_"
    "OR_MARKET_ACCESS"
)
REPEATABILITY_PASS_ACTION = (
    "AUTHORIZE_S3_FULL_CHUNKED_NATIVE_SOURCE_EXTRACTION_ONLY"
)
REPEATABILITY_FAILURE_ACTION = (
    "TERMINAL_REJECT_S3_WITHOUT_FULL_EXTRACTION_OR_MARKET_OPEN"
)

DEFAULT_ATTEMPT = prereg.ATTEMPT_PATH
DEFAULT_REPEATABILITY_RESULT = prereg.REPEATABILITY_RESULT_PATH
DEFAULT_OUTPUT = prereg.RESULT_PATH
DEFAULT_SOURCE_ROWS = prereg.SOURCE_ROWS_PATH
DEFAULT_EMBEDDINGS = prereg.EMBEDDINGS_PATH
DEFAULT_RELATION_LOGITS = prereg.RELATION_LOGITS_PATH
DEFAULT_RELATION_ROWS = prereg.RELATION_ROWS_PATH
DEFAULT_CHECKPOINT_DIRECTORY = prereg.CHECKPOINT_DIRECTORY


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def canonical_json_bytes(payload: Any, *, pretty: bool = True) -> bytes:
    return prereg.canonical_json_bytes(payload, pretty=pretty)


def canonical_hash(payload: Any) -> str:
    return prereg.canonical_hash(payload)


def sha256_file(path: str | Path) -> str:
    target = repository_path(path)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"unsafe RLLM2-S3 file: {path}")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    return base._display_path(path)


def _clean_committed_head() -> str:
    return base._clean_committed_head()


def validate_preregistration() -> dict[str, Any]:
    target = repository_path(PREREGISTRATION)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError("unsafe RLLM2-S3 preregistration artifact")
    raw = target.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PREREGISTRATION_SHA256:
        raise RuntimeError("RLLM2-S3 preregistration SHA changed")
    payload = json.loads(raw.decode("utf-8"))
    if (
        payload != prereg.build_preregistration()
        or payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH
        or payload.get("candidate", {}).get("id") != prereg.STAGE_ID
        or payload.get("scientific_contract", {})
        .get("source_row_roster", {})
        .get("source_row_roster_hash")
        != SOURCE_ROW_ROSTER_HASH
        or payload.get(
            "pre_market_repeatability_and_capacity_gate", {}
        ).get(
            "pass_action"
        )
        != REPEATABILITY_PASS_ACTION
        or payload.get(
            "pre_market_repeatability_and_capacity_gate", {}
        ).get(
            "failure_action"
        )
        != REPEATABILITY_FAILURE_ACTION
        or payload.get("execution_contract", {}).get(
            "s1_or_s2_checkpoint_or_model_outputs_read"
        )
        is not False
        or payload.get("access_boundary", {}).get(
            "market_or_funding_payload_bytes_hashed"
        )
        is not False
    ):
        raise RuntimeError("RLLM2-S3 preregistration contract changed")
    return payload


def _prompt_inputs(processor: Any, prompt: str) -> Any:
    return base._processor_inputs(processor, prompt)


def validate_prompt_capacity(
    processor: Any,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    policy_counts: list[int] = []
    relation_counts: list[int] = []
    policy_max: dict[str, Any] | None = None
    relation_max: dict[str, Any] | None = None
    for row in rows:
        policy_count = int(
            _prompt_inputs(
                processor,
                str(row["policy_prompt"]),
            )["input_ids"].shape[-1]
        )
        relation_count = int(
            _prompt_inputs(
                processor,
                str(row["relation_teacher_prompt"]),
            )["input_ids"].shape[-1]
        )
        if (
            policy_count > prereg.MAXIMUM_INPUT_TOKENS
            or relation_count > prereg.MAXIMUM_INPUT_TOKENS
        ):
            raise RuntimeError("RLLM2-S3 prompt exceeds frozen token cap")
        policy_counts.append(policy_count)
        relation_counts.append(relation_count)
        identity = {
            "row_index": row["row_index"],
            "card_hash": row["card_hash"],
            "decision_at": row["decision_at"],
        }
        if policy_max is None or policy_count > int(policy_max["tokens"]):
            policy_max = {**identity, "tokens": policy_count}
        if (
            relation_max is None
            or relation_count > int(relation_max["tokens"])
        ):
            relation_max = {**identity, "tokens": relation_count}
    if len(policy_counts) != 1_461 or len(relation_counts) != 1_461:
        raise RuntimeError("RLLM2-S3 prompt roster changed")
    return {
        "maximum_input_tokens": prereg.MAXIMUM_INPUT_TOKENS,
        "truncation": False,
        "policy": {
            "count": len(policy_counts),
            "minimum_tokens": min(policy_counts),
            "maximum_tokens": max(policy_counts),
            "mean_tokens": sum(policy_counts) / len(policy_counts),
            "maximum_identity": policy_max,
            "counts": policy_counts,
        },
        "relation_teacher": {
            "count": len(relation_counts),
            "minimum_tokens": min(relation_counts),
            "maximum_tokens": max(relation_counts),
            "mean_tokens": sum(relation_counts) / len(relation_counts),
            "maximum_identity": relation_max,
            "counts": relation_counts,
        },
    }


def prepare_source_only_stage() -> dict[str, Any]:
    preregistration = validate_preregistration()
    runtime = base.validate_local_runtime(load_processor=False)
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        base._model_snapshot(),
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer = base.validate_processor_tokenizer(processor)
    code_ids = {
        code: tokenizer["challenge_code_token_ids"][code]
        for code in s1.RELATION_CODE_ORDER
    }
    if len(set(code_ids.values())) != len(code_ids):
        raise RuntimeError("RLLM2-S3 relation code token IDs collide")
    runtime["tokenizer"] = {
        **tokenizer,
        "relation_code_token_ids": code_ids,
    }
    runtime["chat_template"] = base.validate_chat_template_contract(
        processor
    )
    rows = prereg.build_source_rows()
    roster = s1.source_roster_contract(rows)
    if roster["source_row_roster_hash"] != SOURCE_ROW_ROSTER_HASH:
        raise RuntimeError("RLLM2-S3 source roster changed")
    capacity = validate_prompt_capacity(processor, rows)
    selected = s2.select_equivalence_cases(
        rows,
        policy_token_counts=capacity["policy"]["counts"],
        relation_token_counts=capacity["relation_teacher"]["counts"],
    )
    frozen_cases = s2.validate_frozen_cases(rows)
    if (
        selected != frozen_cases["equivalence_cases"]
        or frozen_cases["equivalence_case_roster_hash"]
        != prereg.build_preregistration()[
            "pre_market_repeatability_and_capacity_gate"
        ][
            "repeatability_case_roster_hash"
        ]
        or capacity["policy"]["counts"][
            int(prereg.CAPACITY_CASE["row_index"])
        ]
        != int(prereg.CAPACITY_CASE["policy_tokens"])
        or capacity["relation_teacher"]["counts"][
            int(prereg.CAPACITY_CASE["row_index"])
        ]
        != int(prereg.CAPACITY_CASE["relation_tokens"])
    ):
        raise RuntimeError("RLLM2-S3 frozen gate cases changed")
    return {
        "preregistration": preregistration,
        "runtime": runtime,
        "processor": processor,
        "rows": rows,
        "roster": roster,
        "prompt_capacity": capacity,
        "repeatability_cases": selected,
        "capacity_case": frozen_cases["capacity_case"],
    }


def _deterministic_npz_bytes(
    arrays: Mapping[str, np.ndarray],
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.save(
                buffer,
                np.asarray(arrays[name]),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(
                f"{name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())
    return output.getvalue()


def _deterministic_jsonl_gzip_bytes(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[bytes, str]:
    decompressed = b"".join(
        canonical_json_bytes(dict(row), pretty=False) + b"\n"
        for row in rows
    )
    return (
        gzip.compress(decompressed, compresslevel=9, mtime=0),
        hashlib.sha256(decompressed).hexdigest(),
    )


def _write_new_bytes(path: Path, raw: bytes) -> None:
    target = repository_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise RuntimeError(f"RLLM2-S3 output already exists: {path}")
    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    with temporary.open("xb") as handle:
        handle.write(raw)
        handle.flush()
    temporary.replace(target)


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_new_bytes(path, canonical_json_bytes(dict(payload), pretty=True))


def _read_exact_json(path: Path) -> dict[str, Any]:
    target = repository_path(path)
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"unsafe RLLM2-S3 JSON: {path}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(  # noqa: TRY004
            f"malformed RLLM2-S3 JSON: {path}"
        )
    return payload


def _attempt_payload(
    *,
    execution_commit: str,
    output_path: Path,
    checkpoint_directory: Path,
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "stage_id": prereg.STAGE_ID,
        "execution_commit": execution_commit,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": sha256_file(Path(__file__)),
        "preregistration": {
            "path": PREREGISTRATION.as_posix(),
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        },
        "predecessor_terminal_result_hash": (
            prereg.S2_FAILURE_RESULT_HASH
        ),
        "source_row_roster_hash": prepared["roster"][
            "source_row_roster_hash"
        ],
        "output": _display_path(output_path),
        "repeatability_result": _display_path(
            repository_path(DEFAULT_REPEATABILITY_RESULT)
        ),
        "artifact_paths": {
            "source_rows": DEFAULT_SOURCE_ROWS.as_posix(),
            "embeddings": DEFAULT_EMBEDDINGS.as_posix(),
            "relation_logits": DEFAULT_RELATION_LOGITS.as_posix(),
            "relation_rows": DEFAULT_RELATION_ROWS.as_posix(),
        },
        "checkpoint_directory": _display_path(checkpoint_directory),
        "checkpoint_shard_size": prereg.CHECKPOINT_SHARD_SIZE,
        "model_id": s1.rllm1.MODEL_ID,
        "model_revision": s1.rllm1.MODEL_REVISION,
        "model_inference_authorized": True,
        "market_access_authorized": False,
        "s1_or_s2_checkpoint_or_model_outputs_read": False,
        "resume_before_repeatability_pass_authorized": False,
        "resume_after_repeatability_pass_authorized": True,
    }
    return {**core, "attempt_hash": canonical_hash(core)}


def _validate_resume_attempt(
    attempt: Mapping[str, Any],
    *,
    execution_commit: str,
    prepared: Mapping[str, Any],
    output_path: Path,
    checkpoint_directory: Path,
) -> None:
    core = {
        key: value for key, value in attempt.items() if key != "attempt_hash"
    }
    expected_keys = {
        "protocol_version",
        "stage_id",
        "execution_commit",
        "started_at_utc",
        "runner_sha256",
        "preregistration",
        "predecessor_terminal_result_hash",
        "source_row_roster_hash",
        "output",
        "repeatability_result",
        "artifact_paths",
        "checkpoint_directory",
        "checkpoint_shard_size",
        "model_id",
        "model_revision",
        "model_inference_authorized",
        "market_access_authorized",
        "s1_or_s2_checkpoint_or_model_outputs_read",
        "resume_before_repeatability_pass_authorized",
        "resume_after_repeatability_pass_authorized",
        "attempt_hash",
    }
    if (
        set(attempt) != expected_keys
        or attempt.get("attempt_hash") != canonical_hash(core)
        or attempt.get("protocol_version") != PROTOCOL_VERSION
        or attempt.get("stage_id") != prereg.STAGE_ID
        or attempt.get("execution_commit") != execution_commit
        or not isinstance(attempt.get("started_at_utc"), str)
        or attempt.get("runner_sha256") != sha256_file(Path(__file__))
        or attempt.get("preregistration")
        != {
            "path": PREREGISTRATION.as_posix(),
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        }
        or attempt.get("predecessor_terminal_result_hash")
        != prereg.S2_FAILURE_RESULT_HASH
        or attempt.get("source_row_roster_hash")
        != prepared["roster"]["source_row_roster_hash"]
        or attempt.get("output") != _display_path(output_path)
        or attempt.get("repeatability_result")
        != _display_path(repository_path(DEFAULT_REPEATABILITY_RESULT))
        or attempt.get("artifact_paths")
        != {
            "source_rows": DEFAULT_SOURCE_ROWS.as_posix(),
            "embeddings": DEFAULT_EMBEDDINGS.as_posix(),
            "relation_logits": DEFAULT_RELATION_LOGITS.as_posix(),
            "relation_rows": DEFAULT_RELATION_ROWS.as_posix(),
        }
        or attempt.get("checkpoint_directory")
        != _display_path(checkpoint_directory)
        or attempt.get("checkpoint_shard_size")
        != prereg.CHECKPOINT_SHARD_SIZE
        or attempt.get("model_id") != s1.rllm1.MODEL_ID
        or attempt.get("model_revision") != s1.rllm1.MODEL_REVISION
        or attempt.get("model_inference_authorized") is not True
        or attempt.get("market_access_authorized") is not False
        or attempt.get("s1_or_s2_checkpoint_or_model_outputs_read")
        is not False
        or attempt.get("resume_before_repeatability_pass_authorized")
        is not False
        or attempt.get("resume_after_repeatability_pass_authorized")
        is not True
    ):
        raise RuntimeError("RLLM2-S3 resume attempt binding changed")


def _checkpoint_binding(
    *,
    attempt_path: Path,
    attempt: Mapping[str, Any],
    prepared: Mapping[str, Any],
    repeatability_result: Mapping[str, Any],
) -> dict[str, Any]:
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "attempt": {
            "path": _display_path(repository_path(attempt_path)),
            "sha256": sha256_file(attempt_path),
            "attempt_hash": attempt["attempt_hash"],
        },
        "execution_commit": attempt["execution_commit"],
        "runner_sha256": attempt["runner_sha256"],
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "preregistration_manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        "repeatability_result": {
            "path": _display_path(
                repository_path(DEFAULT_REPEATABILITY_RESULT)
            ),
            "sha256": sha256_file(DEFAULT_REPEATABILITY_RESULT),
            "result_hash": repeatability_result["result_hash"],
        },
        "source_row_roster_hash": prepared["roster"][
            "source_row_roster_hash"
        ],
        "row_count": len(prepared["rows"]),
        "shard_size": prereg.CHECKPOINT_SHARD_SIZE,
    }
    return {**core, "binding_hash": canonical_hash(core)}


def _create_checkpoint_directory(
    checkpoint_directory: Path,
    binding: Mapping[str, Any],
) -> None:
    target = repository_path(checkpoint_directory)
    if target.exists() or target.is_symlink():
        raise RuntimeError("RLLM2-S3 checkpoint directory already exists")
    target.mkdir(parents=True)
    _write_new_json(target / "binding.json", binding)


def _validate_checkpoint_binding(
    checkpoint_directory: Path,
    expected: Mapping[str, Any],
) -> None:
    target = repository_path(checkpoint_directory)
    if target.is_symlink() or not target.is_dir():
        raise RuntimeError("RLLM2-S3 checkpoint directory is unsafe")
    observed = _read_exact_json(target / "binding.json")
    if observed != dict(expected):
        raise RuntimeError("RLLM2-S3 checkpoint binding changed")


def _shard_paths(
    checkpoint_directory: Path,
    shard_index: int,
) -> tuple[Path, Path]:
    root = repository_path(checkpoint_directory)
    stem = f"shard_{shard_index:04d}"
    return root / f"{stem}.npz", root / f"{stem}.json"


def _canonical_nan_row(width: int) -> np.ndarray:
    return np.full(width, np.float32(np.nan), dtype=np.float32)


def _normalize_relation_logits(values: Sequence[float]) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float32)
    if logits.shape != (len(s1.RELATION_CODE_ORDER),):
        raise RuntimeError("RLLM2-S3 relation logit width changed")
    if not np.all(np.isfinite(logits)):
        return _canonical_nan_row(len(s1.RELATION_CODE_ORDER))
    return logits


def _predicted_code(logits: np.ndarray) -> str | None:
    values = np.asarray(logits, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        return None
    best = float(np.max(values))
    return next(
        code
        for code, value in zip(s1.RELATION_CODE_ORDER, values)
        if float(value) == best
    )


def _validate_shard_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    rows: Sequence[Mapping[str, Any]],
    start: int,
    stop: int,
) -> None:
    expected_names = {
        "embedding",
        "policy_input_tokens",
        "relation_forwarded",
        "relation_input_tokens",
        "relation_logits",
        "row_index",
    }
    if set(arrays) != expected_names:
        raise RuntimeError("RLLM2-S3 checkpoint array names changed")
    count = stop - start
    if (
        arrays["embedding"].shape
        != (count, s1.EMBEDDING_WIDTH)
        or arrays["embedding"].dtype != np.float32
        or arrays["relation_logits"].shape
        != (count, len(s1.RELATION_CODE_ORDER))
        or arrays["relation_logits"].dtype != np.float32
        or arrays["row_index"].shape != (count,)
        or arrays["policy_input_tokens"].shape != (count,)
        or arrays["relation_input_tokens"].shape != (count,)
        or arrays["relation_forwarded"].shape != (count,)
    ):
        raise RuntimeError("RLLM2-S3 checkpoint array shape changed")
    if not np.array_equal(
        arrays["row_index"],
        np.arange(start, stop, dtype=np.int32),
    ):
        raise RuntimeError("RLLM2-S3 checkpoint row order changed")
    if not np.all(np.isfinite(arrays["embedding"])):
        raise RuntimeError("RLLM2-S3 embedding contains nonfinite values")
    for local_index, row in enumerate(rows[start:stop]):
        forwarded = int(arrays["relation_forwarded"][local_index])
        expected_forwarded = int(
            bool(row["relation_teacher_forward_required"])
        )
        logits = arrays["relation_logits"][local_index]
        if forwarded != expected_forwarded:
            raise RuntimeError("RLLM2-S3 relation forward mask changed")
        if forwarded:
            if not (
                np.all(np.isfinite(logits))
                or np.all(np.isnan(logits))
            ):
                raise RuntimeError(
                    "RLLM2-S3 relation logits are partially nonfinite"
                )
        elif not np.all(np.isnan(logits)):
            raise RuntimeError(
                "RLLM2-S3 skipped relation logits are not canonical NaN"
            )
    for name in ("policy_input_tokens", "relation_input_tokens"):
        counts = arrays[name]
        if (
            np.any(counts <= 0)
            or np.any(counts > prereg.MAXIMUM_INPUT_TOKENS)
        ):
            raise RuntimeError("RLLM2-S3 checkpoint token count changed")


def _write_checkpoint_shard(
    checkpoint_directory: Path,
    *,
    shard_index: int,
    start: int,
    stop: int,
    prior_shard_hash: str,
    rows: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    _validate_shard_arrays(arrays, rows=rows, start=start, stop=stop)
    npz_path, metadata_path = _shard_paths(
        checkpoint_directory,
        shard_index,
    )
    raw = _deterministic_npz_bytes(arrays)
    _write_new_bytes(npz_path, raw)
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "shard_index": shard_index,
        "start": start,
        "stop": stop,
        "row_count": stop - start,
        "prior_shard_hash": prior_shard_hash,
        "source_row_hashes": [
            str(row["row_hash"]) for row in rows[start:stop]
        ],
        "arrays_path": _display_path(npz_path),
        "arrays_sha256": hashlib.sha256(raw).hexdigest(),
    }
    payload = {**core, "shard_hash": canonical_hash(core)}
    _write_new_json(metadata_path, payload)
    return payload


def _read_checkpoint_shard(
    checkpoint_directory: Path,
    *,
    shard_index: int,
    expected_start: int,
    expected_prior_hash: str,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    npz_path, metadata_path = _shard_paths(
        checkpoint_directory,
        shard_index,
    )
    metadata = _read_exact_json(metadata_path)
    core = {
        key: value for key, value in metadata.items() if key != "shard_hash"
    }
    stop = min(expected_start + prereg.CHECKPOINT_SHARD_SIZE, len(rows))
    if (
        metadata.get("shard_hash") != canonical_hash(core)
        or metadata.get("shard_index") != shard_index
        or metadata.get("start") != expected_start
        or metadata.get("stop") != stop
        or metadata.get("row_count") != stop - expected_start
        or metadata.get("prior_shard_hash") != expected_prior_hash
        or metadata.get("source_row_hashes")
        != [
            str(row["row_hash"])
            for row in rows[expected_start:stop]
        ]
        or metadata.get("arrays_path") != _display_path(npz_path)
        or metadata.get("arrays_sha256") != sha256_file(npz_path)
    ):
        raise RuntimeError("RLLM2-S3 checkpoint shard binding changed")
    try:
        with np.load(npz_path, allow_pickle=False) as payload:
            arrays = {
                name: np.asarray(payload[name])
                for name in payload.files
            }
    except Exception as exc:
        raise RuntimeError(
            f"RLLM2-S3 checkpoint shard unreadable: {exc}"
        ) from exc
    _validate_shard_arrays(
        arrays,
        rows=rows,
        start=expected_start,
        stop=stop,
    )
    return metadata, arrays


def _remove_uncommitted_checkpoint_files(
    checkpoint_directory: Path,
) -> None:
    root = repository_path(checkpoint_directory)
    metadata_stems = {
        path.stem
        for path in root.glob("shard_*.json")
        if path.is_file() and not path.is_symlink()
    }
    for path in root.iterdir():
        if path.name == "final_staging":
            if path.is_symlink() or not path.is_dir():
                raise RuntimeError("unsafe RLLM2-S3 final staging entry")
            shutil.rmtree(path)
        elif path.name.endswith(".tmp"):
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("unsafe RLLM2-S3 checkpoint temporary")
            path.unlink()
        elif (
            path.name.startswith("shard_")
            and path.suffix == ".npz"
            and path.stem not in metadata_stems
        ):
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("unsafe RLLM2-S3 orphan checkpoint")
            path.unlink()


def _load_checkpoint_prefix(
    checkpoint_directory: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, np.ndarray]], str, int]:
    root = repository_path(checkpoint_directory)
    _remove_uncommitted_checkpoint_files(checkpoint_directory)
    allowed = {"binding.json", "inflight.json"}
    metadata_indices: list[int] = []
    for path in root.iterdir():
        if path.name in allowed:
            continue
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("unsafe RLLM2-S3 checkpoint entry")
        if path.name.startswith("shard_") and path.suffix in {
            ".json",
            ".npz",
        }:
            try:
                index = int(path.stem.split("_", 1)[1])
            except (IndexError, ValueError) as exc:
                raise RuntimeError(
                    "malformed RLLM2-S3 checkpoint filename"
                ) from exc
            if path.suffix == ".json":
                metadata_indices.append(index)
            continue
        raise RuntimeError(
            f"unexpected RLLM2-S3 checkpoint entry: {path.name}"
        )
    ordered = sorted(metadata_indices)
    if ordered != list(range(len(ordered))):
        raise RuntimeError(
            "RLLM2-S3 checkpoints are not a contiguous prefix"
        )
    shards: list[dict[str, np.ndarray]] = []
    prior = canonical_hash(
        {
            "state": "PSIM_D8_RLLM2_S3_CHECKPOINT_CHAIN_START",
            "source_row_roster_hash": SOURCE_ROW_ROSTER_HASH,
        }
    )
    start = 0
    for index in ordered:
        metadata, arrays = _read_checkpoint_shard(
            checkpoint_directory,
            shard_index=index,
            expected_start=start,
            expected_prior_hash=prior,
            rows=rows,
        )
        shards.append(arrays)
        prior = str(metadata["shard_hash"])
        start = int(metadata["stop"])
    return shards, prior, start


def _inflight_path(checkpoint_directory: Path) -> Path:
    return repository_path(checkpoint_directory) / "inflight.json"


def _begin_inflight_row(
    checkpoint_directory: Path,
    *,
    row: Mapping[str, Any],
    prior_shard_hash: str,
) -> dict[str, Any]:
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "row_index": row["row_index"],
        "source_row_hash": row["row_hash"],
        "prior_shard_hash": prior_shard_hash,
        "recorded_before_any_row_forward": True,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    payload = {**core, "inflight_hash": canonical_hash(core)}
    _write_new_json(_inflight_path(checkpoint_directory), payload)
    return payload


def _clear_inflight_row(checkpoint_directory: Path) -> None:
    path = _inflight_path(checkpoint_directory)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("RLLM2-S3 in-flight sentinel is unsafe")
    path.unlink()


def _resolve_inflight_row(
    checkpoint_directory: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    completed_rows: int,
    prior_shard_hash: str,
) -> None:
    path = _inflight_path(checkpoint_directory)
    if not path.exists() and not path.is_symlink():
        return
    payload = _read_exact_json(path)
    core = {
        key: value
        for key, value in payload.items()
        if key != "inflight_hash"
    }
    row_index = payload.get("row_index")
    if (
        payload.get("inflight_hash") != canonical_hash(core)
        or not isinstance(row_index, int)
        or row_index < 0
        or row_index >= len(rows)
        or payload.get("source_row_hash") != rows[row_index]["row_hash"]
        or payload.get("recorded_before_any_row_forward") is not True
    ):
        raise RuntimeError("RLLM2-S3 in-flight binding changed")
    if completed_rows == row_index + 1:
        if prior_shard_hash == payload.get("prior_shard_hash"):
            raise RuntimeError(
                "RLLM2-S3 committed row did not advance checkpoint chain"
            )
        _clear_inflight_row(checkpoint_directory)
        return
    if completed_rows == row_index:
        raise RuntimeError(
            "RLLM2-S3 interrupted row has ambiguous started forwards"
        )
    raise RuntimeError("RLLM2-S3 in-flight row is outside checkpoint prefix")


class Gemma4SourceFeatureScorer:
    def __init__(self, processor: Any) -> None:
        import torch
        from transformers import (
            AutoModelForMultimodalLM,
            BitsAndBytesConfig,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("RLLM2-S3 requires CUDA")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("RLLM2-S3 requires one visible CUDA device")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("RLLM2-S3 requires CUDA BF16")
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        self.torch = torch
        self.processor = processor
        self.closed = False
        self.forward_calls_started = 0
        self.chunk_forwards_started = 0
        self.embedding_forwards_started = 0
        self.relation_forwards_started = 0
        self.embedding_inference_seconds = 0.0
        self.relation_inference_seconds = 0.0
        tokenizer_contract = base.validate_processor_tokenizer(processor)
        self.code_ids = {
            code: tokenizer_contract["challenge_code_token_ids"][code]
            for code in s1.RELATION_CODE_ORDER
        }
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        started = time.perf_counter()
        with disable_transformers_allocator_warmup():
            self.model = AutoModelForMultimodalLM.from_pretrained(
                base._model_snapshot(),
                local_files_only=True,
                quantization_config=quantization,
                device_map={"": 0},
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                trust_remote_code=False,
            ).eval()
        self.load_seconds = time.perf_counter() - started
        self.placement = rllm2_gate.validate_loaded_model_placement(
            self.model,
            torch,
        )
        text_config = self.model.config.get_text_config()
        if int(text_config.sliding_window) != prereg.CHUNK_SIZE:
            raise RuntimeError("RLLM2-S3 frozen sliding window changed")
        self.final_logit_softcapping = (
            text_config.final_logit_softcapping
        )

    def _inputs(self, prompt: str) -> tuple[Any, int]:
        inputs = _prompt_inputs(self.processor, prompt)
        tokens = int(inputs["input_ids"].shape[-1])
        if tokens > prereg.MAXIMUM_INPUT_TOKENS:
            raise RuntimeError("RLLM2-S3 prompt exceeds frozen token cap")
        if int(inputs["attention_mask"][0, -1].item()) != 1:
            raise RuntimeError("RLLM2-S3 final prompt token is padding")
        return inputs.to(self.model.device), tokens

    def _empty_prompt_cache(self) -> None:
        self.torch.cuda.empty_cache()

    def _validate_embedding(self, vector: np.ndarray) -> None:
        if vector.shape != (s1.EMBEDDING_WIDTH,):
            raise RuntimeError("RLLM2-S3 embedding width changed")
        if not np.all(np.isfinite(vector)):
            raise RuntimeError("RLLM2-S3 embedding is nonfinite")

    def _chunked_prompt(
        self,
        prompt: str,
        *,
        relation_logits: bool,
    ) -> tuple[np.ndarray, int, dict[str, Any]]:
        inputs, tokens = self._inputs(prompt)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        reconstructed: list[Any] = []
        past_key_values: Any | None = None
        chunk_count = 0
        started = time.perf_counter()
        final_hidden: Any | None = None
        outputs: Any | None = None
        reconstructed_ids: Any | None = None
        final: Any | None = None
        chunk_ids: Any | None = None
        position_ids: Any | None = None
        full_prefix_mask: Any | None = None
        try:
            with self.torch.inference_mode():
                for start in range(0, tokens, prereg.CHUNK_SIZE):
                    stop = min(start + prereg.CHUNK_SIZE, tokens)
                    chunk_ids = input_ids[:, start:stop]
                    reconstructed.append(chunk_ids)
                    position_ids = self.torch.arange(
                        start,
                        stop,
                        dtype=self.torch.long,
                        device=input_ids.device,
                    ).unsqueeze(0)
                    full_prefix_mask = attention_mask[:, :stop]
                    self.forward_calls_started += 1
                    self.chunk_forwards_started += 1
                    chunk_count += 1
                    outputs = self.model.model(
                        input_ids=chunk_ids,
                        attention_mask=full_prefix_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                    past_key_values = outputs.past_key_values
                    final_hidden = outputs.last_hidden_state[0, -1]
            if final_hidden is None:
                raise RuntimeError("RLLM2-S3 chunk scan produced no hidden state")
            reconstructed_ids = self.torch.cat(reconstructed, dim=1)
            reconstruction_exact = bool(
                self.torch.equal(reconstructed_ids, input_ids)
            )
            if not reconstruction_exact:
                raise RuntimeError("RLLM2-S3 token reconstruction changed")
            if relation_logits:
                with self.torch.inference_mode():
                    final = self.model.lm_head(final_hidden)
                    if self.final_logit_softcapping is not None:
                        final = final / self.final_logit_softcapping
                        final = self.torch.tanh(final)
                        final = final * self.final_logit_softcapping
                value = _normalize_relation_logits(
                    [
                        float(final[self.code_ids[code]].float().item())
                        for code in s1.RELATION_CODE_ORDER
                    ]
                )
            else:
                value = (
                    final_hidden.float()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                self._validate_embedding(value)
            evidence = {
                "input_tokens": tokens,
                "chunk_size_tokens": prereg.CHUNK_SIZE,
                "chunk_count": chunk_count,
                "token_reconstruction_exact": reconstruction_exact,
            }
            return value, tokens, evidence
        finally:
            elapsed = time.perf_counter() - started
            if relation_logits:
                self.relation_inference_seconds += elapsed
            else:
                self.embedding_inference_seconds += elapsed
            past_key_values = None
            reconstructed.clear()
            inputs = None
            input_ids = None
            attention_mask = None
            outputs = None
            reconstructed_ids = None
            final_hidden = None
            final = None
            chunk_ids = None
            position_ids = None
            full_prefix_mask = None
            self._empty_prompt_cache()

    def embed(
        self,
        prompt: str,
    ) -> tuple[np.ndarray, int]:
        self.embedding_forwards_started += 1
        vector, tokens, _ = self._chunked_prompt(
            prompt,
            relation_logits=False,
        )
        return vector, tokens

    def score_relation(
        self,
        prompt: str,
    ) -> tuple[np.ndarray, str | None, int]:
        self.relation_forwards_started += 1
        logits, tokens, _ = self._chunked_prompt(
            prompt,
            relation_logits=True,
        )
        return logits, _predicted_code(logits), tokens

    def chunked_embedding_with_evidence(
        self,
        prompt: str,
    ) -> tuple[np.ndarray, int, dict[str, Any]]:
        self.embedding_forwards_started += 1
        return self._chunked_prompt(prompt, relation_logits=False)

    def chunked_relation_with_evidence(
        self,
        prompt: str,
    ) -> tuple[np.ndarray, str | None, int, dict[str, Any]]:
        self.relation_forwards_started += 1
        logits, tokens, evidence = self._chunked_prompt(
            prompt,
            relation_logits=True,
        )
        return logits, _predicted_code(logits), tokens, evidence

    def reset_peak_memory_stats(self) -> None:
        self.torch.cuda.reset_peak_memory_stats()

    def peak_allocated_bytes(self) -> int:
        return int(self.torch.cuda.max_memory_allocated())

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        model = self.model
        self.model = None
        del model
        gc.collect()
        self.torch.cuda.empty_cache()

    def metrics(self) -> dict[str, Any]:
        properties = self.torch.cuda.get_device_properties(0)
        peak_allocated = int(self.torch.cuda.max_memory_allocated())
        peak_reserved = int(self.torch.cuda.max_memory_reserved())
        if peak_allocated > prereg.MAXIMUM_PEAK_ALLOCATED_BYTES:
            raise RuntimeError("RLLM2-S3 peak VRAM cap exceeded")
        return {
            "gpu_name": properties.name,
            "gpu_total_memory_bytes": int(properties.total_memory),
            "cuda_device_count": self.torch.cuda.device_count(),
            "bf16_supported": self.torch.cuda.is_bf16_supported(),
            "model_load_seconds": self.load_seconds,
            "placement": self.placement,
            "model_forwards_started": self.forward_calls_started,
            "chunk_forwards_started": self.chunk_forwards_started,
            "embedding_forwards_started": (
                self.embedding_forwards_started
            ),
            "relation_forwards_started": self.relation_forwards_started,
            "embedding_inference_seconds": (
                self.embedding_inference_seconds
            ),
            "relation_inference_seconds": self.relation_inference_seconds,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "maximum_peak_allocated_bytes": (
                prereg.MAXIMUM_PEAK_ALLOCATED_BYTES
            ),
        }


def _float32_bytes_sha256(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<f4")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _relation_logits_are_canonical(values: np.ndarray) -> bool:
    normalized = _normalize_relation_logits(values)
    return bool(
        np.all(np.isfinite(normalized)) or np.all(np.isnan(normalized))
    )


def _close_gate_scorer(scorer: Any) -> None:
    close = getattr(scorer, "close", None)
    if not callable(close):
        raise TypeError("RLLM2-S3 scorer lacks destructive close")
    close()
    gc.collect()


def _repeatability_case_result(
    scorer: Any,
    *,
    case: Mapping[str, Any],
    row: Mapping[str, Any],
) -> dict[str, Any]:
    embedding, policy_tokens, policy_scan = (
        scorer.chunked_embedding_with_evidence(str(row["policy_prompt"]))
    )
    logits, code, relation_tokens, relation_scan = (
        scorer.chunked_relation_with_evidence(
            str(row["relation_teacher_prompt"])
        )
    )
    embedding_values = np.asarray(embedding, dtype=np.float32)
    relation_values = _normalize_relation_logits(logits)
    expected_policy_tokens = int(case["policy_tokens"])
    expected_relation_tokens = int(case["relation_tokens"])
    embedding_finite = bool(
        embedding_values.shape == (s1.EMBEDDING_WIDTH,)
        and np.all(np.isfinite(embedding_values))
    )
    relation_canonical = _relation_logits_are_canonical(relation_values)
    passed = bool(
        policy_tokens == expected_policy_tokens
        and relation_tokens == expected_relation_tokens
        and policy_scan.get("token_reconstruction_exact") is True
        and relation_scan.get("token_reconstruction_exact") is True
        and embedding_finite
        and relation_canonical
        and code == _predicted_code(relation_values)
    )
    return {
        "row_index": int(case["row_index"]),
        "row_hash": str(row["row_hash"]),
        "selection_reasons": list(case["selection_reasons"]),
        "policy_tokens": policy_tokens,
        "relation_tokens": relation_tokens,
        "policy_scan": dict(policy_scan),
        "relation_scan": dict(relation_scan),
        "embedding_shape": list(embedding_values.shape),
        "embedding_all_values_finite": embedding_finite,
        "embedding_float32_bytes_sha256": _float32_bytes_sha256(
            embedding_values
        ),
        "relation_logits_finite_or_canonical_nan": relation_canonical,
        "relation_logits_canonical_float32_bytes_sha256": _float32_bytes_sha256(
            relation_values
        ),
        "predicted_relation_code": code,
        "pass": passed,
    }


def _capacity_result(
    scorer: Any,
    *,
    case: Mapping[str, Any],
    row: Mapping[str, Any],
) -> dict[str, Any]:
    scorer.reset_peak_memory_stats()
    embedding, policy_tokens, policy_scan = (
        scorer.chunked_embedding_with_evidence(str(row["policy_prompt"]))
    )
    logits, code, relation_tokens, relation_scan = (
        scorer.chunked_relation_with_evidence(
            str(row["relation_teacher_prompt"])
        )
    )
    peak_allocated = scorer.peak_allocated_bytes()
    embedding_values = np.asarray(embedding, dtype=np.float32)
    relation_values = _normalize_relation_logits(logits)
    embedding_finite = bool(
        embedding_values.shape == (s1.EMBEDDING_WIDTH,)
        and np.all(np.isfinite(embedding_values))
    )
    relation_canonical = _relation_logits_are_canonical(relation_values)
    passed = bool(
        policy_tokens == int(case["policy_tokens"])
        and relation_tokens == int(case["relation_tokens"])
        and policy_scan.get("token_reconstruction_exact") is True
        and relation_scan.get("token_reconstruction_exact") is True
        and embedding_finite
        and relation_canonical
        and code == _predicted_code(relation_values)
        and isinstance(peak_allocated, int)
        and not isinstance(peak_allocated, bool)
        and 0 <= peak_allocated <= prereg.MAXIMUM_PEAK_ALLOCATED_BYTES
    )
    return {
        "row_index": int(case["row_index"]),
        "row_hash": str(row["row_hash"]),
        "policy_tokens": policy_tokens,
        "relation_tokens": relation_tokens,
        "policy_scan": dict(policy_scan),
        "relation_scan": dict(relation_scan),
        "embedding_shape": list(embedding_values.shape),
        "embedding_all_values_finite": embedding_finite,
        "embedding_float32_bytes_sha256": _float32_bytes_sha256(
            embedding_values
        ),
        "relation_logits_finite_or_canonical_nan": relation_canonical,
        "relation_logits_canonical_float32_bytes_sha256": _float32_bytes_sha256(
            relation_values
        ),
        "predicted_relation_code": code,
        "peak_allocated_bytes": peak_allocated,
        "maximum_peak_allocated_bytes": (
            prereg.MAXIMUM_PEAK_ALLOCATED_BYTES
        ),
        "outputs_reused_by_full_extraction": False,
        "pass": passed,
    }


def _cross_load_comparison(
    load_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    complete = len(load_results) == prereg.REPEAT_MODEL_LOAD_COUNT
    same_case_count = bool(
        complete
        and all(
            len(load["case_results"])
            == len(load_results[0]["case_results"])
            for load in load_results
        )
    )
    embedding_hashes_identical = bool(
        same_case_count
        and all(
            left["embedding_float32_bytes_sha256"]
            == right["embedding_float32_bytes_sha256"]
            for left, right in zip(
                load_results[0]["case_results"],
                load_results[1]["case_results"],
                strict=True,
            )
        )
    )
    relation_hashes_identical = bool(
        same_case_count
        and all(
            left["relation_logits_canonical_float32_bytes_sha256"]
            == right["relation_logits_canonical_float32_bytes_sha256"]
            for left, right in zip(
                load_results[0]["case_results"],
                load_results[1]["case_results"],
                strict=True,
            )
        )
    )
    relation_codes_identical = bool(
        same_case_count
        and all(
            left["predicted_relation_code"]
            == right["predicted_relation_code"]
            for left, right in zip(
                load_results[0]["case_results"],
                load_results[1]["case_results"],
                strict=True,
            )
        )
    )
    capacities = [load.get("capacity_result") for load in load_results]
    complete_capacity = bool(
        complete and all(isinstance(value, dict) for value in capacities)
    )
    capacity_embedding_hashes_identical = bool(
        complete_capacity
        and capacities[0]["embedding_float32_bytes_sha256"]
        == capacities[1]["embedding_float32_bytes_sha256"]
    )
    capacity_relation_hashes_identical = bool(
        complete_capacity
        and capacities[0]["relation_logits_canonical_float32_bytes_sha256"]
        == capacities[1]["relation_logits_canonical_float32_bytes_sha256"]
    )
    capacity_relation_codes_identical = bool(
        complete_capacity
        and capacities[0]["predicted_relation_code"]
        == capacities[1]["predicted_relation_code"]
    )
    all_peak_allocated_within_limit = bool(
        complete_capacity
        and all(
            isinstance(value["peak_allocated_bytes"], int)
            and not isinstance(value["peak_allocated_bytes"], bool)
            and 0
            <= value["peak_allocated_bytes"]
            <= prereg.MAXIMUM_PEAK_ALLOCATED_BYTES
            for value in capacities
        )
    )
    passed = bool(
        complete
        and all(load.get("pass") is True for load in load_results)
        and embedding_hashes_identical
        and relation_hashes_identical
        and relation_codes_identical
        and capacity_embedding_hashes_identical
        and capacity_relation_hashes_identical
        and capacity_relation_codes_identical
        and all_peak_allocated_within_limit
    )
    return {
        "model_loads_compared": len(load_results),
        "repeatability_case_count": (
            len(load_results[0]["case_results"]) if complete else 0
        ),
        "embedding_hashes_identical": embedding_hashes_identical,
        "relation_logit_hashes_identical": relation_hashes_identical,
        "relation_codes_identical": relation_codes_identical,
        "capacity_embedding_hashes_identical": (
            capacity_embedding_hashes_identical
        ),
        "capacity_relation_logit_hashes_identical": (
            capacity_relation_hashes_identical
        ),
        "capacity_relation_codes_identical": (
            capacity_relation_codes_identical
        ),
        "all_peak_allocated_within_limit": (
            all_peak_allocated_within_limit
        ),
        "pass": passed,
    }


def _gate_result_payload(
    *,
    execution_commit: str,
    attempt_path: Path,
    attempt: Mapping[str, Any],
    prepared: Mapping[str, Any],
    decision: str,
    load_results: Sequence[Mapping[str, Any]],
    cross_load_comparison: Mapping[str, Any] | None,
    failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    gate = prepared["preregistration"][
        "pre_market_repeatability_and_capacity_gate"
    ]
    core: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "stage_id": prereg.STAGE_ID,
        "execution_commit": execution_commit,
        "runner_sha256": sha256_file(Path(__file__)),
        "attempt": {
            "path": _display_path(repository_path(attempt_path)),
            "sha256": sha256_file(attempt_path),
            "attempt_hash": attempt["attempt_hash"],
        },
        "preregistration": {
            "path": PREREGISTRATION.as_posix(),
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        },
        "decision": decision,
        "terminal_action": (
            REPEATABILITY_PASS_ACTION
            if decision == "pass"
            else REPEATABILITY_FAILURE_ACTION
        ),
        "repeatability_case_roster_hash": gate[
            "repeatability_case_roster_hash"
        ],
        "capacity_case_hash": gate["capacity_case_hash"],
        "repeat_count": gate["repeat_count"],
        "repeat_model_load_count": gate["repeat_model_load_count"],
        "capacity_repeat_count": gate["capacity_repeat_count"],
        "load_results": [dict(result) for result in load_results],
        "cross_load_comparison": (
            None
            if cross_load_comparison is None
            else dict(cross_load_comparison)
        ),
        "gate_outputs_reused_by_full_extraction": False,
        "full_extraction_uses_fresh_third_model_load": True,
        "full_source_extraction_authorized": decision == "pass",
        "market_access_authorized": False,
        "s1_or_s2_checkpoint_or_model_outputs_read": False,
        "access_boundary": {
            "market_or_funding_paths_read": [],
            "market_rows_parsed": 0,
            "funding_rows_parsed": 0,
            "market_or_funding_payload_bytes_hashed": False,
            "rewards_created": 0,
            "economic_metrics_computed": 0,
            "train_2020_outcomes_opened": False,
            "test_outcomes_opened": False,
            "eval_outcomes_opened": False,
        },
    }
    if failure is not None:
        core["failure"] = dict(failure)
    return {**core, "result_hash": canonical_hash(core)}


def _write_gate_result(payload: Mapping[str, Any]) -> None:
    _write_new_json(DEFAULT_REPEATABILITY_RESULT, payload)


def run_pre_market_gates(
    scorer_factory: Callable[[Any], Any],
    processor: Any,
    *,
    execution_commit: str,
    attempt_path: Path,
    attempt: Mapping[str, Any],
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    load_results: list[dict[str, Any]] = []
    active_scorer: Any | None = None
    try:
        for load_index in range(prereg.REPEAT_MODEL_LOAD_COUNT):
            active_scorer = scorer_factory(processor)
            case_results: list[dict[str, Any]] = []
            capacity_result: dict[str, Any] | None = None
            closed = False
            try:
                for case in prepared["repeatability_cases"]:
                    row = prepared["rows"][int(case["row_index"])]
                    case_results.append(
                        _repeatability_case_result(
                            active_scorer,
                            case=case,
                            row=row,
                        )
                    )
                capacity_case = prepared["capacity_case"]
                capacity_row = prepared["rows"][
                    int(capacity_case["row_index"])
                ]
                capacity_result = _capacity_result(
                    active_scorer,
                    case=capacity_case,
                    row=capacity_row,
                )
            finally:
                _close_gate_scorer(active_scorer)
                active_scorer = None
                closed = True
            load_pass = bool(
                len(case_results) == len(prepared["repeatability_cases"])
                and all(result["pass"] for result in case_results)
                and capacity_result is not None
                and capacity_result["pass"]
                and closed
            )
            load_results.append(
                {
                    "load_index": load_index,
                    "model_loaded_independently": True,
                    "scorer_closed_before_next_load": closed,
                    "case_results": case_results,
                    "capacity_result": capacity_result,
                    "pass": load_pass,
                }
            )
            if not load_pass:
                break
        comparison = _cross_load_comparison(load_results)
        decision = "pass" if comparison["pass"] else "reject"
        payload = _gate_result_payload(
            execution_commit=execution_commit,
            attempt_path=attempt_path,
            attempt=attempt,
            prepared=prepared,
            decision=decision,
            load_results=load_results,
            cross_load_comparison=comparison,
            failure=(
                None
                if decision == "pass"
                else {
                    "stage": "PRE_MARKET_REPEATABILITY_AND_CAPACITY",
                    "reason": "frozen repeatability or capacity contract failed",
                }
            ),
        )
        _write_gate_result(payload)
        if decision != "pass":
            raise RuntimeError(
                "RLLM2-S3 pre-market repeatability or capacity failed"
            )
        return payload
    except BaseException as error:
        if active_scorer is not None:
            try:
                _close_gate_scorer(active_scorer)
            finally:
                active_scorer = None
        gate_path = repository_path(DEFAULT_REPEATABILITY_RESULT)
        if not gate_path.exists() and not gate_path.is_symlink():
            payload = _gate_result_payload(
                execution_commit=execution_commit,
                attempt_path=attempt_path,
                attempt=attempt,
                prepared=prepared,
                decision="reject",
                load_results=load_results,
                cross_load_comparison=(
                    _cross_load_comparison(load_results)
                    if load_results
                    else None
                ),
                failure={
                    "stage": "PRE_MARKET_REPEATABILITY_AND_CAPACITY",
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                },
            )
            _write_gate_result(payload)
        raise


def _validate_scan_evidence(
    evidence: Any,
    *,
    expected_tokens: int,
) -> None:
    expected_keys = {
        "input_tokens",
        "chunk_size_tokens",
        "chunk_count",
        "token_reconstruction_exact",
    }
    if (
        not isinstance(evidence, dict)
        or set(evidence) != expected_keys
        or evidence.get("input_tokens") != expected_tokens
        or evidence.get("chunk_size_tokens") != prereg.CHUNK_SIZE
        or evidence.get("chunk_count")
        != (expected_tokens + prereg.CHUNK_SIZE - 1)
        // prereg.CHUNK_SIZE
        or evidence.get("token_reconstruction_exact") is not True
    ):
        raise RuntimeError("RLLM2-S3 chunk scan evidence changed")


def _valid_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_repeatability_case_result(
    expected: Mapping[str, Any],
    observed: Any,
) -> None:
    expected_keys = {
        "row_index",
        "row_hash",
        "selection_reasons",
        "policy_tokens",
        "relation_tokens",
        "policy_scan",
        "relation_scan",
        "embedding_shape",
        "embedding_all_values_finite",
        "embedding_float32_bytes_sha256",
        "relation_logits_finite_or_canonical_nan",
        "relation_logits_canonical_float32_bytes_sha256",
        "predicted_relation_code",
        "pass",
    }
    if (
        not isinstance(observed, dict)
        or set(observed) != expected_keys
        or observed.get("row_index") != expected["row_index"]
        or observed.get("row_hash") != expected["row_hash"]
        or observed.get("selection_reasons")
        != expected["selection_reasons"]
        or observed.get("policy_tokens") != expected["policy_tokens"]
        or observed.get("relation_tokens") != expected["relation_tokens"]
        or observed.get("embedding_shape") != [s1.EMBEDDING_WIDTH]
        or observed.get("embedding_all_values_finite") is not True
        or not _valid_sha256(
            observed.get("embedding_float32_bytes_sha256")
        )
        or observed.get("relation_logits_finite_or_canonical_nan")
        is not True
        or not _valid_sha256(
            observed.get("relation_logits_canonical_float32_bytes_sha256")
        )
        or observed.get("predicted_relation_code")
        not in {None, *s1.RELATION_CODE_ORDER}
        or observed.get("pass") is not True
    ):
        raise RuntimeError("RLLM2-S3 repeatability case evidence changed")
    _validate_scan_evidence(
        observed["policy_scan"],
        expected_tokens=int(expected["policy_tokens"]),
    )
    _validate_scan_evidence(
        observed["relation_scan"],
        expected_tokens=int(expected["relation_tokens"]),
    )


def _validate_capacity_result(
    expected: Mapping[str, Any],
    observed: Any,
) -> None:
    expected_keys = {
        "row_index",
        "row_hash",
        "policy_tokens",
        "relation_tokens",
        "policy_scan",
        "relation_scan",
        "embedding_shape",
        "embedding_all_values_finite",
        "embedding_float32_bytes_sha256",
        "relation_logits_finite_or_canonical_nan",
        "relation_logits_canonical_float32_bytes_sha256",
        "predicted_relation_code",
        "peak_allocated_bytes",
        "maximum_peak_allocated_bytes",
        "outputs_reused_by_full_extraction",
        "pass",
    }
    if (
        not isinstance(observed, dict)
        or set(observed) != expected_keys
        or observed.get("row_index") != expected["row_index"]
        or observed.get("row_hash") != expected["row_hash"]
        or observed.get("policy_tokens") != expected["policy_tokens"]
        or observed.get("relation_tokens") != expected["relation_tokens"]
        or observed.get("embedding_shape") != [s1.EMBEDDING_WIDTH]
        or observed.get("embedding_all_values_finite") is not True
        or not _valid_sha256(
            observed.get("embedding_float32_bytes_sha256")
        )
        or observed.get("relation_logits_finite_or_canonical_nan")
        is not True
        or not _valid_sha256(
            observed.get("relation_logits_canonical_float32_bytes_sha256")
        )
        or observed.get("predicted_relation_code")
        not in {None, *s1.RELATION_CODE_ORDER}
        or not isinstance(observed.get("peak_allocated_bytes"), int)
        or isinstance(observed.get("peak_allocated_bytes"), bool)
        or observed["peak_allocated_bytes"] < 0
        or observed["peak_allocated_bytes"]
        > prereg.MAXIMUM_PEAK_ALLOCATED_BYTES
        or observed.get("maximum_peak_allocated_bytes")
        != prereg.MAXIMUM_PEAK_ALLOCATED_BYTES
        or observed.get("outputs_reused_by_full_extraction") is not False
        or observed.get("pass") is not True
    ):
        raise RuntimeError("RLLM2-S3 capacity evidence changed")
    _validate_scan_evidence(
        observed["policy_scan"],
        expected_tokens=int(expected["policy_tokens"]),
    )
    _validate_scan_evidence(
        observed["relation_scan"],
        expected_tokens=int(expected["relation_tokens"]),
    )


def validate_repeatability_result(
    *,
    attempt_path: Path,
    attempt: Mapping[str, Any],
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _read_exact_json(DEFAULT_REPEATABILITY_RESULT)
    core = {
        key: value for key, value in payload.items() if key != "result_hash"
    }
    gate = prepared["preregistration"][
        "pre_market_repeatability_and_capacity_gate"
    ]
    expected_access_boundary = {
        "market_or_funding_paths_read": [],
        "market_rows_parsed": 0,
        "funding_rows_parsed": 0,
        "market_or_funding_payload_bytes_hashed": False,
        "rewards_created": 0,
        "economic_metrics_computed": 0,
        "train_2020_outcomes_opened": False,
        "test_outcomes_opened": False,
        "eval_outcomes_opened": False,
    }
    expected_top_level_keys = {
        "protocol_version",
        "stage_id",
        "execution_commit",
        "runner_sha256",
        "attempt",
        "preregistration",
        "decision",
        "terminal_action",
        "repeatability_case_roster_hash",
        "capacity_case_hash",
        "repeat_count",
        "repeat_model_load_count",
        "capacity_repeat_count",
        "load_results",
        "cross_load_comparison",
        "gate_outputs_reused_by_full_extraction",
        "full_extraction_uses_fresh_third_model_load",
        "full_source_extraction_authorized",
        "market_access_authorized",
        "s1_or_s2_checkpoint_or_model_outputs_read",
        "access_boundary",
        "result_hash",
    }
    if (
        set(payload) != expected_top_level_keys
        or payload.get("result_hash") != canonical_hash(core)
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("stage_id") != prereg.STAGE_ID
        or payload.get("execution_commit") != attempt["execution_commit"]
        or payload.get("runner_sha256") != attempt["runner_sha256"]
        or payload.get("runner_sha256") != sha256_file(Path(__file__))
        or payload.get("attempt")
        != {
            "path": _display_path(repository_path(attempt_path)),
            "sha256": sha256_file(attempt_path),
            "attempt_hash": attempt["attempt_hash"],
        }
        or payload.get("preregistration")
        != {
            "path": PREREGISTRATION.as_posix(),
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        }
        or payload.get("decision") != "pass"
        or payload.get("terminal_action") != REPEATABILITY_PASS_ACTION
        or payload.get("repeatability_case_roster_hash")
        != gate["repeatability_case_roster_hash"]
        or payload.get("capacity_case_hash") != gate["capacity_case_hash"]
        or payload.get("repeat_count") != prereg.REPEAT_COUNT
        or payload.get("repeat_model_load_count")
        != prereg.REPEAT_MODEL_LOAD_COUNT
        or payload.get("capacity_repeat_count") != prereg.REPEAT_COUNT
        or payload.get("gate_outputs_reused_by_full_extraction") is not False
        or payload.get("full_extraction_uses_fresh_third_model_load")
        is not True
        or payload.get("full_source_extraction_authorized") is not True
        or payload.get("market_access_authorized") is not False
        or payload.get("s1_or_s2_checkpoint_or_model_outputs_read")
        is not False
        or payload.get("access_boundary") != expected_access_boundary
    ):
        raise RuntimeError("RLLM2-S3 repeatability pass binding changed")
    load_results = payload.get("load_results")
    if (
        not isinstance(load_results, list)
        or len(load_results) != prereg.REPEAT_MODEL_LOAD_COUNT
    ):
        raise RuntimeError("RLLM2-S3 repeatability load evidence changed")
    expected_load_keys = {
        "load_index",
        "model_loaded_independently",
        "scorer_closed_before_next_load",
        "case_results",
        "capacity_result",
        "pass",
    }
    for load_index, load in enumerate(load_results):
        if (
            not isinstance(load, dict)
            or set(load) != expected_load_keys
            or load.get("load_index") != load_index
            or load.get("model_loaded_independently") is not True
            or load.get("scorer_closed_before_next_load") is not True
            or load.get("pass") is not True
        ):
            raise RuntimeError("RLLM2-S3 repeatability load evidence changed")
        cases = load.get("case_results")
        if (
            not isinstance(cases, list)
            or len(cases) != len(prepared["repeatability_cases"])
        ):
            raise RuntimeError("RLLM2-S3 repeatability case evidence changed")
        for expected, observed in zip(
            prepared["repeatability_cases"], cases, strict=True
        ):
            _validate_repeatability_case_result(expected, observed)
        _validate_capacity_result(
            prepared["capacity_case"], load.get("capacity_result")
        )
    recomputed = _cross_load_comparison(load_results)
    if payload.get("cross_load_comparison") != recomputed or not recomputed[
        "pass"
    ]:
        raise RuntimeError("RLLM2-S3 cross-load repeatability changed")
    return payload


def _process_shard(
    scorer: Any,
    *,
    rows: Sequence[Mapping[str, Any]],
    policy_counts: Sequence[int],
    relation_counts: Sequence[int],
    start: int,
    stop: int,
) -> dict[str, np.ndarray]:
    embeddings: list[np.ndarray] = []
    relation_logits: list[np.ndarray] = []
    forwarded: list[int] = []
    observed_policy_tokens: list[int] = []
    observed_relation_tokens: list[int] = []
    for row in rows[start:stop]:
        row_index = int(row["row_index"])
        vector, policy_tokens = scorer.embed(str(row["policy_prompt"]))
        if policy_tokens != int(policy_counts[row_index]):
            raise RuntimeError("RLLM2-S3 policy token count changed")
        embeddings.append(np.asarray(vector, dtype=np.float32))
        observed_policy_tokens.append(policy_tokens)
        if bool(row["relation_teacher_forward_required"]):
            logits, _, relation_tokens = scorer.score_relation(
                str(row["relation_teacher_prompt"])
            )
            if relation_tokens != int(relation_counts[row_index]):
                raise RuntimeError(
                    "RLLM2-S3 relation token count changed"
                )
            relation_logits.append(
                _normalize_relation_logits(logits)
            )
            forwarded.append(1)
            observed_relation_tokens.append(relation_tokens)
        else:
            relation_logits.append(
                _canonical_nan_row(len(s1.RELATION_CODE_ORDER))
            )
            forwarded.append(0)
            observed_relation_tokens.append(
                int(relation_counts[row_index])
            )
    return {
        "row_index": np.arange(start, stop, dtype=np.int32),
        "embedding": np.stack(embeddings).astype(
            np.float32,
            copy=False,
        ),
        "relation_logits": np.stack(relation_logits).astype(
            np.float32,
            copy=False,
        ),
        "relation_forwarded": np.asarray(forwarded, dtype=np.uint8),
        "policy_input_tokens": np.asarray(
            observed_policy_tokens,
            dtype=np.int32,
        ),
        "relation_input_tokens": np.asarray(
            observed_relation_tokens,
            dtype=np.int32,
        ),
    }


def _merge_shards(
    shards: Sequence[Mapping[str, np.ndarray]],
    *,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    if not shards:
        raise RuntimeError("RLLM2-S3 has no checkpoint shards")
    names = set(shards[0])
    merged = {
        name: np.concatenate(
            [np.asarray(shard[name]) for shard in shards],
            axis=0,
        )
        for name in names
    }
    _validate_shard_arrays(
        merged,
        rows=rows,
        start=0,
        stop=len(rows),
    )
    return merged


def _relation_rows(
    rows: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        forwarded = bool(arrays["relation_forwarded"][index])
        logits = arrays["relation_logits"][index]
        code = _predicted_code(logits) if forwarded else None
        if not forwarded:
            relation = "INSUFFICIENT_EVIDENCE"
            finite = False
        elif code is None:
            relation = "ABSTAIN"
            finite = False
        else:
            relation = row["relation_teacher_code_to_label"][code]
            finite = True
        core = {
            "schema_version": s1.RELATION_ROW_SCHEMA_VERSION,
            "row_index": index,
            "source_row_hash": row["row_hash"],
            "relation_teacher_forwarded": forwarded,
            "code_to_label": row["relation_teacher_code_to_label"],
            "predicted_code": code,
            "predicted_relation": relation,
            "finite_code_logits": finite,
            "policy_input_tokens": int(
                arrays["policy_input_tokens"][index]
            ),
            "relation_input_tokens": int(
                arrays["relation_input_tokens"][index]
            ),
        }
        output.append({**core, "row_hash": canonical_hash(core)})
    return output


def _artifact_record_from_bytes(
    path: Path,
    raw: bytes,
    *,
    decompressed_sha256: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    target = repository_path(path)
    result = {
        "path": _display_path(target),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }
    if decompressed_sha256 is not None:
        result["decompressed_sha256"] = decompressed_sha256
    if extra:
        result.update(dict(extra))
    return result


def _build_final_artifacts(
    *,
    rows: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    source_raw, source_decompressed_sha = (
        _deterministic_jsonl_gzip_bytes(rows)
    )
    relation_rows = _relation_rows(rows, arrays)
    relation_raw, relation_decompressed_sha = (
        _deterministic_jsonl_gzip_bytes(relation_rows)
    )
    embeddings_raw = _deterministic_npz_bytes(
        {
            "embedding": arrays["embedding"],
            "row_index": arrays["row_index"],
        }
    )
    relation_logits_raw = _deterministic_npz_bytes(
        {
            "forwarded": arrays["relation_forwarded"],
            "logits": arrays["relation_logits"],
            "policy_input_tokens": arrays["policy_input_tokens"],
            "relation_input_tokens": arrays["relation_input_tokens"],
            "row_index": arrays["row_index"],
        }
    )
    records = {
        "source_rows": _artifact_record_from_bytes(
            DEFAULT_SOURCE_ROWS,
            source_raw,
            decompressed_sha256=source_decompressed_sha,
            extra={
                "row_count": len(rows),
                "source_row_roster_hash": SOURCE_ROW_ROSTER_HASH,
            },
        ),
        "embeddings": _artifact_record_from_bytes(
            DEFAULT_EMBEDDINGS,
            embeddings_raw,
            extra={
                "shape": list(arrays["embedding"].shape),
                "dtype": str(arrays["embedding"].dtype),
                "all_values_finite": bool(
                    np.all(np.isfinite(arrays["embedding"]))
                ),
            },
        ),
        "relation_logits": _artifact_record_from_bytes(
            DEFAULT_RELATION_LOGITS,
            relation_logits_raw,
            extra={
                "shape": list(arrays["relation_logits"].shape),
                "dtype": str(arrays["relation_logits"].dtype),
                "forwarded_rows": int(
                    np.sum(arrays["relation_forwarded"])
                ),
                "nonfinite_forwarded_rows": sum(
                    bool(arrays["relation_forwarded"][index])
                    and not np.all(
                        np.isfinite(arrays["relation_logits"][index])
                    )
                    for index in range(len(rows))
                ),
            },
        ),
        "relation_rows": _artifact_record_from_bytes(
            DEFAULT_RELATION_ROWS,
            relation_raw,
            decompressed_sha256=relation_decompressed_sha,
            extra={
                "row_count": len(relation_rows),
                "row_roster_hash": canonical_hash(
                    [row["row_hash"] for row in relation_rows]
                ),
            },
        ),
    }
    raw_by_name = {
        "source_rows": source_raw,
        "embeddings": embeddings_raw,
        "relation_logits": relation_logits_raw,
        "relation_rows": relation_raw,
    }
    return records, raw_by_name


def _publish_journal_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".publish.json")


def _publish_result_staging_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".publish.tmp")


def _publish_internal_paths(output_path: Path) -> tuple[Path, ...]:
    journal = _publish_journal_path(output_path)
    result_staging = _publish_result_staging_path(output_path)
    return (
        journal,
        journal.with_name(journal.name + ".tmp"),
        result_staging,
        result_staging.with_name(result_staging.name + ".tmp"),
    )


def _publish_progress(_: str) -> None:
    """Fault-injection seam for publish transaction tests."""


def _promote_staged_file(source: Path, target: Path) -> None:
    source.replace(target)


def _publish_journal_payload(
    *,
    checkpoint_directory: Path,
    output_path: Path,
    payload: Mapping[str, Any],
    artifact_records: Mapping[str, Mapping[str, Any]],
    result_staging: Path,
) -> dict[str, Any]:
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "stage_id": prereg.STAGE_ID,
        "transaction_state": "PREPARED_FOR_RECOVERABLE_PUBLISH",
        "checkpoint_directory": _display_path(checkpoint_directory),
        "output_path": _display_path(output_path),
        "output_result_hash": payload["result_hash"],
        "result_staging_path": _display_path(result_staging),
        "result_staging_sha256": sha256_file(result_staging),
        "artifact_records": {
            name: dict(artifact_records[name])
            for name in (
                "source_rows",
                "embeddings",
                "relation_logits",
                "relation_rows",
            )
        },
        "authorizing_result_published_last": True,
    }
    return {**core, "transaction_hash": canonical_hash(core)}


def _rollback_publish_transaction(
    *,
    output_path: Path,
    artifact_records: Mapping[str, Mapping[str, Any]],
) -> None:
    if output_path.exists() or output_path.is_symlink():
        raise RuntimeError(
            "RLLM2-S3 cannot roll back a published terminal result"
        )
    for record in artifact_records.values():
        target = repository_path(Path(str(record["path"])))
        if not target.exists() and not target.is_symlink():
            continue
        if (
            target.is_symlink()
            or not target.is_file()
            or sha256_file(target) != record["sha256"]
        ):
            raise RuntimeError(
                "RLLM2-S3 publish rollback found an unsafe artifact"
            )
        target.unlink()
    for path in _publish_internal_paths(output_path):
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                "RLLM2-S3 publish rollback found unsafe metadata"
            )
        path.unlink()


def _validate_publish_journal(
    *,
    checkpoint_directory: Path,
    output_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Mapping[str, Any]],
]:
    journal_path = _publish_journal_path(output_path)
    journal = _read_exact_json(journal_path)
    journal_core = {
        key: value
        for key, value in journal.items()
        if key != "transaction_hash"
    }
    expected_keys = {
        "protocol_version",
        "stage_id",
        "transaction_state",
        "checkpoint_directory",
        "output_path",
        "output_result_hash",
        "result_staging_path",
        "result_staging_sha256",
        "artifact_records",
        "authorizing_result_published_last",
        "transaction_hash",
    }
    result_staging = _publish_result_staging_path(output_path)
    if (
        set(journal) != expected_keys
        or journal.get("transaction_hash") != canonical_hash(journal_core)
        or journal.get("protocol_version") != PROTOCOL_VERSION
        or journal.get("stage_id") != prereg.STAGE_ID
        or journal.get("transaction_state")
        != "PREPARED_FOR_RECOVERABLE_PUBLISH"
        or journal.get("checkpoint_directory")
        != _display_path(checkpoint_directory)
        or journal.get("output_path") != _display_path(output_path)
        or journal.get("result_staging_path")
        != _display_path(result_staging)
        or journal.get("authorizing_result_published_last") is not True
        or result_staging.is_symlink()
        or not result_staging.is_file()
        or journal.get("result_staging_sha256")
        != sha256_file(result_staging)
    ):
        raise RuntimeError("RLLM2-S3 publish journal binding changed")
    payload = _read_exact_json(result_staging)
    payload_core = {
        key: value for key, value in payload.items() if key != "result_hash"
    }
    if (
        payload.get("result_hash") != canonical_hash(payload_core)
        or payload.get("result_hash") != journal["output_result_hash"]
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("stage_id") != prereg.STAGE_ID
        or payload.get("decision") != "pass"
        or payload.get("terminal_action") != PASS_ACTION
    ):
        raise RuntimeError("RLLM2-S3 staged pass result changed")
    artifacts = journal.get("artifact_records")
    if (
        not isinstance(artifacts, dict)
        or set(artifacts)
        != {
            "source_rows",
            "embeddings",
            "relation_logits",
            "relation_rows",
        }
        or payload.get("artifacts") != artifacts
    ):
        raise RuntimeError("RLLM2-S3 publish artifact binding changed")
    return journal, payload, artifacts


def _complete_publish_transaction(
    *,
    checkpoint_directory: Path,
    output_path: Path,
    artifact_records: Mapping[str, Mapping[str, Any]],
) -> None:
    staging = checkpoint_directory / "final_staging"
    for name in (
        "source_rows",
        "embeddings",
        "relation_logits",
        "relation_rows",
    ):
        record = artifact_records[name]
        target = repository_path(Path(str(record["path"])))
        staged = staging / Path(str(record["path"])).name
        if target.exists() or target.is_symlink():
            if (
                target.is_symlink()
                or not target.is_file()
                or sha256_file(target) != record["sha256"]
            ):
                raise RuntimeError(
                    "RLLM2-S3 promoted artifact changed during recovery"
                )
            continue
        if (
            staged.is_symlink()
            or not staged.is_file()
            or sha256_file(staged) != record["sha256"]
        ):
            raise RuntimeError(
                "RLLM2-S3 staged artifact unavailable during recovery"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        _promote_staged_file(staged, target)
        if sha256_file(target) != record["sha256"]:
            raise RuntimeError("RLLM2-S3 promoted artifact hash changed")
        _publish_progress(f"artifact_promoted:{name}")
    if checkpoint_directory.exists() or checkpoint_directory.is_symlink():
        if checkpoint_directory.is_symlink() or not checkpoint_directory.is_dir():
            raise RuntimeError("RLLM2-S3 checkpoint cleanup target is unsafe")
        shutil.rmtree(checkpoint_directory)
    if checkpoint_directory.exists() or checkpoint_directory.is_symlink():
        raise RuntimeError("RLLM2-S3 checkpoint cleanup failed")
    _publish_progress("checkpoint_removed_before_result")
    result_staging = _publish_result_staging_path(output_path)
    if output_path.exists() or output_path.is_symlink():
        raise RuntimeError("RLLM2-S3 terminal result already exists")
    _promote_staged_file(result_staging, output_path)
    _publish_progress("authorizing_result_published")
    journal_path = _publish_journal_path(output_path)
    if journal_path.is_symlink() or not journal_path.is_file():
        raise RuntimeError("RLLM2-S3 publish journal disappeared")
    journal_path.unlink()


def _recover_publish_transaction(
    *,
    checkpoint_directory: Path,
    output_path: Path,
) -> dict[str, Any] | None:
    journal_path = _publish_journal_path(output_path)
    if not journal_path.exists() and not journal_path.is_symlink():
        return None
    _, payload, artifact_records = _validate_publish_journal(
        checkpoint_directory=checkpoint_directory,
        output_path=output_path,
    )
    try:
        _complete_publish_transaction(
            checkpoint_directory=checkpoint_directory,
            output_path=output_path,
            artifact_records=artifact_records,
        )
    except BaseException:
        if not output_path.exists() and not output_path.is_symlink():
            _rollback_publish_transaction(
                output_path=output_path,
                artifact_records=artifact_records,
            )
        raise
    return payload


def _publish_success_atomically(
    *,
    checkpoint_directory: Path,
    output_path: Path,
    payload: Mapping[str, Any],
    artifact_records: Mapping[str, Mapping[str, Any]],
    artifact_bytes: Mapping[str, bytes],
) -> None:
    staging = repository_path(checkpoint_directory) / "final_staging"
    if staging.exists() or staging.is_symlink():
        raise RuntimeError("RLLM2-S3 final staging already exists")
    staging.mkdir()
    for name in (
        "source_rows",
        "embeddings",
        "relation_logits",
        "relation_rows",
    ):
        record = artifact_records[name]
        raw = artifact_bytes[name]
        if (
            hashlib.sha256(raw).hexdigest() != record["sha256"]
            or len(raw) != record["bytes"]
        ):
            raise RuntimeError("RLLM2-S3 artifact record changed")
        staged = staging / Path(str(record["path"])).name
        _write_new_bytes(staged, raw)
        if sha256_file(staged) != record["sha256"]:
            raise RuntimeError("RLLM2-S3 staged artifact hash changed")
    result_raw = canonical_json_bytes(dict(payload), pretty=True)
    result_staging = _publish_result_staging_path(output_path)
    _write_new_bytes(result_staging, result_raw)
    observed = _read_exact_json(result_staging)
    if (
        observed != dict(payload)
        or observed.get("result_hash")
        != canonical_hash(
            {
                key: value
                for key, value in observed.items()
                if key != "result_hash"
            }
        )
    ):
        raise RuntimeError("RLLM2-S3 staged result verification failed")
    journal = _publish_journal_payload(
        checkpoint_directory=checkpoint_directory,
        output_path=output_path,
        payload=payload,
        artifact_records=artifact_records,
        result_staging=result_staging,
    )
    _write_new_json(_publish_journal_path(output_path), journal)
    commit_ready = False
    try:
        _complete_publish_transaction(
            checkpoint_directory=checkpoint_directory,
            output_path=output_path,
            artifact_records=artifact_records,
        )
        commit_ready = True
    except BaseException:
        commit_ready = bool(
            not checkpoint_directory.exists()
            and not checkpoint_directory.is_symlink()
            and all(
                repository_path(Path(str(record["path"]))).is_file()
                and not repository_path(
                    Path(str(record["path"]))
                ).is_symlink()
                and sha256_file(Path(str(record["path"])))
                == record["sha256"]
                for record in artifact_records.values()
            )
        )
        if commit_ready:
            if not output_path.exists() and not output_path.is_symlink():
                _promote_staged_file(result_staging, output_path)
            journal_path = _publish_journal_path(output_path)
            if journal_path.is_file() and not journal_path.is_symlink():
                journal_path.unlink()
            return
        _rollback_publish_transaction(
            output_path=output_path,
            artifact_records=artifact_records,
        )
        raise
    if not commit_ready:
        raise RuntimeError("RLLM2-S3 publish transaction did not commit")


def _success_payload(
    *,
    execution_commit: str,
    attempt_path: Path,
    attempt: Mapping[str, Any],
    prepared: Mapping[str, Any],
    scorer: Any,
    repeatability_result: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    checkpoint_chain_hash: str,
    checkpoint_shards: int,
    resumed: bool,
    resumed_from_rows: int,
) -> dict[str, Any]:
    prompt_capacity = {
        key: value
        for key, value in prepared["prompt_capacity"].items()
        if key not in {"policy", "relation_teacher"}
    }
    prompt_capacity["policy"] = {
        key: value
        for key, value in prepared["prompt_capacity"]["policy"].items()
        if key != "counts"
    }
    prompt_capacity["relation_teacher"] = {
        key: value
        for key, value in prepared["prompt_capacity"][
            "relation_teacher"
        ].items()
        if key != "counts"
    }
    model_metrics = dict(scorer.metrics())
    model_metrics.update(
        {
            "total_embedding_forwards_across_attempt": prepared[
                "roster"
            ]["embedding_forward_count"],
            "total_relation_forwards_across_attempt": prepared[
                "roster"
            ]["relation_teacher_forward_count"],
            "resumed_from_completed_rows": resumed_from_rows,
        }
    )
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "stage_id": prereg.STAGE_ID,
        "execution_commit": execution_commit,
        "attempt": {
            "path": _display_path(repository_path(attempt_path)),
            "sha256": sha256_file(attempt_path),
            "attempt_hash": attempt["attempt_hash"],
        },
        "preregistration": {
            "path": PREREGISTRATION.as_posix(),
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        },
        "predecessor_terminal_result_hash": (
            prereg.S2_FAILURE_RESULT_HASH
        ),
        "repeatability_and_capacity_gate": {
            "path": _display_path(
                repository_path(DEFAULT_REPEATABILITY_RESULT)
            ),
            "sha256": sha256_file(DEFAULT_REPEATABILITY_RESULT),
            "result_hash": repeatability_result["result_hash"],
            "decision": repeatability_result["decision"],
            "model_load_count": prereg.REPEAT_MODEL_LOAD_COUNT,
            "outputs_reused_by_full_extraction": False,
        },
        "decision": "pass",
        "terminal_action": PASS_ACTION,
        "source_feature_seal_authorized": True,
        "open_2020_train_outcomes_authorized": True,
        "open_2021_or_later_outcomes_authorized": False,
        "market_access_authorized_during_this_stage": False,
        "source_roster": prepared["roster"],
        "prompt_capacity": prompt_capacity,
        "runtime": prepared["runtime"],
        "model_metrics": model_metrics,
        "model_load_evidence": {
            "gate_model_loads": prereg.REPEAT_MODEL_LOAD_COUNT,
            "full_extraction_model_load_ordinal": 3,
            "fresh_from_gate_models": True,
        },
        "checkpoint_evidence": {
            "shard_size": prereg.CHECKPOINT_SHARD_SIZE,
            "shard_count": checkpoint_shards,
            "terminal_chain_hash": checkpoint_chain_hash,
            "resumed": resumed,
            "resumed_from_completed_rows": resumed_from_rows,
        },
        "artifacts": dict(artifacts),
        "access_boundary": {
            "source_paths_read": sorted(
                {
                    PREREGISTRATION.as_posix(),
                    prereg.S2_PREREGISTRATION.as_posix(),
                    prereg.S2_ATTEMPT.as_posix(),
                    prereg.S2_GATE.as_posix(),
                    prereg.S2_FAILURE.as_posix(),
                    prereg.S2_FAILURE_LOG.as_posix(),
                    prereg.S2_TERMINAL_REJECTION_DOC.as_posix(),
                    s1.rllm1.D8_CARDS.as_posix(),
                }
            ),
            "model_snapshot_read": str(base._model_snapshot()),
            "s1_or_s2_checkpoint_or_model_outputs_read": False,
            "market_or_funding_paths_read": [],
            "market_rows_parsed": 0,
            "funding_rows_parsed": 0,
            "market_or_funding_payload_bytes_hashed": False,
            "model_forwards_started": int(
                getattr(scorer, "forward_calls_started", 0)
            ),
            "rewards_created": 0,
            "economic_metrics_computed": 0,
            "train_2020_outcomes_opened": False,
            "test_outcomes_opened": False,
            "eval_outcomes_opened": False,
        },
    }
    return {**core, "result_hash": canonical_hash(core)}


def _failure_payload(
    *,
    execution_commit: str,
    attempt_path: Path,
    attempt: Mapping[str, Any],
    stage: str,
    error: BaseException,
    scorer: Any | None,
    completed_rows: int,
) -> dict[str, Any]:
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "stage_id": prereg.STAGE_ID,
        "execution_commit": execution_commit,
        "attempt": {
            "path": _display_path(repository_path(attempt_path)),
            "sha256": sha256_file(attempt_path),
            "attempt_hash": attempt["attempt_hash"],
        },
        "decision": "reject",
        "terminal_action": FAILURE_ACTION,
        "failure": {
            "stage": stage,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "observations": {
            "completed_source_rows": completed_rows,
            "model_forwards_started": int(
                getattr(scorer, "forward_calls_started", 0)
            ),
            "embedding_forwards_started": int(
                getattr(scorer, "embedding_forwards_started", 0)
            ),
            "relation_forwards_started": int(
                getattr(scorer, "relation_forwards_started", 0)
            ),
            "chunk_forwards_started": int(
                getattr(scorer, "chunk_forwards_started", 0)
            ),
        },
        "source_feature_seal_authorized": False,
        "open_2020_train_outcomes_authorized": False,
        "market_access_authorized": False,
        "resume_authorized": False,
        "rerun_authorized": False,
        "access_boundary": {
            "market_or_funding_paths_read": [],
            "market_rows_parsed": 0,
            "funding_rows_parsed": 0,
            "market_or_funding_payload_bytes_hashed": False,
            "s1_or_s2_checkpoint_or_model_outputs_read": False,
            "rewards_created": 0,
            "economic_metrics_computed": 0,
            "train_2020_outcomes_opened": False,
            "test_outcomes_opened": False,
            "eval_outcomes_opened": False,
        },
    }
    return {**core, "result_hash": canonical_hash(core)}


def _all_final_artifact_paths() -> tuple[Path, ...]:
    return (
        DEFAULT_SOURCE_ROWS,
        DEFAULT_EMBEDDINGS,
        DEFAULT_RELATION_LOGITS,
        DEFAULT_RELATION_ROWS,
    )


def _guard_initial_paths() -> None:
    output_path = repository_path(DEFAULT_OUTPUT)
    paths = (
        DEFAULT_ATTEMPT,
        DEFAULT_REPEATABILITY_RESULT,
        DEFAULT_OUTPUT,
        DEFAULT_CHECKPOINT_DIRECTORY,
        *_all_final_artifact_paths(),
        *_publish_internal_paths(output_path),
    )
    if any(
        repository_path(path).exists()
        or repository_path(path).is_symlink()
        for path in paths
    ):
        raise RuntimeError("RLLM2-S3 source feature seal already attempted")


def _guard_resume_paths() -> None:
    attempt = repository_path(DEFAULT_ATTEMPT)
    repeatability = repository_path(DEFAULT_REPEATABILITY_RESULT)
    result = repository_path(DEFAULT_OUTPUT)
    checkpoint = repository_path(DEFAULT_CHECKPOINT_DIRECTORY)
    journal = _publish_journal_path(result)
    result_staging = _publish_result_staging_path(result)
    for path in (
        journal.with_name(journal.name + ".tmp"),
        result_staging.with_name(result_staging.name + ".tmp"),
    ):
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("unsafe RLLM2-S3 publish temporary")
        path.unlink()
    if not journal.exists() and not journal.is_symlink():
        if result_staging.exists() or result_staging.is_symlink():
            if result_staging.is_symlink() or not result_staging.is_file():
                raise RuntimeError("unsafe RLLM2-S3 staged result")
            result_staging.unlink()
        if any(
            repository_path(path).exists()
            or repository_path(path).is_symlink()
            for path in _all_final_artifact_paths()
        ):
            raise RuntimeError(
                "RLLM2-S3 final artifacts exist without publish journal"
            )
    elif (
        journal.is_symlink()
        or not journal.is_file()
        or result_staging.is_symlink()
        or not result_staging.is_file()
    ):
        raise RuntimeError("RLLM2-S3 publish recovery state is unsafe")
    if (
        attempt.is_symlink()
        or not attempt.is_file()
        or repeatability.is_symlink()
        or not repeatability.is_file()
        or result.exists()
        or result.is_symlink()
        or checkpoint.is_symlink()
        or (checkpoint.exists() and not checkpoint.is_dir())
    ):
        raise RuntimeError("RLLM2-S3 resume state is unavailable")
    for path in _all_final_artifact_paths():
        target = repository_path(path)
        if target.is_symlink():
            raise RuntimeError("unsafe RLLM2-S3 partial final artifact")
        if target.exists() and not target.is_file():
            raise RuntimeError("unsafe RLLM2-S3 partial final artifact")


def _remove_partial_final_artifacts() -> None:
    for path in _all_final_artifact_paths():
        target = repository_path(path)
        if target.exists():
            if target.is_symlink() or not target.is_file():
                raise RuntimeError("unsafe RLLM2-S3 partial final artifact")
            target.unlink()


def run_stage(
    *,
    scorer_factory: Callable[[Any], Any] = Gemma4SourceFeatureScorer,
    resume: bool = False,
) -> dict[str, Any]:
    if resume:
        _guard_resume_paths()
    else:
        _guard_initial_paths()
    execution_commit = _clean_committed_head()
    prepared = prepare_source_only_stage()
    attempt_path = repository_path(DEFAULT_ATTEMPT)
    output_path = repository_path(DEFAULT_OUTPUT)
    checkpoint_directory = repository_path(
        DEFAULT_CHECKPOINT_DIRECTORY
    )
    if resume:
        attempt = _read_exact_json(attempt_path)
        _validate_resume_attempt(
            attempt,
            execution_commit=execution_commit,
            prepared=prepared,
            output_path=output_path,
            checkpoint_directory=checkpoint_directory,
        )
    else:
        attempt = _attempt_payload(
            execution_commit=execution_commit,
            output_path=output_path,
            checkpoint_directory=checkpoint_directory,
            prepared=prepared,
        )
        _write_new_json(attempt_path, attempt)
    stage = "PRE_MARKET_REPEATABILITY_AND_CAPACITY"
    scorer: Any | None = None
    repeatability_result: dict[str, Any] | None = None
    completed_rows = 0
    payload: dict[str, Any]
    try:
        if resume:
            repeatability_result = validate_repeatability_result(
                attempt_path=attempt_path,
                attempt=attempt,
                prepared=prepared,
            )
            recovered = _recover_publish_transaction(
                checkpoint_directory=checkpoint_directory,
                output_path=output_path,
            )
            if recovered is not None:
                return recovered
        else:
            repeatability_result = run_pre_market_gates(
                scorer_factory,
                prepared["processor"],
                execution_commit=execution_commit,
                attempt_path=attempt_path,
                attempt=attempt,
                prepared=prepared,
            )
        stage = "CHECKPOINT_BOOTSTRAP"
        binding = _checkpoint_binding(
            attempt_path=attempt_path,
            attempt=attempt,
            prepared=prepared,
            repeatability_result=repeatability_result,
        )
        if resume:
            _remove_partial_final_artifacts()
            if checkpoint_directory.exists():
                _validate_checkpoint_binding(
                    checkpoint_directory,
                    binding,
                )
            else:
                _create_checkpoint_directory(
                    checkpoint_directory,
                    binding,
                )
        else:
            _create_checkpoint_directory(
                checkpoint_directory,
                binding,
            )
        _validate_checkpoint_binding(checkpoint_directory, binding)
        existing_shards, prior_hash, completed_rows = (
            _load_checkpoint_prefix(
                checkpoint_directory,
                rows=prepared["rows"],
            )
        )
        _resolve_inflight_row(
            checkpoint_directory,
            rows=prepared["rows"],
            completed_rows=completed_rows,
            prior_shard_hash=prior_hash,
        )
        resumed_from_rows = completed_rows
        stage = "FRESH_THIRD_MODEL_CONSTRUCTION"
        scorer = scorer_factory(prepared["processor"])
        stage = "SOURCE_FEATURE_FORWARDS"
        shards = list(existing_shards)
        shard_index = len(shards)
        while completed_rows < len(prepared["rows"]):
            stop = min(
                completed_rows + prereg.CHECKPOINT_SHARD_SIZE,
                len(prepared["rows"]),
            )
            if stop != completed_rows + 1:
                raise RuntimeError(
                    "RLLM2-S3 checkpoint is not row granular"
                )
            _begin_inflight_row(
                checkpoint_directory,
                row=prepared["rows"][completed_rows],
                prior_shard_hash=prior_hash,
            )
            arrays = _process_shard(
                scorer,
                rows=prepared["rows"],
                policy_counts=prepared["prompt_capacity"]["policy"][
                    "counts"
                ],
                relation_counts=prepared["prompt_capacity"][
                    "relation_teacher"
                ]["counts"],
                start=completed_rows,
                stop=stop,
            )
            metadata = _write_checkpoint_shard(
                checkpoint_directory,
                shard_index=shard_index,
                start=completed_rows,
                stop=stop,
                prior_shard_hash=prior_hash,
                rows=prepared["rows"],
                arrays=arrays,
            )
            shards.append(arrays)
            prior_hash = str(metadata["shard_hash"])
            completed_rows = stop
            shard_index += 1
            _clear_inflight_row(checkpoint_directory)
        stage = "FINAL_ARTIFACT_ASSEMBLY"
        merged = _merge_shards(shards, rows=prepared["rows"])
        artifacts, artifact_bytes = _build_final_artifacts(
            rows=prepared["rows"],
            arrays=merged,
        )
        payload = _success_payload(
            execution_commit=execution_commit,
            attempt_path=attempt_path,
            attempt=attempt,
            prepared=prepared,
            scorer=scorer,
            repeatability_result=repeatability_result,
            artifacts=artifacts,
            checkpoint_chain_hash=prior_hash,
            checkpoint_shards=len(shards),
            resumed=resume,
            resumed_from_rows=resumed_from_rows,
        )
        stage = "ATOMIC_FINAL_PUBLISH"
        _publish_success_atomically(
            checkpoint_directory=checkpoint_directory,
            output_path=output_path,
            payload=payload,
            artifact_records=artifacts,
            artifact_bytes=artifact_bytes,
        )
    except BaseException as error:
        terminal_interruption = stage == (
            "PRE_MARKET_REPEATABILITY_AND_CAPACITY"
        )
        if (
            isinstance(error, Exception) or terminal_interruption
        ) and not output_path.exists() and not output_path.is_symlink():
            failure = _failure_payload(
                execution_commit=execution_commit,
                attempt_path=attempt_path,
                attempt=attempt,
                stage=stage,
                error=error,
                scorer=scorer,
                completed_rows=completed_rows,
            )
            _write_new_json(output_path, failure)
        if scorer is not None:
            close = getattr(scorer, "close", None)
            if callable(close):
                close()
            scorer = None
            gc.collect()
        raise
    if scorer is not None:
        close = getattr(scorer, "close", None)
        if callable(close):
            close()
        scorer = None
        gc.collect()
    if checkpoint_directory.exists() or checkpoint_directory.is_symlink():
        raise RuntimeError(
            "RLLM2-S3 checkpoint survived successful publish"
        )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate authority, runtime, tokenizer, source roster, and "
            "prompt caps without loading model weights or consuming attempt."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted attempt from its contiguous verified "
            "checkpoint prefix."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_only:
        if args.resume:
            raise RuntimeError("--validate-only and --resume are exclusive")
        _guard_initial_paths()
        prepared = prepare_source_only_stage()
        capacity = {
            key: value
            for key, value in prepared["prompt_capacity"].items()
            if key not in {"policy", "relation_teacher"}
        }
        capacity["policy"] = {
            key: value
            for key, value in prepared["prompt_capacity"]["policy"].items()
            if key != "counts"
        }
        capacity["relation_teacher"] = {
            key: value
            for key, value in prepared["prompt_capacity"][
                "relation_teacher"
            ].items()
            if key != "counts"
        }
        print(
            json.dumps(
                {
                    "stage_id": prereg.STAGE_ID,
                    "runtime": prepared["runtime"],
                    "source_roster": prepared["roster"],
                    "prompt_capacity": capacity,
                    "model_weights_loaded": False,
                    "official_attempt_consumed": False,
                    "market_accessed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    payload = run_stage(resume=args.resume)
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "terminal_action": payload["terminal_action"],
                "source_roster": payload["source_roster"],
                "model_metrics": payload["model_metrics"],
                "artifacts": payload["artifacts"],
                "result_hash": payload["result_hash"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
