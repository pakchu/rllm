#!/usr/bin/env python3
"""Run the resumable PSIM-D8-RLLM2 source-only feature seal."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Callable, Mapping, Sequence
import zipfile

import numpy as np

_BOOTSTRAP_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_REPO_ROOT))

from training import preregister_psim_d8_rllm2_source_feature_seal as prereg
from training import run_psim_d8_rllm1_base_memorization_gate as base
from training import run_psim_d8_rllm2_base_memorization_gate as rllm2_gate
from utils import disable_transformers_allocator_warmup


REPO_ROOT = _BOOTSTRAP_REPO_ROOT
PROTOCOL_VERSION = "psim_d8_rllm2_source_feature_seal_v1"
PREREGISTRATION = prereg.DEFAULT_OUTPUT
PREREGISTRATION_SHA256 = (
    "de50bdd2615cbe7b5232d31fc2388646a50997a3800af465f507bad82c94691c"
)
PREREGISTRATION_MANIFEST_HASH = (
    "4178275879f98cb3ef1b068c8060aa8ad761e3dcd72acd67940c8150bb98a32a"
)
SOURCE_ROW_ROSTER_HASH = (
    "033df68d9067a88cb14eb83f92b7638f0addc2372ef08184ef75e9fe3f7ba47c"
)
PASS_ACTION = (
    "ACCEPT_PSIM_D8_RLLM2_S1_SOURCE_FEATURE_SEAL_"
    "OPEN_2020_TRAIN_OUTCOMES_ONLY"
)
FAILURE_ACTION = (
    "REJECT_PSIM_D8_RLLM2_S1_NO_REPAIR_RERUN_MODEL_SWAP_"
    "OR_MARKET_ACCESS"
)

DEFAULT_ATTEMPT = prereg.ATTEMPT_PATH
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
        raise RuntimeError(f"unsafe RLLM2-S1 file: {path}")
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
        raise RuntimeError("unsafe RLLM2-S1 preregistration artifact")
    raw = target.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PREREGISTRATION_SHA256:
        raise RuntimeError("RLLM2-S1 preregistration SHA changed")
    payload = json.loads(raw.decode("utf-8"))
    if (
        payload != prereg.build_preregistration()
        or payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH
        or payload.get("candidate", {}).get("id") != prereg.STAGE_ID
        or payload.get("source_row_contract", {})
        .get("roster", {})
        .get("source_row_roster_hash")
        != SOURCE_ROW_ROSTER_HASH
        or payload.get("access_boundary", {}).get(
            "market_or_funding_payload_bytes_hashed"
        )
        is not False
    ):
        raise RuntimeError("RLLM2-S1 preregistration contract changed")
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
            raise RuntimeError("RLLM2-S1 prompt exceeds frozen token cap")
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
        raise RuntimeError("RLLM2-S1 prompt roster changed")
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
        for code in prereg.RELATION_CODE_ORDER
    }
    if len(set(code_ids.values())) != len(code_ids):
        raise RuntimeError("RLLM2-S1 relation code token IDs collide")
    runtime["tokenizer"] = {
        **tokenizer,
        "relation_code_token_ids": code_ids,
    }
    runtime["chat_template"] = base.validate_chat_template_contract(
        processor
    )
    rows = prereg.build_source_rows()
    roster = prereg.source_roster_contract(rows)
    if roster["source_row_roster_hash"] != SOURCE_ROW_ROSTER_HASH:
        raise RuntimeError("RLLM2-S1 source roster changed")
    capacity = validate_prompt_capacity(processor, rows)
    return {
        "preregistration": preregistration,
        "runtime": runtime,
        "processor": processor,
        "rows": rows,
        "roster": roster,
        "prompt_capacity": capacity,
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
        raise RuntimeError(f"RLLM2-S1 output already exists: {path}")
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
        raise RuntimeError(f"unsafe RLLM2-S1 JSON: {path}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"malformed RLLM2-S1 JSON: {path}")
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
        "predecessor_result_hash": prereg.RLLM2_RESULT_HASH,
        "source_row_roster_hash": prepared["roster"][
            "source_row_roster_hash"
        ],
        "output": _display_path(output_path),
        "artifact_paths": {
            "source_rows": DEFAULT_SOURCE_ROWS.as_posix(),
            "embeddings": DEFAULT_EMBEDDINGS.as_posix(),
            "relation_logits": DEFAULT_RELATION_LOGITS.as_posix(),
            "relation_rows": DEFAULT_RELATION_ROWS.as_posix(),
        },
        "checkpoint_directory": _display_path(checkpoint_directory),
        "checkpoint_shard_size": prereg.SHARD_SIZE,
        "model_id": prereg.rllm1.MODEL_ID,
        "model_revision": prereg.rllm1.MODEL_REVISION,
        "model_inference_authorized": True,
        "market_access_authorized": False,
        "resume_after_process_interruption_authorized": True,
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
    if (
        attempt.get("attempt_hash") != canonical_hash(core)
        or attempt.get("protocol_version") != PROTOCOL_VERSION
        or attempt.get("stage_id") != prereg.STAGE_ID
        or attempt.get("execution_commit") != execution_commit
        or attempt.get("runner_sha256") != sha256_file(Path(__file__))
        or attempt.get("preregistration", {}).get("sha256")
        != PREREGISTRATION_SHA256
        or attempt.get("preregistration", {}).get("manifest_hash")
        != PREREGISTRATION_MANIFEST_HASH
        or attempt.get("predecessor_result_hash")
        != prereg.RLLM2_RESULT_HASH
        or attempt.get("source_row_roster_hash")
        != prepared["roster"]["source_row_roster_hash"]
        or attempt.get("output") != _display_path(output_path)
        or attempt.get("checkpoint_directory")
        != _display_path(checkpoint_directory)
        or attempt.get("market_access_authorized") is not False
        or attempt.get("resume_after_process_interruption_authorized")
        is not True
    ):
        raise RuntimeError("RLLM2-S1 resume attempt binding changed")


def _checkpoint_binding(
    *,
    attempt_path: Path,
    attempt: Mapping[str, Any],
    prepared: Mapping[str, Any],
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
        "source_row_roster_hash": prepared["roster"][
            "source_row_roster_hash"
        ],
        "row_count": len(prepared["rows"]),
        "shard_size": prereg.SHARD_SIZE,
    }
    return {**core, "binding_hash": canonical_hash(core)}


def _create_checkpoint_directory(
    checkpoint_directory: Path,
    binding: Mapping[str, Any],
) -> None:
    target = repository_path(checkpoint_directory)
    if target.exists() or target.is_symlink():
        raise RuntimeError("RLLM2-S1 checkpoint directory already exists")
    target.mkdir(parents=True)
    _write_new_json(target / "binding.json", binding)


def _validate_checkpoint_binding(
    checkpoint_directory: Path,
    expected: Mapping[str, Any],
) -> None:
    target = repository_path(checkpoint_directory)
    if target.is_symlink() or not target.is_dir():
        raise RuntimeError("RLLM2-S1 checkpoint directory is unsafe")
    observed = _read_exact_json(target / "binding.json")
    if observed != dict(expected):
        raise RuntimeError("RLLM2-S1 checkpoint binding changed")


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
    if logits.shape != (len(prereg.RELATION_CODE_ORDER),):
        raise RuntimeError("RLLM2-S1 relation logit width changed")
    if not np.all(np.isfinite(logits)):
        return _canonical_nan_row(len(prereg.RELATION_CODE_ORDER))
    return logits


def _predicted_code(logits: np.ndarray) -> str | None:
    values = np.asarray(logits, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        return None
    best = float(np.max(values))
    return next(
        code
        for code, value in zip(prereg.RELATION_CODE_ORDER, values)
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
        raise RuntimeError("RLLM2-S1 checkpoint array names changed")
    count = stop - start
    if (
        arrays["embedding"].shape
        != (count, prereg.EMBEDDING_WIDTH)
        or arrays["embedding"].dtype != np.float32
        or arrays["relation_logits"].shape
        != (count, len(prereg.RELATION_CODE_ORDER))
        or arrays["relation_logits"].dtype != np.float32
        or arrays["row_index"].shape != (count,)
        or arrays["policy_input_tokens"].shape != (count,)
        or arrays["relation_input_tokens"].shape != (count,)
        or arrays["relation_forwarded"].shape != (count,)
    ):
        raise RuntimeError("RLLM2-S1 checkpoint array shape changed")
    if not np.array_equal(
        arrays["row_index"],
        np.arange(start, stop, dtype=np.int32),
    ):
        raise RuntimeError("RLLM2-S1 checkpoint row order changed")
    if not np.all(np.isfinite(arrays["embedding"])):
        raise RuntimeError("RLLM2-S1 embedding contains nonfinite values")
    for local_index, row in enumerate(rows[start:stop]):
        forwarded = int(arrays["relation_forwarded"][local_index])
        expected_forwarded = int(
            bool(row["relation_teacher_forward_required"])
        )
        logits = arrays["relation_logits"][local_index]
        if forwarded != expected_forwarded:
            raise RuntimeError("RLLM2-S1 relation forward mask changed")
        if forwarded:
            if not (
                np.all(np.isfinite(logits))
                or np.all(np.isnan(logits))
            ):
                raise RuntimeError(
                    "RLLM2-S1 relation logits are partially nonfinite"
                )
        elif not np.all(np.isnan(logits)):
            raise RuntimeError(
                "RLLM2-S1 skipped relation logits are not canonical NaN"
            )
    for name in ("policy_input_tokens", "relation_input_tokens"):
        counts = arrays[name]
        if (
            np.any(counts <= 0)
            or np.any(counts > prereg.MAXIMUM_INPUT_TOKENS)
        ):
            raise RuntimeError("RLLM2-S1 checkpoint token count changed")


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
    stop = min(expected_start + prereg.SHARD_SIZE, len(rows))
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
        raise RuntimeError("RLLM2-S1 checkpoint shard binding changed")
    try:
        with np.load(npz_path, allow_pickle=False) as payload:
            arrays = {
                name: np.asarray(payload[name])
                for name in payload.files
            }
    except Exception as exc:
        raise RuntimeError(
            f"RLLM2-S1 checkpoint shard unreadable: {exc}"
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
                raise RuntimeError("unsafe RLLM2-S1 final staging entry")
            shutil.rmtree(path)
        elif path.name.endswith(".tmp"):
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("unsafe RLLM2-S1 checkpoint temporary")
            path.unlink()
        elif (
            path.name.startswith("shard_")
            and path.suffix == ".npz"
            and path.stem not in metadata_stems
        ):
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("unsafe RLLM2-S1 orphan checkpoint")
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
            raise RuntimeError("unsafe RLLM2-S1 checkpoint entry")
        if path.name.startswith("shard_") and path.suffix in {
            ".json",
            ".npz",
        }:
            try:
                index = int(path.stem.split("_", 1)[1])
            except (IndexError, ValueError) as exc:
                raise RuntimeError(
                    "malformed RLLM2-S1 checkpoint filename"
                ) from exc
            if path.suffix == ".json":
                metadata_indices.append(index)
            continue
        raise RuntimeError(
            f"unexpected RLLM2-S1 checkpoint entry: {path.name}"
        )
    ordered = sorted(metadata_indices)
    if ordered != list(range(len(ordered))):
        raise RuntimeError(
            "RLLM2-S1 checkpoints are not a contiguous prefix"
        )
    shards: list[dict[str, np.ndarray]] = []
    prior = canonical_hash(
        {
            "state": "PSIM_D8_RLLM2_S1_CHECKPOINT_CHAIN_START",
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
        raise RuntimeError("RLLM2-S1 in-flight sentinel is unsafe")
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
        raise RuntimeError("RLLM2-S1 in-flight binding changed")
    if completed_rows == row_index + 1:
        if prior_shard_hash == payload.get("prior_shard_hash"):
            raise RuntimeError(
                "RLLM2-S1 committed row did not advance checkpoint chain"
            )
        _clear_inflight_row(checkpoint_directory)
        return
    if completed_rows == row_index:
        raise RuntimeError(
            "RLLM2-S1 interrupted row has ambiguous started forwards"
        )
    raise RuntimeError("RLLM2-S1 in-flight row is outside checkpoint prefix")


class Gemma4SourceFeatureScorer:
    def __init__(self, processor: Any) -> None:
        import torch
        from transformers import (
            AutoModelForMultimodalLM,
            BitsAndBytesConfig,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("RLLM2-S1 requires CUDA")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("RLLM2-S1 requires one visible CUDA device")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("RLLM2-S1 requires CUDA BF16")
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        self.torch = torch
        self.processor = processor
        self.forward_calls_started = 0
        self.embedding_forwards_started = 0
        self.relation_forwards_started = 0
        self.embedding_inference_seconds = 0.0
        self.relation_inference_seconds = 0.0
        tokenizer_contract = base.validate_processor_tokenizer(processor)
        self.code_ids = {
            code: tokenizer_contract["challenge_code_token_ids"][code]
            for code in prereg.RELATION_CODE_ORDER
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

    def _inputs(self, prompt: str) -> tuple[Any, int]:
        inputs = _prompt_inputs(self.processor, prompt)
        tokens = int(inputs["input_ids"].shape[-1])
        if tokens > prereg.MAXIMUM_INPUT_TOKENS:
            raise RuntimeError("RLLM2-S1 prompt exceeds frozen token cap")
        if int(inputs["attention_mask"][0, -1].item()) != 1:
            raise RuntimeError("RLLM2-S1 final prompt token is padding")
        return inputs.to(self.model.device), tokens

    def embed(self, prompt: str) -> tuple[np.ndarray, int]:
        inputs, tokens = self._inputs(prompt)
        started = time.perf_counter()
        self.forward_calls_started += 1
        self.embedding_forwards_started += 1
        with self.torch.inference_mode():
            outputs = self.model.model(
                **inputs,
                use_cache=False,
                return_dict=True,
            )
        self.embedding_inference_seconds += time.perf_counter() - started
        vector = (
            outputs.last_hidden_state[0, -1]
            .float()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        if vector.shape != (prereg.EMBEDDING_WIDTH,):
            raise RuntimeError("RLLM2-S1 embedding width changed")
        if not np.all(np.isfinite(vector)):
            raise RuntimeError("RLLM2-S1 embedding is nonfinite")
        return vector, tokens

    def score_relation(
        self,
        prompt: str,
    ) -> tuple[np.ndarray, str | None, int]:
        inputs, tokens = self._inputs(prompt)
        started = time.perf_counter()
        self.forward_calls_started += 1
        self.relation_forwards_started += 1
        with self.torch.inference_mode():
            outputs = self.model(
                **inputs,
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            )
        self.relation_inference_seconds += time.perf_counter() - started
        final = outputs.logits[0, -1].float()
        logits = _normalize_relation_logits(
            [
                float(final[self.code_ids[code]].item())
                for code in prereg.RELATION_CODE_ORDER
            ]
        )
        return logits, _predicted_code(logits), tokens

    def metrics(self) -> dict[str, Any]:
        properties = self.torch.cuda.get_device_properties(0)
        peak_allocated = int(self.torch.cuda.max_memory_allocated())
        peak_reserved = int(self.torch.cuda.max_memory_reserved())
        if peak_allocated > base.MAXIMUM_PEAK_ALLOCATED_BYTES:
            raise RuntimeError("RLLM2-S1 peak VRAM cap exceeded")
        return {
            "gpu_name": properties.name,
            "gpu_total_memory_bytes": int(properties.total_memory),
            "cuda_device_count": self.torch.cuda.device_count(),
            "bf16_supported": self.torch.cuda.is_bf16_supported(),
            "model_load_seconds": self.load_seconds,
            "placement": self.placement,
            "model_forwards_started": self.forward_calls_started,
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
                base.MAXIMUM_PEAK_ALLOCATED_BYTES
            ),
        }


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
            raise RuntimeError("RLLM2-S1 policy token count changed")
        embeddings.append(np.asarray(vector, dtype=np.float32))
        observed_policy_tokens.append(policy_tokens)
        if bool(row["relation_teacher_forward_required"]):
            logits, _, relation_tokens = scorer.score_relation(
                str(row["relation_teacher_prompt"])
            )
            if relation_tokens != int(relation_counts[row_index]):
                raise RuntimeError(
                    "RLLM2-S1 relation token count changed"
                )
            relation_logits.append(
                _normalize_relation_logits(logits)
            )
            forwarded.append(1)
            observed_relation_tokens.append(relation_tokens)
        else:
            relation_logits.append(
                _canonical_nan_row(len(prereg.RELATION_CODE_ORDER))
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
        raise RuntimeError("RLLM2-S1 has no checkpoint shards")
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
            "schema_version": prereg.RELATION_ROW_SCHEMA_VERSION,
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
        raise RuntimeError("RLLM2-S1 final staging already exists")
    staging.mkdir()
    staged_artifacts: dict[str, Path] = {}
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
            raise RuntimeError("RLLM2-S1 artifact record changed")
        staged = staging / Path(str(record["path"])).name
        _write_new_bytes(staged, raw)
        if sha256_file(staged) != record["sha256"]:
            raise RuntimeError("RLLM2-S1 staged artifact hash changed")
        staged_artifacts[name] = staged
    result_raw = canonical_json_bytes(dict(payload), pretty=True)
    staged_result = staging / output_path.name
    _write_new_bytes(staged_result, result_raw)
    observed = _read_exact_json(staged_result)
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
        raise RuntimeError("RLLM2-S1 staged result verification failed")
    for name in (
        "source_rows",
        "embeddings",
        "relation_logits",
        "relation_rows",
    ):
        target = repository_path(
            Path(str(artifact_records[name]["path"]))
        )
        if target.exists() or target.is_symlink():
            raise RuntimeError("RLLM2-S1 final artifact already exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        staged_artifacts[name].replace(target)
        if sha256_file(target) != artifact_records[name]["sha256"]:
            raise RuntimeError("RLLM2-S1 promoted artifact hash changed")
    if output_path.exists() or output_path.is_symlink():
        raise RuntimeError("RLLM2-S1 terminal result already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staged_result.replace(output_path)


def _success_payload(
    *,
    execution_commit: str,
    attempt_path: Path,
    attempt: Mapping[str, Any],
    prepared: Mapping[str, Any],
    scorer: Any,
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
        "predecessor_result_hash": prereg.RLLM2_RESULT_HASH,
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
        "checkpoint_evidence": {
            "shard_size": prereg.SHARD_SIZE,
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
                    prereg.RLLM2_PREREGISTRATION.as_posix(),
                    prereg.RLLM2_ATTEMPT.as_posix(),
                    prereg.RLLM2_RESULT.as_posix(),
                    prereg.rllm1.D8_CARDS.as_posix(),
                }
            ),
            "model_snapshot_read": str(base._model_snapshot()),
            "market_or_funding_paths_read": [],
            "market_rows_parsed": 0,
            "funding_rows_parsed": 0,
            "market_or_funding_payload_bytes_hashed": False,
            "model_outputs_created": (
                prepared["roster"]["embedding_forward_count"]
                + prepared["roster"]["relation_teacher_forward_count"]
            ),
            "rewards_created": 0,
            "economic_metrics_computed": 0,
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
    error: Exception,
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
            "rewards_created": 0,
            "economic_metrics_computed": 0,
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
    paths = (
        DEFAULT_ATTEMPT,
        DEFAULT_OUTPUT,
        DEFAULT_CHECKPOINT_DIRECTORY,
        *_all_final_artifact_paths(),
    )
    if any(
        repository_path(path).exists()
        or repository_path(path).is_symlink()
        for path in paths
    ):
        raise RuntimeError("RLLM2-S1 source feature seal already attempted")


def _guard_resume_paths() -> None:
    attempt = repository_path(DEFAULT_ATTEMPT)
    result = repository_path(DEFAULT_OUTPUT)
    checkpoint = repository_path(DEFAULT_CHECKPOINT_DIRECTORY)
    if (
        attempt.is_symlink()
        or not attempt.is_file()
        or result.exists()
        or result.is_symlink()
        or checkpoint.is_symlink()
        or not checkpoint.is_dir()
    ):
        raise RuntimeError("RLLM2-S1 resume state is unavailable")
    for path in _all_final_artifact_paths():
        target = repository_path(path)
        if target.is_symlink():
            raise RuntimeError("unsafe RLLM2-S1 partial final artifact")
        if target.exists() and not target.is_file():
            raise RuntimeError("unsafe RLLM2-S1 partial final artifact")


def _remove_partial_final_artifacts() -> None:
    for path in _all_final_artifact_paths():
        target = repository_path(path)
        if target.exists():
            if target.is_symlink() or not target.is_file():
                raise RuntimeError("unsafe RLLM2-S1 partial final artifact")
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
    stage = "CHECKPOINT_BOOTSTRAP"
    scorer: Any | None = None
    completed_rows = 0
    payload: dict[str, Any]
    try:
        binding = _checkpoint_binding(
            attempt_path=attempt_path,
            attempt=attempt,
            prepared=prepared,
        )
        if resume:
            _remove_partial_final_artifacts()
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
        stage = "MODEL_CONSTRUCTION"
        scorer = scorer_factory(prepared["processor"])
        stage = "SOURCE_FEATURE_FORWARDS"
        shards = list(existing_shards)
        shard_index = len(shards)
        while completed_rows < len(prepared["rows"]):
            stop = min(
                completed_rows + prereg.SHARD_SIZE,
                len(prepared["rows"]),
            )
            if stop != completed_rows + 1:
                raise RuntimeError(
                    "RLLM2-S1 checkpoint is not row granular"
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
    except Exception as error:
        if not output_path.exists() and not output_path.is_symlink():
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
        raise
    try:
        shutil.rmtree(checkpoint_directory)
    except OSError:
        pass
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
