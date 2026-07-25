"""Preregister the source- and outcome-blind CEFS-D2 successor."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

from training import build_cboe_edge_flip_sequence_policy_support as d1_engine
from training import preregister_cboe_edge_flip_sequence_policy as d1_prereg


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
POLICY_ID = "CEFS-D2"
PROTOCOL_VERSION = "cefs_d2_preregistration_v1"
PRODUCER_SCRIPT = "training/preregister_cboe_edge_flip_sequence_policy_d2.py"
DEFAULT_OUTPUT = (
    "results/cboe_edge_flip_sequence_policy_d2_preregistration_2026-07-25.json"
)

SUCCESSOR_DECISION = "docs/cefs-d2-successor-decision-2026-07-25.md"
SUCCESSOR_DECISION_COMMIT = "b0061fe4c944eefdbdd5e72139e3765b2fd464f1"
SUCCESSOR_DECISION_SHA256 = (
    "60a9f9b01a1943f9a3079885c36ac9af2bfe4686a28e206c8198b36880211aea"
)
BOUNDARY_DOCUMENT = (
    "docs/cboe-edge-flip-sequence-policy-d2-boundary-2026-07-25.md"
)
BOUNDARY_COMMIT = "2ee9831a34d808ba3a3abcdf9f93281ee2dc8ef8"
BOUNDARY_DOCUMENT_SHA256 = (
    "d679d23465963485756bde7f7cc7660607cfbcc1326baaaf367b3eaac34e387b"
)

D1_BOUNDARY = d1_prereg.BOUNDARY_DOCUMENT
D1_BOUNDARY_COMMIT = d1_prereg.BOUNDARY_COMMIT
D1_BOUNDARY_SHA256 = d1_prereg.BOUNDARY_DOCUMENT_SHA256
D1_PREREGISTRATION = d1_prereg.DEFAULT_OUTPUT
D1_PREREGISTRATION_COMMIT = "d60d9744e82f48350be506f1da63bcb2e706cf03"
D1_PREREGISTRATION_SHA256 = (
    "5e515663e99ef4aa322cae25cfb2c07f69b3e24f289bc2f0f79463aca64a8878"
)
D1_PREREGISTRATION_MANIFEST_HASH = (
    "9aa7c891ec241d4733db215068bed3507f41c03cbae7198c906a079ddb6467bf"
)
D1_PRODUCER = d1_prereg.PRODUCER_SCRIPT
D1_PRODUCER_COMMIT = "ec8eae23226d39f0c62b6c5711d6080f2bf990a4"
D1_PRODUCER_SHA256 = (
    "1c4668e72846eadf66011c582c62e8574af3679204500c1ff9631101ecbb7ac1"
)
D1_ENGINE = d1_engine.RUNNER_PATH
D1_ENGINE_COMMIT = "d7213f647128fc6160672bc61f080b3dcf7d1f42"
D1_ENGINE_SHA256 = (
    "2069084d65146540488672115ee09f292cd31e6611bf92a569d534ab8a74c688"
)
D1_REJECTION = d1_engine.REJECTION_REPORT
D1_REJECTION_COMMIT = "0ded877c10d564c2a5c67caf8524f4807b13b28e"
D1_REJECTION_SHA256 = (
    "4c2839e5ac59738367d5116ff05ed50c900d16f06de8c4d4cc724fd25978c169"
)
D1_REJECTION_RESULT_HASH = (
    "9963981f6d56fcff65f1367fc7c3c1fc006b60b821894b3e0ff59c6b9aa35d7b"
)

GIT_EXECUTABLE = "/usr/bin/git"
GIT_EXECUTABLE_SHA256 = (
    "2a8c18fbf43da9f692d75474c72bea9dfd796c260b0f3dfe456376abc3bbd668"
)
D1_SCIENTIFIC_KEYS = (
    "chronology",
    "clock",
    "contingent_economic_chronology",
    "controls",
    "decision",
    "forbidden_access",
    "parsing",
    "phase",
    "policy",
    "relation_language",
    "research_history",
    "sequence_language",
    "sources",
    "support_gates",
)
TERMINAL_ACTIONS = {
    "failure": "retire_cefs_d2_unchanged_before_outcomes",
    "pass": "authorize_cefs_d2_economic_rllm_evaluator_freeze_only",
}


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(repository_path(path).read_bytes()).hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        (GIT_EXECUTABLE, *args),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _assert_committed(path: str, *, expected_commit: str | None = None) -> str:
    _git_output("ls-files", "--error-unmatch", "--", path)
    for args in (
        ("diff", "--quiet", "--", path),
        ("diff", "--cached", "--quiet", "--", path),
    ):
        if subprocess.run(
            (GIT_EXECUTABLE, *args),
            cwd=REPOSITORY_ROOT,
            check=False,
        ).returncode:
            raise RuntimeError(f"CEFS-D2 frozen artifact is dirty: {path}")
    commit = _git_output("log", "-1", "--format=%H", "--", path)
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError(
            f"CEFS-D2 frozen artifact commit mismatch: {path}"
        )
    return commit


def _git_blob_sha256(commit: str, path: str) -> str:
    completed = subprocess.run(
        (GIT_EXECUTABLE, "show", f"{commit}:{path}"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def validate_runtime_authority() -> dict[str, str]:
    git_path = Path(GIT_EXECUTABLE)
    path_value = os.environ.get("PATH")
    if not path_value:
        raise RuntimeError("CEFS-D2 PATH is missing")
    if "/usr/bin" not in path_value.split(os.pathsep):
        raise RuntimeError("CEFS-D2 PATH lacks exact /usr/bin component")
    if git_path.is_symlink() or not git_path.is_file():
        raise RuntimeError("CEFS-D2 Git executable is not a regular file")
    if sha256_file(git_path) != GIT_EXECUTABLE_SHA256:
        raise RuntimeError("CEFS-D2 Git executable hash mismatch")
    version = subprocess.run(
        (GIT_EXECUTABLE, "--version"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not version.startswith("git version "):
        raise RuntimeError("CEFS-D2 Git executable version probe failed")
    return {
        "path": GIT_EXECUTABLE,
        "sha256": GIT_EXECUTABLE_SHA256,
        "version": version,
        "path_component": "/usr/bin",
    }


def _binding(path: str, commit: str, digest: str) -> dict[str, str]:
    return {"path": path, "commit": commit, "sha256": digest}


def validate_frozen_dependencies() -> dict[str, Any]:
    runtime = validate_runtime_authority()
    bindings = (
        (SUCCESSOR_DECISION, SUCCESSOR_DECISION_COMMIT, SUCCESSOR_DECISION_SHA256),
        (BOUNDARY_DOCUMENT, BOUNDARY_COMMIT, BOUNDARY_DOCUMENT_SHA256),
        (D1_BOUNDARY, D1_BOUNDARY_COMMIT, D1_BOUNDARY_SHA256),
        (
            D1_PREREGISTRATION,
            D1_PREREGISTRATION_COMMIT,
            D1_PREREGISTRATION_SHA256,
        ),
        (D1_PRODUCER, D1_PRODUCER_COMMIT, D1_PRODUCER_SHA256),
        (D1_ENGINE, D1_ENGINE_COMMIT, D1_ENGINE_SHA256),
        (D1_REJECTION, D1_REJECTION_COMMIT, D1_REJECTION_SHA256),
    )
    for path, commit, digest in bindings:
        if _assert_committed(path, expected_commit=commit) != commit:
            raise RuntimeError(f"CEFS-D2 dependency commit mismatch: {path}")
        if sha256_file(path) != digest:
            raise RuntimeError(f"CEFS-D2 dependency hash mismatch: {path}")
        if _git_blob_sha256(commit, path) != digest:
            raise RuntimeError(f"CEFS-D2 dependency blob mismatch: {path}")
    d1_payload = json.loads(repository_path(D1_PREREGISTRATION).read_text())
    d1_prereg.validate_manifest(d1_payload)
    expected_d1_producer = _binding(
        D1_PRODUCER,
        D1_PRODUCER_COMMIT,
        D1_PRODUCER_SHA256,
    )
    if (
        d1_payload["manifest_hash"] != D1_PREREGISTRATION_MANIFEST_HASH
        or d1_payload.get("authority", {}).get("producer")
        != expected_d1_producer
        or _git_blob_sha256(D1_PRODUCER_COMMIT, D1_PRODUCER)
        != D1_PRODUCER_SHA256
    ):
        raise RuntimeError("CEFS-D2 D1 preregistration manifest mismatch")
    rejection = json.loads(repository_path(D1_REJECTION).read_text())
    rejection_core = {
        key: value for key, value in rejection.items() if key != "result_hash"
    }
    expected_gate = [
        {
            "checks": {"authority_valid": False},
            "index": 1,
            "name": "authority_forbidden_access",
            "passed": False,
        }
    ]
    if (
        rejection.get("result_hash") != D1_REJECTION_RESULT_HASH
        or rejection["result_hash"] != d1_engine.canonical_hash(rejection_core)
        or rejection.get("decision") != "fail"
        or rejection.get("gates") != expected_gate
        or rejection.get("authority") != {}
        or rejection.get("details") != {}
        or rejection.get("source_output") is not None
        or rejection.get("control_output") is not None
        or rejection.get("error")
        != {
            "message": "[Errno 2] No such file or directory: 'git'",
            "type": "FileNotFoundError",
        }
        or any(rejection["forbidden_counters"].values())
    ):
        raise RuntimeError("CEFS-D2 D1 terminal evidence mismatch")
    d1_prereg._validate_source_anchors()

    d1_preregistration = _binding(
        D1_PREREGISTRATION,
        D1_PREREGISTRATION_COMMIT,
        D1_PREREGISTRATION_SHA256,
    )
    d1_preregistration["manifest_hash"] = D1_PREREGISTRATION_MANIFEST_HASH
    d1_rejection = _binding(
        D1_REJECTION,
        D1_REJECTION_COMMIT,
        D1_REJECTION_SHA256,
    )
    d1_rejection.update(
        {
            "result_hash": D1_REJECTION_RESULT_HASH,
            "source_values_opened": False,
            "outcomes_opened": False,
        }
    )
    return {
        "runtime": runtime,
        "successor_decision": _binding(
            SUCCESSOR_DECISION,
            SUCCESSOR_DECISION_COMMIT,
            SUCCESSOR_DECISION_SHA256,
        ),
        "boundary": _binding(
            BOUNDARY_DOCUMENT,
            BOUNDARY_COMMIT,
            BOUNDARY_DOCUMENT_SHA256,
        ),
        "d1_boundary": _binding(
            D1_BOUNDARY,
            D1_BOUNDARY_COMMIT,
            D1_BOUNDARY_SHA256,
        ),
        "d1_preregistration": d1_preregistration,
        "d1_producer": _binding(
            D1_PRODUCER,
            D1_PRODUCER_COMMIT,
            D1_PRODUCER_SHA256,
        ),
        "d1_engine": _binding(
            D1_ENGINE,
            D1_ENGINE_COMMIT,
            D1_ENGINE_SHA256,
        ),
        "d1_rejection": d1_rejection,
    }


def producer_binding() -> dict[str, str]:
    commit = _assert_committed(PRODUCER_SCRIPT)
    return _binding(PRODUCER_SCRIPT, commit, sha256_file(PRODUCER_SCRIPT))


def d1_scientific_contract() -> dict[str, Any]:
    payload = json.loads(repository_path(D1_PREREGISTRATION).read_text())
    if set(payload) != {
        "authority",
        "manifest_hash",
        "policy_id",
        "protocol_version",
        *D1_SCIENTIFIC_KEYS,
    }:
        raise RuntimeError("CEFS-D2 D1 preregistration fields changed")
    return {key: payload[key] for key in D1_SCIENTIFIC_KEYS}


def scientific_mechanism() -> dict[str, Any]:
    contract = d1_scientific_contract()
    return {
        "inheritance": (
            "byte_exact_D1_scientific_contract_without_identity_or_authority"
        ),
        "d1_preregistration_sha256": D1_PREREGISTRATION_SHA256,
        "d1_preregistration_manifest_hash": (
            D1_PREREGISTRATION_MANIFEST_HASH
        ),
        "d1_scientific_contract": contract,
        "d1_scientific_contract_hash": canonical_hash(contract),
    }


def manifest_core(producer: Mapping[str, str]) -> dict[str, Any]:
    mechanism = scientific_mechanism()
    authority = validate_frozen_dependencies()
    authority["producer"] = dict(producer)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "authority": authority,
        "scientific_mechanism": mechanism,
        "scientific_mechanism_hash": canonical_hash(mechanism),
        "terminal_actions": dict(TERMINAL_ACTIONS),
        "forbidden_access": {
            name: 0 for name in d1_engine.FORBIDDEN_COUNTER_NAMES
        },
        "source_rows_parsed": 0,
        "source_values_opened": False,
        "outcomes_opened": False,
    }


def build_manifest() -> dict[str, Any]:
    core = manifest_core(producer_binding())
    payload = dict(core)
    payload["manifest_hash"] = canonical_hash(core)
    return payload


def validate_manifest(payload: Mapping[str, Any]) -> None:
    expected_fields = {
        "protocol_version",
        "policy_id",
        "authority",
        "scientific_mechanism",
        "scientific_mechanism_hash",
        "terminal_actions",
        "forbidden_access",
        "source_rows_parsed",
        "source_values_opened",
        "outcomes_opened",
        "manifest_hash",
    }
    if set(payload) != expected_fields:
        raise RuntimeError("CEFS-D2 preregistration fields mismatch")
    core = {
        key: value for key, value in payload.items() if key != "manifest_hash"
    }
    if payload["manifest_hash"] != canonical_hash(core):
        raise RuntimeError("CEFS-D2 preregistration manifest hash mismatch")
    forbidden = payload["forbidden_access"]
    if (
        payload["protocol_version"] != PROTOCOL_VERSION
        or payload["policy_id"] != POLICY_ID
        or payload["scientific_mechanism"] != scientific_mechanism()
        or payload["scientific_mechanism_hash"]
        != canonical_hash(payload["scientific_mechanism"])
        or payload["terminal_actions"] != TERMINAL_ACTIONS
        or payload["source_rows_parsed"] != 0
        or payload["source_values_opened"] is not False
        or payload["outcomes_opened"] is not False
        or not isinstance(forbidden, dict)
        or set(forbidden) != set(d1_engine.FORBIDDEN_COUNTER_NAMES)
        or any(forbidden.values())
    ):
        raise RuntimeError("CEFS-D2 preregistration invariant mismatch")
    authority = payload["authority"]
    if not isinstance(authority, dict):
        raise RuntimeError("CEFS-D2 authority is not a mapping")
    producer = authority.get("producer")
    if (
        not isinstance(producer, dict)
        or set(producer) != {"path", "commit", "sha256"}
        or producer.get("path") != PRODUCER_SCRIPT
        or not isinstance(producer.get("commit"), str)
        or len(producer["commit"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in producer["commit"]
        )
        or not isinstance(producer.get("sha256"), str)
        or len(producer["sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in producer["sha256"]
        )
    ):
        raise RuntimeError("CEFS-D2 producer binding grammar mismatch")
    expected_authority = validate_frozen_dependencies()
    observed_authority = dict(authority)
    observed_authority.pop("producer")
    if observed_authority != expected_authority:
        raise RuntimeError("CEFS-D2 preregistration authority mismatch")


def _validate_sealed_producer(payload: Mapping[str, Any]) -> None:
    producer = payload["authority"]["producer"]
    try:
        sealed = subprocess.run(
            (
                GIT_EXECUTABLE,
                "show",
                f"{producer['commit']}:{PRODUCER_SCRIPT}",
            ),
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise RuntimeError("CEFS-D2 producer commit is unreadable") from error
    if hashlib.sha256(sealed).hexdigest() != producer["sha256"]:
        raise RuntimeError("CEFS-D2 sealed producer bytes mismatch")


def write_once(path: str | Path, payload: Mapping[str, Any]) -> str:
    if str(path) != DEFAULT_OUTPUT:
        raise RuntimeError("CEFS-D2 preregistration path is frozen")
    validate_manifest(payload)
    _validate_sealed_producer(payload)
    output = repository_path(path)
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if output.exists():
        if output.is_symlink() or not output.is_file():
            raise RuntimeError("CEFS-D2 preregistration is not a regular file")
        if output.read_bytes() != encoded:
            raise RuntimeError("CEFS-D2 preregistration artifact drift")
        return hashlib.sha256(encoded).hexdigest()
    current = producer_binding()
    if current != payload["authority"]["producer"]:
        raise RuntimeError("CEFS-D2 current producer differs from manifest")
    if _git_output("rev-parse", "HEAD") != current["commit"]:
        raise RuntimeError(
            "CEFS-D2 artifact creation requires sealed producer HEAD"
        )
    return d1_engine.write_once_bytes(path, encoded)


def create(path: str = DEFAULT_OUTPUT) -> dict[str, Any]:
    payload = build_manifest()
    write_once(path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = create(args.output)
    print(
        json.dumps(
            {
                "path": args.output,
                "policy_id": payload["policy_id"],
                "manifest_hash": payload["manifest_hash"],
                "source_values_opened": payload["source_values_opened"],
                "outcomes_opened": payload["outcomes_opened"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
