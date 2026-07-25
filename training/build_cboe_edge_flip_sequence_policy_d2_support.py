"""Run frozen, outcome-blind CEFS-D2 source-language support gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from training import build_cboe_edge_flip_sequence_policy_support as engine
from training import preregister_cboe_edge_flip_sequence_policy_d2 as prereg


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
POLICY_ID = prereg.POLICY_ID
RUNNER_PATH = "training/build_cboe_edge_flip_sequence_policy_d2_support.py"
TEST_PATH = "tests/test_build_cboe_edge_flip_sequence_policy_d2_support.py"
CONTRACT_PATH = "docs/cefs-d2-source-support-implementation-contract-2026-07-25.md"
CONTRACT_COMMIT = "de35d9ce4308bf90301eed71f839dfcc8483fc3c"
CONTRACT_SHA256 = (
    "a616815344df2baac95623c8ba2ba0aad4897148a2b917f1688ff6c3f210087a"
)
PREREGISTRATION_PATH = prereg.DEFAULT_OUTPUT
PREREGISTRATION_COMMIT = "203e6b05ea85e36b099cf1f2e1752344fbcbdb77"
PREREGISTRATION_SHA256 = (
    "9f5afdb4647de01e7f5c5130fba4b68cf0c10824f90f33802b57da33d314de2a"
)
PREREGISTRATION_MANIFEST_HASH = (
    "fa1da6ebd2ecc674d64aa95ca0860434db071a62ffbe3442218c73699d312e00"
)
PREREGISTRATION_PRODUCER_COMMIT = (
    "a7ac3239122adf06ded0f374c582cbca9df253da"
)
PREREGISTRATION_PRODUCER_SHA256 = (
    "ab0021a327c351e030098c388ae6a8d35016edcfc50cba42177c3206b4c6a9e4"
)

EXECUTION_SEAL_PATH = (
    "results/cefs_d2_source_support_execution_seal_2026-07-25.json"
)
SOURCE_OUTPUT = (
    "data/cboe_edge_flip_sequence_policy_d2_source_2020_2023.csv.gz"
)
CONTROL_OUTPUT = (
    "data/cboe_edge_flip_sequence_policy_d2_controls_2020_2023.csv.gz"
)
PASS_REPORT = (
    "results/cboe_edge_flip_sequence_policy_d2_source_support_2026-07-25.json"
)
REJECTION_REPORT = (
    "results/cboe_edge_flip_sequence_policy_d2_source_rejection_2026-07-25.json"
)
SEAL_PROTOCOL = "cefs_d2_source_support_execution_seal_v1"
RESULT_PROTOCOL = "cefs_d2_source_support_result_v1"
FAILURE_ACTION = "retire_cefs_d2_unchanged_before_outcomes"
PASS_ACTION = "authorize_cefs_d2_economic_rllm_evaluator_freeze_only"
GATE_NAMES = (
    "runtime_authority_forbidden_access",
    *engine.GATE_NAMES[1:],
)
HEX40 = re.compile(r"[0-9a-f]{40}\Z", re.ASCII)
HEX64 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(repository_path(path).read_bytes()).hexdigest()


def canonical_bytes(payload: Any) -> bytes:
    return prereg.canonical_bytes(payload)


def canonical_hash(payload: Any) -> str:
    return prereg.canonical_hash(payload)


def _git_output(*args: str) -> str:
    return prereg._git_output(*args)


def _assert_tracked_clean(path: str) -> str:
    return prereg._assert_committed(path)


def _worktree_clean() -> bool:
    return not _git_output("status", "--porcelain", "--untracked-files=all")


def _git_blob_sha256(commit: str, path: str) -> str:
    return prereg._git_blob_sha256(commit, path)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sealed_source_contracts() -> dict[str, Any]:
    payload = json.loads(
        prereg.repository_path(PREREGISTRATION_PATH).read_text()
    )
    return payload["scientific_mechanism"]["d1_scientific_contract"][
        "sources"
    ]


def validate_frozen_authority() -> dict[str, Any]:
    runtime = prereg.validate_runtime_authority()
    dependencies = prereg.validate_frozen_dependencies()
    fixed = (
        (CONTRACT_PATH, CONTRACT_COMMIT, CONTRACT_SHA256),
        (
            prereg.BOUNDARY_DOCUMENT,
            prereg.BOUNDARY_COMMIT,
            prereg.BOUNDARY_DOCUMENT_SHA256,
        ),
        (
            prereg.PRODUCER_SCRIPT,
            PREREGISTRATION_PRODUCER_COMMIT,
            PREREGISTRATION_PRODUCER_SHA256,
        ),
        (
            PREREGISTRATION_PATH,
            PREREGISTRATION_COMMIT,
            PREREGISTRATION_SHA256,
        ),
        (prereg.D1_ENGINE, prereg.D1_ENGINE_COMMIT, prereg.D1_ENGINE_SHA256),
    )
    for path, commit, digest in fixed:
        if (
            _assert_tracked_clean(path) != commit
            or sha256_file(path) != digest
            or _git_blob_sha256(commit, path) != digest
        ):
            raise RuntimeError(f"CEFS-D2 frozen authority mismatch: {path}")
    payload = json.loads(repository_path(PREREGISTRATION_PATH).read_text())
    prereg.validate_manifest(payload)
    prereg._validate_sealed_producer(payload)
    if (
        payload["manifest_hash"] != PREREGISTRATION_MANIFEST_HASH
        or payload["authority"]["producer"]
        != {
            "path": prereg.PRODUCER_SCRIPT,
            "commit": PREREGISTRATION_PRODUCER_COMMIT,
            "sha256": PREREGISTRATION_PRODUCER_SHA256,
        }
        or payload["terminal_actions"] != prereg.TERMINAL_ACTIONS
        or payload["source_values_opened"] is not False
        or payload["outcomes_opened"] is not False
    ):
        raise RuntimeError("CEFS-D2 preregistration binding mismatch")
    return {
        "runtime": runtime,
        "boundary": {
            "path": prereg.BOUNDARY_DOCUMENT,
            "commit": prereg.BOUNDARY_COMMIT,
            "sha256": prereg.BOUNDARY_DOCUMENT_SHA256,
        },
        "contract": {
            "path": CONTRACT_PATH,
            "commit": CONTRACT_COMMIT,
            "sha256": CONTRACT_SHA256,
        },
        "preregistration": {
            "path": PREREGISTRATION_PATH,
            "commit": PREREGISTRATION_COMMIT,
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        },
        "preregistration_producer": {
            "path": prereg.PRODUCER_SCRIPT,
            "commit": PREREGISTRATION_PRODUCER_COMMIT,
            "sha256": PREREGISTRATION_PRODUCER_SHA256,
        },
        "d1_preregistration": dependencies["d1_preregistration"],
        "d1_preregistration_producer": dependencies["d1_producer"],
        "d1_engine": dependencies["d1_engine"],
        "d1_terminal": dependencies["d1_rejection"],
    }


def build_execution_seal() -> dict[str, Any]:
    runner_commit = _assert_tracked_clean(RUNNER_PATH)
    test_commit = _assert_tracked_clean(TEST_PATH)
    if runner_commit != test_commit:
        raise RuntimeError("CEFS-D2 runner and tests must share one commit")
    if not _worktree_clean():
        raise RuntimeError("CEFS-D2 worktree must be clean before seal")
    authority = validate_frozen_authority()
    core = {
        "protocol_version": SEAL_PROTOCOL,
        "policy_id": POLICY_ID,
        "runtime": authority["runtime"],
        "contract": authority["contract"],
        "preregistration": authority["preregistration"],
        "preregistration_producer": authority[
            "preregistration_producer"
        ],
        "d1_preregistration": authority["d1_preregistration"],
        "d1_preregistration_producer": authority[
            "d1_preregistration_producer"
        ],
        "d1_engine": authority["d1_engine"],
        "runner": {
            "path": RUNNER_PATH,
            "commit": runner_commit,
            "sha256": sha256_file(RUNNER_PATH),
        },
        "tests": {
            "path": TEST_PATH,
            "commit": test_commit,
            "sha256": sha256_file(TEST_PATH),
        },
        "source_values_opened": False,
        "outcomes_opened": False,
    }
    payload = dict(core)
    payload["manifest_hash"] = canonical_hash(core)
    return payload


def create_execution_seal() -> dict[str, Any]:
    payload = build_execution_seal()
    engine.write_once_json(EXECUTION_SEAL_PATH, payload)
    return payload


def validate_execution_seal() -> dict[str, Any]:
    path = repository_path(EXECUTION_SEAL_PATH)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("CEFS-D2 execution seal is missing")
    payload = json.loads(path.read_text())
    if payload != build_execution_seal():
        raise RuntimeError("CEFS-D2 execution seal differs from frozen code")
    return payload


def _gate_record(
    index: int,
    name: str,
    checks: Mapping[str, bool],
) -> dict[str, Any]:
    return {
        "index": index,
        "name": name,
        "checks": dict(checks),
        "passed": bool(checks) and all(checks.values()),
    }


def _authority_report(
    seal: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **dict(authority),
        "execution_seal": {
            "path": EXECUTION_SEAL_PATH,
            "manifest_hash": seal["manifest_hash"],
            "runner": seal["runner"],
            "tests": seal["tests"],
            "d1_engine": seal["d1_engine"],
        },
        "sources": _sealed_source_contracts(),
    }


def _result_report(
    *,
    decision: str,
    failure_action: str | None,
    pass_action: str | None,
    authority: Mapping[str, Any],
    gates: Sequence[Mapping[str, Any]],
    details: Mapping[str, Any],
    counters: Mapping[str, int],
    source_hash: str | None,
    control_hash: str | None,
    source_output: Mapping[str, Any] | None,
    control_output: Mapping[str, Any] | None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    core = {
        "protocol_version": RESULT_PROTOCOL,
        "policy_id": POLICY_ID,
        "decision": decision,
        "failure_action": failure_action,
        "pass_action": pass_action,
        "authority": dict(authority),
        "gates": list(gates),
        "details": dict(details),
        "forbidden_counters": dict(counters),
        "source_row_hash": source_hash,
        "control_row_hash": control_hash,
        "source_output": source_output,
        "control_output": control_output,
        "error": (
            None
            if error is None
            else {"type": type(error).__name__, "message": str(error)}
        ),
    }
    payload = dict(core)
    payload["result_hash"] = canonical_hash(core)
    return payload


def _validate_terminal_execution_seal() -> dict[str, Any]:
    path = repository_path(EXECUTION_SEAL_PATH)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("CEFS-D2 terminal execution seal is not regular")
    _assert_tracked_clean(EXECUTION_SEAL_PATH)
    payload = json.loads(path.read_text())
    expected_keys = {
        "protocol_version",
        "policy_id",
        "runtime",
        "contract",
        "preregistration",
        "preregistration_producer",
        "d1_preregistration",
        "d1_preregistration_producer",
        "d1_engine",
        "runner",
        "tests",
        "source_values_opened",
        "outcomes_opened",
        "manifest_hash",
    }
    if set(payload) != expected_keys:
        raise RuntimeError("CEFS-D2 terminal seal fields mismatch")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload["manifest_hash"] != canonical_hash(core):
        raise RuntimeError("CEFS-D2 terminal seal hash mismatch")
    if (
        payload["protocol_version"] != SEAL_PROTOCOL
        or payload["policy_id"] != POLICY_ID
        or payload["contract"]
        != {
            "path": CONTRACT_PATH,
            "commit": CONTRACT_COMMIT,
            "sha256": CONTRACT_SHA256,
        }
        or payload["preregistration"]
        != {
            "path": PREREGISTRATION_PATH,
            "commit": PREREGISTRATION_COMMIT,
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        }
        or payload["preregistration_producer"]
        != {
            "path": prereg.PRODUCER_SCRIPT,
            "commit": PREREGISTRATION_PRODUCER_COMMIT,
            "sha256": PREREGISTRATION_PRODUCER_SHA256,
        }
        or payload["d1_preregistration"]
        != {
            "path": prereg.D1_PREREGISTRATION,
            "commit": prereg.D1_PREREGISTRATION_COMMIT,
            "sha256": prereg.D1_PREREGISTRATION_SHA256,
            "manifest_hash": prereg.D1_PREREGISTRATION_MANIFEST_HASH,
        }
        or payload["d1_preregistration_producer"]
        != {
            "path": prereg.D1_PRODUCER,
            "commit": prereg.D1_PRODUCER_COMMIT,
            "sha256": prereg.D1_PRODUCER_SHA256,
        }
        or payload["d1_engine"]
        != {
            "path": prereg.D1_ENGINE,
            "commit": prereg.D1_ENGINE_COMMIT,
            "sha256": prereg.D1_ENGINE_SHA256,
        }
        or not isinstance(payload["runtime"], dict)
        or set(payload["runtime"])
        != {"path", "path_component", "sha256", "version"}
        or payload["runtime"].get("path") != prereg.GIT_EXECUTABLE
        or payload["runtime"].get("sha256")
        != prereg.GIT_EXECUTABLE_SHA256
        or payload["runtime"].get("path_component") != "/usr/bin"
        or not isinstance(payload["runtime"].get("version"), str)
        or not payload["runtime"]["version"].startswith("git version ")
        or _git_output("--version") != payload["runtime"]["version"]
        or payload["source_values_opened"] is not False
        or payload["outcomes_opened"] is not False
        or Path(prereg.GIT_EXECUTABLE).is_symlink()
        or not Path(prereg.GIT_EXECUTABLE).is_file()
        or sha256_file(prereg.GIT_EXECUTABLE)
        != prereg.GIT_EXECUTABLE_SHA256
    ):
        raise RuntimeError("CEFS-D2 terminal seal authority mismatch")
    frozen = (
        (CONTRACT_PATH, CONTRACT_COMMIT, CONTRACT_SHA256),
        (
            prereg.BOUNDARY_DOCUMENT,
            prereg.BOUNDARY_COMMIT,
            prereg.BOUNDARY_DOCUMENT_SHA256,
        ),
        (
            PREREGISTRATION_PATH,
            PREREGISTRATION_COMMIT,
            PREREGISTRATION_SHA256,
        ),
        (
            prereg.PRODUCER_SCRIPT,
            PREREGISTRATION_PRODUCER_COMMIT,
            PREREGISTRATION_PRODUCER_SHA256,
        ),
        (
            prereg.D1_PREREGISTRATION,
            prereg.D1_PREREGISTRATION_COMMIT,
            prereg.D1_PREREGISTRATION_SHA256,
        ),
        (
            prereg.D1_PRODUCER,
            prereg.D1_PRODUCER_COMMIT,
            prereg.D1_PRODUCER_SHA256,
        ),
        (prereg.D1_ENGINE, prereg.D1_ENGINE_COMMIT, prereg.D1_ENGINE_SHA256),
        (
            prereg.D1_REJECTION,
            prereg.D1_REJECTION_COMMIT,
            prereg.D1_REJECTION_SHA256,
        ),
    )
    for frozen_path, commit, digest in frozen:
        if (
            _assert_tracked_clean(frozen_path) != commit
            or sha256_file(frozen_path) != digest
            or _git_blob_sha256(commit, frozen_path) != digest
        ):
            raise RuntimeError("CEFS-D2 terminal frozen file mismatch")
    for binding, expected_path in (
        (payload["runner"], RUNNER_PATH),
        (payload["tests"], TEST_PATH),
    ):
        if (
            not isinstance(binding, dict)
            or set(binding) != {"path", "commit", "sha256"}
            or binding.get("path") != expected_path
            or not isinstance(binding.get("commit"), str)
            or not HEX40.fullmatch(binding["commit"])
            or not isinstance(binding.get("sha256"), str)
            or not HEX64.fullmatch(binding["sha256"])
            or _git_blob_sha256(binding["commit"], expected_path)
            != binding["sha256"]
            or _assert_tracked_clean(expected_path) != binding["commit"]
            or sha256_file(expected_path) != binding["sha256"]
        ):
            raise RuntimeError("CEFS-D2 terminal runner/test seal mismatch")
    if payload["runner"]["commit"] != payload["tests"]["commit"]:
        raise RuntimeError("CEFS-D2 terminal runner/tests commit mismatch")
    return payload


def _d1_terminal_binding() -> dict[str, Any]:
    return {
        "path": prereg.D1_REJECTION,
        "commit": prereg.D1_REJECTION_COMMIT,
        "sha256": prereg.D1_REJECTION_SHA256,
        "result_hash": prereg.D1_REJECTION_RESULT_HASH,
        "source_values_opened": False,
        "outcomes_opened": False,
    }


def _expected_terminal_authority(
    seal: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "runtime": seal["runtime"],
        "boundary": {
            "path": prereg.BOUNDARY_DOCUMENT,
            "commit": prereg.BOUNDARY_COMMIT,
            "sha256": prereg.BOUNDARY_DOCUMENT_SHA256,
        },
        "contract": {
            "path": CONTRACT_PATH,
            "commit": CONTRACT_COMMIT,
            "sha256": CONTRACT_SHA256,
        },
        "preregistration": {
            "path": PREREGISTRATION_PATH,
            "commit": PREREGISTRATION_COMMIT,
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        },
        "preregistration_producer": {
            "path": prereg.PRODUCER_SCRIPT,
            "commit": PREREGISTRATION_PRODUCER_COMMIT,
            "sha256": PREREGISTRATION_PRODUCER_SHA256,
        },
        "d1_preregistration": {
            "path": prereg.D1_PREREGISTRATION,
            "commit": prereg.D1_PREREGISTRATION_COMMIT,
            "sha256": prereg.D1_PREREGISTRATION_SHA256,
            "manifest_hash": prereg.D1_PREREGISTRATION_MANIFEST_HASH,
        },
        "d1_preregistration_producer": {
            "path": prereg.D1_PRODUCER,
            "commit": prereg.D1_PRODUCER_COMMIT,
            "sha256": prereg.D1_PRODUCER_SHA256,
        },
        "d1_engine": {
            "path": prereg.D1_ENGINE,
            "commit": prereg.D1_ENGINE_COMMIT,
            "sha256": prereg.D1_ENGINE_SHA256,
        },
        "d1_terminal": _d1_terminal_binding(),
        "execution_seal": {
            "path": EXECUTION_SEAL_PATH,
            "manifest_hash": seal["manifest_hash"],
            "runner": seal["runner"],
            "tests": seal["tests"],
            "d1_engine": seal["d1_engine"],
        },
        "sources": _sealed_source_contracts(),
    }


def _validate_terminal_gates(gates: Any, *, decision: str) -> int:
    if (
        not isinstance(gates, list)
        or not gates
        or len(gates) > len(GATE_NAMES)
        or (decision == "pass" and len(gates) != len(GATE_NAMES))
    ):
        raise RuntimeError("CEFS-D2 terminal gate count mismatch")
    for index, (gate, name) in enumerate(zip(gates, GATE_NAMES), start=1):
        if (
            not isinstance(gate, dict)
            or set(gate) != {"index", "name", "checks", "passed"}
            or type(gate.get("index")) is not int
            or gate["index"] != index
            or gate.get("name") != name
            or not isinstance(gate.get("checks"), dict)
            or not gate["checks"]
            or any(type(value) is not bool for value in gate["checks"].values())
            or type(gate.get("passed")) is not bool
            or gate["passed"] != all(gate["checks"].values())
        ):
            raise RuntimeError("CEFS-D2 terminal gate structure mismatch")
    last = len(gates)
    if decision == "pass":
        if not all(gate["passed"] for gate in gates):
            raise RuntimeError("CEFS-D2 terminal pass has failed gate")
    elif any(not gate["passed"] for gate in gates[:-1]) or gates[-1]["passed"]:
        raise RuntimeError("CEFS-D2 terminal rejection violates first stop")
    gate_one = gates[0]
    if last == 1 and decision == "fail":
        if gate_one["checks"] != {"runtime_or_authority_valid": False}:
            raise RuntimeError("CEFS-D2 terminal Gate 1 rejection mismatch")
    else:
        expected = {
            "runtime_authority_valid",
            "absolute_git_valid",
            "authority_valid",
            "seal_valid",
            "worktree_clean",
            *engine.FORBIDDEN_COUNTER_NAMES,
        }
        if (
            set(gate_one["checks"]) != expected
            or not all(gate_one["checks"].values())
        ):
            raise RuntimeError("CEFS-D2 terminal Gate 1 pass mismatch")
    return last


def _validate_terminal_report(path: Path, *, decision: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("CEFS-D2 terminal report is not regular")
    payload = json.loads(path.read_text())
    expected_fields = {
        "protocol_version",
        "policy_id",
        "decision",
        "failure_action",
        "pass_action",
        "authority",
        "gates",
        "details",
        "forbidden_counters",
        "source_row_hash",
        "control_row_hash",
        "source_output",
        "control_output",
        "error",
        "result_hash",
    }
    if set(payload) != expected_fields:
        raise RuntimeError("CEFS-D2 terminal report fields mismatch")
    if (
        payload["protocol_version"] != RESULT_PROTOCOL
        or payload["policy_id"] != POLICY_ID
        or payload["decision"] != decision
    ):
        raise RuntimeError("CEFS-D2 terminal identity mismatch")
    core = {key: value for key, value in payload.items() if key != "result_hash"}
    if payload["result_hash"] != canonical_hash(core):
        raise RuntimeError("CEFS-D2 terminal result hash mismatch")
    last_gate = _validate_terminal_gates(payload["gates"], decision=decision)
    engine._validate_terminal_details(
        payload["details"],
        decision=decision,
        last_gate=last_gate,
    )
    for gate in payload["gates"][1:]:
        if gate["checks"] != engine._terminal_expected_gate_checks(
            gate["index"],
            payload["details"],
        ):
            raise RuntimeError("CEFS-D2 terminal gate checks do not replay")
    counters = payload["forbidden_counters"]
    if (
        not isinstance(counters, dict)
        or set(counters) != set(engine.FORBIDDEN_COUNTER_NAMES)
        or any(
            type(counters[name]) is not int or counters[name] != 0
            for name in engine.FORBIDDEN_COUNTER_NAMES
        )
    ):
        raise RuntimeError("CEFS-D2 terminal forbidden counters mismatch")
    for key in ("source_row_hash", "control_row_hash"):
        value = payload[key]
        if value is not None and (
            not isinstance(value, str) or not HEX64.fullmatch(value)
        ):
            raise RuntimeError("CEFS-D2 terminal row hash grammar mismatch")
    if decision == "pass":
        if (
            payload["failure_action"] is not None
            or payload["pass_action"] != PASS_ACTION
            or payload["error"] is not None
            or payload["source_row_hash"] is None
            or payload["control_row_hash"] is None
        ):
            raise RuntimeError("CEFS-D2 terminal pass action mismatch")
        seal = _validate_terminal_execution_seal()
        if payload["authority"] != _expected_terminal_authority(seal):
            raise RuntimeError("CEFS-D2 terminal pass authority mismatch")
        output_specs = {
            "source_output": (SOURCE_OUTPUT, engine.SOURCE_OUTPUT_COLUMNS),
            "control_output": (CONTROL_OUTPUT, engine.CONTROL_OUTPUT_COLUMNS),
        }
        decoded: dict[str, list[dict[str, str]]] = {}
        for key, (output_path, columns) in output_specs.items():
            binding = payload[key]
            if (
                not isinstance(binding, dict)
                or set(binding) != {"path", "sha256", "rows"}
                or binding.get("path") != output_path
                or not isinstance(binding.get("sha256"), str)
                or not HEX64.fullmatch(binding["sha256"])
                or type(binding.get("rows")) is not int
                or binding["rows"] <= 0
            ):
                raise RuntimeError("CEFS-D2 terminal output binding mismatch")
            final = repository_path(output_path)
            if final.is_symlink() or not final.is_file():
                raise RuntimeError("CEFS-D2 terminal output is not regular")
            if sha256_file(output_path) != binding["sha256"]:
                raise RuntimeError("CEFS-D2 terminal output hash mismatch")
            decoded[key] = engine._terminal_output_records(
                output_path,
                columns,
            )
            if len(decoded[key]) != binding["rows"]:
                raise RuntimeError("CEFS-D2 terminal output count mismatch")
        if (
            payload["source_output"]["rows"]
            != payload["details"]["schedule"]["sequence_ready"]
            or payload["control_output"]["rows"]
            != payload["details"]["controls"]["control_rows"]
        ):
            raise RuntimeError("CEFS-D2 terminal metric row count mismatch")
        if (
            hashlib.sha256(
                engine._canonical_records(decoded["source_output"])
            ).hexdigest()
            != payload["source_row_hash"]
            or hashlib.sha256(
                engine._canonical_records(decoded["control_output"])
            ).hexdigest()
            != payload["control_row_hash"]
        ):
            raise RuntimeError("CEFS-D2 terminal canonical hash mismatch")
        engine._validate_terminal_output_semantics(
            decoded["source_output"],
            decoded["control_output"],
            payload["details"],
        )
    else:
        if (
            payload["failure_action"] != FAILURE_ACTION
            or payload["pass_action"] is not None
            or payload["source_output"] is not None
            or payload["control_output"] is not None
        ):
            raise RuntimeError("CEFS-D2 terminal rejection binding mismatch")
        error = payload["error"]
        if error is not None and (
            not isinstance(error, dict)
            or set(error) != {"type", "message"}
            or not all(isinstance(value, str) for value in error.values())
        ):
            raise RuntimeError("CEFS-D2 terminal rejection error mismatch")
        if (
            (last_gate >= 3 and payload["source_row_hash"] is None)
            or (last_gate < 3 and payload["source_row_hash"] is not None)
            or (last_gate >= 6 and payload["control_row_hash"] is None)
            or (last_gate < 6 and payload["control_row_hash"] is not None)
        ):
            raise RuntimeError("CEFS-D2 rejection row hash stage mismatch")
        if last_gate == 1:
            if payload["authority"] != {} or error is None:
                raise RuntimeError("CEFS-D2 Gate 1 rejection authority mismatch")
        else:
            seal = _validate_terminal_execution_seal()
            if payload["authority"] != _expected_terminal_authority(seal):
                raise RuntimeError("CEFS-D2 rejection authority mismatch")
    return payload


def _path_present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def pre_run_terminal_state() -> dict[str, Any] | None:
    pass_path = repository_path(PASS_REPORT)
    rejection_path = repository_path(REJECTION_REPORT)
    source_path = repository_path(SOURCE_OUTPUT)
    control_path = repository_path(CONTROL_OUTPUT)
    pass_present = _path_present(pass_path)
    rejection_present = _path_present(rejection_path)
    source_present = _path_present(source_path)
    control_present = _path_present(control_path)
    if pass_present and rejection_present:
        raise RuntimeError("CEFS-D2 conflicting terminal reports")
    if pass_present:
        return _validate_terminal_report(pass_path, decision="pass")
    if rejection_present:
        if source_present or control_present:
            raise RuntimeError("CEFS-D2 rejection conflicts with pass outputs")
        return _validate_terminal_report(rejection_path, decision="fail")
    if source_present or control_present:
        raise RuntimeError("CEFS-D2 partial terminal pass state")
    return None


def run_official() -> dict[str, Any]:
    terminal = pre_run_terminal_state()
    if terminal is not None:
        return terminal
    gates: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    authority_report: dict[str, Any] = {}
    counters = engine.forbidden_counters()
    schedules: list[engine.ScheduleRow] = []
    controls: list[engine.ControlRow] = []

    def fail(error: BaseException | None = None) -> dict[str, Any]:
        source_records = [engine.schedule_record(row) for row in schedules]
        control_records = [engine.control_record(row) for row in controls]
        report = _result_report(
            decision="fail",
            failure_action=FAILURE_ACTION,
            pass_action=None,
            authority=authority_report,
            gates=gates,
            details=details,
            counters=counters,
            source_hash=(
                hashlib.sha256(
                    engine._canonical_records(source_records)
                ).hexdigest()
                if source_records
                else None
            ),
            control_hash=(
                hashlib.sha256(
                    engine._canonical_records(control_records)
                ).hexdigest()
                if control_records
                else None
            ),
            source_output=None,
            control_output=None,
            error=error,
        )
        engine.write_once_json(REJECTION_REPORT, report)
        return report

    try:
        runtime = prereg.validate_runtime_authority()
        if prereg._git_output("--version") != runtime["version"]:
            raise RuntimeError("CEFS-D2 absolute Git version drift")
        seal = validate_execution_seal()
        authority = validate_frozen_authority()
        if not _worktree_clean():
            raise RuntimeError("CEFS-D2 worktree is not clean")
        authority_report = _authority_report(seal, authority)
        gate = _gate_record(
            1,
            GATE_NAMES[0],
            {
                "runtime_authority_valid": True,
                "absolute_git_valid": True,
                "authority_valid": True,
                "seal_valid": True,
                "worktree_clean": True,
                **{name: value == 0 for name, value in counters.items()},
            },
        )
        gates.append(gate)
    except Exception as error:
        gates.append(
            _gate_record(
                1,
                GATE_NAMES[0],
                {"runtime_or_authority_valid": False},
            )
        )
        return fail(error)

    try:
        inputs = engine.load_source_inputs()
        common = engine.join_common_rows(inputs)
        parser = engine.parser_metrics(inputs, common)
        details["parser"] = parser
        gate = _gate_record(2, GATE_NAMES[1], engine.parser_checks(parser))
        gates.append(gate)
        if not gate["passed"]:
            return fail()
    except Exception as error:
        gates.append(
            _gate_record(
                2,
                GATE_NAMES[1],
                {"source_parse_completed": False},
            )
        )
        return fail(error)

    schedules = engine.build_schedules(common)
    schedule = engine.schedule_metrics(schedules)
    schedule["schedule_replay"] = engine.schedule_replay_metrics(
        inputs,
        common,
        schedules,
    )
    details["schedule"] = schedule
    gate = _gate_record(3, GATE_NAMES[2], engine.schedule_checks(schedule))
    gates.append(gate)
    if not gate["passed"]:
        return fail()

    edges = engine.edge_metrics(schedules)
    details["edge_support"] = edges
    gate = _gate_record(4, GATE_NAMES[3], engine.edge_checks(edges))
    gates.append(gate)
    if not gate["passed"]:
        return fail()

    diversity = engine.diversity_metrics(schedules)
    details["diversity_stability"] = diversity
    gate = _gate_record(
        5,
        GATE_NAMES[4],
        engine.diversity_checks(diversity),
    )
    gates.append(gate)
    if not gate["passed"]:
        return fail()

    controls = engine.build_controls(schedules)
    control_detail = engine.control_metrics(schedules, controls)
    details["controls"] = control_detail
    gate = _gate_record(
        6,
        GATE_NAMES[5],
        engine.control_checks(control_detail),
    )
    gates.append(gate)
    if not gate["passed"]:
        return fail()

    replay = engine.append_replay_metrics(
        inputs,
        common,
        schedules,
        controls,
    )
    details["determinism_append_replay"] = replay
    gate = _gate_record(
        7,
        GATE_NAMES[6],
        engine._terminal_expected_gate_checks(7, details),
    )
    gates.append(gate)
    if not gate["passed"]:
        return fail()

    source_records = [engine.schedule_record(row) for row in schedules]
    control_records = [engine.control_record(row) for row in controls]
    source_row_bytes = engine._canonical_records(source_records)
    control_row_bytes = engine._canonical_records(control_records)
    source_gzip = engine.deterministic_csv_gzip(
        source_records,
        engine.SOURCE_OUTPUT_COLUMNS,
    )
    control_gzip = engine.deterministic_csv_gzip(
        control_records,
        engine.CONTROL_OUTPUT_COLUMNS,
    )
    source_sha = hashlib.sha256(source_gzip).hexdigest()
    control_sha = hashlib.sha256(control_gzip).hexdigest()
    report = _result_report(
        decision="pass",
        failure_action=None,
        pass_action=PASS_ACTION,
        authority=authority_report,
        gates=gates,
        details=details,
        counters=counters,
        source_hash=hashlib.sha256(source_row_bytes).hexdigest(),
        control_hash=hashlib.sha256(control_row_bytes).hexdigest(),
        source_output={
            "path": SOURCE_OUTPUT,
            "sha256": source_sha,
            "rows": len(source_records),
        },
        control_output={
            "path": CONTROL_OUTPUT,
            "sha256": control_sha,
            "rows": len(control_records),
        },
    )
    engine.publish_write_once_transaction(
        (
            (SOURCE_OUTPUT, source_gzip),
            (CONTROL_OUTPUT, control_gzip),
            (PASS_REPORT, _json_bytes(report)),
        )
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("create-seal")
    subparsers.add_parser("validate-seal")
    subparsers.add_parser("run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "create-seal":
        payload = create_execution_seal()
        print(
            json.dumps(
                {
                    "path": EXECUTION_SEAL_PATH,
                    "manifest_hash": payload["manifest_hash"],
                    "source_values_opened": False,
                    "outcomes_opened": False,
                },
                indent=2,
            )
        )
        return
    if args.command == "validate-seal":
        payload = validate_execution_seal()
        print(
            json.dumps(
                {
                    "path": EXECUTION_SEAL_PATH,
                    "manifest_hash": payload["manifest_hash"],
                    "valid": True,
                },
                indent=2,
            )
        )
        return
    result = run_official()
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "failure_action": result["failure_action"],
                "pass_action": result["pass_action"],
                "gates": result["gates"],
                "result_hash": result["result_hash"],
            },
            indent=2,
        )
    )
    raise SystemExit(0 if result["decision"] == "pass" else 2)


if __name__ == "__main__":
    main()
