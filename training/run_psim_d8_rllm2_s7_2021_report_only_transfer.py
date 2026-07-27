#!/usr/bin/env python3
"""Execute the single sealed PSIM S7 report-only 2021 transfer.

The attempt ledger is published before any 2021 market or funding payload is
opened, read, hashed, or parsed.  The runner evaluates the complete frozen
41-policy family exactly once and never selects, repairs, or refits from the
2021 outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

_BOOTSTRAP_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_REPO_ROOT))

from training import bctp_stage_sources as stage_sources
from training import bctp_strict_economics as strict_economics
from training import (
    preregister_psim_d8_rllm2_s7_2021_report_only_transfer as prereg,
)
from training import psim_report_only_transfer as transfer

PROTOCOL_VERSION = "psim_d8_rllm2_s7_2021_report_only_transfer_v1"
ATTEMPT_PROTOCOL_VERSION = (
    "psim_d8_rllm2_s7_2021_report_only_transfer_attempt_v1"
)
PREREGISTRATION_SHA256 = (
    "daec6511fa318383d0f4988d32d8a91a78d552e48e2581268cb122718337a84e"
)
PREREGISTRATION_MANIFEST_HASH = (
    "2d7683f6f3e0d3c158fb5609924cfb2010756be81ad90f54560dde90fb4655fa"
)
CORE_PATH = Path("training/psim_report_only_transfer.py")
EXPECTED_METRIC_SET_COUNT = len(prereg.FAMILY_IDS) + 4


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else prereg.REPO_ROOT / candidate


def _canonical_bytes(
    payload: Any,
    *,
    pretty: bool = False,
) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8") + (b"\n" if pretty else b"")


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=prereg.REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def assert_committed_clean_runner() -> dict[str, str]:
    head = _git("rev-parse", "HEAD")
    origin_main = _git("rev-parse", "origin/main")
    status = _git("status", "--porcelain", "--untracked-files=no")
    if head != origin_main or status:
        raise RuntimeError(
            "PSIM S7 runner requires clean HEAD equal to origin/main"
        )
    bindings = {"execution_commit": head}
    for name, path in (
        ("runner_sha256", prereg.RUNNER_PATH),
        ("evaluator_core_sha256", CORE_PATH),
    ):
        local_sha = _sha256_file(path)
        committed = subprocess.run(
            ["git", "show", f"{head}:{path.as_posix()}"],
            cwd=prereg.REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(committed).hexdigest() != local_sha:
            raise RuntimeError(f"PSIM S7 committed source changed: {path}")
        bindings[name] = local_sha
    return bindings


def validate_preregistration() -> dict[str, Any]:
    target = repository_path(prereg.DEFAULT_OUTPUT)
    if (
        target.is_symlink()
        or not target.is_file()
        or _sha256_file(target) != PREREGISTRATION_SHA256
    ):
        raise RuntimeError("PSIM S7 preregistration file changed")
    payload = json.loads(target.read_text(encoding="utf-8"))
    core = {
        key: value
        for key, value in payload.items()
        if key != "manifest_hash"
    }
    boundary = payload.get("access_boundary", {})
    execution = payload.get("execution_contract", {})
    if (
        payload.get("manifest_hash") != _canonical_hash(core)
        or payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH
        or payload != prereg.build_preregistration()
        or payload.get("candidate", {}).get("id") != prereg.STAGE_ID
        or payload.get("candidate", {}).get("primary_policy_id")
        != prereg.PRIMARY_POLICY_ID
        or payload.get("frozen_schedule_family", {}).get("family_ids")
        != list(prereg.FAMILY_IDS)
        or payload.get("frozen_schedule_family", {}).get("family_count")
        != len(prereg.FAMILY_IDS)
        or execution.get(
            "attempt_written_before_market_or_funding_open_or_read"
        )
        is not True
        or execution.get(
            "attempt_written_before_market_or_funding_payload_hash_or_parse"
        )
        is not True
        or execution.get("no_2022_or_later_outcome_access") is not True
        or execution.get("no_model_load_or_forward") is not True
        or boundary.get(
            "raw_market_or_funding_paths_opened_or_read_before_attempt"
        )
        != []
        or boundary.get("market_or_funding_payload_bytes_hashed") is not False
        or boundary.get("market_rows_parsed") != 0
        or boundary.get("funding_rows_parsed") != 0
        or boundary.get("economic_metric_sets_computed") != 0
    ):
        raise RuntimeError("PSIM S7 preregistration contract changed")
    return payload


def _write_once(path: str | Path, raw: bytes) -> None:
    target = repository_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if (
            target.is_symlink()
            or not target.is_file()
            or target.read_bytes() != raw
        ):
            raise RuntimeError(f"PSIM S7 write-once artifact drift: {path}")
        return
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=target.parent,
        prefix=f".{target.name}.",
        delete=False,
    ) as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        try:
            os.link(temporary, target)
        except FileExistsError:
            if (
                target.is_symlink()
                or not target.is_file()
                or target.read_bytes() != raw
            ):
                raise RuntimeError(
                    f"PSIM S7 concurrent write-once drift: {path}"
                ) from None
        else:
            directory = os.open(
                target.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_once(
    path: str | Path,
    payload: Mapping[str, Any],
) -> None:
    _write_once(path, _canonical_bytes(dict(payload), pretty=True))


def _attempt_payload(bindings: Mapping[str, str]) -> dict[str, Any]:
    core = {
        "protocol_version": ATTEMPT_PROTOCOL_VERSION,
        "stage_id": prereg.STAGE_ID,
        "execution_commit": bindings["execution_commit"],
        "runner_sha256": bindings["runner_sha256"],
        "evaluator_core_sha256": bindings["evaluator_core_sha256"],
        "preregistration": {
            "path": prereg.DEFAULT_OUTPUT.as_posix(),
            "sha256": PREREGISTRATION_SHA256,
            "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        },
        "authorization": {
            "open_exact_2021_market_and_funding_stage": True,
            "evaluate_fixed_41_policy_family": True,
            "compute_fixed_45_economic_metric_sets": True,
            "select_or_repair_from_2021": False,
            "open_2022_or_later_outcomes": False,
            "load_or_forward_model": False,
        },
        "access_boundary_at_attempt": {
            "raw_market_or_funding_paths_opened_or_read": [],
            "market_or_funding_payload_bytes_hashed": False,
            "market_rows_parsed": 0,
            "funding_rows_parsed": 0,
            "economic_metric_sets_computed": 0,
            "2021_policy_specific_outcomes_opened": False,
            "2022_or_later_outcomes_opened": False,
            "model_loaded": False,
            "model_forwards_started": 0,
        },
    }
    return {**core, "attempt_hash": _canonical_hash(core)}


def _write_or_validate_attempt(
    bindings: Mapping[str, str],
) -> dict[str, Any]:
    payload = _attempt_payload(bindings)
    _write_json_once(prereg.ATTEMPT_PATH, payload)
    return payload


def _verify_post_attempt_inputs(
    bindings: Mapping[str, str],
) -> None:
    expected = (
        (prereg.RUNNER_PATH, bindings["runner_sha256"]),
        (CORE_PATH, bindings["evaluator_core_sha256"]),
        (Path(strict_economics.__file__), prereg.STRICT_ECONOMICS_SHA256),
        (Path(stage_sources.__file__), prereg.STAGE_SOURCES_SHA256),
        (prereg.s4.SCHEDULE_PATH, prereg.S4_BASE_SCHEDULES_SHA256),
        (prereg.s5.SCHEDULE_PATH, prereg.S5_BASE_SCHEDULES_SHA256),
        (prereg.s6r1.SCHEDULE_PATH, prereg.S6R1_BASE_SCHEDULES_SHA256),
        (
            prereg.s6r1.DELAYED_SCHEDULE_PATH,
            prereg.S6R1_DELAYED_SCHEDULE_SHA256,
        ),
    )
    for path, expected_sha in expected:
        if _sha256_file(path) != expected_sha:
            raise RuntimeError(f"PSIM S7 frozen execution input changed: {path}")


def _load_schedule_family() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    return tuple(
        pd.read_csv(repository_path(path), compression="gzip", dtype=str)
        for path in (
            prereg.s4.SCHEDULE_PATH,
            prereg.s5.SCHEDULE_PATH,
            prereg.s6r1.SCHEDULE_PATH,
            prereg.s6r1.DELAYED_SCHEDULE_PATH,
        )
    )  # type: ignore[return-value]


def _expected_access_boundary() -> dict[str, Any]:
    return {
        "raw_market_or_funding_paths_opened_or_read_before_attempt": [],
        "raw_market_or_funding_paths_opened_or_read_after_attempt": [
            prereg.MARKET_PATH.as_posix(),
            prereg.FUNDING_PATH.as_posix(),
        ],
        "market_or_funding_payload_bytes_hashed_after_attempt": True,
        "market_rows_parsed": 105_120,
        "funding_rows_parsed": 1_095,
        "fixed_policy_schedules_parsed": len(prereg.FAMILY_IDS),
        "economic_metric_sets_computed": EXPECTED_METRIC_SET_COUNT,
        "2021_policy_specific_outcomes_opened": True,
        "selection_or_repair_from_2021": False,
        "2022_or_later_outcomes_opened": False,
        "model_loaded": False,
        "model_forwards_started": 0,
    }


def _source_bindings() -> dict[str, Any]:
    return {
        "stage": "2021",
        "manifest": {
            "path": prereg.STAGE_SOURCE_MANIFEST.as_posix(),
            "sha256": prereg.STAGE_SOURCE_MANIFEST_SHA256,
            "manifest_hash": prereg.STAGE_SOURCE_MANIFEST_HASH,
        },
        "market": {
            "path": prereg.MARKET_PATH.as_posix(),
            "gzip_sha256": prereg.MARKET_GZIP_SHA256,
            "rows": 105_120,
        },
        "funding": {
            "path": prereg.FUNDING_PATH.as_posix(),
            "gzip_sha256": prereg.FUNDING_GZIP_SHA256,
            "rows": 1_095,
        },
    }


def _verify_existing_result(
    attempt: Mapping[str, Any],
    *,
    expected_result: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    target = repository_path(prereg.RESULT_PATH)
    if not target.exists():
        return None
    if target.is_symlink() or not target.is_file():
        raise RuntimeError("PSIM S7 result path is unsafe")
    payload = json.loads(target.read_text(encoding="utf-8"))
    core = {
        key: value
        for key, value in payload.items()
        if key != "result_hash"
    }
    evaluation = payload.get("evaluation")
    gate = evaluation.get("gate") if isinstance(evaluation, dict) else None
    checks = gate.get("checks") if isinstance(gate, dict) else None
    passed = gate.get("passed") if isinstance(gate, dict) else None
    expected_decision = "pass" if passed is True else "reject"
    expected_terminal_action = (
        "AUTHORIZE_SEPARATE_FORWARD_OR_LIVE_CANDIDATE_PREREGISTRATION"
        if passed is True
        else "RETIRE_UNCHANGED_S6R1_HYPOTHESIS_NO_2021_REPAIR"
    )
    family_metrics = (
        evaluation.get("base_family_metrics")
        if isinstance(evaluation, dict)
        else None
    )
    primary_metrics = (
        evaluation.get("primary_metrics")
        if isinstance(evaluation, dict)
        else None
    )
    inference = (
        evaluation.get("familywise_max_stat")
        if isinstance(evaluation, dict)
        else None
    )
    strongest = (
        evaluation.get("strongest_nonsemantic_control")
        if isinstance(evaluation, dict)
        else None
    )
    robustness_semantics = (
        evaluation.get("robustness_semantics")
        if isinstance(evaluation, dict)
        else None
    )
    if (
        payload.get("result_hash") != _canonical_hash(core)
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("stage_id") != prereg.STAGE_ID
        or payload.get("execution_commit") != attempt["execution_commit"]
        or payload.get("runner_sha256") != attempt["runner_sha256"]
        or payload.get("evaluator_core_sha256")
        != attempt["evaluator_core_sha256"]
        or payload.get("attempt_hash") != attempt["attempt_hash"]
        or payload.get("preregistration_manifest_hash")
        != PREREGISTRATION_MANIFEST_HASH
        or type(passed) is not bool
        or not isinstance(checks, dict)
        or not checks
        or passed is not all(bool(value) for value in checks.values())
        or payload.get("decision") != expected_decision
        or payload.get("terminal_action") != expected_terminal_action
        or payload.get(
            "authorize_separate_forward_or_live_candidate_preregistration"
        )
        is not passed
        or payload.get("authorize_live_promotion") is not False
        or payload.get("authorize_2022_or_later_outcomes") is not False
        or payload.get("selection_or_repair_from_2021") is not False
        or payload.get("source_bindings") != _source_bindings()
        or payload.get("access_boundary") != _expected_access_boundary()
        or not isinstance(evaluation, dict)
        or evaluation.get("protocol_version")
        != "psim_s7_report_only_transfer_calculation_v1"
        or robustness_semantics
        != {
            "half_metrics": (
                "standalone_reset_to_flat_equity_1_at_each_half_start"
            ),
            "continuous_full_path_subperiod_attribution": False,
        }
        or evaluation.get("primary_policy_id") != prereg.PRIMARY_POLICY_ID
        or evaluation.get("family_ids") != list(prereg.FAMILY_IDS)
        or not isinstance(family_metrics, dict)
        or set(family_metrics) != set(prereg.FAMILY_IDS)
        or len(family_metrics) != len(prereg.FAMILY_IDS)
        or not isinstance(primary_metrics, dict)
        or set(primary_metrics)
        != {
            "base_6bp",
            "stress_10bp",
            "delayed_5m_6bp",
            "first_half_6bp",
            "second_half_6bp",
        }
        or not isinstance(inference, dict)
        or inference.get("family_ids") != list(prereg.FAMILY_IDS)
        or inference.get("draws") != prereg.STATISTICAL_DRAWS
        or inference.get("seed") != prereg.STATISTICAL_SEED
        or not isinstance(strongest, dict)
        or strongest.get("policy_id") not in prereg.NONSEMANTIC_CONTROL_IDS
        or expected_result is None
        or payload != dict(expected_result)
    ):
        raise RuntimeError("PSIM S7 existing result changed")
    return payload


def validate_only() -> dict[str, Any]:
    registration = validate_preregistration()
    bindings = assert_committed_clean_runner()
    return {
        "decision": "validated",
        "stage_id": prereg.STAGE_ID,
        **bindings,
        "preregistration_manifest_hash": registration["manifest_hash"],
        "existing_outputs": [
            path.as_posix()
            for path in (prereg.ATTEMPT_PATH, prereg.RESULT_PATH)
            if repository_path(path).exists()
        ],
        "raw_market_or_funding_paths_opened_or_read": [],
        "market_or_funding_payload_bytes_hashed": False,
        "market_rows_parsed": 0,
        "funding_rows_parsed": 0,
    }


def execute() -> dict[str, Any]:
    validate_preregistration()
    bindings = assert_committed_clean_runner()
    attempt = _write_or_validate_attempt(bindings)

    _verify_post_attempt_inputs(bindings)
    market, funding, diagnostics = stage_sources.load_stage_source(
        "2021",
        output_root=prereg.STAGE_SOURCE_ROOT,
    )
    if (
        _sha256_file(prereg.MARKET_PATH) != prereg.MARKET_GZIP_SHA256
        or _sha256_file(prereg.FUNDING_PATH) != prereg.FUNDING_GZIP_SHA256
        or len(market) != 105_120
        or len(funding) != 1_095
        or diagnostics.get("stage") != "2021"
        or diagnostics.get("manifest_hash")
        != prereg.STAGE_SOURCE_MANIFEST_HASH
    ):
        raise RuntimeError("PSIM S7 opened outcome source changed")
    schedules = _load_schedule_family()
    evaluation = transfer.evaluate_transfer(
        market,
        funding,
        *schedules,
    )
    gate = evaluation.get("gate", {})
    checks = gate.get("checks", {})
    passed = bool(gate.get("passed"))
    if (
        not isinstance(checks, dict)
        or not checks
        or passed is not all(bool(value) for value in checks.values())
    ):
        raise RuntimeError("PSIM S7 evaluation gate is inconsistent")

    result_core = {
        "protocol_version": PROTOCOL_VERSION,
        "stage_id": prereg.STAGE_ID,
        "decision": "pass" if passed else "reject",
        **bindings,
        "attempt_hash": attempt["attempt_hash"],
        "preregistration_manifest_hash": PREREGISTRATION_MANIFEST_HASH,
        "source_bindings": _source_bindings(),
        "stage_source_diagnostics": diagnostics,
        "evaluation": evaluation,
        "terminal_action": (
            "AUTHORIZE_SEPARATE_FORWARD_OR_LIVE_CANDIDATE_PREREGISTRATION"
            if passed
            else "RETIRE_UNCHANGED_S6R1_HYPOTHESIS_NO_2021_REPAIR"
        ),
        "authorize_separate_forward_or_live_candidate_preregistration": (
            passed
        ),
        "authorize_live_promotion": False,
        "authorize_2022_or_later_outcomes": False,
        "selection_or_repair_from_2021": False,
        "access_boundary": _expected_access_boundary(),
    }
    result = {
        **result_core,
        "result_hash": _canonical_hash(result_core),
    }
    existing = _verify_existing_result(
        attempt,
        expected_result=result,
    )
    if existing is not None:
        return existing
    _write_json_once(prereg.RESULT_PATH, result)
    verified = _verify_existing_result(
        attempt,
        expected_result=result,
    )
    if verified != result:
        raise RuntimeError("PSIM S7 published result identity changed")
    return result


def _summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("decision") == "validated":
        return dict(payload)
    evaluation = payload["evaluation"]
    primary = evaluation["primary_metrics"]["base_6bp"]
    return {
        "stage_id": payload["stage_id"],
        "decision": payload["decision"],
        "execution_commit": payload["execution_commit"],
        "attempt_hash": payload["attempt_hash"],
        "result_hash": payload["result_hash"],
        "absolute_return_pct": primary["absolute_return_pct"],
        "cagr_pct": primary["cagr_pct"],
        "strict_mdd_pct": primary["strict_mdd_pct"],
        "cagr_to_strict_mdd": primary["cagr_to_strict_mdd"],
        "directional_entries_including_flips": primary[
            "directional_entries_including_flips"
        ],
        "all_target_changes_including_terminal_flatten": primary[
            "all_target_changes_including_terminal_flatten"
        ],
        "familywise_p_max": evaluation["familywise_max_stat"]["p_max"][
            prereg.PRIMARY_POLICY_ID
        ],
        "gate_checks": evaluation["gate"]["checks"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    payload = validate_only() if args.validate_only else execute()
    print(json.dumps(_summary(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
