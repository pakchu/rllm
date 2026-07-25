"""Chronological BCTP-12H cheap-policy runner.

The command order is intentionally irreversible:

1. open only 2020, create rewards, fit the frozen family, seal 2021 schedules;
2. open 2021 only after that seal, evaluate, and refit/seal 2022 only on pass;
3. open 2022 only after the refit seal and evaluate the selected algorithm.

No 2023 or post-2023 outcome path exists in this runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Final

import numpy as np
import pandas as pd

from training import bctp_family_evaluator as family_evaluator
from training import bctp_policy_family as policy_family
from training import bctp_schedule_seal as schedule_seal
from training import bctp_stage_sources as stage_sources
from training import bctp_transition_labels as transition_labels
from training import freeze_block_clearing_target_position_evaluator as freeze


REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT: Final = Path("results/bctp_mdp_run_2026-07-25")
DEFAULT_LEDGER_ROOT: Final = Path("data/bctp_transition_ledgers")
DEFAULT_STAGE_SOURCE_ROOT: Final = stage_sources.DEFAULT_OUTPUT_ROOT
DEFAULT_SCHEDULE_ROOT: Final = schedule_seal.DEFAULT_OUTPUT_ROOT

FIT_2020_REPORT: Final = "fit_2020_and_seal_2021.json"
TRANSFER_2021_REPORT: Final = "transfer_2021.json"
REFIT_2020_2021_REPORT: Final = "refit_2020_2021_and_seal_2022.json"
TRANSFER_2022_REPORT: Final = "transfer_2022.json"

RUNNER_FILES: Final = (
    "training/bctp_strict_economics.py",
    "training/bctp_cheap_policies.py",
    "training/bctp_stage_sources.py",
    "training/bctp_transition_labels.py",
    "training/bctp_policy_family.py",
    "training/bctp_schedule_seal.py",
    "training/bctp_family_evaluator.py",
    "training/run_block_clearing_target_position_mdp.py",
    "tests/test_bctp_strict_economics.py",
    "tests/test_bctp_cheap_policies.py",
    "tests/test_bctp_stage_sources.py",
    "tests/test_bctp_transition_labels.py",
    "tests/test_bctp_policy_family.py",
    "tests/test_bctp_schedule_seal.py",
    "tests/test_bctp_family_evaluator.py",
    "tests/test_run_block_clearing_target_position_mdp.py",
)


def _path(path: str | Path) -> Path:
    candidate = Path(path)
    return (
        candidate
        if candidate.is_absolute()
        else REPOSITORY_ROOT / candidate
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def assert_runner_committed() -> str:
    tracked = _git(
        "ls-files",
        "--error-unmatch",
        "--",
        *RUNNER_FILES,
    )
    if tracked.returncode:
        raise RuntimeError("BCTP runner files are not fully committed")
    clean = _git("diff", "--quiet", "HEAD", "--", *RUNNER_FILES)
    if clean.returncode:
        raise RuntimeError("BCTP runner differs from committed HEAD")
    head = _git("log", "-1", "--format=%H", "--", *RUNNER_FILES)
    commit = head.stdout.strip()
    if (
        head.returncode
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise RuntimeError("BCTP runner commit cannot be identified")
    return commit


def _report_path(run_root: str | Path, filename: str) -> Path:
    return _path(run_root) / filename


def _manifest(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = freeze.json_normalized(payload)
    return {
        **normalized,
        "manifest_hash": freeze.canonical_hash(normalized),
    }


def _write_report_once(
    path: str | Path,
    core: dict[str, Any],
) -> dict[str, Any]:
    target = _path(path)
    payload = _manifest(core)
    stage_sources._write_json_once(target, payload)
    return load_report(target)


def load_report(path: str | Path) -> dict[str, Any]:
    target = _path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("BCTP run report must be a JSON object")
    core = dict(payload)
    embedded = core.pop("manifest_hash", None)
    if embedded != freeze.canonical_hash(core):
        raise ValueError("BCTP run report manifest hash mismatch")
    return payload


def _load_runner_report(
    path: str | Path,
    *,
    runner_commit: str,
) -> dict[str, Any]:
    report = load_report(path)
    if report.get("runner_commit") != runner_commit:
        raise RuntimeError("BCTP report runner commit binding drifted")
    return report


def _require_report_values(
    report: dict[str, Any],
    expected: dict[str, Any],
    *,
    label: str,
) -> None:
    for key, value in expected.items():
        if report.get(key) != value:
            raise RuntimeError(f"BCTP {label} report {key} binding drifted")


def _load_stage_manifest_only(
    stage: str,
    *,
    stage_source_root: str | Path,
    schedule_seal_binding: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate a stage-source manifest without opening market/funding copies."""

    stage_sources.verify_evaluator_freeze()
    path = _path(stage_source_root) / stage / "source_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("BCTP stage-source manifest must be a JSON object")
    core = dict(payload)
    embedded = core.pop("manifest_hash", None)
    if embedded != stage_sources._canonical_hash(core):
        raise ValueError("BCTP stage-source manifest hash mismatch")
    _require_report_values(
        payload,
        {
            "protocol_version": (
                "bctp_sequential_stage_source_isolation_v1"
            ),
            "stage": stage,
            "strategy_outcomes_calculated": False,
            "post_stage_numeric_rows_parsed": 0,
            "market_or_funding_parent_payload_bytes_hashed": False,
        },
        label="stage-source",
    )
    if payload.get("source_bindings", {}).get(
        "evaluator_manifest_hash"
    ) != stage_sources.EXPECTED_EVALUATOR_MANIFEST_HASH:
        raise RuntimeError("BCTP stage-source evaluator binding drifted")
    schedule = payload.get("target_schedule_binding")
    if stage == "2020":
        if schedule is not None or schedule_seal_binding is not None:
            raise RuntimeError("BCTP 2020 stage-source schedule binding drifted")
    else:
        if not isinstance(schedule, dict) or schedule_seal_binding is None:
            raise RuntimeError("BCTP stage-source schedule binding is missing")
        if (
            schedule.get("manifest_hash")
            != schedule_seal_binding.get("manifest_hash")
            or _path(str(schedule.get("path", ""))).resolve()
            != _path(str(schedule_seal_binding.get("path", ""))).resolve()
        ):
            raise RuntimeError("BCTP stage-source schedule binding drifted")
    return payload


def _validate_transfer_report(
    report: dict[str, Any],
    *,
    stage: str,
    schedule_manifest_hash: str,
    stage_source_manifest_hash: str,
    selected_primary_id: str | None = None,
) -> dict[str, Any]:
    protocol = f"bctp_transfer_{stage}_v1"
    expected = {
        "protocol_version": protocol,
        "stage": stage,
        "schedule_manifest_hash": schedule_manifest_hash,
        "stage_source_manifest_hash": stage_source_manifest_hash,
        f"post_{stage}_outcome_rows_opened": 0,
    }
    if stage == "2022":
        expected["selected_primary_id"] = selected_primary_id
    _require_report_values(report, expected, label=f"transfer-{stage}")
    evaluation = report.get("evaluation")
    if not isinstance(evaluation, dict):
        raise RuntimeError(f"BCTP transfer-{stage} evaluation is missing")
    if evaluation.get("stage") != stage:
        raise RuntimeError(f"BCTP transfer-{stage} evaluation stage drifted")
    passed = evaluation.get("stage_passed")
    if type(passed) is not bool:
        raise RuntimeError(f"BCTP transfer-{stage} pass flag is invalid")
    evaluated_selected = evaluation.get("selected_primary_id")
    if stage == "2021":
        selection_valid = (
            evaluated_selected in policy_family.PROMOTABLE_PRIMARY_IDS
            if passed
            else evaluated_selected is None
        )
        expected_decision = (
            "refit_and_seal_2022"
            if passed
            else "retire_bctp_cheap_policy_unchanged"
        )
    else:
        selection_valid = (
            evaluation.get("requested_primary_id") == selected_primary_id
            and (
                evaluated_selected == selected_primary_id
                if passed
                else evaluated_selected is None
            )
        )
        expected_decision = (
            "authorize_conditional_gemma4_e4b"
            if passed
            else "retire_bctp_cheap_policy_unchanged"
        )
    if not selection_valid:
        raise RuntimeError(f"BCTP transfer-{stage} selection binding drifted")
    if report.get("decision") != expected_decision:
        raise RuntimeError(f"BCTP transfer-{stage} decision binding drifted")
    return evaluation


def _source_states() -> pd.DataFrame:
    if _sha256(freeze.SEQUENCES) != freeze.EXPECTED_STATIC_SHA256[
        str(freeze.SEQUENCES)
    ]:
        raise ValueError("BCTP source sequence artifact changed")
    return transition_labels.load_source_states(freeze.SEQUENCES)


def _stage_states(source: pd.DataFrame, stage: str) -> pd.DataFrame:
    spec = stage_sources.STAGE_SPECS[stage]
    return transition_labels.stage_source_states(
        source,
        start=spec.start,
        end=spec.end,
    )


def _ledger_path(ledger_root: str | Path, stage: str) -> Path:
    return _path(ledger_root) / f"bctp_transition_ledger_{stage}.csv.gz"


def _build_or_load_ledger(
    stage: str,
    states: pd.DataFrame,
    market: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    ledger_root: str | Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    spec = stage_sources.STAGE_SPECS[stage]
    path = _ledger_path(ledger_root, stage)
    if path.exists():
        ledger = transition_labels.read_ledger(path)
        arrays = transition_labels.arrays_from_ledger(states, ledger)
        binding = {
            "path": str(path),
            "sha256": _sha256(path),
            "frame_hash": transition_labels.ledger_frame_hash(ledger),
            "rows": len(ledger),
            "columns": list(transition_labels.LEDGER_COLUMNS),
        }
        return binding, arrays
    built = transition_labels.build_reward_tensor(
        states,
        market,
        funding,
        start=spec.start,
        end=spec.end,
    )
    binding = transition_labels.write_ledger_once(
        path,
        built["ledger"],
    )
    ledger = transition_labels.read_ledger(path)
    if transition_labels.ledger_frame_hash(ledger) != binding["frame_hash"]:
        raise RuntimeError("BCTP transition ledger reload hash changed")
    arrays = transition_labels.arrays_from_ledger(states, ledger)
    return binding, arrays


def _schedule_paths(
    schedule_root: str | Path,
    stage: str,
) -> dict[str, Path]:
    root = _path(schedule_root) / stage
    return {
        "manifest": root / schedule_seal.MANIFEST_FILENAME,
        "base": root / schedule_seal.BASE_SCHEDULE_FILENAME,
        "delayed": (
            root / schedule_seal.DELAYED_SCHEDULE_FILENAME
        ),
    }


def _load_sealed_schedule_frames(
    stage: str,
    *,
    schedule_root: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    seal = schedule_seal.load_transfer_year_schedule_seal(
        stage,
        output_root=schedule_root,
    )
    base = pd.read_csv(
        _path(seal["base_schedules"]["path"]),
        compression="gzip",
        dtype=str,
    )
    delayed = pd.read_csv(
        _path(seal["delayed_primary_schedules"]["path"]),
        compression="gzip",
        dtype=str,
    )
    return seal, base, delayed


def _fit_data(
    stage_states: list[pd.DataFrame],
    stage_arrays: list[dict[str, np.ndarray]],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    if len(stage_states) != len(stage_arrays) or not stage_states:
        raise ValueError("BCTP fit episode inputs changed")
    states = pd.concat(stage_states, ignore_index=True)
    rewards = np.concatenate(
        [arrays["reward_tensor"] for arrays in stage_arrays],
        axis=0,
    )
    terminal = np.concatenate(
        [arrays["terminal"] for arrays in stage_arrays],
        axis=0,
    )
    reachable = np.concatenate(
        [arrays["reachable_mask"] for arrays in stage_arrays],
        axis=0,
    )
    return states, rewards, terminal, reachable


def prepare_2020_and_seal_2021(
    *,
    run_root: str | Path = DEFAULT_RUN_ROOT,
    ledger_root: str | Path = DEFAULT_LEDGER_ROOT,
    stage_source_root: str | Path = DEFAULT_STAGE_SOURCE_ROOT,
    schedule_root: str | Path = DEFAULT_SCHEDULE_ROOT,
) -> dict[str, Any]:
    """Open 2020 only, fit all frozen cheap policies, and seal 2021."""

    runner_commit = assert_runner_committed()
    report_path = _report_path(run_root, FIT_2020_REPORT)
    if report_path.exists():
        report = _load_runner_report(
            report_path,
            runner_commit=runner_commit,
        )
        seal = schedule_seal.load_transfer_year_schedule_seal(
            "2021",
            output_root=schedule_root,
        )
        source_manifest = _load_stage_manifest_only(
            "2020",
            stage_source_root=stage_source_root,
            schedule_seal_binding=None,
        )
        ledger_path = _ledger_path(ledger_root, "2020")
        ledger = transition_labels.read_ledger(ledger_path)
        _require_report_values(
            report,
            {
                "protocol_version": "bctp_fit_2020_and_seal_2021_v1",
                "fit_stage": "2020",
                "target_stage_sealed": "2021",
                "stage_source_manifest_hash": source_manifest[
                    "manifest_hash"
                ],
                "post_2020_outcome_rows_opened": 0,
                "decision": "authorize_2021_transfer_only",
            },
            label="fit-2020",
        )
        if report.get("schedule_seal", {}).get(
            "manifest_hash"
        ) != seal.get("manifest_hash"):
            raise RuntimeError("BCTP fit-2020 schedule binding drifted")
        if (
            report.get("transition_ledger", {}).get("sha256")
            != _sha256(ledger_path)
            or report.get("transition_ledger", {}).get("frame_hash")
            != transition_labels.ledger_frame_hash(ledger)
        ):
            raise RuntimeError("BCTP fit-2020 ledger binding drifted")
        return report

    source = _source_states()
    states_2020 = _stage_states(source, "2020")
    stage_manifest = stage_sources.prepare_stage_source(
        "2020",
        output_root=stage_source_root,
    )
    market, funding, stage_diagnostics = stage_sources.load_stage_source(
        "2020",
        output_root=stage_source_root,
    )
    ledger_binding, arrays_2020 = _build_or_load_ledger(
        "2020",
        states_2020,
        market,
        funding,
        ledger_root=ledger_root,
    )
    fit_states, rewards, terminal, reachable = _fit_data(
        [states_2020],
        [arrays_2020],
    )
    fitted = policy_family.fit_family(
        fit_states,
        rewards,
        terminal,
        reachable,
    )
    states_2021 = _stage_states(source, "2021")
    base, delayed = policy_family.build_transfer_schedules(
        fitted,
        states_2021,
        stage="2021",
    )
    seal = schedule_seal.seal_transfer_year_schedule(
        "2021",
        base,
        delayed,
        output_root=schedule_root,
    )
    return _write_report_once(
        report_path,
        {
            "protocol_version": "bctp_fit_2020_and_seal_2021_v1",
            "runner_commit": runner_commit,
            "evaluator_manifest_hash": (
                stage_sources.EXPECTED_EVALUATOR_MANIFEST_HASH
            ),
            "fit_stage": "2020",
            "target_stage_sealed": "2021",
            "stage_source_manifest_hash": stage_manifest["manifest_hash"],
            "stage_source_diagnostics": stage_diagnostics,
            "transition_ledger": ledger_binding,
            "fit_state_rows": len(fit_states),
            "fitted_estimators": fitted.fitted_estimators,
            "memory_tables_fit": fitted.memory_tables_fit,
            "schedule_seal": {
                "path": seal["path"],
                "manifest_hash": seal["manifest_hash"],
                "file_sha256": seal["file_sha256"],
            },
            "market_rows_opened": len(market),
            "funding_rows_opened": len(funding),
            "post_2020_outcome_rows_opened": 0,
            "decision": "authorize_2021_transfer_only",
        },
    )


def _load_existing_refit_report(
    *,
    selected_primary_id: str,
    run_root: str | Path,
    ledger_root: str | Path,
    schedule_root: str | Path,
    runner_commit: str,
) -> dict[str, Any]:
    report = _load_runner_report(
        _report_path(run_root, REFIT_2020_2021_REPORT),
        runner_commit=runner_commit,
    )
    seal = schedule_seal.load_transfer_year_schedule_seal(
        "2022",
        output_root=schedule_root,
    )
    _require_report_values(
        report,
        {
            "protocol_version": "bctp_refit_2020_2021_seal_2022_v1",
            "selected_primary_id": selected_primary_id,
            "fit_stages": ["2020", "2021"],
            "target_stage_sealed": "2022",
            "post_2021_outcome_rows_opened": 0,
            "decision": "authorize_selected_algorithm_2022_transfer_only",
        },
        label="refit-2020-2021",
    )
    if report.get("schedule_seal", {}).get(
        "manifest_hash"
    ) != seal.get("manifest_hash"):
        raise RuntimeError("BCTP refit schedule binding drifted")
    for stage, report_key in (
        ("2020", "transition_ledger_2020"),
        ("2021", "transition_ledger_2021"),
    ):
        path = _ledger_path(ledger_root, stage)
        ledger = transition_labels.read_ledger(path)
        binding = report.get(report_key, {})
        if (
            binding.get("sha256") != _sha256(path)
            or binding.get("frame_hash")
            != transition_labels.ledger_frame_hash(ledger)
        ):
            raise RuntimeError(f"BCTP refit {stage} ledger binding drifted")
    return report


def _refit_and_seal_2022(
    *,
    selected_primary_id: str,
    source: pd.DataFrame,
    market_2021: pd.DataFrame,
    funding_2021: pd.DataFrame,
    run_root: str | Path,
    ledger_root: str | Path,
    schedule_root: str | Path,
    runner_commit: str,
) -> dict[str, Any]:
    report_path = _report_path(run_root, REFIT_2020_2021_REPORT)
    if report_path.exists():
        return _load_existing_refit_report(
            selected_primary_id=selected_primary_id,
            run_root=run_root,
            ledger_root=ledger_root,
            schedule_root=schedule_root,
            runner_commit=runner_commit,
        )
    states_2020 = _stage_states(source, "2020")
    states_2021 = _stage_states(source, "2021")
    ledger_2020 = transition_labels.read_ledger(
        _ledger_path(ledger_root, "2020")
    )
    arrays_2020 = transition_labels.arrays_from_ledger(
        states_2020,
        ledger_2020,
    )
    binding_2021, arrays_2021 = _build_or_load_ledger(
        "2021",
        states_2021,
        market_2021,
        funding_2021,
        ledger_root=ledger_root,
    )
    states, rewards, terminal, reachable = _fit_data(
        [states_2020, states_2021],
        [arrays_2020, arrays_2021],
    )
    fitted = policy_family.fit_family(
        states,
        rewards,
        terminal,
        reachable,
    )
    states_2022 = _stage_states(source, "2022")
    base, delayed = policy_family.build_transfer_schedules(
        fitted,
        states_2022,
        stage="2022",
    )
    seal = schedule_seal.seal_transfer_year_schedule(
        "2022",
        base,
        delayed,
        output_root=schedule_root,
    )
    return _write_report_once(
        report_path,
        {
            "protocol_version": "bctp_refit_2020_2021_seal_2022_v1",
            "runner_commit": runner_commit,
            "selected_primary_id": selected_primary_id,
            "fit_stages": ["2020", "2021"],
            "target_stage_sealed": "2022",
            "transition_ledger_2020": {
                "path": str(_ledger_path(ledger_root, "2020")),
                "sha256": _sha256(_ledger_path(ledger_root, "2020")),
                "frame_hash": transition_labels.ledger_frame_hash(
                    ledger_2020
                ),
            },
            "transition_ledger_2021": binding_2021,
            "fit_state_rows": len(states),
            "fitted_estimators": fitted.fitted_estimators,
            "memory_tables_fit": fitted.memory_tables_fit,
            "schedule_seal": {
                "path": seal["path"],
                "manifest_hash": seal["manifest_hash"],
                "file_sha256": seal["file_sha256"],
            },
            "post_2021_outcome_rows_opened": 0,
            "decision": "authorize_selected_algorithm_2022_transfer_only",
        },
    )


def evaluate_2021_and_maybe_seal_2022(
    *,
    run_root: str | Path = DEFAULT_RUN_ROOT,
    ledger_root: str | Path = DEFAULT_LEDGER_ROOT,
    stage_source_root: str | Path = DEFAULT_STAGE_SOURCE_ROOT,
    schedule_root: str | Path = DEFAULT_SCHEDULE_ROOT,
) -> dict[str, Any]:
    """Evaluate 2021 and refit/seal 2022 only after a complete gate pass."""

    runner_commit = assert_runner_committed()
    prepare_2020_and_seal_2021(
        run_root=run_root,
        ledger_root=ledger_root,
        stage_source_root=stage_source_root,
        schedule_root=schedule_root,
    )
    seal_2021, base, delayed = _load_sealed_schedule_frames(
        "2021",
        schedule_root=schedule_root,
    )
    report_path = _report_path(run_root, TRANSFER_2021_REPORT)
    if report_path.exists():
        report = _load_runner_report(
            report_path,
            runner_commit=runner_commit,
        )
        stage_manifest = _load_stage_manifest_only(
            "2021",
            stage_source_root=stage_source_root,
            schedule_seal_binding=seal_2021,
        )
        evaluation = _validate_transfer_report(
            report,
            stage="2021",
            schedule_manifest_hash=seal_2021["manifest_hash"],
            stage_source_manifest_hash=stage_manifest["manifest_hash"],
        )
        if not evaluation["stage_passed"]:
            return report
        selected = str(evaluation["selected_primary_id"])
        if _report_path(run_root, REFIT_2020_2021_REPORT).exists():
            refit = _load_existing_refit_report(
                selected_primary_id=selected,
                run_root=run_root,
                ledger_root=ledger_root,
                schedule_root=schedule_root,
                runner_commit=runner_commit,
            )
            return {**report, "refit_2020_2021": refit}
        market, funding, _ = stage_sources.load_stage_source(
            "2021",
            output_root=stage_source_root,
        )
    else:
        stage_manifest = stage_sources.prepare_stage_source(
            "2021",
            output_root=stage_source_root,
            required_schedule_manifest=seal_2021["path"],
        )
        market, funding, stage_diagnostics = stage_sources.load_stage_source(
            "2021",
            output_root=stage_source_root,
        )
        evaluation = family_evaluator.evaluate_bctp_transfer_stage(
            "2021",
            market,
            funding,
            base,
            delayed,
        )
        report = _write_report_once(
            report_path,
            {
                "protocol_version": "bctp_transfer_2021_v1",
                "runner_commit": runner_commit,
                "stage": "2021",
                "schedule_manifest_hash": seal_2021["manifest_hash"],
                "stage_source_manifest_hash": stage_manifest["manifest_hash"],
                "stage_source_diagnostics": stage_diagnostics,
                "evaluation": evaluation,
                "decision": (
                    "refit_and_seal_2022"
                    if evaluation["stage_passed"]
                    else "retire_bctp_cheap_policy_unchanged"
                ),
                "post_2021_outcome_rows_opened": 0,
            },
        )
        if not evaluation["stage_passed"]:
            return report
        selected = str(evaluation["selected_primary_id"])
    source = _source_states()
    refit = _refit_and_seal_2022(
        selected_primary_id=selected,
        source=source,
        market_2021=market,
        funding_2021=funding,
        run_root=run_root,
        ledger_root=ledger_root,
        schedule_root=schedule_root,
        runner_commit=runner_commit,
    )
    return {**report, "refit_2020_2021": refit}


def evaluate_2022_selected_algorithm(
    *,
    run_root: str | Path = DEFAULT_RUN_ROOT,
    ledger_root: str | Path = DEFAULT_LEDGER_ROOT,
    stage_source_root: str | Path = DEFAULT_STAGE_SOURCE_ROOT,
    schedule_root: str | Path = DEFAULT_SCHEDULE_ROOT,
) -> dict[str, Any]:
    """Evaluate the unchanged selected 2021 algorithm once on 2022."""

    runner_commit = assert_runner_committed()
    evaluate_2021_and_maybe_seal_2022(
        run_root=run_root,
        ledger_root=ledger_root,
        stage_source_root=stage_source_root,
        schedule_root=schedule_root,
    )
    transfer_2021 = _load_runner_report(
        _report_path(run_root, TRANSFER_2021_REPORT),
        runner_commit=runner_commit,
    )
    if not transfer_2021["evaluation"]["stage_passed"]:
        raise RuntimeError("BCTP 2021 did not authorize 2022")
    selected = str(transfer_2021["evaluation"]["selected_primary_id"])
    refit = _load_existing_refit_report(
        selected_primary_id=selected,
        run_root=run_root,
        ledger_root=ledger_root,
        schedule_root=schedule_root,
        runner_commit=runner_commit,
    )
    if selected != str(
        refit["selected_primary_id"]
    ):
        raise RuntimeError("BCTP selected algorithm binding drifted")
    seal_2022, base, delayed = _load_sealed_schedule_frames(
        "2022",
        schedule_root=schedule_root,
    )
    report_path = _report_path(run_root, TRANSFER_2022_REPORT)
    if report_path.exists():
        report = _load_runner_report(
            report_path,
            runner_commit=runner_commit,
        )
        stage_manifest = _load_stage_manifest_only(
            "2022",
            stage_source_root=stage_source_root,
            schedule_seal_binding=seal_2022,
        )
        _validate_transfer_report(
            report,
            stage="2022",
            schedule_manifest_hash=seal_2022["manifest_hash"],
            stage_source_manifest_hash=stage_manifest["manifest_hash"],
            selected_primary_id=selected,
        )
        return report
    stage_manifest = stage_sources.prepare_stage_source(
        "2022",
        output_root=stage_source_root,
        required_schedule_manifest=seal_2022["path"],
    )
    market, funding, stage_diagnostics = stage_sources.load_stage_source(
        "2022",
        output_root=stage_source_root,
    )
    evaluation = family_evaluator.evaluate_bctp_transfer_stage(
        "2022",
        market,
        funding,
        base,
        delayed,
        selected_primary_id=selected,
    )
    return _write_report_once(
        report_path,
        {
            "protocol_version": "bctp_transfer_2022_v1",
            "runner_commit": runner_commit,
            "stage": "2022",
            "selected_primary_id": selected,
            "schedule_manifest_hash": seal_2022["manifest_hash"],
            "stage_source_manifest_hash": stage_manifest["manifest_hash"],
            "stage_source_diagnostics": stage_diagnostics,
            "evaluation": evaluation,
            "decision": (
                "authorize_conditional_gemma4_e4b"
                if evaluation["stage_passed"]
                else "retire_bctp_cheap_policy_unchanged"
            ),
            "post_2022_outcome_rows_opened": 0,
        },
    )


def status(
    *,
    run_root: str | Path = DEFAULT_RUN_ROOT,
) -> dict[str, Any]:
    root = _path(run_root)
    reports = {}
    for filename in (
        FIT_2020_REPORT,
        TRANSFER_2021_REPORT,
        REFIT_2020_2021_REPORT,
        TRANSFER_2022_REPORT,
    ):
        path = root / filename
        reports[filename] = (
            {
                "exists": True,
                "manifest_hash": load_report(path)["manifest_hash"],
            }
            if path.exists()
            else {"exists": False}
        )
    return {"run_root": str(root), "reports": reports}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "prepare-2020",
            "evaluate-2021",
            "evaluate-2022",
            "status",
        ),
    )
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--ledger-root", default=str(DEFAULT_LEDGER_ROOT))
    parser.add_argument(
        "--stage-source-root",
        default=str(DEFAULT_STAGE_SOURCE_ROOT),
    )
    parser.add_argument(
        "--schedule-root",
        default=str(DEFAULT_SCHEDULE_ROOT),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    common = {
        "run_root": args.run_root,
    }
    if args.command == "prepare-2020":
        result = prepare_2020_and_seal_2021(
            **common,
            ledger_root=args.ledger_root,
            stage_source_root=args.stage_source_root,
            schedule_root=args.schedule_root,
        )
    elif args.command == "evaluate-2021":
        result = evaluate_2021_and_maybe_seal_2022(
            **common,
            ledger_root=args.ledger_root,
            stage_source_root=args.stage_source_root,
            schedule_root=args.schedule_root,
        )
    elif args.command == "evaluate-2022":
        result = evaluate_2022_selected_algorithm(
            **common,
            ledger_root=args.ledger_root,
            stage_source_root=args.stage_source_root,
            schedule_root=args.schedule_root,
        )
    else:
        result = status(**common)
    print(
        json.dumps(
            result,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
