"""Execute preregistered, outcome-blind CCHR pure-clock exports.

Every selected family is preflighted against its immutable hash-bound
preregistration before any source row loader can run.  The exporter then reads
only the causal allowlists owned by the family module, writes the exact frozen
six-column clock, and publishes a provenance manifest.  It never reads prices
for payoff calculation, outcomes, returns, PnL, or post-2023 source rows.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence, cast

import pandas as pd

from training import cchr_comparator_clock_common as clock_common
from training import export_cchr_dtv_pure_clocks as dtv
from training import export_cchr_far_pure_clocks as far
from training import export_cchr_live_portfolio_pure_clocks as live
from training import export_cchr_pdlh_pure_clocks as pdlh
from training import preregister_cchr_pure_clock_exports as prereg


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPOSITORY_ROOT / "results"

EXPORT_OUTCOME_BOUNDARY = {
    "preregistration_json_read": True,
    "raw_input_bytes_hash_verified": True,
    "causal_source_columns_loaded": True,
    "source_end_exclusive": "2024-01-01T00:00:00Z",
    "outcome_artifacts_parsed": 0,
    "comparator_outcomes_read": 0,
    "return_or_pnl_fields_read": 0,
    "post_2023_rows_loaded": 0,
    "sealed_rows_materialized": 0,
    "sealed_non_timestamp_fields_decoded": 0,
    "first_sealed_timestamp_field_examined": True,
    "network_calls": 0,
    "subprocess_calls": 0,
}


@dataclass(frozen=True)
class ExportPlan:
    family: str
    preregistration: dict[str, Any]
    preregistration_path: Path
    preregistration_sha256: str
    clock_target: Path
    manifest_target: Path


@dataclass(frozen=True)
class StagedExport:
    plan: ExportPlan
    manifest: dict[str, Any]
    clock_temporary: Path
    manifest_temporary: Path


def _absolute_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return Path(os.path.abspath(candidate))


def _protected_paths(payload: Mapping[str, Any]) -> set[Path]:
    protected = {
        _absolute_path(prereg.MASTER_PREREGISTRATION).resolve(),
        _absolute_path(prereg.MASTER_SOURCE).resolve(),
        _absolute_path(prereg.PREREGISTRATION_SOURCE).resolve(),
        _absolute_path(prereg.COMMON_SOURCE).resolve(),
        _absolute_path(prereg.RUNNER_SOURCE).resolve(),
        _absolute_path(prereg.EXPORTER_SOURCES[str(payload["family"])]).resolve(),
    }
    protected.update(
        _absolute_path(binding["path"]).resolve()
        for binding in cast(
            Mapping[str, Mapping[str, Any]], payload["raw_input_bindings"]
        ).values()
    )
    protected.update(
        _absolute_path(binding["path"]).resolve()
        for binding in cast(
            Mapping[str, Mapping[str, Any]], payload["configuration_bindings"]
        ).values()
    )
    return protected


def _validated_result_target(
    path: str | Path,
    *,
    expected_suffix: str,
    protected: set[Path],
) -> Path:
    target = _absolute_path(path)
    results_root = _absolute_path(RESULTS_ROOT)
    if not str(target).endswith(expected_suffix):
        raise ValueError(f"CCHR export output must end with {expected_suffix}")
    if results_root.is_symlink():
        raise ValueError("CCHR export results root cannot be a symlink")
    current = target
    while True:
        if current.is_symlink():
            raise ValueError("CCHR export path cannot contain a symlink")
        if current == results_root or current.parent == current:
            break
        current = current.parent
    resolved_root = results_root.resolve()
    resolved = target.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("CCHR export output must remain under results") from error
    if resolved.parent != resolved_root:
        raise ValueError("CCHR export output must be a direct child of results")
    if resolved in protected:
        raise ValueError("CCHR export output aliases a protected input")
    return resolved


def _verify_runner_binding(payload: Mapping[str, Any]) -> None:
    implementations = cast(
        Mapping[str, Mapping[str, Any]], payload["implementation_bindings"]
    )
    expected = {
        "path": str(prereg.RUNNER_SOURCE),
        "sha256": clock_common.sha256_file(prereg.RUNNER_SOURCE),
    }
    if implementations.get("runner") != expected:
        raise RuntimeError("CCHR export runner differs from preregistration")


def _preflight_family(family: str) -> ExportPlan:
    payload, preregistration_sha256 = prereg.load_preregistration_with_sha256(
        family,
        verify_files=True,
    )
    _verify_runner_binding(payload)
    contract = cast(Mapping[str, Any], payload["output_contract"])
    protected = _protected_paths(payload)
    clock_target = _validated_result_target(
        str(contract["pure_clock"]),
        expected_suffix=".csv.gz",
        protected=protected,
    )
    manifest_target = _validated_result_target(
        str(contract["export_manifest"]),
        expected_suffix=".json",
        protected=protected,
    )
    if clock_target == manifest_target:
        raise ValueError("CCHR clock and export-manifest outputs must be distinct")
    if clock_target.exists() or manifest_target.exists():
        raise FileExistsError("CCHR pure-clock export is immutable")
    return ExportPlan(
        family=family,
        preregistration=payload,
        preregistration_path=_absolute_path(prereg.PREREGISTRATION_OUTPUTS[family]),
        preregistration_sha256=preregistration_sha256,
        clock_target=clock_target,
        manifest_target=manifest_target,
    )


def preflight_exports(families: Sequence[str]) -> tuple[ExportPlan, ...]:
    selected = tuple(families)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("CCHR export families must be non-empty and unique")
    unknown = sorted(set(selected) - set(prereg.FAMILIES))
    if unknown:
        raise KeyError(f"unknown CCHR comparator families: {unknown}")
    plans = tuple(_preflight_family(family) for family in selected)
    targets = [
        target for plan in plans for target in (plan.clock_target, plan.manifest_target)
    ]
    if len(set(targets)) != len(targets):
        raise ValueError("CCHR export outputs must be globally unique")
    return plans


def _build_family_clock(family: str) -> pd.DataFrame:
    if family == "pdlh":
        market, metrics = pdlh.load_causal_inputs()
        return pdlh.build_pdlh_clock(market, metrics)
    if family == "dtv":
        return dtv.build_clock_frame(dtv.load_pre2024())
    if family == "far":
        market, dates, event_rate = far.load_hash_bound_pre2024()
        return far.build_far_clock_frame(market, dates, event_rate)
    if family == "live":
        live.validate_config_hashes()
        inputs = live.load_causal_inputs()
        return live.build_clock(
            inputs["market"],
            inputs["funding"],
            inputs["premium"],
            inputs["upbit"],
        )
    raise KeyError(f"unknown CCHR comparator family: {family}")


def _verify_preregistration_unchanged(plan: ExportPlan) -> None:
    if clock_common.sha256_file(plan.preregistration_path) != (
        plan.preregistration_sha256
    ):
        raise RuntimeError("CCHR export preregistration changed after preflight")


def _coverage(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    coverage: dict[str, dict[str, Any]] = {}
    for split in clock_common.research_splits():
        rows = frame.loc[frame["split"].eq(split.name)]
        coverage[split.name] = {
            "rows": int(len(rows)),
            "observed_member_count": int(rows["candidate_id"].nunique()),
            "first_decision_time": (
                str(rows["decision_time"].min()) if len(rows) else None
            ),
            "last_exit_boundary": str(rows["exit_time"].max()) if len(rows) else None,
        }
    return coverage


def build_export_manifest(
    plan: ExportPlan,
    frame: pd.DataFrame,
    *,
    clock_sha256: str,
) -> dict[str, Any]:
    candidate_map = cast(Mapping[str, Any], plan.preregistration["candidate_map"])
    normalized = clock_common.validate_clock_frame(
        frame,
        expected_candidate_ids=tuple(candidate_map),
    )
    rows_by_candidate = {
        str(candidate_id): int(count)
        for candidate_id, count in normalized.groupby("candidate_id", sort=True)
        .size()
        .items()
    }
    payload: dict[str, Any] = {
        "protocol_version": prereg.EXPORT_MANIFEST_PROTOCOL_VERSION,
        "policy_id": prereg.POLICY_ID,
        "family": plan.family,
        "preregistration": {
            "path": str(prereg.PREREGISTRATION_OUTPUTS[plan.family]),
            "sha256": plan.preregistration_sha256,
            "manifest_hash": plan.preregistration["manifest_hash"],
        },
        "implementation_bindings": plan.preregistration["implementation_bindings"],
        "raw_input_bindings": plan.preregistration["raw_input_bindings"],
        "configuration_bindings": plan.preregistration["configuration_bindings"],
        "candidate_map_sha256": plan.preregistration["candidate_map_sha256"],
        "member_count": plan.preregistration["member_count"],
        "clock": {
            "path": str(plan.preregistration["output_contract"]["pure_clock"]),
            "sha256": clock_sha256,
            "schema": list(clock_common.CLOCK_COLUMNS),
            "compression": "gzip",
            "gzip_mtime": 0,
            "rows": int(len(normalized)),
            "rows_by_candidate": rows_by_candidate,
            "coverage": _coverage(normalized),
        },
        "export_manifest": {
            "path": str(plan.preregistration["output_contract"]["export_manifest"]),
        },
        "authorization": {
            "combined_comparator_freeze": False,
            "outcome_evaluator": False,
            "post_2023_source_access": False,
        },
        "outcomes_opened": False,
        "outcome_boundary": dict(EXPORT_OUTCOME_BOUNDARY),
    }
    payload["manifest_hash"] = clock_common.canonical_hash(payload)
    validate_export_manifest(payload, plan.preregistration)
    return payload


def validate_export_manifest(
    payload: Mapping[str, Any], preregistration: Mapping[str, Any]
) -> None:
    if frozenset(payload) != frozenset(prereg.EXPORT_MANIFEST_TOP_LEVEL_KEYS):
        raise RuntimeError("CCHR export manifest top-level schema drift")
    if payload.get("protocol_version") != prereg.EXPORT_MANIFEST_PROTOCOL_VERSION:
        raise RuntimeError("CCHR export manifest protocol drift")
    if payload.get("policy_id") != prereg.POLICY_ID:
        raise RuntimeError("CCHR export manifest policy drift")
    if payload.get("family") != preregistration.get("family"):
        raise RuntimeError("CCHR export manifest family drift")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != clock_common.canonical_hash(core):
        raise RuntimeError("CCHR export manifest hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("CCHR export manifest opened outcomes")
    if payload.get("outcome_boundary") != EXPORT_OUTCOME_BOUNDARY:
        raise RuntimeError("CCHR export manifest outcome boundary drift")
    if payload.get("implementation_bindings") != preregistration.get(
        "implementation_bindings"
    ):
        raise RuntimeError("CCHR export implementation binding drift")
    if payload.get("raw_input_bindings") != preregistration.get("raw_input_bindings"):
        raise RuntimeError("CCHR export raw-input binding drift")
    if payload.get("configuration_bindings") != preregistration.get(
        "configuration_bindings"
    ):
        raise RuntimeError("CCHR export configuration binding drift")
    if payload.get("candidate_map_sha256") != preregistration.get(
        "candidate_map_sha256"
    ) or payload.get("member_count") != preregistration.get("member_count"):
        raise RuntimeError("CCHR export candidate-map binding drift")
    if payload.get("authorization") != {
        "combined_comparator_freeze": False,
        "outcome_evaluator": False,
        "post_2023_source_access": False,
    }:
        raise RuntimeError("CCHR export authorization drift")


def _temporary_path(target: Path, *, suffix: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=suffix, dir=target.parent
    )
    os.close(descriptor)
    path = Path(name)
    path.unlink()
    return path


def _write_temporary_json(payload: Mapping[str, Any], target: Path) -> Path:
    path = _temporary_path(target, suffix=".json.tmp")
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _same_inode_at(directory_fd: int, left: str, right: str) -> bool:
    try:
        return os.path.samestat(
            os.stat(left, dir_fd=directory_fd, follow_symlinks=False),
            os.stat(right, dir_fd=directory_fd, follow_symlinks=False),
        )
    except FileNotFoundError:
        return False


def _publish_files_create_only(files: Sequence[tuple[Path, Path]]) -> None:
    if not files:
        raise ValueError("CCHR export publication set cannot be empty")
    parents = {path.parent for pair in files for path in pair}
    if len(parents) != 1:
        raise ValueError("CCHR export publication requires one results directory")
    temporary_names = [temporary.name for temporary, _ in files]
    target_names = [target.name for _, target in files]
    if len(set(temporary_names)) != len(temporary_names) or len(
        set(target_names)
    ) != len(target_names):
        raise ValueError("CCHR export publication paths must be unique")
    output_directory = next(iter(parents))
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(output_directory, flags)
    linked: list[tuple[str, str]] = []
    try:
        if not os.path.samestat(
            os.fstat(directory_fd),
            os.stat(output_directory, follow_symlinks=False),
        ):
            raise RuntimeError("CCHR results directory changed before publication")
        for temporary, target in files:
            os.link(
                temporary.name,
                target.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            linked.append((temporary.name, target.name))
        os.fsync(directory_fd)
    except BaseException as error:
        for temporary_name, target_name in reversed(linked):
            if _same_inode_at(directory_fd, temporary_name, target_name):
                os.unlink(target_name, dir_fd=directory_fd)
        if isinstance(error, FileExistsError):
            raise FileExistsError("CCHR pure-clock export is immutable") from error
        raise
    finally:
        for name in temporary_names:
            try:
                os.unlink(name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)


def _publish_pair_create_only(
    clock_temporary: Path,
    clock_target: Path,
    manifest_temporary: Path,
    manifest_target: Path,
) -> None:
    _publish_files_create_only(
        (
            (clock_temporary, clock_target),
            (manifest_temporary, manifest_target),
        )
    )


def _stage_plan(plan: ExportPlan) -> StagedExport:
    _verify_preregistration_unchanged(plan)
    frame = _build_family_clock(plan.family)
    _verify_preregistration_unchanged(plan)
    expected_ids = tuple(cast(Mapping[str, Any], plan.preregistration["candidate_map"]))
    normalized = clock_common.validate_clock_frame(
        frame,
        expected_candidate_ids=expected_ids,
    )
    clock_temporary = _temporary_path(plan.clock_target, suffix=".csv.gz.tmp")
    manifest_temporary: Path | None = None
    try:
        clock_sha256 = clock_common.write_deterministic_gzip_clock(
            normalized,
            clock_temporary,
            expected_candidate_ids=expected_ids,
        )
        manifest = build_export_manifest(
            plan,
            normalized,
            clock_sha256=clock_sha256,
        )
        manifest_temporary = _write_temporary_json(manifest, plan.manifest_target)
        return StagedExport(
            plan=plan,
            manifest=manifest,
            clock_temporary=clock_temporary,
            manifest_temporary=manifest_temporary,
        )
    except BaseException:
        clock_temporary.unlink(missing_ok=True)
        if manifest_temporary is not None:
            manifest_temporary.unlink(missing_ok=True)
        raise


def _publish_staged_exports(staged: Sequence[StagedExport]) -> None:
    _publish_files_create_only(
        tuple(
            pair
            for export in staged
            for pair in (
                (export.clock_temporary, export.plan.clock_target),
                (export.manifest_temporary, export.plan.manifest_target),
            )
        )
    )


def _execute_plan(plan: ExportPlan) -> dict[str, Any]:
    staged = _stage_plan(plan)
    _publish_staged_exports((staged,))
    return staged.manifest


def export_families(families: Sequence[str]) -> dict[str, dict[str, Any]]:
    plans = preflight_exports(families)
    staged: list[StagedExport] = []
    try:
        for plan in plans:
            staged.append(_stage_plan(plan))
        _publish_staged_exports(staged)
    except BaseException:
        for export in staged:
            export.clock_temporary.unlink(missing_ok=True)
            export.manifest_temporary.unlink(missing_ok=True)
        raise
    return {export.plan.family: export.manifest for export in staged}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        action="append",
        choices=prereg.FAMILIES,
        help="export one family; repeat as needed (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    families = tuple(args.family) if args.family else prereg.FAMILIES
    manifests = export_families(families)
    print(
        json.dumps(
            {
                family: {
                    "clock": manifest["clock"],
                    "export_manifest": manifest["export_manifest"],
                    "outcomes_opened": manifest["outcomes_opened"],
                }
                for family, manifest in manifests.items()
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
