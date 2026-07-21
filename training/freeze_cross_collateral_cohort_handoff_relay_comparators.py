"""Freeze every pre-2024 comparator binding required by CCHR-288.

This stage parses only preregistration and export-manifest JSON.  Pure-clock,
raw-input, and outcome-bearing provenance artifacts are hashed as opaque bytes;
their rows and values remain unopened.  The resulting artifact authorizes only
the later source-only CCHR support/novelty stage, never economic outcomes.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
from typing import Any, Callable, Mapping, cast

from training import cchr_comparator_clock_common as clock_common
from training import export_cchr_pure_clocks as exporter
from training import preregister_cchr_pure_clock_exports as export_prereg
from training import preregister_cross_collateral_cohort_handoff_relay as cchr


PROTOCOL_VERSION = "cross_collateral_cohort_handoff_relay_comparator_freeze_v1"
POLICY_ID = cchr.POLICY_ID
AS_OF_DATE = "2026-07-21"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPOSITORY_ROOT / "results"
FREEZE_SOURCE = Path(
    "training/freeze_cross_collateral_cohort_handoff_relay_comparators.py"
)
DEFAULT_OUTPUT = cchr.COMPARATOR_FREEZE

HashFile = Callable[[str | Path], str]
DirectoryIdentity = tuple[int, int]

OUTCOME_BOUNDARY = {
    "master_preregistration_json_parsed": 1,
    "family_preregistration_json_parsed": len(export_prereg.FAMILIES),
    "export_manifest_json_parsed": len(export_prereg.FAMILIES),
    "source_manifest_json_parsed": 0,
    "pure_clock_bytes_hashed": True,
    "pure_clock_rows_read": 0,
    "raw_input_bytes_hashed": True,
    "raw_input_rows_read": 0,
    "outcome_bearing_provenance_bytes_hashed": True,
    "outcome_bearing_provenance_json_parsed": 0,
    "cchr_source_values_read": 0,
    "cchr_incidence_rows_derived": 0,
    "comparator_rows_read": 0,
    "market_rows_loaded": 0,
    "funding_rows_loaded": 0,
    "return_or_pnl_fields_read": 0,
    "post_2023_rows_loaded": 0,
    "network_calls": 0,
    "subprocess_calls": 0,
}

AUTHORIZATION = {
    "source_only_cchr_incidence_after_this_artifact": True,
    "source_only_support_and_novelty": True,
    "outcome_evaluator": False,
    "post_2023_source_access": False,
    "condition": "all path and SHA-256 bindings must remain byte-exact",
}

TOP_LEVEL_KEYS = frozenset(
    {
        "protocol_version",
        "policy_id",
        "as_of_date",
        "master_preregistration",
        "freeze_requirement",
        "freeze_implementation",
        "source_binding",
        "comparator_provenance_bindings",
        "comparator_candidate_map_sha256",
        "comparator_member_count",
        "generated_families",
        "legacy_comparators",
        "authorization",
        "outcomes_opened",
        "outcome_boundary",
        "manifest_hash",
    }
)

REQUIRED_FAMILY_BINDING_KEYS = frozenset(
    cchr.comparator_freeze_requirement()["must_bind"]
)


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return Path(os.path.abspath(candidate))


def _read_json(path: str | Path) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError(f"duplicate JSON key in CCHR freeze input: {key}")
            payload[key] = value
        return payload

    with _repository_path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle, object_pairs_hook=reject_duplicate_keys)
    if not isinstance(payload, dict):
        raise TypeError(f"CCHR freeze input must be a JSON object: {path}")
    return payload


def _read_hash_bound_json(
    path: str | Path,
    *,
    hash_file: HashFile,
) -> tuple[dict[str, Any], str]:
    before = hash_file(path)
    payload = _read_json(path)
    after = hash_file(path)
    if before != after:
        raise RuntimeError(f"CCHR freeze JSON changed while being parsed: {path}")
    return payload, before


def _checked_hash(
    path: str | Path,
    expected: str,
    *,
    hash_file: HashFile,
    cache: dict[str, str],
) -> str:
    key = str(path)
    if key not in cache:
        cache[key] = hash_file(path)
    if cache[key] != expected:
        raise RuntimeError(f"CCHR comparator freeze binding drifted: {path}")
    return cache[key]


def _validate_clock_metadata(
    family: str,
    clock: Mapping[str, Any],
    preregistration: Mapping[str, Any],
) -> None:
    expected_keys = {
        "path",
        "sha256",
        "schema",
        "compression",
        "gzip_mtime",
        "rows",
        "rows_by_candidate",
        "coverage",
    }
    if set(clock) != expected_keys:
        raise RuntimeError(f"{family} CCHR clock metadata schema drift")
    contract = cast(Mapping[str, Any], preregistration["output_contract"])
    if clock.get("path") != contract.get("pure_clock"):
        raise RuntimeError(f"{family} CCHR pure-clock path drift")
    if clock.get("schema") != list(clock_common.CLOCK_COLUMNS):
        raise RuntimeError(f"{family} CCHR pure-clock schema drift")
    if clock.get("compression") != "gzip" or clock.get("gzip_mtime") != 0:
        raise RuntimeError(f"{family} CCHR pure-clock encoding drift")

    rows = clock.get("rows")
    if type(rows) is not int or rows < 0:
        raise RuntimeError(f"{family} CCHR pure-clock row count is invalid")
    candidate_map = cast(Mapping[str, Any], preregistration["candidate_map"])
    rows_by_candidate = clock.get("rows_by_candidate")
    if not isinstance(rows_by_candidate, dict) or set(rows_by_candidate) != set(
        candidate_map
    ):
        raise RuntimeError(f"{family} CCHR rows-by-candidate map drift")
    if any(type(value) is not int or value < 0 for value in rows_by_candidate.values()):
        raise RuntimeError(f"{family} CCHR rows-by-candidate count is invalid")
    if sum(rows_by_candidate.values()) != rows:
        raise RuntimeError(f"{family} CCHR rows-by-candidate total mismatch")

    coverage = clock.get("coverage")
    split_names = {split.name for split in clock_common.research_splits()}
    if not isinstance(coverage, dict) or set(coverage) != split_names:
        raise RuntimeError(f"{family} CCHR coverage split drift")
    split_by_name = {split.name: split for split in clock_common.research_splits()}
    coverage_total = 0
    for split_name, item in coverage.items():
        if not isinstance(item, dict) or set(item) != {
            "rows",
            "observed_member_count",
            "first_decision_time",
            "last_exit_boundary",
        }:
            raise RuntimeError(f"{family} CCHR {split_name} coverage schema drift")
        split_rows = item["rows"]
        observed = item["observed_member_count"]
        if type(split_rows) is not int or split_rows < 0:
            raise RuntimeError(f"{family} CCHR {split_name} coverage rows invalid")
        if type(observed) is not int or not 0 <= observed <= len(candidate_map):
            raise RuntimeError(
                f"{family} CCHR {split_name} observed-member count invalid"
            )
        first = item["first_decision_time"]
        last = item["last_exit_boundary"]
        if split_rows == 0:
            if observed != 0 or first is not None or last is not None:
                raise RuntimeError(
                    f"{family} CCHR empty {split_name} coverage is inconsistent"
                )
        elif observed == 0 or not isinstance(first, str) or not isinstance(last, str):
            raise RuntimeError(
                f"{family} CCHR non-empty {split_name} coverage is inconsistent"
            )
        else:
            try:
                first_time = datetime.strptime(first, "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc
                )
                last_time = datetime.strptime(last, "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc
                )
            except ValueError as error:
                raise RuntimeError(
                    f"{family} CCHR {split_name} coverage timestamp is invalid"
                ) from error
            split = split_by_name[split_name]
            if (
                first_time.minute % 5
                or last_time.minute % 5
                or first_time.second
                or last_time.second
                or first_time > last_time
                or not split.start <= first_time < split.end
                or not split.start < last_time < split.end
            ):
                raise RuntimeError(
                    f"{family} CCHR {split_name} coverage timestamp drift"
                )
        coverage_total += split_rows
    if coverage_total != rows:
        raise RuntimeError(f"{family} CCHR coverage row total mismatch")


def _validate_export_linkage(
    family: str,
    preregistration: Mapping[str, Any],
    preregistration_sha256: str,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    clock_sha256: str,
) -> None:
    exporter.validate_export_manifest(manifest, preregistration)
    contract = cast(Mapping[str, Any], preregistration["output_contract"])
    expected_preregistration = {
        "path": str(export_prereg.PREREGISTRATION_OUTPUTS[family]),
        "sha256": preregistration_sha256,
        "manifest_hash": preregistration["manifest_hash"],
    }
    if manifest.get("preregistration") != expected_preregistration:
        raise RuntimeError(f"{family} CCHR export preregistration linkage drift")
    if manifest.get("export_manifest") != {"path": contract["export_manifest"]}:
        raise RuntimeError(f"{family} CCHR export-manifest path drift")
    clock = cast(Mapping[str, Any], manifest["clock"])
    _validate_clock_metadata(family, clock, preregistration)
    if clock.get("sha256") != clock_sha256:
        raise RuntimeError(f"{family} CCHR pure-clock SHA-256 mismatch")
    if manifest_sha256 != clock_common.sha256_file(contract["export_manifest"]):
        raise RuntimeError(f"{family} CCHR export manifest changed after parsing")


def _verify_master_bindings(
    master: Mapping[str, Any],
    *,
    hash_file: HashFile,
    cache: dict[str, str],
) -> None:
    source = cast(Mapping[str, Any], master["source_binding"])
    for path_key, hash_key in (
        ("path", "sha256"),
        ("manifest", "manifest_sha256"),
        ("builder", "builder_sha256"),
        ("audit", "audit_sha256"),
    ):
        _checked_hash(
            str(source[path_key]),
            str(source[hash_key]),
            hash_file=hash_file,
            cache=cache,
        )
    mechanism = cast(Mapping[str, Any], master["mechanism_decision"])
    _checked_hash(
        str(mechanism["path"]),
        str(mechanism["sha256"]),
        hash_file=hash_file,
        cache=cache,
    )
    preregistration_source = cast(Mapping[str, Any], master["preregistration_source"])
    _checked_hash(
        str(preregistration_source["path"]),
        str(preregistration_source["sha256"]),
        hash_file=hash_file,
        cache=cache,
    )
    provenance = cast(
        Mapping[str, Mapping[str, Any]], master["comparator_provenance_bindings"]
    )
    for binding in provenance.values():
        _checked_hash(
            str(binding["path"]),
            str(binding["sha256"]),
            hash_file=hash_file,
            cache=cache,
        )


def _family_binding(
    family: str,
    preregistration: Mapping[str, Any],
    preregistration_sha256: str,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
) -> dict[str, Any]:
    implementations = cast(
        Mapping[str, Mapping[str, Any]], preregistration["implementation_bindings"]
    )
    clock = cast(Mapping[str, Any], manifest["clock"])
    contract = cast(Mapping[str, Any], preregistration["output_contract"])
    required_bindings = {
        "exporter_sha256": implementations["exporter"]["sha256"],
        "raw_input_path_sha256_and_column_allowlist": preregistration[
            "raw_input_bindings"
        ],
        "export_manifest_sha256": manifest_sha256,
        "pure_clock_sha256": clock["sha256"],
        "coverage": clock["coverage"],
        "member_count": preregistration["member_count"],
        "candidate_map_sha256": preregistration["candidate_map_sha256"],
    }
    if set(required_bindings) != REQUIRED_FAMILY_BINDING_KEYS:
        raise RuntimeError(f"{family} CCHR required freeze bindings are incomplete")
    return {
        "preregistration": {
            "path": str(export_prereg.PREREGISTRATION_OUTPUTS[family]),
            "sha256": preregistration_sha256,
            "manifest_hash": preregistration["manifest_hash"],
        },
        "implementation_bindings": implementations,
        "configuration_bindings": preregistration["configuration_bindings"],
        "paths": {
            "exporter": implementations["exporter"]["path"],
            "export_manifest": contract["export_manifest"],
            "pure_clock": contract["pure_clock"],
        },
        "export_manifest_hash": manifest["manifest_hash"],
        "clock_metadata": {
            "schema": clock["schema"],
            "compression": clock["compression"],
            "gzip_mtime": clock["gzip_mtime"],
            "rows": clock["rows"],
            "rows_by_candidate": clock["rows_by_candidate"],
        },
        "required_bindings": required_bindings,
        "clock_rows_read": 0,
    }


def _legacy_comparator_bindings(master: Mapping[str, Any]) -> dict[str, Any]:
    candidate_map = cast(
        Mapping[str, Mapping[str, Any]], master["comparator_candidate_map"]
    )
    provenance = cast(
        Mapping[str, Mapping[str, Any]], master["comparator_provenance_bindings"]
    )
    contract = cast(Mapping[str, Any], master["policy"])["comparator_contract"]
    result: dict[str, Any] = {}
    for family, binding_name, reader_key in (
        ("ccpr", "ccpr_source_clock", "ccpr_reader_columns"),
        ("dlpd", "dlpd_source_clock", "dlpd_reader_columns"),
    ):
        members = {
            candidate_id: definition
            for candidate_id, definition in candidate_map.items()
            if definition["family"] == family
        }
        source = provenance[binding_name]
        result[family] = {
            "binding_mode": "frozen_legacy_source_projection",
            "generated_export_requirement_applicable": False,
            "clock": {"path": source["path"], "sha256": source["sha256"]},
            "reader_column_allowlist": contract[reader_key],
            "member_count": len(members),
            "candidate_map_sha256": clock_common.candidate_map_hash(members),
            "coverage_validation": (
                "deferred to the authorized source-only novelty reader because "
                "the preregistered legacy artifact is wider than six columns"
            ),
            "clock_rows_read": 0,
        }
    return result


def build_freeze(*, hash_file: HashFile = clock_common.sha256_file) -> dict[str, Any]:
    cache: dict[str, str] = {}
    master, master_sha256 = _read_hash_bound_json(
        export_prereg.MASTER_PREREGISTRATION,
        hash_file=hash_file,
    )
    if master_sha256 != export_prereg.MASTER_PREREGISTRATION_SHA256:
        raise RuntimeError("CCHR master preregistration SHA-256 drift")
    cchr.validate_manifest(
        master,
        verify_sources=False,
        expected_output=export_prereg.MASTER_PREREGISTRATION,
    )
    _verify_master_bindings(master, hash_file=hash_file, cache=cache)

    expected_preregistrations = export_prereg.build_all_manifests()
    generated: dict[str, Any] = {}
    generated_ids: set[str] = set()
    master_map = cast(
        Mapping[str, Mapping[str, Any]], master["comparator_candidate_map"]
    )
    for family in export_prereg.FAMILIES:
        preregistration_path = export_prereg.PREREGISTRATION_OUTPUTS[family]
        preregistration, preregistration_sha256 = _read_hash_bound_json(
            preregistration_path,
            hash_file=hash_file,
        )
        if preregistration != expected_preregistrations[family]:
            raise RuntimeError(f"{family} CCHR export preregistration drift")

        family_map = {
            candidate_id: definition
            for candidate_id, definition in master_map.items()
            if definition["family"] == family
        }
        if preregistration["candidate_map"] != family_map:
            raise RuntimeError(f"{family} CCHR candidate map drift from master")
        generated_ids.update(family_map)

        implementations = cast(
            Mapping[str, Mapping[str, Any]],
            preregistration["implementation_bindings"],
        )
        for binding in implementations.values():
            _checked_hash(
                str(binding["path"]),
                str(binding["sha256"]),
                hash_file=hash_file,
                cache=cache,
            )
        for key in ("raw_input_bindings", "configuration_bindings"):
            bindings = cast(Mapping[str, Mapping[str, Any]], preregistration[key])
            for binding in bindings.values():
                _checked_hash(
                    str(binding["path"]),
                    str(binding["sha256"]),
                    hash_file=hash_file,
                    cache=cache,
                )

        contract = cast(Mapping[str, Any], preregistration["output_contract"])
        manifest_path = str(contract["export_manifest"])
        manifest, manifest_sha256 = _read_hash_bound_json(
            manifest_path,
            hash_file=hash_file,
        )
        clock_sha256 = hash_file(str(contract["pure_clock"]))
        _validate_export_linkage(
            family,
            preregistration,
            preregistration_sha256,
            manifest,
            manifest_sha256,
            clock_sha256,
        )
        generated[family] = _family_binding(
            family,
            preregistration,
            preregistration_sha256,
            manifest,
            manifest_sha256,
        )

    legacy = _legacy_comparator_bindings(master)
    legacy_ids = {
        candidate_id
        for candidate_id, definition in master_map.items()
        if definition["family"] in legacy
    }
    if generated_ids & legacy_ids or generated_ids | legacy_ids != set(master_map):
        raise RuntimeError("CCHR comparator family partition is incomplete")

    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "as_of_date": AS_OF_DATE,
        "master_preregistration": {
            "path": str(export_prereg.MASTER_PREREGISTRATION),
            "sha256": master_sha256,
            "manifest_hash": master["manifest_hash"],
        },
        "freeze_requirement": master["comparator_freeze_requirement"],
        "freeze_implementation": {
            "path": str(FREEZE_SOURCE),
            "sha256": hash_file(FREEZE_SOURCE),
        },
        "source_binding": master["source_binding"],
        "comparator_provenance_bindings": master["comparator_provenance_bindings"],
        "comparator_candidate_map_sha256": master["comparator_candidate_map_hash"],
        "comparator_member_count": len(master_map),
        "generated_families": generated,
        "legacy_comparators": legacy,
        "authorization": dict(AUTHORIZATION),
        "outcomes_opened": False,
        "outcome_boundary": dict(OUTCOME_BOUNDARY),
    }
    payload["manifest_hash"] = clock_common.canonical_hash(payload)
    validate_freeze(payload, verify_files=False)
    return payload


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_path_sha_binding(label: str, binding: object) -> None:
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise RuntimeError(f"{label} CCHR path/SHA binding schema drift")
    if not isinstance(binding["path"], str) or not _is_sha256(binding["sha256"]):
        raise RuntimeError(f"{label} CCHR path/SHA binding is invalid")


def _validate_generated_binding_shape(family: str, binding: object) -> int:
    if not isinstance(binding, dict) or set(binding) != {
        "preregistration",
        "implementation_bindings",
        "configuration_bindings",
        "paths",
        "export_manifest_hash",
        "clock_metadata",
        "required_bindings",
        "clock_rows_read",
    }:
        raise RuntimeError(f"{family} CCHR generated-family schema drift")
    if binding["clock_rows_read"] != 0:
        raise RuntimeError(f"{family} CCHR clock rows were opened")

    registration = binding["preregistration"]
    if not isinstance(registration, dict) or set(registration) != {
        "path",
        "sha256",
        "manifest_hash",
    }:
        raise RuntimeError(f"{family} CCHR preregistration binding schema drift")
    if registration["path"] != str(export_prereg.PREREGISTRATION_OUTPUTS[family]):
        raise RuntimeError(f"{family} CCHR preregistration path drift")
    if not _is_sha256(registration["sha256"]) or not _is_sha256(
        registration["manifest_hash"]
    ):
        raise RuntimeError(f"{family} CCHR preregistration hash is invalid")

    implementations = binding["implementation_bindings"]
    if not isinstance(implementations, dict) or set(implementations) != {
        "common",
        "exporter",
        "runner",
    }:
        raise RuntimeError(f"{family} CCHR implementation binding drift")
    for name, item in implementations.items():
        _validate_path_sha_binding(f"{family} {name}", item)
    if implementations["exporter"]["path"] != str(
        export_prereg.EXPORTER_SOURCES[family]
    ):
        raise RuntimeError(f"{family} CCHR exporter path drift")

    configurations = binding["configuration_bindings"]
    if not isinstance(configurations, dict):
        raise RuntimeError(f"{family} CCHR configuration binding drift")
    for name, item in configurations.items():
        _validate_path_sha_binding(f"{family} {name} configuration", item)

    paths = binding["paths"]
    requirement = cchr.PURE_CLOCK_REQUIREMENTS[family]
    if paths != {
        "exporter": str(export_prereg.EXPORTER_SOURCES[family]),
        "export_manifest": requirement["export_manifest"],
        "pure_clock": requirement["path"],
    }:
        raise RuntimeError(f"{family} CCHR frozen path drift")
    if not _is_sha256(binding["export_manifest_hash"]):
        raise RuntimeError(f"{family} CCHR export manifest hash is invalid")

    required = binding["required_bindings"]
    if not isinstance(required, dict) or set(required) != REQUIRED_FAMILY_BINDING_KEYS:
        raise RuntimeError(f"{family} CCHR required freeze binding drift")
    for key in (
        "exporter_sha256",
        "export_manifest_sha256",
        "pure_clock_sha256",
        "candidate_map_sha256",
    ):
        if not _is_sha256(required[key]):
            raise RuntimeError(f"{family} CCHR {key} is invalid")
    if required["exporter_sha256"] != implementations["exporter"]["sha256"]:
        raise RuntimeError(f"{family} CCHR exporter SHA linkage drift")

    family_map = {
        candidate_id: definition
        for candidate_id, definition in cchr.comparator_candidate_map().items()
        if definition["family"] == family
    }
    expected_member_count = int(requirement["required_member_count"])
    if len(family_map) != expected_member_count:
        raise RuntimeError(f"{family} CCHR master member-count drift")
    if required["member_count"] != expected_member_count:
        raise RuntimeError(f"{family} CCHR frozen member-count drift")
    if required["candidate_map_sha256"] != clock_common.candidate_map_hash(family_map):
        raise RuntimeError(f"{family} CCHR frozen candidate-map drift")

    raw_inputs = required["raw_input_path_sha256_and_column_allowlist"]
    if not isinstance(raw_inputs, dict) or not raw_inputs:
        raise RuntimeError(f"{family} CCHR raw-input binding is missing")
    for name, item in raw_inputs.items():
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "columns"}:
            raise RuntimeError(f"{family} {name} CCHR raw-input schema drift")
        columns = item["columns"]
        if (
            not isinstance(item["path"], str)
            or not _is_sha256(item["sha256"])
            or not isinstance(columns, list)
            or not columns
            or any(not isinstance(column, str) or not column for column in columns)
            or len(columns) != len(set(columns))
        ):
            raise RuntimeError(f"{family} {name} CCHR raw-input binding is invalid")

    metadata = binding["clock_metadata"]
    if not isinstance(metadata, dict):
        raise RuntimeError(f"{family} CCHR clock metadata is invalid")
    synthetic_clock = {
        "path": paths["pure_clock"],
        "sha256": required["pure_clock_sha256"],
        **metadata,
        "coverage": required["coverage"],
    }
    _validate_clock_metadata(
        family,
        synthetic_clock,
        {
            "output_contract": {"pure_clock": paths["pure_clock"]},
            "candidate_map": family_map,
        },
    )
    return expected_member_count


def _validate_legacy_binding_shape(family: str, binding: object) -> int:
    if not isinstance(binding, dict) or set(binding) != {
        "binding_mode",
        "generated_export_requirement_applicable",
        "clock",
        "reader_column_allowlist",
        "member_count",
        "candidate_map_sha256",
        "coverage_validation",
        "clock_rows_read",
    }:
        raise RuntimeError(f"{family} CCHR legacy binding schema drift")
    if binding["binding_mode"] != "frozen_legacy_source_projection":
        raise RuntimeError(f"{family} CCHR legacy binding mode drift")
    if binding["generated_export_requirement_applicable"] is not False:
        raise RuntimeError(f"{family} CCHR generated-export scope drift")
    if binding["clock_rows_read"] != 0:
        raise RuntimeError(f"{family} CCHR legacy comparator rows were opened")

    provenance_name = f"{family}_source_clock"
    expected_clock = cchr.COMPARATOR_PROVENANCE_BINDINGS[provenance_name]
    if binding["clock"] != {
        "path": expected_clock["path"],
        "sha256": expected_clock["sha256"],
    }:
        raise RuntimeError(f"{family} CCHR legacy clock binding drift")
    reader_key = f"{family}_reader_columns"
    if (
        binding["reader_column_allowlist"]
        != cchr.policy()["comparator_contract"][reader_key]
    ):
        raise RuntimeError(f"{family} CCHR legacy reader allowlist drift")
    members = {
        candidate_id: definition
        for candidate_id, definition in cchr.comparator_candidate_map().items()
        if definition["family"] == family
    }
    if binding["member_count"] != len(members):
        raise RuntimeError(f"{family} CCHR legacy member-count drift")
    if binding["candidate_map_sha256"] != clock_common.candidate_map_hash(members):
        raise RuntimeError(f"{family} CCHR legacy candidate-map drift")
    if not isinstance(binding["coverage_validation"], str):
        raise RuntimeError(f"{family} CCHR legacy coverage contract drift")
    return len(members)


def validate_freeze(
    payload: Mapping[str, Any],
    *,
    verify_files: bool = True,
) -> None:
    """Validate structure, and exact bound files unless explicitly disabled.

    ``verify_files=False`` is an internal construction check only.  It is not
    an authorization surface; every published or loaded freeze uses the exact
    deterministic ``build_freeze`` comparison below.
    """
    if frozenset(payload) != TOP_LEVEL_KEYS:
        raise RuntimeError("CCHR comparator freeze top-level schema drift")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("CCHR comparator freeze protocol drift")
    if payload.get("policy_id") != POLICY_ID:
        raise RuntimeError("CCHR comparator freeze policy drift")
    if payload.get("as_of_date") != AS_OF_DATE:
        raise RuntimeError("CCHR comparator freeze as-of date drift")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != clock_common.canonical_hash(core):
        raise RuntimeError("CCHR comparator freeze manifest hash mismatch")
    if payload.get("freeze_requirement") != cchr.comparator_freeze_requirement():
        raise RuntimeError("CCHR comparator freeze requirement drift")
    master = payload.get("master_preregistration")
    if not isinstance(master, dict) or set(master) != {
        "path",
        "sha256",
        "manifest_hash",
    }:
        raise RuntimeError("CCHR master preregistration binding schema drift")
    if (
        master["path"] != str(export_prereg.MASTER_PREREGISTRATION)
        or master["sha256"] != export_prereg.MASTER_PREREGISTRATION_SHA256
    ):
        raise RuntimeError("CCHR master preregistration binding drift")
    if not _is_sha256(master["manifest_hash"]):
        raise RuntimeError("CCHR master preregistration manifest hash is invalid")
    freeze_implementation = payload.get("freeze_implementation")
    _validate_path_sha_binding("freeze implementation", freeze_implementation)
    freeze_binding = cast(Mapping[str, Any], freeze_implementation)
    if freeze_binding["path"] != str(FREEZE_SOURCE):
        raise RuntimeError("CCHR freeze implementation path drift")
    if payload.get("source_binding") != cchr.expected_source_binding():
        raise RuntimeError("CCHR comparator freeze source binding drift")
    if payload.get("comparator_provenance_bindings") != (
        cchr.COMPARATOR_PROVENANCE_BINDINGS
    ):
        raise RuntimeError("CCHR comparator provenance binding drift")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("CCHR comparator freeze opened outcomes")
    if payload.get("outcome_boundary") != OUTCOME_BOUNDARY:
        raise RuntimeError("CCHR comparator freeze outcome boundary drift")
    if payload.get("authorization") != AUTHORIZATION:
        raise RuntimeError("CCHR comparator freeze authorization drift")
    generated = payload.get("generated_families")
    if not isinstance(generated, dict) or set(generated) != set(export_prereg.FAMILIES):
        raise RuntimeError("CCHR comparator freeze generated-family drift")
    generated_member_count = sum(
        _validate_generated_binding_shape(family, binding)
        for family, binding in generated.items()
    )
    legacy = payload.get("legacy_comparators")
    if not isinstance(legacy, dict) or set(legacy) != {"ccpr", "dlpd"}:
        raise RuntimeError("CCHR legacy comparator freeze drift")
    legacy_member_count = sum(
        _validate_legacy_binding_shape(family, binding)
        for family, binding in legacy.items()
    )
    if payload.get("comparator_member_count") != len(cchr.comparator_candidate_map()):
        raise RuntimeError("CCHR comparator freeze member-count drift")
    if (
        generated_member_count + legacy_member_count
        != payload["comparator_member_count"]
    ):
        raise RuntimeError("CCHR comparator freeze family-member total drift")
    if payload.get(
        "comparator_candidate_map_sha256"
    ) != clock_common.candidate_map_hash(cchr.comparator_candidate_map()):
        raise RuntimeError("CCHR comparator freeze candidate-map drift")
    if verify_files and payload != build_freeze():
        raise RuntimeError("CCHR comparator freeze file binding drift")


def _validated_output_target() -> Path:
    target = _repository_path(DEFAULT_OUTPUT)
    results_root = _repository_path(RESULTS_ROOT)
    if target.name != Path(DEFAULT_OUTPUT).name or target.suffix != ".json":
        raise ValueError("CCHR comparator freeze output path drift")
    if results_root.is_symlink():
        raise ValueError("CCHR comparator freeze results root cannot be a symlink")
    current = target
    while True:
        if current.is_symlink():
            raise ValueError("CCHR comparator freeze path cannot contain a symlink")
        if current == results_root or current.parent == current:
            break
        current = current.parent
    resolved_root = results_root.resolve()
    resolved_target = target.resolve()
    if resolved_target.parent != resolved_root:
        raise ValueError("CCHR comparator freeze must be a direct child of results")
    protected = {
        _repository_path(export_prereg.MASTER_PREREGISTRATION).resolve(),
        _repository_path(FREEZE_SOURCE).resolve(),
    }
    if resolved_target in protected:
        raise ValueError("CCHR comparator freeze output aliases a protected input")
    return resolved_target


def _directory_identity(path: Path) -> DirectoryIdentity:
    status = os.stat(path, follow_symlinks=False)
    return status.st_dev, status.st_ino


def _publish_json_create_only(
    payload: Mapping[str, Any],
    target: Path,
    *,
    expected_directory_identity: DirectoryIdentity | None = None,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if expected_directory_identity is None:
        expected_directory_identity = _directory_identity(target.parent)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(target.parent, flags)
    temporary_name: str | None = None
    linked = False
    try:
        opened_status = os.fstat(directory_fd)
        if (opened_status.st_dev, opened_status.st_ino) != (
            expected_directory_identity
        ):
            raise RuntimeError("CCHR results directory identity changed")
        if not os.path.samestat(
            opened_status,
            os.stat(target.parent, follow_symlinks=False),
        ):
            raise RuntimeError("CCHR results directory changed before publication")
        create_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor: int | None = None
        for _ in range(128):
            candidate = f".{target.name}.{secrets.token_hex(12)}.json.tmp"
            try:
                descriptor = os.open(
                    candidate,
                    create_flags,
                    0o600,
                    dir_fd=directory_fd,
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        if descriptor is None or temporary_name is None:
            raise RuntimeError("cannot allocate CCHR freeze temporary file")

        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fchmod(handle.fileno(), 0o644)
            os.fsync(handle.fileno())
        try:
            os.link(
                temporary_name,
                target.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            linked = True
            os.fsync(directory_fd)
            if not os.path.samestat(
                os.fstat(directory_fd),
                os.stat(target.parent, follow_symlinks=False),
            ):
                raise RuntimeError(
                    "CCHR results directory changed during freeze publication"
                )
        except FileExistsError as error:
            raise FileExistsError("CCHR comparator freeze is immutable") from error
        except BaseException:
            if linked:
                try:
                    target_stat = os.stat(
                        target.name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                    temporary_stat = os.stat(
                        temporary_name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                    if os.path.samestat(target_stat, temporary_stat):
                        os.unlink(target.name, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
            raise
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)


def run() -> dict[str, Any]:
    target = _validated_output_target()
    expected_directory_identity = _directory_identity(target.parent)
    if target.exists():
        raise FileExistsError("CCHR comparator freeze is immutable")
    payload = build_freeze()
    validate_freeze(payload, verify_files=True)
    target = _validated_output_target()
    _publish_json_create_only(
        payload,
        target,
        expected_directory_identity=expected_directory_identity,
    )
    if _read_json(target) != payload:
        raise RuntimeError("published CCHR comparator freeze bytes are invalid")
    return payload


def load_freeze(path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if _repository_path(path) != _validated_output_target():
        raise ValueError("CCHR comparator freeze load path drift")
    payload = _read_json(path)
    validate_freeze(payload, verify_files=True)
    return payload


def main() -> None:
    payload = run()
    print(
        json.dumps(
            {
                "path": str(DEFAULT_OUTPUT),
                "manifest_hash": payload["manifest_hash"],
                "comparator_member_count": payload["comparator_member_count"],
                "outcomes_opened": payload["outcomes_opened"],
                "authorization": payload["authorization"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
