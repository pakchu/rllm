"""Freeze hash-only preregistrations for the four CCHR comparator exports.

This stage hashes raw source bytes, implementation bytes, and frozen config
bytes.  It never parses source CSV rows, outcome artifacts, returns, PnL, or
post-2023 values.  A successfully written family artifact authorizes only the
corresponding source-only six-column clock export; outcome evaluation remains
forbidden until the later combined comparator freeze is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

from training import cchr_comparator_clock_common as clock_common
from training import export_cchr_dtv_pure_clocks as dtv
from training import export_cchr_far_pure_clocks as far
from training import export_cchr_live_portfolio_pure_clocks as live
from training import export_cchr_pdlh_pure_clocks as pdlh
from training import preregister_cross_collateral_cohort_handoff_relay as cchr


PROTOCOL_VERSION = "cchr_pure_clock_export_preregistration_v3"
POLICY_ID = "CCHR-288"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPOSITORY_ROOT / "results"
PREREGISTRATION_SOURCE = Path("training/preregister_cchr_pure_clock_exports.py")
COMMON_SOURCE = Path("training/cchr_comparator_clock_common.py")
RUNNER_SOURCE = Path("training/export_cchr_pure_clocks.py")
MASTER_PREREGISTRATION = Path(
    "results/cross_collateral_cohort_handoff_relay_preregistration_2026-07-21.json"
)
MASTER_PREREGISTRATION_SHA256 = (
    "b3792b0f24c4e0e022903c359db143337b0c998034600480edbc29095d039056"
)
MASTER_SOURCE = Path("training/preregister_cross_collateral_cohort_handoff_relay.py")
MASTER_SOURCE_SHA256 = (
    "797dffd052fd9b5f922ad06780c21c94815ee3403212b38d12405076382c800f"
)

FAMILIES = ("pdlh", "dtv", "far", "live")
EXPORTER_SOURCES = {
    "pdlh": Path("training/export_cchr_pdlh_pure_clocks.py"),
    "dtv": Path("training/export_cchr_dtv_pure_clocks.py"),
    "far": Path("training/export_cchr_far_pure_clocks.py"),
    "live": Path("training/export_cchr_live_portfolio_pure_clocks.py"),
}
PREREGISTRATION_OUTPUTS = {
    "pdlh": Path("results/cchr_pdlh_pure_clock_preregistration_v3_2026-07-21.json"),
    "dtv": Path("results/cchr_dtv_pure_clock_preregistration_v3_2026-07-21.json"),
    "far": Path("results/cchr_far_pure_clock_preregistration_v3_2026-07-21.json"),
    "live": Path(
        "results/cchr_live_portfolio_pure_clock_preregistration_v3_2026-07-21.json"
    ),
}

EXPORT_MANIFEST_PROTOCOL_VERSION = "cchr_pure_clock_export_manifest_v1"
EXPORT_MANIFEST_TOP_LEVEL_KEYS = (
    "protocol_version",
    "policy_id",
    "family",
    "preregistration",
    "implementation_bindings",
    "raw_input_bindings",
    "configuration_bindings",
    "candidate_map_sha256",
    "member_count",
    "clock",
    "export_manifest",
    "authorization",
    "outcomes_opened",
    "outcome_boundary",
    "manifest_hash",
)

CAUSAL_ORIGINS = {
    "pdlh": "positioning lifecycle episode start",
    "dtv": "signal time",
    "far": "oldest live eligible cohort settlement; signal-time fallback",
    "live": "signal time",
}
THRESHOLD_CONTRACTS = {
    "pdlh": {
        "kind": "fixed lifecycle constants",
        "fit": "prior-only rolling z-score; no outcome fit",
    },
    "dtv": {
        "kind": "positive-score quantile",
        "fit": "[2020-10-15T00:00:00Z,2023-01-01T00:00:00Z)",
        "quantiles": [0.90, 0.95],
        "minimum_positive_observations": 1_000,
    },
    "far": {
        "kind": "absolute-score quantile",
        "fit": "[2020-10-15T00:00:00Z,2023-01-01T00:00:00Z)",
        "quantiles": [0.90],
        "minimum_observations": 10_000,
    },
    "live": {
        "kind": "frozen numeric config thresholds",
        "fit": "bound by config and exporter hashes; no refit during export",
    },
}

OUTCOME_BOUNDARY = {
    "raw_input_bytes_hashed": True,
    "raw_input_rows_parsed": 0,
    "source_csv_values_read": 0,
    "source_csv_decompressed": 0,
    "outcome_artifacts_parsed": 0,
    "comparator_rows_read": 0,
    "market_rows_loaded": 0,
    "funding_rows_loaded": 0,
    "post_2023_rows_loaded": 0,
    "return_or_pnl_fields_read": 0,
    "signal_incidence_rows_derived": 0,
    "network_calls": 0,
    "subprocess_calls": 0,
}

TOP_LEVEL_KEYS = frozenset(
    {
        "protocol_version",
        "policy_id",
        "family",
        "master_preregistration",
        "preregistration_source",
        "implementation_bindings",
        "raw_input_bindings",
        "configuration_bindings",
        "candidate_map",
        "candidate_map_sha256",
        "member_count",
        "research_splits",
        "clock_contract",
        "threshold_contract",
        "output_contract",
        "authorization",
        "outcomes_opened",
        "outcome_boundary",
        "manifest_hash",
    }
)

HashFile = Callable[[str | Path], str]


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return candidate.resolve()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with repository_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_map(family: str) -> dict[str, dict[str, Any]]:
    if family == "pdlh":
        return pdlh.pdlh_candidate_map()
    if family == "dtv":
        return dtv.comparator_candidate_map()
    if family == "far":
        return far.far_candidate_map()
    if family == "live":
        return live.candidate_map()
    raise KeyError(f"unknown CCHR comparator family: {family}")


def _normalise_bindings(
    bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "path": str(binding["path"]),
            "sha256": str(binding["sha256"]),
            "columns": list(binding["columns"]),
        }
        for name, binding in sorted(bindings.items())
    }


def _raw_input_bindings(family: str) -> dict[str, dict[str, Any]]:
    if family == "pdlh":
        return _normalise_bindings(
            {
                "market": {
                    "path": pdlh.MARKET_PATH,
                    "sha256": pdlh.MARKET_SHA256,
                    "columns": pdlh.MARKET_COLUMNS,
                },
                "metrics": {
                    "path": pdlh.METRICS_PATH,
                    "sha256": pdlh.METRICS_SHA256,
                    "columns": pdlh.METRICS_COLUMNS,
                },
            }
        )
    if family == "dtv":
        return _normalise_bindings(dtv.source_bindings())
    if family == "far":
        return _normalise_bindings(
            {
                "market": {
                    "path": far.MARKET_PATH,
                    "sha256": far.MARKET_SHA256,
                    "columns": far.MARKET_COLUMNS,
                },
                "metrics": {
                    "path": far.METRICS_PATH,
                    "sha256": far.METRICS_SHA256,
                    "columns": far.METRICS_COLUMNS,
                },
                "funding": {
                    "path": far.FUNDING_PATH,
                    "sha256": far.FUNDING_SHA256,
                    "columns": far.FUNDING_COLUMNS,
                },
            }
        )
    if family == "live":
        return _normalise_bindings(live.input_bindings())
    raise KeyError(f"unknown CCHR comparator family: {family}")


def _configuration_bindings(family: str) -> dict[str, dict[str, str]]:
    if family != "live":
        return {}
    return {
        name: {"path": path, "sha256": digest}
        for name, (path, digest) in {
            "portfolio": (
                str(live.PORTFOLIO_CONFIG_PATH),
                live.PORTFOLIO_CONFIG_SHA256,
            ),
            "oi": (str(live.OI_CONFIG_PATH), live.OI_CONFIG_SHA256),
            "funding_premium": (
                str(live.FUNDING_CONFIG_PATH),
                live.FUNDING_CONFIG_SHA256,
            ),
            "rex": (str(live.REX_CONFIG_PATH), live.REX_CONFIG_SHA256),
        }.items()
    }


def _research_splits() -> list[dict[str, str]]:
    return [
        {
            "name": split.name,
            "start_inclusive": clock_common.format_utc(split.start),
            "end_exclusive": clock_common.format_utc(split.end),
        }
        for split in clock_common.research_splits()
    ]


def _output_contract(family: str) -> dict[str, Any]:
    requirement = cchr.PURE_CLOCK_REQUIREMENTS[family]
    return {
        "preregistration": str(PREREGISTRATION_OUTPUTS[family]),
        "pure_clock": str(requirement["path"]),
        "export_manifest": str(requirement["export_manifest"]),
        "combined_freeze": str(cchr.COMPARATOR_FREEZE),
        "clock_format": {
            "schema": list(clock_common.CLOCK_COLUMNS),
            "encoding": "utf-8",
            "line_ending": "LF",
            "compression": "gzip",
            "gzip_mtime": 0,
            "ordering": ["candidate_id", "entry_time"],
            "all_candidate_ids_required": True,
        },
        "export_manifest_contract": {
            "protocol_version": EXPORT_MANIFEST_PROTOCOL_VERSION,
            "required_top_level_keys": list(EXPORT_MANIFEST_TOP_LEVEL_KEYS),
            "manifest_hash": "sha256 canonical JSON excluding manifest_hash",
        },
        "publication": "create-only clock and manifest pair; no overwrite",
    }


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
    actual = cache[key]
    if actual != expected:
        raise RuntimeError(f"frozen CCHR export binding drifted: {path}")
    return actual


def _build_family_manifest(
    family: str,
    *,
    hash_file: HashFile,
    cache: dict[str, str],
) -> dict[str, Any]:
    if family not in FAMILIES:
        raise KeyError(f"unknown CCHR comparator family: {family}")
    _checked_hash(
        MASTER_PREREGISTRATION,
        MASTER_PREREGISTRATION_SHA256,
        hash_file=hash_file,
        cache=cache,
    )
    _checked_hash(
        MASTER_SOURCE,
        MASTER_SOURCE_SHA256,
        hash_file=hash_file,
        cache=cache,
    )

    raw_bindings = _raw_input_bindings(family)
    for binding in raw_bindings.values():
        _checked_hash(
            binding["path"],
            binding["sha256"],
            hash_file=hash_file,
            cache=cache,
        )
    config_bindings = _configuration_bindings(family)
    for binding in config_bindings.values():
        _checked_hash(
            binding["path"],
            binding["sha256"],
            hash_file=hash_file,
            cache=cache,
        )

    members = _candidate_map(family)
    frozen_members = {
        candidate_id: definition
        for candidate_id, definition in cchr.comparator_candidate_map().items()
        if definition["family"] == family
    }
    if members != frozen_members:
        raise RuntimeError(f"{family} exporter candidate map drifted from CCHR")
    requirement = cchr.PURE_CLOCK_REQUIREMENTS[family]
    if len(members) != int(requirement["required_member_count"]):
        raise RuntimeError(f"{family} exporter member count drifted from CCHR")

    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "family": family,
        "master_preregistration": {
            "path": str(MASTER_PREREGISTRATION),
            "sha256": MASTER_PREREGISTRATION_SHA256,
            "source_path": str(MASTER_SOURCE),
            "source_sha256": MASTER_SOURCE_SHA256,
        },
        "preregistration_source": {
            "path": str(PREREGISTRATION_SOURCE),
            "sha256": hash_file(PREREGISTRATION_SOURCE),
        },
        "implementation_bindings": {
            "common": {
                "path": str(COMMON_SOURCE),
                "sha256": hash_file(COMMON_SOURCE),
            },
            "exporter": {
                "path": str(EXPORTER_SOURCES[family]),
                "sha256": hash_file(EXPORTER_SOURCES[family]),
            },
            "runner": {
                "path": str(RUNNER_SOURCE),
                "sha256": hash_file(RUNNER_SOURCE),
            },
        },
        "raw_input_bindings": raw_bindings,
        "configuration_bindings": config_bindings,
        "candidate_map": members,
        "candidate_map_sha256": clock_common.candidate_map_hash(members),
        "member_count": len(members),
        "research_splits": _research_splits(),
        "clock_contract": {
            "schema": list(clock_common.CLOCK_COLUMNS),
            "source_end_exclusive": "2024-01-01T00:00:00Z",
            "prefix_loading": (
                "hash full compressed bytes first; boundary stream isolates each "
                "timestamp field before pandas and stops without materializing the "
                "first sealed row"
            ),
            "signal_bar": "completed five-minute bar identified by open timestamp",
            "decision_delay_bars": 1,
            "entry_delay_bars": 1,
            "exit": "fixed hold_bars after entry; exit boundary must exist",
            "interval": "[entry_time,exit_time)",
            "causal_origin": CAUSAL_ORIGINS[family],
            "scheduling": (
                "split containment first, then non-overlap independently per "
                "candidate_id with entry_time >= prior_exit accepted"
            ),
        },
        "threshold_contract": THRESHOLD_CONTRACTS[family],
        "output_contract": _output_contract(family),
        "authorization": {
            "source_only_clock_export_after_this_artifact": True,
            "outcome_evaluator": False,
            "post_2023_source_access": False,
            "network_access": False,
        },
        "outcomes_opened": False,
        "outcome_boundary": dict(OUTCOME_BOUNDARY),
    }
    payload["manifest_hash"] = clock_common.canonical_hash(payload)
    return payload


def build_family_manifest(family: str) -> dict[str, Any]:
    return _build_family_manifest(family, hash_file=sha256_file, cache={})


def build_all_manifests() -> dict[str, dict[str, Any]]:
    cache: dict[str, str] = {}
    return {
        family: _build_family_manifest(
            family,
            hash_file=sha256_file,
            cache=cache,
        )
        for family in FAMILIES
    }


def validate_manifest(
    payload: Mapping[str, Any],
    *,
    verify_files: bool = True,
) -> None:
    if frozenset(payload) != TOP_LEVEL_KEYS:
        raise RuntimeError("CCHR export preregistration top-level schema drift")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("CCHR export preregistration protocol drift")
    if payload.get("policy_id") != POLICY_ID:
        raise RuntimeError("CCHR export preregistration policy drift")
    family = str(payload.get("family"))
    if family not in FAMILIES:
        raise RuntimeError("CCHR export preregistration family drift")
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != clock_common.canonical_hash(core):
        raise RuntimeError("CCHR export preregistration manifest hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("CCHR export preregistration opened outcomes")
    if payload.get("outcome_boundary") != OUTCOME_BOUNDARY:
        raise RuntimeError("CCHR export preregistration outcome boundary drift")
    if payload.get("candidate_map") != _candidate_map(family):
        raise RuntimeError("CCHR export preregistration candidate map drift")
    if payload.get("candidate_map_sha256") != clock_common.candidate_map_hash(
        _candidate_map(family)
    ):
        raise RuntimeError("CCHR export preregistration candidate-map hash drift")
    if payload.get("raw_input_bindings") != _raw_input_bindings(family):
        raise RuntimeError("CCHR export preregistration raw-input drift")
    if payload.get("configuration_bindings") != _configuration_bindings(family):
        raise RuntimeError("CCHR export preregistration config drift")
    if payload.get("member_count") != len(_candidate_map(family)):
        raise RuntimeError("CCHR export preregistration member-count drift")
    if payload.get("output_contract") != _output_contract(family):
        raise RuntimeError("CCHR export preregistration output-contract drift")
    authorization = payload.get("authorization")
    if not isinstance(authorization, dict) or authorization != {
        "source_only_clock_export_after_this_artifact": True,
        "outcome_evaluator": False,
        "post_2023_source_access": False,
        "network_access": False,
    }:
        raise RuntimeError("CCHR export preregistration authorization drift")
    if verify_files and payload != build_family_manifest(family):
        raise RuntimeError("CCHR export preregistration frozen binding drift")


def _write_temporary_json(payload: Mapping[str, Any], target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _protected_paths(families: Sequence[str]) -> set[Path]:
    protected = {
        repository_path(MASTER_PREREGISTRATION),
        repository_path(MASTER_SOURCE),
        repository_path(PREREGISTRATION_SOURCE),
        repository_path(COMMON_SOURCE),
        repository_path(RUNNER_SOURCE),
        *(repository_path(EXPORTER_SOURCES[family]) for family in families),
    }
    for family in families:
        protected.update(
            repository_path(binding["path"])
            for binding in _raw_input_bindings(family).values()
        )
        protected.update(
            repository_path(binding["path"])
            for binding in _configuration_bindings(family).values()
        )
    return protected


def _validated_output_target(family: str, families: Sequence[str]) -> Path:
    configured = PREREGISTRATION_OUTPUTS[family]
    raw = configured if configured.is_absolute() else REPOSITORY_ROOT / configured
    raw = Path(os.path.abspath(raw))
    results_root = Path(os.path.abspath(RESULTS_ROOT))
    if raw.suffix != ".json":
        raise ValueError("CCHR export preregistration output must be JSON")
    if results_root.is_symlink():
        raise ValueError("CCHR export preregistration results root cannot be a symlink")
    current = raw
    while True:
        if current.is_symlink():
            raise ValueError(
                "CCHR export preregistration path cannot contain a symlink"
            )
        if current == results_root or current.parent == current:
            break
        current = current.parent
    resolved_root = results_root.resolve()
    resolved = raw.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(
            "CCHR export preregistration output must remain under results"
        ) from error
    if resolved.parent != resolved_root:
        raise ValueError(
            "CCHR export preregistration output must be a direct child of results"
        )
    if resolved in _protected_paths(families):
        raise ValueError("CCHR export preregistration output aliases a protected input")
    return resolved


def _publish_create_only(
    temporary: Path,
    target: Path,
    *,
    directory_fd: int,
) -> None:
    try:
        os.link(
            temporary.name,
            target.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except FileExistsError as error:
        raise FileExistsError("CCHR export preregistration is immutable") from error


def _same_inode_at(directory_fd: int, left: str, right: str) -> bool:
    try:
        return os.path.samestat(
            os.stat(left, dir_fd=directory_fd, follow_symlinks=False),
            os.stat(right, dir_fd=directory_fd, follow_symlinks=False),
        )
    except FileNotFoundError:
        return False


def _unlink_at(directory_fd: int, name: str) -> None:
    try:
        os.unlink(name, dir_fd=directory_fd)
    except FileNotFoundError:
        pass


def write_preregistrations(
    families: Sequence[str] = FAMILIES,
) -> dict[str, dict[str, Any]]:
    selected = tuple(families)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("CCHR export families must be non-empty and unique")
    unknown = sorted(set(selected) - set(FAMILIES))
    if unknown:
        raise KeyError(f"unknown CCHR comparator families: {unknown}")
    targets = {
        family: _validated_output_target(family, selected) for family in selected
    }
    if len(set(targets.values())) != len(targets):
        raise ValueError("CCHR export preregistration outputs must be unique")
    for target in targets.values():
        if target.exists():
            raise FileExistsError("CCHR export preregistration is immutable")

    cache: dict[str, str] = {}
    payloads = {
        family: _build_family_manifest(
            family,
            hash_file=sha256_file,
            cache=cache,
        )
        for family in selected
    }
    for payload in payloads.values():
        validate_manifest(payload, verify_files=False)

    temporary: dict[str, Path] = {}
    directory_fd: int | None = None
    try:
        for family, payload in payloads.items():
            temporary[family] = _write_temporary_json(payload, targets[family])
        output_directory = targets[selected[0]].parent
        if any(target.parent != output_directory for target in targets.values()):
            raise ValueError("CCHR preregistrations require one results directory")
        flags = (
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        directory_fd = os.open(output_directory, flags)
        if not os.path.samestat(
            os.fstat(directory_fd),
            os.stat(output_directory, follow_symlinks=False),
        ):
            raise RuntimeError("CCHR results directory changed before publication")
        for family in selected:
            _publish_create_only(
                temporary[family],
                targets[family],
                directory_fd=directory_fd,
            )
        os.fsync(directory_fd)
    except BaseException:
        if directory_fd is not None:
            for family in reversed(selected):
                staged = temporary.get(family)
                if staged is not None and _same_inode_at(
                    directory_fd, staged.name, targets[family].name
                ):
                    _unlink_at(directory_fd, targets[family].name)
            for staged in temporary.values():
                _unlink_at(directory_fd, staged.name)
        else:
            for path in temporary.values():
                path.unlink(missing_ok=True)
        raise
    finally:
        if directory_fd is not None:
            for staged in temporary.values():
                _unlink_at(directory_fd, staged.name)
            os.close(directory_fd)
    return payloads


def load_preregistration(
    family: str,
    *,
    verify_files: bool = True,
) -> dict[str, Any]:
    payload, _ = load_preregistration_with_sha256(
        family,
        verify_files=verify_files,
    )
    return payload


def load_preregistration_with_sha256(
    family: str,
    *,
    verify_files: bool = True,
) -> tuple[dict[str, Any], str]:
    raw = repository_path(PREREGISTRATION_OUTPUTS[family]).read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError("CCHR export preregistration must be a JSON object")
    validate_manifest(payload, verify_files=verify_files)
    return payload, hashlib.sha256(raw).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        action="append",
        choices=FAMILIES,
        help="write one family; repeat as needed (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    families = tuple(args.family) if args.family else FAMILIES
    payloads = write_preregistrations(families)
    print(
        json.dumps(
            {
                family: {
                    "path": str(PREREGISTRATION_OUTPUTS[family]),
                    "manifest_hash": payload["manifest_hash"],
                    "outcomes_opened": payload["outcomes_opened"],
                }
                for family, payload in payloads.items()
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
