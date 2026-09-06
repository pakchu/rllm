"""Write-once, source-blind preregistration for CVVH-432."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from training import cross_venue_volatility_shape_handoff as mechanism


PROTOCOL_VERSION = (
    "cross_venue_volatility_shape_handoff_preregistration_v1"
)
AS_OF_DATE = "2026-07-30"
EXPECTED_BRANCH = "codex/cvvh-volatility-shape-20260730"
EXPECTED_ORIGIN_URL = "https://github.com/pakchu/rllm.git"
DEFAULT_OUTPUT = Path(
    "results/"
    "cross_venue_volatility_shape_handoff_preregistration_2026-07-30.json"
)

COMMON_WINDOW_POLICY = {
    "path": "docs/novelty-comparator-common-window-policy-2026-07-23.md",
    "sha256": (
        "928bce6e04fb34001478b4b4ea84156580b661c88a0f0338065a891c009bd580"
    ),
}

ESDI_PREREGISTRATION = {
    "path": (
        "results/"
        "ethereum_settlement_demand_impulse_preregistration_2026-07-30.json"
    ),
    "sha256": (
        "2a481fc60044d3d468340457d50f92a91f2a52184a464e1a91badfb418bbcaba"
    ),
    "manifest_hash": (
        "d5279f95cc7b92757aa77ecbbc5835d8b1cc4ce34f5a81d6f279abdcf2fcfe8a"
    ),
}

GROSS9_WEIGHTS = {
    "cand_rex_veto_7": 1.6,
    "fresh_kimchi_fx": 2.0,
    "frozen_annual_rank7": 3.0,
    "markov_transition_long": 2.0,
    "rex_taker_low_range_position": 0.4,
}
GROSS9_AUTHORITY_HASH = (
    "b3490c484d3fda1d5b649498e0d84325e203cd2664086e68cebd76509a54957e"
)

SOURCE_ARTIFACTS: dict[str, dict[str, Any]] = {
    "binance_btc_bvol_hourly": {
        "path": (
            "data/binance_btc_bvol_hourly_opdr_2023_2026/"
            "BTCBVOLUSDT_1h_2023-06-20_2026-06-30.csv.gz"
        ),
        "sha256": (
            "40c0d1aecb15119e7fab31aae4108c632d25de136401a6896896852c7f4032b1"
        ),
        "header": [
            "date",
            "feature_available_time_utc",
            "trade_earliest_time_utc",
            "open",
            "high",
            "low",
            "close",
            "source_rows",
            "source_complete",
            "feature_valid",
            "feature_invalid_reason",
        ],
        "header_line_sha256": (
            "b23c2c08e4856d7939fb0fccae8b524f41533d417fc1fdcef3b77b27536c1f14"
        ),
        "parser": {
            "join_clock": "feature_available_time_utc",
            "candle_open_clock": "date",
            "validity": (
                "source_complete=true AND feature_valid=true AND "
                "source_rows=3600"
            ),
            "ohlc": ["open", "high", "low", "close"],
        },
    },
    "binance_btc_bvol_manifest": {
        "path": (
            "data/binance_btc_bvol_hourly_opdr_2023_2026/build_manifest.json"
        ),
        "sha256": (
            "6c62a389cbc8d6524444f5e5fe1d2945c20bafa9fa707b7f2a4801c74221a7e4"
        ),
    },
    "deribit_btc_dvol_hourly": {
        "path": "data/deribit_btc_dvol_1h_2023-06-20_2026-07-01.csv.gz",
        "sha256": (
            "26b768f81c2fa49fd59d9f1a173a829329a7ed5bb94c2d71af7c33b46f4f02cf"
        ),
        "header": ["date", "close_time", "open", "high", "low", "close"],
        "header_line_sha256": (
            "4c782bb5326dbbdba97b985119bad56d917d6d4d8e3a01110890fcbbe202583a"
        ),
        "parser": {
            "join_clock": "close_time",
            "candle_open_clock": "date",
            "ohlc": ["open", "high", "low", "close"],
            "required_end_filter": "close_time < 2026-07-01T00:00:00Z",
        },
    },
    "deribit_btc_dvol_summary": {
        "path": (
            "data/deribit_btc_dvol_1h_2023-06-20_2026-07-01.csv.gz.summary.json"
        ),
        "sha256": (
            "22e0a6e311fcad34a51f5b0844b7807e7c851eecc4a367f89b7a7d6ce438bf74"
        ),
    },
}

PRIOR_VOLATILITY_COMPARATORS: dict[str, dict[str, Any]] = {
    "OPDR-24": {
        "path": "data/options_perpetual_demand_relay_clocks_2023_2026.csv.gz",
        "sha256": (
            "ceb79b206c3e1f6bf78b02cd2ace9a94f875ce930a704cc6e7a5a8b255021b99"
        ),
        "header": [
            "candidate",
            "control",
            "split",
            "decision_time",
            "feature_available_time",
            "entry_time",
            "exit_time",
            "side",
            "log_bvol_dvol_ratio",
            "premium_move_bp",
            "premium_path_range_bp",
            "premium_efficiency",
            "prior_ratio_q20",
            "prior_ratio_q80",
            "prior_move_abs_q80_bp",
            "prior_efficiency_q70",
        ],
        "header_line_sha256": (
            "d93332440131f5f2dcc169dacf90588b686d1705d147f819028c0fe146561f60"
        ),
        "filters": {"candidate": "OPDR-24", "control": "primary"},
        "entry_column": "entry_time",
        "exit_column": "exit_time",
        "side_column": "side",
        "side_parser": {"1": 1, "-1": -1},
        "common_window": [
            "2023-07-01T00:00:00Z",
            "2026-06-01T00:00:00Z",
        ],
    },
    "old_dvol_price_follow": {
        "path": "data/opdr_old_dvol_price_follow_comparator_2023h2.csv.gz",
        "sha256": (
            "a9c9d1c8d32510e63e604dfdc8b9d079f7e7a4bc206fd0a0197cad8c65b03d3d"
        ),
        "header": ["signal_time", "entry_time", "exit_time", "side"],
        "header_line_sha256": (
            "b1902adacd728da49ffe96210821a10c86d43e05bd1a6c7da43ba6fb31775a39"
        ),
        "filters": {},
        "entry_column": "entry_time",
        "exit_column": "exit_time",
        "side_column": "side",
        "side_parser": {"1": 1, "-1": -1},
        "common_window": [
            "2023-07-01T00:00:00Z",
            "2024-01-01T00:00:00Z",
        ],
    },
    "PSR-30/6": {
        "path": "data/premium_snapback_recenter_clocks_2020_2026.csv.gz",
        "sha256": (
            "cb209ed35f9baa08cc2fb3dd5bd60b8e747b1408c09507b774ca275e0b2b2db6"
        ),
        "header": [
            "candidate",
            "split",
            "path_start_time",
            "decision_time",
            "feature_available_time",
            "entry_time",
            "planned_exit_time",
            "direction",
            "prior_center",
            "path_range",
            "efficiency",
            "turns",
            "up_excursion",
            "down_excursion",
            "max_excursion",
            "terminal_deviation",
        ],
        "header_line_sha256": (
            "3afc1ce5df81815a2bed10b1934e356b1e4487f5d35e9c696fdc5ec29067667c"
        ),
        "filters": {"candidate": "PSR-30/6"},
        "entry_column": "entry_time",
        "exit_column": "planned_exit_time",
        "side_column": "direction",
        "side_parser": {"1": 1, "-1": -1},
        "common_window": [
            "2023-07-01T00:00:00Z",
            "2026-06-01T00:00:00Z",
        ],
    },
    "PCBR-12": {
        "path": (
            "data/premium_compression_breakout_relay_clocks_2020_2026.csv.gz"
        ),
        "sha256": (
            "659fc1b6b6e3a20e60031ed1d50f51c8c7d2836956f911f62ad13e4152740cda"
        ),
        "header": [
            "candidate",
            "control",
            "split",
            "context_start_time",
            "decision_time",
            "feature_available_time",
            "entry_time",
            "exit_time",
            "side",
            "context_range",
            "trigger_move",
            "trigger_efficiency",
            "terminal_location",
            "outside_distance",
        ],
        "header_line_sha256": (
            "fec5bab4efda93d624dac7dfa65abf5a58c173ad60b56114bb7931403bd6d193"
        ),
        "filters": {"candidate": "PCBR-12", "control": "primary"},
        "entry_column": "entry_time",
        "exit_column": "exit_time",
        "side_column": "side",
        "side_parser": {"1": 1, "-1": -1},
        "common_window": [
            "2023-07-01T00:00:00Z",
            "2026-06-01T00:00:00Z",
        ],
    },
    "CMSR-36": {
        "path": (
            "data/coinm_next_maturity_shock_relay_clocks_2020_2023.csv.gz"
        ),
        "sha256": (
            "e81450d4e76ffd0ce2ae96edf97106f2f4c473da233be0db18dc2530c8da8e87"
        ),
        "header": [
            "control",
            "signal_time",
            "feature_available_time",
            "entry_time",
            "exit_time",
            "side",
            "pair",
            "next_share_slope",
            "next_flow",
            "next_return",
            "front_return",
            "next_lead_shock",
        ],
        "header_line_sha256": (
            "28703a32b4c55e4e912f54c1bddea62c8c49c658d73d42d3319f491e1e40c365"
        ),
        "filters": {"control": "primary"},
        "entry_column": "entry_time",
        "exit_column": "exit_time",
        "side_column": "side",
        "side_parser": {"1": 1, "-1": -1},
        "common_window": [
            "2023-07-01T00:00:00Z",
            "2024-01-01T00:00:00Z",
        ],
    },
}

PROTOCOL_PATHS = (
    "training/cross_venue_volatility_shape_handoff.py",
    "training/preregister_cross_venue_volatility_shape_handoff.py",
    "tests/test_cross_venue_volatility_shape_handoff.py",
    "tests/test_preregister_cross_venue_volatility_shape_handoff.py",
    (
        "docs/"
        "cross-venue-volatility-shape-handoff-mechanism-decision-2026-07-30.md"
    ),
    (
        "docs/"
        "cross-venue-volatility-shape-handoff-preregistration-2026-07-30.md"
    ),
    COMMON_WINDOW_POLICY["path"],
    ESDI_PREREGISTRATION["path"],
    "pyproject.toml",
    "uv.lock",
)


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def gzip_header(path: str | Path) -> tuple[list[str], str]:
    with gzip.open(path, "rb") as handle:
        raw = handle.readline()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"non-UTF-8 gzip header: {path}") from exc
    if not text.endswith("\n"):
        raise RuntimeError(f"gzip header lacks newline: {path}")
    return text.rstrip("\r\n").split(","), hashlib.sha256(raw).hexdigest()


def validate_hash_bound_artifacts(repo: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    registry = {**SOURCE_ARTIFACTS, **PRIOR_VOLATILITY_COMPARATORS}
    for name, raw_spec in registry.items():
        spec = copy.deepcopy(raw_spec)
        path = repo / str(spec["path"])
        if not path.is_file():
            raise RuntimeError(f"CVVH-432 frozen artifact missing: {name}")
        observed_hash = sha256_file(path)
        if observed_hash != spec["sha256"]:
            raise RuntimeError(f"CVVH-432 frozen artifact hash drift: {name}")
        binding: dict[str, Any] = {
            "path": str(spec["path"]),
            "sha256": observed_hash,
            "bytes": path.stat().st_size,
        }
        if "header" in spec:
            header, header_hash = gzip_header(path)
            if (
                header != spec["header"]
                or header_hash != spec["header_line_sha256"]
            ):
                raise RuntimeError(
                    f"CVVH-432 frozen artifact header drift: {name}"
                )
            binding["header"] = header
            binding["header_line_sha256"] = header_hash
            binding["rows_decoded"] = 0
        output[name] = binding
    return output


def _validated_manifest_payload(raw: Mapping[str, Any], label: str) -> None:
    manifest_hash = raw.get("manifest_hash")
    core = {key: value for key, value in raw.items() if key != "manifest_hash"}
    if not isinstance(manifest_hash, str) or manifest_hash != canonical_hash(core):
        raise RuntimeError(f"{label} manifest hash drift")


def load_gross9_authority(repo: Path) -> dict[str, Any]:
    path = repo / str(ESDI_PREREGISTRATION["path"])
    if sha256_file(path) != ESDI_PREREGISTRATION["sha256"]:
        raise RuntimeError("CVVH-432 ESDI Gross9 authority artifact drift")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise RuntimeError("CVVH-432 ESDI authority is not an object")
    _validated_manifest_payload(raw, "ESDI preregistration")
    if raw.get("manifest_hash") != ESDI_PREREGISTRATION["manifest_hash"]:
        raise RuntimeError("CVVH-432 ESDI authority manifest identity drift")
    gross9 = raw.get("gross9")
    if not isinstance(gross9, Mapping):
        raise RuntimeError("CVVH-432 ESDI Gross9 contract missing")
    authority = gross9.get("authority")
    if not isinstance(authority, Mapping):
        raise RuntimeError("CVVH-432 ESDI Gross9 authority missing")
    if gross9.get("weights") != GROSS9_WEIGHTS:
        raise RuntimeError("CVVH-432 ESDI Gross9 roster drift")
    runtime_closure = authority.get("runtime_code_closure")
    if (
        not isinstance(runtime_closure, Mapping)
        or runtime_closure.get("all_distribution_inventory_count") != 108
        or runtime_closure.get("ast_import_closure_must_match_before_artifact_creation")
        is not True
        or runtime_closure.get("runtime_environment_must_match_before_artifact_creation")
        is not True
    ):
        raise RuntimeError("CVVH-432 ESDI Gross9 closure is incomplete")
    payload = {
        "reference_preregistration": copy.deepcopy(ESDI_PREREGISTRATION),
        "authority": copy.deepcopy(dict(authority)),
        "authority_hash": canonical_hash(authority),
        "weights": copy.deepcopy(GROSS9_WEIGHTS),
        "gross": 9.0,
        "all_five_positive_weight_sleeves_required": True,
        "reuse_exact_esdi_runtime_closure_validation": True,
    }
    _validate_gross9_payload(payload)
    return payload


def _run_git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def repository_identity(repo: Path) -> dict[str, Any]:
    root = Path(_run_git(repo, "rev-parse", "--show-toplevel")).resolve()
    if root != repo.resolve():
        raise RuntimeError("CVVH-432 preregistration ran outside repository root")
    head = _run_git(repo, "rev-parse", "HEAD")
    branch = _run_git(repo, "branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError("CVVH-432 preregistration branch identity drift")
    branch_remote = _run_git(
        repo, "config", "--get", f"branch.{EXPECTED_BRANCH}.remote"
    )
    if branch_remote != "origin":
        raise RuntimeError(
            "CVVH-432 preregistration requires branch remote origin"
        )
    upstream = _run_git(
        repo, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"
    )
    upstream_ref = _run_git(repo, "rev-parse", "--symbolic-full-name", "@{u}")
    expected_upstream = f"origin/{EXPECTED_BRANCH}"
    expected_upstream_ref = f"refs/remotes/{expected_upstream}"
    if upstream != expected_upstream or upstream_ref != expected_upstream_ref:
        raise RuntimeError(
            "CVVH-432 preregistration requires genuine origin remote tracking"
        )
    upstream_remote = _run_git(
        repo,
        "for-each-ref",
        "--format=%(upstream:remotename)",
        f"refs/heads/{EXPECTED_BRANCH}",
    )
    if upstream_remote != "origin":
        raise RuntimeError(
            "CVVH-432 preregistration upstream remotename is not origin"
        )
    fetch_urls = tuple(
        line
        for line in _run_git(repo, "remote", "get-url", "--all", "origin").splitlines()
        if line
    )
    push_urls = tuple(
        line
        for line in _run_git(
            repo, "remote", "get-url", "--push", "--all", "origin"
        ).splitlines()
        if line
    )
    if fetch_urls != (EXPECTED_ORIGIN_URL,) or push_urls != (
        EXPECTED_ORIGIN_URL,
    ):
        raise RuntimeError("CVVH-432 preregistration origin URL set drift")
    upstream_head = _run_git(repo, "rev-parse", "@{u}")
    if head != upstream_head:
        raise RuntimeError("CVVH-432 preregistration requires upstream exact HEAD")
    remote_lines = [
        line
        for line in _run_git(
            repo,
            "ls-remote",
            "--heads",
            "origin",
            f"refs/heads/{EXPECTED_BRANCH}",
        ).splitlines()
        if line
    ]
    expected_remote_line = f"{head}\trefs/heads/{EXPECTED_BRANCH}"
    if remote_lines != [expected_remote_line]:
        raise RuntimeError(
            "CVVH-432 preregistration canonical remote branch is not exact HEAD"
        )
    dirty = _run_git(
        repo, "status", "--porcelain=v1", "--untracked-files=normal"
    )
    if dirty:
        raise RuntimeError(
            "CVVH-432 preregistration requires clean tree including untracked"
        )
    seal: dict[str, Any] = {}
    for relative in PROTOCOL_PATHS:
        path = repo / relative
        if not path.is_file():
            raise RuntimeError(f"CVVH-432 protocol path missing: {relative}")
        _run_git(repo, "ls-files", "--error-unmatch", relative)
        working_blob = _run_git(repo, "hash-object", relative)
        committed_blob = _run_git(repo, "rev-parse", f"HEAD:{relative}")
        if working_blob != committed_blob:
            raise RuntimeError(f"CVVH-432 protocol path not committed: {relative}")
        seal[relative] = {
            "git_blob": committed_blob,
            "sha256": sha256_file(path),
        }
    if (
        seal.get(str(COMMON_WINDOW_POLICY["path"]), {}).get("sha256")
        != COMMON_WINDOW_POLICY["sha256"]
    ):
        raise RuntimeError("CVVH-432 common-window policy hash drift")
    payload = {
        "branch": branch,
        "commit": head,
        "tree": _run_git(repo, "rev-parse", "HEAD^{tree}"),
        "upstream": upstream,
        "upstream_ref": upstream_ref,
        "upstream_remote": upstream_remote,
        "upstream_remote_url": EXPECTED_ORIGIN_URL,
        "upstream_fetch_urls": list(fetch_urls),
        "upstream_push_urls": list(push_urls),
        "upstream_commit": upstream_head,
        "canonical_remote_commit": head,
        "tracked_clean": True,
        "upstream_exact": True,
        "protocol_seal": seal,
        "protocol_seal_hash": canonical_hash(seal),
    }
    _validate_repository_payload(payload)
    return payload


def _hex(value: Any, length: int) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_repository_payload(repository: Mapping[str, Any]) -> None:
    expected_keys = {
        "branch",
        "commit",
        "tree",
        "upstream",
        "upstream_ref",
        "upstream_remote",
        "upstream_remote_url",
        "upstream_fetch_urls",
        "upstream_push_urls",
        "upstream_commit",
        "canonical_remote_commit",
        "tracked_clean",
        "upstream_exact",
        "protocol_seal",
        "protocol_seal_hash",
    }
    if set(repository) != expected_keys:
        raise RuntimeError("CVVH-432 repository identity schema drift")
    if (
        repository.get("branch") != EXPECTED_BRANCH
        or repository.get("upstream") != f"origin/{EXPECTED_BRANCH}"
        or repository.get("upstream_ref")
        != f"refs/remotes/origin/{EXPECTED_BRANCH}"
        or repository.get("upstream_remote") != "origin"
        or repository.get("upstream_remote_url") != EXPECTED_ORIGIN_URL
        or repository.get("upstream_fetch_urls") != [EXPECTED_ORIGIN_URL]
        or repository.get("upstream_push_urls") != [EXPECTED_ORIGIN_URL]
        or repository.get("tracked_clean") is not True
        or repository.get("upstream_exact") is not True
        or repository.get("commit") != repository.get("upstream_commit")
        or repository.get("commit") != repository.get("canonical_remote_commit")
        or not _hex(repository.get("commit"), 40)
        or not _hex(repository.get("tree"), 40)
    ):
        raise RuntimeError("CVVH-432 repository identity values drift")
    seal = repository.get("protocol_seal")
    if not isinstance(seal, Mapping) or set(seal) != set(PROTOCOL_PATHS):
        raise RuntimeError("CVVH-432 protocol seal paths drift")
    for relative, raw in seal.items():
        if (
            not isinstance(relative, str)
            or not isinstance(raw, Mapping)
            or set(raw) != {"git_blob", "sha256"}
            or not _hex(raw.get("git_blob"), 40)
            or not _hex(raw.get("sha256"), 64)
        ):
            raise RuntimeError("CVVH-432 protocol seal entry drift")
    if repository.get("protocol_seal_hash") != canonical_hash(seal):
        raise RuntimeError("CVVH-432 protocol seal hash drift")
    policy = seal.get(str(COMMON_WINDOW_POLICY["path"]))
    if (
        not isinstance(policy, Mapping)
        or policy.get("sha256") != COMMON_WINDOW_POLICY["sha256"]
    ):
        raise RuntimeError("CVVH-432 common-window policy binding drift")


def _validate_artifact_bindings(
    bindings: Mapping[str, Mapping[str, Any]],
) -> None:
    registry = {**SOURCE_ARTIFACTS, **PRIOR_VOLATILITY_COMPARATORS}
    if set(bindings) != set(registry):
        raise RuntimeError("CVVH-432 artifact binding registry drift")
    for name, spec in registry.items():
        binding = bindings.get(name)
        expected_keys = {"path", "sha256", "bytes"}
        if "header" in spec:
            expected_keys.update(
                {"header", "header_line_sha256", "rows_decoded"}
            )
        if (
            not isinstance(binding, Mapping)
            or set(binding) != expected_keys
            or binding.get("path") != spec["path"]
            or binding.get("sha256") != spec["sha256"]
            or not isinstance(binding.get("bytes"), int)
            or isinstance(binding.get("bytes"), bool)
            or int(binding["bytes"]) <= 0
        ):
            raise RuntimeError(f"CVVH-432 artifact binding drift: {name}")
        if "header" in spec and (
            binding.get("header") != spec["header"]
            or binding.get("header_line_sha256")
            != spec["header_line_sha256"]
            or binding.get("rows_decoded") != 0
        ):
            raise RuntimeError(f"CVVH-432 artifact header binding drift: {name}")


def _validate_gross9_payload(gross9: Mapping[str, Any]) -> None:
    expected_keys = {
        "reference_preregistration",
        "authority",
        "authority_hash",
        "weights",
        "gross",
        "all_five_positive_weight_sleeves_required",
        "reuse_exact_esdi_runtime_closure_validation",
    }
    if set(gross9) != expected_keys:
        raise RuntimeError("CVVH-432 Gross9 payload schema drift")
    authority = gross9.get("authority")
    if not isinstance(authority, Mapping):
        raise RuntimeError("CVVH-432 Gross9 authority missing")
    if gross9.get("weights") != GROSS9_WEIGHTS or gross9.get("gross") != 9.0:
        raise RuntimeError("CVVH-432 Gross9 roster drift")
    expected_authority_sections = {
        "base_portfolio",
        "clock_reconstruction",
        "portfolio",
        "pre2025_anchor",
        "runtime",
        "runtime_code_closure",
        "sleeves",
        "transitive_source_manifest",
    }
    closure = authority.get("runtime_code_closure")
    if (
        gross9.get("reference_preregistration") != ESDI_PREREGISTRATION
        or gross9.get("authority_hash") != GROSS9_AUTHORITY_HASH
        or canonical_hash(authority) != GROSS9_AUTHORITY_HASH
        or set(authority) != expected_authority_sections
        or gross9.get("all_five_positive_weight_sleeves_required") is not True
        or gross9.get("reuse_exact_esdi_runtime_closure_validation") is not True
        or not isinstance(closure, Mapping)
        or closure.get("all_distribution_inventory_count") != 108
        or closure.get("ast_import_closure_must_match_before_artifact_creation")
        is not True
        or closure.get("runtime_environment_must_match_before_artifact_creation")
        is not True
    ):
        raise RuntimeError("CVVH-432 Gross9 authority closure drift")


def _artifact_specs_with_bindings(
    specs: Mapping[str, Mapping[str, Any]],
    bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, spec in specs.items():
        if name not in bindings:
            raise RuntimeError(f"CVVH-432 missing artifact binding: {name}")
        combined = copy.deepcopy(dict(spec))
        combined["validated_binding"] = copy.deepcopy(dict(bindings[name]))
        output[name] = combined
    return output


def build_registration(
    *,
    repository: Mapping[str, Any],
    artifact_bindings: Mapping[str, Mapping[str, Any]],
    gross9: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_repository_payload(repository)
    _validate_artifact_bindings(artifact_bindings)
    _validate_gross9_payload(gross9)

    core: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": mechanism.CANDIDATE_ID,
        "as_of_date": AS_OF_DATE,
        "singleton": True,
        "repository": copy.deepcopy(dict(repository)),
        "research_boundary": {
            "bvol_or_dvol_rows_decoded": 0,
            "candidate_incidence_opened": False,
            "comparator_rows_decoded": 0,
            "gross9_clock_rows_opened": 0,
            "btc_execution_rows_opened": 0,
            "funding_rows_opened": 0,
            "return_pnl_cagr_or_drawdown_opened": False,
            "known_prior_mechanisms": [
                "close-level BVOL/DVOL disagreement",
                "BTC price-follow after DVOL movement",
                "OPDR close-ratio plus premium path",
            ],
            "forbidden_repairs": [
                "close-ratio substitution",
                "BTC price-follow feature",
                "premium-path direction or efficiency",
                "24h or 48h hold substitution",
                "polarity inversion",
                "threshold grid",
                "rank2 substitution",
            ],
        },
        "source_contract": {
            "artifacts": _artifact_specs_with_bindings(
                SOURCE_ARTIFACTS, artifact_bindings
            ),
            "join": {
                "type": "exact UTC one-to-one completed-hour inner join",
                "bvol_key": "feature_available_time_utc",
                "dvol_key": "close_time",
                "joint_availability": "max of both exact source clocks",
                "fill_imputation_tolerance_or_nearest": False,
                "duplicate_or_nonmonotonic": "terminal failure",
                "first_post_gap_or_invalid_can_emit": False,
            },
            "decimal_tokens": {
                "parse": "exact Decimal token to exact rational coefficient",
                "coefficient_digits_max": 128,
                "exponent_min": -128,
                "exponent_max": 128,
                "all_ohlc_finite_and_strictly_positive": True,
                "envelope": (
                    "high>=max(open,close) AND low<=min(open,close)"
                ),
                "binary_float_feature_arithmetic": False,
            },
            "allowed_pre_support_reads": [
                "compressed file SHA-256",
                "compressed byte count",
                "single UTF-8 CSV header line",
                "hash-bound JSON metadata bytes",
            ],
            "source_value_or_incidence_reads_before_committed_evaluator": False,
        },
        "mechanism": {
            "body": "(close-open)/open",
            "range": "(high-low)/open",
            "comparison": "exact positive-denominator cross multiplication",
            "primary": {
                "opposite_nonzero_bodies": True,
                "binance_absolute_body_strictly_greater": True,
                "binance_range_strictly_greater": True,
                "binance_positive": "SHORT",
                "binance_negative": "LONG",
                "equality": "NONE",
            },
            "onset": {
                "requires_current_and_previous_valid_consecutive_hours": True,
                "emit": "current in {LONG,SHORT} AND current != previous",
                "continued_same_side": "suppress",
                "opposite_side_transition": "emit",
                "first_after_gap_or_invalid": "suppress",
            },
            "canonical_id": (
                "CVVH-432|<control>|T=<four-digit RFC3339 whole-second UTC Z>"
            ),
            "candidate_sort": [
                "entry_time",
                "signal_time",
                "canonical_id",
                "side",
            ],
            "implementation": (
                "training/cross_venue_volatility_shape_handoff.py"
            ),
        },
        "execution": {
            "entry": "ceil_to_5m(joint_availability)+5 elapsed minutes",
            "aligned_availability_still_waits_minutes": 5,
            "missing_exact_market_open": "terminal economic-stage failure",
            "hold_bars_5m": 432,
            "hold_seconds": 129_600,
            "leverage": "1/2",
            "base_cost_bp_per_notional_side": 6,
            "stress_cost_bp_per_notional_side": 10,
            "reservation": {
                "scope": "one global CVVH position",
                "interval": "[entry,exit)",
                "accept": "entry>=previous accepted exit",
                "suppressed_candidates_queued": False,
            },
            "funding_interval": "entry_time<=funding_time<exit_time",
            "funding_cash": (
                "-side_sign*fixed_quantity*funding_rate*settlement_mark"
            ),
            "early_close_stop_take_profit_trailing_or_pyramiding": False,
        },
        "calendars": {
            "full": [
                "2023-06-01T00:00:00Z",
                "2026-06-01T00:00:00Z",
            ],
            "selection": [
                "2023-06-01T00:00:00Z",
                "2025-01-01T00:00:00Z",
            ],
            "selection_periods": {
                "2023H2": [
                    "2023-06-01T00:00:00Z",
                    "2024-01-01T00:00:00Z",
                ],
                "2024": [
                    "2024-01-01T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                ],
            },
            "support_subperiods": {
                "2024H1": [
                    "2024-01-01T00:00:00Z",
                    "2024-07-01T00:00:00Z",
                ],
                "2024H2": [
                    "2024-07-01T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                ],
            },
            "future25": [
                "2025-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
            ],
            "future26": [
                "2026-01-01T00:00:00Z",
                "2026-06-01T00:00:00Z",
            ],
            "full_calendar_years": 3,
            "interval_eligibility": "entry>=start AND exit<=end; no clipping",
            "pre_source_full_calendar_idleness_included": True,
        },
        "controls": {
            "independent_own_clock": {
                "deribit_led": (
                    "swap leadership; opposite dominant DVOL body maps to side"
                ),
                "body_lead_only": "omit Binance range leadership",
                "range_lead_only": "omit Binance body-magnitude leadership",
                "stale_deribit": (
                    "state[t]=BVOL[t] vs DVOL[t-1]; prior=BVOL[t-1] vs "
                    "DVOL[t-2]; require t,t-1,t-2 valid and retain T availability"
                ),
            },
            "accepted_primary_parent_set": {
                "direction_flip": "flip accepted primary side",
                "deterministic_random_side": (
                    "SHA256 UTF-8(CVVH-432|<primary-id>|RANDOM_SIDE); "
                    "first byte<128 LONG else SHORT"
                ),
                "constant_long": "accepted primary entries fixed LONG",
                "constant_short": "accepted primary entries fixed SHORT",
                "one_bar_delayed_entry": (
                    "shift accepted primary entry and exit +300 seconds; "
                    "do not rerun reservation"
                ),
            },
            "controls_cannot_replace_repair_or_redefine_primary": True,
        },
        "support_gates": {
            "opens_before_novelty_gross9_or_outcomes": True,
            "selection": {
                "total_min": 45,
                "2023H2_min": 12,
                "2024H1_min": 12,
                "2024H2_min": 12,
                "each_side_min": 14,
                "maximum_month_share": "1/5",
            },
            "future25": {
                "total_min": 30,
                "each_side_min": 8,
                "maximum_month_share": "1/4",
            },
            "future26": {
                "total_min": 15,
                "each_side_min": 4,
                "maximum_month_share": "3/10",
            },
            "full": {
                "maximum_accepted_entry_gap_elapsed_days": 90,
                "maximum_same_side_run": 12,
            },
            "structural_control_distinctness_each": {
                "controls": list(mechanism.INDEPENDENT_CONTROLS),
                "exact_entry_jaccard": {"operator": "<", "value": "9/10"},
                "one_to_one_24h_max_matched_share": {
                    "operator": "<",
                    "value": "19/20",
                },
                "matching": (
                    "same cardinality, exact-lag, lexicographic one-to-one "
                    "objective as novelty"
                ),
            },
            "selection_prefix_append_invariance": {
                "cutoff": "2025-01-01T00:00:00Z",
                "byte_identical_fields": [
                    "base_validity",
                    "states",
                    "raw_candidate_ids",
                    "accepted_ids",
                    "sides",
                    "entry_exit_clocks",
                ],
                "later_rows_may_change_prefix": False,
            },
            "undefined_empty_or_zero_denominator": "terminal failure",
            "failure_action": "retire exact CVVH-432 unchanged before novelty",
        },
        "novelty": {
            "opens_only_after_committed_source_support_pass": True,
            "common_window_policy": copy.deepcopy(COMMON_WINDOW_POLICY),
            "common_window_policy_contamination_disclosure": {
                "disclosed": True,
                "fact": (
                    "source-only RMSR verification exposed a valid prior "
                    "comparator interval crossing the 2023/2024 common-window "
                    "boundary"
                ),
                "effect": (
                    "prospective policy motivation only; no CVVH source value, "
                    "incidence, comparator overlap, or outcome was opened"
                ),
                "eligibility_rule_frozen_before_cvvh_incidence": True,
            },
            "prior_volatility_comparators": _artifact_specs_with_bindings(
                PRIOR_VOLATILITY_COMPARATORS, artifact_bindings
            ),
            "minimum_fully_contained_rows_each_clock": 10,
            "raw_artifact_validation_before_window_filter": True,
            "full_containment_only_no_clip_shift_or_split": True,
            "matching": {
                "tolerance_elapsed_seconds": 21_600,
                "one_to_one": True,
                "objective_order": [
                    "maximum matched cardinality",
                    "minimum exact total absolute elapsed seconds",
                    "lexicographically smallest ordered timestamp-pair list",
                ],
                "reported_pair_list_sha256": True,
            },
            "thresholds_each_prior_volatility_and_gross9_sleeve": {
                "exact_entry_jaccard_max": "1/10",
                "one_to_one_6h_max_matched_share_max": "7/20",
                "occupied_5m_bar_jaccard_max": "1/4",
                "absolute_signed_exposure_pearson_max": "7/20",
                "pearson_implementation_gate": "exact square<=49/400",
            },
            "signed_exposure_grid": {
                "interval": "[entry,exit)",
                "frequency_seconds": 300,
                "LONG": 1,
                "SHORT": -1,
                "idle": 0,
            },
            "undefined_metric_or_missing_group": "terminal failure",
            "all_declared_comparators_and_sleeves_must_pass": True,
            "comparator_removal_after_overlap_seen": False,
            "failure_action": "retire exact CVVH-432 unchanged before outcomes",
        },
        "gross9": copy.deepcopy(dict(gross9)),
        "economic_contract": {
            "opens_only_after_committed_clean_exact_novelty_reproduction": True,
            "engine": "exact ESDI strict-open fixed-quantity engine",
            "market_calendar": (
                "complete unique finite BTCUSDT perpetual 5m OHLC, including "
                "period-end boundary open; missing/duplicate/off-grid is terminal"
            ),
            "path_order": [
                "global/pre-entry high-water mark",
                "favorable held OHLC plus funding credits",
                "adverse held OHLC plus funding debits",
                "hypothetical liquidation cost",
                "entry and exit side costs",
            ],
            "sizing": (
                "exit first at exact open; size entries from post-exit "
                "pre-entry equity; quantity remains fixed to exit"
            ),
            "nonpositive_or_nonfinite_equity": "terminal liquidation failure",
            "standalone_gate_base_and_stress_each_gated_stage": {
                "absolute_return": ">0",
                "full_calendar_cagr_to_strict_mdd": ">=3",
                "strict_mdd": "<=3/20",
                "mean_gross_underlying_bp": ">=20",
                "calendar_month_clustered_signflip_p": "<=1/10",
            },
            "signflip": {
                "cluster": "UTC entry month",
                "gated_stages": (
                    "exact exhaustive 2^k sign enumeration; k<=19 including "
                    "combined selection"
                ),
                "stitched_full_if_k_gt_20": (
                    "fixed-seed Monte Carlo allowed because full is descriptive"
                ),
            },
            "source_specific_superiority": {
                "metric": "cagr_to_strict_mdd",
                "operator": "primary > control",
                "periods": ["2023H2", "2024"],
                "costs": ["base", "stress"],
                "primary_must_strictly_exceed": [
                    "body_lead_only",
                    "range_lead_only",
                ],
                "diagnostic_only_cannot_replace": [
                    "deribit_led",
                    "stale_deribit",
                    "one_bar_delayed_entry",
                ],
                "cannot_completely_qualify": [
                    "direction_flip",
                    "deterministic_random_side",
                    "constant_long",
                    "constant_short",
                ],
            },
        },
        "same_gross": {
            "gross9_weights": copy.deepcopy(GROSS9_WEIGHTS),
            "baseline_gross": 9.0,
            "candidate_weights": ["1/4", "1/2", "3/4"],
            "treatment": (
                "scale every Gross9 sleeve by (9-w)/9 and add CVVH at w"
            ),
            "configured_treatment_gross": 9.0,
            "comparison": "unscaled authoritative Gross9 at gross 9",
            "selection_periods_only": ["2023H2", "2024"],
            "requirements_every_period_base_and_stress": {
                "cagr_to_strict_mdd_absolute_improvement_min": "1/20",
                "unscaled_absolute_return_retention_min": "97/100",
                "treatment_absolute_return_positive": True,
                "liquidation_safe": True,
            },
            "strict_mdd_reduced_in_at_least_one_selection_cost_cell": True,
            "ranking": (
                "maximum minimum 2023H2/2024 base/stress improvement; "
                "tie lower candidate weight"
            ),
            "freeze_rank": 1,
            "top1_must_pass_or_terminal": True,
            "rank2_substitution": False,
            "future": {
                "weight": "exact frozen rank1 only",
                "rerank_repair_or_alternate_weight": False,
                "future25_then_future26": True,
                "each_requires_standalone_and_same_gross_base_stress": True,
                "each_requires_strict_mdd_reduction_in_at_least_one_cost_cell": True,
                "failure": "terminal veto",
            },
            "stitched_exact_three_year_report": {
                "required_after_both_future_pass": True,
                "descriptive_non_gating": True,
                "can_repair_or_rerank": False,
            },
        },
        "gross9_evidence_boundary": {
            "reconstruct_only_after_source_support_pass": True,
            "complete_esdi_authority_closure_must_validate": True,
            "full_domain_sleeve_clocks_for_structural_novelty": True,
            "future_rows_used_for_structural_candidate_veto": True,
            "future_rows_used_for_economic_weight_ranking": False,
            "portfolio_return_pnl_metrics_at_novelty": False,
        },
        "stage_sequence": [
            "mechanism_commit_and_push",
            "preregistration_producer_tests_commit_and_push",
            "write_once_canonical_preregistration_commit_and_push",
            "source_support_evaluator_tests_commit_and_push",
            "authoritative_source_support_claim_then_one_run",
            "commit_source_support_or_terminal_failure",
            "novelty_evaluator_tests_commit_and_push",
            "authoritative_novelty_claim_then_one_run",
            "commit_novelty_or_terminal_failure",
            "economics_evaluator_tests_commit_and_push",
            (
                "physical stages 2023H2, 2024, selection, same_gross, "
                "future25, future26, full"
            ),
            "clean_checkout_verification_only_replay_after_success",
            "final_independent_review_commit_and_push",
        ],
        "attempt_and_reproduction_contract": {
            "each_authoritative_stage": {
                "atomic_write_once_claim_before_first_protected_read": True,
                "claim_binds": [
                    "commit",
                    "preregistration",
                    "evaluator closure",
                    "dependency hashes",
                    "prior receipt hashes",
                ],
                "retry_resume_or_fallback_after_claim": False,
                "stop_on_first_failure": True,
            },
            "verification_replay": {
                "allowed_only_after_successful_committed_authoritative_bytes": True,
                "separate_atomic_write_once_claim": True,
                "clean_checkout_and_upstream_exact": True,
                "verification_only": True,
                "canonical_write_ranking_or_repair_authority": False,
                "byte_identical_temp_artifacts_and_receipts_required": True,
                "authoritative_failure_permanently_forbids_replay": True,
                "mismatch": "terminal reproducibility failure",
            },
        },
        "sequence_rules": {
            "stop_at_first_failure": True,
            "parameter_threshold_hold_latency_or_polarity_repair": False,
            "future_can_rank_or_repair": False,
            "control_can_replace_primary": False,
            "ordinary_failure_repair_under_same_identity": False,
        },
        "producer_effects": {
            "network_calls": {
                "source_or_web": 0,
                "read_only_git_ls_remote_attestation": 1,
            },
            "git_metadata_subprocess_calls": "read-only",
            "compressed_artifact_hash_reads": True,
            "gzip_header_lines_decoded": (
                len(
                    [
                        spec
                        for spec in (
                            list(SOURCE_ARTIFACTS.values())
                            + list(PRIOR_VOLATILITY_COMPARATORS.values())
                        )
                        if "header" in spec
                    ]
                )
            ),
            "csv_data_rows_decoded": 0,
            "output_publication": "atomic hard-link write-once",
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_registration(payload: Mapping[str, Any]) -> None:
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("CVVH-432 preregistration protocol drift")
    if payload.get("policy_id") != mechanism.CANDIDATE_ID:
        raise RuntimeError("CVVH-432 preregistration identity drift")
    _validated_manifest_payload(payload, "CVVH-432 preregistration")
    repository = payload.get("repository")
    source_contract = payload.get("source_contract")
    novelty = payload.get("novelty")
    gross9 = payload.get("gross9")
    if (
        not isinstance(repository, Mapping)
        or not isinstance(source_contract, Mapping)
        or not isinstance(novelty, Mapping)
        or not isinstance(gross9, Mapping)
    ):
        raise RuntimeError("CVVH-432 preregistration required section missing")
    source_artifacts = source_contract.get("artifacts")
    prior_artifacts = novelty.get("prior_volatility_comparators")
    if not isinstance(source_artifacts, Mapping) or not isinstance(
        prior_artifacts, Mapping
    ):
        raise RuntimeError("CVVH-432 preregistration artifact registry missing")
    bindings: dict[str, Mapping[str, Any]] = {}
    for name, raw in {**source_artifacts, **prior_artifacts}.items():
        if not isinstance(raw, Mapping) or not isinstance(
            raw.get("validated_binding"), Mapping
        ):
            raise RuntimeError(
                f"CVVH-432 preregistration binding missing: {name}"
            )
        bindings[str(name)] = raw["validated_binding"]
    expected = build_registration(
        repository=repository,
        artifact_bindings=bindings,
        gross9=gross9,
    )
    if dict(payload) != expected:
        raise RuntimeError("CVVH-432 preregistration frozen contract drift")


def encoded_registration(payload: Mapping[str, Any]) -> bytes:
    validate_registration(payload)
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_write_once(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"CVVH-432 preregistration already exists: {path}")
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def generate(repo: Path, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    identity = repository_identity(repo)
    bindings = validate_hash_bound_artifacts(repo)
    gross9 = load_gross9_authority(repo)
    payload = build_registration(
        repository=identity,
        artifact_bindings=bindings,
        gross9=gross9,
    )
    destination = output if output.is_absolute() else repo / output
    atomic_write_once(destination, encoded_registration(payload))
    return payload


def main(arguments: Sequence[str] | None = None) -> int:
    received = tuple(sys.argv[1:] if arguments is None else arguments)
    if received:
        raise SystemExit("CVVH-432 preregistration takes no arguments")
    repo = Path(__file__).resolve().parents[1]
    payload = generate(repo)
    print(
        json.dumps(
            {
                "status": "written_once",
                "output": str(DEFAULT_OUTPUT),
                "manifest_hash": payload["manifest_hash"],
                "source_rows_decoded": 0,
                "outcomes_opened": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AS_OF_DATE",
    "COMMON_WINDOW_POLICY",
    "DEFAULT_OUTPUT",
    "ESDI_PREREGISTRATION",
    "EXPECTED_BRANCH",
    "EXPECTED_ORIGIN_URL",
    "GROSS9_AUTHORITY_HASH",
    "GROSS9_WEIGHTS",
    "PRIOR_VOLATILITY_COMPARATORS",
    "PROTOCOL_PATHS",
    "PROTOCOL_VERSION",
    "SOURCE_ARTIFACTS",
    "atomic_write_once",
    "build_registration",
    "canonical_hash",
    "encoded_registration",
    "generate",
    "gzip_header",
    "load_gross9_authority",
    "repository_identity",
    "sha256_file",
    "validate_hash_bound_artifacts",
    "validate_registration",
]
