from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import pytest

from training import preregister_tron_usdt_supply_impulse as p

esdi = p.esdi


def fake_repository_identity() -> dict[str, Any]:
    paths = sorted(str(path) for path in p.committed_identity_paths())
    git_blobs = {
        path: hashlib.sha1(path.encode("utf-8")).hexdigest() for path in paths
    }
    sha256 = {
        path: hashlib.sha256(path.encode("utf-8")).hexdigest() for path in paths
    }
    sha256[str(p.SOURCE_DECISION_PATH)] = p.SOURCE_DECISION_SHA256
    sha256[str(p.MECHANISM_DECISION_PATH)] = p.MECHANISM_DECISION_SHA256
    sha256[str(p.ESDI_HELPER_PATH)] = p.ESDI_HELPER_SHA256
    sha256[str(p.ESDI_PREREGISTRATION_PATH)] = (
        p.ESDI_PREREGISTRATION_SHA256
    )
    seal = {"git_blobs": git_blobs, "sha256": sha256}
    return {
        "branch": p.EXPECTED_BRANCH,
        "head_commit": "a" * 40,
        "head_tree": "b" * 40,
        "upstream": f"origin/{p.EXPECTED_BRANCH}",
        "upstream_ref": f"refs/remotes/origin/{p.EXPECTED_BRANCH}",
        "upstream_commit": "a" * 40,
        "head_equals_upstream_required": True,
        "git_blobs": git_blobs,
        "sha256": sha256,
        "whole_worktree_clean_required": True,
        "bound_paths_clean_against_head_required": True,
        "protocol_seal_hash": p.canonical_hash(seal),
    }


def manifest() -> dict[str, Any]:
    return p.build_manifest(fake_repository_identity())


@pytest.mark.parametrize(
    ("helper_variant", "expected_error"),
    [
        ("tampered_regular", "helper preflight SHA-256 drift"),
        ("working_tree_symlink", "helper preflight path is unsafe"),
        (
            "committed_blob_mismatch",
            "helper differs from its committed Git blob",
        ),
    ],
)
def test_cold_import_rejects_unsafe_helper_before_exec(
    tmp_path: Path,
    helper_variant: str,
    expected_error: str,
) -> None:
    repo = tmp_path / "repo"
    training = repo / "training"
    training.mkdir(parents=True)
    (training / "__init__.py").write_text("", encoding="utf-8")
    producer_source = Path(__file__).resolve().parents[1] / p.PRODUCER_PATH
    shutil.copyfile(producer_source, repo / p.PRODUCER_PATH)
    sentinel = tmp_path / "helper-executed"
    helper_payload = (
        "from pathlib import Path\n"
        "import os\n"
        "Path(os.environ['TUSI_SENTINEL']).write_text('executed')\n"
    )
    helper_path = repo / p.ESDI_HELPER_PATH
    if helper_variant in {"tampered_regular", "committed_blob_mismatch"}:
        helper_path.write_text(helper_payload, encoding="utf-8")
    else:
        helper_source = Path(__file__).resolve().parents[1] / p.ESDI_HELPER_PATH
        shutil.copyfile(helper_source, helper_path)
    _run_git(repo, "init", "-b", p.EXPECTED_BRANCH)
    _run_git(repo, "config", "user.email", "cold@example.invalid")
    _run_git(repo, "config", "user.name", "Cold Import Test")
    _run_git(repo, "add", "training")
    _run_git(repo, "commit", "-m", "helper preflight")
    if helper_variant == "working_tree_symlink":
        linked_helper = tmp_path / "linked-helper.py"
        linked_helper.write_text(helper_payload, encoding="utf-8")
        helper_path.unlink()
        helper_path.symlink_to(linked_helper)
    elif helper_variant == "committed_blob_mismatch":
        helper_source = Path(__file__).resolve().parents[1] / p.ESDI_HELPER_PATH
        shutil.copyfile(helper_source, helper_path)

    script = f"""
import pathlib
import sys

helper_exec_events = []
def audit(event, args):
    if event == "exec":
        filename = getattr(args[0], "co_filename", "")
        if filename.endswith({str(p.ESDI_HELPER_PATH)!r}):
            helper_exec_events.append(filename)

sys.addaudithook(audit)
try:
    from training import preregister_tron_usdt_supply_impulse
except RuntimeError as error:
    if {expected_error!r} not in str(error):
        raise
else:
    raise AssertionError("unsafe helper import unexpectedly succeeded")
if pathlib.Path({str(sentinel)!r}).exists():
    raise AssertionError("unsafe helper side effect executed")
if helper_exec_events:
    raise AssertionError("unsafe helper bytecode executed")
print("preflight-blocked-before-helper-exec")
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo)
    environment["TUSI_SENTINEL"] = str(sentinel)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "preflight-blocked-before-helper-exec"
    assert not sentinel.exists()


def test_cold_import_real_pushed_create_has_exact_effect_allowlist(
    tmp_path: Path,
) -> None:
    source_root = Path(__file__).resolve().parents[1]
    remote = tmp_path / "remote.git"
    _run_git(tmp_path, "init", "--bare", str(remote))
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    _run_git(repository_root, "init", "-b", p.EXPECTED_BRANCH)
    _run_git(repository_root, "config", "user.email", "audit@example.invalid")
    _run_git(repository_root, "config", "user.name", "Cold Audit Test")
    identity_paths = tuple(
        sorted(str(path) for path in p.committed_identity_paths())
    )
    for relative in identity_paths:
        source = source_root / relative
        destination = repository_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    _run_git(repository_root, "add", *identity_paths)
    _run_git(repository_root, "commit", "-m", "pushed producer identity")
    _run_git(repository_root, "remote", "add", "origin", str(remote))
    _run_git(
        repository_root,
        "push",
        "-u",
        "origin",
        p.EXPECTED_BRANCH,
    )
    head_commit = _run_git(repository_root, "rev-parse", "HEAD")

    producer_source = repository_root / p.PRODUCER_PATH
    package_source = repository_root / "training/__init__.py"
    allowed_repository_files = {
        *(repository_root / path for path in identity_paths),
        Path(importlib.util.cache_from_source(str(producer_source))),
        Path(importlib.util.cache_from_source(str(package_source))),
    }
    distribution_metadata = {
        Path(distribution._path) / "METADATA"
        for distribution in esdi.importlib_metadata.distributions()
    }
    assert len(distribution_metadata) == p.FROZEN_DISTRIBUTION_INVENTORY_COUNT
    assert all(path.is_file() for path in distribution_metadata)
    site_packages_directories = {
        path.parent.parent for path in distribution_metadata
    }
    assert len(site_packages_directories) == 1

    script = f"""
import http.client
import os
from pathlib import Path
import re
import socket
import sys
import urllib.request

repository_root = Path({str(repository_root)!r})
helper_path = repository_root / {str(p.ESDI_HELPER_PATH)!r}
artifact_name = {p.DEFAULT_OUTPUT.name!r}
artifact_path = repository_root / {str(p.DEFAULT_OUTPUT)!r}
identity_paths = tuple({list(identity_paths)!r})
allowed_repository_files = {{
    Path(path) for path in {sorted(map(str, allowed_repository_files))!r}
}}
distribution_metadata = {{
    Path(path) for path in {sorted(map(str, distribution_metadata))!r}
}}
site_packages_directories = {{
    Path(path) for path in {sorted(map(str, site_packages_directories))!r}
}}
allowed_repository_directories = {{repository_root}}
for path in allowed_repository_files:
    parent = path.parent
    while parent != repository_root:
        allowed_repository_directories.add(parent)
        parent = parent.parent
allowed_repository_directories.add(repository_root / "results")
allowed_relative_parts = {{
    part
    for path in allowed_repository_files
    for part in path.relative_to(repository_root).parts
}}
allowed_relative_parts.update({{"results", artifact_name}})
base_prefix = Path(sys.base_prefix)
expected_git = [
    ("git", "rev-parse", "--show-toplevel", "HEAD"),
    (
        "git",
        "ls-tree",
        "-z",
        {head_commit!r},
        "--",
        {str(p.ESDI_HELPER_PATH)!r},
    ),
    (
        "git",
        "status",
        "--porcelain=v2",
        "--branch",
        "--ahead-behind",
        "--untracked-files=all",
    ),
    ("git", "rev-parse", "--show-toplevel", {"HEAD^{tree}"!r}),
    ("git", "ls-tree", "-z", {head_commit!r}, "--", *identity_paths),
    (
        "git",
        "status",
        "--porcelain=v2",
        "--branch",
        "--ahead-behind",
        "--untracked-files=all",
    ),
]
git_calls = []
helper_exec_count = 0
output_link_count = 0
output_remove_count = 0
temporary_pattern = re.compile(
    rf"\\.{{re.escape(artifact_name)}}\\.{{os.getpid()}}\\."
    rf"[0-9a-f]{{{{16}}}}\\.tmp"
)
write_flags = (
    os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
)

def blocked_effect(*_args, **_kwargs):
    raise AssertionError("network or HTTP effect attempted")

socket.socket = blocked_effect
socket.create_connection = blocked_effect
urllib.request.urlopen = blocked_effect
http.client.HTTPConnection.connect = blocked_effect
http.client.HTTPSConnection.connect = blocked_effect

def is_within(path, root):
    return path == root or root in path.parents

def decoded_path(value):
    if isinstance(value, bytes):
        return os.fsdecode(value)
    return os.fspath(value)

def audit(event, args):
    global helper_exec_count, output_link_count, output_remove_count
    if event.startswith("socket.") or event.startswith("http."):
        raise AssertionError(f"network audit event: {{event}}")
    if event == "subprocess.Popen":
        executable, arguments, cwd, environment = args
        call = tuple(os.fspath(argument) for argument in arguments)
        index = len(git_calls)
        if (
            executable != "git"
            or index >= len(expected_git)
            or call != expected_git[index]
            or Path(cwd) != repository_root
            or environment is not None
        ):
            raise AssertionError(f"unexpected subprocess: {{args!r}}")
        git_calls.append(call)
        return
    if event == "exec":
        filename = getattr(args[0], "co_filename", "")
        if filename == str(helper_path):
            if git_calls != expected_git[:2]:
                raise AssertionError("helper executed before complete preflight")
            helper_exec_count += 1
        return
    if event == "open":
        target, mode, flags = args
        if isinstance(target, int):
            return
        raw = decoded_path(target)
        writing = bool(
            (isinstance(flags, int) and flags & write_flags)
            or (isinstance(mode, str) and any(flag in mode for flag in "wax+"))
        )
        if os.path.isabs(raw):
            path = Path(os.path.normpath(raw))
            if is_within(path, repository_root):
                allowed_output = (
                    path == artifact_path
                    or (
                        path.parent == artifact_path.parent
                        and temporary_pattern.fullmatch(path.name)
                    )
                )
                if allowed_output:
                    if writing and not temporary_pattern.fullmatch(path.name):
                        raise AssertionError(
                            f"unexpected output write: {{args!r}}"
                        )
                    return
                allowed_dependency = (
                    path in allowed_repository_files
                    or path in allowed_repository_directories
                )
                if not allowed_dependency or writing:
                    raise AssertionError(
                        f"forbidden repository/dependency open: {{args!r}}"
                    )
                return
            if path in distribution_metadata and not writing:
                return
            if is_within(path, base_prefix) and not writing:
                return
            raise AssertionError(f"unexpected external open: {{args!r}}")
        if temporary_pattern.fullmatch(raw):
            if not writing:
                raise AssertionError(
                    f"temporary output not opened for write: {{args!r}}"
                )
            return
        if raw not in allowed_relative_parts or writing:
            raise AssertionError(f"unexpected relative open: {{args!r}}")
        return
    if event in {{"os.listdir", "os.scandir"}}:
        target = args[0]
        if isinstance(target, int):
            return
        raw = decoded_path(target)
        if not os.path.isabs(raw):
            raise AssertionError(f"unexpected relative directory scan: {{args!r}}")
        path = Path(os.path.normpath(raw))
        if (
            path in allowed_repository_directories
            or path in site_packages_directories
            or is_within(path, base_prefix)
        ):
            return
        raise AssertionError(f"unexpected directory scan: {{args!r}}")
    if event == "os.link":
        source, destination, source_fd, destination_fd = args
        if (
            not temporary_pattern.fullmatch(decoded_path(source))
            or decoded_path(destination) != artifact_name
            or not isinstance(source_fd, int)
            or not isinstance(destination_fd, int)
        ):
            raise AssertionError(f"unexpected hard link: {{args!r}}")
        output_link_count += 1
        return
    if event == "os.remove":
        target, directory_fd = args
        if (
            not temporary_pattern.fullmatch(decoded_path(target))
            or not isinstance(directory_fd, int)
        ):
            raise AssertionError(f"unexpected remove: {{args!r}}")
        output_remove_count += 1
        return
    if event in {{"os.rename", "os.mkdir", "os.rmdir"}}:
        raise AssertionError(f"unexpected filesystem mutation: {{event}} {{args!r}}")

if (
    "training.preregister_tron_usdt_supply_impulse" in sys.modules
    or "training._tusi_verified_esdi_authority" in sys.modules
):
    raise AssertionError("producer/helper imported before audit hook")
sys.addaudithook(audit)

from training import preregister_tron_usdt_supply_impulse as producer

if producer.REPOSITORY_ROOT != repository_root:
    raise AssertionError("producer resolved the wrong repository")
status, payload = producer.write_once()
if status != "created":
    raise AssertionError(f"unexpected write_once status: {{status}}")
if artifact_path.read_bytes() != producer.canonical_manifest_bytes(payload):
    raise AssertionError("synthetic artifact bytes drift")
if git_calls != expected_git:
    raise AssertionError(f"Git calls drift: {{git_calls!r}}")
if helper_exec_count != 1:
    raise AssertionError(f"helper exec count drift: {{helper_exec_count}}")
if output_link_count != 1 or output_remove_count != 1:
    raise AssertionError("atomic output effects drift")
if payload["producer_effects"]["git_read_only_subprocess_calls"] != len(
    expected_git
):
    raise AssertionError("manifest Git subprocess count drift")
print("cold-real-audit-create-passed")
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository_root)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "cold-real-audit-create-passed"


def test_exact_identity_contract_and_deterministic_bytes() -> None:
    first = manifest()
    second = manifest()
    assert first == second
    assert p.canonical_manifest_bytes(first) == p.canonical_manifest_bytes(second)
    core = {key: value for key, value in first.items() if key != "manifest_hash"}
    assert first["manifest_hash"] == p.canonical_hash(core)
    assert first["policy_id"] == "TUSI-168"
    assert first["protocol_version"] == (
        "tron_usdt_supply_impulse_preregistration_v1"
    )
    assert set(first) == set(p.PREREGISTRATION_TOP_LEVEL_TYPES)
    assert "write_once_artifacts" not in first
    assert p.DEFAULT_OUTPUT == Path(
        "results/tron_usdt_supply_impulse_preregistration_2026-07-30.json"
    )
    canonical = p.canonical_manifest_bytes(first)
    assert canonical.endswith(b"\n")
    assert b"https://" not in canonical
    assert b"/jsonrpc" not in canonical
    assert b"synthetic-provider-secret-token" not in canonical
    p.validate_manifest(first)


def test_source_chain_boundaries_chunks_topics_transports_and_pairing() -> None:
    source = manifest()["source"]
    assert source["authority"] == "TRON mainnet"
    assert source["chain_id"] == "0x2b6653dc"
    assert source["contract"] == {
        "base58": "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t",
        "log_address": "0xa614f803b6fd780986a42c78ec9c7f77e6ded13c",
        "zero_address": "0x0000000000000000000000000000000000000000",
        "decimals": 6,
    }
    assert source["first_source_block"] == 47_313_358
    assert source["exclusive_end_block"] == 83_201_056
    assert source["confirmation_blocks"] == 64
    assert source["last_admissible_event_block"] == 83_200_991
    assert source["last_confirmation_block"] == 83_201_055
    assert source["chunks"] == {
        "count": 7_178,
        "maximum_blocks": 5_000,
        "full_size_chunk_count": 7_177,
        "full_size_chunk_blocks": 5_000,
        "final_chunk": {
            "inclusive": [83_198_358, 83_200_991],
            "block_count": 2_634,
        },
        "inclusive_contiguous_fixed": True,
        "gap_overlap_dynamic_split_or_partial_final_chunk": False,
    }
    assert [row["first_block_at_or_after"] for row in source["boundaries"]] == [
        47_313_358,
        51_652_374,
        57_811_194,
        68_346_198,
        78_854_231,
        83_201_056,
    ]
    assert len(source["topics"]) == 5
    assert all(len(topic) == 66 for topic in source["topics"].values())
    assert source["raw_log_envelope"] == {
        "inclusive": [47_313_358, 83_200_991],
        "inclusive_block_count": 35_887_634,
        "confirmation_only_not_event_queried": [83_200_992, 83_201_055],
        "exclusive_boundary_header": 83_201_056,
    }
    assert source["transports"] == [
        {
            "role": "primary",
            "scheme": "https",
            "hostname": "api.trongrid.io",
            "port": 443,
        },
        {
            "role": "verification",
            "scheme": "https",
            "hostname": "tron-mainnet.core.chainstack.com",
            "port": 443,
        },
    ]
    assert source["maximum_json_rpc_batch_by_role"] == {
        "primary": 100,
        "verification": 30,
    }
    serialized_transports = repr(source["transports"])
    assert "https://" not in serialized_transports
    assert "/jsonrpc" not in serialized_transports
    runtime_transport = source["transport_runtime_configuration"]
    assert runtime_transport[
        "url_path_query_userinfo_or_credential_serialized"
    ] is False
    assert "exactly one" in source["pairing"]["Issue"]
    assert "exactly one" in source["pairing"]["Redeem"]
    assert source["pairing"]["bidirectional_exact"] is True
    assert source["pairing"]["orphan_semantic_events"] == 0
    assert source["pairing"]["orphan_zero_address_transfers"] == 0
    assert source["pairing"]["transaction_receipt_status_success_required"] is True
    assert source["pairing"]["companion_transfer_is_second_economic_event"] is False
    assert source["normalized_fields_only"][:4] == [
        "event_type",
        "supply_direction",
        "actor_address",
        "amount_raw",
    ]
    assert source["replay"]["attempts_per_request"] == 1
    assert source["replay"]["inter_batch_throttle"] == {
        "elapsed_seconds": 0.25,
        "applies_per_transport_after_first_batch": True,
        "cli_parameter": False,
        "bound_in_replay_claim_and_source_manifest": True,
    }
    assert source["replay"][
        "retry_backoff_provider_substitution_checkpoint_resume"
    ] is False
    assert source["raw_shape_validation"] == {
        "address": "exact frozen 20-byte log address",
        "Issue": {
            "topic_count": 1,
            "data_bytes": 32,
            "data_words": ["positive_uint256_amount_raw"],
        },
        "Redeem": {
            "topic_count": 1,
            "data_bytes": 32,
            "data_words": ["positive_uint256_amount_raw"],
        },
        "DestroyedBlackFunds": {
            "topic_count": 1,
            "data_bytes": 64,
            "data_words": [
                "exact_left_zero_padded_20_byte_actor_address",
                "positive_uint256_amount_raw",
            ],
        },
        "Deprecate": {
            "topic_count": 1,
            "data_bytes": 32,
            "data_words": [
                "exact_left_zero_padded_20_byte_replacement_address"
            ],
        },
        "Transfer": {
            "topic_count": 3,
            "data_bytes": 32,
            "indexed_address_words": [
                "exact_left_zero_padded_20_byte_from",
                "exact_left_zero_padded_20_byte_to",
            ],
            "data_words": ["positive_uint256_amount_raw"],
        },
        "quantities_topics_and_data_canonical": True,
        "extra_topics_short_or_long_data_nonzero_address_padding_or_zero_amount": (
            "terminal"
        ),
        "removed_must_be_false": True,
    }
    assert source["exact_event_labels"] == [
        "Issue",
        "Redeem",
        "DestroyedBlackFunds",
        "Deprecate",
    ]
    assert source["semantic_supply_direction"] == {
        "Issue": 1,
        "Redeem": -1,
        "DestroyedBlackFunds": -1,
        "Deprecate": 0,
    }
    boundary = source["pre_replay_evidence_boundary"]
    assert boundary["prior_contract_logs_or_receipts_opened"] == 0
    assert "metadata-only" in boundary["prior_precutoff_rpc"]
    assert boundary["replay_claim_required_before"] == [
        "first precutoff eth_getLogs request",
        "first precutoff eth_getTransactionReceipt request",
        "any event or source incidence",
    ]


def test_transport_serialization_rejects_synthetic_secret_surfaces() -> None:
    p.validate_sanitized_transports(copy.deepcopy(p.TRANSPORTS))
    for field, value in [
        ("url", "https://api.example.invalid/synthetic-secret-token"),
        ("path", "/synthetic-secret-token"),
        ("query", "token=synthetic-secret-token"),
        ("userinfo", "synthetic-user:synthetic-secret-token"),
        ("credential", "synthetic-secret-token"),
    ]:
        tampered = copy.deepcopy(p.TRANSPORTS)
        tampered[0][field] = value
        with pytest.raises(RuntimeError, match="not sanitized"):
            p.validate_sanitized_transports(tampered)


def test_grouping_side_execution_calendars_support_and_controls_are_exact() -> None:
    payload = manifest()
    feature = payload["feature_and_signal"]
    assert feature["group_key"] == "exact candidate_entry_open"
    assert feature["net_supply_raw"] == (
        "sum(Issue.amount_raw)-sum(Redeem.amount_raw)"
    )
    assert feature["side"] == {
        "positive": "LONG",
        "negative": "SHORT",
        "zero": "ABSTAIN",
    }
    assert feature[
        "rank_quantile_threshold_clipping_winsorization_or_amount_floor"
    ] is False
    assert feature["source_identity"]["sort_key"] == [
        "block_number",
        "transaction_index",
        "log_index",
        "transaction_hash",
        "event_type",
        "amount_raw",
    ]
    assert feature["source_identity"]["row_shape"] == "six-element JSON array"
    assert feature["source_identity"]["digest"] == (
        "SHA-256 of canonical constituent bytes"
    )
    execution = payload["execution"]
    assert execution["hold_hours"] == 168
    assert execution["hold_bars_5m"] == 2_016
    assert execution["standalone_leverage"] == 0.5
    assert execution["reservation"]["interval"] == "[entry_time,exit_time)"
    assert execution["reservation"]["suppressed_candidates_queued"] is False
    assert execution["candidate_order"] == [
        "entry_time",
        "decision_time",
        "source_identity",
        "side",
    ]
    scheduler = execution["scheduler"]
    assert scheduler["runs_per_independent_construction"] == 1
    assert scheduler["diagnostic_or_economic_period_nonoverlap_rerun"] is False
    assert scheduler["projected_periods"] == [
        "2023H2",
        "2024",
        "2024H1",
        "2024H2",
        "2025H1",
        "2025H2",
    ]
    assert "selection,future25,future26" in scheduler["steps"][1]
    assert "accepted union" in scheduler["steps"][3]
    assert "never truncated or replaced" in scheduler[
        "diagnostic_boundary_crossing_selection_trade"
    ]
    assert payload["calendars"]["selection_reports"] == {
        "2023H2": ["2023-06-01T00:00:00Z", "2024-01-01T00:00:00Z"],
        "2024H1": ["2024-01-01T00:00:00Z", "2024-07-01T00:00:00Z"],
        "2024H2": ["2024-07-01T00:00:00Z", "2025-01-01T00:00:00Z"],
    }
    support = payload["support_gates"]
    assert all(
        value == 0 for value in support["source_exact_zero_differences"].values()
    )
    assert support["accepted_trade_minimums"] == {
        "selection": 8,
        "2023H2": 2,
        "2024H1": 2,
        "2024H2": 2,
        "future25": 4,
        "2025H1": 1,
        "2025H2": 1,
        "future26": 2,
    }
    assert support["utc_entry_month_share"] == {
        "periods": ["selection", "future25", "future26"],
        "numerator": (
            "maximum count sharing the same UTC entry_time calendar month"
        ),
        "denominator": "all accepted primary entries in that period",
        "maximum_exact_inclusive": {"numerator": 1, "denominator": 2},
        "comparison": "integer cross multiplication",
        "zero_denominator": "terminal failure",
    }
    assert support["full_accepted_entry_gap"] == {
        "entries": "accepted primary entries contained in full",
        "sort": "strictly increasing entry_time integer UTC seconds",
        "differences": "consecutive entry timestamps only",
        "boundary_to_first_and_last_to_boundary_included": False,
        "minimum_entries": 2,
        "fewer_than_minimum": "terminal failure",
        "maximum_seconds_exact_inclusive": 240 * 86_400,
    }
    append = support["future_append_invariance"]
    assert append["fresh_constructions"] == {
        "prefix": (
            "source rows with available_at_utc < 2025-01-01T00:00:00Z"
        ),
        "full": "complete source artifact",
    }
    assert append["independent_constructions"] == [
        "primary",
        "issue_only",
        "redeem_only",
        "include_destroyed_black_funds",
        "count_net_side",
    ]
    assert append["compared_views"] == [
        "every raw candidate assigned to selection",
        "every accepted selection clock row",
    ]
    assert append["row_order_field_or_sha256_differences_allowed"] == 0
    controls = payload["controls"]
    assert controls["independent_own_bucket_and_scheduler"] == [
        "issue_only",
        "redeem_only",
        "include_destroyed_black_funds",
        "count_net_side",
    ]
    assert controls["same_primary_parent_no_regroup_or_nonoverlap_rerun"] == [
        "exact_direction_flip",
        "deterministic_random_side",
        "constant_long",
        "constant_short",
        "one_bar_delayed_entry",
    ]
    assert set(controls["definitions"]) == {
        "issue_only",
        "redeem_only",
        "include_destroyed_black_funds",
        "count_net_side",
        "exact_direction_flip",
        "deterministic_random_side",
        "constant_long",
        "constant_short",
        "one_bar_delayed_entry",
    }
    assert (
        controls["definitions"]["deterministic_random_side"]
        == "SHA256 UTF-8(source_identity|TUSI-168|RANDOM_SIDE); first byte "
        "<128 LONG else SHORT"
    )
    assert "+300 seconds" in controls["definitions"]["one_bar_delayed_entry"]
    assert controls["one_bar_delayed_projection"] == {
        "all_shifted_parent_rows_retained_in_support_artifact": True,
        "parent_main_window_label_role": "provenance only",
        "each_diagnostic_and_economic_projection": (
            "recheck both shifted timestamps against its own half-open period"
        ),
        "shifted_boundary_crosser_action": (
            "drop before any market or funding lookup"
        ),
        "dropped_report_fields": ["count", "source_identity"],
        "may_suppress_replace_or_reschedule_another_parent": False,
    }


def test_novelty_rationals_economics_and_same_gross_are_exact() -> None:
    payload = manifest()
    novelty = payload["novelty"]
    assert novelty["prior_source_family_thresholds_exact_inclusive"] == {
        "exact_entry_jaccard": {"numerator": 1, "denominator": 5},
        "candidate_24h_containment": {"numerator": 1, "denominator": 2},
        "squared_signed_exposure_pearson": {
            "numerator": 4,
            "denominator": 25,
        },
    }
    assert novelty[
        "gross9_each_positive_weight_sleeve_thresholds_exact_inclusive"
    ] == {
        "exact_entry_jaccard": {"numerator": 1, "denominator": 10},
        "candidate_6h_containment": {"numerator": 7, "denominator": 20},
        "occupied_bar_jaccard": {"numerator": 1, "denominator": 4},
        "squared_signed_exposure_pearson": {
            "numerator": 49,
            "denominator": 400,
        },
    }
    assert novelty["comparator_min_common_domain_entries_to_gate"] == 10
    assert novelty["minimum_applied_after_common_domain_filtering"] is True
    assert novelty["metric_contract"]["inclusive_threshold_comparison"] == (
        "exact integer cross multiplication"
    )
    assert novelty["metric_contract"]["terminal_undefined_inputs"] == [
        "duplicate_or_unsorted_timestamps",
        "empty_metric_denominator",
        "unequal_exposure_vectors",
        "zero_pearson_variance",
    ]
    gates = payload["economic_contract"][
        "standalone_gate_base_and_stress_each_opened_period"
    ]
    assert gates == {
        "absolute_return": ">0",
        "full_calendar_cagr_to_strict_mdd": ">=3.0",
        "strict_mdd": "<=0.15",
        "mean_gross_underlying_bp": ">=20",
        "calendar_month_clustered_signflip_p": "<=0.10",
    }
    economics = payload["economic_contract"]
    assert economics["accounting_code_authority"] == {
        "path": (
            "training/evaluate_ethereum_settlement_demand_"
            "impulse_economics.py"
        ),
        "sha256": (
            "fba7de6a26ede945edfe63c32dd4a0c"
            "88760c6459ac0d4f079dd12d546580235"
        ),
        "tusi_imports_strict_pure_helpers": True,
        "bound_by_later_source_replay_claim": True,
        "included_in_preregistration_repository_identity": False,
    }
    assert len(economics["standalone_accounting"]["strict_mdd_bar_order"]) == 5
    assert economics["standalone_accounting"][
        "nonpositive_liquidation_envelope_equity"
    ] == "terminal failure"
    accounting = economics["standalone_accounting"]
    assert accounting["position_formulas"]["quantity"] == (
        "allocated_equity * 0.5 / O"
    )
    assert accounting["position_formulas"]["funding_cash"] == (
        "-s * quantity * funding_rate * settlement_mark"
    )
    assert accounting["strict_mdd_formulas"]["upper_t"] == (
        "max(E_pre, E_pre - C_in + F_plus + A_plus)"
    )
    assert accounting["strict_mdd_formulas"]["lower_t"] == (
        "min(E_pre, E_pre - C_in - C_out + F_plus + F_minus "
        "+ A_minus - Q * P_bad * cost_rate)"
    )
    assert accounting["strict_mdd_formulas"][
        "C_out_is_single_exit_cost_event_not_second_charge"
    ] is True
    assert accounting["period_metrics"]["calendar_years_full"] == 3.0
    assert accounting["period_metrics"]["nonpositive_cagr_to_mdd"] == 0
    sign_flip = economics["calendar_month_clustered_sign_flip"]
    assert sign_flip["helper"] == "calendar_month_clustered_signflip"
    assert sign_flip["trade_record_order"] == [
        "entry_time",
        "exit_time",
        "source_identity",
    ]
    assert sign_flip["cluster"] == "UTC entry month"
    assert sign_flip["ordered_vector"] == (
        "NumPy float64 in ascending YYYY-MM order"
    )
    assert sign_flip["discard_only_when_absolute_sum_at_most"] == 1e-15
    assert sign_flip["observed"] == "left-to-right ordered-vector sum"
    assert sign_flip["observed_total_nonpositive_p"] == 1
    assert sign_flip["exact_enumeration_when_nonzero_months_at_most"] == 20
    assert sign_flip["exact_sign_stream"] == (
        "itertools.product((-1.0,1.0),repeat=m) in native order"
    )
    assert sign_flip["rng"] == "numpy.default_rng(20260730)"
    assert sign_flip["sign_vectors"] == 10_000
    assert sign_flip["batch_rows"] == [4_096, 4_096, 1_808]
    assert sign_flip["random_draw"] == (
        "rng.integers(0,2,size=(batch,m),dtype=np.int8)"
    )
    assert sign_flip["sign_conversion"] == (
        "signs.astype(np.float64)*2.0-1.0"
    )
    assert sign_flip["matrix_evaluation"] == "row-major signs @ ordered"
    assert sign_flip["monte_carlo_p"] == "(exceed+1)/10001"
    superiority = economics["independent_control_superiority"]
    assert superiority["metric"] == "full_calendar_CAGR / strict_MDD"
    assert superiority["comparison"] == "primary strictly greater"
    assert superiority["period_scope"] == "every opened standalone period"
    assert superiority["cost_scope"] == ["base", "stress"]
    assert superiority["gate_when_contained_accepted_trades_at_least"] == 1
    assert superiority["undefined_nonzero_support_metric"] == "terminal failure"
    same_parent = economics["same_primary_parent_complete_qualification"]
    assert same_parent["any_control_may_completely_qualify"] is False
    assert "all five standalone gates" in same_parent["definition"]
    assert "timing sensitivity" in same_parent["one_bar_delayed_entry"]
    gross9 = payload["gross9"]
    assert gross9["weights"] == {
        "cand_rex_veto_7": 1.6,
        "fresh_kimchi_fx": 2.0,
        "frozen_annual_rank7": 3.0,
        "markov_transition_long": 2.0,
        "rex_taker_low_range_position": 0.4,
    }
    assert sum(gross9["weights"].values()) == 9.0
    assert gross9["candidate_weights"] == [0.25, 0.50, 0.75, 1.00]
    assert "(9-w)/9" in gross9["treatment"]
    assert gross9["same_configured_gross"] == 9.0
    assert gross9["requirements"] == {
        "applies_to_every_candidate_weight": True,
        "periods": ["2023H2", "2024"],
        "cost_settings": ["base", "stress"],
        "base_and_stress_cagr_mdd_improvement_min": 0.05,
        "unscaled_absolute_return_retention_min": 0.97,
        "base_and_stress_absolute_return_positive": True,
        "strict_mdd_reduced_in_at_least_one_of_four_period_cost_cells": True,
    }
    assert gross9["future_contract"] == {
        "periods_evaluated_independently": ["future25", "future26"],
        "cost_settings": ["base", "stress"],
        "cagr_mdd_improvement_min_each_cost": 0.05,
        "baseline_absolute_return_retention_min_each_cost": 0.97,
        "treatment_return_positive_each_cost": True,
        "liquidation_safety_each_cost": True,
        "strict_mdd_lower_than_baseline_in_at_least_one_cost_per_period": True,
    }
    assert gross9["future_weight_grid_opened"] is False
    assert gross9["esdi_artifact_binding"] == {
        "path": str(p.ESDI_PREREGISTRATION_PATH),
        "file_sha256": p.ESDI_PREREGISTRATION_SHA256,
        "manifest_hash": p.ESDI_MANIFEST_HASH,
        "gross9_subtree_sha256": p.ESDI_GROSS9_SUBTREE_SHA256,
        "authority_subtree_sha256": p.ESDI_GROSS9_AUTHORITY_SHA256,
        "runtime_closure_subtree_sha256": p.ESDI_RUNTIME_CLOSURE_SHA256,
        "complete_authority_closure_and_environment_deep_copied": True,
        "current_closure_and_environment_match_required": True,
    }


def test_reused_novelty_functions_are_exact_and_fail_closed() -> None:
    assert p.entries_in_domain([0, 300, 600], 300, 900) == (300, 600)
    assert p.exact_entry_jaccard([0, 300], [300, 600]) == Fraction(1, 3)
    assert p.bidirectional_entry_containment(
        [0, 1_000], [100, 2_000], 150
    ) == Fraction(1, 2)
    assert p.fraction_at_most(Fraction(1, 2), 1, 2) is True
    assert p.fraction_at_most(Fraction(501, 1_000), 1, 2) is False
    left = p.signed_exposure_5m([(0, 600, 1)], 0, 1_200)
    right = p.signed_exposure_5m([(300, 900, -1)], 0, 1_200)
    assert p.occupied_bar_jaccard(left, right) == Fraction(1, 3)
    assert p.squared_signed_exposure_pearson(left, right) == Fraction(0, 1)

    with pytest.raises(ValueError, match="strictly increasing"):
        p.exact_entry_jaccard([300, 300], [300])
    with pytest.raises(ValueError, match="strictly increasing"):
        p.entries_in_domain([600, 300], 0, 900)
    with pytest.raises(ValueError, match="empty union"):
        p.exact_entry_jaccard([], [])
    with pytest.raises(ValueError, match="two nonempty"):
        p.bidirectional_entry_containment([], [300], 300)
    with pytest.raises(ValueError, match="equal length"):
        p.occupied_bar_jaccard([1], [1, 0])
    with pytest.raises(ValueError, match="share length"):
        p.squared_signed_exposure_pearson([1], [1, -1])
    with pytest.raises(ValueError, match="zero variance"):
        p.squared_signed_exposure_pearson([0, 0], [0, 1])


def test_esdi_registry_gross9_and_runtime_authorities_are_deep_copied() -> None:
    first_authority = p.load_esdi_preregistration_authority()
    second_authority = p.load_esdi_preregistration_authority()
    registry = first_authority["frozen_comparator_artifacts"]
    assert registry == esdi.frozen_comparator_registry()
    assert len(registry) == p.ESDI_COMPARATOR_COUNT == 18
    assert p.canonical_hash(registry) == p.ESDI_COMPARATOR_SUBTREE_SHA256
    assert p.canonical_hash(first_authority["gross9"]) == (
        p.ESDI_GROSS9_SUBTREE_SHA256
    )
    assert p.canonical_hash(first_authority["gross9"]["authority"]) == (
        p.ESDI_GROSS9_AUTHORITY_SHA256
    )
    closure = first_authority["gross9"]["authority"]["runtime_code_closure"]
    assert p.canonical_hash(closure) == p.ESDI_RUNTIME_CLOSURE_SHA256
    frozen = manifest()["frozen_preregistration"][
        "esdi_preregistration_authority"
    ]
    assert frozen["path"] == str(p.ESDI_PREREGISTRATION_PATH)
    assert frozen["file_sha256"] == p.ESDI_PREREGISTRATION_SHA256
    assert frozen["manifest_hash"] == p.ESDI_MANIFEST_HASH
    assert frozen["comparator_registry"]["item_count"] == 18
    assert frozen["comparator_registry"][
        "canonical_compact_sorted_sha256"
    ] == p.ESDI_COMPARATOR_SUBTREE_SHA256
    assert frozen["gross9_subtree"][
        "canonical_compact_sorted_sha256"
    ] == p.ESDI_GROSS9_SUBTREE_SHA256
    assert frozen["gross9_authority_subtree"][
        "canonical_compact_sorted_sha256"
    ] == p.ESDI_GROSS9_AUTHORITY_SHA256
    assert frozen["runtime_closure_subtree"][
        "canonical_compact_sorted_sha256"
    ] == p.ESDI_RUNTIME_CLOSURE_SHA256

    registry["CAIM"]["sha256"] = "0" * 64
    first_authority["gross9"]["authority"]["sleeves"].clear()
    assert second_authority["frozen_comparator_artifacts"]["CAIM"]["sha256"] != (
        "0" * 64
    )
    assert len(second_authority["gross9"]["authority"]["sleeves"]) == 5

    first = manifest()
    assert first["gross9"]["authority"] == second_authority["gross9"]["authority"]
    assert first["gross9"]["authority"] is not second_authority["gross9"]["authority"]
    first["novelty"]["frozen_comparator_artifacts"].clear()
    first["gross9"]["authority"]["sleeves"].clear()
    second = manifest()
    assert len(second["novelty"]["frozen_comparator_artifacts"]) == 18
    assert len(second["gross9"]["authority"]["sleeves"]) == 5
    closure = second["gross9"]["authority"]["runtime_code_closure"]
    assert closure["paths"] == [
        str(path) for path in p.RUNTIME_CODE_CLOSURE_PATHS
    ]
    assert closure["exact_runtime_environment"] == p.current_runtime_environment()


def test_future_protocol_paths_are_metadata_only_and_authorities_are_bound() -> None:
    identity = fake_repository_identity()
    paths = set(identity["sha256"])
    assert set(p.committed_identity_paths()) == {
        *p.PREREGISTRATION_PATHS,
        *p.RUNTIME_CODE_CLOSURE_PATHS,
    }
    assert set(p.FUTURE_PROTOCOL_PATHS).isdisjoint(
        {Path(path) for path in paths}
    )
    assert Path(
        "training/evaluate_ethereum_settlement_demand_impulse_economics.py"
    ) not in p.committed_identity_paths()
    assert p.PRODUCER_PATH in {Path(path) for path in paths}
    assert p.TEST_PATH in {Path(path) for path in paths}
    assert p.ESDI_HELPER_PATH in {Path(path) for path in paths}
    assert p.ESDI_PREREGISTRATION_PATH in {Path(path) for path in paths}
    assert set(p.RUNTIME_CODE_CLOSURE_PATHS) <= {Path(path) for path in paths}
    assert identity["sha256"][str(p.SOURCE_DECISION_PATH)] == (
        p.SOURCE_DECISION_SHA256
    )
    assert identity["sha256"][str(p.MECHANISM_DECISION_PATH)] == (
        p.MECHANISM_DECISION_SHA256
    )
    assert identity["sha256"][str(p.ESDI_HELPER_PATH)] == p.ESDI_HELPER_SHA256
    assert identity["sha256"][str(p.ESDI_PREREGISTRATION_PATH)] == (
        p.ESDI_PREREGISTRATION_SHA256
    )
    p.validate_repository_identity(identity)

    post_prereg = manifest()["frozen_preregistration"][
        "expected_post_preregistration_protocol"
    ]
    assert post_prereg["paths_metadata_only"] == [
        str(path) for path in p.FUTURE_PROTOCOL_PATHS
    ]
    assert post_prereg["included_in_preregistration_repository_identity"] is False
    assert post_prereg["existence_required_at_preregistration_creation"] is False
    assert post_prereg["bytes_or_hashes_read_by_preregistration_producer"] is False
    assert "artifact file SHA-256 and manifest_hash" in post_prereg[
        "required_later_binding"
    ]
    assert "seals their Git blobs and SHA-256" in post_prereg[
        "separate_source_replay_claim"
    ]
    sequence = manifest()["strict_sequence"]
    assert "committed and pushed" in sequence[0]
    assert sequence[1] == (
        "write_once_preregistration_artifact created committed and pushed"
    )
    assert "future builder evaluators and tests" in sequence[2]
    assert "artifact SHA-256 and manifest_hash" in sequence[2]
    assert "separate source-replay claim" in sequence[3]
    assert "first precutoff event/log/receipt replay RPC" in sequence[3]
    assert "metadata-only boundary headers acknowledged" in sequence[3]
    assert sequence[4] == "complete_source_replay_once"
    artifacts = manifest()["frozen_preregistration"][
        "downstream_artifact_contracts"
    ]
    assert artifacts["paths"] == {
        "preregistration": str(p.DEFAULT_OUTPUT),
        "source_replay_claim": (
            "results/tron_usdt_supply_events_source_replay_"
            "claim_2026-07-30.json"
        ),
        "primary_support_clock": (
            "results/tron_usdt_supply_impulse_primary_"
            "clock_2026-07-30.csv.gz"
        ),
        "control_support_clocks": (
            "results/tron_usdt_supply_impulse_control_"
            "clocks_2026-07-30.csv.gz"
        ),
        "source_support_report": (
            "results/tron_usdt_supply_impulse_source_"
            "support_2026-07-30.json"
        ),
    }
    circularity = artifacts["circularity_boundary"]
    assert circularity["future_protocol_paths_are_metadata_only"] is True
    assert circularity["not_yet_final_evaluator_hashes_in_preregistration"] is False
    assert circularity[
        "later_evaluators_hardcode_preregistration_artifact_file_sha256"
    ] is True
    assert artifacts["support_clock_csv"]["header_order"] == [
        "policy_id",
        "control",
        "window",
        "constituent_identities_json",
        "source_identity",
        "constituent_count",
        "bucket_amount_raw",
        "decision_time_utc",
        "entry_time_utc",
        "exit_time_utc",
        "side",
    ]
    assert artifacts["support_clock_csv"]["control_file_order"] == list(
        p.FROZEN_CONTROL_ORDER
    )
    assert artifacts["support_clock_csv"]["primary_file_control_values"] == [
        "primary"
    ]
    prereg_schema = artifacts["preregistration_json_schema"]
    assert prereg_schema["top_level_keys_and_types"] == (
        p.PREREGISTRATION_TOP_LEVEL_TYPES
    )
    assert prereg_schema["unknown_or_missing_keys"] == "terminal failure"
    report_schema = artifacts["source_support_report_json_schema"]
    assert report_schema["clock_artifact_keys"] == [
        "primary_sha256",
        "controls_sha256",
    ]
    assert report_schema["period_diagnostic_keys"] == [
        "selection",
        "2023H2",
        "2024",
        "2024H1",
        "2024H2",
        "future25",
        "2025H1",
        "2025H2",
        "future26",
        "full",
    ]

    tampered = copy.deepcopy(identity)
    tampered["sha256"][str(p.ESDI_HELPER_PATH)] = "0" * 64
    tampered["protocol_seal_hash"] = p.canonical_hash(
        {"git_blobs": tampered["git_blobs"], "sha256": tampered["sha256"]}
    )
    with pytest.raises(RuntimeError, match="helper hash drift"):
        p.validate_repository_identity(tampered)

    tampered = copy.deepcopy(identity)
    tampered["sha256"][str(p.ESDI_PREREGISTRATION_PATH)] = "0" * 64
    tampered["protocol_seal_hash"] = p.canonical_hash(
        {"git_blobs": tampered["git_blobs"], "sha256": tampered["sha256"]}
    )
    with pytest.raises(RuntimeError, match="preregistration hash drift"):
        p.validate_repository_identity(tampered)


def test_documents_helper_and_esdi_artifact_have_the_frozen_hashes() -> None:
    assert p.sha256_file(p.SOURCE_DECISION_PATH) == p.SOURCE_DECISION_SHA256
    assert p.sha256_file(p.MECHANISM_DECISION_PATH) == (
        p.MECHANISM_DECISION_SHA256
    )
    assert p.sha256_file(p.ESDI_HELPER_PATH) == p.ESDI_HELPER_SHA256
    assert p.sha256_file(p.ESDI_PREREGISTRATION_PATH) == (
        p.ESDI_PREREGISTRATION_SHA256
    )
    p.validate_frozen_documents_and_helper()


@pytest.mark.parametrize("tamper_target", ["registry", "gross9"])
def test_esdi_artifact_subtree_tamper_fails_before_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_target: str,
) -> None:
    source = Path(__file__).resolve().parents[1] / p.ESDI_PREREGISTRATION_PATH
    payload = json.loads(source.read_bytes())
    assert isinstance(payload, dict)
    if tamper_target == "registry":
        payload["novelty"]["frozen_comparator_artifacts"]["CAIM"]["sha256"] = (
            "0" * 64
        )
        message = "comparator registry drift"
    else:
        payload["gross9"]["weights"]["cand_rex_veto_7"] = 1.5
        message = "Gross9 authority drift"
    core = {
        key: value for key, value in payload.items() if key != "manifest_hash"
    }
    payload["manifest_hash"] = p.canonical_hash(core)
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
    artifact = tmp_path / "esdi.json"
    artifact.write_bytes(encoded)
    monkeypatch.setattr(p, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(p, "ESDI_PREREGISTRATION_PATH", Path("esdi.json"))
    monkeypatch.setattr(
        p, "ESDI_PREREGISTRATION_SHA256", hashlib.sha256(encoded).hexdigest()
    )
    monkeypatch.setattr(p, "ESDI_MANIFEST_HASH", payload["manifest_hash"])
    with pytest.raises(RuntimeError, match=message):
        p.load_esdi_preregistration_authority()


def test_wrong_document_hash_and_symlinked_dependency_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(p, "REPOSITORY_ROOT", tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "source.md"
    source.write_bytes(b"source")
    mechanism = docs / "mechanism.md"
    mechanism.write_bytes(b"wrong")
    helper = tmp_path / "helper.py"
    helper.write_bytes(b"helper")
    monkeypatch.setattr(p, "SOURCE_DECISION_PATH", Path("docs/source.md"))
    monkeypatch.setattr(p, "MECHANISM_DECISION_PATH", Path("docs/mechanism.md"))
    monkeypatch.setattr(p, "ESDI_HELPER_PATH", Path("helper.py"))
    monkeypatch.setattr(
        p, "SOURCE_DECISION_SHA256", hashlib.sha256(b"source").hexdigest()
    )
    monkeypatch.setattr(p, "MECHANISM_DECISION_SHA256", "0" * 64)
    monkeypatch.setattr(
        p, "ESDI_HELPER_SHA256", hashlib.sha256(b"helper").hexdigest()
    )
    with pytest.raises(RuntimeError, match="frozen dependency changed"):
        p.validate_frozen_documents_and_helper()

    mechanism.write_bytes(b"mechanism")
    monkeypatch.setattr(
        p,
        "MECHANISM_DECISION_SHA256",
        hashlib.sha256(b"mechanism").hexdigest(),
    )
    real = tmp_path / "real"
    real.mkdir()
    (real / "unsafe").write_bytes(b"x")
    (tmp_path / "linked").symlink_to(real, target_is_directory=True)
    with pytest.raises(RuntimeError, match="missing or unsafe"):
        p.sha256_file("linked/unsafe")


def test_manifest_tamper_and_evidence_boundary_are_rejected() -> None:
    payload = manifest()
    payload["unknown"] = False
    with pytest.raises(RuntimeError, match="top-level schema drift"):
        p.validate_manifest(payload)

    payload = manifest()
    del payload["source"]
    with pytest.raises(RuntimeError, match="top-level schema drift"):
        p.validate_manifest(payload)

    payload = manifest()
    payload["singleton"] = 1
    with pytest.raises(RuntimeError, match="top-level schema drift"):
        p.validate_manifest(payload)

    payload = manifest()
    payload["execution"]["hold_hours"] = 167
    with pytest.raises(RuntimeError, match="differs from frozen code"):
        p.validate_manifest(payload)

    payload = manifest()
    payload["outcomes_opened"] = True
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    payload["manifest_hash"] = p.canonical_hash(core)
    with pytest.raises(RuntimeError):
        p.validate_manifest(payload)
    assert all(manifest()[name] is False for name in p.EVIDENCE_BOUNDARIES)


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/out.json",
        "../results/out.json",
        "results/../out.json",
        "~/out.json",
        "results/other.json",
    ],
)
def test_output_path_escape_and_noncanonical_paths_fail(path: str) -> None:
    with pytest.raises(RuntimeError):
        p._output_relative(path)
    assert p._output_relative(p.DEFAULT_OUTPUT) == p.DEFAULT_OUTPUT


def _prepare_write_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], Path]:
    identity = fake_repository_identity()
    esdi_authority = p.load_esdi_preregistration_authority()
    monkeypatch.setattr(p, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        p, "frozen_repository_identity", lambda: copy.deepcopy(identity)
    )
    monkeypatch.setattr(p, "validate_frozen_documents_and_helper", lambda: None)
    monkeypatch.setattr(
        p,
        "load_esdi_preregistration_authority",
        lambda: copy.deepcopy(esdi_authority),
    )
    monkeypatch.setattr(
        p,
        "validate_existing_artifact_repository",
        lambda _identity, _artifact_bytes: None,
    )
    monkeypatch.setattr(
        p,
        "validate_creation_publish_state",
        lambda _identity, _temporary: None,
    )
    (tmp_path / "results").mkdir()
    return identity, tmp_path / p.DEFAULT_OUTPUT


def test_write_once_atomic_create_verify_and_overwrite_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity, artifact = _prepare_write_once(tmp_path, monkeypatch)
    payload = p.build_manifest(identity)
    status, written = p.write_once(p.DEFAULT_OUTPUT, payload)
    assert status == "created"
    assert written == payload
    assert artifact.read_bytes() == p.canonical_manifest_bytes(payload)
    assert stat_mode(artifact) == 0o444
    assert p.write_once(p.DEFAULT_OUTPUT, payload)[0] == "verified_existing"

    artifact.chmod(0o600)
    artifact.write_bytes(b"drift\n")
    with pytest.raises(RuntimeError, match="existing preregistration drift"):
        p.write_once(p.DEFAULT_OUTPUT, payload)


def test_existing_artifact_rejects_inode_swap_during_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity, artifact = _prepare_write_once(tmp_path, monkeypatch)
    payload = p.build_manifest(identity)
    assert p.write_once(payload=payload)[0] == "created"
    canonical = artifact.read_bytes()

    def swap_artifact(
        _identity: dict[str, Any],
        _artifact_bytes: bytes,
    ) -> None:
        artifact.rename(artifact.with_suffix(".original"))
        artifact.write_bytes(canonical)

    monkeypatch.setattr(
        p,
        "validate_existing_artifact_repository",
        swap_artifact,
    )
    with pytest.raises(RuntimeError, match="race drift"):
        p.write_once(payload=payload)


def _prepare_pushed_producer_with_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, tuple[Path, ...], dict[str, Any], str, Path]:
    esdi_authority = p.load_esdi_preregistration_authority()
    remote = tmp_path / "remote.git"
    _run_git(tmp_path, "init", "--bare", str(remote))
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(repo, "init", "-b", p.EXPECTED_BRANCH)
    _run_git(repo, "config", "user.email", "artifact@example.invalid")
    _run_git(repo, "config", "user.name", "Artifact Test")

    paths = (
        p.SOURCE_DECISION_PATH,
        p.MECHANISM_DECISION_PATH,
        p.PRODUCER_PATH,
        p.TEST_PATH,
        p.ESDI_HELPER_PATH,
        p.ESDI_PREREGISTRATION_PATH,
    )
    contents = {
        p.SOURCE_DECISION_PATH: b"source\n",
        p.MECHANISM_DECISION_PATH: b"mechanism\n",
        p.PRODUCER_PATH: b"producer\n",
        p.TEST_PATH: b"tests\n",
        p.ESDI_HELPER_PATH: b"helper\n",
        p.ESDI_PREREGISTRATION_PATH: b"esdi artifact\n",
    }
    for path, content in contents.items():
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    _run_git(repo, "add", *[str(path) for path in paths])
    _run_git(repo, "commit", "-m", "producer seal")
    _run_git(repo, "remote", "add", "origin", str(remote))
    _run_git(repo, "push", "-u", "origin", p.EXPECTED_BRANCH)
    producer_commit = _run_git(repo, "rev-parse", "HEAD")

    monkeypatch.setattr(p, "REPOSITORY_ROOT", repo)
    monkeypatch.setattr(p, "committed_identity_paths", lambda: paths)
    monkeypatch.setattr(
        p,
        "SOURCE_DECISION_SHA256",
        hashlib.sha256(contents[p.SOURCE_DECISION_PATH]).hexdigest(),
    )
    monkeypatch.setattr(
        p,
        "MECHANISM_DECISION_SHA256",
        hashlib.sha256(contents[p.MECHANISM_DECISION_PATH]).hexdigest(),
    )
    monkeypatch.setattr(
        p,
        "ESDI_HELPER_SHA256",
        hashlib.sha256(contents[p.ESDI_HELPER_PATH]).hexdigest(),
    )
    monkeypatch.setattr(
        p,
        "ESDI_PREREGISTRATION_SHA256",
        hashlib.sha256(contents[p.ESDI_PREREGISTRATION_PATH]).hexdigest(),
    )
    monkeypatch.setattr(p, "_BOOTSTRAP_HEAD_COMMIT", producer_commit)
    monkeypatch.setattr(
        p,
        "_BOOTSTRAP_HELPER_GIT_BLOB",
        _run_git(repo, "rev-parse", f"HEAD:{p.ESDI_HELPER_PATH}"),
    )
    monkeypatch.setattr(p, "validate_runtime_code_closure", lambda: None)
    monkeypatch.setattr(p, "validate_runtime_environment", lambda: None)
    monkeypatch.setattr(
        p,
        "load_esdi_preregistration_authority",
        lambda: copy.deepcopy(esdi_authority),
    )
    status, payload = p.write_once()
    assert status == "created"
    artifact = repo / p.DEFAULT_OUTPUT
    return repo, paths, payload, producer_commit, artifact


def test_existing_artifact_verifies_after_artifact_and_future_commits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, paths, payload, producer_commit, artifact = (
        _prepare_pushed_producer_with_artifact(tmp_path, monkeypatch)
    )
    _run_git(repo, "add", str(p.DEFAULT_OUTPUT))
    _run_git(repo, "commit", "-m", "artifact only")
    _run_git(repo, "push", "origin", p.EXPECTED_BRANCH)
    artifact_commit = _run_git(repo, "rev-parse", "HEAD")
    assert artifact_commit != producer_commit
    assert _run_git(repo, "rev-parse", "HEAD^") == producer_commit
    assert _run_git(
        repo,
        "diff",
        "--name-only",
        producer_commit,
        "HEAD",
    ) == str(p.DEFAULT_OUTPUT)

    artifact_status, artifact_verified = p.write_once(payload=payload)
    assert artifact_status == "verified_existing"
    assert artifact_verified == payload

    future_path = p.FUTURE_PROTOCOL_PATHS[0]
    assert future_path not in paths
    future_file = repo / future_path
    future_file.parent.mkdir(parents=True, exist_ok=True)
    future_file.write_text("PREREGISTRATION_SHA256 = 'synthetic'\\n")
    _run_git(repo, "add", str(future_path))
    _run_git(repo, "commit", "-m", "future protocol")
    _run_git(repo, "push", "origin", p.EXPECTED_BRANCH)
    assert _run_git(repo, "rev-parse", "HEAD^") == artifact_commit
    assert _run_git(
        repo,
        "diff",
        "--name-only",
        artifact_commit,
        "HEAD",
    ) == str(future_path)

    future_status, future_verified = p.write_once(payload=payload)
    assert future_status == "verified_existing"
    assert future_verified == payload
    assert artifact.read_bytes() == p.canonical_manifest_bytes(payload)

    (repo / p.PRODUCER_PATH).write_text("changed producer\\n")
    _run_git(repo, "add", str(p.PRODUCER_PATH))
    _run_git(repo, "commit", "-m", "invalid bound descendant")
    _run_git(repo, "push", "origin", p.EXPECTED_BRANCH)
    with pytest.raises(RuntimeError, match="committed Git blob"):
        p.write_once(payload=payload)


@pytest.mark.parametrize(
    ("scenario", "expected_error"),
    [
        ("non_direct_child", "not the producer direct child"),
        ("extra_artifact_commit_path", "changed non-artifact paths"),
        ("later_artifact_drift", "changed after its write-once"),
        ("later_bound_drift", "bound path changed after"),
    ],
)
def test_existing_artifact_rejects_invalid_pushed_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    expected_error: str,
) -> None:
    repo, paths, payload, _producer_commit, artifact = (
        _prepare_pushed_producer_with_artifact(tmp_path, monkeypatch)
    )
    future_path = p.FUTURE_PROTOCOL_PATHS[0]
    assert future_path not in paths
    future_file = repo / future_path

    if scenario == "non_direct_child":
        future_file.parent.mkdir(parents=True, exist_ok=True)
        future_file.write_text("intermediate = True\n")
        _run_git(repo, "add", str(future_path))
        _run_git(repo, "commit", "-m", "intermediate future path")
        _run_git(repo, "add", str(p.DEFAULT_OUTPUT))
        _run_git(repo, "commit", "-m", "late artifact add")
    elif scenario == "extra_artifact_commit_path":
        future_file.parent.mkdir(parents=True, exist_ok=True)
        future_file.write_text("extra = True\n")
        _run_git(
            repo,
            "add",
            str(p.DEFAULT_OUTPUT),
            str(future_path),
        )
        _run_git(repo, "commit", "-m", "artifact plus extra path")
    elif scenario == "later_artifact_drift":
        original = artifact.read_bytes()
        _run_git(repo, "add", str(p.DEFAULT_OUTPUT))
        _run_git(repo, "commit", "-m", "artifact only")
        artifact.chmod(0o644)
        artifact.write_bytes(original + b" ")
        _run_git(repo, "add", str(p.DEFAULT_OUTPUT))
        _run_git(repo, "commit", "-m", "artifact drift")
        artifact.write_bytes(original)
        artifact.chmod(0o444)
        _run_git(repo, "add", str(p.DEFAULT_OUTPUT))
        _run_git(repo, "commit", "-m", "artifact restored")
    else:
        bound_file = repo / p.PRODUCER_PATH
        original = bound_file.read_bytes()
        _run_git(repo, "add", str(p.DEFAULT_OUTPUT))
        _run_git(repo, "commit", "-m", "artifact only")
        bound_file.write_bytes(b"mutated producer\n")
        _run_git(repo, "add", str(p.PRODUCER_PATH))
        _run_git(repo, "commit", "-m", "bound path drift")
        bound_file.write_bytes(original)
        _run_git(repo, "add", str(p.PRODUCER_PATH))
        _run_git(repo, "commit", "-m", "bound path restored")

    _run_git(repo, "push", "origin", p.EXPECTED_BRANCH)
    with pytest.raises(RuntimeError, match=expected_error):
        p.write_once(payload=payload)


def test_existing_artifact_rejects_git_ref_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, paths, payload, _producer_commit, _artifact = (
        _prepare_pushed_producer_with_artifact(tmp_path, monkeypatch)
    )
    _run_git(repo, "add", str(p.DEFAULT_OUTPUT))
    _run_git(repo, "commit", "-m", "artifact only")
    _run_git(repo, "push", "origin", p.EXPECTED_BRANCH)

    real_git = p._git
    raced = False

    def racing_git(*arguments: str) -> bytes:
        nonlocal raced
        if arguments == ("rev-parse", "HEAD", "@{upstream}") and not raced:
            raced = True
            race_path = p.FUTURE_PROTOCOL_PATHS[1]
            assert race_path not in paths
            target = repo / race_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("race = True\n")
            _run_git(repo, "add", str(race_path))
            _run_git(repo, "commit", "-m", "racing future commit")
            _run_git(repo, "push", "origin", p.EXPECTED_BRANCH)
        return real_git(*arguments)

    monkeypatch.setattr(p, "_git", racing_git)
    with pytest.raises(RuntimeError, match="Git identity changed"):
        p.write_once(payload=payload)
    assert raced is True


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_write_once_rejects_symlink_parent_target_and_fifo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = fake_repository_identity()
    esdi_authority = p.load_esdi_preregistration_authority()
    monkeypatch.setattr(p, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        p, "frozen_repository_identity", lambda: copy.deepcopy(identity)
    )
    monkeypatch.setattr(p, "validate_frozen_documents_and_helper", lambda: None)
    monkeypatch.setattr(
        p,
        "load_esdi_preregistration_authority",
        lambda: copy.deepcopy(esdi_authority),
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "results").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="parent is unsafe"):
        p.write_once()

    (tmp_path / "results").unlink()
    (tmp_path / "results").mkdir()
    artifact = tmp_path / p.DEFAULT_OUTPUT
    artifact.symlink_to(outside / "target.json")
    with pytest.raises(RuntimeError, match="unsafe"):
        p.write_once()

    artifact.unlink()
    os.mkfifo(artifact)
    with pytest.raises(RuntimeError, match="regular file"):
        p.write_once()


def _run_git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_repository_identity_binds_head_upstream_clean_blobs_and_rejects_dirty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = tmp_path / "remote.git"
    _run_git(tmp_path, "init", "--bare", str(remote))
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(repo, "init", "-b", p.EXPECTED_BRANCH)
    _run_git(repo, "config", "user.email", "tusi@example.invalid")
    _run_git(repo, "config", "user.name", "TUSI Test")

    paths = (
        p.SOURCE_DECISION_PATH,
        p.MECHANISM_DECISION_PATH,
        p.ESDI_HELPER_PATH,
        p.ESDI_PREREGISTRATION_PATH,
        Path("training/sealed.py"),
    )
    contents = {
        p.SOURCE_DECISION_PATH: b"source\n",
        p.MECHANISM_DECISION_PATH: b"mechanism\n",
        p.ESDI_HELPER_PATH: b"helper\n",
        p.ESDI_PREREGISTRATION_PATH: b"esdi artifact\n",
        Path("training/sealed.py"): b"sealed = True\n",
    }
    for path, content in contents.items():
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    _run_git(repo, "add", *[str(path) for path in paths])
    _run_git(repo, "commit", "-m", "seal")
    _run_git(repo, "remote", "add", "origin", str(remote))
    _run_git(repo, "push", "-u", "origin", p.EXPECTED_BRANCH)

    monkeypatch.setattr(p, "REPOSITORY_ROOT", repo)
    monkeypatch.setattr(p, "committed_identity_paths", lambda: paths)
    monkeypatch.setattr(
        p,
        "SOURCE_DECISION_SHA256",
        hashlib.sha256(contents[p.SOURCE_DECISION_PATH]).hexdigest(),
    )
    monkeypatch.setattr(
        p,
        "MECHANISM_DECISION_SHA256",
        hashlib.sha256(contents[p.MECHANISM_DECISION_PATH]).hexdigest(),
    )
    monkeypatch.setattr(
        p,
        "ESDI_HELPER_SHA256",
        hashlib.sha256(contents[p.ESDI_HELPER_PATH]).hexdigest(),
    )
    monkeypatch.setattr(
        p,
        "ESDI_PREREGISTRATION_SHA256",
        hashlib.sha256(contents[p.ESDI_PREREGISTRATION_PATH]).hexdigest(),
    )
    monkeypatch.setattr(
        p,
        "_BOOTSTRAP_HEAD_COMMIT",
        _run_git(repo, "rev-parse", "HEAD"),
    )
    monkeypatch.setattr(
        p,
        "_BOOTSTRAP_HELPER_GIT_BLOB",
        _run_git(repo, "rev-parse", f"HEAD:{p.ESDI_HELPER_PATH}"),
    )
    monkeypatch.setattr(p, "validate_runtime_code_closure", lambda: None)
    monkeypatch.setattr(p, "validate_runtime_environment", lambda: None)
    identity = p.frozen_repository_identity()
    assert identity["head_commit"] == identity["upstream_commit"]
    assert identity["upstream_ref"] == (
        f"refs/remotes/origin/{p.EXPECTED_BRANCH}"
    )
    assert identity["whole_worktree_clean_required"] is True
    assert set(identity["git_blobs"]) == {str(path) for path in paths}

    expected_temporary = p.DEFAULT_OUTPUT.parent / ".publish.tmp"
    (repo / expected_temporary).write_text("temporary\n")
    late_unbound = repo / "late-unbound.txt"
    late_unbound.write_text("late dirty\n")
    with pytest.raises(RuntimeError, match="changed before artifact publish"):
        p.validate_creation_publish_state(identity, expected_temporary)
    (repo / expected_temporary).unlink()
    late_unbound.unlink()

    unbound = repo / "unbound.txt"
    unbound.write_text("dirty unbound\n")
    with pytest.raises(RuntimeError, match="clean whole worktree"):
        p.frozen_repository_identity()
    unbound.unlink()

    (repo / "training/sealed.py").write_bytes(b"dirty = True\n")
    with pytest.raises(RuntimeError, match="clean whole worktree"):
        p.frozen_repository_identity()


def test_repository_identity_ignores_future_paths_but_rejects_missing_bound_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = fake_repository_identity()
    paths = tuple(Path(path) for path in sorted(expected["git_blobs"]))
    monkeypatch.setattr(p, "committed_identity_paths", lambda: paths)
    monkeypatch.setattr(p, "validate_frozen_documents_and_helper", lambda: None)
    monkeypatch.setattr(p, "validate_runtime_code_closure", lambda: None)
    monkeypatch.setattr(p, "validate_runtime_environment", lambda: None)
    monkeypatch.setattr(p, "_BOOTSTRAP_HEAD_COMMIT", "a" * 40)
    monkeypatch.setattr(
        p,
        "_BOOTSTRAP_HELPER_GIT_BLOB",
        expected["git_blobs"][str(p.ESDI_HELPER_PATH)],
    )
    monkeypatch.setattr(
        p,
        "_committed_file_sha256",
        lambda path, _blob: expected["sha256"][str(path)],
    )
    calls: list[tuple[str, ...]] = []

    def fake_git_complete(*args: str) -> bytes:
        calls.append(args)
        if args[0] == "status":
            return (
                f"# branch.oid {'a' * 40}\n"
                f"# branch.head {p.EXPECTED_BRANCH}\n"
                f"# branch.upstream origin/{p.EXPECTED_BRANCH}\n"
                "# branch.ab +0 -0\n"
            ).encode()
        if args[0] == "rev-parse" and "--show-toplevel" in args:
            return f"{p.REPOSITORY_ROOT}\n{'b' * 40}\n".encode()
        records = []
        for path, blob in expected["git_blobs"].items():
            records.append(f"100644 blob {blob}\t{path}".encode())
        return b"\0".join(records) + b"\0"

    monkeypatch.setattr(p, "_git", fake_git_complete)
    assert p.frozen_repository_identity() == expected
    scoped_arguments = {
        argument
        for call in calls
        if call[0] in {"status", "ls-tree"}
        for argument in call
    }
    assert all(str(path) not in scoped_arguments for path in p.FUTURE_PROTOCOL_PATHS)

    def fake_git_missing(*args: str) -> bytes:
        result = fake_git_complete(*args)
        if args[0] != "ls-tree":
            return result
        records = []
        for path, blob in list(expected["git_blobs"].items())[:-1]:
            records.append(f"100644 blob {blob}\t{path}".encode())
        return b"\0".join(records) + b"\0"

    monkeypatch.setattr(p, "_git", fake_git_missing)
    with pytest.raises(RuntimeError, match="blobs are incomplete"):
        p.frozen_repository_identity()


def test_build_manifest_opens_no_network_source_comparator_or_outcome_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_dependency_bytes = p._dependency_bytes
    opened: list[Path] = []

    def tracked_dependency_bytes(path: str | Path) -> bytes:
        candidate = Path(path)
        opened.append(candidate)
        assert candidate == p.ESDI_PREREGISTRATION_PATH
        return original_dependency_bytes(candidate)

    def forbidden_subprocess(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("build_manifest attempted a subprocess")

    monkeypatch.setattr(p, "_dependency_bytes", tracked_dependency_bytes)
    monkeypatch.setattr(p.subprocess, "run", forbidden_subprocess)
    payload = manifest()
    assert opened == [p.ESDI_PREREGISTRATION_PATH]
    assert payload["producer_effects"] == {
        "network_calls": 0,
        "non_git_subprocess_calls": 0,
        "helper_import_preflight_git_read_only_subprocess_calls": 2,
        "artifact_creation_git_read_only_subprocess_calls": 4,
        "git_read_only_subprocess_calls": 6,
        "source_csv_or_source_rows_opened": 0,
        "esdi_preregistration_metadata_artifacts_opened": 1,
        "comparator_data_rows_opened": 0,
        "gross9_data_rows_opened": 0,
        "market_rows_opened": 0,
        "funding_rows_opened": 0,
        "future_protocol_files_opened_or_hashed": 0,
        "bound_committed_paths_hashed": len(p.committed_identity_paths()),
    }
    result_paths = {
        path
        for path in p.committed_identity_paths()
        if str(path).startswith("results/")
    }
    assert result_paths == {p.ESDI_PREREGISTRATION_PATH}
    assert not any(
        str(path).startswith("data/") for path in p.committed_identity_paths()
    )
    assert set(p.FUTURE_PROTOCOL_PATHS).isdisjoint(
        p.committed_identity_paths()
    )
