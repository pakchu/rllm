"""Probe the outcome-blind PSIM-D3 targeted batch-hydration transport."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(
    "results/protocol_specification_intent_maturity_d3_transport_probe_"
    "2026-07-25.json"
)
PROTOCOL_VERSION = "psim_d3_batch_hydration_transport_probe_v1"
GIT_VERSION = "git version 2.43.0"
GIT_BINARY = Path("/usr/bin/git")
GIT_BINARY_SHA256 = (
    "2a8c18fbf43da9f692d75474c72bea9dfd796c260b0f3dfe456376abc3bbd668"
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")

BULK_FETCH_ARGUMENTS = (
    "-c",
    "fetch.negotiationAlgorithm=noop",
    "fetch",
    "origin",
    "--no-tags",
    "--no-write-fetch-head",
    "--recurse-submodules=no",
    "--filter=blob:none",
    "--no-auto-maintenance",
    "--stdin",
)


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def canonical_json_bytes(payload: Any, *, pretty: bool = True) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_hash(payload: Any) -> str:
    return sha256_bytes(canonical_json_bytes(payload, pretty=False).rstrip(b"\n"))


def _git_environment(
    additions: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "LANG": "C",
            "LC_ALL": "C",
        }
    )
    if additions:
        environment.update(additions)
    return environment


def _run(
    arguments: Sequence[str],
    *,
    cwd: Path | None = None,
    input_bytes: bytes | None = None,
    additions: Mapping[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        list(arguments),
        cwd=cwd,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_git_environment(additions),
        check=False,
        timeout=60,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"PSIM-D3 synthetic command failed ({completed.returncode}): "
            f"{' '.join(arguments)}: {detail}"
        )
    return completed


def _git(
    repo: Path | None,
    arguments: Sequence[str],
    **kwargs: Any,
) -> subprocess.CompletedProcess[bytes]:
    command = [str(GIT_BINARY)]
    if repo is not None:
        command.extend(["-C", str(repo)])
    command.extend(arguments)
    return _run(command, **kwargs)


def _git_text(repo: Path | None, arguments: Sequence[str]) -> str:
    return _git(repo, arguments).stdout.decode("utf-8").strip()


def _pack_roster(repo: Path, suffix: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.name
            for path in (repo / "objects" / "pack").glob(f"*.{suffix}")
        )
    )


def _loose_object_roster(repo: Path) -> tuple[str, ...]:
    object_root = repo / "objects"
    rows: list[str] = []
    for prefix in sorted(object_root.iterdir()):
        if not prefix.is_dir() or re.fullmatch(r"[0-9a-f]{2}", prefix.name) is None:
            continue
        for candidate in sorted(prefix.iterdir()):
            if (
                re.fullmatch(r"[0-9a-f]{38}", candidate.name) is None
                or candidate.is_symlink()
                or not candidate.is_file()
            ):
                raise RuntimeError(
                    "PSIM-D3 loose-object roster contains an unsafe entry"
                )
            rows.append(f"{prefix.name}/{candidate.name}")
    return tuple(rows)


def _pack_objects(pack_path: Path) -> dict[str, str]:
    output = _git(None, ["verify-pack", "-v", str(pack_path)]).stdout.decode(
        "utf-8"
    )
    values: dict[str, str] = {}
    for line in output.splitlines():
        fields = line.split()
        if (
            len(fields) >= 2
            and HEX40.fullmatch(fields[0])
            and fields[1] in {"blob", "commit", "tag", "tree"}
        ):
            values[fields[0]] = fields[1]
    return dict(sorted(values.items()))


def _local_objects(repo: Path) -> dict[str, str]:
    output = _git(
        repo,
        [
            "cat-file",
            "--batch-all-objects",
            "--batch-check=%(objectname) %(objecttype)",
        ],
        additions={"GIT_NO_LAZY_FETCH": "1"},
    ).stdout.decode("utf-8")
    values: dict[str, str] = {}
    for line in output.splitlines():
        fields = line.split()
        if (
            len(fields) == 2
            and HEX40.fullmatch(fields[0])
            and fields[1] in {"blob", "commit", "tag", "tree"}
        ):
            values[fields[0]] = fields[1]
        else:
            raise RuntimeError(
                "PSIM-D3 local object inventory is malformed"
            )
    return dict(sorted(values.items()))


def _trace_child_arguments(path: Path) -> list[tuple[str, ...]]:
    children: list[tuple[str, ...]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if payload.get("event") != "child_start":
            continue
        arguments = payload.get("argv")
        if not isinstance(arguments, list) or not arguments or not all(
            isinstance(value, str) and value for value in arguments
        ):
            raise RuntimeError("PSIM-D3 trace child argv is ambiguous")
        children.append(tuple(arguments))
    return children


def _is_maintenance_child(arguments: Sequence[str]) -> bool:
    executable = Path(arguments[0]).name if arguments else ""
    return (
        executable in {"git-gc", "git-maintenance"}
        or "maintenance" in arguments
        or ("gc" in arguments and "--auto" in arguments)
    )


def _is_fetch_child(arguments: Sequence[str]) -> bool:
    if not arguments:
        return False
    executable = Path(arguments[0]).name
    return (
        executable.startswith("git-remote-")
        or executable in {"git-fetch", "git-fetch-pack", "git-upload-pack"}
        or "fetch" in arguments
        or "fetch-pack" in arguments
    )


def _ref_roster(repo: Path) -> tuple[str, ...]:
    output = _git_text(
        repo,
        ["for-each-ref", "--format=%(refname) %(objectname)"],
    )
    return tuple(sorted(line for line in output.splitlines() if line))


def _validate_hydration_delta(
    *,
    before_packs: Sequence[str],
    after_packs: Sequence[str],
    before_promisors: Sequence[str],
    after_promisors: Sequence[str],
    before_loose: Sequence[str],
    after_loose: Sequence[str],
    before_objects: Mapping[str, str],
    after_objects: Mapping[str, str],
    new_pack_objects: Mapping[str, Mapping[str, str]],
    requested: Sequence[str],
) -> dict[str, Any]:
    before_pack_set = set(before_packs)
    after_pack_set = set(after_packs)
    before_promisor_set = set(before_promisors)
    after_promisor_set = set(after_promisors)
    before_loose_set = set(before_loose)
    after_loose_set = set(after_loose)
    if (
        not before_pack_set.issubset(after_pack_set)
        or not before_promisor_set.issubset(after_promisor_set)
        or not before_loose_set.issubset(after_loose_set)
        or any(
            after_objects.get(oid) != object_type
            for oid, object_type in before_objects.items()
        )
    ):
        raise RuntimeError("PSIM-D3 hydration removed or replaced local objects")

    new_packs = tuple(sorted(after_pack_set - before_pack_set))
    new_promisors = tuple(
        sorted(after_promisor_set - before_promisor_set)
    )
    new_loose = tuple(sorted(after_loose_set - before_loose_set))
    expected_promisors = tuple(
        sorted(name.removesuffix(".pack") + ".promisor" for name in new_packs)
    )
    if (
        not new_packs
        or set(new_pack_objects) != set(new_packs)
        or new_promisors != expected_promisors
        or new_loose
    ):
        raise RuntimeError("PSIM-D3 hydration pack/promisor roster changed")

    expected = {oid: "blob" for oid in requested}
    packed_union: dict[str, str] = {}
    for objects in new_pack_objects.values():
        for oid, object_type in objects.items():
            prior = packed_union.setdefault(oid, object_type)
            if prior != object_type:
                raise RuntimeError(
                    "PSIM-D3 new packs disagree on an object type"
                )
    object_store_delta = {
        oid: object_type
        for oid, object_type in after_objects.items()
        if oid not in before_objects
    }
    if packed_union != expected or object_store_delta != expected:
        raise RuntimeError(
            "PSIM-D3 hydration object set differs from requested blobs"
        )
    return {
        "new_loose_objects": len(new_loose),
        "new_pack_count": len(new_packs),
        "new_pack_names": list(new_packs),
        "new_promisor_count": len(new_promisors),
        "new_total_object_count": len(object_store_delta),
    }


def _object_store_snapshot(repo: Path) -> dict[str, Any]:
    return {
        "fetch_head_absent": not (repo / "FETCH_HEAD").exists(),
        "loose_objects": list(_loose_object_roster(repo)),
        "objects": _local_objects(repo),
        "packs": list(_pack_roster(repo, "pack")),
        "promisors": list(_pack_roster(repo, "promisor")),
        "refs": list(_ref_roster(repo)),
    }


def _assert_post_read_invariant(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> None:
    if before != after or after.get("fetch_head_absent") is not True:
        raise RuntimeError("PSIM-D3 post-hydration object store changed")


def _make_synthetic_origin(root: Path) -> tuple[Path, Path, tuple[str, ...], str]:
    work = root / "work"
    origin = root / "origin.git"
    _git(None, ["init", "-q", "-b", "master", str(work)])
    _git(work, ["config", "user.name", "PSIM Synthetic"])
    _git(work, ["config", "user.email", "psim@example.invalid"])
    (work / "proposals").mkdir()
    (work / "noise").mkdir()
    for index in range(1, 9):
        token = f"{index:02d}"
        (work / "proposals" / f"p-{token}.md").write_text(
            f"proposal {token}\npayload-{token}-{index:040d}\n",
            encoding="utf-8",
        )
        (work / "noise" / f"n-{token}.txt").write_text(
            f"noise {token}\n",
            encoding="utf-8",
        )
        _git(work, ["add", "."])
        timestamp = f"2021-01-{token}T00:00:00Z"
        _git(
            work,
            ["commit", "-q", "-m", f"synthetic {token}"],
            additions={
                "GIT_AUTHOR_DATE": timestamp,
                "GIT_COMMITTER_DATE": timestamp,
            },
        )
    _git(None, ["clone", "-q", "--bare", str(work), str(origin)])
    _git(origin, ["config", "uploadpack.allowFilter", "true"])
    _git(origin, ["config", "uploadpack.allowAnySHA1InWant", "true"])
    requested = tuple(
        sorted(
            _git_text(work, ["rev-parse", f"HEAD:proposals/p-{index:02d}.md"])
            for index in (1, 2, 3, 4, 5, 6)
        )
    )
    nonrequested = _git_text(
        work,
        ["rev-parse", "HEAD:proposals/p-08.md"],
    )
    if (
        len(requested) != 6
        or len(set(requested)) != 6
        or any(HEX40.fullmatch(oid) is None for oid in requested)
        or HEX40.fullmatch(nonrequested) is None
    ):
        raise RuntimeError("PSIM-D3 synthetic blob identities are malformed")
    return work, origin, requested, nonrequested


def _partial_clone(origin: Path, destination: Path) -> None:
    _git(
        None,
        [
            "clone",
            "-q",
            "--bare",
            "--filter=blob:none",
            "--single-branch",
            "--branch",
            "master",
            "--no-tags",
            f"file://{origin}",
            str(destination),
        ],
    )


def _assert_oids_absent_from_object_store(
    repo: Path,
    object_ids: Sequence[str],
) -> None:
    local_objects = _local_objects(repo)
    if set(local_objects).intersection(object_ids):
        raise RuntimeError(
            "PSIM-D3 blob:none clone unexpectedly contains target blobs"
        )


def _bulk_fetch_probe(
    root: Path,
    origin: Path,
    requested: Sequence[str],
    nonrequested: str,
) -> dict[str, Any]:
    clone = root / "bulk.git"
    trace_fetch = root / "bulk-fetch.trace.json"
    trace_read = root / "bulk-read.trace.json"
    _partial_clone(origin, clone)
    _assert_oids_absent_from_object_store(
        clone,
        [*requested, nonrequested],
    )

    before_packs = _pack_roster(clone, "pack")
    before_promisors = _pack_roster(clone, "promisor")
    before_loose = _loose_object_roster(clone)
    before_objects = _local_objects(clone)
    before_refs = _ref_roster(clone)
    fetch_head_absent_before = not (clone / "FETCH_HEAD").exists()
    request_bytes = ("\n".join(requested) + "\n").encode("ascii")
    _git(
        clone,
        BULK_FETCH_ARGUMENTS,
        input_bytes=request_bytes,
        additions={"GIT_TRACE2_EVENT": str(trace_fetch)},
    )

    after_packs = _pack_roster(clone, "pack")
    after_promisors = _pack_roster(clone, "promisor")
    after_loose = _loose_object_roster(clone)
    after_objects = _local_objects(clone)
    after_refs = _ref_roster(clone)
    new_pack_names = sorted(set(after_packs) - set(before_packs))
    new_pack_objects = {
        name: _pack_objects(clone / "objects" / "pack" / name)
        for name in new_pack_names
    }
    delta = _validate_hydration_delta(
        before_packs=before_packs,
        after_packs=after_packs,
        before_promisors=before_promisors,
        after_promisors=after_promisors,
        before_loose=before_loose,
        after_loose=after_loose,
        before_objects=before_objects,
        after_objects=after_objects,
        new_pack_objects=new_pack_objects,
        requested=requested,
    )
    if (
        before_refs != after_refs
        or (clone / "FETCH_HEAD").exists()
        or not fetch_head_absent_before
    ):
        raise RuntimeError("PSIM-D3 bulk-fetch boundary changed")

    read_input = ("\n".join(requested) + "\n").encode("ascii")
    _git(
        clone,
        ["cat-file", "--batch"],
        input_bytes=read_input,
        additions={
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_TRACE2_EVENT": str(trace_read),
        },
    )
    after_read_snapshot = _object_store_snapshot(clone)
    expected_after_read_snapshot = {
        "fetch_head_absent": True,
        "loose_objects": list(after_loose),
        "objects": after_objects,
        "packs": list(after_packs),
        "promisors": list(after_promisors),
        "refs": list(after_refs),
    }
    _assert_post_read_invariant(
        expected_after_read_snapshot,
        after_read_snapshot,
    )
    fetch_children = _trace_child_arguments(trace_fetch)
    read_children = _trace_child_arguments(trace_read)
    if (
        any(_is_maintenance_child(row) for row in fetch_children)
        or any(_is_fetch_child(row) for row in read_children)
    ):
        raise RuntimeError("PSIM-D3 post-hydration local-only read changed")

    return {
        "command": [
            str(GIT_BINARY),
            "-C",
            "<fresh-bare-root>",
            *BULK_FETCH_ARGUMENTS,
        ],
        "fetch_invocations": 1,
        "fetch_head_absent": True,
        "maintenance_child_processes": sum(
            _is_maintenance_child(row) for row in fetch_children
        ),
        "new_loose_object_count": delta["new_loose_objects"],
        "new_pack_exact_requested_blobs": True,
        "new_pack_object_count": delta["new_total_object_count"],
        "new_total_object_store_exact_requested_blobs": True,
        "nonrequested_blob_present": (
            nonrequested in set(after_objects) - set(before_objects)
        ),
        "pack_delta": len(after_packs) - len(before_packs),
        "post_hydration_fetch_child_processes": sum(
            _is_fetch_child(row) for row in read_children
        ),
        "post_hydration_object_store_unchanged": True,
        "post_hydration_pack_delta": (
            len(after_read_snapshot["packs"]) - len(after_packs)
        ),
        "promisor_pack_delta": len(after_promisors) - len(before_promisors),
        "refs_unchanged": (
            before_refs
            == after_refs
            == tuple(after_read_snapshot["refs"])
        ),
        "requested_blob_count": len(requested),
        "stderr_consumption": "subprocess_run_communicate",
    }


def _no_lazy_fetch_probe(
    root: Path,
    origin: Path,
    missing_oid: str,
) -> dict[str, Any]:
    clone = root / "no-lazy.git"
    trace = root / "no-lazy.trace.json"
    _partial_clone(origin, clone)
    _assert_oids_absent_from_object_store(clone, [missing_oid])
    before = _pack_roster(clone, "pack")
    before_objects = _local_objects(clone)
    completed = _git(
        clone,
        ["cat-file", "-e", missing_oid],
        additions={
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_TRACE2_EVENT": str(trace),
        },
        check=False,
    )
    after = _pack_roster(clone, "pack")
    after_objects = _local_objects(clone)
    children = _trace_child_arguments(trace)
    stderr = completed.stderr.decode("utf-8", errors="replace")
    supported = (
        completed.returncode != 0
        and before == after
        and before_objects == after_objects
        and "lazy fetching disabled" in stderr
        and not any(_is_fetch_child(row) for row in children)
    )
    if not supported:
        raise RuntimeError(
            "PSIM-D3 bound Git binary lacks no-lazy-fetch semantics"
        )
    return {
        "environment": "GIT_NO_LAZY_FETCH=1",
        "fetch_child_processes": 0,
        "missing_object_exit_nonzero": True,
        "pack_delta": 0,
        "semantic_probe_passed": True,
        "upstream_version_contract": (
            "behaviorally_bound_local_backport_not_upstream_2.43_cli"
        ),
    }


def _buffered_cat_file_control(
    root: Path,
    origin: Path,
    requested: Sequence[str],
) -> dict[str, Any]:
    clone = root / "buffer-control.git"
    _partial_clone(origin, clone)
    _assert_oids_absent_from_object_store(clone, requested)
    before = _pack_roster(clone, "promisor")
    commands = (
        "".join(f"contents {oid}\n" for oid in requested) + "flush\n"
    ).encode("ascii")
    _git(
        clone,
        [
            "-c",
            "maintenance.auto=false",
            "-c",
            "gc.auto=0",
            "cat-file",
            "--batch-command",
            "--buffer",
        ],
        input_bytes=commands,
    )
    after = _pack_roster(clone, "promisor")
    delta = len(after) - len(before)
    if delta != len(requested):
        raise RuntimeError(
            "PSIM-D3 buffered cat-file control no longer fetches per object"
        )
    return {
        "command": "git cat-file --batch-command --buffer",
        "interpretation": (
            "stdout_buffering_only_not_network_want_batching"
        ),
        "promisor_pack_delta": delta,
        "requested_blob_count": len(requested),
    }


def build_probe(root: Path) -> dict[str, Any]:
    if _git_text(None, ["--version"]) != GIT_VERSION:
        raise RuntimeError("PSIM-D3 Git version changed")
    selected_git = shutil.which("git")
    if (
        selected_git is None
        or Path(selected_git).resolve() != GIT_BINARY.resolve()
        or GIT_BINARY.resolve() != Path("/usr/bin/git")
    ):
        raise RuntimeError("PSIM-D3 Git binary path changed")
    if sha256_file(GIT_BINARY) != GIT_BINARY_SHA256:
        raise RuntimeError("PSIM-D3 Git binary hash changed")

    root.mkdir(parents=True, exist_ok=False)
    _work, origin, requested, nonrequested = _make_synthetic_origin(root)
    payload: dict[str, Any] = {
        "access_boundary": {
            "official_eip_bip_source_accessed": False,
            "market_data_accessed": False,
            "model_accessed": False,
            "outcomes_accessed": False,
        },
        "buffered_cat_file_control": _buffered_cat_file_control(
            root,
            origin,
            requested,
        ),
        "bulk_fetch_probe": _bulk_fetch_probe(
            root,
            origin,
            requested,
            nonrequested,
        ),
        "candidate": {
            "id": "PSIM-D3",
            "name": (
                "Protocol Specification Intent-Maturity relation RLLM, "
                "targeted batch-hydration bare replay"
            ),
            "transport_only_successor": True,
        },
        "git_binding": {
            "binary_path": str(GIT_BINARY),
            "binary_sha256": GIT_BINARY_SHA256,
            "version": GIT_VERSION,
        },
        "synthetic_remote_contract": {
            "transport": "file",
            "uploadpack_allow_any_sha1_in_want": True,
            "uploadpack_allow_filter": True,
        },
        "no_lazy_fetch_probe": _no_lazy_fetch_probe(
            root,
            origin,
            nonrequested,
        ),
        "protocol_version": PROTOCOL_VERSION,
        "requested_blob_oid_manifest_hash": canonical_hash(list(requested)),
        "synthetic_only": True,
    }
    payload["result_hash"] = canonical_hash(payload)
    return payload


def run_probe() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="psim-d3-batch-hydration-probe-"
    ) as temporary:
        return build_probe(Path(temporary) / "root")


def write_probe(output: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    payload = run_probe()
    target = repository_path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(payload)
    if target.exists() and target.read_bytes() != raw:
        raise RuntimeError("existing PSIM-D3 transport probe differs")
    target.write_bytes(raw)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    arguments = parser.parse_args()
    payload = write_probe(arguments.output)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "result_hash": payload["result_hash"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
