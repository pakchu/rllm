"""Build and gate the outcome-blind BCIMS 2020-2023 source."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

from training import preregister_bitcoin_core_immutable_merge_surface as protocol


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = Path("training/build_bitcoin_core_immutable_merge_surface_source.py")
PROTOCOL_PATH = protocol.DEFAULT_OUTPUT
DEFAULT_SOURCE = Path(
    "data/bitcoin_core_immutable_merge_surface_2020_2023.jsonl.gz"
)
DEFAULT_MANIFEST = Path(
    "results/bitcoin_core_immutable_merge_surface_source_manifest_2026-07-22.json"
)
DEFAULT_SUPPORT = Path(
    "results/bitcoin_core_immutable_merge_surface_source_support_2026-07-22.json"
)
SOURCE_START = datetime(2020, 1, 1, tzinfo=timezone.utc)
SOURCE_END_EXCLUSIVE = datetime(2024, 1, 1, tzinfo=timezone.utc)

HEX40 = re.compile(r"[0-9a-f]{40}", re.ASCII)
IDENTITY_TIME_PATTERN = re.compile(
    rb" (?P<epoch>[0-9]+) (?P<offset>[+-][0-9]{4})\Z"
)
RAW_DIFF_HEADER_PATTERN = re.compile(
    rb":(?P<old_mode>[0-7]{6}) (?P<new_mode>[0-7]{6}) "
    rb"(?P<old_oid>[0-9a-f]{40}) (?P<new_oid>[0-9a-f]{40}) "
    rb"(?P<status>[A-Z])\Z"
)


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(_repo_path(path).read_bytes())


def canonical_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(raw)


def _run_git(
    repo: Path,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    env = os.environ.copy()
    env.update({"LC_ALL": "C", "LANG": "C", "GIT_OPTIONAL_LOCKS": "0"})
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )
    if check and result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"BCIMS git {' '.join(args)} failed: {detail}")
    return result


def _used_gib(path: Path) -> int:
    stats = os.statvfs(path)
    used = (stats.f_blocks - stats.f_bfree) * stats.f_frsize
    return used // (1024**3)


def enforce_disk_guard(path: Path) -> int:
    used_gib = _used_gib(path)
    if used_gib >= protocol.DISK_LIMIT_GIB:
        raise RuntimeError(
            f"BCIMS disk guard rejected {used_gib} GiB used at "
            f"{protocol.DISK_LIMIT_GIB} GiB"
        )
    return used_gib


def git_object_sha1(kind: str, raw: bytes) -> str:
    header = f"{kind} {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw, usedforsecurity=False).hexdigest()


def _parse_identity_time(raw_header: bytes, label: str) -> tuple[int, str]:
    match = IDENTITY_TIME_PATTERN.search(raw_header)
    if match is None:
        raise ValueError(f"BCIMS {label} header has no exact timestamp suffix")
    epoch = int(match.group("epoch"))
    offset = match.group("offset").decode("ascii")
    hours = int(offset[1:3])
    minutes = int(offset[3:5])
    if hours > 14 or minutes > 59:
        raise ValueError(f"BCIMS {label} header has an invalid timezone offset")
    return epoch, offset


def _utc_iso_from_epoch(epoch: int) -> str:
    try:
        parsed = datetime.fromtimestamp(epoch, tz=timezone.utc)
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError("BCIMS commit timestamp is outside datetime range") from exc
    return parsed.isoformat().replace("+00:00", "Z")


def parse_commit_object(expected_hash: str, raw: bytes) -> dict[str, Any]:
    if HEX40.fullmatch(expected_hash) is None:
        raise ValueError("BCIMS expected commit hash is malformed")
    if git_object_sha1("commit", raw) != expected_hash:
        raise ValueError("BCIMS raw commit object does not match its SHA-1 identity")
    if b"\x00" in raw:
        raise ValueError("BCIMS raw commit object contains NUL")
    try:
        header_raw, message_raw = raw.split(b"\n\n", 1)
    except ValueError as exc:
        raise ValueError("BCIMS raw commit object has no header/message boundary") from exc

    headers: dict[bytes, list[bytes]] = defaultdict(list)
    current_key: bytes | None = None
    for line in header_raw.split(b"\n"):
        if line.startswith(b" "):
            if current_key is None:
                raise ValueError("BCIMS commit header begins with a continuation")
            headers[current_key][-1] += b"\n" + line
            continue
        try:
            key, value = line.split(b" ", 1)
        except ValueError as exc:
            raise ValueError("BCIMS commit header line is malformed") from exc
        current_key = key
        headers[key].append(value)

    for key in (b"tree", b"author", b"committer"):
        if len(headers.get(key, [])) != 1:
            raise ValueError(f"BCIMS commit {key.decode()} header is not singular")
    if b"encoding" in headers:
        raise ValueError("BCIMS commit declares a non-default message encoding")

    tree_hash = headers[b"tree"][0].decode("ascii", errors="strict")
    parent_hashes = [value.decode("ascii", errors="strict") for value in headers.get(b"parent", [])]
    if HEX40.fullmatch(tree_hash) is None or any(
        HEX40.fullmatch(value) is None for value in parent_hashes
    ):
        raise ValueError("BCIMS commit contains a malformed tree or parent hash")

    author_epoch, author_offset = _parse_identity_time(headers[b"author"][0], "author")
    committer_epoch, committer_offset = _parse_identity_time(
        headers[b"committer"][0], "committer"
    )
    try:
        message = message_raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("BCIMS commit message is not strict UTF-8") from exc
    subject = message.split("\n", 1)[0]
    return {
        "commit_hash": expected_hash,
        "tree_hash": tree_hash,
        "parent_hashes": parent_hashes,
        "parent_count": len(parent_hashes),
        "author_epoch": author_epoch,
        "author_offset": author_offset,
        "author_time_utc": _utc_iso_from_epoch(author_epoch),
        "committer_epoch": committer_epoch,
        "committer_offset": committer_offset,
        "committer_time_utc": _utc_iso_from_epoch(committer_epoch),
        "subject": subject,
        "full_message": message,
        "raw_commit_sha256": sha256_bytes(raw),
        "raw_commit_base64": base64.b64encode(raw).decode("ascii"),
    }


def parse_raw_path_delta(raw: bytes) -> list[dict[str, str]]:
    if not raw:
        return []
    tokens = raw.split(b"\x00")
    if tokens[-1] != b"":
        raise ValueError("BCIMS raw path delta is not NUL-terminated")
    tokens.pop()
    if len(tokens) % 2:
        raise ValueError("BCIMS raw path delta does not contain header/path pairs")

    changes: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for index in range(0, len(tokens), 2):
        header, path_raw = tokens[index], tokens[index + 1]
        match = RAW_DIFF_HEADER_PATTERN.fullmatch(header)
        if match is None:
            raise ValueError("BCIMS raw path-delta header is malformed")
        status = match.group("status").decode("ascii")
        if status in {"C", "R"}:
            raise ValueError("BCIMS path delta unexpectedly used rename detection")
        if status not in {"A", "D", "M", "T", "U", "X", "B"}:
            raise ValueError("BCIMS path delta has an unsupported status")
        try:
            path = path_raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("BCIMS Git path is not strict UTF-8") from exc
        surface = protocol.path_surface(path)
        if path in seen_paths:
            raise ValueError("BCIMS path delta repeats a path")
        seen_paths.add(path)
        changes.append(
            {
                "old_mode": match.group("old_mode").decode("ascii"),
                "new_mode": match.group("new_mode").decode("ascii"),
                "old_oid": match.group("old_oid").decode("ascii"),
                "new_oid": match.group("new_oid").decode("ascii"),
                "status": status,
                "path": path,
                "surface": surface,
            }
        )
    return changes


def _cat_file_batch(repo: Path, hashes: Sequence[str]) -> Iterator[tuple[str, bytes]]:
    env = os.environ.copy()
    env.update({"LC_ALL": "C", "LANG": "C", "GIT_OPTIONAL_LOCKS": "0"})
    process = subprocess.Popen(
        ["git", "-C", str(repo), "cat-file", "--batch"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    if process.stdin is None or process.stdout is None or process.stderr is None:
        process.kill()
        raise RuntimeError("BCIMS failed to open git cat-file pipes")
    try:
        for expected_hash in hashes:
            process.stdin.write(expected_hash.encode("ascii") + b"\n")
            process.stdin.flush()
            header = process.stdout.readline().rstrip(b"\n")
            parts = header.split(b" ")
            if len(parts) != 3 or parts[0].decode("ascii") != expected_hash:
                raise RuntimeError("BCIMS git cat-file returned an unexpected header")
            if parts[1] != b"commit" or not parts[2].isdigit():
                raise RuntimeError("BCIMS git cat-file did not return a commit object")
            size = int(parts[2])
            raw = process.stdout.read(size)
            if len(raw) != size or process.stdout.read(1) != b"\n":
                raise RuntimeError("BCIMS git cat-file returned a truncated object")
            yield expected_hash, raw
        process.stdin.close()
        stderr = process.stderr.read()
        return_code = process.wait()
        if return_code != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"BCIMS git cat-file failed: {detail}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def _historical_floor(
    committer_time_utc: str,
    running_day: date | None,
) -> tuple[date, datetime]:
    parsed = datetime.fromisoformat(committer_time_utc.replace("Z", "+00:00"))
    commit_day = parsed.astimezone(timezone.utc).date()
    next_running_day = commit_day if running_day is None else max(running_day, commit_day)
    floor = datetime.combine(
        next_running_day + timedelta(days=2),
        time(12),
        tzinfo=timezone.utc,
    )
    return next_running_day, floor


def verify_and_refresh_source_repo(repo: Path) -> dict[str, Any]:
    if not repo.is_dir():
        raise RuntimeError("BCIMS source repository does not exist")
    used_before = enforce_disk_guard(repo)
    remote = _run_git(repo, "remote", "get-url", "origin").stdout.decode().strip()
    if remote != protocol.OFFICIAL_REMOTE:
        raise RuntimeError("BCIMS origin URL differs from the official remote")
    object_format = _run_git(repo, "rev-parse", "--show-object-format").stdout.decode().strip()
    if object_format != "sha1":
        raise RuntimeError("BCIMS repository object format is not frozen SHA-1")
    promisor = _run_git(repo, "config", "--get", "remote.origin.promisor").stdout.decode().strip()
    partial_filter = _run_git(
        repo, "config", "--get", "remote.origin.partialclonefilter"
    ).stdout.decode().strip()
    if promisor != "true" or partial_filter != "blob:none":
        raise RuntimeError("BCIMS clone is not the frozen blobless promisor clone")

    remote_probe = _run_git(repo, "ls-remote", "--symref", "origin", "HEAD").stdout
    remote_lines = remote_probe.decode("utf-8", errors="strict").splitlines()
    if not remote_lines or remote_lines[0] != "ref: refs/heads/master\tHEAD":
        raise RuntimeError("BCIMS remote default branch symref changed")
    remote_heads = [line.split("\t", 1)[0] for line in remote_lines[1:] if line.endswith("\tHEAD")]
    if len(remote_heads) != 1 or HEX40.fullmatch(remote_heads[0]) is None:
        raise RuntimeError("BCIMS remote HEAD identity is malformed")

    enforce_disk_guard(repo)
    _run_git(repo, "fetch", "--filter=blob:none", "--no-tags", "origin", "master")
    sealed = _run_git(
        repo, "rev-parse", f"{protocol.PROBE_SEALED_TIP}^{{commit}}"
    ).stdout.decode().strip()
    if sealed != protocol.PROBE_SEALED_TIP:
        raise RuntimeError("BCIMS sealed tip identity changed")
    ancestor = _run_git(
        repo,
        "merge-base",
        "--is-ancestor",
        protocol.PROBE_SEALED_TIP,
        "refs/remotes/origin/master",
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("BCIMS sealed tip is not reachable from fetched master")
    fsck = _run_git(repo, "fsck", "--connectivity-only", "--no-dangling")
    git_version = subprocess.run(
        ["git", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout.decode("ascii", errors="strict").strip()
    return {
        "remote": remote,
        "remote_default_symref": "refs/heads/master",
        "remote_head_at_fetch": remote_heads[0],
        "remote_probe_sha256": sha256_bytes(remote_probe),
        "sealed_tip": sealed,
        "object_format": object_format,
        "promisor": True,
        "partial_clone_filter": partial_filter,
        "git_version": git_version,
        "fsck_connectivity_passed": True,
        "fsck_stdout_sha256": sha256_bytes(fsck.stdout),
        "fsck_stderr_sha256": sha256_bytes(fsck.stderr),
        "used_gib_before_fetch": used_before,
        "used_gib_after_fetch": enforce_disk_guard(repo),
    }


def collect_source_rows(repo: Path) -> list[dict[str, Any]]:
    rev_list_raw = _run_git(
        repo, "rev-list", "--first-parent", "--reverse", protocol.PROBE_SEALED_TIP
    ).stdout
    hashes = [line for line in rev_list_raw.decode("ascii", errors="strict").splitlines() if line]
    if not hashes or hashes[-1] != protocol.PROBE_SEALED_TIP:
        raise RuntimeError("BCIMS first-parent traversal does not end at the sealed tip")
    if len(hashes) != len(set(hashes)) or any(HEX40.fullmatch(value) is None for value in hashes):
        raise RuntimeError("BCIMS first-parent traversal contains invalid identities")

    running_day: date | None = None
    selected: list[dict[str, Any]] = []
    for commit_hash, raw in _cat_file_batch(repo, hashes):
        commit = parse_commit_object(commit_hash, raw)
        running_day, floor = _historical_floor(commit["committer_time_utc"], running_day)
        if floor < SOURCE_START:
            continue
        if floor >= SOURCE_END_EXCLUSIVE:
            continue
        commit["causal_availability_utc"] = floor.isoformat().replace("+00:00", "Z")
        selected.append(commit)
    if not selected:
        raise RuntimeError("BCIMS frozen interval contains no first-parent commits")

    seen_prs: dict[str, set[int]] = {
        "primary_core": set(),
        "gui_comparator": set(),
    }
    rows: list[dict[str, Any]] = []
    for position, commit in enumerate(selected):
        parsed_subject = protocol.parse_merge_subject(commit["subject"])
        if parsed_subject is not None and commit["parent_count"] != 2:
            raise RuntimeError("BCIMS exact merge subject does not have two parents")
        if not commit["parent_hashes"]:
            raise RuntimeError("BCIMS in-range first-parent commit has no parent")
        stratum = "audit_only" if parsed_subject is None else parsed_subject["stratum"]
        pr_number: int | None = None
        repository: str | None = None
        title: str | None = None
        if parsed_subject is not None:
            parsed_pr_number = parsed_subject["pr_number"]
            parsed_repository = parsed_subject["repository"]
            parsed_title = parsed_subject["title"]
            if (
                not isinstance(parsed_pr_number, int)
                or not isinstance(parsed_repository, str)
                or not isinstance(parsed_title, str)
                or stratum not in seen_prs
            ):
                raise RuntimeError("BCIMS merge-subject parser returned malformed data")
            pr_number = parsed_pr_number
            repository = parsed_repository
            title = parsed_title
            if pr_number in seen_prs[stratum]:
                raise RuntimeError("BCIMS repeats a PR number within a stratum")
            seen_prs[stratum].add(pr_number)

        parent_one = commit["parent_hashes"][0]
        raw_delta = _run_git(
            repo,
            "diff-tree",
            "--no-commit-id",
            "--raw",
            "-r",
            "--no-renames",
            "-z",
            parent_one,
            commit["commit_hash"],
        ).stdout
        changes = parse_raw_path_delta(raw_delta)
        if stratum == "primary_core" and not changes:
            raise RuntimeError("BCIMS primary event has an empty path delta")
        surfaces = sorted({change["surface"] for change in changes})
        rows.append(
            {
                "schema_version": "bcims_source_row_v1",
                "event_position": position,
                "event_id": commit["commit_hash"],
                **commit,
                "stratum": stratum,
                "repository": repository,
                "pr_number": pr_number,
                "title": title,
                "path_changes": changes,
                "path_change_count": len(changes),
                "top_level_surfaces": surfaces,
                "raw_path_delta_sha256": sha256_bytes(raw_delta),
                "raw_path_delta_base64": base64.b64encode(raw_delta).decode("ascii"),
            }
        )
    return rows


def _gate(
    gate_id: str,
    observed: Any,
    requirement: str,
    passed: bool,
) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "observed": observed,
        "requirement": requirement,
        "passed": bool(passed),
    }


def evaluate_support(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("BCIMS support requires source rows")
    years = (2020, 2021, 2022, 2023)
    primary = [row for row in rows if row["stratum"] == "primary_core"]
    gui = [row for row in rows if row["stratum"] == "gui_comparator"]
    audit = [row for row in rows if row["stratum"] == "audit_only"]

    def availability(row: Mapping[str, Any]) -> datetime:
        value = row["causal_availability_utc"]
        if not isinstance(value, str):
            raise ValueError("BCIMS source row availability is malformed")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed < SOURCE_START or parsed >= SOURCE_END_EXCLUSIVE:
            raise ValueError("BCIMS source row escaped the frozen interval")
        return parsed

    parsed_times = {id(row): availability(row) for row in rows}
    primary_year = Counter(parsed_times[id(row)].year for row in primary)
    primary_quarter = Counter(
        f"{parsed_times[id(row)].year}-Q{((parsed_times[id(row)].month - 1) // 3) + 1}"
        for row in primary
    )
    primary_month = Counter(
        parsed_times[id(row)].strftime("%Y-%m") for row in primary
    )
    primary_days_by_year: dict[int, set[str]] = defaultdict(set)
    primary_surfaces_by_year: dict[int, set[str]] = defaultdict(set)
    surface_weight: dict[str, float] = defaultdict(float)
    for row in primary:
        observed_at = parsed_times[id(row)]
        primary_days_by_year[observed_at.year].add(observed_at.date().isoformat())
        surfaces_raw = row["top_level_surfaces"]
        if not isinstance(surfaces_raw, list) or not surfaces_raw:
            raise ValueError("BCIMS primary source row has no path surfaces")
        surfaces = {str(surface) for surface in surfaces_raw}
        primary_surfaces_by_year[observed_at.year].update(surfaces)
        weight = 1.0 / len(surfaces)
        for surface in surfaces:
            surface_weight[surface] += weight

    gui_year = Counter(parsed_times[id(row)].year for row in gui)
    gui_days = {parsed_times[id(row)].date().isoformat() for row in gui}
    unknown_fraction = len(audit) / len(rows)
    unknown_fraction_by_year: dict[int, float] = {}
    for year in years:
        year_rows = [row for row in rows if parsed_times[id(row)].year == year]
        unknown_fraction_by_year[year] = (
            sum(row["stratum"] == "audit_only" for row in year_rows) / len(year_rows)
            if year_rows
            else 1.0
        )
    maximum_month_share = max(primary_month.values(), default=0) / max(len(primary), 1)
    surface_shares = {
        surface: weight / max(len(primary), 1)
        for surface, weight in sorted(surface_weight.items())
    }
    maximum_surface_share = max(surface_shares.values(), default=0.0)

    gates = protocol.build_manifest()["source_quality_gates"]
    primary_gates = gates["primary_core"]
    gui_gates = gates["gui_comparator"]
    integrity_gates = gates["integrity"]
    checks = [
        _gate(
            "unknown_fraction_overall",
            unknown_fraction,
            f"<= {integrity_gates['unknown_first_parent_fraction_max']}",
            unknown_fraction <= integrity_gates["unknown_first_parent_fraction_max"],
        ),
        _gate(
            "unknown_fraction_each_year",
            {str(year): unknown_fraction_by_year[year] for year in years},
            f"each <= {integrity_gates['unknown_first_parent_fraction_max_each_year']}",
            all(
                unknown_fraction_by_year[year]
                <= integrity_gates["unknown_first_parent_fraction_max_each_year"]
                for year in years
            ),
        ),
        _gate(
            "primary_total",
            len(primary),
            f">= {primary_gates['minimum_events']}",
            len(primary) >= primary_gates["minimum_events"],
        ),
        _gate(
            "primary_each_year",
            {str(year): primary_year[year] for year in years},
            f"each >= {primary_gates['minimum_events_each_year']}",
            all(
                primary_year[year] >= primary_gates["minimum_events_each_year"]
                for year in years
            ),
        ),
        _gate(
            "primary_each_quarter",
            dict(sorted(primary_quarter.items())),
            f"each of 16 quarters >= {primary_gates['minimum_events_each_quarter']}",
            all(
                primary_quarter[f"{year}-Q{quarter}"]
                >= primary_gates["minimum_events_each_quarter"]
                for year in years
                for quarter in range(1, 5)
            ),
        ),
        _gate(
            "primary_unique_days_each_year",
            {str(year): len(primary_days_by_year[year]) for year in years},
            f"each >= {primary_gates['minimum_unique_availability_days_each_year']}",
            all(
                len(primary_days_by_year[year])
                >= primary_gates["minimum_unique_availability_days_each_year"]
                for year in years
            ),
        ),
        _gate(
            "primary_maximum_month_share",
            maximum_month_share,
            f"<= {primary_gates['maximum_calendar_month_share']}",
            maximum_month_share <= primary_gates["maximum_calendar_month_share"],
        ),
        _gate(
            "primary_distinct_surfaces_each_year",
            {str(year): len(primary_surfaces_by_year[year]) for year in years},
            f"each >= {primary_gates['minimum_distinct_top_level_surfaces_each_year']}",
            all(
                len(primary_surfaces_by_year[year])
                >= primary_gates["minimum_distinct_top_level_surfaces_each_year"]
                for year in years
            ),
        ),
        _gate(
            "primary_maximum_fractional_surface_share",
            maximum_surface_share,
            f"<= {primary_gates['maximum_fractional_top_level_surface_share']}",
            maximum_surface_share
            <= primary_gates["maximum_fractional_top_level_surface_share"],
        ),
        _gate(
            "gui_total",
            len(gui),
            f">= {gui_gates['minimum_events']}",
            len(gui) >= gui_gates["minimum_events"],
        ),
        _gate(
            "gui_each_year",
            {str(year): gui_year[year] for year in years},
            f"each >= {gui_gates['minimum_events_each_year']}",
            all(
                gui_year[year] >= gui_gates["minimum_events_each_year"]
                for year in years
            ),
        ),
        _gate(
            "gui_unique_days",
            len(gui_days),
            f">= {gui_gates['minimum_unique_availability_days']}",
            len(gui_days) >= gui_gates["minimum_unique_availability_days"],
        ),
    ]
    all_passed = all(check["passed"] for check in checks)
    return {
        "source_id": "BCIMS",
        "outcomes_opened": False,
        "market_clocks_opened": False,
        "semantic_model_opened": False,
        "row_count": len(rows),
        "stratum_counts": {
            "primary_core": len(primary),
            "gui_comparator": len(gui),
            "audit_only": len(audit),
        },
        "primary_month_counts": dict(sorted(primary_month.items())),
        "primary_fractional_surface_shares": surface_shares,
        "gates": checks,
        "all_gates_passed": all_passed,
        "status": (
            "PASS_ADVANCE_TO_SEMANTIC_FREEZE"
            if all_passed
            else "REJECT_NO_REPAIR"
        ),
    }


def _canonical_json_line(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_source_once(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output = _repo_path(path)
    if output.exists():
        raise RuntimeError("refusing to overwrite frozen BCIMS source rows")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    uncompressed_hasher = hashlib.sha256()
    try:
        with temporary.open("xb") as raw_handle:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_handle,
                mtime=0,
            ) as compressed:
                for row in rows:
                    encoded = _canonical_json_line(row)
                    uncompressed_hasher.update(encoded)
                    compressed.write(encoded)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(path),
        "row_count": len(rows),
        "uncompressed_jsonl_sha256": uncompressed_hasher.hexdigest(),
        "compressed_sha256": sha256_file(output),
        "compressed_bytes": output.stat().st_size,
        "gzip_mtime": 0,
    }


def write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    output = _repo_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )


def verify_committed_builder() -> dict[str, str]:
    relative_path = str(BUILDER_PATH)
    status = _run_git(
        REPO_ROOT,
        "status",
        "--porcelain=v1",
        "--",
        relative_path,
    ).stdout
    if status:
        raise RuntimeError("BCIMS builder must be committed and clean before incidence")
    head = _run_git(REPO_ROOT, "rev-parse", "HEAD").stdout.decode().strip()
    committed = _run_git(REPO_ROOT, "show", f"HEAD:{relative_path}").stdout
    working_sha256 = sha256_file(BUILDER_PATH)
    committed_sha256 = sha256_bytes(committed)
    if committed_sha256 != working_sha256:
        raise RuntimeError("BCIMS committed builder differs from working bytes")
    return {
        "path": relative_path,
        "sha256": working_sha256,
        "git_commit": head,
        "committed_blob_sha256": committed_sha256,
    }


def build_source(
    source_repo: Path,
    source_output: Path = DEFAULT_SOURCE,
    manifest_output: Path = DEFAULT_MANIFEST,
    support_output: Path = DEFAULT_SUPPORT,
) -> dict[str, Any]:
    protocol_payload = json.loads(_repo_path(PROTOCOL_PATH).read_text(encoding="utf-8"))
    protocol.validate_manifest(protocol_payload)
    builder_binding = verify_committed_builder()
    for path in (source_output, manifest_output, support_output):
        if _repo_path(path).exists():
            raise RuntimeError(f"BCIMS output already exists: {path}")

    source_verification = verify_and_refresh_source_repo(source_repo)
    rows = collect_source_rows(source_repo)
    source_file = write_source_once(source_output, rows)
    support_core = evaluate_support(rows)
    support_core.update(
        {
            "protocol_manifest_hash": protocol_payload["manifest_hash"],
            "protocol_file_sha256": sha256_file(PROTOCOL_PATH),
            "source_file": source_file,
        }
    )
    support_payload = {
        **support_core,
        "result_hash": canonical_hash(support_core),
    }
    manifest_core = {
        "manifest_version": "bcims_source_manifest_v1",
        "source_id": "BCIMS",
        "outcomes_opened": False,
        "market_clocks_opened": False,
        "semantic_model_opened": False,
        "protocol_manifest_hash": protocol_payload["manifest_hash"],
        "protocol_file_sha256": sha256_file(PROTOCOL_PATH),
        "builder": builder_binding,
        "source_verification": source_verification,
        "source_file": source_file,
        "support_result_hash": support_payload["result_hash"],
    }
    manifest_payload = {
        **manifest_core,
        "manifest_hash": canonical_hash(manifest_core),
    }
    write_json_once(manifest_output, manifest_payload)
    write_json_once(support_output, support_payload)
    return {
        "source_id": "BCIMS",
        "status": support_payload["status"],
        "all_gates_passed": support_payload["all_gates_passed"],
        "row_count": len(rows),
        "stratum_counts": support_payload["stratum_counts"],
        "source_file": source_file,
        "manifest_hash": manifest_payload["manifest_hash"],
        "result_hash": support_payload["result_hash"],
        "outcomes_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", required=True, type=Path)
    parser.add_argument("--source-output", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--support-output", type=Path, default=DEFAULT_SUPPORT)
    args = parser.parse_args()
    result = build_source(
        args.source_repo,
        args.source_output,
        args.manifest_output,
        args.support_output,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
