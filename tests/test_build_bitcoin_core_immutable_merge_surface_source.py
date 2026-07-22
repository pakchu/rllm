from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from training import build_bitcoin_core_immutable_merge_surface_source as builder


def _commit_object(
    *,
    tree: str = "1" * 40,
    parents: tuple[str, ...] = ("2" * 40, "3" * 40),
    timestamp: int = 1_600_000_000,
    message: str = "Merge bitcoin/bitcoin#123: net: harden transport\n",
) -> tuple[str, bytes]:
    headers = [f"tree {tree}", *(f"parent {parent}" for parent in parents)]
    headers.extend(
        [
            f"author Alice <alice@example.invalid> {timestamp} +0000",
            f"committer Bob <bob@example.invalid> {timestamp} +0000",
        ]
    )
    raw = ("\n".join(headers) + "\n\n" + message).encode()
    object_hash = hashlib.sha1(
        f"commit {len(raw)}\0".encode() + raw,
        usedforsecurity=False,
    ).hexdigest()
    return object_hash, raw


def _row(
    *,
    year: int,
    month: int,
    day: int,
    stratum: str,
    surfaces: tuple[str, ...] = ("src", "test"),
) -> dict[str, object]:
    availability = datetime(year, month, day, 12, tzinfo=timezone.utc)
    return {
        "stratum": stratum,
        "causal_availability_utc": availability.isoformat().replace("+00:00", "Z"),
        "top_level_surfaces": list(surfaces),
    }


def _git(repo: Path, *args: str, commit_date: str | None = None) -> str:
    env = os.environ.copy()
    if commit_date is not None:
        env["GIT_AUTHOR_DATE"] = commit_date
        env["GIT_COMMITTER_DATE"] = commit_date
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    return result.stdout.decode().strip()


def test_parse_commit_object_verifies_identity_and_extracts_immutable_fields() -> None:
    object_hash, raw = _commit_object()
    parsed = builder.parse_commit_object(object_hash, raw)
    assert parsed["commit_hash"] == object_hash
    assert parsed["tree_hash"] == "1" * 40
    assert parsed["parent_hashes"] == ["2" * 40, "3" * 40]
    assert parsed["parent_count"] == 2
    assert parsed["subject"] == "Merge bitcoin/bitcoin#123: net: harden transport"
    assert base64.b64decode(parsed["raw_commit_base64"]) == raw
    assert parsed["raw_commit_sha256"] == hashlib.sha256(raw).hexdigest()


def test_parse_commit_object_rejects_hash_timestamp_encoding_and_utf8_faults() -> None:
    object_hash, raw = _commit_object()
    with pytest.raises(ValueError, match="SHA-1"):
        builder.parse_commit_object("f" * 40, raw)

    bad_time = raw.replace(b" +0000\ncommitter", b" +9999\ncommitter", 1)
    bad_time_hash = builder.git_object_sha1("commit", bad_time)
    with pytest.raises(ValueError, match="timezone"):
        builder.parse_commit_object(bad_time_hash, bad_time)

    encoded = raw.replace(b"committer Bob", b"encoding ISO-8859-1\ncommitter Bob")
    encoded_hash = builder.git_object_sha1("commit", encoded)
    with pytest.raises(ValueError, match="encoding"):
        builder.parse_commit_object(encoded_hash, encoded)

    invalid_utf8 = raw[:-1] + b"\xff\n"
    invalid_hash = builder.git_object_sha1("commit", invalid_utf8)
    with pytest.raises(ValueError, match="UTF-8"):
        builder.parse_commit_object(invalid_hash, invalid_utf8)


def test_parse_raw_path_delta_preserves_exact_no_rename_records() -> None:
    raw = (
        b":100644 100644 " + b"1" * 40 + b" " + b"2" * 40 + b" M\x00src/net.cpp\x00"
        b":000000 100644 " + b"0" * 40 + b" " + b"3" * 40 + b" A\x00README.md\x00"
    )
    changes = builder.parse_raw_path_delta(raw)
    assert [change["path"] for change in changes] == ["src/net.cpp", "README.md"]
    assert [change["surface"] for change in changes] == ["src", "__root__"]
    assert [change["status"] for change in changes] == ["M", "A"]


@pytest.mark.parametrize(
    "raw, match",
    [
        (b"not-terminated", "NUL-terminated"),
        (
            b":100644 100644 " + b"1" * 40 + b" " + b"2" * 40 + b" R\x00src/a\x00",
            "rename",
        ),
        (
            b":100644 100644 " + b"1" * 40 + b" " + b"2" * 40 + b" U\x00src/a\x00",
            "unsupported",
        ),
        (
            b":100644 100644 " + b"1" * 40 + b" " + b"2" * 40 + b" X\x00src/a\x00",
            "unsupported",
        ),
        (
            b":100644 100644 " + b"1" * 40 + b" " + b"2" * 40 + b" B\x00src/a\x00",
            "unsupported",
        ),
        (
            b":100644 100644 "
            + b"1" * 40
            + b" "
            + b"2" * 40
            + b" M\x00src/a\x00"
            + b":100644 100644 "
            + b"1" * 40
            + b" "
            + b"2" * 40
            + b" M\x00src/a\x00",
            "repeats",
        ),
    ],
)
def test_parse_raw_path_delta_fails_closed(raw: bytes, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        builder.parse_raw_path_delta(raw)


def test_collect_source_rows_replays_first_parent_merge_and_audit_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "source.git"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "master")
    _git(repo, "config", "user.name", "Fixture")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    (repo / "README.md").write_text("base\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", "base", commit_date="2019-12-20T00:00:00+00:00")

    _git(repo, "checkout", "-q", "-b", "feature")
    (repo / "src").mkdir()
    (repo / "src" / "net.cpp").write_text("transport\n")
    _git(repo, "add", "src/net.cpp")
    _git(
        repo,
        "commit",
        "-q",
        "-m",
        "net: fixture",
        commit_date="2019-12-29T00:00:00+00:00",
    )
    _git(repo, "checkout", "-q", "master")
    _git(
        repo,
        "merge",
        "--no-ff",
        "-q",
        "feature",
        "-m",
        "Merge bitcoin/bitcoin#123: net: harden transport",
        commit_date="2019-12-30T00:00:00+00:00",
    )
    (repo / "README.md").write_text("base\naudit\n")
    _git(repo, "add", "README.md")
    _git(
        repo,
        "commit",
        "-q",
        "-m",
        "direct audit commit",
        commit_date="2020-01-01T00:00:00+00:00",
    )
    sealed_tip = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(builder.protocol, "PROBE_SEALED_TIP", sealed_tip)

    rows = builder.collect_source_rows(repo)
    assert [row["stratum"] for row in rows] == ["primary_core", "audit_only"]
    assert rows[0]["causal_availability_utc"] == "2020-01-01T12:00:00Z"
    assert rows[0]["pr_number"] == 123
    assert rows[0]["top_level_surfaces"] == ["src"]
    assert rows[0]["path_changes"][0]["path"] == "src/net.cpp"
    assert base64.b64decode(rows[0]["raw_commit_base64"])
    assert base64.b64decode(rows[0]["raw_path_delta_base64"])
    assert rows[1]["parent_count"] == 1
    assert rows[1]["repository"] is None


def test_support_evaluator_passes_only_the_exact_frozen_battery() -> None:
    rows: list[dict[str, object]] = []
    surfaces = ("src", "test", "doc", "depends", "contrib", "cmake")
    for year in range(2020, 2024):
        for month in range(1, 13):
            for index in range(50):
                rows.append(
                    _row(
                        year=year,
                        month=month,
                        day=(index % 25) + 1,
                        stratum="primary_core",
                        surfaces=(surfaces[index % len(surfaces)], surfaces[(index + 1) % len(surfaces)]),
                    )
                )
        for index in range(8):
            rows.append(
                _row(
                    year=year,
                    month=(index % 8) + 1,
                    day=index + 1,
                    stratum="gui_comparator",
                    surfaces=("src",),
                )
            )

    result = builder.evaluate_support(rows)
    assert result["all_gates_passed"] is True
    assert result["status"] == "PASS_ADVANCE_TO_SEMANTIC_FREEZE"
    assert result["stratum_counts"] == {
        "primary_core": 2400,
        "gui_comparator": 32,
        "audit_only": 0,
    }
    assert result["outcomes_opened"] is False
    assert result["market_clocks_opened"] is False
    assert result["semantic_model_opened"] is False

    del rows[:101]
    rejected = builder.evaluate_support(rows)
    assert rejected["all_gates_passed"] is False
    assert rejected["status"] == "REJECT_NO_REPAIR"


def test_support_rejects_unknown_concentration() -> None:
    rows = [
        _row(year=year, month=1, day=1, stratum="audit_only", surfaces=("src",))
        for year in range(2020, 2024)
        for _ in range(2)
    ]
    rows.extend(
        _row(year=year, month=1, day=2, stratum="primary_core", surfaces=("src",))
        for year in range(2020, 2024)
        for _ in range(18)
    )
    result = builder.evaluate_support(rows)
    gate = next(item for item in result["gates"] if item["gate_id"] == "unknown_fraction_overall")
    assert gate["observed"] == pytest.approx(0.10)
    assert gate["passed"] is False


def test_source_writer_is_deterministic_and_write_once(tmp_path: Path) -> None:
    rows = [
        {
            "event_id": "a" * 40,
            "stratum": "primary_core",
            "causal_availability_utc": "2020-01-03T12:00:00Z",
        }
    ]
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"
    first_meta = builder.write_source_once(first, rows)
    second_meta = builder.write_source_once(second, rows)
    assert first.read_bytes() == second.read_bytes()
    assert first_meta["compressed_sha256"] == second_meta["compressed_sha256"]
    assert gzip.decompress(first.read_bytes()) == builder._canonical_json_line(rows[0])
    with pytest.raises(RuntimeError, match="overwrite"):
        builder.write_source_once(first, rows)


def test_disk_guard_rejects_at_frozen_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(builder, "_used_gib", lambda _: builder.protocol.DISK_LIMIT_GIB)
    with pytest.raises(RuntimeError, match="disk guard"):
        builder.enforce_disk_guard(tmp_path)


def test_local_object_inventory_detects_blobs(tmp_path: Path) -> None:
    repo = tmp_path / "objects.git"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "master")
    _git(repo, "config", "user.name", "Fixture")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    (repo / "tracked.txt").write_text("blob\n")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-q", "-m", "tracked")
    inventory = builder.local_object_inventory(repo)
    assert inventory["object_counts"]["blob"] == 1
    with pytest.raises(RuntimeError, match="contains blobs"):
        builder.require_no_local_blobs(inventory, "fixture")


def test_result_hash_excludes_no_required_contract_fields() -> None:
    rows = [
        _row(year=year, month=1, day=1, stratum="primary_core")
        for year in range(2020, 2024)
    ]
    result = builder.evaluate_support(rows)
    result_hash = builder.canonical_hash(result)
    mutated = dict(result)
    mutated["outcomes_opened"] = True
    assert builder.canonical_hash(mutated) != result_hash


def test_build_source_replay_mismatch_fails_before_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.jsonl.gz"
    manifest = tmp_path / "manifest.json"
    support = tmp_path / "support.json"
    row = {
        "event_id": "a" * 40,
        "raw_commit_sha256": "b" * 64,
        "raw_path_delta_sha256": "c" * 64,
    }
    changed = {**row, "event_id": "d" * 40}
    calls = iter(([row], [changed]))
    monkeypatch.setattr(builder, "verify_committed_builder", lambda: {"sha256": "x"})
    monkeypatch.setattr(builder, "verify_and_refresh_source_repo", lambda _: {})
    monkeypatch.setattr(builder, "collect_source_rows", lambda _: next(calls))

    with pytest.raises(RuntimeError, match="deterministic replay"):
        builder.build_source(tmp_path, source, manifest, support)
    assert not source.exists()
    assert not manifest.exists()
    assert not support.exists()


def test_build_source_writes_only_after_identical_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.jsonl.gz"
    manifest = tmp_path / "manifest.json"
    support = tmp_path / "support.json"
    row = {
        "event_id": "a" * 40,
        "raw_commit_sha256": "b" * 64,
        "raw_path_delta_sha256": "c" * 64,
    }
    inventory = {
        "object_counts": {"blob": 0, "commit": 1, "tag": 0, "tree": 1},
        "enumeration_stdout_sha256": "d" * 64,
        "enumeration_stderr_sha256": "e" * 64,
    }
    monkeypatch.setattr(
        builder,
        "verify_committed_builder",
        lambda: {"path": "builder.py", "sha256": "f" * 64},
    )
    monkeypatch.setattr(builder, "verify_and_refresh_source_repo", lambda _: {})
    monkeypatch.setattr(builder, "collect_source_rows", lambda _: [dict(row)])
    monkeypatch.setattr(builder, "local_object_inventory", lambda _: inventory)
    monkeypatch.setattr(
        builder,
        "evaluate_support",
        lambda _: {
            "source_id": "BCIMS",
            "status": "PASS_ADVANCE_TO_SEMANTIC_FREEZE",
            "all_gates_passed": True,
            "stratum_counts": {
                "primary_core": 1,
                "gui_comparator": 0,
                "audit_only": 0,
            },
            "outcomes_opened": False,
        },
    )

    result = builder.build_source(tmp_path, source, manifest, support)
    assert result["all_gates_passed"] is True
    assert source.exists() and manifest.exists() and support.exists()
    manifest_payload = json.loads(manifest.read_text())
    assert manifest_payload["source_verification"]["deterministic_replay"]["passed"] is True
    assert "remote_head_at_fetch" not in manifest_payload["source_verification"]


def test_committed_builder_binding_rejects_dirty_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = builder._run_git

    def fake_run_git(
        repo: Path,
        *args: str,
        check: bool = True,
    ) -> subprocess.CompletedProcess[bytes]:
        if args[:2] == ("status", "--porcelain=v1"):
            return subprocess.CompletedProcess(
                ["git", *args],
                0,
                stdout=b" M training/builder.py\n",
                stderr=b"",
            )
        return original(repo, *args, check=check)

    monkeypatch.setattr(builder, "_run_git", fake_run_git)
    with pytest.raises(RuntimeError, match="committed and clean"):
        builder.verify_committed_builder()
