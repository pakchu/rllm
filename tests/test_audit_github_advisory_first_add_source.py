from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from training import audit_github_advisory_first_add_source as audit


GHSA_A = "GHSA-2345-6789-cfgh"
GHSA_B = "GHSA-2345-6789-cfgj"
GHSA_C = "GHSA-2345-6789-cfgm"


def _git(
    repo: Path,
    *args: str,
    author_date: str | None = None,
    committer_date: str | None = None,
) -> str:
    environment = dict(os.environ)
    if author_date is not None:
        environment["GIT_AUTHOR_DATE"] = author_date
    if committer_date is not None:
        environment["GIT_COMMITTER_DATE"] = committer_date
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    return completed.stdout.decode("utf-8", errors="strict").strip()


def _init_repo(path: Path) -> Path:
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.name", "Fixture")
    _git(path, "config", "user.email", "fixture@example.invalid")
    (path / "README.md").write_text("fixture\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(
        path,
        "commit",
        "-q",
        "-m",
        "base",
        author_date="2022-02-12T00:00:00Z",
        committer_date="2022-02-12T00:00:00Z",
    )
    return path


def _path(repo: Path, identity: str, *, bucket: str = "2022/02") -> Path:
    return (
        repo
        / "advisories"
        / "github-reviewed"
        / bucket
        / identity
        / f"{identity}.json"
    )


def _advisory(
    identity: str,
    *,
    published: str = "2022-02-18T00:00:00Z",
    modified: str = "2022-02-18T01:00:00Z",
    ecosystem: str = "PyPI",
    summary: str = "summary",
    details: str = "details",
    withdrawn: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "1.6.0",
        "id": identity,
        "modified": modified,
        "published": published,
        "aliases": [],
        "summary": summary,
        "details": details,
        "severity": [{"type": "CVSS_V3", "score": "fixture"}],
        "affected": [
            {
                "package": {
                    "ecosystem": ecosystem,
                    "name": "fixture-package",
                    "purl": "pkg:fixture/example",
                },
                "ranges": [],
                "versions": [],
            }
        ],
        "references": [],
        "credits": [],
        "database_specific": {},
        "ecosystem_specific": {},
    }
    if withdrawn is not None:
        payload["withdrawn"] = withdrawn
    return payload


def _write_advisory(
    repo: Path,
    identity: str,
    *,
    bucket: str = "2022/02",
    raw: bytes | None = None,
    **kwargs: object,
) -> Path:
    path = _path(repo, identity, bucket=bucket)
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw is None:
        raw = (
            json.dumps(
                _advisory(identity, **kwargs),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    path.write_bytes(raw)
    return path


def _commit_all(
    repo: Path,
    message: str,
    *,
    author_date: str,
    committer_date: str,
) -> str:
    _git(repo, "add", "-A")
    _git(
        repo,
        "commit",
        "-q",
        "-m",
        message,
        author_date=author_date,
        committer_date=committer_date,
    )
    return _git(repo, "rev-parse", "HEAD")


def _candidate(
    *,
    identity: str = GHSA_A,
    published_commit_time: str = "2022-02-20T00:00:00Z",
) -> audit.FirstAddCandidate:
    return audit.FirstAddCandidate(
        identity=identity,
        path=(
            f"advisories/github-reviewed/2022/02/{identity}/{identity}.json"
        ),
        commit_hash="a" * 40,
        blob_oid="b" * 40,
        first_parent_position=1,
        committer_time_utc=published_commit_time,
        ordered_committer_time_utc=published_commit_time,
    )


def _event(
    year: int,
    month: int,
    day: int,
    index: int,
    *,
    ecosystem: str,
    withdrawn: bool = False,
    complete_text: bool = True,
) -> audit.SourceEvent:
    availability = datetime(year, month, day, 12, tzinfo=timezone.utc)
    published = availability - timedelta(days=1)
    digest = hashlib.sha256(
        f"{year}-{month}-{day}-{index}".encode("ascii")
    ).hexdigest()
    return audit.SourceEvent(
        identity_digest=digest,
        first_add_commit_digest=hashlib.sha256(
            f"commit-{digest}".encode("ascii")
        ).hexdigest(),
        initial_blob_sha1=hashlib.sha1(
            digest.encode("ascii"), usedforsecurity=False
        ).hexdigest(),
        raw_sha256=digest,
        structural_sha256=hashlib.sha256(
            f"struct-{digest}".encode("ascii")
        ).hexdigest(),
        published_at_utc=published.isoformat().replace("+00:00", "Z"),
        modified_at_utc=published.isoformat().replace("+00:00", "Z"),
        availability_at_utc=availability.isoformat().replace("+00:00", "Z"),
        schema_version="1.6.0",
        ecosystems=(ecosystem,),
        withdrawn=withdrawn,
        summary_nonempty=complete_text,
        details_nonempty=complete_text,
        summary_utf8_bytes=7 if complete_text else 0,
        details_utf8_bytes=7 if complete_text else 0,
        severity_types=("CVSS_V3",),
    )


def _passing_events() -> list[audit.SourceEvent]:
    rows: list[audit.SourceEvent] = []
    ecosystems = ("PyPI", "npm", "Go", "Maven", "crates.io")
    for index in range(500):
        day = datetime(2022, 2, 13, 12, tzinfo=timezone.utc) + timedelta(
            days=index % 300
        )
        rows.append(
            _event(
                day.year,
                day.month,
                day.day,
                index,
                ecosystem=ecosystems[index % len(ecosystems)],
            )
        )
    ordinal = 10_000
    for year in (2023, 2024, 2025):
        for month in range(1, 13):
            for index in range(84):
                rows.append(
                    _event(
                        year,
                        month,
                        (index % 28) + 1,
                        ordinal,
                        ecosystem=ecosystems[ordinal % len(ecosystems)],
                    )
                )
                ordinal += 1
    return rows


def _paths(root: Path) -> audit.AuditPaths:
    return audit.AuditPaths(
        sentinel=root / "attempt.started",
        manifest=root / "manifest.ndjson",
        raw_dir=root / "raw",
        report=root / "report.json",
    )


def test_frozen_boundary_and_source_constants() -> None:
    assert audit.BOUNDARY_SHA256 == (
        "b167da46a43308a5ce6be70563c455b1c4209499ae5a0423efbdad15080bb25f"
    )
    assert audit.OFFICIAL_REMOTE == (
        "https://github.com/github/advisory-database.git"
    )
    assert audit.FROZEN_COMMIT == "40e5791b176b832cb09323d3962abe2fe3249e34"
    assert audit.FROZEN_TREE == "283dcf468588e3f9fd4a1d7a671df11527788dfc"
    assert audit.FROZEN_PARENT == "0ab828c5a28c008f4c6f3344a8bb783484c41378"
    assert audit.SOURCE_START.isoformat() == "2022-02-11T22:59:38+00:00"
    assert audit.SOURCE_END_EXCLUSIVE.isoformat() == "2026-01-01T00:00:00+00:00"


def test_first_parent_merge_clock_ignores_side_branch_and_author_dates(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path / "repo")
    _git(repo, "checkout", "-q", "-b", "side")
    _write_advisory(repo, GHSA_A)
    side_commit = _commit_all(
        repo,
        "side add",
        author_date="2020-01-01T00:00:00Z",
        committer_date="2022-02-13T00:00:00Z",
    )

    _git(repo, "checkout", "-q", "main")
    (repo / "README.md").write_text("fixture\nmain\n", encoding="utf-8")
    prior_main = _commit_all(
        repo,
        "later main clock",
        author_date="2030-01-01T00:00:00Z",
        committer_date="2022-02-20T00:00:00Z",
    )
    _git(
        repo,
        "merge",
        "--no-ff",
        "-q",
        "side",
        "-m",
        "merge side",
        author_date="2019-01-01T00:00:00Z",
        committer_date="2022-02-15T00:00:00Z",
    )
    merge_commit = _git(repo, "rev-parse", "HEAD")

    scan = audit.collect_first_add_candidates(repo, merge_commit)
    assert scan.first_parent_commit_count == 3
    assert len(scan.candidates) == 1
    candidate = scan.candidates[0]
    assert candidate.commit_hash == merge_commit
    assert candidate.commit_hash not in {side_commit, prior_main}
    assert candidate.committer_time_utc == "2022-02-15T00:00:00Z"
    assert candidate.ordered_committer_time_utc == "2022-02-20T00:00:00Z"

    raw = audit.read_git_blob(repo, candidate.blob_oid)
    event = audit.parse_initial_blob(candidate, raw)
    assert event.availability_at_utc == "2022-02-21T12:00:00Z"


def test_published_clock_can_dominate_and_next_day_is_noon_utc() -> None:
    candidate = _candidate(published_commit_time="2022-02-20T23:59:59Z")
    raw = json.dumps(
        _advisory(
            GHSA_A,
            published="2022-02-21T00:00:01Z",
            modified="2022-02-21T00:00:02Z",
        ),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    candidate = replace(
        candidate,
        blob_oid=audit.git_blob_sha1(raw),
    )
    event = audit.parse_initial_blob(candidate, raw)
    assert event.availability_at_utc == "2022-02-22T12:00:00Z"


def test_rfc3339_nanoseconds_parse_and_preserve_modified_ordering() -> None:
    parsed = audit.parse_rfc3339_utc(
        "2022-02-21T00:00:01.123456789Z",
        field="fixture",
    )
    assert parsed.isoformat() == "2022-02-21T00:00:01.123456+00:00"

    payload = _advisory(
        GHSA_A,
        published="2022-02-21T00:00:01.123456789Z",
        modified="2022-02-21T00:00:01.123456788Z",
    )
    raw = json.dumps(payload, separators=(",", ":")).encode()
    candidate = replace(_candidate(), blob_oid=audit.git_blob_sha1(raw))
    with pytest.raises(audit.StructureError, match="modified"):
        audit.parse_initial_blob(candidate, raw)

    payload["modified"] = "2022-02-21T00:00:01.123456790Z"
    raw = json.dumps(payload, separators=(",", ":")).encode()
    candidate = replace(_candidate(), blob_oid=audit.git_blob_sha1(raw))
    event = audit.parse_initial_blob(candidate, raw)
    assert event.published_at_utc.endswith(".123456789Z")


def test_readd_move_and_edit_do_not_create_new_first_add_event(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path / "repo")
    first_path = _write_advisory(repo, GHSA_A)
    first_commit = _commit_all(
        repo,
        "first add",
        author_date="2022-02-14T00:00:00Z",
        committer_date="2022-02-14T00:00:00Z",
    )
    payload = json.loads(first_path.read_text(encoding="utf-8"))
    payload["modified"] = "2022-02-15T00:00:00Z"
    first_path.write_text(json.dumps(payload), encoding="utf-8")
    _commit_all(
        repo,
        "edit",
        author_date="2022-02-15T00:00:00Z",
        committer_date="2022-02-15T00:00:00Z",
    )
    first_path.unlink()
    _commit_all(
        repo,
        "delete",
        author_date="2022-02-16T00:00:00Z",
        committer_date="2022-02-16T00:00:00Z",
    )
    _write_advisory(repo, GHSA_A, bucket="2022/03")
    tip = _commit_all(
        repo,
        "readd elsewhere",
        author_date="2022-03-01T00:00:00Z",
        committer_date="2022-03-01T00:00:00Z",
    )

    scan = audit.collect_first_add_candidates(repo, tip)
    assert len(scan.candidates) == 1
    assert scan.candidates[0].commit_hash == first_commit
    assert scan.active_path_count == 1
    current = audit._verify_current_tree(repo, tip)
    assert current["regular_blob_count"] == scan.active_path_count
    assert current["tree_listing_sha256"] == scan.active_tree_sha256
    assert scan.mutation_counts == {
        "addition": 2,
        "deletion": 1,
        "modification": 1,
        "readdition": 1,
        "type_change": 0,
    }


def test_duplicate_identity_in_same_transition_rejects(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write_advisory(repo, GHSA_A, bucket="2022/02/one")
    _write_advisory(repo, GHSA_A, bucket="2022/02/two")
    tip = _commit_all(
        repo,
        "duplicate add",
        author_date="2022-02-14T00:00:00Z",
        committer_date="2022-02-14T00:00:00Z",
    )
    with pytest.raises(audit.StructureError, match="duplicate identity"):
        audit.collect_first_add_candidates(repo, tip)


def test_later_concurrent_identity_copy_rejects_without_prior_deletion(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write_advisory(repo, GHSA_A, bucket="2022/02/one")
    _commit_all(
        repo,
        "first add",
        author_date="2022-02-14T00:00:00Z",
        committer_date="2022-02-14T00:00:00Z",
    )
    _write_advisory(repo, GHSA_A, bucket="2022/03/two")
    tip = _commit_all(
        repo,
        "concurrent duplicate",
        author_date="2022-03-01T00:00:00Z",
        committer_date="2022-03-01T00:00:00Z",
    )
    with pytest.raises(audit.StructureError, match="active duplicate identity"):
        audit.collect_first_add_candidates(repo, tip)


@pytest.mark.parametrize(
    "relative_path",
    [
        "advisories/github-reviewed/2022/02/not-json.txt",
        "advisories/github-reviewed/2022/02/ghsa-2345-6789-cfgh.json",
        "advisories/github-reviewed/2022/02/GHSA-2345-6789-cfgh.JSON",
        "advisories/github-reviewed/2022/02/GHSA-abcd-efgh-ijkl.json",
    ],
)
def test_unauthorized_reviewed_path_rejects(
    tmp_path: Path,
    relative_path: str,
) -> None:
    repo = _init_repo(tmp_path / "repo")
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")
    tip = _commit_all(
        repo,
        "bad path",
        author_date="2022-02-14T00:00:00Z",
        committer_date="2022-02-14T00:00:00Z",
    )
    with pytest.raises(audit.StructureError, match="authorized"):
        audit.collect_first_add_candidates(repo, tip)


def test_strict_json_rejects_path_id_duplicate_key_nonfinite_and_encoding() -> None:
    candidate = _candidate()

    mismatch = json.dumps(_advisory(GHSA_B)).encode()
    mismatch_candidate = replace(candidate, blob_oid=audit.git_blob_sha1(mismatch))
    with pytest.raises(audit.StructureError, match="path/id"):
        audit.parse_initial_blob(mismatch_candidate, mismatch)

    duplicate = (
        b'{"schema_version":"1.6.0","id":"'
        + GHSA_A.encode()
        + b'","id":"'
        + GHSA_A.encode()
        + b'","modified":"2022-02-18T01:00:00Z",'
        b'"published":"2022-02-18T00:00:00Z","affected":[{"package":'
        b'{"ecosystem":"PyPI","name":"x"}}]}'
    )
    duplicate_candidate = replace(
        candidate, blob_oid=audit.git_blob_sha1(duplicate)
    )
    with pytest.raises(audit.StructureError, match="duplicate JSON key"):
        audit.parse_initial_blob(duplicate_candidate, duplicate)

    nonfinite = json.dumps(_advisory(GHSA_A)).encode().replace(
        b'"database_specific": {}',
        b'"database_specific": {"x": NaN}',
    )
    nonfinite_candidate = replace(
        candidate, blob_oid=audit.git_blob_sha1(nonfinite)
    )
    with pytest.raises(audit.StructureError, match="finite"):
        audit.parse_initial_blob(nonfinite_candidate, nonfinite)

    for raw, match in (
        (b"\xef\xbb\xbf" + json.dumps(_advisory(GHSA_A)).encode(), "BOM"),
        (b"\xff", "UTF-8"),
    ):
        invalid_candidate = replace(
            candidate, blob_oid=audit.git_blob_sha1(raw)
        )
        with pytest.raises(audit.StructureError, match=match):
            audit.parse_initial_blob(invalid_candidate, raw)


@pytest.mark.parametrize(
    "mutator, match",
    [
        (lambda row: row.pop("published"), "published"),
        (lambda row: row.update({"affected": []}), "affected"),
        (
            lambda row: row.update(
                {
                    "affected": [
                        {"package": {"ecosystem": "", "name": "fixture"}}
                    ]
                }
            ),
            "package",
        ),
        (
            lambda row: row.update({"summary": "nul\u0000inside"}),
            "NUL",
        ),
        (
            lambda row: row.update(
                {
                    "published": "2022-02-19T00:00:00Z",
                    "modified": "2022-02-18T00:00:00Z",
                }
            ),
            "modified",
        ),
    ],
)
def test_required_structure_rejects(
    mutator: object,
    match: str,
) -> None:
    payload = _advisory(GHSA_A)
    assert callable(mutator)
    mutator(payload)
    raw = json.dumps(payload, ensure_ascii=False).encode()
    candidate = replace(_candidate(), blob_oid=audit.git_blob_sha1(raw))
    with pytest.raises(audit.StructureError, match=match):
        audit.parse_initial_blob(candidate, raw)


def test_source_window_requires_both_published_and_availability() -> None:
    rows = [
        replace(
            _event(2022, 2, 12, 0, ecosystem="PyPI"),
            published_at_utc="2022-02-11T22:59:38Z",
        ),
        _event(2025, 12, 31, 1, ecosystem="PyPI"),
        replace(
            _event(2025, 12, 31, 2, ecosystem="PyPI"),
            published_at_utc="2026-01-01T00:00:00Z",
        ),
        replace(
            _event(2025, 12, 31, 3, ecosystem="PyPI"),
            availability_at_utc="2026-01-01T12:00:00Z",
        ),
        replace(
            _event(2022, 2, 12, 4, ecosystem="PyPI"),
            published_at_utc="2022-02-11T22:59:37Z",
        ),
    ]
    selected = [row for row in rows if audit.in_source_window(row)]
    assert selected == rows[:2]


def test_2026_first_add_blob_remains_opaque_and_cannot_reject_source(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write_advisory(repo, GHSA_A)
    _commit_all(
        repo,
        "valid source-window advisory",
        author_date="2022-02-18T00:00:00Z",
        committer_date="2022-02-18T00:00:00Z",
    )
    backfill_path = _path(repo, GHSA_B, bucket="2025/12")
    backfill_path.parent.mkdir(parents=True, exist_ok=True)
    backfill_path.write_text(
        json.dumps(
            {
                "id": GHSA_B,
                "published": "2026-01-02T00:00:00.123456789Z",
                "affected": "TOP_SECRET_BACKFILL_PACKAGE",
                "summary": "nul\u0000would-fail-full-parser",
                "unknown_future_field": {"semantic": "must-not-open"},
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    _commit_all(
        repo,
        "pre-2026 commit with post-window published clock",
        author_date="2025-12-30T00:00:00Z",
        committer_date="2025-12-30T00:00:00Z",
    )
    opaque_path = _path(repo, GHSA_C, bucket="2026/01")
    opaque_path.parent.mkdir(parents=True, exist_ok=True)
    opaque_path.write_bytes(
        b"\xff malformed post-window body TOP_SECRET_2026_PACKAGE"
    )
    tip = _commit_all(
        repo,
        "opaque post-window advisory",
        author_date="2026-01-02T00:00:00Z",
        committer_date="2026-01-02T00:00:00Z",
    )
    scan = audit.collect_first_add_candidates(repo, tip)
    corpus = audit.parse_candidate_corpus(repo, scan.candidates)
    assert len(corpus.selected_events) == 1
    assert corpus.candidate_count == 3
    assert corpus.opaque_postwindow_count == 2
    assert corpus.postwindow_count == 2
    assert corpus.prewindow_count == 0

    support = audit.evaluate_support(
        corpus.selected_events,
        source_event_count=corpus.candidate_count,
        prewindow_count=corpus.prewindow_count,
        postwindow_count=corpus.postwindow_count,
        opaque_postwindow_count=corpus.opaque_postwindow_count,
        candidate_raw_hashes_sha256=corpus.candidate_raw_hashes_sha256,
    )
    encoded = json.dumps(support, sort_keys=True)
    assert "TOP_SECRET_BACKFILL_PACKAGE" not in encoded
    assert "TOP_SECRET_2026_PACKAGE" not in encoded
    assert support["aggregate"]["not_selected_opaque_postwindow_events"] == 2


def test_support_battery_passes_and_each_major_gate_can_fail() -> None:
    rows = _passing_events()
    passed = audit.evaluate_support(rows)
    assert passed["decision"] == "SOURCE_SUPPORT_PASS"
    assert passed["all_gates_passed"] is True
    assert all(gate["passed"] for gate in passed["gates"])

    too_few_2025 = [
        row
        for row in rows
        if not (
            row.availability_at_utc.startswith("2025-")
            and int(row.identity_digest[:2], 16) % 2 == 0
        )
    ]
    failed = audit.evaluate_support(too_few_2025)
    assert failed["decision"] == "TERMINAL_REJECT"
    assert any(
        gate["gate_id"] == "events_2025" and not gate["passed"]
        for gate in failed["gates"]
    )

    dominant = [replace(row, ecosystems=("PyPI",)) for row in rows]
    failed = audit.evaluate_support(dominant)
    assert any(
        gate["gate_id"] == "ecosystem_dominance" and not gate["passed"]
        for gate in failed["gates"]
    )

    incomplete = [
        replace(row, summary_nonempty=False, details_nonempty=False)
        if index % 10 == 0
        else row
        for index, row in enumerate(rows)
    ]
    failed = audit.evaluate_support(incomplete)
    assert any(
        gate["gate_id"] == "text_completeness" and not gate["passed"]
        for gate in failed["gates"]
    )


def test_attempt_guard_is_exclusive_hash_chained_and_tamper_evident(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path / "fixture")
    guard = audit.reserve_attempt(
        paths=paths,
        verifier_commit="a" * 40,
        runner_blob="b" * 40,
        started_at_utc=datetime(2026, 7, 24, tzinfo=timezone.utc),
        run_id="00000000-0000-4000-8000-000000000001",
    )
    guard.append("fixture", {"safe": True})
    guard.validate()
    assert paths.sentinel.exists()
    assert paths.manifest.read_bytes().endswith(b"\n")
    with pytest.raises(audit.ProtocolError, match="exists"):
        audit.reserve_attempt(
            paths=paths,
            verifier_commit="a" * 40,
            runner_blob="b" * 40,
        )
    paths.manifest.write_bytes(paths.manifest.read_bytes().replace(b"true", b"fals"))
    with pytest.raises(audit.ProtocolError):
        guard.validate()


def test_fixture_cannot_use_production_paths_or_authorize_mechanism(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write_advisory(repo, GHSA_A)
    tip = _commit_all(
        repo,
        "one advisory",
        author_date="2022-02-18T00:00:00Z",
        committer_date="2022-02-18T00:00:00Z",
    )
    with pytest.raises(audit.ProtocolError, match="disjoint"):
        audit.run_fixture_audit(
            source_repo=repo,
            pinned_commit=tip,
            paths=audit.PRODUCTION_PATHS,
            verifier_commit="a" * 40,
            runner_blob="b" * 40,
        )

    report = audit.run_fixture_audit(
        source_repo=repo,
        pinned_commit=tip,
        paths=_paths(tmp_path / "fixture"),
        verifier_commit="a" * 40,
        runner_blob="b" * 40,
    )
    assert report["execution_authority"] == "offline_fixture"
    assert report["source_audit_authoritative"] is False
    assert report["mechanism_preregistration_authorized"] is False
    assert not report["bindings"]["manifest_artifact"].startswith("/")
    assert not report["bindings"]["sentinel_artifact"].startswith("/")
    encoded = json.dumps(report, sort_keys=True)
    assert GHSA_A not in encoded
    assert "fixture-package" not in encoded
    assert "pkg:fixture/example" not in encoded


def test_deterministic_source_fingerprint_excludes_operational_run_identity(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write_advisory(repo, GHSA_A)
    tip = _commit_all(
        repo,
        "one advisory",
        author_date="2022-02-18T00:00:00Z",
        committer_date="2022-02-18T00:00:00Z",
    )
    scan = audit.collect_first_add_candidates(repo, tip)
    current_tree = audit._verify_current_tree(repo, tip)
    events = audit.parse_candidates(repo, scan.candidates)
    support = audit.evaluate_support(events)
    replay = audit._events_replay_fingerprint(scan, events)
    stable_transport = {
        "candidate_blob_count": 1,
        "candidate_blob_oids_sha256": "c" * 64,
        "candidate_manifest_sha256": "d" * 64,
        "network_fetch_count": 2,
    }
    first = audit.deterministic_source_fingerprint(
        repository_identity={"commit": tip, "tree": "e" * 40},
        current_tree=current_tree,
        scan=scan,
        replay=replay,
        support=support,
        transport=stable_transport,
    )
    second = audit.deterministic_source_fingerprint(
        repository_identity={"commit": tip, "tree": "e" * 40},
        current_tree=current_tree,
        scan=scan,
        replay=replay,
        support=support,
        transport={**stable_transport, "network_fetch_count": 999},
    )
    assert first == second


def test_production_has_no_injectable_transport_and_requires_isolated_child() -> None:
    parameters = inspect.signature(audit._run_bound_source_audit).parameters
    assert tuple(parameters) == ("execution_mode", "isolated_authority")
    assert "fetcher" not in parameters
    assert "disk_guard" not in parameters
    before = {
        path: audit.repository_path(path).exists()
        for path in (
            audit.DEFAULT_SENTINEL,
            audit.DEFAULT_MANIFEST,
            audit.DEFAULT_RAW_DIR,
            audit.DEFAULT_REPORT,
        )
    }
    with pytest.raises(audit.ProtocolError, match="isolated CLI child"):
        audit._run_bound_source_audit(
            execution_mode=audit._PRODUCTION_EXECUTION,
            isolated_authority="0" * 64,
        )
    after = {
        path: audit.repository_path(path).exists()
        for path in before
    }
    assert after == before


def test_transport_environment_and_commands_are_sealed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "path-poison"))
    monkeypatch.setenv("HOME", str(tmp_path / "home-poison"))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "credential.helper")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "fixture-stealer")
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy.invalid")
    monkeypatch.setenv("GIT_HTTP_COOKIEFILE", str(tmp_path / "cookies"))
    environment = audit.sealed_git_environment(tmp_path)
    assert environment["PATH"] == "/usr/bin:/bin"
    assert environment["HOME"] == str(tmp_path.resolve())
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_CONFIG_GLOBAL"] == os.devnull
    assert environment["GIT_TERMINAL_PROMPT"] == "0"
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert environment["GIT_LFS_SKIP_SMUDGE"] == "1"
    assert not any("PROXY" in key.upper() for key in environment)
    assert not any("TOKEN" in key.upper() for key in environment)
    assert not any("COOKIE" in key.upper() for key in environment)
    assert "GIT_CONFIG_COUNT" not in environment
    assert audit.assert_git_executable()["sha256"] == (
        audit.GIT_EXECUTABLE_SHA256
    )

    commit_command = audit.fetch_objects_command(
        tmp_path / "repo.git",
        allow_lazy_fetch=False,
    )
    blob_command = audit.fetch_objects_command(
        tmp_path / "repo.git",
        allow_lazy_fetch=True,
    )
    for command in (commit_command, blob_command):
        assert command[0] == str(audit.GIT_EXECUTABLE)
        assert "--filter=blob:none" in command
        assert "--no-tags" in command
        assert "--no-write-fetch-head" in command
        assert "--stdin" in command
        assert "checkout" not in command
        assert "http.followRedirects=false" in command


def test_two_phase_transport_derives_oids_before_fetching_only_candidate_blobs(
    tmp_path: Path,
) -> None:
    source = _init_repo(tmp_path / "source")
    _write_advisory(source, GHSA_A)
    tip = _commit_all(
        source,
        "one advisory",
        author_date="2022-02-18T00:00:00Z",
        committer_date="2022-02-18T00:00:00Z",
    )
    _git(source, "config", "uploadpack.allowFilter", "true")
    target = tmp_path / "target.git"
    home = tmp_path / "sealed-home"
    home.mkdir()
    environment = audit.sealed_git_environment(home)
    environment["GIT_NO_LAZY_FETCH"] = "1"
    subprocess.run(
        ["git", "-c", "init.templateDir=", "init", "--bare", str(target)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    subprocess.run(
        ["git", "-C", str(target), "remote", "add", "origin", str(source)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    for key, value in (
        ("remote.origin.promisor", "true"),
        ("remote.origin.partialclonefilter", "blob:none"),
    ):
        subprocess.run(
            ["git", "-C", str(target), "config", key, value],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )

    subprocess.run(
        audit.fetch_objects_command(target, allow_lazy_fetch=False),
        input=(tip + "\n").encode(),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    before = audit._object_inventory(target, environment=environment)
    assert before["blob_oids"] == set()

    scan = audit.collect_first_add_candidates(target, tip)
    assert len(scan.candidates) == 1
    candidate_oids = {candidate.blob_oid for candidate in scan.candidates}
    blob_environment = dict(environment)
    blob_environment["GIT_NO_LAZY_FETCH"] = "0"
    subprocess.run(
        audit.fetch_objects_command(target, allow_lazy_fetch=True),
        input=b"".join(oid.encode() + b"\n" for oid in sorted(candidate_oids)),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=blob_environment,
    )
    after = audit._object_inventory(target, environment=environment)
    assert after["blob_oids"] == candidate_oids
    assert len(audit.parse_candidates(target, scan.candidates)) == 1


def test_disk_and_object_caps_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usage = type(
        "Usage",
        (),
        {
            "total": 1 << 50,
            "used": audit.DISK_USED_LIMIT,
            "free": audit.DISK_FREE_FLOOR,
        },
    )()
    monkeypatch.setattr(audit.shutil, "disk_usage", lambda _: usage)
    with pytest.raises(audit.DiskGuardError):
        audit.assert_disk_guard(tmp_path)

    object_dir = tmp_path / "objects"
    object_dir.mkdir()
    (object_dir / "pack").mkdir()
    (object_dir / "pack" / "oversized.pack").write_bytes(b"x")
    monkeypatch.setattr(
        audit,
        "directory_bytes",
        lambda _: audit.GIT_OBJECT_STORE_CAP,
    )
    with pytest.raises(audit.DiskGuardError, match="object store"):
        audit.assert_object_store_guard(object_dir)

    assert audit.plan_candidate_fetch_chunk(
        remaining_count=10_000,
        materialized_raw_bytes=0,
        candidate_manifest_bytes=1024,
        retrieval_manifest_bytes=1024,
    ) <= audit.CANDIDATE_FETCH_CHUNK_MAX
    with pytest.raises(audit.DiskGuardError, match="headroom"):
        audit.plan_candidate_fetch_chunk(
            remaining_count=1,
            materialized_raw_bytes=(
                audit.CANDIDATE_MATERIAL_CAP
                - audit.MANIFEST_GROWTH_RESERVE
                - audit.SINGLE_BLOB_CAP
            ),
            candidate_manifest_bytes=audit.SINGLE_BLOB_CAP,
            retrieval_manifest_bytes=1,
        )


def test_fixture_parent_alias_cannot_reach_production_results(
    tmp_path: Path,
) -> None:
    aliased = _paths(tmp_path / "fixture")
    aliased = replace(
        aliased,
        sentinel=(
            audit.REPOSITORY_ROOT
            / "outside"
            / ".."
            / "results"
            / ".ghad-fixture-alias.started"
        ),
    )
    with pytest.raises(audit.ProtocolError, match="parent segment"):
        audit.validate_fixture_paths(aliased)


def test_fixture_source_repo_cannot_reference_official_or_production_clone(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path / "repo")
    _git(repo, "remote", "add", "origin", audit.OFFICIAL_REMOTE)
    with pytest.raises(audit.ProtocolError, match="official source"):
        audit.validate_fixture_source_repo(repo)

    with pytest.raises(audit.ProtocolError, match="production source"):
        audit.validate_fixture_source_repo(
            audit.repository_path(audit.DEFAULT_RAW_DIR) / "repository.git"
        )
