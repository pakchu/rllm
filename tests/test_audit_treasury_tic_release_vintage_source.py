from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
import inspect
import io
import json
from pathlib import Path
import stat
import struct
from typing import Callable
import zipfile

import pytest

from training import audit_treasury_tic_release_vintage_source as audit


FIXED_NOW = datetime(2026, 7, 24, 0, 0, tzinfo=timezone.utc)


def _index_html(
    *,
    row_mutator: Callable[[str, str, str | None], tuple[str, str, str | None]]
    | None = None,
) -> bytes:
    rows: list[str] = []
    for label, filename, released in audit.EXPECTED_RELEASE_ROWS:
        if row_mutator is not None:
            label, filename, released = row_mutator(label, filename, released)
        annotation = f" (released {released})" if released is not None else ""
        rows.append(
            f'<a href="{audit.START_ARCHIVE_PREFIX}{filename}">'
            f"{label}</a> TIC synthetic release{annotation}<br>"
        )
    return ("<html><body><p>" + "".join(rows) + "</p></body></html>").encode()


def _zip_bytes(
    *,
    release_date: date,
    members: tuple[tuple[str, bytes], ...] = (("stable.txt", b"source-only"),),
    compression: int = zipfile.ZIP_DEFLATED,
    extra: bytes = b"",
    archive_comment: bytes = b"",
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=compression,
        allowZip64=False,
    ) as archive:
        archive.comment = archive_comment
        for name, payload in members:
            info = zipfile.ZipInfo(
                name,
                (
                    release_date.year,
                    release_date.month,
                    release_date.day,
                    12,
                    0,
                    0,
                ),
            )
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = compression
            info.extra = extra
            archive.writestr(info, payload)
    return output.getvalue()


def _payload(
    url: str,
    *,
    status: int,
    raw: bytes,
    content_type: str | None = None,
    location: str | None = None,
    extra_headers: tuple[tuple[str, str], ...] = (),
) -> audit.HttpPayload:
    headers: list[tuple[str, str]] = [("Content-Length", str(len(raw)))]
    if content_type is not None:
        headers.append(("Content-Type", content_type))
    if location is not None:
        headers.append(("Location", location))
    headers.extend(extra_headers)
    return audit.HttpPayload(
        requested_url=url,
        status=status,
        headers=tuple(headers),
        raw=raw,
        request_started_at_utc=FIXED_NOW,
        response_completed_at_utc=FIXED_NOW,
    )


class _SyntheticCorpus:
    def __init__(
        self,
        *,
        index_raw: bytes | None = None,
        archive_mutator: Callable[[audit.ReleaseIdentity, int, bytes], bytes]
        | None = None,
        redirect_mutator: Callable[
            [audit.ReleaseIdentity, audit.HttpPayload], audit.HttpPayload
        ]
        | None = None,
    ) -> None:
        self.index_raw = index_raw or _index_html()
        self.identity_audit = audit.parse_archive_index(self.index_raw)
        self.archive_mutator = archive_mutator
        self.redirect_mutator = redirect_mutator
        self.by_start = {
            row.starting_url: (index, row)
            for index, row in enumerate(self.identity_audit.identities)
        }
        self.by_final = {
            row.final_url: (index, row)
            for index, row in enumerate(self.identity_audit.identities)
        }
        self.responses: dict[str, audit.HttpPayload] = {
            audit.ARCHIVE_INDEX_URL: _payload(
                audit.ARCHIVE_INDEX_URL,
                status=200,
                raw=self.index_raw,
                content_type="text/html; charset=UTF-8",
            )
        }
        for url, (_, identity) in self.by_start.items():
            payload = _payload(
                url,
                status=302,
                raw=b"",
                location=identity.final_url,
            )
            if self.redirect_mutator is not None:
                payload = self.redirect_mutator(identity, payload)
            self.responses[url] = payload
        for url, (index, identity) in self.by_final.items():
            raw = _zip_bytes(
                release_date=identity.release_identity_date,
                members=(("stable.txt", f"release-{index}".encode()),),
            )
            if self.archive_mutator is not None:
                raw = self.archive_mutator(identity, index, raw)
            self.responses[url] = _payload(
                url,
                status=200,
                raw=raw,
                content_type="application/zip",
            )


def _paths(tmp_path: Path) -> audit.AuditPaths:
    return audit.AuditPaths(
        sentinel=tmp_path / "attempt.started",
        manifest=tmp_path / "manifest.ndjson",
        raw_dir=tmp_path / "raw",
        report=tmp_path / "report.json",
    )


def _request_intent_count(paths: audit.AuditPaths) -> int:
    return sum(
        json.loads(line)["event"] == "request_intent"
        for line in paths.manifest.read_text().splitlines()
    )


def _run_fixture(
    tmp_path: Path,
    corpus: _SyntheticCorpus,
) -> dict[str, object]:
    return audit.run_fixture_audit(
        paths=_paths(tmp_path),
        responses=corpus.responses,
        verifier_commit="a" * 40,
        runner_blob="b" * 40,
    )


def _mutate_first_local_and_central_name(raw: bytes) -> bytes:
    value = bytearray(raw)
    central = value.index(b"PK\x01\x02")
    local_name_length = struct.unpack_from("<H", value, 26)[0]
    assert local_name_length > 0
    value[30] ^= 0x01
    central_name_length = struct.unpack_from("<H", value, central + 28)[0]
    assert central_name_length > 0
    return bytes(value)


def _mutate_flags(raw: bytes, flags: int) -> bytes:
    value = bytearray(raw)
    central = value.index(b"PK\x01\x02")
    struct.pack_into("<H", value, 6, flags)
    struct.pack_into("<H", value, central + 8, flags)
    return bytes(value)


def test_boundary_constants_and_frozen_release_inventory() -> None:
    assert audit.sha256_file(audit.BOUNDARY_PATH) == audit.BOUNDARY_SHA256
    assert len(audit.EXPECTED_RELEASE_ROWS) == 54
    assert audit.EXPECTED_REQUESTS == 109
    assert audit.EXPECTED_RELEASE_ROWS[0] == (
        "01/18/2022",
        "ticrel_20220118.zip",
        None,
    )
    assert audit.EXPECTED_RELEASE_ROWS[-1] == (
        "06/18/2026",
        "ticrel_20260618.zip",
        None,
    )


def test_exact_index_parses_filename_mismatch_delayed_corelease_and_clock() -> None:
    result = audit.parse_archive_index(_index_html())
    assert len(result.identities) == 54
    assert result.strictly_later_transitions == 52
    assert result.same_day_transitions == 1

    february = next(
        row for row in result.identities if row.index_label_date == date(2022, 2, 15)
    )
    assert february.filename_date == date(2022, 2, 22)
    assert february.release_identity_date == date(2022, 2, 22)
    assert february.available_at_utc.isoformat() == "2022-02-23T05:00:00+00:00"

    october = next(
        row for row in result.identities if row.index_label_date == date(2025, 10, 17)
    )
    november = next(
        row for row in result.identities if row.index_label_date == date(2025, 11, 18)
    )
    assert october.explicit_public_release_date == date(2025, 11, 18)
    assert october.available_at_utc == november.available_at_utc
    assert october.available_at_utc.isoformat() == "2025-11-19T05:00:00+00:00"


@pytest.mark.parametrize(
    "mutator",
    [
        lambda label, filename, released: (
            label,
            "ticrel_20220215.zip",
            released,
        )
        if label == "02/15/2022"
        else (label, filename, released),
        lambda label, filename, released: (
            label,
            filename,
            None,
        )
        if label == "10/17/2025"
        else (label, filename, released),
        lambda label, filename, released: (
            label,
            filename,
            "11-18-2025",
        )
        if label == "09/18/2025"
        else (label, filename, released),
    ],
)
def test_index_rejects_repaired_or_changed_frozen_exceptions(
    mutator: Callable[[str, str, str | None], tuple[str, str, str | None]]
) -> None:
    with pytest.raises(audit.IndexContractError):
        audit.parse_archive_index(_index_html(row_mutator=mutator))


def test_index_rejects_nonexact_delayed_release_parenthetical() -> None:
    raw = _index_html().replace(
        b"(released 11-18-2025)",
        b"(Released 11-18-2025)",
    )
    with pytest.raises(audit.IndexContractError):
        audit.parse_archive_index(raw)
    extra = _index_html().replace(
        b"TIC synthetic release<br>",
        b"TIC synthetic release (Public Release 01-18-2022)<br>",
        1,
    )
    with pytest.raises(audit.IndexContractError):
        audit.parse_archive_index(extra)


def test_http_validation_rejects_ambiguity_and_accepts_optional_media_parameter() -> None:
    good = _payload(
        audit.ARCHIVE_INDEX_URL,
        status=200,
        raw=b"abc",
        content_type="text/html; charset=UTF-8",
    )
    audit.validate_http_payload(
        good,
        expected_statuses=frozenset({200}),
        body_cap=10,
        allowed_content_types=frozenset({"text/html"}),
    )

    duplicate_length = replace(
        good,
        headers=good.headers + (("Content-Length", "3"),),
    )
    with pytest.raises(audit.TransportError):
        audit.validate_http_payload(
            duplicate_length,
            expected_statuses=frozenset({200}),
            body_cap=10,
            allowed_content_types=frozenset({"text/html"}),
        )

    encoded = replace(
        good,
        headers=good.headers + (("Content-Encoding", "gzip"),),
    )
    with pytest.raises(audit.TransportError):
        audit.validate_http_payload(
            encoded,
            expected_statuses=frozenset({200}),
            body_cap=10,
            allowed_content_types=frozenset({"text/html"}),
        )
    partial = replace(
        good,
        headers=good.headers + (("Content-Range", "bytes 0-2/3"),),
    )
    with pytest.raises(audit.TransportError):
        audit.validate_http_payload(
            partial,
            expected_statuses=frozenset({200}),
            body_cap=10,
            allowed_content_types=frozenset({"text/html"}),
        )
    with pytest.raises(audit.TransportError):
        audit.validate_http_payload(
            good,
            expected_statuses=frozenset({200}),
            body_cap=2,
            allowed_content_types=frozenset({"text/html"}),
        )


def test_rate_limiter_counts_every_request_including_manual_redirect() -> None:
    now = [0.0]
    sleeps: list[float] = []

    def monotonic() -> float:
        return now[0]

    def sleep(delay: float) -> None:
        sleeps.append(delay)
        now[0] += delay

    limiter = audit._RateLimiter(monotonic=monotonic, sleep=sleep)
    limiter.wait()
    limiter.wait()
    now[0] += 0.25
    limiter.wait()
    assert sleeps == [1.0, 0.75]


def test_direct_https_request_plan_is_exact_and_unauthenticated() -> None:
    host, path, headers = audit._direct_request_plan(audit.ARCHIVE_INDEX_URL)
    assert host == "home.treasury.gov"
    assert path == "/archives-of-tic-monthly-data-releases"
    assert headers == {
        "User-Agent": audit.CONTACT_USER_AGENT,
        "Accept-Encoding": "identity",
        "Accept": "*/*",
        "Connection": "close",
    }
    assert "Authorization" not in headers
    assert "Cookie" not in headers
    assert not hasattr(audit, "_DirectHttpsTransport")


@pytest.mark.parametrize(
    "url",
    [
        "https://WWW.treasury.gov/resource-center/data-chart-center/tic/Documents/x",
        "https://www.treasury.gov./resource-center/data-chart-center/tic/Documents/x",
        "https://user@www.treasury.gov/resource-center/data-chart-center/tic/Documents/x",
        "https://www.treasury.gov:443/resource-center/data-chart-center/tic/Documents/x",
        "https://www.treasury.gov/resource-center/%2Ftic/Documents/x",
        "https://www.treasury.gov/resource-center/../tic/Documents/x",
    ],
)
def test_strict_url_rejects_authority_encoding_and_path_variants(url: str) -> None:
    with pytest.raises(audit.TransportError):
        audit._strict_url(
            url,
            exact_host="www.treasury.gov",
            exact_path="/resource-center/data-chart-center/tic/Documents/x",
        )


def test_valid_store_and_deflate_archives_pass_byte_level_audit() -> None:
    for compression in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        raw = _zip_bytes(
            release_date=date(2025, 1, 17),
            compression=compression,
        )
        result = audit.audit_zip_archive(
            raw,
            release_identity_date=date(2025, 1, 17),
        )
        assert result.machine_family_keys == frozenset({"stable.txt"})
        assert result.realized_uncompressed_bytes == len(b"source-only")
        assert result.members[0].sha256 == audit.sha256_bytes(b"source-only")


@pytest.mark.parametrize(
    "members",
    [
        (("../escape.txt", b"x"),),
        (("folder/file.txt", b"x"),),
        (("C:drive.txt", b"x"),),
        (("bad.bin", b"x"),),
        (("Stable.txt", b"x"), ("stable.txt", b"y")),
        (("stable.txt", b"PK\x03\x04nested"),),
        (("stable.txt", b""), ("common.pdf", b"%PDF-1.7")),
    ],
)
def test_zip_rejects_unsafe_names_suffixes_nested_payloads_and_empty_machine_support(
    members: tuple[tuple[str, bytes], ...],
) -> None:
    raw = _zip_bytes(release_date=date(2025, 1, 17), members=members)
    with pytest.raises((audit.ZipContractError, audit.SupportError)):
        audit.audit_zip_archive(
            raw,
            release_identity_date=date(2025, 1, 17),
        )


def test_zip_rejects_central_local_name_and_flag_disagreement() -> None:
    raw = _zip_bytes(release_date=date(2025, 1, 17))
    with pytest.raises(audit.ZipContractError):
        audit.audit_zip_archive(
            _mutate_first_local_and_central_name(raw),
            release_identity_date=date(2025, 1, 17),
        )
    with pytest.raises(audit.ZipContractError):
        audit.audit_zip_archive(
            _mutate_flags(raw, 0x0002),
            release_identity_date=date(2025, 1, 17),
        )

    version = bytearray(raw)
    central = version.index(b"PK\x01\x02")
    struct.pack_into("<H", version, 4, 45)
    struct.pack_into("<H", version, central + 6, 45)
    with pytest.raises(audit.ZipContractError):
        audit.audit_zip_archive(
            bytes(version),
            release_identity_date=date(2025, 1, 17),
        )


def test_zip_rejects_archive_comment_prefix_suffix_and_late_timestamp() -> None:
    commented = _zip_bytes(
        release_date=date(2025, 1, 17),
        archive_comment=b"x",
    )
    with pytest.raises(audit.ZipContractError):
        audit.audit_zip_archive(
            commented,
            release_identity_date=date(2025, 1, 17),
        )

    plain = _zip_bytes(release_date=date(2025, 1, 17))
    for malformed in (b"x" + plain, plain + b"x"):
        with pytest.raises(audit.ZipContractError):
            audit.audit_zip_archive(
                malformed,
                release_identity_date=date(2025, 1, 17),
            )

    late = _zip_bytes(release_date=date(2025, 1, 25))
    with pytest.raises(audit.ZipContractError):
        audit.audit_zip_archive(
            late,
            release_identity_date=date(2025, 1, 17),
        )


def test_zip_timestamp_extra_must_match_dos_calendar_date() -> None:
    same_day_epoch = int(
        datetime(2025, 1, 17, 12, 0, tzinfo=timezone.utc).timestamp()
    )
    valid_extra = struct.pack("<HHBI", 0x5455, 5, 1, same_day_epoch)
    valid = _zip_bytes(
        release_date=date(2025, 1, 17),
        extra=valid_extra,
    )
    audit.audit_zip_archive(
        valid,
        release_identity_date=date(2025, 1, 17),
    )

    other_day_epoch = int(
        datetime(2025, 1, 18, 12, 0, tzinfo=timezone.utc).timestamp()
    )
    invalid_extra = struct.pack("<HHBI", 0x5455, 5, 1, other_day_epoch)
    invalid = _zip_bytes(
        release_date=date(2025, 1, 17),
        extra=invalid_extra,
    )
    with pytest.raises(audit.ZipContractError):
        audit.audit_zip_archive(
            invalid,
            release_identity_date=date(2025, 1, 17),
        )

    flags_claim_optional_times_but_central_stores_mtime_only = (
        bytes([0x07]) + struct.pack("<I", same_day_epoch)
    )
    assert audit._timestamp_dates_from_fields(
        {0x5455: flags_claim_optional_times_but_central_stores_mtime_only},
        central_directory=True,
    ) == (date(2025, 1, 17),)
    with pytest.raises(audit.ZipContractError):
        audit._timestamp_dates_from_fields(
            {0x5455: flags_claim_optional_times_but_central_stores_mtime_only},
            central_directory=False,
        )


def test_zip_rejects_crc_damage_extreme_ratio_and_declared_symlink() -> None:
    raw = _zip_bytes(
        release_date=date(2025, 1, 17),
        members=(("stable.txt", b"nondeterministic-source-payload"),),
    )
    entry = audit.parse_zip_entries(
        raw,
        release_identity_date=date(2025, 1, 17),
    )[0]
    damaged = bytearray(raw)
    damaged[entry.data_offset + entry.compressed_size // 2] ^= 0x80
    with pytest.raises(audit.ZipContractError):
        audit.audit_zip_archive(
            bytes(damaged),
            release_identity_date=date(2025, 1, 17),
        )

    ratio_bomb = _zip_bytes(
        release_date=date(2025, 1, 17),
        members=(("stable.txt", b"\x00" * 200_000),),
    )
    with pytest.raises(audit.ZipContractError):
        audit.audit_zip_archive(
            ratio_bomb,
            release_identity_date=date(2025, 1, 17),
        )

    symlink = bytearray(_zip_bytes(release_date=date(2025, 1, 17)))
    central = symlink.index(b"PK\x01\x02")
    struct.pack_into(
        "<I",
        symlink,
        central + 38,
        (stat.S_IFLNK | 0o777) << 16,
    )
    with pytest.raises(audit.ZipContractError):
        audit.audit_zip_archive(
            bytes(symlink),
            release_identity_date=date(2025, 1, 17),
        )

    for nonregular_dos_attribute in (0x08, 0x40):
        nonregular = bytearray(
            _zip_bytes(release_date=date(2025, 1, 17))
        )
        central = nonregular.index(b"PK\x01\x02")
        struct.pack_into(
            "<I",
            nonregular,
            central + 38,
            nonregular_dos_attribute,
        )
        with pytest.raises(audit.ZipContractError):
            audit.audit_zip_archive(
                bytes(nonregular),
                release_identity_date=date(2025, 1, 17),
            )


def test_attempt_guard_is_exclusive_and_detects_manifest_tampering(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    guard = audit.reserve_attempt(
        paths=paths,
        verifier_commit="a" * 40,
        runner_blob="b" * 40,
        started_at_utc=FIXED_NOW,
        run_id="00000000-0000-4000-8000-000000000001",
    )
    assert paths.sentinel.exists()
    assert paths.manifest.read_bytes() == b""
    guard.append("fixture", {"safe": True})
    guard.validate()
    paths.manifest.write_bytes(paths.manifest.read_bytes().replace(b"true", b"fals"))
    with pytest.raises(audit.ProtocolError):
        guard.validate()
    with pytest.raises(audit.ProtocolError):
        audit.reserve_attempt(
            paths=paths,
            verifier_commit="a" * 40,
            runner_blob="b" * 40,
        )

    canonical_paths = _paths(tmp_path / "canonical")
    canonical_guard = audit.reserve_attempt(
        paths=canonical_paths,
        verifier_commit="a" * 40,
        runner_blob="b" * 40,
        started_at_utc=FIXED_NOW,
        run_id="00000000-0000-4000-8000-000000000002",
    )
    canonical_guard.append("fixture", {"safe": True})
    original = canonical_paths.manifest.read_bytes()
    canonical_paths.manifest.write_bytes(b"{ " + original[1:])
    with pytest.raises(audit.ProtocolError):
        canonical_guard.validate()


def test_production_guards_are_not_injectable_and_fixtures_cannot_use_production_paths(
    tmp_path: Path,
) -> None:
    assert not hasattr(audit, "run_source_audit")
    bound_parameters = inspect.signature(audit._run_bound_source_audit).parameters
    assert "fetcher" not in bound_parameters
    assert "transport" not in bound_parameters
    assert "disk_guard" not in bound_parameters
    assert not hasattr(audit, "_execute_source_audit")
    assert not hasattr(audit, "_fetch_and_record")
    assert not hasattr(audit, "_SEALED_PRODUCTION_PATHS")
    assert audit._run_bound_source_audit.__closure__ is None
    corpus = _SyntheticCorpus()
    with pytest.raises(audit.ProtocolError, match="disjoint"):
        audit.run_fixture_audit(
            paths=audit.PRODUCTION_PATHS,
            responses=corpus.responses,
            verifier_commit="a" * 40,
            runner_blob="b" * 40,
        )

    aliased_target = (
        audit.REPOSITORY_ROOT
        / "results"
        / f".fixture-parent-alias-{tmp_path.name}.started"
    )
    aliased = _paths(tmp_path / "aliased")
    aliased = replace(
        aliased,
        sentinel=(
            audit.REPOSITORY_ROOT
            / "outside"
            / ".."
            / "results"
            / aliased_target.name
        ),
    )
    with pytest.raises(audit.ProtocolError, match="parent segment"):
        audit.run_fixture_audit(
            paths=aliased,
            responses=corpus.responses,
            verifier_commit="a" * 40,
            runner_blob="b" * 40,
        )
    assert not aliased_target.exists()

    with pytest.raises(audit.ProtocolError, match="production execution binding"):
        audit._run_bound_source_audit(
            execution_mode=audit._PRODUCTION_EXECUTION,
            paths=_paths(tmp_path / "network-bypass"),
        )

    original_public_paths = audit.PRODUCTION_PATHS
    audit.PRODUCTION_PATHS = _paths(tmp_path / "rebound-public-constant")
    try:
        with pytest.raises(
            audit.ProtocolError,
            match="production execution binding",
        ):
            audit._run_bound_source_audit(
                execution_mode=audit._PRODUCTION_EXECUTION,
                paths=audit.PRODUCTION_PATHS,
            )
    finally:
        audit.PRODUCTION_PATHS = original_public_paths

    production_sentinel = audit.repository_path(audit.DEFAULT_SENTINEL)
    descendant = _paths(tmp_path / "descendant")
    descendant = replace(
        descendant,
        sentinel=production_sentinel / "child",
    )
    with pytest.raises(audit.ProtocolError, match="disjoint"):
        audit.run_fixture_audit(
            paths=descendant,
            responses=corpus.responses,
            verifier_commit="a" * 40,
            runner_blob="b" * 40,
        )

    ancestor = _paths(tmp_path / "ancestor")
    ancestor = replace(
        ancestor,
        sentinel=audit.REPOSITORY_ROOT,
    )
    with pytest.raises(audit.ProtocolError, match="disjoint"):
        audit.run_fixture_audit(
            paths=ancestor,
            responses=corpus.responses,
            verifier_commit="a" * 40,
            runner_blob="b" * 40,
        )


def test_production_requires_isolated_cli_child_before_reservation() -> None:
    with pytest.raises(audit.ProtocolError, match="isolated CLI child"):
        audit._run_bound_source_audit(
            execution_mode=audit._PRODUCTION_EXECUTION,
            isolated_authority="0" * 64,
        )
    for path in (
        audit.DEFAULT_SENTINEL,
        audit.DEFAULT_MANIFEST,
        audit.DEFAULT_RAW_DIR,
        audit.DEFAULT_REPORT,
    ):
        assert not audit.repository_path(path).exists()


def test_attempt_reservation_rejects_symlinked_artifact_parent(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    paths = audit.AuditPaths(
        sentinel=linked / "attempt.started",
        manifest=linked / "manifest.ndjson",
        raw_dir=linked / "raw",
        report=linked / "report.json",
    )
    with pytest.raises(audit.ProtocolError, match="symlink"):
        audit.reserve_attempt(
            paths=paths,
            verifier_commit="a" * 40,
            runner_blob="b" * 40,
        )

    broken = tmp_path / "broken"
    broken.symlink_to(tmp_path / "missing", target_is_directory=True)
    broken_paths = audit.AuditPaths(
        sentinel=broken / "attempt.started",
        manifest=broken / "manifest.ndjson",
        raw_dir=broken / "raw",
        report=broken / "report.json",
    )
    with pytest.raises(audit.ProtocolError, match="symlink"):
        audit.reserve_attempt(
            paths=broken_paths,
            verifier_commit="a" * 40,
            runner_blob="b" * 40,
        )


def test_full_synthetic_one_shot_passes_exact_109_gets_and_support_gates(
    tmp_path: Path,
) -> None:
    corpus = _SyntheticCorpus()
    report = _run_fixture(tmp_path, corpus)
    assert report["decision"] == "SOURCE_SUPPORT_PASS"
    assert _request_intent_count(_paths(tmp_path)) == 109
    assert audit.LIVE_METADATA_URL not in corpus.responses
    assert report["request_contract"]["observed_get_requests"] == 109
    assert report["source_clock"]["strictly_later_transitions"] == 52
    assert report["source_clock"]["same_day_transitions"] == 1
    assert report["archive_support"]["stable_machine_families"] == ["stable.txt"]
    assert report["execution_authority"] == "offline_fixture"
    assert report["source_audit_authoritative"] is False
    assert report["mechanism_preregistration_authorized"] is False
    assert all(value is False for value in report["outcome_boundary"].values())
    assert len(report["archive_support"]["archive_hashes"]) == 54
    assert len(set(report["archive_support"]["archive_hashes"])) == 54
    assert _paths(tmp_path).sentinel.exists()
    assert _paths(tmp_path).report.exists()
    assert report["bindings"]["manifest_sha256"] == audit.sha256_file(
        _paths(tmp_path).manifest
    )

    manifest_records = [
        json.loads(line)
        for line in _paths(tmp_path).manifest.read_text().splitlines()
    ]
    identity_index = next(
        index
        for index, record in enumerate(manifest_records)
        if record["event"] == "identity_finalization"
    )
    first_archive_request = next(
        index
        for index, record in enumerate(manifest_records)
        if record["event"] == "request_intent"
        and record["payload"]["kind"] == "archive_redirect"
    )
    assert identity_index < first_archive_request
    report_without_hash = dict(report)
    claimed_hash = report_without_hash.pop("manifest_hash_without_self")
    assert claimed_hash == audit.sha256_bytes(
        audit.canonical_json_bytes(report_without_hash, newline=False)
    )


def test_sentinel_and_request_intent_are_the_first_durable_source_records(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    corpus = _SyntheticCorpus()
    report = audit.run_fixture_audit(
        paths=paths,
        responses=corpus.responses,
        verifier_commit="a" * 40,
        runner_blob="b" * 40,
    )
    assert report["decision"] == "SOURCE_SUPPORT_PASS"
    assert paths.sentinel.exists()
    records = [
        json.loads(line)
        for line in paths.manifest.read_text().splitlines()
    ]
    assert records[0]["event"] == "request_intent"
    assert records[0]["payload"]["kind"] == "index"
    assert records[1]["event"] == "response_receipt"


def test_full_source_support_rejects_fuzzy_only_family_and_duplicate_archive_hash(
    tmp_path: Path,
) -> None:
    def unique_family(
        identity: audit.ReleaseIdentity,
        index: int,
        _raw: bytes,
    ) -> bytes:
        return _zip_bytes(
            release_date=identity.release_identity_date,
            members=((f"table_{index:02d}.txt", f"row-{index}".encode()),),
        )

    fuzzy_corpus = _SyntheticCorpus(archive_mutator=unique_family)
    fuzzy_paths = _paths(tmp_path / "fuzzy")
    fuzzy_report = _run_fixture(tmp_path / "fuzzy", fuzzy_corpus)
    assert fuzzy_report["decision"] == "TERMINAL_REJECT"
    assert fuzzy_report["failure"] == {
        "exception_class": "SupportError",
        "stage": "support",
    }
    assert _request_intent_count(fuzzy_paths) == 109

    cached: list[bytes] = []

    def duplicate_hash(
        _identity: audit.ReleaseIdentity,
        index: int,
        raw: bytes,
    ) -> bytes:
        if index == 0:
            cached.append(raw)
            return raw
        if index == 1:
            return cached[0]
        return raw

    duplicate_corpus = _SyntheticCorpus(archive_mutator=duplicate_hash)
    duplicate_paths = _paths(tmp_path / "duplicate")
    duplicate_report = _run_fixture(tmp_path / "duplicate", duplicate_corpus)
    assert duplicate_report["decision"] == "TERMINAL_REJECT"
    assert duplicate_report["failure"] == {
        "exception_class": "SupportError",
        "stage": "support",
    }
    assert _request_intent_count(duplicate_paths) == 5


def test_terminal_reject_artifact_is_generic_and_never_authorizes_mechanism(
    tmp_path: Path,
) -> None:
    secret_member = "secret-country-table.txt"

    def mutate(
        _identity: audit.ReleaseIdentity,
        index: int,
        raw: bytes,
    ) -> bytes:
        if index != 0:
            return raw
        return _zip_bytes(
            release_date=date(2022, 1, 18),
            members=(
                (
                    secret_member,
                    b"BTC_RETURN_FUTURE_TARGET_SECRET",
                ),
            ),
            archive_comment=b"reject-me",
        )

    corpus = _SyntheticCorpus(archive_mutator=mutate)
    report = _run_fixture(tmp_path, corpus)
    serialized = json.dumps(report, sort_keys=True)
    assert report["decision"] == "TERMINAL_REJECT"
    assert report["failure"] == {
        "exception_class": "ZipContractError",
        "stage": "zip",
    }
    assert report["mechanism_preregistration_authorized"] is False
    assert report["retry_or_resume_authorized"] is False
    assert "manifest_records" not in serialized
    assert "observed_get_requests" not in serialized
    assert "request_count" not in serialized
    for forbidden in (
        secret_member,
        "BTC_RETURN_FUTURE_TARGET_SECRET",
        audit.START_ARCHIVE_PREFIX,
        "ticrel_20220118.zip",
        "reject-me",
    ):
        assert forbidden not in serialized


def test_redirect_target_drift_terminally_rejects_before_final_get(
    tmp_path: Path,
) -> None:
    def mutate(
        identity: audit.ReleaseIdentity,
        payload: audit.HttpPayload,
    ) -> audit.HttpPayload:
        if identity.index_label_date != date(2022, 1, 18):
            return payload
        return replace(
            payload,
            headers=(
                ("Content-Length", "0"),
                ("Location", identity.final_url + "?changed=1"),
            ),
        )

    corpus = _SyntheticCorpus(redirect_mutator=mutate)
    report = _run_fixture(tmp_path, corpus)
    assert report["decision"] == "TERMINAL_REJECT"
    assert report["failure"] == {
        "exception_class": "TransportError",
        "stage": "transport",
    }
    assert _request_intent_count(_paths(tmp_path)) == 2


def test_direct_cli_help_is_importable_from_repo_root() -> None:
    with pytest.raises(SystemExit) as caught:
        audit.main(["--help"])
    assert caught.value.code == 0


def test_imported_main_cannot_start_authoritative_source_audit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert audit.main([]) == 2
    assert capsys.readouterr().out.strip() == "TICRV source audit: PRECHECK_REJECT"
    for path in (
        audit.DEFAULT_SENTINEL,
        audit.DEFAULT_MANIFEST,
        audit.DEFAULT_RAW_DIR,
        audit.DEFAULT_REPORT,
    ):
        assert not audit.repository_path(path).exists()
