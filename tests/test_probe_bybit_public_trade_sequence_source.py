from __future__ import annotations

import gzip
import inspect
import json
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from training import probe_bybit_public_trade_sequence_source as probe


@pytest.fixture(autouse=True)
def _safe_disk_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probe, "used_gib", lambda _: 250)


class FakeResponse:
    def __init__(self, url: str, payload: bytes, content_type: str) -> None:
        self.status = 200
        self._url = url
        self._stream = BytesIO(payload)
        self.headers = {
            "Content-Length": str(len(payload)),
            "Content-Type": content_type,
            "Last-Modified": "Wed, 22 Jul 2026 00:00:00 GMT",
            "ETag": '"fixture"',
            "Accept-Ranges": "bytes",
        }

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def geturl(self) -> str:
        return self._url

    def getcode(self) -> int:
        return self.status

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None


def directory_payload(*, complete: bool = True, duplicate: bool = False) -> bytes:
    names = list(probe.expected_2023_names())
    if not complete:
        names.pop(100)
    if duplicate:
        names.append(names[0])
    return "\n".join(f'<a href="{name}">{name}</a>' for name in names).encode()


def archive_payload(
    header: str = (
        "timestamp,symbol,side,size,price,tickDirection,trdMatchID,"
        "grossValue,homeNotional,foreignNotional"
    ),
) -> bytes:
    values = [
        "1672531200.123",
        "BTCUSDT",
        "Buy",
        "0.001",
        "16500.5",
        "PlusTick",
        "fixture-trade-id",
        "1",
        "0.001",
        "16.5005",
    ]
    field_count = len(header.split(","))
    while len(values) < field_count:
        values.append("false" if header.endswith(",RPI") else "fixture-extra")
    values = values[:field_count]
    row = ",".join(values)
    return gzip.compress(f"{header}\n{row}\ntrailing,data\n".encode(), mtime=0)


def make_opener(
    monkeypatch: pytest.MonkeyPatch,
    *,
    complete_directory: bool = True,
    duplicate_directory: bool = False,
    header: str | None = None,
    recent_rpi: bool = False,
) -> tuple[probe.OpenUrl, list[str]]:
    urls: list[str] = []
    archive_payloads: dict[str, bytes] = {}
    for day in probe.PROBE_DAYS:
        selected_header = header
        if selected_header is None:
            selected_header = ",".join(probe.FROZEN_BASE_HEADER)
            if recent_rpi and day == probe.PROBE_DAYS[-1]:
                selected_header += ",RPI"
        archive_payloads[day.isoformat()] = archive_payload(selected_header)
    monkeypatch.setattr(
        probe,
        "V1_PREFIX_SHA256",
        {
            day: probe.sha256_bytes(payload[: probe.READ_CHUNK_BYTES])
            for day, payload in archive_payloads.items()
        },
    )

    def open_url(url: str, _: float) -> FakeResponse:
        urls.append(url)
        if url == probe.DIRECTORY_URL:
            return FakeResponse(
                url,
                directory_payload(
                    complete=complete_directory,
                    duplicate=duplicate_directory,
                ),
                "text/html",
            )
        day = next(
            day
            for day in probe.PROBE_DAYS
            if url == probe.archive_url(day)
        )
        return FakeResponse(
            url,
            archive_payloads[day.isoformat()],
            "application/gzip",
        )

    return open_url, urls


def test_frozen_urls_and_calendar_are_exact() -> None:
    assert probe.PROBE_DAYS == (
        date(2020, 3, 25),
        date(2023, 1, 1),
        date(2026, 7, 22),
    )
    assert len(probe.expected_2023_names()) == 365
    assert probe.expected_2023_names()[0] == "BTCUSDT2023-01-01.csv.gz"
    assert probe.expected_2023_names()[-1] == "BTCUSDT2023-12-31.csv.gz"


def test_pass_artifact_retains_schema_and_hashes_but_not_source_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opener, urls = make_opener(monkeypatch)
    artifact = probe.build_artifact(
        probe.ProbeConfig(output=tmp_path / "result.json"),
        open_url=opener,
    )
    assert artifact["decision"] == "SOURCE_FEASIBILITY_PASS"
    assert artifact["directory"]["complete_2023"] is True
    assert len(artifact["probes"]) == 3
    assert urls == [probe.DIRECTORY_URL] + [
        probe.archive_url(day) for day in probe.PROBE_DAYS
    ]
    for result in artifact["probes"]:
        assert result["canonical_field_mapping"] == {
            "timestamp": "timestamp",
            "symbol": "symbol",
            "side": "side",
            "size": "size",
            "price": "price",
            "execution_id": "trdMatchID",
        }
        assert result["first_record_values_retained"] is False
        assert "1672531200" not in str(result)
        assert "fixture-trade-id" not in str(result)
        assert len(result["first_record_raw_sha256"]) == 64
        assert result["logical_csv_records_decompressed"] == 2
        assert result["bytes_decompressed_after_first_record"] == 0
    assert artifact["candidate_incidence_opened"] is False
    assert artifact["market_outcomes_opened"] is False
    assert artifact["returns_or_pnl_opened"] is False
    assert artifact["v1_decision_authoritative"] is False
    assert artifact["prefix_binding_enforced"] is True
    assert artifact["disk"]["guard_enforced"] is True
    assert artifact["disk"]["guard_filesystem"] == str(probe.REPO_ROOT)
    probe.validate_manifest_hash(artifact)


def test_missing_required_side_rejects_without_trying_later_days(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    header = (
        "timestamp,symbol,size,price,tickDirection,trdMatchID,"
        "grossValue,homeNotional,foreignNotional"
    )
    opener, urls = make_opener(monkeypatch, header=header)
    artifact = probe.build_artifact(
        probe.ProbeConfig(output=tmp_path / "reject.json"),
        open_url=opener,
    )
    assert artifact["decision"] == "REJECT_NO_REPAIR"
    assert artifact["probes"] == []
    assert "required field 'side'" in artifact["failures"][0]
    assert urls == [probe.DIRECTORY_URL, probe.archive_url(probe.PROBE_DAYS[0])]


def test_incomplete_2023_directory_rejects_even_when_headers_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opener, _ = make_opener(monkeypatch, complete_directory=False)
    artifact = probe.build_artifact(
        probe.ProbeConfig(output=tmp_path / "reject.json"),
        open_url=opener,
    )
    assert artifact["decision"] == "REJECT_NO_REPAIR"
    assert len(artifact["probes"]) == 3
    assert artifact["directory"]["observed_2023_files"] == 364
    assert artifact["failures"] == [
        "official directory lacks exact 2023 daily coverage"
    ]


def test_disk_guard_fails_before_any_network_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opener, urls = make_opener(monkeypatch)
    monkeypatch.setattr(probe, "used_gib", lambda _: 300)
    with pytest.raises(probe.SourceProbeError, match="disk guard"):
        probe.build_artifact(
            probe.ProbeConfig(output=tmp_path / "result.json"),
            open_url=opener,
        )
    assert urls == []


def test_recent_rpi_suffix_is_explicitly_mapped_and_excluded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opener, _ = make_opener(monkeypatch, recent_rpi=True)
    artifact = probe.build_artifact(
        probe.ProbeConfig(output=tmp_path / "result.json"),
        open_url=opener,
    )
    assert artifact["decision"] == "SOURCE_FEASIBILITY_PASS"
    drift = artifact["schema_drift"]
    assert drift["observed"] is True
    assert drift["explicitly_classified"] is True
    assert drift["optional_fields_excluded_from_primary"] == ["RPI"]
    assert drift["by_day"][-1] == {
        "day": "2026-07-22",
        "classification": "explicit_recent_optional_suffix",
        "added_fields": ["RPI"],
        "removed_fields": [],
        "base_fields_reordered": False,
        "accepted": True,
    }


def test_rpi_suffix_on_old_days_rejects_full_header_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    header = ",".join(probe.FROZEN_BASE_HEADER) + ",RPI"
    opener, _ = make_opener(monkeypatch, header=header)
    artifact = probe.build_artifact(
        probe.ProbeConfig(output=tmp_path / "reject.json"),
        open_url=opener,
    )
    assert artifact["decision"] == "REJECT_NO_REPAIR"
    assert artifact["schema_drift"]["by_day"][0]["classification"] == (
        "unclassified_drift"
    )
    assert artifact["schema_drift"]["by_day"][1]["classification"] == (
        "unclassified_drift"
    )
    assert artifact["schema_drift"]["by_day"][2]["accepted"] is True


def test_unclassified_extra_field_rejects_full_header_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    header = ",".join(probe.FROZEN_BASE_HEADER) + ",mystery"
    opener, _ = make_opener(monkeypatch, header=header)
    artifact = probe.build_artifact(
        probe.ProbeConfig(output=tmp_path / "reject.json"),
        open_url=opener,
    )
    assert artifact["decision"] == "REJECT_NO_REPAIR"
    assert "not explicitly classified" in artifact["failures"][-1]


def test_directory_requires_unique_anchor_hrefs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opener, _ = make_opener(monkeypatch, duplicate_directory=True)
    artifact = probe.build_artifact(
        probe.ProbeConfig(output=tmp_path / "reject.json"),
        open_url=opener,
    )
    assert artifact["decision"] == "REJECT_NO_REPAIR"
    assert artifact["directory"]["complete_2023"] is False
    assert artifact["directory"]["duplicate_2023_hrefs"] == [
        "BTCUSDT2023-01-01.csv.gz"
    ]


def test_csv_reader_stops_at_two_logical_records_with_quoted_newline() -> None:
    raw = (
        b'a,b,note\r\n1,2,"line one\nline two"\r\n3,4,forbidden\r\n'
    )
    payload = gzip.compress(raw, mtime=0)
    exact, raw_header, raw_first = probe._decompress_two_csv_records(payload)
    assert exact == b'a,b,note\r\n1,2,"line one\nline two"\r\n'
    assert raw_header == b"a,b,note\r\n"
    assert raw_first == b'1,2,"line one\nline two"\r\n'
    assert b"forbidden" not in exact


def test_decompression_bound_applies_before_large_record_allocation() -> None:
    payload = gzip.compress(b"a" * (probe.MAX_DECOMPRESSED_PREFIX_BYTES + 1), mtime=0)
    with pytest.raises(probe.SourceProbeError, match="decompressed prefix"):
        probe._decompress_two_csv_records(payload)


def test_prefix_mismatch_rejects_before_source_schema_decompression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opener, urls = make_opener(monkeypatch)
    expected = dict(probe.V1_PREFIX_SHA256)
    expected[probe.PROBE_DAYS[0].isoformat()] = "0" * 64
    monkeypatch.setattr(probe, "V1_PREFIX_SHA256", expected)
    artifact = probe.build_artifact(
        probe.ProbeConfig(output=tmp_path / "reject.json"),
        open_url=opener,
    )
    assert artifact["decision"] == "REJECT_NO_REPAIR"
    assert artifact["probes"] == []
    assert artifact["opened_source_days"] == []
    assert artifact["source_values_opened"] is False
    assert artifact["prefix_rejections"] == [
        {
            "day": probe.PROBE_DAYS[0].isoformat(),
            "observed_compressed_prefix_sha256": probe.sha256_bytes(
                archive_payload()[: probe.READ_CHUNK_BYTES]
            ),
            "expected_compressed_prefix_sha256": "0" * 64,
            "source_schema_decompressed": False,
        }
    ]
    assert urls == [probe.DIRECTORY_URL, probe.archive_url(probe.PROBE_DAYS[0])]


def test_build_artifact_has_no_test_only_source_or_disk_bypass() -> None:
    signature = inspect.signature(probe.build_artifact)
    assert set(signature.parameters) == {"cfg", "open_url"}


def test_manifest_scope_survives_write_and_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opener, _ = make_opener(monkeypatch)
    artifact = probe.build_artifact(
        probe.ProbeConfig(output=tmp_path / "result.json"),
        open_url=opener,
    )
    output = tmp_path / "artifact.json"
    probe.write_artifact(output, artifact)
    loaded = json.loads(output.read_text())
    probe.validate_manifest_hash(loaded)
    loaded["decision"] = "tampered"
    with pytest.raises(probe.SourceProbeError, match="manifest"):
        probe.validate_manifest_hash(loaded)


def test_redirect_handler_rejects_before_target_request() -> None:
    handler = probe._NoRedirectHandler()
    request = probe.urllib.request.Request(probe.DIRECTORY_URL)
    with pytest.raises(probe.SourceProbeError, match="redirect rejected"):
        handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://example.com/redirected",
        )


def test_v2_is_bound_to_committed_correction_and_invalid_v1() -> None:
    probe.validate_correction_bindings()
    assert probe.sha256_file(probe.CORRECTION_PATH) == probe.CORRECTION_SHA256
    assert probe.sha256_file(probe.V1_INVALID_PATH) == probe.V1_INVALID_FILE_SHA256


def test_unknown_url_and_non_frozen_day_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(probe.SourceProbeError, match="official"):
        probe.validate_official_url("https://example.com/trading/BTCUSDT/")
    opener, _ = make_opener(monkeypatch)
    with pytest.raises(probe.SourceProbeError, match="outside"):
        probe.inspect_archive_day(
            date(2023, 1, 2),
            expected_prefix_sha256="0" * 64,
            timeout_seconds=1,
            open_url=opener,
        )
