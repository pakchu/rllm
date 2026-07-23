from __future__ import annotations

import gzip
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from training import probe_bybit_public_trade_sequence_source as probe


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


def directory_payload(*, complete: bool = True) -> bytes:
    names = list(probe.expected_2023_names())
    if not complete:
        names.pop(100)
    return "\n".join(f'<a href="{name}">{name}</a>' for name in names).encode()


def archive_payload(
    header: str = (
        "timestamp,symbol,side,size,price,tickDirection,trdMatchID,"
        "grossValue,homeNotional,foreignNotional"
    ),
) -> bytes:
    row = (
        "1672531200.123,BTCUSDT,Buy,0.001,16500.5,PlusTick,"
        "fixture-trade-id,1,0.001,16.5005"
    )
    return gzip.compress(f"{header}\n{row}\ntrailing,data\n".encode(), mtime=0)


def make_opener(
    *, complete_directory: bool = True, header: str | None = None
) -> tuple[probe.OpenUrl, list[str]]:
    urls: list[str] = []

    def open_url(url: str, _: float) -> FakeResponse:
        urls.append(url)
        if url == probe.DIRECTORY_URL:
            return FakeResponse(url, directory_payload(complete=complete_directory), "text/html")
        return FakeResponse(
            url,
            archive_payload() if header is None else archive_payload(header),
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


def test_pass_artifact_retains_schema_and_hashes_but_not_source_values(tmp_path: Path) -> None:
    opener, urls = make_opener()
    artifact = probe.build_artifact(
        probe.ProbeConfig(output=tmp_path / "result.json"),
        open_url=opener,
        disk_used_gib=250,
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
        assert len(result["first_record_sha256"]) == 64
    assert artifact["candidate_incidence_opened"] is False
    assert artifact["market_outcomes_opened"] is False
    assert artifact["returns_or_pnl_opened"] is False


def test_missing_required_side_rejects_without_trying_later_days(tmp_path: Path) -> None:
    header = (
        "timestamp,symbol,size,price,tickDirection,trdMatchID,"
        "grossValue,homeNotional,foreignNotional"
    )
    opener, urls = make_opener(header=header)
    artifact = probe.build_artifact(
        probe.ProbeConfig(output=tmp_path / "reject.json"),
        open_url=opener,
        disk_used_gib=250,
    )
    assert artifact["decision"] == "REJECT_NO_REPAIR"
    assert artifact["probes"] == []
    assert "required field 'side'" in artifact["failures"][0]
    assert urls == [probe.DIRECTORY_URL, probe.archive_url(probe.PROBE_DAYS[0])]


def test_incomplete_2023_directory_rejects_even_when_headers_pass(tmp_path: Path) -> None:
    opener, _ = make_opener(complete_directory=False)
    artifact = probe.build_artifact(
        probe.ProbeConfig(output=tmp_path / "reject.json"),
        open_url=opener,
        disk_used_gib=250,
    )
    assert artifact["decision"] == "REJECT_NO_REPAIR"
    assert len(artifact["probes"]) == 3
    assert artifact["directory"]["observed_2023_files"] == 364
    assert artifact["failures"] == [
        "official directory lacks exact 2023 daily coverage"
    ]


def test_disk_guard_fails_before_any_network_access(tmp_path: Path) -> None:
    opener, urls = make_opener()
    with pytest.raises(probe.SourceProbeError, match="disk guard"):
        probe.build_artifact(
            probe.ProbeConfig(output=tmp_path / "result.json"),
            open_url=opener,
            disk_used_gib=300,
        )
    assert urls == []


def test_unknown_url_and_non_frozen_day_fail_closed() -> None:
    with pytest.raises(probe.SourceProbeError, match="official"):
        probe.validate_official_url("https://example.com/trading/BTCUSDT/")
    opener, _ = make_opener()
    with pytest.raises(probe.SourceProbeError, match="outside"):
        probe.inspect_archive_day(
            date(2023, 1, 2),
            timeout_seconds=1,
            open_url=opener,
        )
