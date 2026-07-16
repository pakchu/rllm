from __future__ import annotations

import gzip
import json

import pytest

from training import download_coinmetrics_btc_blockspace_security_daily as dl


def row(day: str, completion: int, *, fee: str = "10.5") -> dict[str, str]:
    return {
        "time": f"{day}T00:00:00.000000000Z",
        "AssetEODCompletionTime": str(completion),
        "FeeTotNtv": fee,
        "IssTotNtv": "900",
        "BlkCnt": "144",
        "TxCnt": "250000",
    }


def test_source_url_requests_only_frozen_raw_metrics() -> None:
    cfg = dl.Config(output="x", manifest="y")
    url = dl.source_url(cfg)
    assert "FeeTotNtv%2CIssTotNtv%2CBlkCnt%2CTxCnt%2CAssetEODCompletionTime" in url
    assert "FlowInEx" not in url
    assert "PriceUSD" not in url


def test_download_paginates_deduplicates_and_sorts() -> None:
    cfg = dl.Config(output="x", manifest="y")
    completion = 1_609_588_800  # 2021-01-02 UTC
    pages = [
        {"data": [row("2021-01-02", completion + 86400)], "next_page_url": "/next"},
        {
            "data": [
                row("2021-01-01", completion),
                row("2021-01-02", completion + 86400, fee="11.0"),
            ]
        },
    ]
    seen: list[str] = []

    def fetch(url: str) -> dict:
        seen.append(url)
        return pages[len(seen) - 1]

    rows, first = dl.download_rows(cfg, fetch=fetch)
    assert first == dl.source_url(cfg)
    assert len(seen) == 2
    assert [item["observation_date"] for item in rows] == [
        "2021-01-01 00:00:00",
        "2021-01-02 00:00:00",
    ]
    assert rows[1]["FeeTotNtv"] == "11"


def test_completion_time_cannot_precede_daily_close() -> None:
    with pytest.raises(ValueError, match="precedes required daily lag"):
        dl._normalise_row(row("2021-01-01", 1_609_459_200))


def test_invalid_numbers_fail_closed() -> None:
    bad = row("2021-01-01", 1_609_545_600)
    bad["BlkCnt"] = "0"
    with pytest.raises(ValueError, match="BlkCnt must be positive"):
        dl._normalise_row(bad)
    bad = row("2021-01-01", 1_609_545_600, fee="-1")
    with pytest.raises(ValueError, match="FeeTotNtv must be finite and nonnegative"):
        dl._normalise_row(bad)


def test_run_writes_deterministic_gzip_and_manifest(tmp_path, monkeypatch) -> None:
    output = tmp_path / "security.csv.gz"
    manifest = tmp_path / "security.json"
    cfg = dl.Config(output=str(output), manifest=str(manifest))
    records = [dl._normalise_row(row("2021-01-01", 1_609_545_600))]
    monkeypatch.setattr(dl, "download_rows", lambda _: (records, "https://source"))
    first = dl.run(cfg)
    first_bytes = output.read_bytes()
    second = dl.run(cfg)
    assert output.read_bytes() == first_bytes
    assert first["sha256"] == second["sha256"]
    assert json.loads(manifest.read_text())["excluded_on_purpose"]["price_or_market_cap_metrics"]
    with gzip.open(output, "rt") as handle:
        header = handle.readline().strip().split(",")
    assert header == list(dl.OUTPUT_COLUMNS)
