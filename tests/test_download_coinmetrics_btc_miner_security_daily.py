from __future__ import annotations

import gzip
import json

import pytest

from training import download_coinmetrics_btc_miner_security_daily as dl


def row(day: str, completion: int, *, hash_rate: str = "500000000") -> dict[str, str]:
    return {
        "time": f"{day}T00:00:00.000000000Z",
        "AssetEODCompletionTime": str(completion),
        "HashRate": hash_rate,
        "IssTotNtv": "900",
        "FeeTotNtv": "10.5",
        "BlkCnt": "144",
    }


def test_source_url_requests_only_frozen_miner_metrics() -> None:
    cfg = dl.Config(output="x", manifest="y")
    url = dl.source_url(cfg)
    assert "HashRate%2CIssTotNtv%2CFeeTotNtv%2CBlkCnt%2CAssetEODCompletionTime" in url
    assert "PriceUSD" not in url
    assert "FlowInEx" not in url


def test_download_paginates_deduplicates_and_sorts() -> None:
    cfg = dl.Config(output="x", manifest="y")
    completion = 1_609_588_800
    pages = [
        {"data": [row("2021-01-02", completion + 86400)], "next_page_url": "/next"},
        {
            "data": [
                row("2021-01-01", completion),
                row("2021-01-02", completion + 86400, hash_rate="600000000"),
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
    assert rows[1]["HashRate"] == "600000000"


def test_completion_time_cannot_precede_daily_close() -> None:
    with pytest.raises(ValueError, match="precedes required daily lag"):
        dl._normalise_row(row("2021-01-01", 1_609_459_200))


def test_invalid_numbers_fail_closed() -> None:
    bad = row("2021-01-01", 1_609_545_600, hash_rate="0")
    with pytest.raises(ValueError, match="HashRate must be positive"):
        dl._normalise_row(bad)
    bad = row("2021-01-01", 1_609_545_600)
    bad["FeeTotNtv"] = "-1"
    with pytest.raises(ValueError, match="FeeTotNtv must be finite and nonnegative"):
        dl._normalise_row(bad)


def test_run_writes_deterministic_gzip_and_manifest(tmp_path, monkeypatch) -> None:
    output = tmp_path / "miner.csv.gz"
    manifest = tmp_path / "miner.json"
    cfg = dl.Config(output=str(output), manifest=str(manifest))
    records = [dl._normalise_row(row("2021-01-01", 1_609_545_600))]
    monkeypatch.setattr(dl, "download_rows", lambda _: (records, "https://source"))
    first = dl.run(cfg)
    first_bytes = output.read_bytes()
    second = dl.run(cfg)
    assert output.read_bytes() == first_bytes
    assert first["sha256"] == second["sha256"]
    payload = json.loads(manifest.read_text())
    assert payload["excluded_on_purpose"]["post_2023_rows"]
    with gzip.open(output, "rt") as handle:
        header = handle.readline().strip().split(",")
    assert header == list(dl.OUTPUT_COLUMNS)
