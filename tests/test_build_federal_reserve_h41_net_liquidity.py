from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from training import build_federal_reserve_h41_net_liquidity as builder


def _index(*dates: str) -> bytes:
    by_year: dict[str, list[str]] = {}
    for value in dates:
        by_year.setdefault(value[:4], []).append(value)
    document = [
        {
            "yearValue": year,
            "Months": [{"MonthName": "fixture", "MonthValue": year, "Dates": values}],
        }
        for year, values in sorted(by_year.items(), reverse=True)
    ]
    return json.dumps(document, separators=(",", ":")).encode()


def _legacy_page(
    observation_date: str = "Dec 28, 2022",
    *,
    assets: str = "8,000,000",
    rrp: str = "2,000,000",
    tga: str = "500,000",
) -> bytes:
    return f"""<!doctype html><html><body><pre>
5. Consolidated Statement of Condition of All Federal Reserve Banks
Millions of dollars
Assets, liabilities, and capital             Eliminations Wednesday
                                                 from {observation_date} Wednesday
Assets
Total assets                                      (0)    {assets}   - 1
5. Consolidated Statement of Condition of All Federal Reserve Banks (continued)
Liabilities
  Reverse repurchase agreements (12)                     {rrp}   + 1
    U.S. Treasury, General Account                        {tga}   - 1
6. Statement of Condition of Each Federal Reserve Bank
</pre></body></html>""".encode()


def _modern_page(
    observation_date: str = "Jan 4, 2023",
    *,
    assets: str = "8,100,000",
    rrp: str = "2,100,000",
    tga: str = "600,000",
) -> bytes:
    return f"""<!doctype html><html><body>
<table><tr><th>Assets, liabilities, and capital</th><th>Eliminations from consolidation</th><th>Wednesday {observation_date}</th><th>Change since</th></tr>
<tr><th>Assets</th><td></td><td></td></tr>
<tr><td>Total assets</td><td>(0)</td><td>{assets}</td><td>- 1</td></tr></table>
<table><tr><th>Assets, liabilities, and capital</th><th>Eliminations from consolidation</th><th>Wednesday {observation_date}</th><th>Change since</th></tr>
<tr><th>Liabilities</th><td></td><td></td></tr>
<tr><td>Reverse repurchase agreements12</td><td></td><td>{rrp}</td></tr>
<tr><td>U.S. Treasury, General Account</td><td></td><td>{tga}</td></tr></table>
</body></html>""".encode()


class _FakeFetcher:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads

    def __call__(self, url: str, *, retries: int, timeout: int) -> bytes:
        del retries, timeout
        return self.payloads[url]


def test_release_url_and_index_enforce_frozen_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert builder.release_url("20231228") == (
        "https://www.federalreserve.gov/releases/h41/20231228/"
    )
    with pytest.raises(ValueError, match="YYYYMMDD"):
        builder.release_url("2023-12-28")

    monkeypatch.setattr(
        builder,
        "FROZEN_YEAR_COVERAGE",
        {2023: (2, "20230105", "20231228")},
    )
    assert builder.parse_release_dates(
        _index("20231228", "20230105"), start_year=2023, end_year=2023
    ) == ["20230105", "20231228"]
    with pytest.raises(ValueError, match="coverage changed"):
        builder.parse_release_dates(
            _index("20230105"), start_year=2023, end_year=2023
        )


def test_parse_legacy_and_modern_consolidated_statements_are_causal() -> None:
    legacy = builder.parse_release_page(
        _legacy_page(),
        release_date="20221229",
    )
    assert legacy == {
        "release_date": "2022-12-29",
        "observation_date": "2022-12-28",
        "available_at_utc": "2022-12-29T21:35:00+00:00",
        "total_assets_usd_millions": 8_000_000,
        "treasury_general_account_usd_millions": 500_000,
        "reverse_repurchase_agreements_usd_millions": 2_000_000,
        "net_liquidity_usd_millions": 5_500_000,
        "source_format": "legacy_pre",
    }

    modern = builder.parse_release_page(
        _modern_page(),
        release_date="20230105",
    )
    assert modern["available_at_utc"] == "2023-01-05T21:35:00+00:00"
    assert modern["net_liquidity_usd_millions"] == 5_400_000
    assert modern["source_format"] == "modern_html_tables"


def test_legacy_parser_ignores_release_notice_references() -> None:
    payload = _legacy_page().replace(
        b"<pre>",
        b"<pre>Consolidated Statement of Condition of All Federal Reserve Banks "
        b"(table 5) was revised. January 7, 2021\n",
    )
    parsed = builder.parse_release_page(payload, release_date="20221229")
    assert parsed["observation_date"] == "2022-12-28"


def test_parse_uses_dst_and_rejects_future_or_noncausal_observation() -> None:
    summer = builder.parse_release_page(
        _modern_page("Jul 5, 2023"),
        release_date="20230706",
    )
    assert summer["available_at_utc"] == "2023-07-06T20:35:00+00:00"
    with pytest.raises(ValueError, match="not causal"):
        builder.parse_release_page(
            _modern_page("Jan 5, 2023"),
            release_date="20230105",
        )
    with pytest.raises(ValueError, match=r"2024\+"):
        builder.build(builder.BuildConfig(start_year=2023, end_year=2024))


def test_build_replays_byte_identically_and_rejects_snapshot_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = _index("20221229", "20230105")
    payloads = {
        builder.INDEX_URL: index,
        builder.release_url("20221229"): _legacy_page(),
        builder.release_url("20230105"): _modern_page(),
    }
    monkeypatch.setattr(
        builder,
        "FROZEN_YEAR_COVERAGE",
        {
            2022: (1, "20221229", "20221229"),
            2023: (1, "20230105", "20230105"),
        },
    )
    cfg = builder.BuildConfig(
        start_year=2022,
        end_year=2023,
        output_dir=str(tmp_path),
    )
    first = builder.build(cfg, fetcher=_FakeFetcher(payloads))
    output_bytes = Path(first["output"]).read_bytes()
    build_manifest_bytes = (tmp_path / "build_manifest.json").read_bytes()
    source_manifest_bytes = (tmp_path / "source_manifest.json").read_bytes()

    replay = builder.build(cfg, from_snapshot=True)
    assert Path(replay["output"]).read_bytes() == output_bytes
    assert (tmp_path / "build_manifest.json").read_bytes() == build_manifest_bytes
    assert (tmp_path / "source_manifest.json").read_bytes() == source_manifest_bytes
    assert first["rows"] == 2
    assert first["legacy_pre_rows"] == 1
    assert first["modern_html_rows"] == 1
    assert first["protocol"]["outcomes_opened"] is False
    assert first["protocol"]["crypto_market_fields_opened"] is False
    assert first["output_sha256"] == hashlib.sha256(output_bytes).hexdigest()
    with gzip.open(first["output"], "rt", encoding="utf-8") as handle:
        output_text = handle.read()
    assert "5500000,legacy_pre" in output_text
    assert "5400000,modern_html_tables" in output_text

    mutated = dict(payloads)
    mutated[builder.release_url("20230105")] = _modern_page(assets="8,100,001")
    with pytest.raises(ValueError, match="changed from the frozen snapshot"):
        builder.build(cfg, fetcher=_FakeFetcher(mutated))


def test_snapshot_corruption_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = _index("20230105")
    payloads = {
        builder.INDEX_URL: index,
        builder.release_url("20230105"): _modern_page(),
    }
    monkeypatch.setattr(
        builder,
        "FROZEN_YEAR_COVERAGE",
        {2023: (1, "20230105", "20230105")},
    )
    cfg = builder.BuildConfig(start_year=2023, end_year=2023, output_dir=str(tmp_path))
    builder.build(cfg, fetcher=_FakeFetcher(payloads))
    snapshot = tmp_path / "raw" / "releases" / "20230105.html.gz"
    snapshot.write_bytes(b"not gzip")
    with pytest.raises(ValueError, match="invalid H.4.1 source snapshot"):
        builder.build(cfg, from_snapshot=True)
