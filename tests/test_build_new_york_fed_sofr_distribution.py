from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from training import build_new_york_fed_sofr_distribution as builder


def _raw(
    effective_date: str,
    *,
    p1: object = 4.9,
    p25: object = 4.95,
    rate: object = 5.0,
    p75: object = 5.05,
    p99: object = 5.1,
    volume: object = 1000,
    revision: object = "",
) -> dict[str, object]:
    return {
        "effectiveDate": effective_date,
        "type": "SOFR",
        "percentRate": rate,
        "percentPercentile1": p1,
        "percentPercentile25": p25,
        "percentPercentile75": p75,
        "percentPercentile99": p99,
        "volumeInBillions": volume,
        "revisionIndicator": revision,
    }


def _payload(*rows: dict[str, object]) -> bytes:
    return json.dumps({"refRates": list(rows)}, separators=(",", ":")).encode()


class _FakeFetcher:
    def __init__(self, payloads: dict[int, bytes]) -> None:
        self.payloads = payloads

    def __call__(self, url: str, *, retries: int, timeout: int) -> bytes:
        del retries, timeout
        year = int(url.split("startDate=")[1][:4])
        return self.payloads[year]


def test_annual_url_uses_official_sofr_endpoint_and_exact_year() -> None:
    assert builder.annual_url(2023) == (
        "https://markets.newyorkfed.org/api/rates/secured/sofr/search.json?"
        "startDate=2023-01-01&endDate=2023-12-31"
    )


def test_parse_accepts_descending_rows_and_preserves_official_na() -> None:
    rows = builder.parse_annual_response(
        _payload(
            _raw("2023-01-04", revision="R"),
            _raw("2023-01-03", p1="NA", p25="NA", p75="NA", p99="NA"),
        ),
        year=2023,
    )
    assert [row["effective_date"].isoformat() for row in rows] == [
        "2023-01-03",
        "2023-01-04",
    ]
    assert rows[0]["source_complete"] is False
    assert rows[0]["percentile_1_percent"] is None
    assert rows[1]["revision_indicator"] == "R"


def test_parse_rejects_wrong_type_duplicate_and_bad_distribution() -> None:
    wrong = _raw("2023-01-03")
    wrong["type"] = "TGCR"
    with pytest.raises(ValueError, match="expected SOFR"):
        builder.parse_annual_response(_payload(wrong), year=2023)

    with pytest.raises(ValueError, match="duplicate"):
        builder.parse_annual_response(
            _payload(_raw("2023-01-03"), _raw("2023-01-03")), year=2023
        )

    with pytest.raises(ValueError, match="not ordered"):
        builder.parse_annual_response(
            _payload(_raw("2023-01-03", p99=4.0)), year=2023
        )

    with pytest.raises(ValueError, match="partially missing"):
        builder.parse_annual_response(
            _payload(_raw("2023-01-03", p1="NA")), year=2023
        )


def test_causal_panel_uses_next_business_observation_and_post_revision_time() -> None:
    winter_rows = []
    for day in ("2023-01-06", "2023-01-09"):
        winter_rows.extend(
            builder.parse_annual_response(_payload(_raw(day)), year=2023)
        )
    winter = builder.causal_panel(winter_rows)
    assert winter[0]["publication_date"] == "2023-01-09"
    assert winter[0]["sofr_available_at_utc"] == "2023-01-09T20:00:00+00:00"
    assert winter[0]["summary_available_at_utc"] == "2023-07-01T21:00:00+00:00"

    summer_rows = []
    for day in ("2023-07-03", "2023-07-05"):
        summer_rows.extend(
            builder.parse_annual_response(_payload(_raw(day)), year=2023)
        )
    summer = builder.causal_panel(summer_rows)
    assert summer[0]["publication_date"] == "2023-07-05"
    assert summer[0]["sofr_available_at_utc"] == "2023-07-05T19:00:00+00:00"
    assert summer[0]["summary_available_at_utc"] == "2024-01-01T21:00:00+00:00"
    assert summer[-1]["effective_date"] == "2023-07-03"


def test_causal_panel_rejects_unexpected_source_gap() -> None:
    rows = []
    for day in ("2023-01-03", "2023-01-09"):
        rows.extend(builder.parse_annual_response(_payload(_raw(day)), year=2023))
    with pytest.raises(ValueError, match="unexpected business-day gap"):
        builder.causal_panel(rows)


def test_build_is_byte_deterministic_and_opens_no_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = {
        2022: _payload(_raw("2022-12-29"), _raw("2022-12-30")),
        2023: _payload(_raw("2023-01-03", revision="R")),
    }
    monkeypatch.setattr(
        builder,
        "FROZEN_YEAR_COVERAGE",
        {
            2022: (2, "2022-12-29", "2022-12-30"),
            2023: (1, "2023-01-03", "2023-01-03"),
        },
    )
    monkeypatch.setattr(
        builder,
        "FROZEN_RESPONSE_SHA256",
        {year: hashlib.sha256(payload).hexdigest() for year, payload in payloads.items()},
    )
    cfg = builder.BuildConfig(
        start_year=2022,
        end_year=2023,
        output_dir=str(tmp_path),
    )
    first = builder.build(cfg, fetcher=_FakeFetcher(payloads))
    first_output = Path(first["output"]).read_bytes()
    first_manifest = (tmp_path / "build_manifest.json").read_bytes()
    second = builder.build(cfg, fetcher=_FakeFetcher(payloads))
    assert Path(second["output"]).read_bytes() == first_output
    assert (tmp_path / "build_manifest.json").read_bytes() == first_manifest
    assert first["output_sha256"] == hashlib.sha256(first_output).hexdigest()
    assert first["fetched_rows"] == 3
    assert first["rows"] == 2
    assert first["revised_rows"] == 0
    assert first["protocol"]["outcomes_opened"] is False
    assert first["protocol"]["crypto_market_fields_opened"] is False
    assert first["source_snapshot_date"] == "2026-07-17"
    assert len(first["builder_sha256"]) == 64
    for source in first["sources"]:
        raw_path = Path(source["raw_path"])
        assert raw_path.exists()
        assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == source[
            "response_sha256"
        ]
    with gzip.open(first["output"], "rt", encoding="utf-8") as handle:
        text = handle.read()
    assert (
        "2022-12-29,2022-12-30,2022-12-30T20:00:00+00:00,"
        "2023-04-01T21:00:00+00:00" in text
    )
    assert not any(line.startswith("2023-01-03,") for line in text.splitlines())


def test_build_rejects_2024_and_coverage_drift(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"2024\+"):
        builder.build(
            builder.BuildConfig(
                start_year=2023,
                end_year=2024,
                output_dir=str(tmp_path),
            ),
            fetcher=_FakeFetcher({}),
        )

    with pytest.raises(ValueError, match="coverage changed"):
        builder.build(
            builder.BuildConfig(
                start_year=2023,
                end_year=2023,
                output_dir=str(tmp_path),
            ),
            fetcher=_FakeFetcher({2023: _payload(_raw("2023-01-03"))}),
        )


def test_build_rejects_mutated_historical_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _payload(_raw("2023-01-03"))
    monkeypatch.setattr(
        builder,
        "FROZEN_YEAR_COVERAGE",
        {2023: (1, "2023-01-03", "2023-01-03")},
    )
    monkeypatch.setattr(builder, "FROZEN_RESPONSE_SHA256", {2023: "0" * 64})
    with pytest.raises(ValueError, match="changed from the frozen snapshot"):
        builder.build(
            builder.BuildConfig(
                start_year=2023,
                end_year=2023,
                output_dir=str(tmp_path),
            ),
            fetcher=_FakeFetcher({2023: payload}),
        )
