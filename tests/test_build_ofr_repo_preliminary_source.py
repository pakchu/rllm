from __future__ import annotations

import gzip
import json
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from training import build_ofr_repo_preliminary_source as source


UTC = timezone.utc


def metadata(mnemonic: str, name: str, *, vintage: str = "Preliminary") -> dict:
    return {
        "mnemonic": mnemonic,
        "description": {
            "vintage_approach": vintage,
            "vintage": vintage,
            "notes": "Synthetic source-only fixture",
            "description": "Synthetic repo series",
            "subsetting": "Tenor",
            "subtype": "Interest Rate" if "_AR_" in mnemonic else "Transaction Volume",
            "name": name,
        },
        "rights": {"description": ""},
        "schedule": {
            "observation_period": "Single Day",
            "seasonal_adjustment": "None",
            "observation_frequency": "Daily",
            "start_date": "2019-01-01",
            "last_update": "2026-07-23 14:00:00",
        },
        "parents": [],
        "release": {
            "long_name": "OFR U.S. Repo Markets Data Release",
            "href": "/short-term-funding-monitor/datasets/repo/",
            "frequency": "Daily",
            "short_name": "U.S. Repo Markets",
        },
        "children": [],
        "unit": {
            "type": "Rate" if "_AR_" in mnemonic else "Volume",
            "magnitude": 0,
            "display_magnitude": 0,
            "name": "Percent" if "_AR_" in mnemonic else "USD",
            "precision": 2,
        },
    }


def fixture_payloads() -> dict[str, bytes]:
    names = {
        "REPO-DVP_AR_OO-P": "DVP Average Rate (Preliminary)",
        "REPO-GCF_TV_T-P": "GCF Transaction Volume (Preliminary)",
    }
    mnemonics = [
        {"mnemonic": mnemonic, "series_name": name}
        for mnemonic, name in names.items()
    ] + [
        {"mnemonic": "REPO-DVP_AR_OO-F", "series_name": "DVP Average Rate (Final)"},
        {
            "mnemonic": "REPO-GCF_TV_T-F",
            "series_name": "GCF Transaction Volume (Final)",
        },
    ]
    dataset = {
        "short_name": "U.S. Repo Markets",
        "long_name": "OFR U.S. Repo Markets Data Release",
        "timeseries": {
            "REPO-DVP_AR_OO-P": {
                "timeseries": {
                    "aggregation": [
                        ["2019-01-02", 1.25],
                        ["2019-01-03", None],
                    ],
                    "disclosure_edits": [["2019-01-03", None]],
                },
                "metadata": metadata(
                    "REPO-DVP_AR_OO-P", names["REPO-DVP_AR_OO-P"]
                ),
            },
            "REPO-GCF_TV_T-P": {
                "timeseries": {
                    "aggregation": [
                        ["2019-01-02", 100],
                        ["2019-01-03", 110.5],
                    ]
                },
                "metadata": metadata(
                    "REPO-GCF_TV_T-P", names["REPO-GCF_TV_T-P"]
                ),
            },
        },
    }
    return {
        "mnemonics": json.dumps(mnemonics).encode(),
        "preliminary": json.dumps(dataset).encode(),
    }


def response(url: str, body: bytes) -> source.FetchResponse:
    return source.FetchResponse(
        body=body,
        final_url=url,
        status=200,
        content_type="application/json; charset=utf-8",
    )


def test_frozen_urls_select_only_preliminary_2019_2023() -> None:
    assert source.MNEMONICS_URL.endswith("/metadata/mnemonics?dataset=repo")
    assert source.DATASET_URL == (
        "https://data.financialresearch.gov/v1/series/dataset?"
        "dataset=repo&vintage=p&start_date=2019-01-01&end_date=2023-12-31"
    )
    assert "periodicity=" not in source.DATASET_URL
    assert "remove_nulls=" not in source.DATASET_URL


def test_response_validation_rejects_drift() -> None:
    good = response(source.DATASET_URL, b"{}")
    source._validate_response(source.DATASET_URL, good)
    with pytest.raises(RuntimeError, match="outside official host"):
        source._validate_response(
            source.DATASET_URL,
            source.FetchResponse(b"{}", "https://example.com/x", 200, "application/json"),
        )
    with pytest.raises(RuntimeError, match="URL changed"):
        source._validate_response(
            source.DATASET_URL,
            source.FetchResponse(
                b"{}", source.DATASET_URL + "&extra=1", 200, "application/json"
            ),
        )
    with pytest.raises(RuntimeError, match="not JSON"):
        source._validate_response(
            source.DATASET_URL,
            source.FetchResponse(b"<html>", source.DATASET_URL, 200, "text/html"),
        )
    with pytest.raises(RuntimeError, match="redirect chain"):
        source._validate_response(
            source.DATASET_URL,
            source.FetchResponse(
                b"{}",
                source.DATASET_URL,
                200,
                "application/json",
                ("https://data.financialresearch.gov/redirect",),
            ),
        )


def test_mnemonics_filter_preliminary_and_validate_shape() -> None:
    payloads = fixture_payloads()
    names = source.parse_mnemonics(payloads["mnemonics"])
    assert set(names) == {"REPO-DVP_AR_OO-P", "REPO-GCF_TV_T-P"}
    assert source._parse_mnemonic("REPO-TRIV1_OV_T-P") == (
        "TRIV1",
        "OV",
        "T",
    )
    with pytest.raises(RuntimeError, match="non-preliminary"):
        source._parse_mnemonic("REPO-DVP_AR_OO-F")
    foreign = json.dumps(
        [{"mnemonic": "SOFR-P", "series_name": "Not repo"}]
    ).encode()
    with pytest.raises(RuntimeError, match="foreign mnemonic"):
        source.parse_mnemonics(foreign)
    catalog = source._parse_mnemonic_catalog(payloads["mnemonics"])
    source._validate_preliminary_final_correspondence(catalog)
    catalog["REPO-DVP_AR_OO-F"] = "Different Definition (Final)"
    with pytest.raises(RuntimeError, match="series names disagree"):
        source._validate_preliminary_final_correspondence(catalog)


def test_production_metadata_shape_is_frozen_before_candidate_work() -> None:
    assert source.EXPECTED_METADATA_SERIES == 164
    assert source.EXPECTED_PRELIMINARY_SERIES == 82
    assert source.EXPECTED_FINAL_SERIES == 82
    assert source.EXPECTED_SERIES_BY_SEGMENT == {
        "DVP": 18,
        "GCF": 24,
        "TRI": 20,
        "TRIV1": 20,
    }
    assert source.EXPECTED_SERIES_BY_MEASURE == {"AR": 34, "OV": 14, "TV": 34}
    definitions, _, _ = source.build_panel(fixture_payloads())
    with pytest.raises(RuntimeError, match="metadata series count changed"):
        source.validate_expected_source_shape(
            fixture_payloads()["mnemonics"], definitions
        )


def test_dataset_parser_preserves_nulls_and_waits_eight_days() -> None:
    definitions, rows, audit = source.build_panel(fixture_payloads())
    assert len(definitions) == 2
    assert len(rows) == 4
    first = next(
        row
        for row in rows
        if row.mnemonic == "REPO-DVP_AR_OO-P"
        and row.observation_date.isoformat() == "2019-01-02"
    )
    assert first.value == Decimal("1.25")
    assert first.available_at_utc == source.PRELIMINARY_FEED_FLOOR_UTC
    missing = next(row for row in rows if row.value is None)
    assert missing.disclosure_edit is True
    assert missing.available_at_utc == source.PRELIMINARY_FEED_FLOOR_UTC
    assert source._availability(date(2021, 1, 2)) == datetime(
        2021, 1, 10, tzinfo=UTC
    )
    assert audit.disclosure_markers_total == 1
    assert audit.disclosure_markers_retained == 1


def test_transport_gzip_is_decoded_once_and_bounded() -> None:
    payloads = fixture_payloads()
    compressed = dict(payloads)
    compressed["preliminary"] = gzip.compress(payloads["preliminary"])
    definitions, observations, _ = source.build_panel(compressed)
    assert len(definitions) == 2
    assert len(observations) == 4
    with pytest.raises(RuntimeError, match="transport gzip is invalid"):
        source.parse_dataset(source.GZIP_MAGIC + b"broken", {})
    nested = gzip.compress(gzip.compress(payloads["preliminary"]))
    with pytest.raises(RuntimeError, match="transport gzip is nested"):
        source.parse_dataset(nested, {})


def test_out_of_window_disclosure_markers_are_audited_not_normalized() -> None:
    payloads = fixture_payloads()
    document = json.loads(payloads["preliminary"])
    document["timeseries"]["REPO-DVP_AR_OO-P"]["timeseries"][
        "disclosure_edits"
    ] = [
        ["2018-12-31", None],
        ["2019-01-03", None],
        ["2024-01-01", None],
    ]
    payloads["preliminary"] = json.dumps(document).encode()
    _, observations, audit = source.build_panel(payloads)
    assert audit.disclosure_markers_total == 3
    assert audit.disclosure_markers_retained == 1
    assert audit.disclosure_markers_before_window == 1
    assert audit.disclosure_markers_after_window == 1
    assert sum(row.disclosure_edit for row in observations) == 1
    assert all(
        source.START_DATE <= row.observation_date <= source.END_DATE
        for row in observations
    )


def test_dataset_parser_rejects_final_duplicate_future_and_negative_volume() -> None:
    payloads = fixture_payloads()
    names = source.parse_mnemonics(payloads["mnemonics"])
    document = json.loads(payloads["preliminary"])
    document["timeseries"]["REPO-DVP_AR_OO-P"]["metadata"]["description"][
        "vintage"
    ] = "Final"
    with pytest.raises(RuntimeError, match="not preliminary"):
        source.parse_dataset(json.dumps(document).encode(), names)

    document = json.loads(payloads["preliminary"])
    document["timeseries"]["REPO-DVP_AR_OO-P"]["timeseries"]["aggregation"].append(
        ["2019-01-03", 1.3]
    )
    with pytest.raises(RuntimeError, match="duplicate date"):
        source.parse_dataset(json.dumps(document).encode(), names)

    document = json.loads(payloads["preliminary"])
    document["timeseries"]["REPO-DVP_AR_OO-P"]["timeseries"]["aggregation"][1][
        0
    ] = "2024-01-01"
    document["timeseries"]["REPO-DVP_AR_OO-P"]["timeseries"]["disclosure_edits"][
        0
    ][0] = "2024-01-01"
    with pytest.raises(RuntimeError, match="frozen source window"):
        source.parse_dataset(json.dumps(document).encode(), names)

    document = json.loads(payloads["preliminary"])
    document["timeseries"]["REPO-GCF_TV_T-P"]["timeseries"]["aggregation"][0][
        1
    ] = -1
    with pytest.raises(RuntimeError, match="volume must be nonnegative"):
        source.parse_dataset(json.dumps(document).encode(), names)

    document = json.loads(payloads["preliminary"])
    document["timeseries"]["REPO-DVP_AR_OO-P"]["metadata"]["release"][
        "href"
    ] = "/short-term-funding-monitor/datasets/not-repo/"
    with pytest.raises(RuntimeError, match="release href changed"):
        source.parse_dataset(json.dumps(document).encode(), names)


def test_fetch_cache_is_exact_and_offline_replayable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(source, "REPOSITORY_ROOT", tmp_path)
    payloads = fixture_payloads()
    calls: list[str] = []

    def fetcher(url: str, timeout: int) -> source.FetchResponse:
        assert timeout == 7
        calls.append(url)
        name = "mnemonics" if url == source.MNEMONICS_URL else "preliminary"
        return response(url, payloads[name])

    cfg = source.Config(output_dir="artifacts", fetch=True, request_timeout_seconds=7)
    fetched, ledger = source.acquire_sources(
        cfg,
        fetcher=fetcher,
        clock=lambda: datetime(2026, 7, 23, 0, 0, tzinfo=UTC),
    )
    assert fetched == payloads
    assert calls == [source.MNEMONICS_URL, source.DATASET_URL]
    assert len(ledger) == 2
    offline, offline_ledger = source.acquire_sources(
        source.Config(output_dir="artifacts")
    )
    assert offline == payloads
    assert offline_ledger == ledger
    with pytest.raises(RuntimeError, match="refusing refresh"):
        source.acquire_sources(cfg, fetcher=fetcher)


def test_partial_or_tampered_cache_fails_closed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(source, "REPOSITORY_ROOT", tmp_path)
    root = tmp_path / "artifacts/raw"
    root.mkdir(parents=True)
    (root / "repo_mnemonics.json.gz").write_bytes(gzip.compress(b"[]"))
    with pytest.raises(RuntimeError, match="partial OFR source cache"):
        source.acquire_sources(source.Config(output_dir="artifacts"))


def test_cache_replay_rejects_type_tampering_and_symlink_escape(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(source, "REPOSITORY_ROOT", tmp_path)
    payloads = fixture_payloads()

    def fetcher(url: str, _timeout: int) -> source.FetchResponse:
        name = "mnemonics" if url == source.MNEMONICS_URL else "preliminary"
        return response(url, payloads[name])

    cfg = source.Config(output_dir="cache", fetch=True)
    source.acquire_sources(
        cfg,
        fetcher=fetcher,
        clock=lambda: datetime(2026, 7, 23, tzinfo=UTC),
    )
    ledger_path = tmp_path / "cache/raw/fetch_ledger.json"
    ledger = json.loads(ledger_path.read_text())
    ledger[0]["http_status"] = "200"
    ledger_path.write_text(json.dumps(ledger))
    with pytest.raises(RuntimeError, match="field types changed"):
        source.acquire_sources(source.Config(output_dir="cache"))

    escaped_root = tmp_path.parent / f"{tmp_path.name}-escaped"
    escaped_root.mkdir()
    safe_root = tmp_path / "symlinked"
    safe_root.mkdir()
    (safe_root / "raw").symlink_to(escaped_root, target_is_directory=True)
    with pytest.raises(RuntimeError, match="inside repository"):
        source.acquire_sources(source.Config(output_dir="symlinked"))


def test_output_artifacts_are_deterministic_and_outcome_blind(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(source, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(source, "SCRIPT_PATH", Path("builder.py"))
    (tmp_path / "builder.py").write_text("# fixture\n")
    definitions, observations, audit = source.build_panel(fixture_payloads())
    ledger = [
        {
            "name": name,
            "request_url": url,
            "final_url": url,
            "retrieved_at_utc": "2026-07-23T00:00:00+00:00",
            "http_status": 200,
            "content_type": "application/json",
            "redirect_chain": [],
            "bytes": len(fixture_payloads()[name]),
            "sha256": source.sha256_bytes(fixture_payloads()[name]),
        }
        for name, url in source._request_specs()
    ]
    first = source.write_outputs(
        source.Config(output_dir="out", fetch=True, request_timeout_seconds=7),
        definitions,
        observations,
        audit,
        ledger,
    )
    first_panel = (tmp_path / first["observations"]["path"]).read_bytes()
    second = source.write_outputs(
        source.Config(output_dir="out", fetch=False, request_timeout_seconds=999),
        definitions,
        observations,
        audit,
        ledger,
    )
    second_panel = (tmp_path / second["observations"]["path"]).read_bytes()
    assert first["manifest_hash"] == second["manifest_hash"]
    assert first_panel == second_panel
    assert first["config"] == {"output_dir": "out"}
    assert first["research_boundary"]["candidate_incidence_opened"] is False
    assert first["research_boundary"]["btc_market_rows_read"] == 0


def test_repository_path_and_source_decision_are_bound() -> None:
    with pytest.raises(RuntimeError, match="repository-relative"):
        source._repository_path("../outside")
    assert source.sha256_file(source.SOURCE_DECISION) == source.SOURCE_DECISION_SHA256
