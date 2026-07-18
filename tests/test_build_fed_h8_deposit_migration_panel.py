from __future__ import annotations

import json
import gzip
import base64
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from training import build_fed_h8_deposit_migration_panel as builder


SAMPLE_ROOT = Path("/tmp/h8-samples")


def _record(name: str) -> builder.ArchiveRecord:
    body = (SAMPLE_ROOT / f"{name}.html").read_bytes()
    return builder.ArchiveRecord(
        release_calendar_key=date.fromisoformat(
            f"{name[:4]}-{name[4:6]}-{name[6:]}"
        ).isoformat(),
        archive_path_date=date.fromisoformat(
            f"{name[:4]}-{name[4:6]}-{name[6:]}"
        ).isoformat(),
        url=builder.ARCHIVE_URL.format(release_date=name),
        response_sha256=builder.sha256_bytes(body),
        body=body,
    )


@pytest.mark.skipif(
    not (SAMPLE_ROOT / "20190104.html").exists(), reason="local official sample absent"
)
def test_legacy_archive_page_parses_and_balances() -> None:
    row = builder.parse_archive_page(_record("20190104"))
    assert row["html_schema"] == "legacy"
    assert row["release_time_utc"] == "2019-01-04T21:15:00+00:00"
    assert row["prior_week_ending"] == "2018-12-19"
    assert row["latest_week_ending"] == "2018-12-26"
    assert Decimal(row["sa_large_other_deposits_latest"]) == Decimal("6835.5")
    assert Decimal(row["sa_small_other_deposits_latest"]) == Decimal("3658.4")


@pytest.mark.skipif(
    not (SAMPLE_ROOT / "20231229.html").exists(), reason="local official sample absent"
)
def test_modern_archive_page_parses_and_balances() -> None:
    row = builder.parse_archive_page(_record("20231229"))
    assert row["html_schema"] == "modern"
    assert row["release_time_utc"] == "2023-12-29T21:15:00+00:00"
    assert row["prior_week_ending"] == "2023-12-13"
    assert row["latest_week_ending"] == "2023-12-20"
    assert Decimal(row["sa_large_other_deposits_latest"]) == Decimal("9990.4")
    assert Decimal(row["sa_small_other_deposits_latest"]) == Decimal("4616.1")


def test_release_date_parser_is_ordered_and_rejects_coverage_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = [
        {
            "yearValue": 2023,
            "Months": [
                {
                    "MonthName": "January",
                    "MonthValue": "202301",
                    "Dates": ["20230113", "20230106"],
                }
            ],
        }
    ]
    monkeypatch.setattr(builder, "START_YEAR", 2023)
    monkeypatch.setattr(builder, "END_YEAR", 2023)
    monkeypatch.setattr(builder, "FROZEN_COVERAGE", (2, "2023-01-06", "2023-01-13"))
    parsed = builder.parse_release_dates(json.dumps(source).encode())
    assert parsed == [date(2023, 1, 6), date(2023, 1, 13)]
    monkeypatch.setattr(builder, "FROZEN_COVERAGE", (3, "2023-01-06", "2023-01-13"))
    with pytest.raises(ValueError, match="coverage changed"):
        builder.parse_release_dates(json.dumps(source).encode())


def test_accounting_validator_fails_closed() -> None:
    values = {}
    for adjustment in builder.ADJUSTMENTS:
        for metric in builder.METRICS:
            values[(adjustment, "large", metric)] = (
                Decimal("120"),
                Decimal("120"),
            )
            values[(adjustment, "small", metric)] = (
                Decimal("80"),
                Decimal("80"),
            )
            values[(adjustment, "domestic", metric)] = (
                Decimal("200"),
                Decimal("200"),
            )
        for group, scale in (
            ("large", Decimal("1.2")),
            ("small", Decimal("0.8")),
            ("domestic", Decimal("2")),
        ):
            values[(adjustment, group, "deposits")] = (100 * scale, 100 * scale)
            values[(adjustment, group, "large_time_deposits")] = (
                40 * scale,
                40 * scale,
            )
            values[(adjustment, group, "other_deposits")] = (
                60 * scale,
                60 * scale,
            )
    builder._validate_accounting(values)
    values[("sa", "small", "borrowings")] = (Decimal("101"), Decimal("100"))
    with pytest.raises(ValueError, match=r"large\+small identity"):
        builder._validate_accounting(values)


def test_frozen_source_artifacts_and_manifest_replay() -> None:
    paths = builder.artifact_paths(builder.BuildConfig.output_dir)
    assert (
        builder.sha256_file(paths["release_dates"])
        == builder.FROZEN_RELEASE_DATES_SNAPSHOT_SHA256
    )
    assert (
        builder.sha256_file(paths["archive"])
        == builder.FROZEN_ARCHIVE_SNAPSHOT_SHA256
    )
    assert builder.sha256_file(paths["panel"]) == builder.FROZEN_PANEL_SHA256
    manifest = json.loads(paths["manifest"].read_text())
    core = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    assert manifest["manifest_hash"] == builder.canonical_hash(core)
    assert manifest["panel"]["rows"] == 365
    assert manifest["source_contract"]["market_or_funding_rows_read"] == 0

    with gzip.open(paths["archive"], "rt", encoding="utf-8") as handle:
        item = json.loads(next(handle))
    body = base64.b64decode(item["body_b64"], validate=True)
    record = builder.ArchiveRecord(
        release_calendar_key=item["release_calendar_key"],
        archive_path_date=item["archive_path_date"],
        url=item["url"],
        response_sha256=item["response_sha256"],
        body=body,
    )
    row = builder.parse_archive_page(record)
    assert row["release_date"] == "2017-01-06"
    assert row["html_schema"] == "legacy"
