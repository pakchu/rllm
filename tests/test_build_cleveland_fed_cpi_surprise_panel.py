from __future__ import annotations

import csv
import gzip
import io
import json
from datetime import date

import pytest

from training import build_cleveland_fed_cpi_surprise_panel as builder


def _chart() -> dict[str, object]:
    labels = ["01/11", "01/12", "01/13"]

    def series(name: str, values: list[str]) -> dict[str, object]:
        return {"seriesname": name, "data": [{"value": value} for value in values]}

    return {
        "chart": {"subcaption": "2020-12"},
        "categories": [{"category": [{"label": label} for label in labels]}],
        "dataset": [
            series("CPI Inflation", ["0.20", "0.25", ""]),
            series("Core CPI Inflation", ["0.30", "0.35", ""]),
            series("Actual CPI Inflation", ["", "", "0.10"]),
            series("Actual Core CPI Inflation", ["", "", "0.20"]),
        ],
    }


def test_date_labels_roll_into_year_after_reference_month() -> None:
    assert builder._calendar_date("12/31", date(2020, 12, 1)) == date(2020, 12, 31)
    assert builder._calendar_date("01/13", date(2020, 12, 1)) == date(2021, 1, 13)


def test_parser_uses_last_strictly_pre_release_nowcast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(builder, "FROZEN_COVERAGE", (1, "2021-01-13", "2021-01-13"))
    bls = {
        date(2020, 12, 1): {
            "release_time_utc": "2021-01-13 13:30:00+00:00",
            "source_complete": "True",
        }
    }
    payload = json.dumps([_chart()]).encode()
    panel = builder.parse_response(payload, bls_rows=bls)
    rows = list(csv.DictReader(io.StringIO(panel.decode())))
    assert len(rows) == 1
    row = rows[0]
    assert row["latest_nowcast_date"] == "2021-01-12"
    assert float(row["headline_surprise_pct"]) == pytest.approx(-0.15)
    assert float(row["core_surprise_pct"]) == pytest.approx(-0.15)
    assert float(row["composite_surprise_pct"]) == pytest.approx(-0.15)
    assert row["surprise_sign_concordant"] == "1"


def test_parser_rejects_release_date_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(builder, "FROZEN_COVERAGE", (1, "2021-01-14", "2021-01-14"))
    bls = {
        date(2020, 12, 1): {
            "release_time_utc": "2021-01-14 13:30:00+00:00",
            "source_complete": "True",
        }
    }
    with pytest.raises(ValueError, match="disagrees"):
        builder.parse_response(json.dumps([_chart()]).encode(), bls_rows=bls)


def test_frozen_source_artifacts_replay() -> None:
    paths = builder.artifact_paths("data/cleveland_fed_cpi_surprise_2019_2023")
    assert builder.sha256_file(paths["raw"]) == builder.FROZEN_RAW_SHA256
    assert builder.sha256_file(paths["panel"]) == builder.FROZEN_PANEL_SHA256
    raw = builder.read_gzip(paths["raw"])
    replay = builder.parse_response(raw, bls_rows=builder._read_bls_panel())
    assert replay == builder.read_gzip(paths["panel"])
    with gzip.open(paths["panel"], "rt") as handle:
        assert sum(1 for _ in handle) == builder.FROZEN_COVERAGE[0] + 1
    manifest = json.loads(paths["manifest"].read_text())
    core = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    assert manifest["manifest_hash"] == builder.canonical_hash(core)
    assert manifest["source_contract"]["market_or_funding_rows_read"] == 0
