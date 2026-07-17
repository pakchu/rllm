from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from training import build_cboe_option_flow_panel as builder


def _options_data() -> dict[str, object]:
    ratios = [
        {"name": name, "value": value}
        for name, value in (
            ("TOTAL PUT/CALL RATIO", "0.80"),
            ("INDEX PUT/CALL RATIO", "1.00"),
            ("EQUITY PUT/CALL RATIO", "0.50"),
            ("CBOE VOLATILITY INDEX (VIX) PUT/CALL RATIO", "0.50"),
            ("SPX + SPXW PUT/CALL RATIO", "1.00"),
        )
    ]
    return {
        "ratios": ratios,
        "SUM OF ALL PRODUCTS": [
            {"name": "VOLUME", "call": 1_000, "put": 800, "total": 1_800}
        ],
        "INDEX OPTIONS": [
            {"name": "VOLUME", "call": 400, "put": 400, "total": 800}
        ],
        "EQUITY OPTIONS": [
            {"name": "VOLUME", "call": 400, "put": 200, "total": 600}
        ],
        "CBOE VOLATILITY INDEX (VIX)": [
            {"name": "VOLUME", "call": 100, "put": 50, "total": 150}
        ],
        "SPX + SPXW": [
            {"name": "VOLUME", "call": 250, "put": 250, "total": 500}
        ],
    }


def _flight(options_data: object) -> bytes:
    chunk = f'25:["$",{{"optionsData":{json.dumps(options_data)}}}]'
    encoded = json.dumps(chunk)
    return f"<script>self.__next_f.push([1,{encoded}])</script>".encode()


def test_source_url_is_date_addressable_official_page() -> None:
    assert builder.source_url("2023-09-27") == (
        "https://www.cboe.com/us/options/market_statistics/daily/"
        "?dt=2023-09-27"
    )
    with pytest.raises(ValueError, match="horizon"):
        builder.source_url("2024-01-02")


def test_parse_html_response_distinguishes_no_data_date() -> None:
    assert builder.parse_html_response(_flight(None)) is None
    parsed = builder.parse_html_response(_flight(_options_data()))
    assert parsed is not None
    assert parsed["INDEX OPTIONS"][0]["total"] == 800


def test_normalize_validates_ratios_and_volume_hierarchy() -> None:
    row = builder.normalize_options_data(
        _options_data(), observation="2023-09-27", response_sha256="a" * 64
    )
    assert tuple(row) == builder.PANEL_COLUMNS
    assert row["index_pcr"] == "1.000000"
    assert row["vix_call_volume"] == "100"

    broken = _options_data()
    broken["INDEX OPTIONS"] = [
        {"name": "VOLUME", "call": 400, "put": 400, "total": 799}
    ]
    with pytest.raises(ValueError, match=r"call\+put"):
        builder.normalize_options_data(
            broken, observation="2023-09-27", response_sha256="b" * 64
        )


def test_panel_is_sorted_unique_and_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        builder, "FROZEN_COVERAGE", (2, "2023-09-27", "2023-09-28")
    )
    base = builder.normalize_options_data(
        _options_data(), observation="2023-09-27", response_sha256="a" * 64
    )
    later = dict(base)
    later["observation_date"] = "2023-09-28"
    later["response_sha256"] = "b" * 64
    compact = [
        {"date": row["observation_date"], **{k: v for k, v in row.items() if k != "observation_date"}}
        for row in (later, base)
    ]
    left = builder.panel_bytes(compact)
    right = builder.panel_bytes(list(reversed(compact)))
    assert left == right
    assert left.decode().splitlines()[1].startswith("2023-09-27,")


def test_deterministic_gzip_round_trip(tmp_path: Path) -> None:
    payload = b"observation_date,value\n2023-09-27,1\n"
    left = tmp_path / "left.csv.gz"
    right = tmp_path / "right.csv.gz"
    builder.write_gzip(left, payload)
    builder.write_gzip(right, payload)
    assert left.read_bytes() == right.read_bytes()
    with gzip.open(left, "rb") as handle:
        assert handle.read() == payload


def test_frozen_option_flow_artifacts_replay() -> None:
    paths = builder.artifact_paths("data/cboe_option_flow_2020_2023")
    assert builder.sha256_file(paths["panel"]) == builder.FROZEN_PANEL_SHA256
    payload, rows = builder.validate_snapshot(paths["panel"])
    assert len(rows) == builder.FROZEN_COVERAGE[0]
    assert builder.sha256_bytes(paths["panel"].read_bytes()) == builder.FROZEN_PANEL_SHA256
    assert payload.startswith(b"observation_date,total_pcr,index_pcr,")

    manifest = json.loads(paths["manifest"].read_text())
    core = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    assert manifest["manifest_hash"] == builder.canonical_hash(core)
    assert manifest["source_contract"]["market_or_label_rows_read"] == 0
