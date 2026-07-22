from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from training import preregister_texas_metar_thermal_relay as tmtr


def _rehash(payload: dict[str, object]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    payload["manifest_hash"] = tmtr.canonical_hash(core)


def test_archive_envelope_is_exact_and_complete() -> None:
    labels = tmtr.archive_labels()
    assert len(labels) == 5844
    assert labels[0] == datetime(2020, 1, 1, tzinfo=timezone.utc)
    assert labels[-1] == datetime(2023, 12, 31, 18, tzinfo=timezone.utc)
    assert {label.hour for label in labels} == set(tmtr.ARCHIVE_HOURS)
    assert len(set(labels)) == len(labels)


def test_archive_url_is_exact_and_rejects_out_of_contract_labels() -> None:
    label = datetime(2023, 1, 1, 6, tzinfo=timezone.utc)
    assert tmtr.archive_url(label) == (
        "https://madis-data.ncep.noaa.gov/madisPublic/data/archive/"
        "2023/01/01/point/metar/netcdf/20230101_0600.gz"
    )
    with pytest.raises(ValueError, match="hour set"):
        tmtr.archive_url(label.replace(hour=5))
    with pytest.raises(ValueError, match="interval"):
        tmtr.archive_url(label.replace(year=2024))
    with pytest.raises(ValueError, match="UTC-aware"):
        tmtr.archive_url(label.replace(tzinfo=None))


@pytest.mark.parametrize(
    ("station", "raw"),
    [
        (
            "KMAF",
            "KMAF 312353Z 24012KT 10SM CLR 18/03 A2985 RMK AO2 SLP084",
        ),
        (
            "KACT",
            "METAR KACT 010051Z 20008KT 10SM CLR 20/07 A2981",
        ),
    ],
)
def test_raw_metar_identity_parser_accepts_only_frozen_panel(
    station: str,
    raw: str,
) -> None:
    parsed = tmtr.parse_raw_metar(station, raw)
    assert parsed["station"] == station
    assert 1 <= parsed["day"] <= 31


@pytest.mark.parametrize(
    ("station", "raw", "match"),
    [
        ("KDFW", "KDFW 312353Z 18005KT 10SM CLR 20/10 A3000", "station"),
        ("KMAF", "KABI 312353Z 18005KT 10SM CLR 20/10 A3000", "identity"),
        ("KMAF", "KMAF 322353Z 18005KT 10SM CLR 20/10 A3000", "identity"),
        ("KMAF", "KMAF 312460Z 18005KT 10SM CLR 20/10 A3000", "identity"),
        ("KMAF", "KMAF 312353Z 18005KT\n10SM CLR", "control"),
    ],
)
def test_raw_metar_parser_fails_closed(station: str, raw: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        tmtr.parse_raw_metar(station, raw)


def test_temporal_envelope_accepts_receipt_time_not_archive_time() -> None:
    archive = datetime(2023, 1, 1, tzinfo=timezone.utc)
    observed = archive - timedelta(minutes=9)
    received = archive - timedelta(minutes=2)
    tmtr.validate_temporal_envelope(archive, observed, received)


@pytest.mark.parametrize(
    ("observed_delta", "received_delta", "match"),
    [
        (timedelta(minutes=-76), timedelta(minutes=-70), "observation"),
        (timedelta(minutes=-10), timedelta(minutes=-11), "receipt"),
        (timedelta(minutes=-20), timedelta(minutes=16), "receipt"),
        (timedelta(minutes=-40), timedelta(minutes=-5), "30 minutes"),
    ],
)
def test_temporal_envelope_rejects_noncausal_or_late_rows(
    observed_delta: timedelta,
    received_delta: timedelta,
    match: str,
) -> None:
    archive = datetime(2023, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match=match):
        tmtr.validate_temporal_envelope(
            archive,
            archive + observed_delta,
            archive + received_delta,
        )


def test_manifest_is_source_only_and_probe_is_disclosed() -> None:
    payload = tmtr.build_manifest()
    tmtr.validate_manifest(payload)
    assert payload["source_id"] == "TMTR"
    assert payload["outcomes_opened"] is False
    assert payload["market_clocks_opened"] is False
    assert payload["historical_source_incidence_opened"] is False
    assert payload["thermal_parser_opened"] is False
    probe = payload["source_only_probe"]
    assert probe["target_station_rows"] == 4
    assert probe["full_interval_incidence_counted"] is False
    assert probe["thermal_values_parsed"] is False
    assert probe["market_or_outcomes_opened"] is False


def test_source_contract_retains_raw_reports_and_forbids_decoded_weather() -> None:
    contract = tmtr.build_manifest()["source_contract"]
    assert contract["archive_interval"]["expected_objects"] == 5844
    assert [row["station"] for row in contract["station_panel"]] == list(tmtr.STATIONS)
    assert "rawMETAR" in contract["retained_fields"]
    assert "decoded MADIS meteorological variables or QC summaries" in contract["forbidden_fields"]
    assert "BTC bars/returns/funding/PnL" in contract["forbidden_fields"]


def test_quality_gates_are_frozen_before_incidence() -> None:
    gates = tmtr.build_manifest()["source_quality_gates"]
    assert gates["transport"]["minimum_object_fraction_each_year"] == 0.995
    assert gates["transport"]["minimum_object_fraction_each_month"] == 0.98
    assert gates["panel"] == {
        "minimum_station_fraction_each_year": 0.95,
        "minimum_all_four_fraction_each_year": 0.90,
        "minimum_all_four_fraction_each_month": 0.85,
        "maximum_consecutive_missing_all_four_anchors": 4,
        "retained_raw_syntax_fraction": 1.0,
        "retained_temporal_envelope_fraction": 1.0,
        "conflicting_target_reports_allowed": 0,
        "imputation_or_forward_fill_allowed": False,
    }
    assert gates["failure_effect"] == "REJECT_NO_REPAIR"


def test_later_mechanism_cannot_repair_source_or_select_on_eval() -> None:
    boundary = tmtr.build_manifest()["later_mechanism_boundary"]
    assert boundary["authorized_now"] is False
    assert boundary["eligible_input"] == "rawMETAR only"
    assert "repair missing stations" in boundary["llm_or_rllm_may_not"]
    assert "select direction or threshold from eval outcomes" in boundary["llm_or_rllm_may_not"]


def test_manifest_hash_and_contract_binding_reject_mutation() -> None:
    payload = tmtr.build_manifest()
    payload["source_quality_gates"]["panel"]["minimum_all_four_fraction_each_year"] = 0.1
    with pytest.raises(RuntimeError, match="hash mismatch"):
        tmtr.validate_manifest(payload)

    payload = tmtr.build_manifest()
    payload["source_contract"]["station_panel"].pop()
    _rehash(payload)
    with pytest.raises(RuntimeError, match="differs from code"):
        tmtr.validate_manifest(payload)


def test_repository_bindings_match_bytes() -> None:
    payload = tmtr.build_manifest()
    assert tmtr.sha256_file(payload["decision_binding"]["path"]) == payload["decision_binding"]["sha256"]
    assert tmtr.sha256_file(payload["implementation_binding"]["path"]) == payload["implementation_binding"]["sha256"]


def test_write_once_is_deterministic_and_fail_closed(tmp_path: Path) -> None:
    output = tmp_path / "tmtr.json"
    payload = tmtr.build_manifest()
    assert tmtr.write_manifest_once(output, payload) == "created"
    assert tmtr.write_manifest_once(output, tmtr.build_manifest()) == "verified_existing"
    assert json.loads(output.read_text()) == payload

    mutated = tmtr.build_manifest()
    mutated["historical_source_incidence_opened"] = True
    _rehash(mutated)
    with pytest.raises(RuntimeError, match="must keep .*false"):
        tmtr.write_manifest_once(tmp_path / "mutated.json", mutated)


def test_repository_artifact_matches_code() -> None:
    artifact = json.loads((tmtr.REPO_ROOT / tmtr.DEFAULT_OUTPUT).read_text())
    tmtr.validate_manifest(artifact)
    assert artifact == tmtr.build_manifest()
