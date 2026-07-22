from __future__ import annotations

import json
import math
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


def test_realtime_url_freezes_public_live_route_without_historical_interval() -> None:
    label = datetime(2026, 7, 22, 12, tzinfo=timezone.utc)
    assert tmtr.realtime_url(label) == (
        "https://madis-data.ncep.noaa.gov/madisPublic/data/point/metar/netcdf/"
        "20260722_1200.gz"
    )
    with pytest.raises(ValueError, match="hour set"):
        tmtr.realtime_url(label.replace(hour=11))


def _schema() -> tuple[dict[str, int | None], dict[str, dict[str, object]]]:
    dimensions = dict(tmtr.NETCDF_REQUIRED_DIMENSIONS)
    variables = {
        name: {
            "dimensions": tuple(spec["dimensions"]),
            "typecode": spec["typecode"],
            "fill_value": spec["fill_value"],
        }
        for name, spec in tmtr.NETCDF_REQUIRED_VARIABLES.items()
    }
    return dimensions, variables


def test_classic_netcdf_magic_and_exact_required_schema_are_executable() -> None:
    tmtr.validate_netcdf_magic(b"CDF\x01payload")
    with pytest.raises(ValueError, match="CDF1"):
        tmtr.validate_netcdf_magic(b"CDF\x02payload")

    dimensions, variables = _schema()
    tmtr.validate_netcdf_schema(dimensions, variables)
    variables["rawMETAR"]["dimensions"] = ("recNum", "wrong")
    with pytest.raises(ValueError, match="rawMETAR"):
        tmtr.validate_netcdf_schema(dimensions, variables)


def test_schema_permits_unread_dataset_fields_but_never_aliases_required_fields() -> None:
    dimensions, variables = _schema()
    dimensions["otherWidth"] = 12
    variables["temperature"] = {
        "dimensions": ("recNum",),
        "typecode": "f",
        "fill_value": -9999.0,
    }
    tmtr.validate_netcdf_schema(dimensions, variables)
    variables["station"] = variables.pop("stationName")
    with pytest.raises(ValueError, match="stationName.*missing"):
        tmtr.validate_netcdf_schema(dimensions, variables)


@pytest.mark.parametrize(
    ("raw", "width", "expected"),
    [
        (b"KMAF\x00", 5, "KMAF"),
        (b"METAR ", 6, "METAR"),
        (b"KMAF 312353Z AUTO\x00   ", 21, "KMAF 312353Z AUTO"),
    ],
)
def test_fixed_ascii_decoder_uses_c_order_trailing_trim_only(
    raw: bytes,
    width: int,
    expected: str,
) -> None:
    assert tmtr.decode_fixed_ascii(raw, width=width) == expected


@pytest.mark.parametrize(
    ("raw", "width", "match"),
    [
        (b"     ", 5, "empty"),
        (b"K\x00AF ", 5, "embedded NUL"),
        (b"K\nAF ", 5, "control"),
        (b"K\xffAF ", 5, "not ASCII"),
        (b"KMAF", 5, "width"),
    ],
)
def test_fixed_ascii_decoder_fails_closed(
    raw: bytes,
    width: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        tmtr.decode_fixed_ascii(raw, width=width)


def test_epoch_and_correction_decoders_reject_fill_fractional_and_nonzero() -> None:
    assert tmtr.epoch_seconds_to_utc(
        1672530780.0,
        fill_value=1.7976931348623157e308,
    ) == datetime(2022, 12, 31, 23, 53, tzinfo=timezone.utc)
    for invalid in (math.nan, math.inf, 1.5, 1.7976931348623157e308):
        with pytest.raises(ValueError, match="finite non-fill integer"):
            tmtr.epoch_seconds_to_utc(
                invalid,
                fill_value=1.7976931348623157e308,
            )
    assert tmtr.validate_correction(0) == 0
    with pytest.raises(ValueError, match="integer zero"):
        tmtr.validate_correction(1)
    with pytest.raises(ValueError, match="integer"):
        tmtr.validate_correction(0.0)


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
        (
            "KMAF",
            "KMAF 312353Z AUTO 24012KT 10SM CLR 18/03 A2985",
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
        ("KMAF", "KMAF 312353Z NIL", "forbidden"),
        ("KMAF", "KMAF 312353Z COR 18005KT 10SM CLR", "forbidden"),
        ("KMAF", "SPECI KMAF 312353Z 18005KT 10SM CLR", "forbidden"),
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
        (timedelta(minutes=-16), timedelta(minutes=-10), "observation"),
        (timedelta(minutes=-10), timedelta(minutes=-11), "receipt"),
        (timedelta(minutes=-10), timedelta(minutes=16), "receipt"),
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


def test_combined_membership_binds_raw_timestamp_to_timeobs_across_year_rollover() -> None:
    archive = datetime(2023, 1, 1, tzinfo=timezone.utc)
    observed = datetime(2022, 12, 31, 23, 53, tzinfo=timezone.utc)
    received = datetime(2022, 12, 31, 23, 58, tzinfo=timezone.utc)
    raw = "KMAF 312353Z AUTO 24012KT 10SM CLR 18/03 A2985"
    tmtr.validate_metar_row_membership(
        archive_label=archive,
        station="KMAF",
        raw_metar=raw,
        observation_time=observed,
        receipt_time=received,
        report_type="METAR",
        correction=0,
    )
    with pytest.raises(ValueError, match="differs from timeObs"):
        tmtr.validate_metar_row_membership(
            archive_label=archive,
            station="KMAF",
            raw_metar=raw,
            observation_time=observed.replace(minute=54),
            receipt_time=received,
            report_type="METAR",
            correction=0,
        )
    with pytest.raises(ValueError, match="reportType"):
        tmtr.validate_metar_row_membership(
            archive_label=archive,
            station="KMAF",
            raw_metar=raw,
            observation_time=observed,
            receipt_time=received,
            report_type="SPECI",
            correction=0,
        )


def test_public_availability_is_conservative_object_publication_floor() -> None:
    archive = datetime(2023, 1, 1, tzinfo=timezone.utc)
    receipt = archive - timedelta(minutes=2)
    assert tmtr.public_availability_time(archive, receipt) == (
        archive + timedelta(minutes=60)
    )


def test_canonical_row_identity_is_exact_and_receipt_sensitive() -> None:
    archive = datetime(2023, 1, 1, tzinfo=timezone.utc)
    observed = archive - timedelta(minutes=7)
    raw = "KMAF 312353Z 24012KT 10SM CLR 18/03 A2985"
    first = tmtr.canonical_row_identity(
        archive_label=archive,
        station="KMAF",
        observation_time=observed,
        receipt_time=archive - timedelta(minutes=2),
        report_type="METAR",
        correction=0,
        raw_metar=raw,
    )
    second = tmtr.canonical_row_identity(
        archive_label=archive,
        station="KMAF",
        observation_time=observed,
        receipt_time=archive - timedelta(minutes=1),
        report_type="METAR",
        correction=0,
        raw_metar=raw,
    )
    assert len(first) == 64
    assert first != second


def test_object_cardinality_deduplicates_exact_rows_and_rejects_conflicts() -> None:
    first = "a" * 64
    second = "b" * 64
    assert tmtr.resolve_object_row_identities(
        [("KMAF", first), ("KMAF", first), ("KACT", second)]
    ) == {"KMAF": first, "KACT": second}
    with pytest.raises(ValueError, match="conflicting.*KMAF"):
        tmtr.resolve_object_row_identities(
            [("KMAF", first), ("KMAF", second)]
        )


def test_expected_coverage_denominators_include_every_missing_label() -> None:
    labels = tmtr.archive_labels()
    panels: dict[datetime, set[str] | None] = {
        label: set(tmtr.STATIONS) for label in labels
    }
    panels[labels[0]] = None
    panels[labels[1]] = {"KMAF"}
    coverage = tmtr.summarize_expected_coverage(panels)
    assert coverage["expected_labels"] == 5844
    assert coverage["years"]["2020"] == {
        "expected_objects": 1464,
        "available_objects": 1463,
        "complete_panels": 1462,
        "station_counts": {
            "KMAF": 1463,
            "KABI": 1462,
            "KLBB": 1462,
            "KACT": 1462,
        },
    }
    assert coverage["months"]["2020-01"] == {
        "expected_objects": 124,
        "available_objects": 123,
        "complete_panels": 122,
    }
    assert coverage["maximum_consecutive_incomplete_panels"] == 2


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
    assert probe["target_report_type"] == "METAR"
    assert probe["target_correction_value"] == 0
    assert probe["full_interval_incidence_counted"] is False
    assert probe["thermal_values_parsed"] is False
    assert probe["market_or_outcomes_opened"] is False
    availability = payload["availability_contract"]
    assert availability["source_row_availability"] == (
        "max(timeReceived, archive_label + 60 minutes)"
    )
    assert availability[
        "historical_archive_alone_proves_point_in_time_availability"
    ] is False
    assert payload["disk_contract"]["maximum_concurrent_objects"] == 1


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
    assert gates["transport"]["netcdf_schema_corruption_is_fatal"] is True
    assert gates["denominators"][
        "fetched_or_successful_rows_never_define_a_denominator"
    ] is True
    assert gates["denominators"]["panel_streak"] == (
        "walk every expected label in chronological order"
    )


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
