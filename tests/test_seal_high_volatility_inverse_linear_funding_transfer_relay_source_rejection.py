import io
import zipfile

from training import seal_high_volatility_inverse_linear_funding_transfer_relay_source_rejection as seal


def fixture_archive() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(
            "BTCUSD_PERP-fundingRate-2023-01.csv",
            "calc_time,funding_interval_hours,last_funding_rate\n"
            "1672531200005,8,-0.00001609\n",
        )
    return output.getvalue()


def test_build_report_is_terminal_before_incidence_or_outcomes() -> None:
    report = seal.build_report(fixture_archive(), {
        "url": seal.FIRST_ARCHIVE_URL,
        "zip_sha256": "a" * 64,
        "checksum_url": seal.FIRST_ARCHIVE_URL + ".CHECKSUM",
        "checksum_response_sha256": "b" * 64,
    })
    assert report["failed_contract"]["offset_milliseconds"] == 5
    assert report["decision"]["status"] == "terminal_source_contract_rejection"
    assert report["decision"]["repair_authorized"] is False
    assert report["access_boundary"]["candidate_incidence_derived"] is False
    assert report["access_boundary"]["economic_outcomes_opened"] is False
