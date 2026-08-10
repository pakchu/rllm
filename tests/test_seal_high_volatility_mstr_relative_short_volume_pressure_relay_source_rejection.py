import hashlib

import pytest

from training import seal_high_volatility_mstr_relative_short_volume_pressure_relay_source_rejection as seal


def test_report_binds_terminal_first_transport_failure() -> None:
    raw = b"forbidden"
    report = seal.build_report(403, raw, {"etag": "x", "last_modified": None})
    assert report["source_binding"]["url"] == seal.FIRST_URL
    assert report["source_binding"]["response_sha256"] == hashlib.sha256(raw).hexdigest()
    assert report["failed_contract"]["first_failure_short_circuit"] is True
    assert report["access_boundary"]["finra_target_rows_opened"] == 0
    assert report["access_boundary"]["postgres_connected"] is False
    assert report["decision"]["status"] == "terminal_source_contract_rejection"
    assert report["decision"]["repair_authorized"] is False
    assert report["manifest_hash"] == seal.support.canonical_hash(
        {key: value for key, value in report.items() if key != "manifest_hash"}
    )


def test_report_rejects_status_drift() -> None:
    with pytest.raises(RuntimeError, match="status drift"):
        seal.build_report(404, b"", {"etag": None, "last_modified": None})
