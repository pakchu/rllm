from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import probe_bybit_public_trade_sequence_source as probe


RESULT = Path(
    "results/bybit_public_trade_sequence_source_feasibility_v2_2026-07-23.json"
)
RESULT_FILE_SHA256 = (
    "916a55f7cd957eff39e84b2ac383c2b49cb342e2012a0f8bc15c3af98b3b3cb0"
)
MANIFEST_HASH = (
    "c36f46c8399692b62d202a7331c9215fc3a5684cc3b2d57ca04d7fc7c83a5f84"
)
SCRIPT_SHA256 = (
    "a808ff71ddbdce447764b8a5ed173a7816a10be2dc0add51628c5b08655f9bee"
)
PREFIXES = {
    "2020-03-25": "dfde89d406b7d179c03fef8116d2668ab52999199e39e07cdd55175a6e749821",
    "2023-01-01": "e91c088cdf75a06fcaecb42be75d59178706798029cdec098ef67e49cc2e7455",
    "2026-07-22": "ad70af6e771252b3c5331234882d7a60defece0813e9ca189c91b64bc93e6039",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact() -> dict[str, object]:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_v2_artifact_is_file_and_manifest_hash_bound() -> None:
    assert sha256(RESULT) == RESULT_FILE_SHA256
    payload = artifact()
    assert payload["manifest_hash_without_self"] == MANIFEST_HASH
    probe.validate_manifest_hash(payload)
    bindings = payload["bindings"]
    assert bindings["script_sha256"] == SCRIPT_SHA256
    assert sha256(Path(bindings["script_path"])) == SCRIPT_SHA256
    assert sha256(Path(bindings["decision_path"])) == bindings["decision_sha256"]
    assert sha256(Path(bindings["correction_path"])) == bindings["correction_sha256"]
    assert sha256(Path(bindings["invalid_v1_path"])) == bindings[
        "invalid_v1_file_sha256"
    ]


def test_v2_replay_matches_exact_frozen_prefixes_and_schema() -> None:
    payload = artifact()
    assert payload["decision"] == "SOURCE_FEASIBILITY_PASS"
    assert payload["failures"] == []
    assert payload["prefix_rejections"] == []
    assert payload["prefix_binding_enforced"] is True
    probes = payload["probes"]
    assert [row["day"] for row in probes] == list(PREFIXES)
    assert {
        row["day"]: row["compressed_prefix_sha256"] for row in probes
    } == PREFIXES
    for row in probes:
        assert row["compressed_prefix_bytes_consumed"] == probe.READ_CHUNK_BYTES
        assert row["logical_csv_records_decompressed"] == 2
        assert row["bytes_decompressed_after_first_record"] == 0
        assert row["first_record_values_retained"] is False
        assert row["canonical_field_mapping"] == {
            "timestamp": "timestamp",
            "symbol": "symbol",
            "side": "side",
            "size": "size",
            "price": "price",
            "execution_id": "trdMatchID",
        }

    drift = payload["schema_drift"]
    assert drift["explicitly_classified"] is True
    assert drift["optional_fields_excluded_from_primary"] == ["RPI"]
    assert [row["classification"] for row in drift["by_day"]] == [
        "frozen_base_header",
        "frozen_base_header",
        "explicit_recent_optional_suffix",
    ]


def test_v2_artifact_preserves_outcome_and_storage_boundary() -> None:
    payload = artifact()
    assert payload["candidate_incidence_opened"] is False
    assert payload["binance_comparator_opened"] is False
    assert payload["market_outcomes_opened"] is False
    assert payload["returns_or_pnl_opened"] is False
    assert payload["v1_decision_authoritative"] is False
    assert payload["directory"]["complete_2023"] is True
    assert payload["directory"]["expected_2023_files"] == 365
    assert payload["directory"]["observed_2023_files"] == 365
    assert payload["disk"]["guard_enforced"] is True
    assert payload["disk"]["limit_gib"] == 300
    assert payload["disk"]["used_gib_before_probe"] < 300
    assert payload["disk"]["raw_archives_persisted"] is False
