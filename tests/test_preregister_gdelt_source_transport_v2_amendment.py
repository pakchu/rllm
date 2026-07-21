from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import preregister_gdelt_source_transport_v2_amendment as amendment


def test_amendment_changes_only_sparse_bin_transport() -> None:
    payload = amendment.build_payload()
    assert (
        payload["decision"] == "amend_transport_only_for_empirical_global_outage_bins"
    )
    assert payload["transport"]["only_added_contract_field"] == "sparse_bin_policy"
    assert payload["date_only_diagnostic"]["missing_dates"] == [
        "2020-10-20",
        "2023-03-23",
    ]
    assert all(value is False for value in payload["policy_invariants"].values())
    assert payload["outcome_boundary"]["outcomes_opened"] is False
    assert payload["outcome_boundary"]["source_feature_values_inspected"] is False


def test_amendment_is_write_once_and_self_hashed(tmp_path: Path) -> None:
    output = tmp_path / "amendment.json"
    payload = amendment.write_once(output)
    restored = json.loads(output.read_text(encoding="utf-8"))
    assert restored == payload
    unhashed = dict(restored)
    assert unhashed.pop("manifest_hash") == restored["manifest_hash"]
    assert amendment.canonical_hash(unhashed) == restored["manifest_hash"]
    with pytest.raises(FileExistsError, match="write-once"):
        amendment.write_once(output)


def test_amendment_is_bound_to_committed_inputs() -> None:
    assert amendment.sha256_file(amendment.V1_BUILDER) == amendment.V1_BUILDER_SHA256
    assert amendment.sha256_file(amendment.V2_BUILDER) == amendment.V2_BUILDER_SHA256
    assert (
        amendment.sha256_file(amendment.PREREGISTRATION)
        == amendment.PREREGISTRATION_SHA256
    )
    assert (
        amendment.sha256_file(amendment.AMENDMENT_DOCUMENT)
        == amendment.AMENDMENT_DOCUMENT_SHA256
    )
