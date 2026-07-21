from __future__ import annotations

import json

from training import (
    freeze_blockspace_load_settlement_relay_support_evaluator as freeze,
)


EXPECTED_FREEZE_SOURCE_SHA256 = (
    "db33dda004504ec1ed12e7f05fd8e8d1c3bc34c0e67a7650aa13584c2dd2d77c"
)
EXPECTED_ARTIFACT_SHA256 = (
    "d5eaceea1ed35907b65f8f47427c9a554c13edd506d9c0f763e865467ac66d13"
)
EXPECTED_MANIFEST_HASH = (
    "b8b4a6962010557a187e67b9fca7bd0b65f7e14b215834c398535ecef93466c5"
)


def test_frozen_blsr_evaluator_freeze_artifact() -> None:
    assert freeze.evaluate.sha256_file(freeze.DEFAULT_OUTPUT) == (
        EXPECTED_ARTIFACT_SHA256
    )
    assert (
        freeze.evaluate.sha256_file(
            "training/freeze_blockspace_load_settlement_relay_support_evaluator.py"
        )
        == EXPECTED_FREEZE_SOURCE_SHA256
    )
    payload = json.loads(
        freeze.evaluate._repository_path(freeze.DEFAULT_OUTPUT).read_text(
            encoding="utf-8"
        )
    )
    freeze.validate_manifest(payload)

    assert payload["manifest_hash"] == EXPECTED_MANIFEST_HASH
    assert payload["evaluator_source_commit"] == freeze.EXPECTED_EVALUATOR_COMMIT
    assert payload["evaluator_source_sha256"] == freeze.EXPECTED_EVALUATOR_SHA256
    assert payload["mutable_parameters"] == []
    assert payload["source_support_artifact_exists_before_freeze"] is False


def test_frozen_blsr_evaluator_freeze_opened_no_incidence_or_outcomes() -> None:
    payload = json.loads(
        freeze.evaluate._repository_path(freeze.DEFAULT_OUTPUT).read_text(
            encoding="utf-8"
        )
    )
    boundary = payload["freeze_boundary"]
    assert boundary["source_artifact_bytes_hashed"] is True
    assert boundary["comparator_artifact_bytes_hashed"] is True
    for field in (
        "source_value_rows_read",
        "source_feature_rows_derived",
        "candidate_event_rows_derived",
        "comparator_event_rows_read",
        "support_or_novelty_verdicts_produced",
        "btc_market_rows_loaded",
        "funding_rows_loaded",
        "return_pnl_or_equity_rows_loaded",
        "economic_simulations_run",
        "network_calls",
    ):
        assert boundary[field] == 0
