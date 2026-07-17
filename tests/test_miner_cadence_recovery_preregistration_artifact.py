from __future__ import annotations

import json

from training import preregister_miner_cadence_recovery as prereg


ARTIFACT = "results/miner_cadence_recovery_preregistration_2026-07-17.json"
EXPECTED_PROTOCOL_HASH = "20c5aa201e36169c775d64c2882361e3e21bb30c9b0ea88b2888d1b7281d14a1"


def test_frozen_mcr7_preregistration_identity() -> None:
    payload = json.load(open(ARTIFACT))
    prereg.validate_manifest(payload)
    assert payload["manifest_hash"] == EXPECTED_PROTOCOL_HASH
    assert payload["outcomes_opened"] is False
    assert payload["evidence_boundary"]["2024_or_later_miner_source_opened"] is False
    assert payload["support_freeze_before_returns"]["market_or_funding_rows_loaded"] == 0
    assert payload["selection_protocol"]["no_parameter_repair"] is True
