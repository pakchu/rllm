from __future__ import annotations

import json
from pathlib import Path

from training import preregister_cboe_cross_surface_pressure_grammar as p


ARTIFACT = Path(
    "results/cboe_cross_surface_pressure_grammar_"
    "preregistration_2026-07-24.json"
)
ARTIFACT_SHA256 = (
    "fb3cc457b77e731b0f5ce8d6cdc692f1bcadfc7ea65a059ac4fb7d9cd45ffa33"
)
MANIFEST_HASH = (
    "7d0b0f44afa49468d01b14f93e68ed08e8b2a531278e4d871b29b7c68148d037"
)


def test_frozen_cspg_preregistration_matches_code_and_hash() -> None:
    assert p.sha256_file(ARTIFACT) == ARTIFACT_SHA256
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    p.validate_manifest(payload)
    assert payload == p.build_manifest()
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert payload["research_history_boundary"][
        "new_cspg_pressure_or_token_seen"
    ] is False
    assert payload["research_history_boundary"][
        "new_cspg_market_outcome_seen"
    ] is False
    assert payload["outcome_boundary"]["source_values_decoded"] == 0
    assert payload["outcome_boundary"]["market_rows_loaded"] == 0
    assert payload["outcome_boundary"]["funding_rows_loaded"] == 0
    assert payload["outcome_boundary"]["comparator_rows_decoded"] == 0
