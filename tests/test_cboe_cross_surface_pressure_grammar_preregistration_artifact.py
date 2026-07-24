from __future__ import annotations

import json
from pathlib import Path

from training import preregister_cboe_cross_surface_pressure_grammar as p


ARTIFACT = Path(
    "results/cboe_cross_surface_pressure_grammar_"
    "preregistration_2026-07-24.json"
)
ARTIFACT_SHA256 = (
    "aa22b964af179dad2daf617496344eb7c335b2f63cfbf4a32f893c065e58d229"
)
MANIFEST_HASH = (
    "2145ff3cda4632cbc1bd824bc0ecefcc95cca12e201443c3c8e1401689ef02ef"
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
