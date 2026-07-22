from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import preregister_wbtc_stablecoin_finalized_confirmation_relay as prereg


ARTIFACT = Path(
    "results/wbtc_stablecoin_finalized_confirmation_relay_"
    "preregistration_2026-07-23.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_preregistration_is_hash_bound_and_reproducible() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    prereg.validate_preregistration(payload)
    assert _sha256(ARTIFACT) == (
        "b105051e2b3bdf806c3abff30312889656534f49914ca4e4f584cb9723fb2fe0"
    )
    assert payload["manifest_hash"] == (
        "1466ec5118df70985dda8692df1496d2d03285449ecadd5f8fcdec216b3f978f"
    )
    assert payload["policy_hash"] == (
        "fadc7bf149767416995a9756941e53ba80e81889b5c4b5612d9c4667ea86460d"
    )
    assert payload["preregistration_source"]["sha256"] == (
        "5d10adb12011522c586f63070fc7b963409f8460e7d1995bc64eedbcc8de45fa"
    )
    assert payload["mechanism_decision"]["sha256"] == (
        "5a453f30897a27b9d96fad1af0b8f99d3d5993e439fcfa99927457f11d6ff9ee"
    )


def test_frozen_artifact_preserves_research_boundary_and_comparators() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["source_family_values_previously_opened"] is True
    assert payload["exact_source_incidence_opened"] is False
    assert payload["outcomes_opened"] is False
    assert payload["performance_values_opened"] is False
    assert payload["prior_research_disclosure"][
        "source_family_hypothesis_number"
    ] == 3
    assert payload["outcome_boundary"]["btc_market_rows_read"] == 0
    assert payload["outcome_boundary"]["funding_rows_read"] == 0
    assert payload["outcome_boundary"]["future_return_rows_read"] == 0
    assert [view["name"] for view in payload["comparator_bindings"]] == [
        "wcdr_primary",
        "wtsl_primary",
        "ugci_primary",
        "sealed_prior_stablecoin_bundle",
        "live_portfolio_pure_clocks",
    ]
