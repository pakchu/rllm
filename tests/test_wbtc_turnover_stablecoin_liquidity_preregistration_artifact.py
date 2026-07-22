from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import preregister_wbtc_turnover_stablecoin_liquidity as prereg


ARTIFACT = Path(
    "results/wbtc_turnover_stablecoin_liquidity_"
    "preregistration_2026-07-23.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_preregistration_is_hash_bound_and_reproducible() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    prereg.validate_preregistration(payload)
    assert _sha256(ARTIFACT) == (
        "23a1c884306fbde2ef90d02f20de985229c334e1a21992796206d3db6413f92c"
    )
    assert payload["manifest_hash"] == (
        "81f41c68b526a2e22a4da769e973026255d44251f7df996e7cdbc5eb8a66ac4a"
    )
    assert payload["policy_hash"] == (
        "884cc1e5a3674e7aac52f367fac5ee1ba2fa8a541f09de4af900fec1947065a6"
    )
    assert payload["preregistration_source"]["sha256"] == (
        "2bd2b457f7d0dd98db0c854558b1c164b930552d02818b4e2859db8327aca870"
    )
    assert payload["mechanism_decision"]["sha256"] == (
        "f656d2a88964baaf3dead52d89beba25a9a04d653bcf59d7ce2d35e1ca653e2a"
    )


def test_frozen_artifact_cannot_be_misreported_as_clean_preregistration() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["source_incidence_opened"] is True
    assert payload["source_incidence_disclosure"] == {
        "market_outcomes_opened": False,
        "pre_2024_primary_candidates": 236,
        "selection_2023_side_counts": {"long": 12, "short": 13},
        "side_counts": {"long": 189, "short": 47},
        "source_incidence_opened": True,
        "source_support_is_confirmatory": False,
        "year_counts": {"2021": 168, "2022": 43, "2023": 25},
    }
    assert payload["outcomes_opened"] is False
    assert payload["performance_values_opened"] is False
    assert payload["outcome_boundary"]["btc_market_rows_read"] == 0
    assert payload["outcome_boundary"]["funding_rows_read"] == 0
    assert payload["outcome_boundary"]["future_return_rows_read"] == 0
    assert payload["policy"]["research_status"] == "source-seen_outcome-blind"
