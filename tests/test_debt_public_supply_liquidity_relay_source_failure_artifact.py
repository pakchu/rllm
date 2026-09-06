import json
from decimal import Decimal

import pytest

from training import build_debt_public_supply_liquidity_relay_support as support


def test_exact_official_source_failure_is_reproducible():
    raw = json.loads(support.RAW_RESPONSE.read_text())
    failures = []
    for row in raw["data"]:
        difference = (
            Decimal(row["debt_held_public_amt"])
            + Decimal(row["intragov_hold_amt"])
            - Decimal(row["tot_pub_debt_out_amt"])
        )
        if difference:
            failures.append((row["record_date"], difference))
    assert len(raw["data"]) == 893
    assert failures == [("2025-08-04", Decimal("10000000000.00"))]
    with pytest.raises(RuntimeError, match="debt identity failed"):
        support.parse_response(support.RAW_RESPONSE.read_bytes())


def test_terminal_failure_artifact_keeps_all_downstream_data_sealed():
    payload = json.loads(support.RESULT.read_text())
    assert payload["policy_id"] == "DPSLR-24"
    assert payload["support_passed"] is False
    assert payload["advance_to_gross9_novelty"] is False
    assert payload["advance_to_economic_outcomes"] is False
    assert payload["candidate_incidence_computed"] is False
    assert payload["btc_preentry_rows_opened"] == 0
    assert payload["postentry_return_pnl_execution_price_opened"] is False
    assert payload["gross9_rows_opened"] is False
    manifest_hash = payload.pop("manifest_hash")
    assert support.canonical_hash(payload) == manifest_hash
