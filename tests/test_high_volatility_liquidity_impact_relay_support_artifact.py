import json

from training import build_high_volatility_liquidity_impact_relay_support as support


def test_support_artifact_passes_frozen_gates():
    payload = json.loads(support.RESULT.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == support.canonical_hash(core)
    assert payload["support_passed"] is True
    assert payload["advance_to_gross9_novelty"] is True
    assert payload["advance_to_economic_outcomes"] is False
    assert all(payload["support_checks"].values())
    assert payload["postentry_return_pnl_execution_price_opened"] is False
    assert payload["gross9_rows_opened"] is False


def test_support_counts_are_balanced_and_distributed():
    payload = json.loads(support.RESULT.read_text())
    for split, minimum in support.MINIMUM.items():
        stats = payload["support"][split]
        assert stats["events"] >= minimum
        assert stats["minority_side_share"] >= 0.20
        assert stats["max_month_share"] <= 0.45
