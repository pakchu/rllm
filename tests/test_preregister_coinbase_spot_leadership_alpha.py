from __future__ import annotations

from training import preregister_coinbase_spot_leadership_alpha as csl


def test_policy_grid_is_fixed_unique_directional_and_bounded() -> None:
    policies = csl.policy_grid()
    assert len(policies) == 16
    assert policies == sorted(policies)
    assert len(set(policies)) == 16
    assert [policy.policy_id for policy in policies] == [f"P{i:02d}" for i in range(1, 17)]
    assert {policy.side for policy in policies} == {-1, 1}
    assert {policy.hold_bars for policy in policies} == {1, 3}
    assert {policy.family for policy in policies} == {
        "relative_return_lead",
        "premium_shock",
        "activity_confirmed_relative",
        "activity_premium_confluence",
        "return_premium_confluence",
    }


def test_manifest_keeps_2023_and_future_sealed() -> None:
    manifest = csl.build_manifest()
    csl.validate_manifest(manifest)
    protocol = manifest["selection_protocol"]
    assert manifest["outcomes_opened"] is False
    assert protocol["selection"][1] == "2023-01-01"
    assert protocol["sealed_holdout"] == ["2023-01-01", "2024-01-01"]
    assert protocol["future_2024_plus_sealed"] is True
    assert protocol["multiple_testing_hypotheses"] == 16


def test_contract_limits_claim_and_currency_contamination() -> None:
    manifest = csl.build_manifest()
    assert "true price discovery" in manifest["claim_boundary"]["not_claimed"]
    currency = manifest["currency_contract"]
    assert currency["raw_cross_venue_ratio_is_not_pure_coinbase_alpha"] is True
    assert "USD/USDT" in currency["interpretation"]
    assert "quote_asset_volume" in manifest["source_contract"]["binance_signal_leg"]
    assert (
        "Coinbase BTC volume * Coinbase close"
        == manifest["feature_contract"]["coinbase_quote_notional"]
    )


def test_support_is_frozen_before_forward_returns() -> None:
    support = csl.build_manifest()["support_freeze_before_returns"]
    assert support["paired_family_nonoverlap_events_min_total"] == 120
    assert support["paired_family_nonoverlap_events_min_each_year"] == 25
    assert support["minimum_each_side_share"] == 0.20
    assert support["global_missing_or_quarantined_fraction_max"] == 0.01
    assert support["failure_action"].endswith("without computing forward trade returns")


def test_execution_is_next_bar_with_correct_account_cost_and_strict_mdd() -> None:
    execution = csl.build_manifest()["execution_contract"]
    assert execution["entry_delay_bars"] == 1
    assert execution["base_cost_notional_per_side"] == 0.0006
    assert execution["base_cost_account_per_side_at_half_leverage"] == 0.0003
    assert execution["realized_funding"] is True
    assert "pre-entry high-water" in execution["strict_mdd"]
    assert "hypothetical liquidation" in execution["strict_mdd"]


def test_selection_requires_annual_halfyear_and_adjusted_significance() -> None:
    selection = csl.build_manifest()["selection_protocol"]
    gates = selection["selection_gates"]
    assert gates["every_calendar_year_absolute_return_positive"] is True
    assert gates["positive_half_years_min_of_six"] == 5
    assert gates["strict_mdd_pct_max_each_year"] == 10.0
    assert gates["familywise_weekly_cluster_signflip_p_max"] == 0.10
    assert selection["familywise_adjustment"].startswith("Bonferroni")
