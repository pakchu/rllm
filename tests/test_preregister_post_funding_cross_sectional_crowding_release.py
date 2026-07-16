from __future__ import annotations

from training import preregister_post_funding_cross_sectional_crowding_release as pfcr


def test_protocol_is_deterministic_and_singleton() -> None:
    first = pfcr.protocol()
    second = pfcr.protocol()
    assert first == second
    assert pfcr.canonical_hash(first) == pfcr.canonical_hash(second)
    assert first["selection_2023_2024"]["singleton_no_parameter_ranking"] is True
    assert first["feature_formula"]["spread_reference"]["current_event_excluded"] is True


def test_protocol_delays_entry_beyond_observed_settlement() -> None:
    frozen = pfcr.protocol()
    assert frozen["clock"]["feature_available_time"] == "settlement timestamp + 5 minutes"
    assert frozen["clock"]["entry_time"] == "settlement timestamp + 10 minutes"
    assert frozen["clock"]["same_timestamp_funding"].startswith(
        "current settlement is not earned"
    )


def test_protocol_has_no_btc_leg_and_requires_actual_orthogonality() -> None:
    frozen = pfcr.protocol()
    assert "BTCUSDT" not in frozen["universe"]["symbols"]
    assert frozen["universe"]["position"].endswith("no BTC leg")
    orthogonality = frozen["orthogonality_after_standalone_pass"]
    assert orthogonality["absolute_daily_pnl_pearson_at_most"] == 0.30
    assert orthogonality["synchronized_portfolio_marginal_improvement_required"] is True
