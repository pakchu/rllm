from training.rex_horizon_sweep import _legacy_rank, _stability_rank


def _row(family, train_ratio, val_ratio, train_cagr=10.0, val_cagr=20.0, train_mdd=10.0, val_mdd=8.0, val_trades=60, p=0.1):
    return {
        'family': family,
        'train': {'sim': {'trade_entries': 200, 'cagr_pct': train_cagr, 'strict_mdd_pct': train_mdd, 'cagr_to_strict_mdd': train_ratio}},
        'val': {'sim': {'trade_entries': val_trades, 'cagr_pct': val_cagr, 'strict_mdd_pct': val_mdd, 'cagr_to_strict_mdd': val_ratio}, 'trade_stats': {'p_value_mean_ret_approx': p}},
    }


def test_stability_rank_penalizes_validation_spikes_more_than_legacy_rank():
    spike = _row('rex_multiscale_location_revert', train_ratio=0.1, val_ratio=5.0, val_trades=140)
    stable = _row('rex_htf_deep_pullback_resume', train_ratio=1.0, val_ratio=3.0, val_trades=45)

    assert _legacy_rank(spike) > _legacy_rank(stable)
    assert _stability_rank(stable) > _stability_rank(spike)
