from training.event_candidate_regime_family_selector import _history_family_score


def _fold_result(ratio, trades=40, cagr=10.0, mdd=5.0, p=0.2):
    return {
        'sim': {
            'trade_entries': trades,
            'cagr_pct': cagr,
            'strict_mdd_pct': mdd,
            'cagr_to_strict_mdd': ratio,
            'side_counts': {'LONG': trades, 'SHORT': 0},
        },
        'trade_stats': {'p_value_mean_ret_approx': p, 'mean_trade_ret_pct': 0.1},
    }


def test_history_family_score_penalizes_declining_location_reversion():
    folds = [{'name': 'old'}, {'name': 'recent'}]
    nearest = [(0.0, folds[0]), (0.0, folds[1])]
    results = {
        'old': {
            'rex_multiscale_location_revert': {'fold': _fold_result(5.0)},
            'rex_htf_pullback_reclaim': {'fold': _fold_result(2.0)},
        },
        'recent': {
            'rex_multiscale_location_revert': {'fold': _fold_result(1.0)},
            'rex_htf_pullback_reclaim': {'fold': _fold_result(1.8)},
        },
    }

    location_score, _ = _history_family_score('rex_multiscale_location_revert', nearest=nearest, family_fold_results=results, min_trades=20)
    reclaim_score, _ = _history_family_score('rex_htf_pullback_reclaim', nearest=nearest, family_fold_results=results, min_trades=20)

    assert reclaim_score > location_score
