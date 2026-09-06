from training import build_pposm_symbolic_router_data as symbolic


def test_predicates_use_frozen_comparison_directions():
    features = {
        "htf_1w_return_1": -2.0,
        "rex_576_range_width_pct": 3.0,
        "quote_vol_z_1d": 0.0,
        "premium_index_change": 4.0,
        "rex_576_range_pos": 5.0,
    }
    thresholds = {
        "htf_1w_return_1_q50": -1.0,
        "rex_576_range_width_pct_q50": 2.0,
        "quote_vol_z_1d_q20": -1.0,
        "premium_index_change_q67": 3.0,
        "rex_576_range_pos_q67": 4.0,
    }
    assert symbolic.predicates(features, thresholds) == {
        "week_low": True,
        "range_wide": True,
        "quote_dry": False,
        "premium_hot": True,
        "range_high": True,
    }


def test_symbolic_prompt_contains_no_numeric_outcome():
    prompt = symbolic.symbolic_prompt(
        {"week_low": True, "range_wide": False, "quote_dry": True, "premium_hot": False, "range_high": False}
    )
    assert "causal_predicates" in prompt and "priority" in prompt
    assert "return TP4" in prompt and "return SKIP" in prompt
    assert "net_return" not in prompt and "future" not in prompt.lower()
