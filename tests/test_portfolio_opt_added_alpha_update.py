import numpy as np

from training.portfolio_opt_added_alpha_update import (
    Config,
    LIVE_WEIGHTS,
    NEW_SLEEVES,
    SLEEVES,
    exact_pre2025_rows,
    pre2025_passes,
    pre2025_selection_key,
    quantize_weights,
    strict_metric,
    valid_weights,
    years_for,
)


def metric(return_pct, cagr, mdd, ratio, trades):
    return {
        "absolute_return_pct": return_pct,
        "cagr_pct": cagr,
        "strict_mdd_pct": mdd,
        "cagr_to_strict_mdd": ratio,
        "trades": trades,
    }


def test_pre2025_selection_key_cannot_read_future_metrics():
    cfg = Config()
    stats = {
        "train": metric(100, 20, 10, 2, 200),
        "test2024": metric(30, 30, 6, 5, 100),
        "eval2025": metric(10, 10, 5, 2, 20),
        "ytd2026": metric(5, 12, 4, 3, 10),
    }
    before = pre2025_selection_key(stats, cfg)
    stats["eval2025"] = metric(-99, -99, 99, -1, 1)
    stats["ytd2026"] = metric(999, 999, 1, 999, 999)
    assert pre2025_selection_key(stats, cfg) == before


def test_pre2025_exact_constraints_are_explicitly_enforced():
    cfg = Config()
    stats = {
        "train": metric(100, 20, 10, 2, 200),
        "test2024": metric(30, 30, 6, 5, 100),
        "eval2025": metric(10, 10, 5, 2, 20),
        "ytd2026": metric(5, 12, 4, 3, 10),
    }
    assert pre2025_passes(stats, cfg)
    stats["test2024"] = metric(30, 30, 20.01, 5, 100)
    assert not pre2025_passes(stats, cfg)


def test_weight_contract_enforces_grid_min_gross_and_family_cap():
    cfg = Config()
    assert valid_weights(LIVE_WEIGHTS, cfg)
    assert quantize_weights({"frozen_annual_rank7": 0.274}, cfg) == {
        "frozen_annual_rank7": 0.25
    }
    assert not valid_weights({"frozen_annual_rank7": 0.2}, cfg)
    assert not valid_weights({"frozen_annual_rank7": 10.05}, cfg)
    assert not valid_weights(
        {
            "new_long_minimal_funding_premium": 1.5,
            "markov_transition_long": 0.55,
        },
        cfg,
    )


def test_candidate_universe_contains_added_alpha_sleeves():
    assert set(NEW_SLEEVES).issubset(SLEEVES)
    assert set(LIVE_WEIGHTS).issubset(SLEEVES)


def test_ytd_cagr_clock_counts_full_authoritative_calendar():
    expected = (
        np.datetime64("2026-06-03") - np.datetime64("2026-01-01")
    ) / np.timedelta64(1, "D") / 365.25
    assert np.isclose(years_for("ytd2026"), expected)


def test_strict_metric_uses_same_bar_upper_before_adverse_lower():
    returns = np.zeros((len(SLEEVES), 2))
    adverse = np.zeros_like(returns)
    index = SLEEVES.index("frozen_annual_rank7")
    returns[index] = [0.10, 0.0]
    adverse[index] = [-0.10, 0.0]
    data = {
        "R": returns,
        "A": adverse,
        "U": np.maximum(returns, 0.0),
        "counts": np.eye(1, len(SLEEVES), index, dtype=int).ravel(),
        "wins": np.eye(1, len(SLEEVES), index, dtype=int).ravel(),
    }
    result = strict_metric(data, 1.0, {"frozen_annual_rank7": 1.0})
    # Peak 1.10 is carried before the same bar's adverse 0.90 envelope.
    assert np.isclose(result["strict_mdd_pct"], (1.0 - 0.9 / 1.1) * 100.0)


def test_strict_metric_retains_intrabar_favorable_peak():
    returns = np.zeros((len(SLEEVES), 1))
    adverse = np.zeros_like(returns)
    favorable = np.zeros_like(returns)
    index = SLEEVES.index("fresh_kimchi_fx")
    adverse[index, 0] = -0.10
    favorable[index, 0] = 0.20
    data = {
        "R": returns,
        "A": adverse,
        "U": favorable,
        "counts": np.zeros(len(SLEEVES), dtype=int),
        "wins": np.zeros(len(SLEEVES), dtype=int),
    }
    result = strict_metric(data, 1.0, {"fresh_kimchi_fx": 1.0})
    assert np.isclose(result["strict_mdd_pct"], 25.0)


def test_exact_pre2025_ranks_every_generated_candidate_on_bar_clock():
    bars = 24
    returns = np.zeros((len(SLEEVES), bars))
    adverse = np.zeros_like(returns)
    favorable = np.zeros_like(returns)
    first = SLEEVES.index("frozen_annual_rank7")
    second = SLEEVES.index("fresh_kimchi_fx")
    returns[first] = 0.001
    returns[second] = 0.0005
    favorable[:] = np.maximum(returns, 0.0)
    data = {
        "R": returns,
        "A": adverse,
        "U": favorable,
        "counts": np.full(len(SLEEVES), 100, dtype=int),
        "wins": np.zeros(len(SLEEVES), dtype=int),
    }
    candidates = [
        {"fresh_kimchi_fx": 0.25},
        {"frozen_annual_rank7": 0.25},
    ]
    rows = exact_pre2025_rows({"train": data, "test2024": data}, candidates, Config())
    assert len(rows) == 2
    assert rows[0]["weights"] == {"frozen_annual_rank7": 0.25}
