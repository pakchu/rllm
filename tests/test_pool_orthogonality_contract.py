from __future__ import annotations

import json
from pathlib import Path


POOL_DIR = Path("research/pools")


def load_pool(name: str) -> dict:
    return json.loads((POOL_DIR / name).read_text())


def by_id(pool: dict, entry_id: str) -> dict:
    return next(entry for entry in pool["entries"] if entry["id"] == entry_id)


def test_alpha_pool_uses_actual_trade_orthogonality_and_records_fresh_kimchi():
    pool = load_pool("alpha_pool.json")
    contract = pool["protocol"]["trade_orthogonality"]
    assert contract["decision_unit"] == "Actual executable trades, not raw feature activation."
    assert contract["default_limits"]["minimum_nonzero_pnl_days"] == 10
    assert "cannot establish portfolio independence" in contract["feature_correlation_role"]

    candidate = by_id(
        pool, "funding_fx_bidirectional_kimchi_local_impulse_gate_20260712"
    )
    audit = candidate["trade_orthogonality"]
    assert audit["exact_entry_jaccard"] == 0.0
    assert audit["position_jaccard"] < contract["default_limits"]["position_jaccard_max"]
    assert abs(audit["daily_marked_pnl_pearson"]) < contract["default_limits"][
        "absolute_daily_pnl_pearson_max"
    ]


def test_orthogonal_but_unprofitable_book_clock_stays_beta_feature():
    pool = load_pool("feature_pool.json")
    candidate = by_id(pool, "cross_collateral_near_pressure_event_clock_20260716")

    assert candidate["feature_tier"] == "beta_feature"
    assert candidate["status"] == "weak"
    assert candidate["orthogonality"]["passes_declared_limits"] is True
    assert "not an alpha" in candidate["tier_rationale"]
    assert any("lost 25.0844%" in failure for failure in candidate["known_failures"])


def test_fixed_orthogonal_portfolio_is_shadow_candidate_not_live():
    pool = load_pool("portfolio_pool.json")
    candidate = by_id(pool, "rank7_75_fresh_kimchi_25_fixed_shadow_20260716")

    assert candidate["status"] == "candidate"
    assert candidate["weights"] == {
        "expanding_extratrees_frozen_annual_rank7": 0.75,
        "funding_fx_bidirectional_kimchi_local_impulse_gate_20260712": 0.25,
    }
    assert candidate["construction_recipe"]["evaluation_protocol"]["weights"] == (
        "single hash-pinned 75/25 cell"
    )
    future = candidate["stats"]["future_2025_2026h1"]
    assert future["cagr_mdd"] > 4.0
    assert future["marginal_vs_rank7"]["ratio_delta"] > 0.0
    assert future["marginal_vs_rank7"]["strict_mdd_delta_pct"] < 0.0
    assert candidate["contamination_risk"] == "high"
    result = json.loads(
        Path("results/rank7_fresh_kimchi_fixed_portfolio_2026-07-16.json").read_text()
    )
    assert candidate["portfolio_spec_hash"] == result["portfolio_spec_hash"]
