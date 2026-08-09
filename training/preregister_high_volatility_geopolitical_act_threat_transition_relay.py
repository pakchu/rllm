"""Outcome-blind preregistration for HVGATA-24."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

POLICY_ID = "HVGATA-24"
DEFAULT_OUTPUT = Path("results/high_volatility_geopolitical_act_threat_transition_relay_preregistration_2026-08-09.json")
SOURCE = Path("data/global_daily_gpr_recent_1985_2026_aug.xls")
MARKET = Path("data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz")
MARKET_SHA = "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"
URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_geopolitical_act_threat_transition_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-09",
        "singleton": True,
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "mechanism": {
            "claim": "An extreme daily transition between geopolitical threats and realized geopolitical acts during elevated BTC variation identifies a change in demand for Bitcoin's conditional geopolitical hedge. Follow the completed act-minus-threat transition for one day.",
            "side": "strict sign of the completed daily change in log((GPR acts + 1)/(GPR threats + 1))",
            "why_distinct": "No repository alpha uses the official Caldara-Iacoviello daily geopolitical acts-versus-threats decomposition; EPU, OFAC, FX, Treasury, Cboe, crypto derivatives, and on-chain candidates use different economic objects.",
            "why_suited_to_volatile_regimes": "published daily evidence links extreme geopolitical-risk states and jumps to Bitcoin returns and volatility; the candidate additionally requires causal high BTC variation",
            "why_low_gross9_overlap_is_plausible": "a conservatively delayed official newspaper geopolitical transition clock is absent from Gross9",
        },
        "features": {
            "source": "official Caldara-Iacoviello Recent daily Geopolitical Risk XLS",
            "source_url": URL,
            "observation": "finite nonnegative daily GPR acts and GPR threats components for calendar day D",
            "availability": "D+2 00:00 UTC, deliberately later than the next-day newspaper index observation; no same-day or D+1 use",
            "valid_change": "D and D-1 exact consecutive calendar observations with both components present",
            "transition": "log((acts[D]+1)/(threats[D]+1)) - log((acts[D-1]+1)/(threats[D-1]+1)), strict nonzero",
            "transition_rank": "strict-prior midrank of abs(transition) over at most 270 valid changes, minimum 180, current excluded; rank>=0.75",
            "btc_variation": "sqrt(sum squared exact completed 5m BTC log returns over UTC day D+1, ending at decision)",
            "btc_variation_rank": "strict-prior midrank over at most 270 source-valid decisions, minimum 180, current excluded; rank>=0.65",
            "no_imputation": True,
        },
        "clock": {
            "decision": "D+2 00:00 UTC",
            "entry": "decision+5m BTCUSDT open",
            "hold": "24 elapsed hours",
            "side": "geopolitical act-minus-threat transition sign",
            "reservation": "daily nonoverlapping; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not signal input; exact settlements after novelty",
        },
        "policy": {
            "transition_history_days": 270,
            "minimum_history_days": 180,
            "transition_rank_min": 0.75,
            "btc_variation_rank_min": 0.65,
            "publication_delay_days": 2,
            "entry_delay_minutes": 5,
            "hold_hours": 24,
            "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        "stages": {
            "train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
        },
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.2, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.1, "candidate_near_6h_share_max": 0.35, "occupied_5m_bar_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.1, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "accounting": "fixed quantity, exact funding, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "post_stage_volatility_audit": {"prerequisite": "unchanged candidate passes all stages", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "diagnostic_controls": {"definitions": {"no_transition_tail": "transition direction and high BTC variation without transition rank", "no_btc_variation_gate": "transition tail without BTC variation rank", "one_day_stale_transition": "prior exact transition with current BTC variation gate", "direction_flip": "negative primary side", "same_clock_forced_long": "side +1 on primary clock"}, "cannot_be_promoted": True},
        "source_plan": {"gpr": {"url": URL, "destination": str(SOURCE), "download_after_preregistration_commit": True, "read_only_snapshot": True}, "historical_market": {"path": str(MARKET), "sha256": MARKET_SHA}, "live_extension": "read-only Postgres BTCUSDT 1m through 2026-08-01", "execution_prices": "sealed until source and novelty pass"},
        "research_boundary": {"paper": "Effects of the geopolitical risks on Bitcoin returns and volatility, Research in International Business and Finance 47 (2019) 511-518", "paper_url": "https://doi.org/10.1016/j.ribaf.2018.09.011", "source_methodology_url": "https://www.matteoiacoviello.com/gpr.htm", "prior_macro_outcomes_known": True, "prior_event_sets_or_controls_reused": False, "exact_hvgata_incidence_or_outcomes_known": False, "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False, "selection_basis": "independent official daily geopolitical acts-versus-threats transition mechanism"},
        "stopping_rule": "terminal first failure; no source, delay, component, rank, variation, side, clock, hold, subset, threshold, comparator, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    if value.get("manifest_hash") != canonical_hash({k: v for k, v in value.items() if k != "manifest_hash"}):
        raise RuntimeError("HVGATA prereg drift")
    if SOURCE.exists():
        raise RuntimeError("HVGATA source must not exist before preregistration freeze")
    if hashlib.sha256(MARKET.read_bytes()).hexdigest() != MARKET_SHA:
        raise RuntimeError("HVGATA market drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
