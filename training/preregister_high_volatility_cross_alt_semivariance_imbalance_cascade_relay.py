"""Outcome-blind preregistration for HVCASI-6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVCASI-6"
SLUG = "high_volatility_cross_alt_semivariance_imbalance_cascade_relay"
ALTS = ("ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT")
DEFAULT_OUTPUT = Path(f"results/{SLUG}_preregistration_2026-08-13.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": f"{SLUG}_v1",
        "policy_id": POLICY_ID,
        "slug": SLUG,
        "as_of_date": "2026-08-13",
        "singleton": True,
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "mechanism": {
            "claim": "At three fixed daily decisions, simultaneous upper-tail signed semivariance imbalance across at least four liquid alt perpetuals identifies a broad directional volatility cascade rather than symmetric noise. During elevated BTC variation, BTC follows the imbalance-weighted common downside-versus-upside variance direction for six hours.",
            "side": "strict sign of the selected-alt absolute-semivariance-imbalance-weighted consensus",
            "why_distinct": "Variance-shock propagation ranks unsigned one-hour quadratic variation, cojump propagation thresholds individual five-minute returns, return-tail asymmetry uses a cross-sectional cubic endpoint statistic, and the BTC semivariance candidate is single-asset reversal. HVCASI applies signed upside-versus-downside realized-semivariance imbalance separately to each alt's completed six-hour path, selects own-history absolute-imbalance tails, and transfers only their broad directional consensus into BTC. It reuses no event set or control and uses no BTC return direction, volume, flow, funding, OI, premium, or fitted outcome.",
            "why_suited_to_volatile_regimes": "BTC completed twenty-four-hour realized variation must rank in its causal upper 35 percent against strictly prior scheduled decisions, while the alt filter distinguishes directional variance asymmetry from symmetric noise.",
            "why_low_gross9_overlap_is_plausible": "01:05, 09:05 and 17:05 UTC entries conditioned on cross-alt signed semivariance imbalance are absent from Gross9 primitive clocks.",
        },
        "features": {
            "bars": "exact five-minute OHLC aggregated from five consecutive bars_binance one-minute rows; all seven symbols complete, finite, positive, coherent; no imputation",
            "decision_times": "every calendar day at exact 01:00, 09:00 and 17:00 UTC",
            "five_minute_return": "log(close/open)",
            "alt_path": "72 completed five-minute returns ending at decision for each alt",
            "signed_semivariance_imbalance": "(sum(positive_return^2)-sum(negative_return^2))/sum(return^2), finite strict nonzero with positive denominator",
            "semivariance_imbalance_rank": "for each alt independently, strict-prior midrank of absolute signed semivariance imbalance over at most 120 prior jointly valid scheduled decisions, minimum 60, current excluded; selected when rank>=0.75",
            "score": "sum(abs(imbalance)*sign(imbalance) over selected alts)/sum(abs(imbalance) over selected alts)",
            "consensus_gate": "at least four selected alts, abs(score)>=0.60, and at least three selected imbalance signs equal score sign",
            "btc_realized_variation": "sqrt(sum squared exact BTC five-minute log(close/open) returns over [D-24h,D))",
            "variation_rank": "strict-prior midrank over 20 prior jointly source-valid scheduled decisions, minimum 15, current excluded; rank>=0.65",
            "no_imputation": True,
        },
        "clock": {
            "decision": "exact daily 01:00, 09:00 or 17:00 UTC",
            "entry": "exact BTCUSDT perpetual decision+5m open",
            "side": "score sign",
            "hold": "6 elapsed hours",
            "reservation": "global chronological half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not signal input; exact settlements only after novelty",
        },
        "policy": {
            "path_bars": 72,
            "semivariance_imbalance_prior_decisions": 120,
            "semivariance_imbalance_minimum_decisions": 60,
            "semivariance_imbalance_rank_min": 0.75,
            "minimum_selected_alts": 4,
            "score_absolute_min": 0.60,
            "minimum_agreeing_alts": 3,
            "variation_bars": 288,
            "variation_prior_decisions": 20,
            "variation_minimum_decisions": 15,
            "variation_rank_min": 0.65,
            "decision_hours_utc": [1, 9, 17],
            "decision_minute": 0,
            "entry_delay_minutes": 5,
            "hold_hours": 6,
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
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.1, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "accounting": "fixed quantity, exact funding, 6bp/10bp per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "post_stage_volatility_audit": {"prerequisite": "unchanged all-stage pass", "rv20_q90_entry_filter": False, "minimum_q90_trades": 8, "candidate_q90_absolute_return_positive": True, "identical_clock_forced_long_residual_positive": True},
        "diagnostic_controls": {"names": ["no_variation_gate", "no_semivariance_imbalance_tail", "three_alt_consensus", "equal_weighted_score", "one_decision_stale_consensus", "direction_flip", "forced_long"], "cannot_be_promoted": True},
        "source_plan": {"market": {"table": "bars_binance", "symbols": ["BTCUSDT", *ALTS], "interval": "1m", "columns": ["ts", "symbol", "open", "high", "low", "close"], "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"]}, "read_after_preregistration_commit": True, "execution_prices": "sealed until source and novelty pass"},
        "research_boundary": {"prior_cross_alt_and_single_asset_semivariance_imbalance_outcomes_known": True, "repository_exact_cross_alt_semivariance_imbalance_consensus_event_found": False, "prior_event_sets_or_controls_reused": False, "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False, "selection_basis": "independent cross-alt signed semivariance imbalance spillover mechanism selected by outcome-blind primitive-gap audit"},
        "stopping_rule": "terminal first failure; no universe, path, statistic, history, rank, consensus, variation gate, direction, clock, hold, subset, comparator, or control repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVCASI prereg drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build()
    validate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if args.output.exists() and args.output.read_bytes() != encoded:
        raise RuntimeError(f"refusing overwrite {args.output}")
    args.output.write_bytes(encoded)
    print(args.output)
