"""Outcome-blind preregistration for HVGCDR-24."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_high_volatility_energy_technology_spillover_relay as template


DEFAULT_OUTPUT = Path("results/high_volatility_global_correlation_demand_relay_preregistration_2026-08-12.json")
EQUITIES = ("SPY", "EFA", "EEM")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    core = copy.deepcopy(template.build()); core.pop("manifest_hash")
    core.update(
        protocol_version="high_volatility_global_correlation_demand_relay_v1",
        policy_id="HVGCDR-24",
        as_of_date="2026-08-12",
        mechanism={
            "claim": (
                "Peer-reviewed out-of-sample evidence identifies change in Bitcoin-stock correlation as the only "
                "consistently meaningful public predictor among common candidates: rising correlation suppresses "
                "next-day Bitcoin diversification demand and falling correlation boosts it. The paper further "
                "reports improvement from aggregating global information. Follow the negative sign of an equal-weight "
                "SPY/EFA/EEM correlation change only in extreme-change and high-BTC-variation states."
            ),
            "side": "negative strict sign of the equal-weight global correlation change",
            "external_support": {
                "paper": "Yae et al. (2022), Out-of-sample forecasting of cryptocurrency returns: A comprehensive comparison of predictors and algorithms, Physica A 598, 127379",
                "doi": "10.1016/j.physa.2022.127379",
                "reported_fact": (
                    "An increase in stock correlation suppresses next-day Bitcoin diversification demand and return; "
                    "a decrease boosts them. Correlation change is the only consistently meaningful tested predictor, "
                    "and global-information aggregation improves its OOS performance."
                ),
                "inference_disclosure": (
                    "SPY/EFA/EEM, a nonparametric 20-session Pearson window, equal weighting, strict-prior magnitude "
                    "rank, high-variation gate, and 24-hour execution are preregistered adaptations."
                ),
            },
            "why_distinct": (
                "BSCBR-24 used one SPY series and a fitted bivariate DCC-GARCH with fixed 0.02 changes. HVGCDR uses "
                "three regional equity portfolios, no fitted model, and the change in their equal-weight trailing "
                "Pearson correlations. Exact scans found no EFA/EEM or global-correlation BTC clock."
            ),
            "why_suited_to_volatile_regimes": "Only upper-35% completed BTC variation and upper-20% absolute global correlation changes are admitted.",
            "why_low_gross9_overlap_is_plausible": "Sparse multi-region US-close diversification-demand shocks are absent from Gross9 primitives.",
        },
        features={
            "sessions": "official common US cash sessions on which SPY, EFA and EEM all have complete adjusted OHLCV rows",
            "equity_returns": "for each ETF, log split/dividend-adjusted close_t / adjusted close_(t-1 common session)",
            "btc_returns": "log BTC close at current official US cash close / BTC close at previous common-session close",
            "correlations": "three Pearson correlations of BTC returns with each ETF over the 20 completed common-session return pairs ending at t; finite nonzero variances required",
            "global_correlation": "equal arithmetic mean of the three trailing correlations",
            "global_correlation_change": "global_correlation_t-global_correlation_(previous common session), strict nonzero",
            "magnitude_rank": "strict-prior midrank of absolute global_correlation_change versus at most 270 previous valid sessions; minimum 180; current excluded; rank>=0.80",
            "btc_variation": "sqrt(sum squared BTCUSDT 1m log(close/open)) over exact [decision-24h,decision)",
            "btc_variation_rank": "strict-prior 270/180 midrank over valid sessions; current excluded; rank>=0.65",
            "missing": "missing common session, adjustment, BTC minute, zero variance, nonfinite or duplicate data rejects; no imputation",
        },
        clock={
            "decision": "16:05 America/New_York after the official common-session cash close and all matched BTC data",
            "entry": "exact BTCUSDT five-minute open 5 minutes after decision",
            "hold": "24 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium": "not signal inputs; exact funding only after novelty passes",
            "no_imputation": True,
        },
        policy={
            "equity_symbols": list(EQUITIES), "correlation_sessions": 20,
            "magnitude_prior_sessions": 270, "magnitude_prior_minimum": 180, "magnitude_midrank_min": 0.80,
            "variation_prior_sessions": 270, "variation_prior_minimum": 180, "variation_midrank_min": 0.65,
            "feature_delay_minutes": 5, "entry_delay_minutes": 5, "hold_hours": 24,
            "gross_exposure": 0.5, "base_cost_per_notional_side": 0.0006, "stress_cost_per_notional_side": 0.001,
        },
        source_plan={
            "equities": {
                "provider": "Yahoo chart API current adjusted history", "symbols": list(EQUITIES),
                "interval": "1d", "window": ["2022-01-01", "2026-08-01"],
                "official_session_validation": "NYSE schedules and frozen early closes", "read_after_preregistration": True,
            },
            "btc_1m": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "close"], "read_only": True},
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        diagnostic_controls={
            "names": ["no_btc_volatility_gate", "global_correlation_direction_flip", "one_session_stale_change", "spy_only_correlation_change", "correlation_level", "same_clock_forced_long"],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        research_boundary={
            "paper_definition_opened": True,
            "prior_single_spy_bscbr_support_incidence_known": True,
            "prior_single_spy_formula_event_set_or_control_reused": False,
            "prior_bscbr_failure_used_to_set_global_formula_or_threshold": False,
            "exact_global_candidate_source_incidence_opened": False,
            "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False,
            "repository_global_correlation_candidate_found": False, "candidate_count": 1, "grid": False,
            "repair_of_prior_candidate": False, "promoted_prior_control": False,
            "selection_basis": "paper's separate global-information aggregation result, causal close availability, volatile-state targeting, and exact EFA/EEM repository absence",
        },
        stopping_rule="terminal first-failure sequence: source support, Gross9 novelty, train/test/eval/final strict economics, then RV20 q90; no symbols, window, weighting, rank, threshold, side, hold, clock, subset, source, or control repair",
    )
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {k: v for k, v in value.items() if k != "manifest_hash"}
    if value != build() or value.get("manifest_hash") != canonical_hash(core): raise RuntimeError("HVGCDR preregistration drift")
    if value["outcomes_opened"] is not False or value["source_incidence_opened"] is not False: raise RuntimeError("HVGCDR boundary drift")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); args=parser.parse_args()
    result=build(); validate(result); args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n"); print(args.output)
