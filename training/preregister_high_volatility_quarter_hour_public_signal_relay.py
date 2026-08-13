"""Outcome-blind preregistration for HVQHPS-12."""
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

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template


POLICY_ID = "HVQHPS-12"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_quarter_hour_public_signal_relay_preregistration_2026-08-13.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    contract = copy.deepcopy(template.build())
    contract.pop("manifest_hash")
    contract.update(
        protocol_version="high_volatility_quarter_hour_public_signal_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-13",
        mechanism={
            "claim": (
                "Observable pre-boundary price-volume state explains a distinct component of scheduled "
                "quarter-hour opening order flow. An unusually strong causal estimate of that public-signal "
                "component should relay in its direction for twelve hours during elevated BTC variation."
            ),
            "side": "strict sign of the causal TI28 public-signal opening-imbalance component",
            "why_distinct": (
                "HVQHLF-4 uses only twelve prior opening-flow lags and is terminal. QHOIR/HVQHOFR use "
                "current opening flow. Standalone technical-indicator candidates trade one indicator's "
                "canonical crossing on four-hour bars. HVQHPS jointly uses the paper's fixed 28-dimensional "
                "15-minute price-volume state only as one block in a causal opening-imbalance decomposition; "
                "the current boundary flow, future return, prior event sets, and controls are never signal inputs."
            ),
            "why_suited_to_volatile_regimes": (
                "completed twenty-four-hour BTC variation must occupy its causal upper thirty-five percent"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "rare onset entries from a phase-specific fitted public-flow tail are absent from Gross9 primitives"
            ),
        },
        external_basis={
            "paper": "Kim and Hansen (2026), The Quarter-Hour Effect: Periodic Algorithmic Trading and Return Predictability in Cryptocurrency Futures",
            "arxiv": "https://arxiv.org/abs/2607.09426",
            "reported_facts": [
                "the first stage projects quarter-hour opening imbalance on twelve own lags and a fixed 28-indicator price-volume block",
                "the public-signal component has a positive 12-hour coefficient with block-bootstrap t=2.5",
                "its cross-asset mean 12-hour interquartile effect is 16.9bp and contribution share is 65 percent",
                "the public-signal share rises from four to twelve hours in all six contracts",
            ],
            "untested_adaptations": [
                "one-minute candle flow replaces first-ten-second aggregate-trade imbalance",
                "strictly-prior rolling OLS replaces the paper's full-sample decomposition",
                "high-variation and public-component tail onset gates are research-policy sparsifiers",
            ],
            "independent_selection_timing": "the public-signal 12-hour result was documented before HVQHLF-4 economic outcomes were opened",
        },
        features={
            "decision_grid": "every exact UTC quarter-hour boundary T",
            "opening_imbalance_response": "BTCUSDT one-minute [T,T+1m) normalized aggressive base flow; used only as responses of strictly earlier OLS samples, never the current signal",
            "lag_block": "twelve exact prior quarter-hour opening imbalances at T-15m,...,T-180m",
            "indicator_source": "exact 15-minute OHLCV bars aggregated from coherent one-minute rows; at T the newest admitted indicator bar is [T-30m,T-15m), so [T-15m,T) is excluded",
            "momentum_5": "Wilder RSI(24); stochastic %K(24) and SMA6 %D; stochastic RSI from Wilder RSI(24), rolling 24 range, then two sequential SMA6 smoothings; CCI(24) of typical price with 0.015 mean absolute deviation",
            "trend_9": "close/SMA-1 for windows 4,6,12,20,32,48,96; normalized EMA8(close)-EMA32(close) divided by close; that MACD minus its EMA6",
            "volume_10": "volume/SMA-1 for windows 4,6,12,16,24,32,48; normalized EMA8(volume)-EMA32(volume) divided by volume; that MACD minus its EMA6; (EMA4(ADL)-EMA32(ADL))/SMA32(volume), where ADL cumulatively adds ((2C-H-L)/(H-L))*volume and zero-range bars add zero",
            "volatility_4": "on 24 bars, lower=mean-2*population_std, middle=mean, upper=mean+2*population_std; (close-lower)/lower, (close-middle)/middle, (close-upper)/upper, and (upper-lower)/middle; all four denominators strict positive",
            "smoothing_conventions": "all SMA and population-dispersion windows require their full stated bars; recursive EMA uses alpha=2/(span+1), adjust=False, seeded by the first finite input; Wilder RSI uses alpha=1/24 after a 24-change arithmetic-mean seed; stochastic smoothers are full-window SMA6",
            "indicator_validity": "all 28 finite, all required denominators strict nonzero, uninterrupted exact source history, no imputation",
            "causal_rolling_ols": "intercept plus twelve lag-block columns plus all 28 indicators, fitted on at most 8,640 jointly valid quarter-hour samples strictly before T, minimum 5,760, current response excluded, full column rank required",
            "public_signal_component": "sum of the 28 fitted indicator coefficients times current TI28 values; intercept and twelve-lag fitted contribution excluded; finite strict nonzero",
            "public_strength_rank": "strict-prior midrank of abs(public_signal_component) over at most 8,640 earlier model-valid quarter-hours, minimum 5,760, current excluded; rank>=0.95",
            "btc_variation": "sqrt(sum squared exact BTCUSDT one-minute open-to-close log returns)) over [T-24h,T)",
            "variation_rank": "strict-prior midrank over at most 8,640 earlier source-valid quarter-hours, minimum 5,760, current excluded; rank>=0.65",
            "eligibility": "model-valid, strict public-component sign, public-strength rank>=0.95, and variation rank>=0.65",
            "onset": "eligible now and the immediately preceding model-valid quarter-hour ineligible; insufficient history counts as ineligible",
            "no_imputation": True,
        },
        clock={
            "decision": "exact UTC quarter-hour T",
            "feature_available": "T; newest TI input ended at T-15m and newest lagged opening flow ended by T-14m",
            "entry": "exact BTCUSDT perpetual T+5m open",
            "side": "sign(public_signal_component)",
            "hold": "12 elapsed hours",
            "reservation": "global chronological half-open first-eligible reservation; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        policy={
            "quarter_hour_lags": 12,
            "technical_indicator_count": 28,
            "ols_history_observations": 8640,
            "ols_minimum_observations": 5760,
            "rank_history_observations": 8640,
            "rank_minimum_observations": 5760,
            "public_strength_rank_min": 0.95,
            "variation_minutes": 1440,
            "variation_rank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 12,
            "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        diagnostic_controls={
            "names": [
                "lagged_flow_component",
                "include_ols_intercept",
                "no_public_strength_tail",
                "no_variation_gate",
                "latest_indicator_bar_included",
                "one_quarter_stale_public_component",
                "direction_flip",
                "same_clock_forced_long",
            ],
            "cannot_be_promoted": True,
        },
        source_plan={
            "btc": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close", "volume", "taker_buy_base"],
                "window": ["2023-04-01T00:00:00Z", "2026-08-01T00:00:00Z"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        research_boundary={
            "paper_and_exact_ti28_menu_read": True,
            "repository_exact_public_signal_decomposition_candidate_found": False,
            "adjacent_single_indicator_and_quarter_hour_candidates_known": True,
            "hvqhlf_terminal_outcome_known": True,
            "public_signal_formula_horizon_and_direction_selected_before_hvqhlf_outcome": True,
            "adjacent_outcomes_used_to_set_formula_side_hold_clock_or_threshold": False,
            "prior_event_sets_reused": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "the paper's independently reported TI28 public-flow component and strongest reported twelve-hour contribution under the requested high-variation regime",
        },
        stopping_rule=(
            "Terminal first failure; no indicator definition, information cutoff, lag block, OLS history, "
            "component allocation, rank history, tail, variation, onset, side, hold, clock, subset, "
            "threshold, or control repair."
        ),
    )
    return {**contract, "manifest_hash": canonical_hash(contract)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value != build() or value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVQHPS preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build()
    validate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
