"""Outcome-blind preregistration for HVQHLF-4."""
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


POLICY_ID = "HVQHLF-4"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_quarter_hour_lagged_flow_relay_preregistration_2026-08-13.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    contract = copy.deepcopy(template.build())
    contract.pop("manifest_hash")
    contract.update(
        protocol_version="high_volatility_quarter_hour_lagged_flow_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-13",
        mechanism={
            "claim": (
                "Periodic execution leaves a causally forecastable component in BTC quarter-hour "
                "opening order flow. An unusually strong forecast formed only from the preceding "
                "twelve quarter-hour opening imbalances should relay in its direction for four hours "
                "when completed BTC variation is elevated."
            ),
            "side": "strict sign of the causal twelve-lag predicted opening-imbalance component",
            "why_distinct": (
                "QHOIR-8 and HVQHOFR use the current completed opening imbalance, while HVCAQF and "
                "HVDQOFS use current cross-alt or daily opening flow. HVQHLF never reads current "
                "boundary flow: it trades only a rolling prior-data OLS projection from twelve "
                "already completed quarter-hour imbalances, requires an extreme predicted-component "
                "onset, and holds four hours. No prior event set or diagnostic control is reused."
            ),
            "why_suited_to_volatile_regimes": (
                "completed twenty-four-hour BTC variation must occupy its causal upper thirty-five percent"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "rare onset entries from a phase-specific predicted-flow tail are absent from Gross9 primitives"
            ),
        },
        external_basis={
            "paper": "Kim and Hansen (2026), The Quarter-Hour Effect: Periodic Algorithmic Trading and Return Predictability in Cryptocurrency Futures",
            "arxiv": "https://arxiv.org/abs/2607.09426",
            "reported_facts": [
                "the first-stage lagged-flow component uses twelve own quarter-hour opening-imbalance lags",
                "the lagged-flow component has a positive 4-hour coefficient and the largest mean 4-hour contribution share",
                "the paper reports a 5.0bp cross-asset mean interquartile effect at four hours",
            ],
            "untested_adaptations": [
                "one-minute candle flow replaces the paper's first-ten-second aggregate-trade imbalance",
                "rolling strictly-prior OLS replaces the paper's full-sample decomposition",
                "the high-variation and predicted-component tail onset gates are research-policy sparsifiers",
            ],
        },
        features={
            "decision_grid": "every exact UTC quarter-hour boundary T",
            "opening_imbalance": "for each prior boundary Q, exact BTCUSDT one-minute bar [Q,Q+1m): (2*taker_buy_base-volume)/volume; positive finite volume and coherent taker flow required",
            "lag_vector": "the twelve exact prior quarter-hour imbalances at T-15m,...,T-180m; all must be source-valid",
            "ols_samples": "for each earlier quarter-hour s, response imbalance[s] and its twelve exact prior quarter-hour imbalances; rows are admitted only when all thirteen values are source-valid",
            "causal_rolling_ols": "intercept plus twelve lag coefficients fitted by ordinary least squares on at most 8,640 admitted samples strictly before T, minimum 5,760; current T response is never used; full column rank required",
            "predicted_lagged_flow": "sum of the twelve fitted lag coefficients times the current twelve-lag vector; intercept excluded exactly as the paper's lagged-flow component; finite strict nonzero",
            "flow_strength_rank": "strict-prior midrank of abs(predicted_lagged_flow) over at most 8,640 earlier model-valid quarter-hours, minimum 5,760, current excluded; rank>=0.95",
            "btc_variation": "sqrt(sum squared exact BTCUSDT one-minute open-to-close log returns)) over 1,440 completed bars [T-24h,T)",
            "variation_rank": "strict-prior midrank over at most 8,640 earlier source-valid quarter-hours, minimum 5,760, current excluded; rank>=0.65",
            "eligibility": "model-valid, strict predicted sign, flow-strength rank>=0.95, and variation rank>=0.65",
            "onset": "eligible now and the immediately preceding quarter-hour model-valid observation ineligible; insufficient history counts as ineligible",
            "no_imputation": True,
        },
        clock={
            "decision": "exact UTC quarter-hour T; all signal inputs end no later than T",
            "feature_available": "T because the newest input bar [T-15m,T-14m) is already complete",
            "entry": "exact BTCUSDT perpetual T+5m open",
            "side": "sign(predicted_lagged_flow)",
            "hold": "4 elapsed hours",
            "reservation": "global chronological half-open first-eligible reservation; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        policy={
            "quarter_hour_lags": 12,
            "ols_history_observations": 8640,
            "ols_minimum_observations": 5760,
            "rank_history_observations": 8640,
            "rank_minimum_observations": 5760,
            "flow_strength_rank_min": 0.95,
            "variation_minutes": 1440,
            "variation_rank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 4,
            "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        diagnostic_controls={
            "names": [
                "raw_current_opening_imbalance",
                "include_ols_intercept",
                "no_flow_strength_tail",
                "no_variation_gate",
                "shifted_phase_plus_2m",
                "one_quarter_stale_prediction",
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
                "columns": ["ts", "open", "close", "volume", "taker_buy_base"],
                "window": ["2023-04-01T00:00:00Z", "2026-08-01T00:00:00Z"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        research_boundary={
            "paper_read_before_candidate": True,
            "repository_exact_causal_lagged_flow_component_candidate_found": False,
            "adjacent_raw_quarter_hour_candidate_failures_known": True,
            "adjacent_failures_used_to_set_formula_side_hold_clock_or_threshold": False,
            "prior_event_sets_reused": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "published twelve-lag periodic-flow decomposition and its strongest reported four-hour contribution under the requested high-variation regime",
        },
        stopping_rule=(
            "Terminal first failure; no source window, imbalance definition, lag count, OLS history, "
            "intercept treatment, rank history, tail, variation, onset, side, hold, clock, subset, "
            "threshold, or control repair."
        ),
    )
    return {**contract, "manifest_hash": canonical_hash(contract)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value != build() or value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVQHLF preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build()
    validate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(args.output)
