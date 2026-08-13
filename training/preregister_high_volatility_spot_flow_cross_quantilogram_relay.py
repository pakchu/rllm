"""Outcome-blind preregistration for HVSFCQ-8."""
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

from training import preregister_high_volatility_bds_nonlinear_dependence_relay as template


DEFAULT_OUTPUT = Path(
    "results/high_volatility_spot_flow_cross_quantilogram_relay_preregistration_2026-08-13.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = copy.deepcopy(template.build())
    core.pop("manifest_hash")
    core.update(
        protocol_version="high_volatility_spot_flow_cross_quantilogram_relay_v1",
        policy_id="HVSFCQ-8",
        as_of_date="2026-08-13",
        mechanism={
            "claim": (
                "A positive lag-one cross-quantilogram from extreme Binance spot aggressive-flow minutes to "
                "same-tail next-minute perpetual returns identifies nonlinear cash-to-leverage price discovery in "
                "the active direction. During elevated BTC variation, follow that direction when final-hour spot "
                "flow and both venue returns confirm it."
            ),
            "side": "strict sign of completed final-hour normalized spot aggressive quote flow",
            "external_basis": {
                "paper": (
                    "Han, Linton, Oka and Whang (2016), The cross-quantilogram: Measuring quantile dependence and "
                    "testing directional predictability between time series, Journal of Econometrics 193, 251-270"
                ),
                "doi": "10.1016/j.jeconom.2016.03.001",
                "used_fact": (
                    "the cross-quantilogram measures directional predictability between quantile-hit processes "
                    "without requiring finite higher moments or a linear conditional-mean model"
                ),
                "adaptation_disclosure": (
                    "the spot-flow predictor, perpetual-return response, within-block quartiles, causal ranks, "
                    "confirmation, clock and hold are preregistered BTC adaptations"
                ),
            },
            "why_distinct": (
                "Spot-flow leadership compares two-hour aggregate spot and perpetual imbalance magnitudes; flow "
                "disagreement uses one five-minute opposite-sign imbalance; spot-return leadership uses linear "
                "correlation or categorical sign transitions. HVSFCQ instead estimates separate lagged upper- and "
                "lower-tail hit correlations from minute spot flow to next-minute perpetual return and selects the "
                "tail only from current completed spot-flow direction. It reuses no event or control and uses no "
                "funding, OI, premium, fitted post-entry outcome, or future data."
            ),
            "why_suited_to_volatile_regimes": (
                "completed perpetual variation and active-direction tail predictability must occupy causal upper tails"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "02:05/10:05/18:05 UTC cash-to-leverage tail-predictability onsets are absent from Gross9 clocks"
            ),
        },
        features={
            "decision_grid": "exact 02:00/10:00/18:00 UTC boundaries D",
            "block": (
                "480 exact aligned unique coherent BTCUSDT one-minute rows from bars_binance_spot and "
                "bars_binance over [D-8h,D); no imputation"
            ),
            "spot_flow": (
                "(2*taker_buy_quote-quote_asset_volume)/quote_asset_volume each minute; denominator strict positive, "
                "finite and in [-1,1]"
            ),
            "perpetual_return": "log(close/open) each minute, finite",
            "predictor_response_pairs": (
                "479 pairs (spot_flow_t, perpetual_return_t+1), t=0..478; the final spot-flow minute is excluded"
            ),
            "within_block_quartiles": (
                "linear-interpolation q25 and q75 separately for the 479 predictor values and 479 response values; "
                "each q25 strictly below q75"
            ),
            "lower_hits": (
                "centered indicators 1(spot_flow_t<spot_q25)-0.25 and "
                "1(perpetual_return_t+1<return_q25)-0.25"
            ),
            "upper_hits": (
                "centered indicators 1(spot_flow_t>spot_q75)-0.25 and "
                "1(perpetual_return_t+1>return_q75)-0.25"
            ),
            "cross_quantilogram": (
                "sum(predictor_hit*response_hit)/sqrt(sum predictor_hit^2*sum response_hit^2), separately lower "
                "and upper; denominators strict positive"
            ),
            "final_hour_spot_flow": (
                "(2*sum taker_buy_quote-sum quote_asset_volume)/sum quote_asset_volume over [D-1h,D), strict nonzero"
            ),
            "active_score": "upper cross-quantilogram for positive final-hour flow, lower for negative final-hour flow",
            "active_score_rank": (
                "strict-prior midrank against the matching-tail score history over at most 270 source-valid "
                "decisions, minimum 180, current excluded; active score strict positive and rank>=0.75"
            ),
            "direction_confirmation": (
                "final-hour spot and perpetual log returns both have the strict sign of final-hour spot flow"
            ),
            "perpetual_variation": "sqrt(sum squared 480 perpetual minute returns), finite strict positive",
            "variation_rank": "strict-prior 270/180 midrank, current excluded; rank>=0.65",
            "onset": (
                "eligible now and immediately preceding scheduled jointly source-valid decision ineligible; missing "
                "or invalid prior cannot trigger"
            ),
            "no_imputation": True,
        },
        clock={
            "decision": "completed aligned eight-hour boundary D",
            "entry": "exact BTCUSDT perpetual D+5m open",
            "side": "strict sign of completed final-hour spot aggressive flow",
            "hold": "8 elapsed hours",
            "reservation": "global chronological half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium": "not signal inputs; exact funding only after novelty",
        },
        policy={
            "tail_probability": 0.25,
            "lag_minutes": 1,
            "prior_decisions": 270,
            "minimum_prior_decisions": 180,
            "active_score_rank_min": 0.75,
            "variation_rank_min": 0.65,
            "decision_hours_utc": [2, 10, 18],
            "entry_delay_minutes": 5,
            "hold_hours": 8,
            "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        source_plan={
            "spot": {
                "table": "bars_binance_spot", "symbol": "BTCUSDT", "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close", "quote_asset_volume", "taker_buy_quote"],
            },
            "perpetual": {
                "table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close"],
            },
            "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            "read_after_preregistration_commit": True,
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        diagnostic_controls={
            "names": [
                "no_cross_quantilogram_tail",
                "no_variation_gate",
                "same_minute_cross_quantilogram",
                "one_decision_stale_dependence",
                "direction_flip",
                "forced_long",
            ],
            "cannot_be_promoted": True,
        },
        research_boundary={
            "prior_spot_flow_leadership_disagreement_and_return_transmission_outcomes_known": True,
            "repository_exact_spot_flow_to_next_perpetual_return_cross_quantilogram_found": False,
            "prior_event_sets_or_controls_reused": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "reversal_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": (
                "independent published tail-directional-dependence primitive and complete causal source audit; no "
                "prior outcome selected its tails, lag, direction, rank, clock, confirmation, or hold"
            ),
        },
        stopping_rule=(
            "terminal first failure; no venue, flow definition, return, lag, tail probability, hit process, score, "
            "history, rank, variation, confirmation, onset, side, clock, hold, subset, source, or control repair"
        ),
    )
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVSFCQ preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
