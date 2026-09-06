"""Outcome-blind preregistration for HVCARTAR-8."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVCARTAR-8"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_cross_alt_return_tail_asymmetry_relay_preregistration_2026-08-11.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    contract = copy.deepcopy(template.build())
    contract.pop("manifest_hash")
    contract.update(
        protocol_version="high_volatility_cross_alt_return_tail_asymmetry_relay_v1",
        policy_id=POLICY_ID,
        mechanism={
            "claim": (
                "During elevated BTC variation, a strongly asymmetric cross-sectional tail in "
                "completed liquid-alt returns identifies the direction in which speculative risk "
                "is being repriced first. Follow that alt-tail direction in BTC for eight hours."
            ),
            "side": "strict sign of the median-centered cubic alt-return tail mass",
            "why_distinct": (
                "Cross-alt breadth counts signs, dispersion discards direction, tail dependence "
                "measures joint BTC-alt extremes through time, and wick consensus uses intrabar "
                "rejection geometry. HVCARTAR uses the signed asymmetry of the six-alt completed-"
                "return cross section, no BTC directional return, flow, volume, funding, OI, "
                "fitted outcome, prior event set, or promoted control."
            ),
            "why_suited_to_volatile_regimes": (
                "BTC completed variation must be in its causal upper 35%, while the absolute "
                "alt-tail asymmetry must enter its causal upper quartile."
            ),
            "why_low_gross9_overlap_is_plausible": (
                "sparse offset eight-hour cross-alt tail-asymmetry onsets are absent from Gross9"
            ),
        },
        features={
            "decision_grid": "exact 04:00/12:00/20:00 UTC boundaries",
            "symbols": ["ADAUSDT", "BNBUSDT", "DOGEUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
            "source_window": (
                "480 exact coherent bars_binance one-minute rows per symbol in [D-8h,D)"
            ),
            "alt_return": "log(last completed close / first completed open) independently per symbol",
            "cross_section_center": "median of the six finite alt returns",
            "centered_return": "alt return minus the cross-sectional median",
            "signed_tail_mass": "sum(centered_return cubed)",
            "absolute_tail_mass": "sum(abs(centered_return) cubed), required finite and strict positive",
            "tail_asymmetry": (
                "signed_tail_mass / absolute_tail_mass, required finite and strict nonzero"
            ),
            "tail_asymmetry_rank": (
                "strict-prior midrank of abs(tail_asymmetry) over at most 270 earlier source-valid "
                "blocks, minimum 180, current excluded; rank>=0.75"
            ),
            "btc_realized_variation": (
                "sum squared BTCUSDT one-minute log(close/open) returns over the same completed "
                "block, finite strict positive"
            ),
            "variation_rank": (
                "strict-prior midrank over at most 270 earlier source-valid blocks, minimum 180, "
                "current excluded; rank>=0.65"
            ),
            "eligible_state": "tail-asymmetry-rank and variation-rank gates pass",
            "onset": (
                "eligible now and immediately previous exact source-valid block ineligible; "
                "missing prior cannot trigger"
            ),
            "no_imputation": True,
        },
        clock={
            "decision": "completed eight-hour boundary",
            "entry": "exact BTCUSDT D+5m open",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        policy={
            "history_blocks": 270,
            "minimum_history_blocks": 180,
            "tail_asymmetry_rank_min": 0.75,
            "variation_rank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 8,
            "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        diagnostic_controls={
            "names": [
                "no_tail_asymmetry_gate",
                "no_variation_gate",
                "mean_centered_cubic_mass",
                "one_boundary_stale_returns",
                "direction_flip",
            ],
            "cannot_be_promoted": True,
        },
        source_plan={
            "bars": {
                "table": "bars_binance",
                "symbols": [
                    "BTCUSDT",
                    "ADAUSDT",
                    "BNBUSDT",
                    "DOGEUSDT",
                    "ETHUSDT",
                    "SOLUSDT",
                    "XRPUSDT",
                ],
                "interval": "1m",
                "columns": ["ts", "symbol", "open", "high", "low", "close"],
                "window": ["2023-04-01T00:00:00Z", "2026-08-01T00:00:00Z"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        research_boundary={
            "prior_cross_alt_return_outcomes_known": True,
            "repository_cross_alt_return_tail_asymmetry_candidate_found": False,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_rank_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": (
                "research-supported cross-market order-flow propagation translated into a "
                "source-available, outcome-blind cross-sectional tail-leadership state"
            ),
        },
        stopping_rule=(
            "Terminal first failure; no symbol basket, return formula, center, tail power, rank, "
            "side, hold, clock, subset, threshold, or control repair."
        ),
    )
    return {**contract, "manifest_hash": canonical_hash(contract)}


def validate(contract: dict[str, Any]) -> None:
    core = {key: value for key, value in contract.items() if key != "manifest_hash"}
    if contract.get("manifest_hash") != canonical_hash(core) or contract != build():
        raise RuntimeError("HVCARTAR preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build()
    validate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(args.output)
