"""Outcome-blind preregistration for HVDCSATPCLR-8."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from training import preregister_high_volatility_directional_change_scarcity_ticket_participation_acceleration_relay as base


POLICY_ID = "HVDCSATPCLR-8"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_directional_change_ticket_participation_close_location_router_preregistration_2026-08-16.json"
)
canonical_hash = base.canonical_hash


def build():
    source = copy.deepcopy(base.build())
    source.pop("manifest_hash")
    source["protocol_version"] = "high_volatility_directional_change_ticket_participation_close_location_router_v1"
    source["policy_id"] = POLICY_ID
    source["mechanism"] = {
        "claim": (
            "Keep the frozen high-variation directional-change-scarcity and joint ticket-participation acceleration "
            "eligibility. Route with the confirmed eight-hour direction only when the final-hour range closes in that "
            "direction's outer quartile; otherwise route against it as terminal-auction rejection."
        ),
        "side": (
            "confirmed long follows when final-hour close location>=0.75 and fades otherwise; confirmed short follows "
            "when final-hour close location<=0.25 and fades otherwise"
        ),
        "why_distinct": (
            "A frozen intrinsic-time scarcity plus ticket participation state is combined with a final-hour OHLC auction "
            "acceptance/rejection router; no outcome-fitted magnitude or promoted diagnostic control is used."
        ),
        "why_suited_to_volatile_regimes": (
            "coherent volatile paths with rising size and participation are separated by whether the terminal auction "
            "accepts the directional extreme or rejects it with an opposing wick"
        ),
        "why_low_gross9_overlap_is_plausible": (
            "the underlying sparse HVDCSATPA clock passed Gross9 and close-location polarity changes exposure shape"
        ),
    }
    source["features"]["final_hour_close_location"] = (
        "(last close - minimum low)/(maximum high - minimum low) over exact final 60 one-minute bars; "
        "strict positive range required"
    )
    source["features"]["close_location_router"] = (
        "follow confirmed long at >=0.75 or confirmed short at <=0.25; otherwise fade; fixed quartiles"
    )
    source["clock"]["side"] = "final-hour close-location routed confirmed completed-return sign"
    source["policy"]["long_acceptance_close_location_min"] = 0.75
    source["policy"]["short_acceptance_close_location_max"] = 0.25
    source["diagnostic_controls"]["names"] = [
        "always_follow_confirmed_direction",
        "always_fade_confirmed_direction",
        "no_joint_ticket_participation_acceleration_gate",
        "no_directional_change_scarcity_tail",
        "no_variation_gate",
        "one_block_stale_geometry",
        "direction_flip",
        "forced_long",
    ]
    boundary = source["research_boundary"]
    boundary.pop("exact_hvdcs_joint_ticket_participation_acceleration_incidence_or_outcomes_known", None)
    boundary["prior_hvdcs_joint_ticket_participation_train_pass_test_fail_known"] = True
    boundary["exact_close_location_router_incidence_or_outcomes_known"] = False
    boundary["candidate_incidence_opened"] = False
    boundary["postentry_return_or_pnl_opened"] = False
    boundary["gross9_rows_opened"] = False
    boundary["repair_of_prior_candidate"] = False
    boundary["selection_basis"] = (
        "causal intrinsic-time coherence, ticket participation sponsorship, and terminal-auction acceptance polarity"
    )
    source["stopping_rule"] = (
        "Terminal first-failure sequence: source support, Gross9 novelty, strict economics; no formula, threshold, "
        "side, hold, clock, subset, or control repair."
    )
    return {**source, "manifest_hash": canonical_hash(source)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)
