"""Outcome-blind preregistration for HVCAOFTAR-8."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVCAOFTAR-8"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_cross_alt_order_flow_tail_asymmetry_relay_preregistration_2026-08-11.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    contract = copy.deepcopy(template.build())
    contract.pop("manifest_hash")
    contract.update(
        protocol_version="high_volatility_cross_alt_order_flow_tail_asymmetry_relay_v1",
        policy_id=POLICY_ID,
        mechanism={
            "claim": (
                "When the cross section of liquid-alt aggressive flow develops an unusually "
                "strong one-sided tail while overall flow intensity is elevated, speculative "
                "risk transfer is arriving outside BTC first. Follow that alt-flow tail in BTC "
                "for eight hours."
            ),
            "side": "strict sign of the median-centered cubic alt-order-flow tail mass",
            "why_distinct": (
                "Cross-alt flow leadership counts a sign majority and requires opposing BTC flow; "
                "flow centrality estimates dynamic leaders; flow geometry uses dispersion or "
                "diffusion. HVCAOFTAR uses the signed cubic tail asymmetry of six normalized alt "
                "flows and their cross-sectional RMS intensity, with no BTC flow or price, return, "
                "funding, OI, fitted outcome, prior event set, or promoted control."
            ),
            "why_suited_to_volatile_regimes": (
                "Both one-sided flow-tail asymmetry and cross-alt aggressive-flow intensity must "
                "enter causal upper tails; causal BTC RV20 q90 remains a post-stage audit."
            ),
            "why_low_gross9_overlap_is_plausible": (
                "data-driven hourly cross-alt flow-tail onsets are absent from Gross9 primitives"
            ),
        },
        features={
            "decision_grid": "every exact UTC hour D",
            "symbols": ["ADAUSDT", "BNBUSDT", "DOGEUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
            "source_window": (
                "60 exact bars_binance one-minute rows per symbol in [D-1h,D)"
            ),
            "normalized_alt_flow": (
                "sum(2*taker_buy_quote-quote_asset_volume)/sum(quote_asset_volume) independently "
                "per symbol; denominator strict positive and result finite"
            ),
            "cross_section_center": "median of the six normalized alt flows",
            "centered_flow": "normalized alt flow minus the cross-sectional median",
            "signed_tail_mass": "sum(centered_flow cubed)",
            "absolute_tail_mass": "sum(abs(centered_flow) cubed), finite and strict positive",
            "flow_tail_asymmetry": (
                "signed_tail_mass/absolute_tail_mass, finite and strict nonzero"
            ),
            "flow_asymmetry_rank": (
                "strict-prior midrank of abs(flow_tail_asymmetry) over at most 720 earlier valid "
                "hours, minimum 672, current excluded; rank>=0.90"
            ),
            "cross_alt_flow_intensity": "sqrt(mean(square(normalized_alt_flow)))",
            "flow_intensity_rank": (
                "strict-prior 720/672 midrank, current excluded; rank>=0.65"
            ),
            "eligible_state": "flow-asymmetry-rank and flow-intensity-rank gates pass",
            "onset": (
                "eligible now and immediately previous exact source-valid block ineligible; "
                "missing prior cannot trigger"
            ),
            "no_imputation": True,
        },
        clock={
            "decision": "completed hourly boundary",
            "entry": "exact BTCUSDT D+5m open",
            "hold": "8 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        policy={
            "history_blocks": 720,
            "minimum_history_blocks": 672,
            "flow_asymmetry_rank_min": 0.90,
            "flow_intensity_rank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 8,
            "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        diagnostic_controls={
            "names": [
                "no_flow_asymmetry_gate",
                "no_flow_intensity_gate",
                "mean_centered_cubic_mass",
                "one_hour_stale_flows",
                "direction_flip",
            ],
            "cannot_be_promoted": True,
        },
        source_plan={
            "bars": {
                "table": "bars_binance",
                "symbols": [
                    "ADAUSDT",
                    "BNBUSDT",
                    "DOGEUSDT",
                    "ETHUSDT",
                    "SOLUSDT",
                    "XRPUSDT",
                ],
                "interval": "1m",
                "columns": ["ts", "symbol", "quote_asset_volume", "taker_buy_quote"],
                "window": ["2023-05-01T00:00:00Z", "2026-08-01T00:00:00Z"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        research_boundary={
            "prior_cross_alt_flow_outcomes_known": True,
            "repository_cross_alt_order_flow_tail_asymmetry_candidate_found": False,
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
            "Terminal first failure; no symbol basket, flow formula, center, tail power, rank, "
            "side, hold, clock, subset, threshold, or control repair."
        ),
    )
    return {**contract, "manifest_hash": canonical_hash(contract)}


def validate(contract: dict[str, Any]) -> None:
    core = {key: value for key, value in contract.items() if key != "manifest_hash"}
    if contract.get("manifest_hash") != canonical_hash(core) or contract != build():
        raise RuntimeError("HVCAOFTAR preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build()
    validate(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(args.output)
