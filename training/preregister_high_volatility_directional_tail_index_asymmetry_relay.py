"""Outcome-blind preregistration for HVDTIAR-6."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template


POLICY_ID = "HVDTIAR-6"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_directional_tail_index_asymmetry_relay_preregistration_2026-08-10.json"
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = copy.deepcopy(template.build())
    core.pop("manifest_hash")
    core.update(
        protocol_version="high_volatility_directional_tail_index_asymmetry_relay_v1",
        policy_id=POLICY_ID,
        mechanism={
            "claim": "In a volatile completed BTC block, the return direction with the heavier scale-free extreme tail identifies the active liquidity vacuum. Follow that direction for six hours.",
            "side": "strict sign of positive-return Hill index minus negative-return Hill index",
            "why_distinct": "Standardized tail breadth counts RMS exceedances; realized skew and semivariance use full-distribution cubic or squared mass; price-impact asymmetry divides returns by turnover. HVDTIAR estimates separate scale-free positive and negative return-tail decay indices and uses no volume, flow, funding, OI, cross-asset, prior event, fitted outcome, or control.",
            "why_suited_to_volatile_regimes": "completed realized variation must be in its causal upper 35% while directional tail-index asymmetry enters its upper 40%",
            "why_low_gross9_overlap_is_plausible": "twice-daily signed Hill-tail-asymmetry onsets are absent from Gross9 primitives",
        },
        features={
            "decision_grid": "exact 02:00 and 14:00 UTC boundaries",
            "block": "720 exact coherent bars_binance BTCUSDT one-minute rows [D-12h,D)",
            "minute_return": "log(close/open), finite; zero values excluded only from signed-tail samples",
            "signed_samples": "positive returns and absolute negative returns, each requiring at least 25 observations",
            "hill_index": "for each signed sample sorted descending, fixed k=24 mean log(x_j/x_25) for j=1..24; x_25 strict positive and index finite",
            "tail_index_contrast": "positive-return Hill index minus negative-return Hill index, finite strict nonzero",
            "tail_asymmetry_rank": "strict-prior midrank of abs(tail_index_contrast) over at most 180 earlier source-valid decisions, minimum 120, current excluded; rank>=0.60",
            "realized_variation": "sqrt(sum squared minute returns), finite strict positive",
            "variation_rank": "strict-prior 180/120 midrank, current excluded; rank>=0.65",
            "eligible_state": "tail-asymmetry and variation gates pass",
            "onset": "eligible now and immediately previous exact source-valid decision ineligible; missing prior cannot trigger",
            "no_imputation": True,
        },
        clock={
            "decision": "completed twelve-hour boundary",
            "entry": "exact BTCUSDT D+5m open",
            "hold": "6 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        policy={
            "history_blocks": 180,
            "minimum_history_blocks": 120,
            "tail_asymmetry_rank_min": 0.60,
            "variation_rank_min": 0.65,
            "hill_k": 24,
            "minimum_signed_observations": 25,
            "entry_delay_minutes": 5,
            "hold_hours": 6,
            "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        diagnostic_controls={
            "names": [
                "no_tail_asymmetry_rank",
                "no_variation_gate",
                "fixed_k_tail_mass_asymmetry",
                "one_boundary_stale_tail",
                "direction_flip",
                "same_clock_forced_long",
            ],
            "cannot_be_promoted": True,
        },
        source_plan={
            "bars": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close"],
                "window": ["2023-04-01T00:00:00Z", "2026-08-01T00:00:00Z"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        research_boundary={
            "prior_tail_breadth_moment_semivariance_and_impact_outcomes_known": True,
            "repository_directional_tail_index_candidate_found": False,
            "prior_event_sets_reused": False,
            "prior_candidate_outcomes_used_to_set_formula_rank_side_hold_or_clock": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "signed scale-free extreme-tail decay contrast",
        },
        stopping_rule="Terminal first failure; no k, sample split, tail formula, rank, side, hold, clock, subset, or control repair.",
    )
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVDTIAR preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    registration = build()
    validate(registration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(registration, indent=2, allow_nan=False) + "\n")
    print(args.output)
