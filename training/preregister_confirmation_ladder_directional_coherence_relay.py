"""Outcome-blind preregistration for CLDCR-6."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "CLDCR-6"
DEFAULT_OUTPUT = Path(
    "results/confirmation_ladder_directional_coherence_relay_preregistration_2026-08-09.json"
)


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "confirmation_ladder_directional_coherence_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": "A large BTC price impulse whose direction persists across at least five of six consecutive canonical confirmation intervals represents block-time price discovery rather than one wall-clock bar; follow the cumulative direction for six elapsed hours.",
            "side": "strict sign of the sum of six confirmation-interval log returns",
            "why_distinct": "CLDCR uses ordered return coherence on an external canonical block-time ladder. It is not fixed-clock sign persistence, a wall-clock breakout, range/variance ratio, volume, flow, spot/perp leadership, funding, premium, OI, block throughput, CLCS source substitution, or a prior control.",
            "volatile_market_target": "q90 cumulative block-time impulse plus five-of-six direction agreement directly selects coherent volatile discovery; causal RV20 q90 remains only a later audit",
            "why_low_gross9_overlap_is_plausible": "height-modulo anchors, six irregular confirmation intervals, q90 onset, and six-hour reservation form a sparse asynchronous clock",
        },
        "features": {
            "anchor": "canonical height H divisible by 36; ladder is blocks H through H+6",
            "ladder_intervals": "six strictly increasing header-timestamp intervals (H+k-1,H+k], each 60 to 1800 seconds",
            "completed_minutes": "for each interval use exact complete UTC perpetual minutes whose opens are >=ceil(left/60)*60 and whose closes are <=floor(right/60)*60",
            "interval_return": "R_k=log(last completed-minute close/first completed-minute open), finite and strict nonzero",
            "cumulative_return": "C=sum(R_k), finite and strict nonzero",
            "path_confirmation": "at least five of six R_k have strict sign sign(C)",
            "prior_ranks": "strict-prior midrank of abs(C) over at most 168 prior valid anchors, minimum 112, current excluded",
            "eligible_state": "abs(C) rank>=0.90 and path confirmation true",
            "onset": "current eligible state true and immediately preceding valid anchor state false; invalid anchors preserve prior valid state",
            "confirmation": "block H+6 is contained before signal availability",
            "raw_availability": "prefix maximum header timestamp and mediantime through H+6 plus 7200 seconds",
            "decision": "raw availability ceiled to the next exact five-minute UTC boundary D",
            "side": "strict sign of C",
            "source_valid": "contiguous canonical chain and exact coherent perpetual minute grid, no imputation",
        },
        "rv20_stress_slice": {
            "rv20": "sqrt(365*mean exact daily returns^2 over t-20 through t-1)",
            "threshold": "numpy linear q90 over 756 strictly prior available RV20 observations",
            "entry_filter": False,
            "future_use": "only after all sequential full-calendar stages pass",
        },
        "clock": {
            "entry": "exact BTCUSDT decision+5m open",
            "hold": "6 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not an input; exact realized funding only after novelty",
        },
        "stages": {
            "train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
        },
        "source_support_gates": {
            "minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8},
            "minority_side_share_min": 0.2,
            "max_month_share": 0.45,
        },
        "novelty_gates": {
            "exact_entry_jaccard_max": 0.1,
            "candidate_near_6h_share_max": 0.35,
            "occupied_5m_bar_jaccard_max": 0.25,
            "absolute_signed_exposure_pearson_max": 0.35,
            "must_pass_before_economics": True,
        },
        "economic_gates": {
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_min": 3.0,
            "strict_mdd_max_pct": 15.0,
            "mean_gross_underlying_min_bp": 20.0,
            "weekly_signflip_one_sided_p_max": 0.1,
            "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5,
            "each_calendar_half_positive": True,
            "stop_on_first_failure": True,
            "accounting": (
                "fixed quantity, exact funding, 6bp base and 10bp stress per notional "
                "side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "post_stage_volatility_audit": {
            "prerequisite": "unchanged candidate passes train, test, eval, final",
            "persistent_long_vol_comparator": "same accepted clock and 0.5 gross, side forced long",
            "full_calendar_decomposition": "candidate minus comparator net return",
            "rv20_q90_decomposition": "same decomposition on causal RV20 q90 decisions",
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "candidate_specific_q90_residual_positive": True,
            "comparator_cannot_satisfy_candidate_claim": True,
        },
        "diagnostic_controls": {
            "definitions": {
                "magnitude_only": "abs(C) rank>=0.90 onset without five-of-six coherence",
                "coherence_only": "five-of-six coherence onset without magnitude rank",
                "one_anchor_stale_ladder": "primary valid-anchor state and side shifted one anchor before onset",
                "direction_flip": "negative primary side",
            },
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "blocks": {
                "source": "mempool Esplora-compatible canonical block metadata",
                "columns": [
                    "height", "id", "previousblockhash", "timestamp", "mediantime",
                    "tx_count", "size", "weight",
                ],
                "read_after_preregistration": True,
            },
            "bars": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "high", "low", "close"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "research_boundary": {
            "prior_sign_persistence_and_block_families_known": True,
            "exact_cldcr_incidence_known": False,
            "exact_cldcr_outcomes_known": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
        },
        "stopping_rule": (
            "terminal first failure; no block predicate, threshold, confirmation, embargo, "
            "side, clock, hold, RV20, subset, comparator, control, or gate repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("CLDCR preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(args.output)
