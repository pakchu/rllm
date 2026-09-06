"""Outcome-blind preregistration for HVETCC-8."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_ticket_elasticity_sponsorship_relay as template

POLICY_ID = "HVETCC-8"
DEFAULT_OUTPUT = Path(
    "results/high_volatility_equal_turnover_clock_concordance_relay_preregistration_2026-08-13.json"
)


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build() -> dict[str, Any]:
    contract = copy.deepcopy(template.build())
    contract.pop("manifest_hash")
    contract.update(
        protocol_version="high_volatility_equal_turnover_clock_concordance_relay_v1",
        policy_id=POLICY_ID,
        as_of_date="2026-08-13",
        mechanism={
            "claim": "In a completed volatile eight-hour BTC auction, one common direction across four sequential equal-quote-turnover intrinsic-time segments means price discovery persisted through every participation quartile rather than being produced by one physical-time burst; follow that direction for eight elapsed hours.",
            "side": "common strict sign of all four equal-turnover-clock segment returns",
            "why_distinct": "HVETCC partitions one fixed BTC block by cumulative quote-turnover quartiles and signs the within-minute returns assigned to those endogenous participation-time segments. Dynamic volume-bucket toxicity uses irregular buckets and taker-flow imbalance; volume-weighted median migration estimates an accepted-value center; equal-variance-clock concordance partitions by squared returns; temporal persistence uses fixed elapsed-time segments. HVETCC uses no taker split, funding, OI, premium, fitted outcome, reused event set, or promoted control.",
            "why_suited_to_volatile_regimes": "completed eight-hour realized variation must occupy its causal upper thirty-five percent and directional persistence must survive all four participation-time quartiles",
            "why_low_gross9_overlap_is_plausible": "offset fixed UTC clocks filtered by endogenous equal-turnover segment unanimity are absent from Gross9 primitives",
        },
        external_basis={
            "origin": "volume-clock and intrinsic-time transformation",
            "fixed_definition": "use cumulative BTCUSDT perpetual one-minute quote turnover as the clock and split ordered indivisible minutes at 25%, 50%, and 75% of completed total turnover",
            "selection_use": "the equal-turnover time change only; no directional return, incidence, or outcome claim is imported",
        },
        features={
            "decision_grid": "exact 02:00, 10:00, and 18:00 UTC boundaries",
            "source_block": "480 exact coherent BTCUSDT perpetual one-minute rows over [D-8h,D), each with finite positive OHLC and finite strictly positive quote_asset_volume",
            "minute_return": "log(close/open) within each completed minute, finite",
            "quote_turnover": "quote_asset_volume for each completed minute; completed block total finite and strictly positive",
            "turnover_clock_assignment": "minute i belongs to segment min(3,floor(4*cumulative_quote_turnover_before_i/total_quote_turnover)); minutes are indivisible and assignment is fixed before adding minute i turnover",
            "segment_validity": "each of four ordered turnover-clock segments contains at least one minute",
            "segment_return": "sum of one-minute log(close/open) returns assigned to each turnover-clock segment, each finite and strictly nonzero",
            "concordance": "all four segment returns have one common strict sign",
            "variation": "sqrt(sum squared one-minute log(close/open) returns over [D-8h,D)), finite and strictly positive",
            "variation_rank": "strict-prior midrank over at most 270 earlier source-valid decisions, minimum 180, current excluded; rank>=0.65",
            "eligibility": "source-valid, four nonempty strict-nonzero segments, concordance, and variation-rank gate",
            "onset": "eligible now and the immediately previous source-valid decision was ineligible; absence of a rank after insufficient history counts as ineligible",
            "no_imputation": True,
        },
        clock={
            "feature_available": "decision boundary after all 480 source minutes complete",
            "entry": "exact BTCUSDT perpetual D+5m open",
            "side": "common segment-return sign",
            "hold": "8 elapsed hours",
            "reservation": "global half-open first-eligible reservation; exit first on equal open",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty passes",
        },
        policy={
            "bar_minutes": 1,
            "path_minutes": 480,
            "turnover_segments": 4,
            "decision_hours": [2, 10, 18],
            "variation_history_decisions": 270,
            "minimum_history_decisions": 180,
            "variation_rank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 8,
            "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        diagnostic_controls={
            "names": [
                "no_variation_gate",
                "three_of_four_concordance",
                "equal_physical_time",
                "one_decision_stale_geometry",
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
                "columns": ["ts", "open", "high", "low", "close", "quote_asset_volume"],
                "window": ["2022-12-31T18:00:00Z", "2026-08-01T00:00:00Z"],
                "read_after_preregistration": True,
            },
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        research_boundary={
            "volume_clock_definition_read": True,
            "repository_exact_equal_turnover_clock_concordance_candidate_found": False,
            "adjacent_volume_bucket_value_anchor_equal_variance_and_temporal_persistence_candidates_known": True,
            "adjacent_candidate_outcomes_used_to_set_formula_side_hold_clock_or_threshold": False,
            "prior_event_sets_reused": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "fixed four-segment quote-turnover intrinsic-clock concordance under the requested high-variation regime",
        },
        stopping_rule="Terminal first failure; no source block, turnover clock, segment count, concordance, variation rank, onset, side, hold, clock, subset, threshold, or control repair.",
    )
    return {**contract, "manifest_hash": canonical_hash(contract)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVETCC preregistration drift")


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
