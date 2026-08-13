"""Outcome-blind preregistration for HVAMF-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ALTS = ("ADAUSDT", "BNBUSDT", "DOGEUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
DEFAULT_OUTPUT = Path(
    "results/high_volatility_alt_modularity_fragmentation_relay_preregistration_2026-08-13.json"
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
    core = {
        "protocol_version": "high_volatility_alt_modularity_fragmentation_relay_v1",
        "policy_id": "HVAMF-8",
        "as_of_date": "2026-08-13",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "When the full positive-correlation network of six liquid alt perpetuals newly separates into two "
                "unusually modular communities, crypto risk transmission is fragmented rather than governed by one "
                "market mode. During elevated BTC variation, follow the completed final-hour direction of the "
                "dynamically dominant community for eight elapsed hours."
            ),
            "side": "strict sign of the larger-absolute-median-return community in the completed final hour",
            "why_distinct": (
                "HVAMST discards all but five minimum-distance edges and tests unique-node degree centralization. "
                "HVCMMI measures one spectral market-mode share, while correlation-fracture candidates aggregate "
                "pairwise changes. HVAMF instead retains every positive correlation edge, exhaustively maximizes "
                "weighted two-community Newman-Girvan modularity, and obtains direction only after the partition "
                "from the dynamically dominant module. It reuses no prior event or control and uses no BTC "
                "direction, volume, flow, funding, OI, premium, fitted outcome, or post-entry data."
            ),
            "why_suited_to_volatile_regimes": (
                "completed BTC variation must occupy its causal upper 35 percent and exact network modularity its "
                "causal upper quartile"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "04:05/12:05/20:05 UTC full-network fragmentation onsets are absent from Gross9 clocks"
            ),
        },
        "features": {
            "decision_grid": "exact 04:00/12:00/20:00 UTC boundaries D",
            "aligned_window": (
                "96 exact epoch-aligned five-minute bars from 480 unique coherent bars_binance one-minute rows "
                "[D-8h,D) for BTC and six alts"
            ),
            "alt_return": (
                "log(five-minute close/open), 96 observations per alt; each sample variance strict positive"
            ),
            "correlation": "six-by-six float64 sample Pearson correlation matrix, finite with unit diagonal",
            "adjacency": "off-diagonal max(correlation,0), zero diagonal; total undirected edge weight strict positive",
            "partition_universe": (
                "all unordered bipartitions of six frozen-index nodes with both communities containing at least two "
                "nodes; the community containing frozen node index zero is emitted first"
            ),
            "modularity": (
                "for each partition, sum over communities of internal_edge_weight/total_edge_weight minus "
                "(community_weighted_degree/(2*total_edge_weight))^2"
            ),
            "unique_partition": (
                "one partition must have a strict float64 maximum modularity; exact maximum ties reject"
            ),
            "modularity_rank": (
                "strict-prior midrank over at most 270 source-valid decisions, minimum 180, current excluded; "
                "rank>=0.75"
            ),
            "community_final_hour_return": (
                "cross-sectional median, separately within each community, of member alts' sums of the final "
                "12 five-minute returns; both medians finite and strict nonzero"
            ),
            "dominant_community": (
                "community with strictly larger absolute final-hour median return; equal magnitudes reject"
            ),
            "btc_realized_variation": "sqrt(sum squared 96 BTC five-minute returns), finite strict positive",
            "variation_rank": "strict-prior 270/180 midrank, current excluded; rank>=0.65",
            "onset": (
                "eligible now and immediately preceding scheduled source-valid decision ineligible; missing or "
                "invalid prior cannot trigger"
            ),
            "no_imputation": True,
        },
        "clock": {
            "decision": "completed aligned eight-hour boundary D",
            "entry": "exact BTCUSDT perpetual D+5m open",
            "side": "sign of dynamically dominant community final-hour median return",
            "hold": "8 elapsed hours",
            "reservation": "global chronological half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium": "not signal inputs; exact funding only after novelty",
        },
        "policy": {
            "alt_symbols": list(ALTS),
            "minimum_community_size": 2,
            "prior_decisions": 270,
            "minimum_prior_decisions": 180,
            "modularity_rank_min": 0.75,
            "variation_rank_min": 0.65,
            "decision_hours_utc": [4, 12, 20],
            "entry_delay_minutes": 5,
            "hold_hours": 8,
            "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
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
            "future_can_rank_repair_or_reselect": False,
            "accounting": (
                "fixed quantity, exact funding, 6bp base and 10bp stress per notional side, every held 5m "
                "favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "post_stage_volatility_audit": {
            "prerequisite": "unchanged candidate passes train, test, eval and final",
            "rv20_q90_entry_filter": False,
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "identical_clock_forced_long_residual_positive": True,
        },
        "source_plan": {
            "market": {
                "table": "bars_binance",
                "symbols": ["BTCUSDT", *ALTS],
                "interval": "1m",
                "columns": ["ts", "symbol", "open", "high", "low", "close"],
                "window": ["2023-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            },
            "read_after_preregistration_commit": True,
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        "diagnostic_controls": {
            "names": [
                "no_modularity_tail",
                "no_variation_gate",
                "all_alt_final_hour_median",
                "one_decision_stale_partition",
                "direction_flip",
                "forced_long",
            ],
            "cannot_be_promoted": True,
        },
        "research_boundary": {
            "prior_mst_market_mode_correlation_fracture_and_cross_alt_outcomes_known": True,
            "repository_exact_full_positive_correlation_modularity_candidate_found": False,
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
                "independent full-network community-fragmentation primitive selected from formula-absence and "
                "complete-source audit only; no prior candidate outcome selected its partition, side, clock, or gate"
            ),
        },
        "stopping_rule": (
            "terminal first failure; no universe, aggregation, correlation, adjacency, partition universe, "
            "modularity, tie rule, history, rank, threshold, variation, onset, dominant community, direction, "
            "clock, hold, subset, source, or control repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVAMF preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
