"""Preregister CRRC-72 before opening any post-entry return.

CRRC-72 is an outcome-blind cross-venue order-book state.  It asks whether
both Binance USD-M and COIN-M books simultaneously refill the inner same-side
shell while outer same-side liquidity is withdrawn and the refill remains
credible rather than flickering.  This module freezes the single selected
support cell; it never loads a price, funding, return, or equity series.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "results/cross_venue_radial_refill_compression_preregistration_2026-07-17.json"
)
DEFAULT_DOCS = Path(
    "docs/cross-venue-radial-refill-compression-preregistration-2026-07-17.md"
)

SHELL_SOURCE_SHA256 = (
    "ead931ec8ce2bbd73c946b8660e16d7750ce73051e60ce4989467a7c5bc68342"
)
CREDIBILITY_SOURCE_SHA256 = (
    "45026cc02620d9a0c67f250804f2a06705bf0e824f72257d6c2414f40ab7d429"
)

# The incidence counts below were inspected without loading market prices or
# post-entry outcomes.  They are retained so the support-cell choice is fully
# auditable and cannot be silently rewritten after an outcome is opened.
SUPPORT_GRID: tuple[dict[str, float | int], ...] = (
    {"q_add": 0.95, "q_withdraw": 0.90, "q_net": 0.70, "q_flicker": 0.60, "events": 0},
    {"q_add": 0.90, "q_withdraw": 0.85, "q_net": 0.60, "q_flicker": 0.70, "events": 0},
    {"q_add": 0.90, "q_withdraw": 0.80, "q_net": 0.60, "q_flicker": 0.75, "events": 4},
    {"q_add": 0.85, "q_withdraw": 0.80, "q_net": 0.55, "q_flicker": 0.75, "events": 22},
    {"q_add": 0.85, "q_withdraw": 0.75, "q_net": 0.55, "q_flicker": 0.80, "events": 77},
    {"q_add": 0.80, "q_withdraw": 0.70, "q_net": 0.50, "q_flicker": 0.85, "events": 338},
    {"q_add": 0.80, "q_withdraw": 0.75, "q_net": 0.50, "q_flicker": 0.80, "events": 162},
    {"q_add": 0.80, "q_withdraw": 0.75, "q_net": 0.55, "q_flicker": 0.80, "events": 164},
    {"q_add": 0.80, "q_withdraw": 0.75, "q_net": 0.50, "q_flicker": 0.85, "events": 292},
    {"q_add": 0.80, "q_withdraw": 0.70, "q_net": 0.55, "q_flicker": 0.85, "events": 338},
    {"q_add": 0.85, "q_withdraw": 0.70, "q_net": 0.50, "q_flicker": 0.85, "events": 188},
    {"q_add": 0.85, "q_withdraw": 0.75, "q_net": 0.50, "q_flicker": 0.85, "events": 158},
    {"q_add": 0.85, "q_withdraw": 0.70, "q_net": 0.55, "q_flicker": 0.85, "events": 187},
    {"q_add": 0.85, "q_withdraw": 0.75, "q_net": 0.55, "q_flicker": 0.85, "events": 156},
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


def protocol() -> dict[str, Any]:
    """Return the immutable CRRC-72 singleton protocol."""

    return {
        "protocol_version": "crrc72_v1_2026-07-17",
        "name": "CRRC-72 — Cross-Venue Radial Refill Compression",
        "claim": (
            "A same-side inner-book refill that occurs simultaneously in USD-M and "
            "COIN-M while outer same-side depth withdraws and inner depth remains "
            "credible predicts a six-hour move toward the refilled side."
        ),
        "evidence_boundary": {
            "book_feature_rows_inspected": True,
            "support_incidence_grid_inspected": True,
            "incidence_correction_before_outcomes": (
                "eight looser-cell scratch counts were replaced by the canonical "
                "quarter-contained t+10m scheduler replay; the selected cell and its "
                "156-event count did not change"
            ),
            "post_entry_price_return_or_equity_opened": False,
            "2023_selection_outcomes_opened": False,
            "2024_test_outcomes_opened": False,
            "2025_eval_outcomes_opened": False,
            "2026_holdout_outcomes_opened": False,
            "underlying_book_family_results_seen_elsewhere": True,
            "historical_results_can_promote_live": False,
            "minimum_forward_shadow_required": True,
        },
        "novelty_boundary": {
            "versus_rex": (
                "contemporaneous cross-venue depth-shell mechanics, not price-action "
                "exhaustion and reclaim"
            ),
            "versus_oi_funding_premium_kimchi": (
                "the signal uses no OI, funding, premium, spot basis, Kimchi, FX, or DXY"
            ),
            "versus_pdf10": (
                "same-side radial add/withdraw compression, not displayed-depth versus "
                "firmness divergence"
            ),
            "versus_rlwc144": (
                "one contemporaneous state, not a temporal outer-to-middle-to-inner cascade"
            ),
            "versus_cclh": (
                "flow and credibility conjunction, not persistent full-depth geometry hysteresis"
            ),
            "versus_near_pressure": (
                "unsigned four-condition venue agreement, not a signed weighted shell-flow score"
            ),
            "not_a_new_family_claim": (
                "multiple adjacent order-book candidates and their outcomes already exist; "
                "CRRC-72 must report outcome-blind clock overlap and later PnL correlation"
            ),
            "llm_tree_or_markov_dependency": False,
        },
        "universe": {
            "signal_venues": ["Binance USD-M BTCUSDT", "Binance COIN-M BTCUSD_PERP"],
            "execution_instrument": "Binance USD-M BTCUSDT perpetual only",
            "execution_role_of_coin_m": "signal-only; no inverse-contract PnL is executed",
            "side": "long, short, or flat",
            "gross_when_active": 0.5,
            "maximum_concurrent_positions": 1,
        },
        "source_contract": {
            "shell_panel": {
                "path": (
                    "data/binance_cross_collateral_book_shells_btc_2023/"
                    "BTC_cross_collateral_book_shells_5m_2023.csv.gz"
                ),
                "sha256": SHELL_SOURCE_SHA256,
                "rows": 105120,
            },
            "credibility_panel": {
                "path": (
                    "data/binance_cross_collateral_book_credibility_btc_2023/"
                    "BTC_cross_collateral_book_credibility_5m_2023.csv.gz"
                ),
                "sha256": CREDIBILITY_SOURCE_SHA256,
                "rows": 105120,
            },
            "joint_complete_rows_observed": 101649,
            "selection_end_exclusive": "2024-01-01 00:00:00 UTC",
            "outcome_columns_forbidden_during_support": [
                "open",
                "high",
                "low",
                "close",
                "funding_rate",
                "return",
                "pnl",
                "equity",
            ],
            "nonjoint_or_nonfinite_row": "fail closed",
        },
        "feature_formula": {
            "bar": "completed 5-minute bookDepth aggregate built from nominal 30-second snapshots",
            "sides": {"m": "bid", "p": "ask"},
            "per_venue_and_side": {
                "inner_add": "shell_flow_add_side1 + 0.5 * shell_flow_add_side2",
                "outer_withdraw": "shell_flow_withdraw_side4 + shell_flow_withdraw_side5",
                "inner_net": "0.5 * (log_net_side1 + log_net_side2)",
                "inner_flicker": (
                    "0.25 * (log_mad_side1 + log_mad_side2 + "
                    "log_step_side1 + log_step_side2); arithmetic mean, never division"
                ),
            },
            "lagged_quantile": {
                "window_rows": 8640,
                "minimum_finite_prior_rows": 6912,
                "quantile_interpolation": "linear",
                "shift_rows": 1,
                "current_row_excluded": True,
                "zero_nan_or_infinite_threshold": "fail closed",
            },
            "selected_thresholds": {
                "inner_add_at_least_q": 0.85,
                "outer_withdraw_at_least_q": 0.75,
                "inner_net_at_least_q": 0.55,
                "inner_flicker_at_most_q": 0.85,
            },
            "venue_side_pass": "all four inequalities pass on one venue and side",
            "bid_both": "UM bid pass AND CM bid pass",
            "ask_both": "UM ask pass AND CM ask pass",
            "long": "bid_both AND NOT ask_both",
            "short": "ask_both AND NOT bid_both",
            "conflict": "bid_both AND ask_both is flat",
        },
        "support_selection": {
            "grid": [dict(cell) for cell in SUPPORT_GRID],
            "candidate_cells_inspected": len(SUPPORT_GRID),
            "selection_rule": (
                "among cells satisfying every support gate, lexicographically maximize "
                "q_add, then q_withdraw, then q_net, then minimize q_flicker"
            ),
            "selected_cell": {
                "q_add": 0.85,
                "q_withdraw": 0.75,
                "q_net": 0.55,
                "q_flicker": 0.85,
                "observed_nonoverlap_events": 156,
            },
            "selection_used_outcomes": False,
            "no_threshold_or_hold_repair_after_freeze": True,
        },
        "clock": {
            "signal_bar": "t covers [t, t+5m)",
            "feature_available_time": "t+5m",
            "entry_time": "t+10m open; one full 5m bar after feature availability",
            "hold_bars": 72,
            "exit_time": "entry open + 72 completed 5m bars = six hours",
            "scheduler": "greedy non-overlap, reset flat at each UTC calendar quarter",
            "quarter_containment": "signal, entry, full hold, and exit must remain in one quarter",
            "source_gap_after_entry": (
                "does not cancel a live position; only entry formation requires joint completeness"
            ),
        },
        "support_gate": {
            "nonoverlap_events_at_least": 150,
            "events_each_half_at_least": 50,
            "events_each_quarter_at_least": 25,
            "minimum_side_share": 0.35,
            "maximum_month_share": 0.20,
            "maximum_quarter_share": 0.40,
            "all_signal_inputs_current_joint_complete_and_finite": True,
            "outcome_columns_forbidden": True,
        },
        "outcome_blind_independence_gate": {
            "compare_event_clocks_against": [
                "PDF-10",
                "CCLH",
                "RLWC-144",
                "selected near-pressure",
            ],
            "exact_entry_jaccard_at_most_each": 0.05,
            "twelve_bar_tolerant_candidate_match_share_at_most_each": 0.35,
            "position_time_jaccard_at_most_each": 0.25,
            "rlwc_zero_event_clock_is_reported_but_not_evidence_of_independence": True,
            "post_entry_return_or_pnl_forbidden": True,
            "qualification_not_claim_of_pnl_orthogonality": True,
        },
        "execution": {
            "base_cost_bp_per_notional_side": 6.0,
            "stress_cost_bp_per_notional_side": 10.0,
            "cost_applied": "entry and exit on 0.5x absolute USD-M notional",
            "funding_interval": "entry_time < exact USD-M funding_time <= exit_time",
            "funding_completeness": "no forward fill; missing expected held settlement invalidates evaluation",
            "strict_mdd": (
                "global/pre-entry HWM; entry, exit, funding cash, hypothetical liquidation "
                "cost, and held OHLC with favorable-before-adverse ordering"
            ),
            "cagr": "full declared wall-clock including warm-up and idle cash",
            "tp_sl": None,
            "liquidation_model": None,
        },
        "selection_2023": {
            "singleton_no_parameter_ranking": True,
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_at_least": 3.0,
            "strict_mdd_at_most_pct": 15.0,
            "trades_at_least": 150,
            "each_calendar_quarter_absolute_return_positive": True,
            "long_only_absolute_return_positive": True,
            "short_only_absolute_return_positive": True,
            "ten_bp_stress_absolute_return_positive": True,
            "entry_and_exit_delay_plus_5m_absolute_return_positive": True,
            "direction_flip_cagr_lower": True,
            "monthly_cluster_signflip_p_at_most": 0.10,
        },
        "controls_not_repair_candidates": [
            "UM signal venue only",
            "COIN-M signal venue only",
            "credibility conditions removed",
            "inner-add condition only",
            "outer-withdraw condition only",
            "exact long-short direction flip",
            "entry and exit delayed five minutes",
            "ten basis points per notional side",
        ],
        "sequential_oos": {
            "2024_opened_only_after_complete_2023_pass": True,
            "2024_absolute_return_positive": True,
            "2024_cagr_to_strict_mdd_at_least": 3.0,
            "2024_strict_mdd_at_most_pct": 15.0,
            "2024_trades_at_least": 150,
            "2024_each_calendar_half_absolute_return_positive": True,
            "2025_opened_only_after_complete_2024_pass": True,
            "2026_opened_only_after_complete_2025_pass": True,
            "no_sign_threshold_hold_scale_or_feature_repair": True,
            "minimum_forward_shadow_days_for_promotion": 90,
        },
        "orthogonality_after_standalone_pass": {
            "compare_against": "all promoted/live sleeves on one synchronized strict ledger",
            "absolute_daily_pnl_pearson_at_most": 0.30,
            "absolute_weekly_pnl_pearson_at_most": 0.40,
            "synchronized_portfolio_marginal_improvement_required": True,
        },
        "live_parity_contract": {
            "archive_is_not_live": True,
            "collector_required": (
                "live UM and CM local books recreating cumulative +/-1..5 percent depth "
                "at nominal 30-second snapshots"
            ),
            "aggregation_required": (
                "the same 5m shell-flow and credibility transforms, UTC boundaries, "
                "joint completeness, lagged quantiles, and t+10m entry clock"
            ),
            "parity_before_promotion": (
                "shadow collector must reproduce archive transforms within documented "
                "numeric tolerances and pass stale/gap/reconnect tests"
            ),
            "official_archive": "https://github.com/binance/binance-public-data",
            "official_local_book_spec": (
                "https://developers.binance.com/en/docs/products/derivatives-trading/"
                "usds-futures/websocket-market-streams/How-to-manage-a-local-order-book-correctly"
            ),
        },
        "stop_rule": (
            "Reject before outcomes if support or outcome-blind independence fails. "
            "Otherwise hash-freeze the event clock, physically isolated 2023 execution "
            "sources, and evaluator, then open 2023 exactly once. Retire without repair "
            "on the first failed frozen gate. Open 2024, 2025, and 2026 only after each "
            "preceding window completely passes. Historical success still requires live "
            "collector parity and at least 90 forward-shadow days."
        ),
    }


def markdown(payload: dict[str, Any]) -> str:
    return f"""# CRRC-72 preregistration — 2026-07-17

## Mechanism

CRRC-72 looks for a contemporaneous radial compression of the Binance BTC
order book. On one side, both USD-M and COIN-M must show strong inner-shell
adds, outer-shell withdrawals, positive inner net depth, and non-extreme
inner flicker. Bid agreement is long, ask agreement is short, and a two-sided
conflict is flat.

Every rolling threshold uses the previous 8,640 rows only, requires 6,912
finite prior rows, and excludes the current row. A completed signal bar is
available at `t+5m`; the trade enters USD-M BTCUSDT at `t+10m`, holds six
hours, and is greedily non-overlapping within each calendar quarter.

## Frozen support choice

Fourteen incidence-only cells were inspected without any price or return.
Among cells passing >=150 events, >=50 per half, >=25 per quarter, balanced
sides, and concentration gates, the deterministic rule maximized add,
withdrawal, and net quantiles in order, then minimized the flicker quantile.
It selected `(0.85, 0.75, 0.55, 0.85)` with 156 scheduled events. No outcome
was used.

Before any outcome was opened, a canonical replay corrected eight scratch
incidence counts for looser cells whose earlier temporary scheduler was not
the frozen `t+10m`, quarter-contained clock. The selected cell and its 156
events were unchanged.

## Novelty boundary

This is distinct from REX, OI/funding/premium/Kimchi signals and differs from
PDF-10, CCLH, RLWC-144, and the signed near-pressure score. It is not claimed
to be a globally new family because adjacent book-depth experiments already
exist. Their causal clocks must be replayed and pass frozen overlap gates
before 2023 returns can be opened; PnL correlation is tested only after a
standalone pass.

## Evaluation and stop rule

Strict MDD uses the global/pre-entry HWM, held OHLC with
favorable-before-adverse ordering, exact funding cash, all entry/exit costs,
and hypothetical liquidation costs. CAGR spans the full calendar including
warm-up and idle periods. The singleton must be positive in every 2023
quarter, in both side sleeves, under 10bp stress and a +5m delay, while
reaching CAGR/strict-MDD >=3 and strict MDD <=15%. The first failed gate
retires it without sign, threshold, hold, or feature repair and keeps later
years sealed.

## Live limitation

Binance Vision archives are not a live feed. Promotion requires a live UM/CM
local-order-book collector that reproduces cumulative +/-1..5% depth at
nominal 30-second snapshots and the exact 5m transforms, completeness rules,
quantiles, and clock. Official references:

- https://github.com/binance/binance-public-data
- https://developers.binance.com/en/docs/products/derivatives-trading/usds-futures/websocket-market-streams/How-to-manage-a-local-order-book-correctly

Protocol hash: `{payload['protocol_hash']}`
"""


def run(
    output: str | Path = DEFAULT_OUTPUT,
    docs_output: str | Path = DEFAULT_DOCS,
) -> dict[str, Any]:
    frozen = protocol()
    payload = {
        "protocol": frozen,
        "protocol_hash": canonical_hash(frozen),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    docs_path = Path(docs_output)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(markdown(payload))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--docs-output", default=str(DEFAULT_DOCS))
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.docs_output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
