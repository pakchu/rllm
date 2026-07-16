"""Preregister DCRM-1 before opening any post-entry return."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "results/dispersion_conditioned_residual_momentum_preregistration_2026-07-17.json"
)
DEFAULT_DOCS = Path(
    "docs/dispersion-conditioned-residual-momentum-preregistration-2026-07-17.md"
)


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def protocol() -> dict[str, Any]:
    """Return the immutable DCRM-1 singleton protocol."""

    return {
        "protocol_version": "dcrm_v1_2026-07-17",
        "name": "DCRM-1 — Dispersion-Conditioned Cross-Alt Residual Momentum",
        "claim": (
            "A weekly beta-residual 30-day cross-alt momentum spread persists for seven "
            "days, while a strictly-prior dispersion state identifies weeks that should "
            "carry only quarter gross rather than full gross."
        ),
        "evidence_boundary": {
            "underlying_alt_market_rows_seen_elsewhere": True,
            "outcome_blind_weekly_feature_clock_inspected": True,
            "post_entry_returns_or_equity_opened": False,
            "support_only_scale_candidates_inspected": [
                "abstain above strictly-prior 26-state q80",
                "quarter gross above strictly-prior 26-state q80",
            ],
            "selection_basis": (
                "quarter gross preserved 92 support events versus 59 under abstention; "
                "no post-entry price or return was calculated"
            ),
            "2023_selection_outcomes_opened": False,
            "2024_test_outcomes_opened": False,
            "2025_eval_outcomes_opened": False,
            "2026_holdout_outcomes_opened": False,
            "historical_results_can_promote_live": False,
            "minimum_forward_shadow_required": True,
        },
        "novelty_boundary": {
            "versus_rex": "cross-alt weekly rank, not BTC price-action exhaustion/reclaim",
            "versus_oi_funding_premium_kimchi": (
                "uses no OI, funding signal, premium index, spot premium, Kimchi, FX, or DXY"
            ),
            "versus_lore_lorc_cres": (
                "30-day residual continuation with a weekly clock, not short-horizon "
                "residual exhaustion, continuation event, or expert switcher"
            ),
            "versus_current_active_portfolio": (
                "one market-neutral alt pair, no BTC position, low weekly turnover, and "
                "a distinct seven-day holding interval"
            ),
            "not_globally_orthogonal": (
                "shares six-alt residual/beta machinery with LORE/LORC; outcome-blind "
                "clock overlap and eventual PnL correlation must be reported"
            ),
            "llm_or_tree_dependency": False,
        },
        "universe": {
            "venue": "Binance USD-M perpetual futures",
            "symbols": [
                "ETHUSDT",
                "SOLUSDT",
                "BNBUSDT",
                "XRPUSDT",
                "ADAUSDT",
                "DOGEUSDT",
            ],
            "position": "one long alt and one short alt; no BTC leg",
            "maximum_active_pairs": 1,
            "maximum_gross": 1.0,
            "minimum_gross_when_active": 0.25,
        },
        "source_contract": {
            "market": "exact completed Binance USD-M 5m OHLC",
            "funding": "exact reported rate and millisecond settlement timestamp per held symbol",
            "selection_source": "physically frozen 2023-2024 LORE prefix",
            "required_complete_symbols": 6,
            "no_cross_venue_or_stale_macro_input": True,
            "no_2024_outcome_source_before_2023_pass": True,
            "no_2025_source_before_2024_pass": True,
        },
        "feature_formula": {
            "weekly_boundary": "Monday 00:00 UTC",
            "last_observable_bar": "Sunday 23:55 UTC 5m bar, completed at Monday 00:00",
            "row_cutoff": (
                "latest feature row open timestamp is decision_time - 5m; every market "
                "row with timestamp >= decision_time is forbidden"
            ),
            "momentum": "log(close[t-5m] / close[t-30d-5m])",
            "hourly_return": "log ratio of consecutive completed xx:55 5m closes",
            "factor": "per-symbol median completed hourly return of the other five alts",
            "factor_30d": (
                "sum of that symbol-specific leave-one-out hourly factor over the 720 "
                "completed hours ending at t-5m"
            ),
            "beta": {
                "lookback_completed_hours": 720,
                "minimum_completed_hours": 336,
                "shift_completed_hours": 1,
                "clip": [0.25, 2.5],
            },
            "score": "symbol 30-day return - clipped beta * symbol-specific factor_30d",
            "long_symbol": "lexically tie-broken maximum score",
            "short_symbol": "lexically tie-broken minimum score",
            "dispersion": "population standard deviation of the six scores",
            "dispersion_reference": {
                "lookback_prior_weekly_states": 26,
                "minimum_prior_weekly_states": 8,
                "quantile": 0.8,
                "quantile_interpolation": "linear",
                "current_state_excluded": True,
            },
            "gross_scale": (
                "1.0 when current dispersion <= strictly-prior q80; otherwise 0.25"
            ),
            "base_weights": (
                "positive gross-one weights solving long_beta*w_long = "
                "short_beta*w_short"
            ),
            "executed_weights": "base long and short weights multiplied by gross_scale",
        },
        "clock": {
            "feature_available_time": "Monday 00:00 UTC",
            "entry_time": "Monday 00:05 UTC open",
            "exit_time": "next Monday 00:05 UTC open",
            "hold_days": 7,
            "nonoverlap": True,
            "same_boundary_order": "close old pair before opening the new pair",
            "backtest_fill_proxy": (
                "exact Monday 00:05 5m open; no signal uses that row or any later row"
            ),
            "live_fill_contract": (
                "marketable orders during a fixed five-minute window beginning 00:05; "
                "partial or missed fills are logged and never backfilled or repaired"
            ),
            "same_timestamp_funding": (
                "funding at Monday 00:00 precedes the new entry and belongs only to an "
                "already-held pair"
            ),
        },
        "execution": {
            "base_cost_bp_per_notional_side": 6.0,
            "stress_cost_bp_per_notional_side": 10.0,
            "cost_applied": "entry and exit on each leg, proportional to absolute notional",
            "funding_interval": "entry_time < funding_time <= exit_time",
            "funding_completeness": (
                "no forward fill; a missing expected held funding settlement on either "
                "leg invalidates the evaluation rather than silently skipping a trade"
            ),
            "strict_mdd": (
                "global/pre-entry HWM, entry/exit and hypothetical liquidation cost, "
                "funding cash, and held favorable-before-adverse two-leg OHLC; funding "
                "credits cannot create an intratrade favorable peak"
            ),
            "cagr": "full declared wall-clock including warm-up, idle cash, and low-gross weeks",
            "tp_sl": None,
            "liquidation_model": None,
        },
        "support_gate": {
            "events_2023_2024_at_least": 85,
            "events_each_year_at_least": 35,
            "events_each_half_at_least": 10,
            "unique_ordered_pairs_at_least": 12,
            "maximum_ordered_pair_share_at_most": 0.20,
            "maximum_month_share_at_most": 0.15,
            "all_six_symbols_long_and_short_required": True,
            "outcome_columns_forbidden": True,
        },
        "selection_2023": {
            "singleton_no_parameter_ranking": True,
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_at_least": 2.0,
            "strict_mdd_at_most_pct": 15.0,
            "trades_at_least": 35,
            "each_calendar_half_absolute_return_positive": True,
            "ten_bp_stress_absolute_return_positive": True,
            "entry_and_exit_delay_plus_5m_absolute_return_positive": True,
            "direction_flip_cagr_lower": True,
            "weekly_cluster_signflip_p_at_most": 0.10,
        },
        "sequential_oos": {
            "2024_opened_only_after_2023_pass": True,
            "2024_absolute_return_positive": True,
            "2024_cagr_to_strict_mdd_at_least": 3.0,
            "2024_strict_mdd_at_most_pct": 15.0,
            "2024_trades_at_least": 40,
            "2024_each_calendar_half_absolute_return_positive": True,
            "combined_2023_2024_cagr_to_strict_mdd_at_least": 3.0,
            "2025_opened_only_after_2024_pass": True,
            "2026_opened_only_after_2025_pass": True,
            "no_sign_lookback_hold_scale_pair_or_beta_repair": True,
            "minimum_forward_shadow_days_for_promotion": 90,
        },
        "controls_not_repair_candidates": [
            "exact long-short direction flip",
            "entry and exit delayed five minutes",
            "ten basis points per notional side",
            "dispersion scaling disabled at full gross",
            "dispersion scaling inverted",
        ],
        "orthogonality_after_standalone_pass": {
            "compare_against": "all promoted/live sleeves on one synchronized strict ledger",
            "exact_entry_jaccard_at_most": 0.02,
            "position_jaccard_at_most": 0.20,
            "absolute_daily_pnl_pearson_at_most": 0.30,
            "absolute_weekly_pnl_pearson_at_most": 0.40,
            "synchronized_portfolio_marginal_improvement_required": True,
        },
        "outcome_blind_overlap_before_outcomes": {
            "compare_clocks_against": ["LORE", "LORC", "all promoted/live sleeves"],
            "post_entry_return_or_pnl_forbidden": True,
            "report_exact_entry_jaccard": True,
            "report_position_time_jaccard": True,
            "qualification_not_claim_of_pnl_orthogonality": True,
        },
        "research_context": {
            "supportive_primary_source": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4322637",
            "dispersion_conditioning_preprint": (
                "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6648082"
            ),
            "adverse_realism_check": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4675565",
            "horizon_check": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3913263",
            "version_note": (
                "The 2026 dispersion and realism papers are recent working papers; they "
                "motivate a falsifiable test and do not establish this implementation's alpha."
            ),
        },
        "stop_rule": (
            "Reject before outcomes if support fails. Open 2023 once only; if any frozen "
            "selection gate fails, retire without opening 2024 or changing sign, lookback, "
            "hold, scale, pair rule, or beta. Open 2024 only after a complete 2023 pass, "
            "then 2025 and 2026 sequentially. Test portfolio orthogonality only after the "
            "standalone candidate passes."
        ),
    }


def markdown(payload: dict[str, Any]) -> str:
    return f"""# DCRM-1 preregistration — 2026-07-17

## Mechanism

DCRM-1 is a weekly, market-neutral cross-alt pair. At Monday 00:00 UTC it
uses only completed bars through Sunday 23:55, ranks six alts by beta-residual
30-day momentum, then buys the maximum and shorts the minimum at the Monday
00:05 open. The pair exits seven days later.

A strictly-prior 26-week dispersion q80 controls risk, not direction: gross is
1.0 below or at q80 and 0.25 above it. The current week is excluded from the
reference distribution. The weights are positive and beta-neutral before that
gross scale is applied.

## Why this is structurally different

The candidate has no BTC leg and uses no REX, OI, funding signal, premium,
Kimchi, FX, DXY, tree model, Markov state, or LLM. Its weekly cross-sectional
ranking and seven-day holding period also differ from the active event-driven
sleeves. Correlation is nevertheless an outcome and will be opened only after
the standalone gates pass.

## Evidence boundary

Only the causal weekly feature/support clock was inspected. Two outcome-blind
risk treatments were compared: abstention above q80 left 59 events, while
quarter gross retained 92. No post-entry price, trade return, or equity curve
was calculated. The latter is frozen as DCRM-1.

## Qualification

Support must first pass at least 85 events, 35 per year, 10 per half, 12
ordered pairs, <=20% pair concentration, <=15% month concentration, and all
six symbols on both sides. Then 2023 is opened exactly once. It must be
positive, have CAGR/strict-MDD >=2, strict MDD <=15%, at least 35 trades,
positive halves and all controls. Only a complete pass opens 2024, which must
reach CAGR/strict-MDD >=3; 2025 and 2026 remain sequentially sealed.

Strict MDD uses the global/pre-entry HWM, held two-leg OHLC with
favorable-before-adverse ordering, funding, and hypothetical liquidation
costs. CAGR always spans the full declared calendar.

## Research context

- 30-day cross-sectional continuation with a seven-day horizon:
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4322637
- dispersion-conditioned momentum (recent 2026 working paper):
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6648082
- adverse realism check on implementable crypto momentum (2026 working paper):
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4675565
- horizon dependence and reversal warning:
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3913263

These sources motivate the test; they do not validate this candidate.

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
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
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
