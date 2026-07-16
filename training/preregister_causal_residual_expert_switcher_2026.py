"""Preregister the one-shot 2026 CRES-1 confirmation before opening outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OUTPUT = "results/causal_residual_expert_switcher_2026_preregistration_2026-07-17.json"
DOCS_OUTPUT = "docs/causal-residual-expert-switcher-2026-preregistration-2026-07-17.md"
DEVELOPMENT_RESULT = "results/causal_residual_expert_switcher_development_2026-07-17.json"
EXPECTED_DEVELOPMENT_SHA256 = "30065bed3a7ebd0e6150aadcd198364ac63222184b53c71656d297b9ff366af5"


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol() -> dict[str, Any]:
    return {
        "protocol_version": "cres_v1_2026_one_shot_2026-07-17",
        "name": "CRES-1 — Causal Residual Expert Switcher",
        "claim": (
            "The continuation-versus-reversion sign of an extreme factor-neutral alt residual "
            "is conditionally predictable from the event state and only already-completed expert "
            "outcomes; high-confidence events retain positive market-neutral pair returns after "
            "costs while a causal range-risk scaler limits tail exposure."
        ),
        "evidence_boundary": {
            "development_2023_2025_opened": True,
            "development_2023_2025_contaminated_for_confirmation": True,
            "development_result_path": DEVELOPMENT_RESULT,
            "development_result_sha256": EXPECTED_DEVELOPMENT_SHA256,
            "confirmation_2026_post_entry_returns_opened": False,
            "confirmation_window": [
                "2026-01-01T00:00:00Z",
                "2026-07-01T00:00:00Z",
            ],
            "repository_wide_human_pristine_claim": False,
            "one_new_family_one_2026_open": True,
        },
        "development_disclosure": {
            "selection_windows": "2023 warm-up; 2024 and 2025 development",
            "selected_2024_2025": {
                "absolute_return_pct": 36.63173004797691,
                "cagr_pct": 16.877099539423625,
                "strict_mdd_pct": 5.437082267827098,
                "cagr_to_strict_mdd": 3.1040728663038,
                "trades": 49,
            },
            "selected_2025": {
                "absolute_return_pct": 13.789728955914683,
                "cagr_pct": 13.799797616049148,
                "strict_mdd_pct": 5.304715382869862,
                "cagr_to_strict_mdd": 2.6014209283709824,
                "trades": 24,
            },
            "ten_bp_2024_2025": {
                "absolute_return_pct": 32.33361219389488,
                "cagr_pct": 15.02532598527171,
                "strict_mdd_pct": 5.53290500267154,
                "cagr_to_strict_mdd": 2.7156305734540527,
                "trades": 49,
            },
            "multiplicity": (
                "The research process screened rolling and exponentially weighted experts, "
                "ridge windows/penalties, abstention levels, regime rules, stops, and risk "
                "scalers. Nothing in 2023-2025 is OOS evidence."
            ),
        },
        "universe": {
            "venue": "Binance USD-M perpetual futures",
            "symbols": ["ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"],
            "position": "one factor-beta-neutral long/short alt pair or flat; no BTC leg",
            "structural_orthogonality": (
                "cross-sectional relative value rather than BTC direction, REX, BTC OI, "
                "funding/premium gate, Kimchi/FX, or LLM directional prediction"
            ),
        },
        "source_contract": {
            "physical_prefix": ["2025-01-01T00:00:00Z", "2026-07-01T00:00:00Z"],
            "warmup": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "confirmation": ["2026-01-01T00:00:00Z", "2026-07-01T00:00:00Z"],
            "base_interval": "5m bar-open timestamps",
            "required_market_columns": [
                "date",
                "open",
                "high",
                "low",
                "close",
                "quote_asset_volume",
                "taker_buy_quote",
                "tic",
            ],
            "all_six_symbols_required": True,
            "exact_complete_grid": True,
            "no_fill_or_nearest_join": True,
            "funding": "exact symbol funding timestamps; no fill or synthetic rate",
            "support_builder_may_not_compute_post_entry_returns": True,
        },
        "base_event": {
            "completed_hour": "12 exact completed 5m bars, right-labelled at signal time",
            "leave_one_out_factor": "median hourly return of the other five symbols",
            "beta": {
                "lookback_hours": 720,
                "minimum_hours": 336,
                "shift_hours": 1,
                "clip": [0.25, 2.5],
            },
            "residual_horizon_hours": 12,
            "residual_z": {"lookback_hours": 2160, "minimum_hours": 1080, "history_shift": 1},
            "flow": "2*12h taker_buy_quote/12h quote_asset_volume-1",
            "flow_z": {"lookback_hours": 2160, "minimum_hours": 1080, "history_shift": 1},
            "winner_residual_z_at_least": 1.5,
            "loser_residual_z_at_most": -1.5,
            "winner_price_minus_flow_z_at_least": 1.0,
            "loser_flow_minus_price_z_at_least": 1.0,
            "minimum_leg_weight_at_gross_one": 0.25,
            "reservation": "start flat at 2026-01-01; reserve non-overlapping 12h events",
        },
        "online_policy": {
            "policy_id": "CRES01",
            "two_experts": ["continuation", "reversion"],
            "target": "continuation net log return minus reversion net log return at 6bp/side with exact funding",
            "full_information_update": (
                "after each event exit, both directional counterfactuals are observable from the "
                "same frozen OHLC/funding path and may train later events"
            ),
            "target_availability": "prior event exit + 5m <= current signal",
            "initial_training_seed": "all frozen 2023-2025 event features and expert outcomes",
            "minimum_training_rows": 52,
            "maximum_recent_training_rows": 104,
            "ridge_alpha": 300.0,
            "feature_standardization": "training rows only; zero for zero-variance columns",
            "target_centering": "subtract training target mean for fit; do not add mean/intercept to prediction",
            "features": [
                "loser_residual_z",
                "winner_residual_z",
                "loser_flow_z",
                "winner_flow_z",
                "setup_score",
                "long_weight",
                "short_weight_abs",
                "long_beta",
                "short_beta",
                "edge_mean_8",
                "edge_mean_16",
                "edge_mean_24",
                "edge_std_24",
            ],
            "confidence": "abs(prediction) > 82.5th percentile of abs in-sample fitted predictions",
            "direction": "positive=continuation; negative=reversion; otherwise flat",
            "hold_hours": 12,
        },
        "risk_policy": {
            "measure": "maximum across proposed legs of RMS log(high/low) over 864 completed 5m bars",
            "reference": "median risk of prior 52 base events, independent of outcomes",
            "gross_scale": "clip(reference/current, 0.25, 1.0)",
            "apply_equally_to_both_legs": True,
            "factor_beta_neutral_after_scaling": True,
            "no_stop_loss_or_take_profit": True,
        },
        "execution": {
            "entry": "5m open at signal + 5m",
            "exit": "5m open at entry + 12h",
            "base_cost_bp_per_notional_side": 6.0,
            "stress_cost_bp_per_notional_side": 10.0,
            "funding_interval": "entry < funding timestamp <= exit",
            "strict_mdd": (
                "global/pre-entry HWM; entry cost first; funding cash included; favorable "
                "long-high/short-low before adverse long-low/short-high for every held bar; "
                "hypothetical liquidation cost at every mark; scheduled exit cost"
            ),
            "cagr": "full 2026-01-01 through 2026-07-01 wall-clock including idle periods",
        },
        "support_gate": {
            "events_at_least": 35,
            "q1_events_at_least": 15,
            "q2_events_at_least": 15,
            "unique_ordered_pairs_at_least": 8,
            "maximum_ordered_pair_share": 0.20,
            "symbols_seen_each_side_at_least": 5,
            "outcome_columns_forbidden": True,
        },
        "confirmation_gate": {
            "absolute_return_positive": True,
            "annualized_cagr_to_strict_mdd_at_least": 3.0,
            "strict_mdd_at_most_pct": 15.0,
            "executed_trades_at_least": 10,
            "q1_absolute_return_positive": True,
            "q2_absolute_return_positive": True,
            "ten_bp_cost_stress_absolute_return_positive": True,
            "entry_delay_plus_5m_absolute_return_positive": True,
            "direction_flip_cagr_lower_than_primary": True,
        },
        "orthogonality_after_strategy_pass": {
            "daily_mark_to_market_pearson_to_btc_abs_at_most": 0.30,
            "daily_mark_to_market_beta_to_btc_abs_at_most": 0.15,
            "exact_entry_jaccard_to_current_portfolio_at_most": 0.02,
            "candidate_entries_near_6h_of_portfolio_at_most": 0.25,
            "position_bar_jaccard_at_most": 0.15,
            "marginal_synchronized_portfolio_improvement_required": True,
        },
        "deployment_boundary": {
            "research_or_shadow_only": True,
            "reason": (
                "the current executor does not yet provide atomic two-leg alt pair execution, "
                "partial-fill neutralization, and pair-level reservation"
            ),
        },
        "stop_rule": (
            "Freeze source, outcome-blind base clock, historical seed, evaluator source, and tests "
            "before opening 2026 post-entry returns. Run the exact evaluator once. If any strategy "
            "gate fails, retire CRES-1; do not repair thresholds, sign, hold, risk scaling, symbols, "
            "or model hyperparameters on 2026. Orthogonality is evaluated only after strategy pass."
        ),
    }


def markdown(payload: dict[str, Any]) -> str:
    dev = payload["protocol"]["development_disclosure"]
    selected = dev["selected_2024_2025"]
    return f"""# CRES-1 2026 one-shot preregistration

## Evidence boundary

All 2023-2025 outcomes are development data. The selected development row made
{selected['absolute_return_pct']:.2f}% absolute return, {selected['cagr_pct']:.2f}% full-calendar
CAGR, {selected['strict_mdd_pct']:.2f}% strict MDD, ratio
{selected['cagr_to_strict_mdd']:.2f}, and {selected['trades']} trades over 2024-2025. Calendar
2025 alone had ratio {dev['selected_2025']['cagr_to_strict_mdd']:.2f}; it is not hidden or
relabelled OOS. The search multiplicity makes 2026H1 the first possible
confirmation of this new meta-family.

## Exact frozen policy

- six Binance USD-M alts; one factor-beta-neutral pair or flat, no BTC leg;
- unchanged 12h residual/flow event and fixed 12h hold;
- online ridge uses only outcomes available at least 5m after prior exits;
- last 104 rows, minimum 52, ridge alpha 300, no target-mean/intercept drift;
- 13 fixed current-state and lagged expert-edge features;
- trade only above the fixed 82.5% fitted-confidence quantile;
- causal 3-day range-risk scale versus prior 52-event median, gross 0.25..1.0;
- 6 bp/side base, 10 bp stress, exact funding, full-wall-clock CAGR and strict
  favorable-before-adverse intratrade MDD.

## One-shot 2026 gate

The evaluator covers 2026-01-01 through 2026-07-01. It must have positive
absolute return, annualized CAGR/strict-MDD >= 3, strict MDD <= 15%, at least 10
executed trades, positive Q1 and Q2, positive 10-bp stress, positive +5m delay,
and a worse direction-flip control. A failure retires CRES-1 without repair.

Even on a strategy pass, portfolio promotion additionally requires low BTC and
current-portfolio marked-PnL/timing overlap. Live deployment remains blocked on
atomic two-leg execution and partial-fill neutralization.

Protocol hash: `{payload['protocol_hash']}`
"""


def run(output: str = OUTPUT, docs_output: str = DOCS_OUTPUT) -> dict[str, Any]:
    actual = sha256_file(DEVELOPMENT_RESULT)
    if actual != EXPECTED_DEVELOPMENT_SHA256:
        raise RuntimeError("CRES development result changed before preregistration")
    frozen = protocol()
    payload = {
        "protocol": frozen,
        "protocol_hash": canonical_hash(frozen),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    Path(docs_output).parent.mkdir(parents=True, exist_ok=True)
    Path(docs_output).write_text(markdown(payload))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--docs-output", default=DOCS_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.docs_output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
