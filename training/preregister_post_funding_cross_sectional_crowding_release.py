"""Preregister PFCR-1 before its exact pair returns are opened."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_OUTPUT = Path(
    "results/post_funding_cross_sectional_crowding_release_preregistration_2026-07-17.json"
)
DEFAULT_DOCS = Path(
    "docs/post-funding-cross-sectional-crowding-release-preregistration-2026-07-17.md"
)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def protocol() -> dict[str, Any]:
    symbols = ["ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"]
    return {
        "protocol_version": "pfcr_v1_2026-07-17",
        "name": "PFCR-1 — Post-Funding Cross-Sectional Crowding Release",
        "claim": (
            "After an observed common funding settlement, the highest-funding alt "
            "underperforms the lowest-funding alt over four hours when the current "
            "cross-sectional funding spread is extreme relative to prior settlements."
        ),
        "evidence_boundary": {
            "underlying_alt_market_and_funding_rows_seen_elsewhere": True,
            "exact_pfcr_pair_clock_or_returns_opened": False,
            "2023_fit_opened": False,
            "2024_test_opened": False,
            "2025_eval_opened": False,
            "2026_holdout_opened": False,
            "historical_results_can_promote_live": False,
            "forward_shadow_required": True,
        },
        "novelty_boundary": {
            "versus_afch": (
                "post-settlement four-hour price release, not 28-day pre-settlement carry harvest"
            ),
            "versus_lore_lorc_cres": (
                "funding-settlement event and current funding-rank pair, not residual/flow extremes"
            ),
            "versus_btc_funding_settlement_families": (
                "factor-beta-neutral cross-alt pair with no BTC directional position"
            ),
            "versus_current_portfolio": (
                "no BTC, REX, OI, premium, Kimchi/FX, tree, Markov, or LLM direction"
            ),
        },
        "universe": {
            "venue": "Binance USD-M perpetual futures",
            "symbols": symbols,
            "position": "one long alt and one short alt; no BTC leg",
            "maximum_active_pairs": 1,
            "maximum_gross": 1.0,
        },
        "source_contract": {
            "market": "exact completed Binance USD-M 5m OHLC",
            "funding": "exact reported rate and millisecond settlement timestamp per symbol",
            "common_settlement": "all six symbols have exactly one row at the same timestamp",
            "selection_source": "physically frozen 2023-2024 LORE prefix",
            "no_2025_source_before_selection_pass": True,
        },
        "feature_formula": {
            "current_spread": "max current settled rate minus min current settled rate",
            "spread_reference": {
                "lookback_common_settlements": 180,
                "minimum_prior_settlements": 90,
                "quantile": 0.90,
                "current_event_excluded": True,
            },
            "eligible": "current spread > strictly-prior rolling q90 and spread > 0",
            "short_symbol": "lexically tie-broken current maximum funding rate",
            "long_symbol": "lexically tie-broken current minimum funding rate",
            "factor": "median completed hourly return of the other five alts",
            "beta": {
                "lookback_hours": 720,
                "minimum_hours": 336,
                "shift_hours": 1,
                "clip": [0.25, 2.5],
            },
            "weights": "positive gross-one weights solving long_beta*w_long = short_beta*w_short",
        },
        "clock": {
            "feature_available_time": "settlement timestamp + 5 minutes",
            "entry_time": "settlement timestamp + 10 minutes",
            "hold_hours": 4,
            "nonoverlap": True,
            "same_timestamp_funding": "current settlement is not earned or paid because entry is later",
        },
        "execution": {
            "base_cost_bp_per_notional_side": 6.0,
            "stress_cost_bp_per_notional_side": 10.0,
            "funding_interval": "entry_time < funding_time <= exit_time",
            "strict_mdd": (
                "global/pre-entry HWM, entry/exit and hypothetical liquidation cost, "
                "funding cash, and held favorable-before-adverse two-leg OHLC"
            ),
            "cagr": "full declared wall-clock including idle time",
            "tp_sl": None,
        },
        "support_gate": {
            "events_2023_2024_at_least": 60,
            "events_each_year_at_least": 25,
            "unique_ordered_pairs_at_least": 8,
            "maximum_ordered_pair_share_at_most": 0.25,
            "long_symbols_at_least": 4,
            "short_symbols_at_least": 4,
            "maximum_month_share_at_most": 0.20,
            "outcome_columns_forbidden": True,
        },
        "selection_2023_2024": {
            "singleton_no_parameter_ranking": True,
            "each_year_absolute_return_positive": True,
            "each_year_cagr_to_strict_mdd_at_least": 1.5,
            "combined_cagr_to_strict_mdd_at_least": 3.0,
            "combined_strict_mdd_at_most_pct": 15.0,
            "combined_trades_at_least": 60,
            "ten_bp_stress_absolute_return_positive": True,
            "entry_delay_plus_5m_absolute_return_positive": True,
            "direction_flip_cagr_lower": True,
            "weekly_cluster_signflip_p_at_most": 0.10,
        },
        "sequential_oos": {
            "2025_opened_only_after_2023_2024_pass": True,
            "2026_opened_only_after_2025_pass": True,
            "no_threshold_sign_hold_pair_or_beta_repair": True,
            "minimum_forward_shadow_days_for_promotion": 90,
        },
        "orthogonality_after_standalone_pass": {
            "exact_entry_jaccard_at_most": 0.02,
            "candidate_entries_near_6h_at_most": 0.25,
            "position_jaccard_at_most": 0.15,
            "absolute_daily_pnl_pearson_at_most": 0.30,
            "synchronized_portfolio_marginal_improvement_required": True,
        },
        "controls_not_repair_candidates": [
            "exact long-short direction flip",
            "entry and exit delayed five minutes",
            "ten basis points per notional side",
            "event clock shifted four hours",
        ],
        "stop_rule": (
            "Reject before outcomes if support fails. Reject before 2025 if the singleton "
            "fails 2023-2024. Reject before 2026 if it fails 2025. Never repair on a "
            "consumed future window."
        ),
    }


def markdown(payload: dict[str, Any]) -> str:
    return f"""# PFCR-1 preregistration — 2026-07-17

## Mechanism

At a common Binance USD-M funding settlement, PFCR reads the six rates only
after settlement. If the current cross-sectional spread exceeds the strictly
prior 180-event q90, it buys the lowest-funding alt and shorts the highest-
funding alt. Causal 30-day betas set gross-one factor-neutral weights.

The signal becomes available at settlement +5 minutes, enters at +10 minutes,
and exits four hours later. The just-observed settlement belongs to neither
leg because entry occurs afterward. This tests leveraged crowding release, not
funding carry.

## Qualification

Support is checked without post-entry returns. The singleton must then be
positive in both 2023 and 2024, achieve each-year ratio >=1.5, combined full-
calendar CAGR/strict-MDD >=3, strict MDD <=15%, at least 60 trades, positive
10 bp and +5m-delay controls, inferior direction flip, and weekly cluster
p<=0.10. Only a pass can open 2025; only a 2025 pass can open 2026.

## Boundary

The six-alt rows were used by adjacent research, so historical performance is
not pristine enough for live promotion. Even a full historical pass requires
atomic two-leg execution parity and at least 90 forward-shadow days. No sign,
threshold, hold, beta, or pair repair is allowed after an outcome window opens.

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
