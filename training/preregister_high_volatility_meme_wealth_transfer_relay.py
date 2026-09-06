"""Outcome-blind preregistration for HVMWTR-24."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_high_volatility_energy_technology_spillover_relay as template


DEFAULT_OUTPUT = Path(
    "results/high_volatility_meme_wealth_transfer_relay_preregistration_2026-08-12.json"
)
MEMES = ("GME", "AMC")
BENCHMARK = "SPY"


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
    core = copy.deepcopy(template.build())
    core.pop("manifest_hash")
    core.update(
        protocol_version="high_volatility_meme_wealth_transfer_relay_v1",
        policy_id="HVMWTR-24",
        as_of_date="2026-08-12",
        gross9_rows_opened=False,
        mechanism={
            "claim": (
                "Peer-reviewed evidence identifies a unidirectional wealth transfer from meme stocks to Bitcoin. "
                "Follow the sign of the equal-weight GME/AMC idiosyncratic cash-close shock into the subsequent "
                "BTC interval only when that shock is unusually large and completed BTC variation is high."
            ),
            "side": "strict sign of the equal-weight current SPY-beta-hedged GME/AMC residual",
            "external_support": {
                "paper": (
                    "Spillovers between Bitcoin and Meme stocks (2022), "
                    "Finance Research Letters 50, 103218"
                ),
                "doi": "10.1016/j.frl.2022.103218",
                "reported_fact": (
                    "The study reports a unidirectional wealth-transfer phenomenon from meme stocks to Bitcoin "
                    "and common adverse-news spillovers."
                ),
                "inference_disclosure": (
                    "GME/AMC, strict-prior rolling SPY betas, equal residual weighting, daily adjusted returns, "
                    "rank gates, exact close latency and a 24-hour BTC hold are preregistered adaptations."
                ),
            },
            "why_distinct": (
                "HVMWTR isolates retail-crowding shocks in two canonical meme equities after removing broad SPY "
                "exposure. It is not a mining-equity, broad stock-correlation, MSTR, on-chain, derivatives, "
                "crypto order-flow, lottery-sales, or Gross9 primitive."
            ),
            "why_suited_to_volatile_regimes": (
                "Only upper-35% completed BTC variation and upper-30% absolute meme residuals are admitted."
            ),
            "why_low_gross9_overlap_is_plausible": (
                "Sparse idiosyncratic meme-equity shocks become available only at official US cash closes."
            ),
        },
        features={
            "sessions": (
                "official common US cash sessions on which GME, AMC and SPY all have complete adjusted OHLCV rows"
            ),
            "equity_returns": (
                "for GME, AMC and SPY, log split/dividend-adjusted close_t / adjusted close_(t-1 common session)"
            ),
            "strict_prior_beta": (
                "for each meme, covariance(meme_return,SPY_return)/variance(SPY_return) over at most 60 previous "
                "valid common-session pairs, minimum 40, current pair excluded; finite positive SPY variance required"
            ),
            "meme_residual": "current meme return minus its strict-prior SPY beta times current SPY return",
            "wealth_transfer_residual": "equal arithmetic mean of current GME and AMC residuals; strict nonzero",
            "magnitude_rank": (
                "strict-prior midrank of absolute wealth_transfer_residual versus at most 270 previous valid sessions; "
                "minimum 180; current excluded; rank>=0.70"
            ),
            "btc_variation": (
                "sqrt(sum squared BTCUSDT 1m log(close/open)) over exact 24h ending at official cash close"
            ),
            "btc_variation_rank": (
                "strict-prior 270/180 midrank over valid sessions; current excluded; rank>=0.65"
            ),
            "missing": (
                "missing common session, adjustment, BTC minute, beta history, zero variance, nonfinite or duplicate data rejects; no imputation"
            ),
        },
        clock={
            "decision": "official common-session cash close plus 5 elapsed minutes",
            "entry": "exact BTCUSDT five-minute open 5 minutes after decision",
            "hold": "24 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium": "not signal inputs; exact funding only after novelty passes",
            "no_imputation": True,
        },
        policy={
            "meme_symbols": list(MEMES),
            "benchmark_symbol": BENCHMARK,
            "beta_prior_sessions": 60,
            "beta_prior_minimum": 40,
            "magnitude_prior_sessions": 270,
            "magnitude_prior_minimum": 180,
            "magnitude_midrank_min": 0.70,
            "variation_prior_sessions": 270,
            "variation_prior_minimum": 180,
            "variation_midrank_min": 0.65,
            "feature_delay_minutes": 5,
            "entry_delay_minutes": 5,
            "hold_hours": 24,
            "gross_exposure": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        source_plan={
            "equities": {
                "provider": "Yahoo chart API current adjusted history",
                "symbols": [*MEMES, BENCHMARK],
                "interval": "1d",
                "window": ["2022-01-01", "2026-08-01"],
                "official_session_validation": "NYSE schedules and frozen early closes",
                "read_after_preregistration": True,
            },
            "btc_1m": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "close"],
                "read_only": True,
            },
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        diagnostic_controls={
            "names": [
                "no_btc_volatility_gate",
                "direction_flip",
                "one_session_stale_residual",
                "raw_equal_mean_meme_return",
                "gme_only_residual",
                "same_clock_forced_long",
            ],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        research_boundary={
            "paper_definition_opened": True,
            "repository_exact_gme_amc_spy_candidate_found": False,
            "yahoo_transport_documentation_opened": True,
            "gme_amc_spy_rows_returns_residual_rank_or_event_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": (
                "paper's meme-stock-to-Bitcoin transfer direction, volatile-state relevance, and exact repository absence"
            ),
        },
        stopping_rule=(
            "terminal first-failure sequence: source support, Gross9 novelty, train/test/eval/final strict economics, "
            "then RV20 q90; no symbols, benchmark, beta, weighting, rank, threshold, side, hold, clock, subset, source, or control repair"
        ),
    )
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value != build() or value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVMWTR preregistration drift")
    if value["outcomes_opened"] is not False or value["source_incidence_opened"] is not False:
        raise RuntimeError("HVMWTR boundary drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
