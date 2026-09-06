"""Outcome-blind preregistration for HVMELR-24."""
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
    "results/high_volatility_mining_equity_leadership_relay_preregistration_2026-08-12.json"
)
MINERS = ("RIOT", "MARA", "HUT")


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
        protocol_version="high_volatility_mining_equity_leadership_relay_v1",
        policy_id="HVMELR-24",
        as_of_date="2026-08-12",
        gross9_rows_opened=False,
        mechanism={
            "claim": (
                "Peer-reviewed connectedness evidence identifies RIOT, MARA and HUT as typical net "
                "spillover transmitters and finds stronger crypto-mining-stock connectedness in turbulent "
                "periods. Follow the sign of their equal-weight, BTC-beta-hedged close shock into the "
                "subsequent BTC holding interval only when that shock and completed BTC variation are high."
            ),
            "side": "strict sign of the equal-weight current miner beta-hedged residual",
            "external_support": {
                "paper": (
                    "Selmi et al. (2024), Environmental attention and uncertainties of cryptocurrency "
                    "market: Examining linkages with crypto-mining stocks, Finance Research Letters 59, 104672"
                ),
                "doi": "10.1016/j.frl.2023.104672",
                "reported_fact": (
                    "RIOT, MARA and HUT are typically net spillover transmitters, and mining-stock return "
                    "connectedness increases during turbulent periods."
                ),
                "inference_disclosure": (
                    "Daily adjusted returns, strict-prior rolling BTC betas, equal residual weighting, "
                    "rank gates, exact close latency and a 24-hour BTC hold are preregistered adaptations."
                ),
            },
            "why_distinct": (
                "HVMELR isolates issuer-specific shocks in three listed proof-of-work operators. It is not "
                "a broad stock-correlation, MSTR short-volume, on-chain miner, hash-rate, energy-sector, "
                "derivatives, order-flow, or Gross9 primitive."
            ),
            "why_suited_to_volatile_regimes": (
                "Only upper-35% completed BTC variation and upper-20% absolute miner residuals are admitted."
            ),
            "why_low_gross9_overlap_is_plausible": (
                "Sparse issuer-specific shocks become available only at official US miner-equity closes."
            ),
        },
        features={
            "sessions": (
                "official common US cash sessions on which RIOT, MARA and HUT all have complete adjusted OHLCV rows"
            ),
            "miner_returns": (
                "for each miner, log split/dividend-adjusted close_t / adjusted close_(t-1 common session)"
            ),
            "btc_returns": (
                "log BTC close at current official US cash close / BTC close at previous common-session close"
            ),
            "strict_prior_beta": (
                "for each miner, covariance(miner_return,BTC_return)/variance(BTC_return) over at most 60 "
                "previous valid common-session pairs, minimum 40, current pair excluded; finite positive BTC variance required"
            ),
            "miner_residual": "current miner_return minus its strict-prior beta times current BTC_return",
            "leadership_residual": "equal arithmetic mean of the three current miner residuals; strict nonzero",
            "magnitude_rank": (
                "strict-prior midrank of absolute leadership_residual versus at most 270 previous valid sessions; "
                "minimum 180; current excluded; rank>=0.80"
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
            "miner_symbols": list(MINERS),
            "beta_prior_sessions": 60,
            "beta_prior_minimum": 40,
            "magnitude_prior_sessions": 270,
            "magnitude_prior_minimum": 180,
            "magnitude_midrank_min": 0.80,
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
            "miners": {
                "provider": "Yahoo chart API current adjusted history",
                "symbols": list(MINERS),
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
                "raw_equal_mean_miner_return",
                "mara_only_residual",
                "same_clock_forced_long",
            ],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        research_boundary={
            "paper_definition_opened": True,
            "repository_exact_riot_mara_hut_candidate_found": False,
            "yahoo_transport_schema_date_row_coverage_preflight_opened": True,
            "preflight_observation": (
                "two immediate normalized downloads per RIOT/MARA/HUT returned 1148 rows each and were byte-identical"
            ),
            "miner_return_residual_rank_or_event_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": (
                "paper's named net-transmitter miners, volatile-state relevance, reproducible source preflight, and exact repository absence"
            ),
        },
        stopping_rule=(
            "terminal first-failure sequence: source support, Gross9 novelty, train/test/eval/final strict economics, "
            "then RV20 q90; no symbols, beta, weighting, rank, threshold, side, hold, clock, subset, source, or control repair"
        ),
    )
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value != build() or value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVMELR preregistration drift")
    if value["outcomes_opened"] is not False or value["source_incidence_opened"] is not False:
        raise RuntimeError("HVMELR boundary drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
