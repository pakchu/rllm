"""Outcome-blind preregistration for HVPASR-24."""
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
    "results/high_volatility_palladium_asymmetric_spillover_relay_preregistration_2026-08-12.json"
)
ASSET = "PALL"
BENCHMARK = "GLD"


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
        protocol_version="high_volatility_palladium_asymmetric_spillover_relay_v1",
        policy_id="HVPASR-24",
        as_of_date="2026-08-12",
        gross9_rows_opened=False,
        mechanism={
            "claim": (
                "Peer-reviewed high-frequency connectedness evidence identifies palladium as the largest net "
                "precious-metal spillover contributor while Bitcoin is a net recipient, with asymmetric transmission. "
                "Follow a sign reversal in the PALL return residual after removing GLD exposure during high BTC variation."
            ),
            "side": "strict sign of current GLD-beta-hedged PALL residual after a strict sign reversal",
            "external_support": {
                "paper": (
                    "Mensi et al. (2020), High-frequency asymmetric volatility connectedness between Bitcoin "
                    "and major precious metals markets, North American Journal of Economics and Finance 52, 101031"
                ),
                "doi": "10.1016/j.najef.2019.101031",
                "reported_fact": (
                    "Palladium is the largest net spillover contributor, Bitcoin is a net recipient, and transmission "
                    "differs between positive and negative semivolatility."
                ),
                "inference_disclosure": (
                    "PALL/GLD ETFs, strict-prior beta residuals, the residual sign-reversal event, BTC variation gate, "
                    "cash-close latency and a 24-hour BTC hold are preregistered adaptations."
                ),
            },
            "why_distinct": (
                "HVPASR isolates palladium-specific shock polarity after removing broad precious-metal exposure. "
                "It is not the prior gold-platinum level ratio, broad equity correlation, crypto flow, derivative, "
                "calendar, mining, or Gross9 primitive."
            ),
            "why_suited_to_volatile_regimes": (
                "Only residual polarity transitions during upper-35% completed BTC variation are admitted."
            ),
            "why_low_gross9_overlap_is_plausible": (
                "Sparse palladium-specific polarity transitions become available only at official US cash closes."
            ),
        },
        features={
            "sessions": (
                "official common US cash sessions on which PALL and GLD both have complete adjusted OHLCV rows"
            ),
            "equity_returns": (
                "for PALL and GLD, log split/dividend-adjusted close_t / adjusted close_(t-1 common session)"
            ),
            "strict_prior_beta": (
                "covariance(PALL_return,GLD_return)/variance(GLD_return) over at most 60 previous valid common-session "
                "pairs, minimum 40, current pair excluded; finite positive GLD variance required"
            ),
            "palladium_residual": "current PALL return minus its strict-prior beta times current GLD return",
            "transition": "current and previous valid-session PALL residuals are strict nonzero and have opposite signs",
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
            "asset_symbol": ASSET,
            "benchmark_symbol": BENCHMARK,
            "beta_prior_sessions": 60,
            "beta_prior_minimum": 40,
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
                "symbols": [ASSET, BENCHMARK],
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
                "one_session_stale_transition",
                "raw_pall_return_transition",
                "residual_level_without_transition",
                "same_clock_forced_long",
            ],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        research_boundary={
            "paper_definition_opened": True,
            "repository_exact_pall_gld_residual_transition_candidate_found": False,
            "yahoo_transport_documentation_opened": True,
            "pall_gld_rows_returns_residual_or_event_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": (
                "paper's palladium net-transmitter result, asymmetric transition mechanism, volatile-state relevance, and exact repository absence"
            ),
        },
        stopping_rule=(
            "terminal first-failure sequence: source support, Gross9 novelty, train/test/eval/final strict economics, "
            "then RV20 q90; no symbols, benchmark, beta, transition, threshold, side, hold, clock, subset, source, or control repair"
        ),
    )
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value != build() or value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVPASR preregistration drift")
    if value["outcomes_opened"] is not False or value["source_incidence_opened"] is not False:
        raise RuntimeError("HVPASR boundary drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
