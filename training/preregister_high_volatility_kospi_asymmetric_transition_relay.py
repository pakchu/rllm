"""Outcome-blind preregistration for HVKATR-24."""
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
    "results/high_volatility_kospi_asymmetric_transition_relay_preregistration_2026-08-12.json"
)
KOSPI_YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/%5EKS11"


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
        protocol_version="high_volatility_kospi_asymmetric_transition_relay_v1",
        policy_id="HVKATR-24",
        as_of_date="2026-08-12",
        gross9_rows_opened=False,
        mechanism={
            "claim": (
                "Peer-reviewed connectedness evidence identifies the Korean stock market as a significant "
                "volatility emitter to Bitcoin in an asymmetric East-Asian spillover system. Follow the current "
                "KOSPI return direction when a large completed-session return reverses the prior session's sign "
                "during elevated completed BTC variation."
            ),
            "side": "strict sign of the current KOSPI close-to-close return after a strict sign transition",
            "external_support": {
                "paper": (
                    "Zeng (2023), Volatility spillover effect between Bitcoin and the East Asian stock market, "
                    "International Journal of Managerial Finance"
                ),
                "doi": "10.1108/IJMF-03-2021-0161",
                "reported_fact": (
                    "The study reports two-way asymmetric spillovers and identifies the Korean market as a "
                    "significant volatility emitter in the Bitcoin/East-Asian stock system."
                ),
                "inference_disclosure": (
                    "The Yahoo KOSPI proxy, strict-prior absolute-return rank, sign-transition event, BTC variation "
                    "gate, five-minute publication buffer and 24-hour BTC hold are preregistered adaptations."
                ),
            },
            "why_distinct": (
                "HVKATR uses the Korean cash equity index at its native 15:30 Asia/Seoul close. Existing Korean "
                "candidates use Upbit/Binance crypto microstructure, premium, volume or venue leadership; the old "
                "cross-asset portability battery trades KODEX 200 and explicitly does not use KOSPI as a BTC signal."
            ),
            "why_suited_to_volatile_regimes": (
                "Both a top-40% KOSPI absolute-return shock and upper-35% completed BTC variation are required."
            ),
            "why_low_gross9_overlap_is_plausible": (
                "Sparse KOSPI polarity transitions are released at the native 06:30 UTC cash close, a primitive and "
                "clock absent from Gross9."
            ),
        },
        features={
            "sessions": "Yahoo ^KS11 completed exchange sessions, dated in Asia/Seoul",
            "kospi_return": "log(close_t / close_(t-1 observed exchange session)), strict nonzero",
            "kospi_shock_rank": (
                "strict-prior midrank of abs(KOSPI return) over at most 252 valid sessions, minimum 126, current "
                "excluded; rank>=0.60"
            ),
            "transition": "current and previous valid-session KOSPI returns are strict nonzero and opposite in sign",
            "btc_variation": (
                "sqrt(sum squared BTCUSDT 1m log(close/open)) over the exact 24 hours ending at 15:30 Asia/Seoul"
            ),
            "btc_variation_rank": (
                "strict-prior midrank over at most 270 valid KOSPI decisions, minimum 180, current excluded; rank>=0.65"
            ),
            "availability": (
                "the completed KOSPI daily bar is treated as available five elapsed minutes after the native close"
            ),
            "missing": (
                "duplicate, weekend, nonpositive, nonfinite or implausibly gapped KOSPI sessions and missing BTC "
                "minutes reject; no imputation"
            ),
        },
        clock={
            "decision": "KOSPI native 15:30 Asia/Seoul cash close plus 5 elapsed minutes",
            "entry": "exact BTCUSDT five-minute open 5 minutes after decision (normally 06:40 UTC)",
            "hold": "24 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium": "not signal inputs; exact funding only after novelty passes",
            "no_imputation": True,
        },
        policy={
            "asset_symbol": "^KS11",
            "asset_timezone": "Asia/Seoul",
            "cash_close_local": "15:30",
            "shock_prior_sessions": 252,
            "shock_prior_minimum": 126,
            "shock_midrank_min": 0.60,
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
            "equity": {
                "provider": "Yahoo chart API current KOSPI history",
                "symbol": "^KS11",
                "interval": "1d",
                "window": ["2021-01-01", "2026-08-01"],
                "exchange_timezone": "Asia/Seoul",
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
                "no_kospi_shock_gate",
                "one_session_stale_transition",
                "return_level_without_transition",
                "direction_flip",
                "same_clock_forced_long",
            ],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        research_boundary={
            "paper_definition_opened": True,
            "repository_exact_kospi_btc_transition_candidate_found": False,
            "yahoo_transport_schema_only_opened": True,
            "kospi_rows_returns_ranks_or_event_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": (
                "published Korean-market volatility-emitter result, asymmetric transition mechanism, volatile-state "
                "relevance, native-close novelty and exact repository absence"
            ),
        },
        stopping_rule=(
            "terminal first-failure sequence: source support, Gross9 novelty, train/test/eval/final strict economics, "
            "then RV20 q90; no source, market, threshold, transition, side, hold, clock, subset or control repair"
        ),
    )
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value != build() or value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVKATR preregistration drift")
    if value["outcomes_opened"] is not False or value["source_incidence_opened"] is not False:
        raise RuntimeError("HVKATR boundary drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
