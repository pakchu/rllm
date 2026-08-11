"""Outcome-blind preregistration for HVEBPR-24."""
from __future__ import annotations

import argparse, copy, hashlib, json
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import preregister_high_volatility_energy_technology_spillover_relay as template

DEFAULT_OUTPUT = Path("results/high_volatility_equity_breadth_participation_relay_preregistration_2026-08-12.json")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    core = copy.deepcopy(template.build()); core.pop("manifest_hash")
    core.update(
        protocol_version="high_volatility_equity_breadth_participation_relay_v1",
        policy_id="HVEBPR-24", as_of_date="2026-08-12", gross9_rows_opened=False,
        mechanism={
            "claim": (
                "RSP and SPY track the same S&P 500 stock universe under equal weighting and float-adjusted "
                "market-cap weighting respectively. An unusually large completed RSP-minus-SPY return therefore "
                "isolates broad constituent participation from mega-cap index direction. During elevated completed "
                "BTC variation, follow that broad risk-participation direction for twenty-four hours."
            ),
            "side": "strict sign of the completed adjusted-close RSP return minus adjusted-close SPY return",
            "external_support": {
                "rsp_official": "https://www.invesco.com/us/financial-products/etfs/product-detail?productId=RSP",
                "rsp_definition": "RSP is based on the S&P 500 Equal Weight Index",
                "spy_official": "https://www.ssga.com/us/en/individual/etfs/state-street-spdr-sp-500-etf-trust-spy",
                "spy_definition": "SPY tracks the float-adjusted market-capitalization-weighted S&P 500 Index",
                "inference_disclosure": "the relative-return tail, BTC variation gate, direction, latency and hold are untested preregistered adaptations",
            },
            "why_distinct": (
                "No repository candidate uses RSP or the equal-weight-versus-cap-weight S&P 500 spread. Existing "
                "equity candidates use sector returns, regional correlations, single-index volatility, meme/miner "
                "residuals or KOSPI; HVEBPR measures same-universe US equity participation breadth."
            ),
            "why_suited_to_volatile_regimes": "the breadth shock and completed BTC variation must both lie in causal upper tails",
            "why_low_gross9_overlap_is_plausible": "a sparse official US cash-close same-universe breadth shock is absent from Gross9 primitives",
        },
        features={
            "sessions": "official common US cash sessions with complete adjusted RSP and SPY OHLCV rows",
            "returns": "log(adjusted_close_t/adjusted_close_(t-1 common session)) independently for RSP and SPY",
            "breadth_return": "RSP adjusted return minus SPY adjusted return, strict nonzero",
            "breadth_shock_rank": "strict-prior midrank of abs(breadth_return) over at most 252 valid common sessions, minimum 126, current excluded; rank>=0.70",
            "btc_variation": "sqrt(sum squared BTCUSDT 1m log(close/open)) over exact 24h ending at official cash close",
            "btc_variation_rank": "strict-prior midrank over at most 270 valid sessions, minimum 180, current excluded; rank>=0.65",
            "missing": "missing common session, adjustment, BTC minute, duplicate, nonfinite or nonpositive data rejects; no imputation",
        },
        clock={
            "decision": "official common-session cash close plus 5 elapsed minutes",
            "entry": "exact BTCUSDT five-minute open 5 minutes after decision",
            "hold": "24 elapsed hours", "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip", "gross_exposure": 0.5,
            "funding_oi_premium": "not signal inputs; exact funding only after novelty passes", "no_imputation": True,
        },
        policy={
            "breadth_symbol": "RSP", "cap_weight_symbol": "SPY",
            "breadth_prior_sessions": 252, "breadth_prior_minimum": 126, "breadth_midrank_min": 0.70,
            "variation_prior_sessions": 270, "variation_prior_minimum": 180, "variation_midrank_min": 0.65,
            "feature_delay_minutes": 5, "entry_delay_minutes": 5, "hold_hours": 24,
            "gross_exposure": 0.5, "base_cost_per_notional_side": 0.0006, "stress_cost_per_notional_side": 0.001,
        },
        source_plan={
            "equities": {"provider": "Yahoo chart API current adjusted history", "symbols": ["RSP", "SPY"], "interval": "1d", "window": ["2021-01-01", "2026-08-01"], "official_session_validation": "NYSE schedules and frozen early closes", "read_after_preregistration": True},
            "btc_1m": {"table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m", "columns": ["ts", "open", "close"], "read_only": True},
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        diagnostic_controls={
            "names": ["no_btc_volatility_gate", "no_breadth_shock_gate", "rsp_return_only", "one_session_stale_breadth", "direction_flip", "same_clock_forced_long"],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        research_boundary={
            "official_product_definitions_opened": True, "repository_exact_rsp_spy_breadth_candidate_found": False,
            "yahoo_transport_documentation_opened": True, "rsp_spy_rows_returns_ranks_or_event_incidence_opened": False,
            "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "candidate_count": 1,
            "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False,
            "selection_basis": "official same-universe weighting contrast, volatile-state relevance and exact repository absence",
        },
        stopping_rule="terminal first-failure sequence: source support, Gross9 novelty, train/test/eval/final strict economics, then RV20 q90; no symbol, spread, threshold, side, hold, clock, subset, source or control repair",
    )
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {k: v for k, v in value.items() if k != "manifest_hash"}
    if value != build() or value.get("manifest_hash") != canonical_hash(core): raise RuntimeError("HVEBPR preregistration drift")
    if value["outcomes_opened"] is not False or value["source_incidence_opened"] is not False: raise RuntimeError("HVEBPR boundary drift")


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); a=p.parse_args()
    r=build(); validate(r); a.output.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+"\n"); print(a.output)
