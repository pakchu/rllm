"""Outcome-blind preregistration for HVECSP-24."""
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


DEFAULT_OUTPUT = Path("results/high_volatility_energy_commodity_return_spillover_relay_preregistration_2026-08-11.json")
USO_YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/USO"
BNO_YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/BNO"
UNG_YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/UNG"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = copy.deepcopy(template.build())
    core.pop("manifest_hash")
    core.update(
        protocol_version="high_volatility_energy_commodity_return_spillover_relay_v1",
        policy_id="HVECSP-24",
        as_of_date="2026-08-11",
        mechanism={
            "claim": (
                "Completed US cash-session energy-commodity returns identify a signed cross-market impulse: "
                "published evidence maps higher natural-gas returns to higher future Bitcoin returns and higher "
                "WTI/Brent returns to lower future Bitcoin returns. During elevated causal BTC variation, follow "
                "the signed gas-minus-oil impulse for twenty-four hours."
            ),
            "side": "sign of log(UNG close/open)-0.5*(log(USO close/open)+log(BNO close/open)); zero is ineligible",
            "external_support": {
                "paper": (
                    "Derbali, Jamel, Ben Ltaifa, and Elnagar (2020), Return, Volatility and Shock Spillovers "
                    "of Bitcoin with Energy Commodities"
                ),
                "journal": "International Journal of Finance, Insurance and Risk Management 10(3), 157-170",
                "primary_url": (
                    "https://journalfirm.com/journal/228/download/Return%2C%2BVolatility%2Band%2BShock%2B"
                    "Spillovers%2Bof%2BBitcoin%2B%2Bwith%2BEnergy%2BCommodities.pdf"
                ),
                "paper_fixed_facts": [
                    "the paper studies return, volatility, and shock spillovers between Bitcoin, WTI, Brent, and natural gas",
                    "past energy-commodity returns are reported to spill unilaterally into Bitcoin returns",
                    "higher past WTI and Brent returns predict lower Bitcoin returns",
                    "higher past natural-gas returns predict higher Bitcoin returns",
                ],
                "implementation_choices_not_claimed_as_replication": [
                    "USO, BNO, and UNG as liquid investable WTI, Brent, and natural-gas proxies",
                    "same-session raw open-to-close log returns rather than the paper's daily futures-price VAR",
                    "equal-weight WTI/Brent oil leg and a signed gas-minus-oil score",
                    "absolute causal z-score threshold of 0.75",
                    "causal BTC realized-variation rank gate of 0.65",
                    "24-hour BTC hold beginning ten minutes after the actual NYSE close",
                ],
            },
            "why_distinct": (
                "Repository collision scans found no WTI/Brent/natural-gas, USO/BNO/UNG, or signed gas-minus-oil "
                "Bitcoin candidate. The terminal HVETSR candidate used a same-sign energy-plus-technology equity "
                "factor; this independently sourced commodity mechanism freezes opposing published gas and oil "
                "directions and is not a predictor replacement or repair of HVETSR."
            ),
            "why_suited_to_volatile_regimes": (
                "The paper models volatility and shock spillovers across energy commodities and Bitcoin; the frozen "
                "implementation requires prior-24-hour BTC realized-variation rank >= 0.65."
            ),
            "why_low_gross9_overlap_is_plausible": (
                "A completed cash-session signed natural-gas-versus-oil commodity shock is absent from Gross9 primitives."
            ),
        },
        features={
            "uso_intraday_return": "log(raw unadjusted USO close/open) on the completed common session",
            "bno_intraday_return": "log(raw unadjusted BNO close/open) on the completed common session",
            "ung_intraday_return": "log(raw unadjusted UNG close/open) on the completed common session",
            "oil_return": "0.5*(uso_intraday_return+bno_intraday_return)",
            "spillover_score": "ung_intraday_return-oil_return",
            "spillover_score_z": (
                "causal z-score against at most 252 previous finite common-session scores; minimum 126; "
                "current excluded; population standard deviation; zero variance is ineligible"
            ),
            "event": "absolute spillover_score_z >= 0.75; side is sign(spillover_score)",
            "btc_variation": (
                "sqrt(sum squared log(close/open)) over 1,440 exact BTCUSDT 1m bars ending at but excluding "
                "the actual NYSE close"
            ),
            "btc_variation_rank": (
                "strict-prior midrank against at most 252 prior valid common-session variations; minimum 126; "
                "current excluded; rank >= 0.65"
            ),
            "common_rows": "exact common USO/BNO/UNG NYSE dates with finite positive raw opens/closes; no imputation",
        },
        clock={
            "source_session": "actual NYSE regular-session close for each common USO/BNO/UNG date, including early closes",
            "feature_available": "five minutes after the actual NYSE close",
            "entry": "exact BTCUSDT five-minute open ten minutes after the actual NYSE close",
            "hold": "24 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium": "not signal inputs; exact funding only after novelty passes",
            "no_imputation": True,
        },
        policy={
            "commodity_weights": {"USO": -0.5, "BNO": -0.5, "UNG": 1.0},
            "spillover_score_z_prior_sessions": 252,
            "spillover_score_z_prior_minimum": 126,
            "spillover_score_abs_z_min": 0.75,
            "variation_prior_sessions": 252,
            "variation_prior_minimum": 126,
            "variation_midrank_min": 0.65,
            "feature_delay_minutes": 5,
            "entry_delay_minutes_after_feature": 5,
            "hold_hours": 24,
            "gross_exposure": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        source_plan={
            "energy_commodity_proxies": {
                "urls": {"USO": USO_YAHOO_URL, "BNO": BNO_YAHOO_URL, "UNG": UNG_YAHOO_URL},
                "fields": ["timestamp", "open", "high", "low", "close", "volume"],
                "raw_unadjusted_only": True,
                "download_after_preregistration": True,
            },
            "btc_1m": {
                "table": "bars_binance", "symbol": "BTCUSDT", "interval": "1m",
                "columns": ["ts", "open", "close"], "read_only": True,
            },
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        diagnostic_controls={
            "names": [
                "no_btc_volatility_gate", "inverse_oil_only", "natural_gas_only",
                "one_session_stale_score", "direction_flip",
            ],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        research_boundary={
            "source_schema_and_transport_checked": False,
            "source_values_used_to_select_rule": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "repository_energy_commodity_spillover_candidate_found": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "primary published directional energy-commodity-to-Bitcoin evidence plus exact repository absence",
        },
        stopping_rule=(
            "terminal first-failure sequence: source support, Gross9 novelty, train/test/eval/final strict economics, "
            "then RV20 q90 audit; no commodity proxy, predictor, weighting, threshold, side, hold, clock, subset, "
            "or control repair"
        ),
    )
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVECSP preregistration drift")
    if value["outcomes_opened"] is not False or value["source_incidence_opened"] is not False:
        raise RuntimeError("HVECSP boundary drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
