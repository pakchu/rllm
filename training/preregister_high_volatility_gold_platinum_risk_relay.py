"""Outcome-blind preregistration for HVAUPTR-24."""
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


DEFAULT_OUTPUT = Path("results/high_volatility_gold_platinum_risk_relay_preregistration_2026-08-11.json")
GLD_YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/GLD"
PPLT_YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/PPLT"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = copy.deepcopy(template.build())
    core.pop("manifest_hash")
    core.update(
        protocol_version="high_volatility_gold_platinum_risk_relay_v1",
        policy_id="HVAUPTR-24",
        as_of_date="2026-08-11",
        mechanism={
            "claim": (
                "A completed US cash-session increase in gold relative to platinum identifies an aggregate-risk "
                "impulse that published evidence maps to higher future Bitcoin returns. During elevated causal "
                "BTC variation, follow the signed gold/platinum impulse for twenty-four hours."
            ),
            "side": "sign of log(GLD close/open)-log(PPLT close/open); zero is ineligible",
            "external_support": {
                "paper": "Huynh, Burggraf, and Wang (2020), Gold, platinum, and expected Bitcoin returns",
                "doi": "10.1016/j.mulfin.2020.100628",
                "paper_fixed_facts": [
                    "the gold-to-platinum price ratio is examined as an aggregate-market-risk state variable",
                    "the ratio predicts future Bitcoin returns across models and Bitcoin data sources",
                    "when gold rises relative to platinum, future Bitcoin return rises",
                    "gold and platinum volatility can influence Bitcoin volatility",
                ],
                "implementation_choices_not_claimed_as_replication": [
                    "GLD and PPLT as liquid investable metal-price proxies",
                    "same-session raw open-to-close log relative return rather than the paper's ratio-level regressions",
                    "absolute causal z-score threshold of 0.75",
                    "causal BTC realized-variation rank gate of 0.65",
                    "24-hour BTC hold beginning ten minutes after the actual NYSE close",
                ],
            },
            "why_distinct": (
                "Repository collision scans found no gold/platinum, GLD/PPLT, PPLT, XPT, or precious-metal-ratio "
                "candidate. Existing gold candidates compare gold with QQQ or implied gold/oil volatility and do "
                "not use platinum or the published aggregate-risk ratio channel."
            ),
            "why_suited_to_volatile_regimes": (
                "The paper reports precious-metal volatility transmission into Bitcoin; the frozen implementation "
                "requires prior-24-hour BTC realized-variation rank >= 0.65."
            ),
            "why_low_gross9_overlap_is_plausible": (
                "A completed cash-session gold-versus-platinum shock is absent from Gross9 primitives."
            ),
        },
        features={
            "gld_intraday_return": "log(raw unadjusted GLD close/open) on the completed common session",
            "pplt_intraday_return": "log(raw unadjusted PPLT close/open) on the completed common session",
            "ratio_impulse": "gld_intraday_return-pplt_intraday_return",
            "ratio_impulse_z": (
                "causal z-score against at most 252 previous finite common-session impulses; minimum 126; "
                "current excluded; population standard deviation; zero variance is ineligible"
            ),
            "event": "absolute ratio_impulse_z >= 0.75; side is sign(ratio_impulse)",
            "btc_variation": (
                "sqrt(sum squared log(close/open)) over 1,440 exact BTCUSDT 1m bars ending at but excluding "
                "the actual NYSE close"
            ),
            "btc_variation_rank": (
                "strict-prior midrank against at most 252 prior valid common-session variations; minimum 126; "
                "current excluded; rank >= 0.65"
            ),
            "common_rows": "exact common GLD/PPLT NYSE dates with finite positive raw opens/closes; no imputation",
        },
        clock={
            "source_session": "actual NYSE regular-session close for each common GLD/PPLT date, including early closes",
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
            "ratio_impulse_z_prior_sessions": 252,
            "ratio_impulse_z_prior_minimum": 126,
            "ratio_impulse_abs_z_min": 0.75,
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
            "metals": {
                "urls": {"GLD": GLD_YAHOO_URL, "PPLT": PPLT_YAHOO_URL},
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
            "names": ["no_btc_volatility_gate", "gld_only", "inverse_pplt_only", "one_session_stale_impulse", "direction_flip"],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        research_boundary={
            "source_schema_and_transport_checked": False,
            "source_values_used_to_select_rule": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "repository_gold_platinum_candidate_found": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "primary published positive gold/platinum-to-Bitcoin predictability plus exact repository absence",
        },
        stopping_rule=(
            "terminal first-failure sequence: source support, Gross9 novelty, train/test/eval/final strict economics, "
            "then RV20 q90 audit; no metal proxy, threshold, side, hold, clock, subset, or control repair"
        ),
    )
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(value: dict[str, Any]) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core) or value != build():
        raise RuntimeError("HVAUPTR preregistration drift")
    if value["outcomes_opened"] is not False or value["source_incidence_opened"] is not False:
        raise RuntimeError("HVAUPTR boundary drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
