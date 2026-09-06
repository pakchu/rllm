"""Outcome-blind preregistration for SERBR-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT = Path("results/sterling_euro_risk_beta_relay_preregistration_2026-08-09.json")


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "sterling_euro_risk_beta_relay_v1",
        "policy_id": "SERBR-12",
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "Sterling is the higher-beta European currency while the euro is the regional "
                "benchmark. Their completed London-session relative return cancels the common "
                "US-dollar component and measures a fresh European risk-beta impulse. GBP "
                "outperformance maps long BTC and underperformance maps short BTC, only while "
                "BTC is already in a high realized-variation regime."
            ),
            "side": "sign of completed GBPUSD log return minus completed EURUSD log return",
            "external_support": {
                "economic_channel": (
                    "GBP commonly carries greater cyclical and global-risk sensitivity than EUR; "
                    "the relative cross removes their shared dollar move"
                ),
                "implementation_is_not_a_published_replication": True,
                "untested_adaptation": (
                    "the completed London relative move is tested as a twelve-hour BTC relay"
                ),
            },
            "why_distinct": (
                "No prior repository alpha uses GBPUSD-minus-EURUSD relative session return. "
                "HVDBR used six-pair common-dollar breadth and was terminally rejected at source "
                "support; SERBR removes the common dollar factor and neither reuses its event set "
                "nor promotes a control. MXRBR used one emerging-market currency."
            ),
            "why_suited_to_volatile_regimes": (
                "BTC prior-24h realized variation must rank at least 0.65 causally"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "weekday 16:05 London entries conditioned on external relative FX are absent from "
                "Gross9 primitives"
            ),
        },
        "features": {
            "fx_session": (
                "exact GBPUSD and EURUSD Polygon 1m bars [08:00,16:00) Europe/London on each "
                "Monday-Friday; all 480 timestamps required for both symbols"
            ),
            "relative_return": (
                "log(GBPUSD close at 15:59/open at 08:00) minus log(EURUSD close at 15:59/open "
                "at 08:00); zero is ineligible"
            ),
            "availability": "16:00 Europe/London after both completed source paths",
            "btc_variation": (
                "sqrt(sum squared log(close/open)) over exact BTCUSDT 1m bars in the prior 24 "
                "elapsed hours ending at decision"
            ),
            "btc_variation_rank": (
                "strict-prior midrank against at most 252 previous valid weekday decision "
                "variations; minimum 126; current excluded; rank>=0.65"
            ),
            "missing_duplicate_or_nonpositive": "ineligible or source failure; no imputation",
        },
        "clock": {
            "decision": (
                "each Monday-Friday 16:00 Europe/London with complete FX and BTC source paths"
            ),
            "entry": "exact 16:05 Europe/London BTCUSDT 5m open",
            "hold": "12 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding_oi_premium": "not signal inputs; exact funding only after novelty passes",
            "no_imputation": True,
        },
        "stages": {
            "train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
        },
        "source_support_gates": {
            "minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8},
            "minority_side_share_min": 0.2,
            "max_month_share": 0.45,
        },
        "novelty_gates": {
            "exact_entry_jaccard_max": 0.1,
            "candidate_near_6h_share_max": 0.35,
            "occupied_5m_jaccard_max": 0.25,
            "absolute_signed_exposure_pearson_max": 0.35,
            "must_pass_before_economics": True,
        },
        "economic_gates": {
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_min": 3.0,
            "strict_mdd_max_pct": 15.0,
            "mean_gross_underlying_min_bp": 20.0,
            "weekly_signflip_one_sided_p_max": 0.1,
            "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5,
            "each_calendar_half_positive": True,
            "stop_on_first_failure": True,
            "future_can_rank_repair_or_reselect": False,
            "accounting": (
                "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, "
                "every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "source_plan": {
            "fx": {
                "table": "bars_polygon",
                "symbols": ["GBPUSD", "EURUSD"],
                "interval": "1m",
                "columns": ["ts", "open", "close"],
                "read_only": True,
            },
            "btc": {
                "table": "bars_binance",
                "symbol": "BTCUSDT",
                "interval": "1m",
                "columns": ["ts", "open", "close"],
                "read_only": True,
            },
            "execution_price": "sealed until source support and Gross9 novelty pass",
        },
        "diagnostic_controls": {
            "names": [
                "no_volatility_gate",
                "common_dollar_basket",
                "one_session_stale_relative_return",
                "direction_flip",
            ],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        "research_boundary": {
            "database_metadata_only_opened_before_preregistration": True,
            "fx_values_used_to_select_rule": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": (
                "independent common-dollar-neutral European risk-beta channel plus user-required "
                "high volatility"
            ),
        },
        "stopping_rule": (
            "terminal first-failure sequence: source support, Gross9 novelty, strict economics; "
            "no session, threshold, side, hold, timing, volatility, or subset repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(registration: dict[str, Any]) -> None:
    core = {key: value for key, value in registration.items() if key != "manifest_hash"}
    if (
        registration.get("manifest_hash") != canonical_hash(core)
        or registration.get("outcomes_opened") is not False
        or registration.get("source_incidence_opened") is not False
    ):
        raise RuntimeError("SERBR preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(args.output)
