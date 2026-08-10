"""Outcome-blind preregistration for HVMRSVP-24."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "results/high_volatility_mstr_relative_short_volume_pressure_relay_preregistration_2026-08-10.json"
)


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_mstr_relative_short_volume_pressure_relay_v1",
        "policy_id": "HVMRSVP-24",
        "as_of_date": "2026-08-10",
        "oos_outcomes_opened": False,
        "oos_source_incidence_opened": False,
        "gross9_rows_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "FINRA off-exchange MSTR short-sale execution pressure relative to QQQ isolates "
                "Bitcoin-treasury-equity hedging from broad Nasdaq market-making. During volatile "
                "BTC trading, a sharp completed increase is followed short and a sharp decrease "
                "is followed long for one day. This is short-sale volume, not short interest."
            ),
            "side": "negative strict sign of the one-source-day change in relative short-volume pressure",
            "why_distinct": (
                "Gross9 and prior candidates do not use FINRA Reg SHO MSTR or QQQ daily short-sale "
                "volume; the source, publication clock, and equity-hedging mechanism are new"
            ),
            "volatile_market_target": "strict-prior prior-24h BTC realized-variation rank >=0.65",
            "why_low_gross9_overlap_is_plausible": (
                "events occur only after official US equity reporting days at next-day 00:05 UTC "
                "and require a sparse MSTR-versus-QQQ pressure-change onset"
            ),
        },
        "source_contract": {
            "official_page": "https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data/daily-short-sale-volume-files",
            "official_template": "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt",
            "dataset": "FINRA Consolidated NMS Daily Short Sale Volume",
            "symbols": ["MSTR", "QQQ"],
            "window": ["2023-01-01", "2026-08-01"],
            "schema": ["Date", "Symbol", "ShortVolume", "ShortExemptVolume", "TotalVolume", "Market"],
            "row_validity": (
                "one exact MSTR row and one exact QQQ row per official Consolidated NMS file; "
                "Date equals filename date; finite ShortVolume>=0 and TotalVolume>0; "
                "ShortVolume<=TotalVolume; no duplicate symbol/date"
            ),
            "source_day": (
                "a UTC calendar date with an official HTTP-200 Consolidated NMS file; weekends "
                "and HTTP-404 dates emit no source day; any other HTTP status fails closed"
            ),
            "causal_availability": (
                "FINRA states same-trade-date posting no later than 18:00 ET; decision is fixed "
                "conservatively at 00:00 UTC on the following calendar day and entry at 00:05 UTC"
            ),
            "transport": (
                "download every candidate date directly from official FINRA CDN, retain response "
                "SHA256 and HTTP metadata, parse strict pipe-delimited schema, and hash normalized panel"
            ),
            "revision_boundary": (
                "FINRA may publish later Updated files; official direct-file bytes freeze this replay "
                "vintage and no historical first-seen revision archive is claimed"
            ),
            "license_boundary": "FINRA identifies the data as free for non-commercial use",
            "bounded_probe_disclosure": (
                "before preregistration, official documentation, response headers for four dates, "
                "and the 2026-08-07 header plus web-rendered rows A through AES were opened; no MSTR "
                "or QQQ row, candidate feature, event incidence, BTC outcome, or Gross9 row was read"
            ),
        },
        "feature_contract": {
            "short_volume_share": "ShortVolume divided by TotalVolume separately for MSTR and QQQ",
            "relative_pressure": (
                "logit of MSTR short-volume share minus logit of QQQ share, with each share clipped "
                "only for numerical evaluation to [1e-9,1-1e-9]"
            ),
            "pressure_change": (
                "current relative_pressure minus the immediately previous valid official source-day "
                "relative_pressure; strict nonzero"
            ),
            "absolute_pressure_change_rank": (
                "strict-prior midrank of abs(pressure_change), current excluded, at most 252 valid "
                "source days and minimum 126"
            ),
            "btc_variation": (
                "sum squared close-to-close log returns from the exact 1,440 completed BTCUSDT "
                "bars_binance 1m rows ending at decision time"
            ),
            "btc_variation_rank": "same strict-prior 252/126 source-day rule",
            "eligible_state": (
                "absolute_pressure_change_rank>=0.80, btc_variation_rank>=0.65, and pressure_change nonzero"
            ),
            "onset": "current source day eligible and immediately previous source day ineligible",
            "no_imputation": True,
        },
        "oos_clock": {
            "start": "2023-07-01T00:00:00Z",
            "decision": "00:00 UTC on calendar day after FINRA trade date",
            "entry": "decision+5m BTCUSDT perpetual open",
            "side": "negative sign(pressure_change)",
            "hold": "24 elapsed hours",
            "reservation": (
                "chronological first eligible onset while flat; intervals half-open and exit first "
                "on an equal-time entry"
            ),
            "funding": "opened only after novelty passes",
        },
        "policy": {
            "history_source_days": 252,
            "minimum_history_source_days": 126,
            "absolute_pressure_change_rank_min": 0.80,
            "btc_variation_rank_min": 0.65,
            "entry_delay_minutes": 5,
            "hold_hours": 24,
            "leverage": 0.5,
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
        },
        "stages": {
            "train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
        },
        "source_support_gates": {
            "minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8},
            "minority_side_share_min": 0.20,
            "max_month_share": 0.45,
            "complete_target_pair_per_source_day_required": True,
        },
        "novelty_gates": {
            "exact_entry_jaccard_max": 0.10,
            "candidate_near_6h_share_max": 0.35,
            "occupied_5m_bar_jaccard_max": 0.25,
            "absolute_signed_exposure_pearson_max": 0.35,
            "must_pass_before_economics": True,
        },
        "economic_gates": {
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_min": 3.0,
            "strict_mdd_max_pct": 15.0,
            "mean_gross_underlying_min_bp": 20.0,
            "weekly_signflip_one_sided_p_max": 0.10,
            "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5,
            "each_calendar_half_positive": True,
            "stop_on_first_failure": True,
            "accounting": (
                "fixed quantity, exact funding, 6bp/10bp per notional side, favorable-then-adverse "
                "held 5m path, global HWM, full-calendar CAGR"
            ),
        },
        "post_stage_volatility_audit": {
            "prerequisite": "unchanged passes every sequential economic stage",
            "rv20_q90_entry_filter": False,
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "identical_clock_forced_long_residual_positive": True,
        },
        "diagnostic_controls": {
            "names": [
                "no_volatility_gate", "no_pressure_magnitude_gate", "mstr_share_change_only",
                "one_source_day_stale_features", "direction_flip", "forced_long",
            ],
            "cannot_be_promoted": True,
        },
        "research_boundary": {
            "prior_finra_mstr_qqq_short_volume_outcomes_known": False,
            "candidate_specific_target_rows_opened": False,
            "candidate_specific_incidence_opened": False,
            "candidate_specific_postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": (
                "independent official equity short-sale execution-pressure mechanism fixed before "
                "target-symbol values, incidence, or outcomes"
            ),
        },
        "stopping_rule": (
            "freeze preregistration, source support, Gross9 novelty, and sequential economics; "
            "terminal first failure with no symbol, benchmark, share transform, history, rank, "
            "volatility gate, onset, side, delay, hold, subset, or control repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)
