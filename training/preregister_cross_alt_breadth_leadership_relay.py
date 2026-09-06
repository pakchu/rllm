"""Outcome-blind preregistration for CABLR-8."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT = Path("results/cross_alt_breadth_leadership_relay_preregistration_2026-08-09.json")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "cross_alt_breadth_leadership_relay_v1",
        "policy_id": "CABLR-8", "as_of_date": "2026-08-09",
        "outcomes_opened": False, "source_incidence_opened": False, "singleton": True,
        "mechanism": {
            "claim": "When at least four of six liquid alt perpetuals move together against BTC during a completed high-variation eight-hour block, broad speculative price discovery leads the lagging BTC leg; follow the alt majority for eight hours.",
            "side": "strict majority sign of the six alt completed-block returns",
            "why_distinct": "CABER-12 required BTC-alt confirmation plus larger alt magnitude and faded BTC once daily. CABLR-8 requires cross-alt consensus opposed to BTC and follows the alt majority at three settlement-aligned clocks. AFGI uses price-free flow and a fitted learner; residual and funding families trade alt pairs. CABLR uses only contemporaneous price-sign diffusion into directional BTC.",
            "why_suited_to_volatile_regimes": "completed BTC block realized-variation strict-prior rank must be at least 0.65",
            "why_low_gross9_overlap_is_plausible": "sparse alt-versus-BTC disagreement events differ from Gross9 source triggers despite settlement-aligned decisions",
        },
        "features": {
            "universe": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT"],
            "blocks": "exact bars_binance 1m paths [00:00,08:00), [08:00,16:00), and [16:00,24:00) UTC; all 480 timestamps for all seven symbols",
            "returns": "log(last close/first open) independently per symbol; BTC and confirming alt returns strict nonzero",
            "breadth": "at least four of six alt return signs equal one strict sign",
            "disagreement": "BTC return sign is strict nonzero and opposite the qualifying alt-majority sign",
            "btc_variation": "sqrt(sum squared BTC 1m log(close/open)) over completed block",
            "btc_variation_rank": "strict-prior midrank among at most 810 previous valid blocks; minimum 540; current excluded; rank>=0.65",
            "availability": "block end after all exact paths complete", "no_imputation": True,
        },
        "clock": {"decision": "00:00, 08:00, or 16:00 UTC block end", "entry": "decision+5m BTCUSDT perpetual 5m open", "hold": "8 elapsed hours", "reservation": "global half-open; exit first on equal open", "split_crossing_action": "skip", "gross_exposure": 0.5, "funding": "exact settlements only after novelty passes"},
        "policy": {"history_observations": 810, "minimum_history_observations": 540, "variation_rank_min": 0.65, "breadth_min": 4, "entry_delay_minutes": 5, "hold_hours": 8, "leverage": 0.5, "base_cost_per_notional_side": 0.0006, "stress_cost_per_notional_side": 0.001},
        "stages": {"train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"], "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"], "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"], "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"]},
        "source_support_gates": {"minimum_events": {"train": 8, "test": 12, "eval": 12, "final": 8}, "minority_side_share_min": 0.2, "max_month_share": 0.45},
        "novelty_gates": {"exact_entry_jaccard_max": 0.1, "candidate_near_6h_share_max": 0.35, "occupied_5m_jaccard_max": 0.25, "absolute_signed_exposure_pearson_max": 0.35, "must_pass_before_economics": True},
        "economic_gates": {"absolute_return_positive": True, "cagr_to_strict_mdd_min": 3.0, "strict_mdd_max_pct": 15.0, "mean_gross_underlying_min_bp": 20.0, "weekly_signflip_one_sided_p_max": 0.1, "stress_absolute_return_positive": True, "stress_cagr_to_strict_mdd_min": 2.5, "each_calendar_half_positive": True, "stop_on_first_failure": True, "future_can_rank_repair_or_reselect": False, "accounting": "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, every held 5m favorable then adverse, global HWM, full-calendar CAGR"},
        "source_plan": {"table": "bars_binance", "interval": "1m", "columns": ["ts", "open", "high", "low", "close"], "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT"], "read_only": True, "execution_price": "sealed until source support and Gross9 novelty pass"},
        "diagnostic_controls": {"names": ["no_volatility_gate", "three_of_six_breadth", "btc_confirmation", "one_block_stale_geometry", "direction_flip"], "diagnostic_controls_cannot_be_promoted": True},
        "research_boundary": {"database_metadata_only_opened_before_preregistration": True, "market_values_used_to_select_rule": False, "candidate_incidence_opened": False, "postentry_return_or_pnl_opened": False, "gross9_rows_opened": False, "candidate_count": 1, "grid": False, "repair_of_prior_candidate": False, "promoted_prior_control": False, "selection_basis": "independent cross-alt disagreement diffusion mechanism plus user-required high volatility"},
        "stopping_rule": "terminal first-failure sequence: source support, Gross9 novelty, strict economics; no breadth, disagreement, block, side, hold, timing, volatility, or subset repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(result: dict[str, Any]) -> None:
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    if result.get("manifest_hash") != canonical_hash(core) or result.get("outcomes_opened") is not False or result.get("source_incidence_opened") is not False:
        raise RuntimeError("CABLR preregistration drift")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
