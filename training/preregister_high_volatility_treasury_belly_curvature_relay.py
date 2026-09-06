"""Outcome-blind preregistration for HVTBCR-24."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POLICY_ID = "HVTBCR-24"
DEFAULT_OUTPUT = Path("results/high_volatility_treasury_belly_curvature_relay_preregistration_2026-08-10.json")
SOURCE_DIR = Path("data/treasury_parallel_yield_shock_relay_sources_2023_2026/official_xml")
SOURCE_HASHES = {
    "daily_treasury_yield_curve_2023.xml": "9260f47cc662f161931792254e25bcfe35588731f8abce85307b2ecff7033c81",
    "daily_treasury_yield_curve_2024.xml": "3d77ebdb32fba00bf4f2211af3ea8924f275518c00f2faa53698b9336d180c36",
    "daily_treasury_yield_curve_2025.xml": "ac0a8ea1a70b6a22e05aa52513188076b2eddfde5e340bd17db71b48cb3df5a5",
    "daily_treasury_yield_curve_2026.xml": "4ef6e46360173d960a9f01c3dd6690b914e86b3d52635e4e877b48f377434c3a",
}
SOURCE_MANIFEST = Path("data/treasury_parallel_yield_shock_relay_sources_2023_2026/manifest.json")
SOURCE_MANIFEST_SHA = "fb6db916c42724cc3b74e8aef262d422c651537fd9d835f379b233ce2c80b9fd"
MARKET = Path("data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz")
MARKET_SHA = "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_treasury_belly_curvature_relay_v1",
        "policy_id": POLICY_ID,
        "as_of_date": "2026-08-10",
        "singleton": True,
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "gross9_rows_opened": False,
        "mechanism": {
            "claim": (
                "A daily change in the official 2s5s10s Treasury butterfly isolates belly "
                "repricing from level and end-point slope moves. Rising belly curvature is "
                "intermediate-horizon discount-rate tightening and maps short BTC; falling "
                "curvature maps long BTC for one day when completed BTC variation is elevated."
            ),
            "side": "negative strict sign of the daily change in 2*5y-2y-10y",
            "why_distinct": (
                "TPYSR used the same-sign 2y/10y level factor and HVTCTR used only opposite-sign "
                "2y/10y twists. HVTBCR introduces the official 5y belly and the three-point "
                "curvature factor; it reuses no event set or diagnostic control."
            ),
            "literature_context": {
                "yield_curve_factors": (
                    "Litterman and Scheinkman (1991), Common Factors Affecting Bond Returns, "
                    "DOI 10.3905/jfi.1991.692347: level, steepness and curvature are distinct factors"
                ),
                "bitcoin_monetary_policy": (
                    "Kang, Ratti and Vespignani (2023), Monetary policy and Bitcoin, "
                    "DOI 10.1016/j.jimonfin.2023.102880"
                ),
                "implementation_is_not_a_published_replication": True,
            },
            "why_suited_to_volatile_regimes": (
                "completed seven-day BTC realized variation must rank in its causal upper 35%"
            ),
            "why_low_gross9_overlap_is_plausible": (
                "a sparse next-day official sovereign-belly curvature clock is absent from Gross9"
            ),
        },
        "features": {
            "source": "hash-bound official US Treasury Daily Treasury Par Yield Curve Rates XML",
            "transition": "consecutive valid Treasury observations one to five calendar days apart",
            "curvature": "2*yield_5y-yield_2y-yield_10y",
            "curvature_change": "current curvature minus immediately previous valid curvature",
            "eligible": "finite nonzero curvature_change",
            "direction": "-sign(curvature_change)",
            "availability": "source day D plus one calendar day at 12:00 UTC",
            "btc_variation": (
                "sqrt(sum squared exact completed 5m BTC close/open log returns over seven elapsed days ending at decision)"
            ),
            "variation_rank": (
                "strict-prior midrank over at most 270 daily 12:00 UTC states, minimum 180; "
                "current excluded; rank>=0.65"
            ),
            "missing_duplicate_nonpositive": "ineligible; no imputation",
        },
        "clock": {
            "decision": "D+1 12:00 UTC",
            "entry": "decision+5m exact BTCUSDT open",
            "hold": "24 elapsed hours",
            "reservation": "global half-open; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
            "funding": "not a signal input; exact settlements only after novelty",
        },
        "policy": {
            "variation_history_days": 270,
            "minimum_history_days": 180,
            "variation_rank_min": 0.65,
            "availability_delay_hours": 36,
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
            "minority_side_share_min": 0.2,
            "max_month_share": 0.45,
        },
        "novelty_gates": {
            "exact_entry_jaccard_max": 0.1,
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
            "weekly_signflip_one_sided_p_max": 0.1,
            "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5,
            "each_calendar_half_positive": True,
            "stop_on_first_failure": True,
            "accounting": (
                "fixed quantity, exact funding, 6bp base and 10bp stress per notional side, "
                "every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "post_stage_volatility_audit": {
            "prerequisite": "unchanged candidate passes train, test, eval and final",
            "rv20_q90_entry_filter": False,
            "minimum_q90_trades": 8,
            "candidate_q90_absolute_return_positive": True,
            "identical_clock_forced_long_residual_positive": True,
        },
        "diagnostic_controls": {
            "names": [
                "no_variation_gate",
                "curvature_level",
                "one_observation_stale_curvature_change",
                "direction_flip",
                "same_clock_forced_long",
            ],
            "cannot_be_promoted": True,
        },
        "source_plan": {
            "treasury_xml": {
                str(SOURCE_DIR / name): digest for name, digest in SOURCE_HASHES.items()
            },
            "treasury_manifest": {"path": str(SOURCE_MANIFEST), "sha256": SOURCE_MANIFEST_SHA},
            "historical_market": {"path": str(MARKET), "sha256": MARKET_SHA},
            "live_extension": "read-only Postgres BTCUSDT 1m through 2026-08-01",
            "execution_prices": "sealed until source support and novelty pass",
        },
        "research_boundary": {
            "prior_treasury_level_and_twist_source_results_known": True,
            "prior_treasury_outcomes_used": False,
            "prior_event_sets_or_controls_reused": False,
            "exact_candidate_incidence_or_outcomes_known": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
            "selection_basis": "independent three-maturity Treasury curvature factor",
        },
        "stopping_rule": (
            "terminal first failure; no source, maturity, curvature, side, variation, delay, "
            "clock, hold, subset, threshold, comparator or control repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload["manifest_hash"] != canonical_hash(core):
        raise RuntimeError("HVTBCR preregistration hash mismatch")
    for name, digest in SOURCE_HASHES.items():
        if hashlib.sha256((SOURCE_DIR / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"HVTBCR source drift: {name}")
    for path, digest in ((SOURCE_MANIFEST, SOURCE_MANIFEST_SHA), (MARKET, MARKET_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"HVTBCR source drift: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)
