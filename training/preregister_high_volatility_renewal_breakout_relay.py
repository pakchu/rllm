"""Outcome-sequenced preregistration for HVRBR-12."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "results/high_volatility_renewal_breakout_relay_preregistration_2026-08-09.json"
)
SOURCE_BINDINGS = {
    "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz": (
        "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"
    ),
}


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_renewal_breakout_relay_v1",
        "policy_id": "HVRBR-12",
        "as_of_date": "2026-08-09",
        "oos_outcomes_opened": False,
        "oos_source_incidence_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "A completed hour whose realized variation jumps into its strict-prior q90 tail "
                "after at least six consecutive below-median hours marks renewal of urgent price "
                "discovery after liquidity has adapted to quiet conditions; its direction should "
                "continue for twelve hours."
            ),
            "side": "sign of the completed breakout-hour open-to-close log return",
            "why_distinct": (
                "HVRBR is an endogenous renewal-time event defined by the age of a quiet spell and "
                "a fresh hourly realized-variation crossing. HVDPR used fixed eight-hour implied-vol "
                "confirmation; HVVCR used variance concentration inside an already volatile day and "
                "traded reversal. HVRBR uses no implied volatility, flow, OI, funding, cross-asset "
                "feed, calendar slot, terminal candidate repair, or prior control promotion."
            ),
            "why_low_gross9_overlap_is_plausible": (
                "entries occur only at endogenous quiet-to-q90 renewal times rather than Gross9's "
                "dense or fixed structural clocks"
            ),
        },
        "causal_features": {
            "decision_grid": "every completed xx:55 UTC 5m bar; next exact hour is entry",
            "five_minute_return": "log(close[t]/close[t-1])",
            "hourly_realized_variation": (
                "sqrt(sum of squared five-minute returns over the 12 bars ending at decision); "
                "requires all 12 exact bars"
            ),
            "reference": (
                "strictly preceding 2160 complete hourly observations, current excluded; minimum "
                "1440 observations; linear q50 and q90"
            ),
            "quiet_spell": (
                "the immediately preceding six complete hourly realized-variation observations are "
                "each <= their own causally available strict-prior q50"
            ),
            "renewal_crossing": (
                "current hourly realized variation >= its strict-prior q90 and immediately previous "
                "hourly realized variation < its own strict-prior q90"
            ),
            "direction": "nonzero completed breakout-hour log return; zero is inactive",
            "missing": "inactive; no interpolation or forward fill",
            "grid": False,
        },
        "oos_clock": {
            "domain_start": "2023-07-01T00:00:00Z",
            "entry": "next exact-hour 5m open after the completed decision hour",
            "hold": "12 elapsed hours",
            "reservation": "global half-open; chronological signals; exit first on equal open",
            "split_crossing_action": "skip",
            "gross_exposure": 0.5,
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
                "fixed quantity, exact funding marks, 6bp base and 10bp stress per notional side, "
                "every held 5m favorable then adverse, global HWM, full-calendar CAGR"
            ),
        },
        "diagnostic_controls": {
            "names": [
                "no_quiet_spell",
                "no_renewal_crossing",
                "one_hour_stale_direction",
                "direction_flip",
            ],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        "source_plan": {
            "historical_market": "hash-bound 5m cache through 2026-06-01",
            "live_extension": "read-only Postgres completed bars through 2026-08-01",
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "source_bindings": SOURCE_BINDINGS,
        "research_boundary": {
            "prior_volatility_candidate_outcomes_known": True,
            "exact_renewal_breakout_outcomes_known": False,
            "oos_candidate_incidence_opened": False,
            "oos_post_entry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
        },
        "stopping_rule": (
            "freeze this singleton, open source incidence, then Gross9 novelty, then stage outcomes "
            "sequentially; terminal first failure with no threshold, quiet-age, direction, or hold repair"
        ),
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVRBR preregistration hash mismatch")
    for raw, expected in SOURCE_BINDINGS.items():
        if hashlib.sha256(Path(raw).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"HVRBR source drift: {raw}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
