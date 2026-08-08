"""Outcome-blind preregistration for HVDCR-72."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "results/high_volatility_donchian_reclaim_relay_preregistration_2026-08-09.json"
)
SOURCE_BINDINGS = {
    "preprocessing/market_features.py": "f9091ecb080656c69a08ac3b4d07f7316cc2ddcc1fe4efacb9e10e8334d5cafa",
    "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz": "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c",
}


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def build() -> dict[str, Any]:
    core = {
        "protocol_version": "high_volatility_donchian_reclaim_relay_v1",
        "policy_id": "HVDCR-72",
        "as_of_date": "2026-08-09",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "singleton": True,
        "mechanism": {
            "claim": (
                "A high-volatility break of a completed three-day price channel that subsequently "
                "pulls back inside the old range without losing its directional half, then reclaims "
                "the frozen boundary, represents tested price discovery and should continue over 72 hours."
            ),
            "side": "original breakout direction",
            "why_distinct": (
                "HVDCR is a causal multi-step episode with a frozen old boundary, defended pullback, "
                "and later reclaim. It is not the rejected raw Donchian crossing plus ATR exit, a "
                "single-bar range breakout, or a terminal candidate/control modification."
            ),
            "prior_art_boundary": (
                "the repository forbids retrying raw 5m Donchian+ATR grids but explicitly permits a "
                "materially different trend-pullback recovery or slower execution horizon"
            ),
        },
        "clock": {
            "decision": "every completed 5m bar; entry is the next exact 5m open",
            "channel": "prior 864 completed 5m bars, shifted one bar; current bar excluded",
            "breakout": (
                "false-to-true close above prior high or below prior low while range_vol is at or "
                "above frozen 2023H1 q60; freeze high, low, midpoint, direction, and breakout time"
            ),
            "pullback_window": "within 24 elapsed hours after breakout",
            "defended_pullback": (
                "long: close<=frozen high and close>=frozen midpoint; short: close>=frozen low and "
                "close<=frozen midpoint"
            ),
            "reclaim_window": "within 24 elapsed hours after the first defended pullback",
            "reclaim": "long close>frozen high; short close<frozen low",
            "episode_rules": (
                "one pending episode globally; expiry or a close through the opposite frozen boundary "
                "cancels; no replacement while pending"
            ),
            "calibration": "2023H1 source-only range_vol q60; no forward return or PnL",
            "hold": "72 elapsed hours",
            "reservation": "global half-open after accepted reclaim; exit first on equal open",
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
            "names": ["no_volatility_gate", "no_midpoint_defense", "one_bar_stale_geometry", "direction_flip"],
            "diagnostic_controls_cannot_be_promoted": True,
        },
        "source_plan": {
            "historical_market": "hash-bound 5m cache through 2026-06-01",
            "live_extension": "read-only Postgres completed bars through 2026-08-01",
            "execution_prices": "sealed until source support and Gross9 novelty pass",
        },
        "source_bindings": SOURCE_BINDINGS,
        "research_boundary": {
            "raw_donchian_atr_outcomes_known_and_not_reused": True,
            "exact_episode_outcomes_known": False,
            "candidate_incidence_opened": False,
            "postentry_return_or_pnl_opened": False,
            "gross9_rows_opened": False,
            "candidate_count": 1,
            "grid": False,
            "repair_of_prior_candidate": False,
            "promoted_prior_control": False,
        },
        "stopping_rule": "terminal first failure; no channel, timing, side, hold, or gate repair",
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def validate(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("HVDCR preregistration hash mismatch")
    for raw, expected in SOURCE_BINDINGS.items():
        if hashlib.sha256(Path(raw).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"HVDCR source drift: {raw}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build()
    validate(result)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(args.output)
