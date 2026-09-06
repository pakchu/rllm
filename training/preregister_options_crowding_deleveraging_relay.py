"""Write the outcome-blind OCDR-12 preregistration."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "results/options_crowding_deleveraging_relay_preregistration_2026-08-08.json"
)


@dataclass(frozen=True)
class Policy:
    policy_id: str = "OCDR-12"
    prior_hour_window: int = 720
    prior_hour_min_periods: int = 672
    oi_change_quantile: float = 0.75
    prior_funding_events: int = 270
    prior_funding_min_events: int = 252
    absolute_funding_quantile: float = 0.75
    entry_delay_minutes: int = 5
    hold_hours: int = 12
    leverage: float = 0.5
    base_cost_per_notional_side: float = 0.0006
    stress_cost_per_notional_side: float = 0.001


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
        "protocol_version": "options_crowding_deleveraging_relay_v1",
        "as_of_date": "2026-08-08",
        "outcomes_opened": False,
        "policy": asdict(Policy()),
        "mechanism": {
            "claim": (
                "an options-led volatility expansion accompanied by rising perpetual "
                "open interest and an extreme signed funding crowd identifies crowded "
                "directional risk that is vulnerable to a twelve-hour deleveraging unwind"
            ),
            "side": "negative sign of the latest causally available funding rate",
            "why_not_ovepr": (
                "OCDR never uses premium-index direction or premium efficiency; OVEPR's "
                "known direction-flip diagnostic is forbidden as selection evidence"
            ),
            "why_not_gross9": (
                "the required joint BVOL/DVOL body-lead state and extreme funding-plus-OI "
                "crowding onset is absent from all five Gross9 sleeve clocks"
            ),
        },
        "causal_clock": {
            "hour": "T is the boundary after a completed UTC hour",
            "volatility_state": (
                "normalized completed-hour BVOL and DVOL bodies are both positive and "
                "the DVOL normalized body is strictly larger"
            ),
            "normalized_body": "(close-open)/open on one completed source hour",
            "oi_change": (
                "sum_open_interest at T-5m divided by T-65m minus one; both rows must be "
                "source-valid and observed_at strictly before the T+5m entry"
            ),
            "oi_gate": (
                "positive OI change >= the strictly-prior 720-hour q75, requiring 672 "
                "finite prior completed-hour observations"
            ),
            "funding_event": (
                "latest BTCUSDT funding event with funding_time<=T and observed_at, when "
                "present, strictly before T+5m; zero funding is invalid"
            ),
            "funding_gate": (
                "absolute latest rate >= its strictly-prior 270-event q75, requiring 252 "
                "prior events; the current event is excluded"
            ),
            "trigger": "false-to-true onset of the complete joint state only",
            "entry": "BTCUSDT perpetual open at exactly T+5m",
            "reservation": "global half-open [entry,entry+12h), exit first on equal open",
            "no_imputation": True,
        },
        "source_plan": {
            "bvol": (
                "official checksum-verified Binance BTCBVOLUSDT hourly archives, extended "
                "through 2026-08-01 only after this preregistration is committed"
            ),
            "dvol": (
                "official Deribit BTC volatility-index hourly API, close_time availability, "
                "extended through 2026-08-01 only after commit"
            ),
            "oi": (
                "Postgres open_interest_binance BTCUSDT 5m rows materialized after commit; "
                "bind ts, observed_at, source and values; no live-table substitution"
            ),
            "funding": (
                "Postgres funding_rates_binance BTCUSDT rows materialized after commit with "
                "funding_time, observed_at when available, rate and settlement mark"
            ),
            "market": "opened only after source-support and Gross9 novelty both pass",
        },
        "stages": {
            "train": ["2023-07-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "test": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "eval": ["2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "final": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
        },
        "economic_gates": {
            "each_stage": {
                "absolute_return_positive": True,
                "cagr_to_strict_mdd_min": 3.0,
                "strict_mdd_max_pct": 15.0,
                "mean_gross_underlying_min_bp": 20.0,
                "weekly_cluster_signflip_one_sided_p_max": 0.10,
                "stress_absolute_return_positive": True,
                "stress_cagr_to_strict_mdd_min": 2.5,
                "each_calendar_half_positive": True,
            },
            "accounting": (
                "fixed quantity, exact funding cash and settlement marks, 6bp/side base, "
                "10bp stress, every held 5m favorable-then-adverse strict path, global HWM, "
                "full declared-calendar CAGR"
            ),
            "stop_on_first_failure": True,
            "future_can_rank_repair_or_reselect": False,
        },
        "novelty_gates": {
            "exact_entry_jaccard_max": 0.10,
            "candidate_near_6h_share_max": 0.45,
            "occupied_5m_jaccard_max": 0.30,
            "absolute_signed_exposure_pearson_max": 0.35,
            "must_pass_before_economics": True,
        },
        "research_boundary": {
            "ovepr_train_result_known": True,
            "ovepr_direction_flip_control_forbidden": True,
            "ocdr_price_or_return_rows_opened": False,
            "ocdr_incidence_opened": False,
            "candidate_count": 1,
            "threshold_hold_direction_grid": False,
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
