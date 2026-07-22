"""Freeze IVFHR-72 before opening its source incidence or any market outcome."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = (
    "results/intrinsic_volume_flow_handoff_relay_preregistration_2026-07-23.json"
)
MARKET_MANIFEST = "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
MARKET_SOURCE = (
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)


@dataclass(frozen=True)
class Policy:
    policy_id: str = "IVFHR-72"
    utc_day_volume_lookback_days: int = 28
    utc_day_volume_min_days: int = 21
    intrinsic_volume_fraction: float = 0.50
    latest_anchor_minute_utc: int = 17 * 60 + 55
    event_reference_days: int = 180
    event_reference_min_days: int = 90
    current_flow_quantile: float = 0.60
    prior_state_min_anchors: int = 3
    entry_delay_bars: int = 1
    hold_bars: int = 72
    leverage: float = 0.50
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _core_manifest() -> dict[str, Any]:
    return {
        "protocol_version": "intrinsic_volume_flow_handoff_relay_v1",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "policy": asdict(Policy()),
        "research_history_boundary": {
            "repo_wide_btc_history_already_seen": True,
            "ivlir_source_support_seen": True,
            "ivlir_post_entry_outcomes_seen": False,
            "exact_ivfhr72_clock_or_outcomes_seen": False,
            "claim_scope": (
                "source-seen successor candidate, not a pristine discovery or market holdout"
            ),
            "design_origin": (
                "IVLIR-72 was rejected because cumulative-flow levels produced a 26-event "
                "same-side run; IVFHR makes a state transition the event rather than adding "
                "an anti-persistence gate to the rejected identity"
            ),
            "llm_used_in_this_stage": False,
        },
        "mechanism": {
            "name": "intrinsic-volume flow handoff relay",
            "economic_object": (
                "after at least three consecutive eligible equal-notional daily anchors "
                "share one cumulative taker-flow sign, the first strong opposite-sign anchor "
                "is a transfer of aggressive inventory control; if price from the UTC-day "
                "open still points against the new flow, trade the new flow for delayed catch-up"
            ),
            "new_geometry": [
                "forward daily equal-notional first-passage anchors",
                "a sign transition after a completed persistent source state is the event",
                "current price non-confirmation is required before following the new flow",
                "one event per transition episode rather than repeated level-triggered entries",
            ],
            "explicitly_not": [
                "a repaired IVLIR-72 level filter",
                "a backward rolling volume-clock jump gate",
                "a fixed UTC session handoff",
                "dual-intrinsic-clock event-count divergence",
                "a generic price moving-average crossover",
                "an LLM-generated side, threshold, entry, exit, or hold",
            ],
        },
        "source_contract": {
            "market_manifest": MARKET_MANIFEST,
            "market_manifest_sha256": (
                "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
            ),
            "market": MARKET_SOURCE,
            "market_sha256": (
                "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
            ),
            "market_rows": 420_768,
            "interval": ["2020-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "bar_interval": "5min",
            "timestamp_semantics": (
                "date is the Binance bar-open timestamp in UTC; a row stamped t covers "
                "[t,t+5min) and becomes decision-available only at t+5min"
            ),
            "required_columns": [
                "date",
                "open",
                "high",
                "low",
                "close",
                "quote_asset_volume",
                "taker_buy_quote",
            ],
            "missing_bar_policy": "fail closed; no interpolation or forward fill",
        },
        "causal_feature_contract": {
            "daily_expected_volume": (
                "median total quote_asset_volume of the previous 28 complete UTC days; "
                "current day excluded; at least 21 complete days required"
            ),
            "intrinsic_target": "0.50 * daily_expected_volume",
            "anchor": (
                "first completed five-minute bar of the UTC day whose cumulative quote "
                "volume reaches the target; its date is the bar-open timestamp; reject "
                "anchors whose bar-open timestamp is after 17:55 UTC"
            ),
            "cumulative_flow": (
                "sum(2*taker_buy_quote-quote_asset_volume) from UTC-day start through "
                "the anchor divided by cumulative quote volume over the same bars"
            ),
            "flow_side": "LONG when cumulative_flow>0; SHORT when cumulative_flow<0",
            "invalid_anchor": (
                "a missing/incomplete UTC day, target not reached by the 17:55-open bar, "
                "non-finite required value, non-positive cumulative volume, or exactly zero "
                "cumulative flow is state-invalid; it emits no anchor, is excluded from the "
                "event reference, and resets the prior-state run"
            ),
            "prior_state": (
                "the immediately preceding calendar-consecutive run of valid daily anchors "
                "with one flow_side; it must contain at least three anchors; invalid or "
                "ineligible calendar days break rather than skip the run"
            ),
            "handoff": (
                "current flow_side differs from the immediately preceding eligible anchor; "
                "only the first anchor after that completed prior state can qualify"
            ),
            "event_reference": (
                "previous 180 eligible daily anchors, excluding current; at least 90 required; "
                "NumPy linear quantile without clipping or winsorization"
            ),
            "strong_new_flow": (
                "abs(current cumulative_flow) >= q60 absolute cumulative flow from the "
                "strictly-prior event reference"
            ),
            "price_lag": (
                "new_flow_sign * log(anchor_close / UTC-day first-bar open) <= 0; "
                "price has not yet confirmed the new aggressive-flow side"
            ),
            "side": "fixed to the current/new cumulative-flow sign",
            "current_anchor_excluded_from_every_reference": True,
            "future_bar_used_by_signal": False,
        },
        "execution_contract": {
            "decision": (
                "for anchor bar-open timestamp t, after [t,t+5min) closes at t+5min"
            ),
            "entry": "BTCUSDT perpetual open at t+5min",
            "exit": "scheduled open at entry + 72*5min",
            "hold": "six hours fixed",
            "maximum_one_candidate_per_utc_day": True,
            "maximum_one_candidate_per_transition_episode": True,
            "split_contained_nonoverlap": True,
            "stop_or_take_profit": None,
            "leverage": 0.50,
            "base_cost": "6bp/notional/side",
            "stress_cost": "10bp/notional/side",
            "funding_rule": "entry_time <= funding_time < exit_time",
            "cagr": "full declared calendar including warmup and idle cash",
            "strict_mdd": (
                "global/pre-entry HWM with entry cost, exact funding, every held-bar "
                "favorable-then-adverse path, virtual adverse exit cost, and actual exit"
            ),
        },
        "source_support_gate": {
            "train_window": ["2020-01-01T00:00:00Z", "2023-01-01T00:00:00Z"],
            "selection_window": ["2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
            "train_events_min": 60,
            "each_train_year_events_min": 12,
            "selection_events_min": 18,
            "each_selection_half_events_min": 7,
            "each_side_share_range_all_train_selection": [0.25, 0.75],
            "active_months_min": 24,
            "maximum_single_month_share": 0.12,
            "maximum_single_quarter_share": 0.25,
            "maximum_calendar_gap_days": 90,
            "maximum_same_side_run": 2,
            "statistic_definitions": {
                "event_timestamp": "accepted clock entry_time in UTC",
                "calendar_subperiod": "entry_time membership with exit_time <= subperiod end",
                "side_share": "side count divided by accepted events in the named window",
                "active_months": "distinct UTC calendar months containing an accepted entry",
                "month_or_quarter_share": (
                    "accepted entries in the UTC month or quarter divided by all accepted "
                    "primary entries across train plus selection"
                ),
                "calendar_gap_days": (
                    "maximum elapsed days between consecutive accepted primary entry_time "
                    "values; warmup/source endpoints are excluded"
                ),
                "same_side_run": (
                    "maximum chronological run of equal side among accepted primary entries"
                ),
            },
            "failure_action": "reject without calculating a post-entry return",
            "gate_domain": (
                "the chronologically accepted, split-contained, non-overlapping primary "
                "clock after signal construction; train/selection and calendar subwindows "
                "use entry_time containment and require exit_time <= window end"
            ),
            "controls_affect_primary_pass": False,
        },
        "source_only_controls": {
            "exact_side_flip": (
                "same primary timestamps with the old/prior flow side; report-only"
            ),
            "any_handoff": (
                "drop prior-state length and current-flow-strength requirements; retain price "
                "lag; report-only"
            ),
            "no_price_lag": (
                "retain persistent-state handoff and strong new flow; drop price-lag gate; "
                "report-only"
            ),
            "no_flow_strength": (
                "retain persistent-state handoff and price lag; drop current q60 gate; "
                "report-only"
            ),
            "persistence_level": (
                "same persistent prior state, strong current flow, and price lag, but require "
                "no sign change; report-only"
            ),
            "fixed_noon_handoff": (
                "replace first-passage anchor by the 11:55-open bar, require the intrinsic "
                "target to have been reached by that bar, and preserve strictly-prior expected "
                "volume, state, and reference calculations; report-only"
            ),
            "deterministic_random_side": (
                "primary timestamps with NumPy default_rng seed from SHA256('IVFHR-72|side'); "
                "report-only"
            ),
            "shared_control_contract": (
                "any_handoff, no_price_lag, no_flow_strength, and persistence_level require "
                "the same 90-anchor strictly-prior reference warmup as primary even when a "
                "control drops the q60 test; fixed_noon_handoff builds its own identical "
                "strictly-prior warmup; exact_side_flip and deterministic_random_side reuse "
                "primary timestamps exactly; every control uses the same next-open latency, "
                "72-bar hold, chronological non-overlap, split containment, and clock-only "
                "schema as primary; malformed control clocks fail the build, but control "
                "incidence cannot rescue or reject a primary source-support decision"
            ),
        },
        "strict_sequence": {
            "phase_1": "source support and controls only",
            "phase_2": "hash-freeze strict evaluator before any post-entry outcome",
            "phase_3": "open train 2020-2022 once",
            "phase_4": "open selection 2023 only if train passes",
            "phase_5": "open 2024, then 2025, then 2026 YTD sequentially",
            "stop_at_first_failed_stage": True,
            "no_parameter_repair": True,
        },
        "economic_gates": {
            "each_opened_stage_absolute_return_positive": True,
            "each_opened_stage_cagr_to_strict_mdd_min": 3.0,
            "each_opened_stage_strict_mdd_pct_max": 15.0,
            "base_and_stress_cost_positive": True,
            "one_extra_bar_delay_positive": True,
            "mean_gross_underlying_bp_min": 15.0,
            "weekly_cluster_signflip_p_max": 0.10,
            "train_each_calendar_year_positive": True,
            "selection_h1_and_h2_positive": True,
            "primary_mean_gross_margin_over_best_component_bp_min": 5.0,
            "component_control_universe": [
                "any_handoff",
                "no_price_lag",
                "no_flow_strength",
                "persistence_level",
                "fixed_noon_handoff",
            ],
            "component_margin_statistic": (
                "primary mean gross underlying basis points minus the maximum mean gross "
                "underlying basis points among the frozen component_control_universe, measured "
                "within the same opened stage before leverage, funding, or transaction cost"
            ),
        },
        "llm_boundary": {
            "allowed_only_after_standalone_train_and_selection_pass": True,
            "later_action_space": ["TRADE_FIXED_SIDE", "ABSTAIN"],
            "llm_may_not_change": [
                "candidate clock",
                "side",
                "entry",
                "hold",
                "leverage",
                "cost model",
            ],
            "reason": (
                "the LLM is reserved for symbolic context arbitration after the event itself "
                "shows causal economic edge; it cannot rescue a failed base event"
            ),
        },
    }


def build_manifest() -> dict[str, Any]:
    core = _core_manifest()
    return {
        **core,
        "manifest_hash": canonical_hash(core),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def validate_manifest(payload: dict[str, Any]) -> None:
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"manifest_hash", "created_at"}
    }
    if canonical_hash(core) != payload.get("manifest_hash"):
        raise RuntimeError("IVFHR-72 preregistration hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("IVFHR-72 preregistration cannot open outcomes")
    if payload.get("source_incidence_opened") is not False:
        raise RuntimeError("IVFHR-72 source incidence was opened before freeze")
    if payload.get("policy") != asdict(Policy()):
        raise RuntimeError("IVFHR-72 policy differs from code")


def write_once(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text())
        validate_manifest(existing)
        if existing["manifest_hash"] != payload["manifest_hash"]:
            raise RuntimeError("refusing to overwrite frozen IVFHR-72 preregistration")
        return "verified_existing"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_manifest()
    status = write_once(args.output, payload)
    print(
        json.dumps(
            {
                "status": status,
                "policy_id": payload["policy"]["policy_id"],
                "outcomes_opened": payload["outcomes_opened"],
                "source_incidence_opened": payload["source_incidence_opened"],
                "manifest_hash": payload["manifest_hash"],
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
