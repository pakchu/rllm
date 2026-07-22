"""Freeze IVLIR-72 before opening its source incidence or any market outcome."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = (
    "results/intrinsic_volume_latent_impact_relay_preregistration_2026-07-23.json"
)
MARKET_MANIFEST = "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
MARKET_SOURCE = (
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)


@dataclass(frozen=True)
class Policy:
    policy_id: str = "IVLIR-72"
    utc_day_volume_lookback_days: int = 28
    utc_day_volume_min_days: int = 21
    intrinsic_volume_fraction: float = 0.50
    latest_anchor_minute_utc: int = 17 * 60 + 55
    event_reference_days: int = 180
    event_reference_min_days: int = 90
    absolute_flow_quantile: float = 0.60
    maximum_impact_quantile: float = 0.70
    rolling_extrema_bars: int = 2_016
    long_max_range_position: float = 0.80
    short_min_range_position: float = 0.20
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
        "protocol_version": "intrinsic_volume_latent_impact_relay_v1",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "policy": asdict(Policy()),
        "research_history_boundary": {
            "repo_wide_btc_history_already_seen": True,
            "exact_ivlir72_clock_or_outcomes_seen": False,
            "claim_scope": (
                "candidate-level frozen test, not a globally pristine market holdout"
            ),
            "prior_llm_event_arbitration_seen": True,
            "llm_used_in_this_stage": False,
        },
        "mechanism": {
            "name": "intrinsic-volume latent-impact relay",
            "economic_object": (
                "the first daily crossing of a fixed notional target derived only from "
                "prior complete UTC days; strong cumulative aggressive flow that has "
                "moved price in the same direction but with below-tail-normalized impact "
                "is treated as buffered inventory pressure whose price adjustment is delayed"
            ),
            "new_geometry": [
                "forward accumulation from a fixed UTC-day origin",
                "one first-passage event per UTC day",
                "equalized prior-volume target rather than a backward fixed-time window",
                "direction from signed flow and admission from causal impact under-response",
            ],
            "explicitly_not": [
                "the backward rolling volume-clock jump gate",
                "a fixed 00/08/16 UTC session handoff",
                "dual-intrinsic-clock price-versus-flow event counts",
                "execution-metronome spectral regularity",
                "a raw rolling-extrema threshold strategy",
                "an LLM-generated side, entry, exit, or threshold",
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
                "volume reaches the intrinsic target; reject anchors after 17:55 UTC"
            ),
            "cumulative_flow": (
                "sum(2*taker_buy_quote-quote_asset_volume) from UTC-day start through "
                "the anchor divided by cumulative quote volume over the same bars"
            ),
            "anchor_return": "log(anchor_close / UTC-day first-bar open)",
            "directional_return": "sign(cumulative_flow) * anchor_return",
            "impact_ratio": (
                "directional_return / max(abs(cumulative_flow), 1e-12)"
            ),
            "event_reference": (
                "the previous 180 eligible daily anchors, excluding current; at least "
                "90 required; NumPy linear quantiles without clipping or winsorization"
            ),
            "flow_gate": (
                "abs(cumulative_flow) >= prior-event q60 absolute cumulative flow"
            ),
            "alignment_gate": "directional_return >= 0",
            "under_response_gate": (
                "impact_ratio <= prior-event q70 impact ratio among finite aligned anchors"
            ),
            "rolling_extrema": (
                "rolling 2,016 completed bars including the anchor; range_position = "
                "(anchor_close-low)/(high-low); zero-width ranges are ineligible"
            ),
            "headroom_gate": (
                "LONG requires range_position <= 0.80; SHORT requires range_position >= 0.20"
            ),
            "side": "LONG when cumulative_flow>0; SHORT when cumulative_flow<0",
            "current_anchor_excluded_from_every_reference": True,
            "future_bar_used_by_signal": False,
        },
        "execution_contract": {
            "decision": "after the intrinsic-volume anchor bar closes",
            "entry": "next BTCUSDT perpetual five-minute open",
            "exit": "scheduled open exactly 72 five-minute bars after entry",
            "hold": "six hours fixed",
            "maximum_one_candidate_per_utc_day": True,
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
            "train_events_min": 120,
            "each_train_year_events_min": 30,
            "selection_events_min": 35,
            "each_selection_half_events_min": 15,
            "each_side_share_range_all_train_selection": [0.25, 0.75],
            "active_months_min": 30,
            "maximum_single_month_share": 0.10,
            "maximum_single_quarter_share": 0.20,
            "maximum_same_side_run": 15,
            "failure_action": "reject without calculating a post-entry return",
        },
        "source_only_controls": {
            "exact_side_flip": "same primary timestamps with opposite side",
            "flow_only": "drop alignment, impact, and rolling-extrema headroom gates",
            "no_under_response": "retain flow, alignment, and headroom; drop impact gate",
            "no_headroom": "retain flow, alignment, and impact; drop range-position gate",
            "fixed_noon": (
                "replace first-passage anchor with completed 11:55 UTC bar while retaining "
                "strictly-prior daily/event references and all signal gates"
            ),
            "stale_previous_anchor_side": (
                "primary clock with the immediately preceding eligible anchor flow side"
            ),
            "deterministic_random_side": (
                "primary clock with NumPy default_rng seed from SHA256('IVLIR-72|side')"
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
                "prior Gemma SFT/DPO event selectors repeatedly learned class priors; "
                "the base event must first demonstrate causal economic edge"
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
        raise RuntimeError("IVLIR-72 preregistration hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("IVLIR-72 preregistration cannot open outcomes")
    if payload.get("source_incidence_opened") is not False:
        raise RuntimeError("IVLIR-72 source incidence was opened before freeze")
    if payload.get("policy") != asdict(Policy()):
        raise RuntimeError("IVLIR-72 policy differs from code")


def write_once(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text())
        validate_manifest(existing)
        if existing["manifest_hash"] != payload["manifest_hash"]:
            raise RuntimeError("refusing to overwrite frozen IVLIR-72 preregistration")
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
