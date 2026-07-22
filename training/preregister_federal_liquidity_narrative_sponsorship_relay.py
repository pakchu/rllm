"""Freeze FLNSR-2016 before opening its source incidence or any market outcome."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = (
    "results/federal_liquidity_narrative_sponsorship_relay_"
    "preregistration_2026-07-23.json"
)
H41_SOURCE = (
    "data/federal_reserve_h41_net_liquidity_2018_2023/"
    "federal_reserve_h41_net_liquidity_2018-01-04_2023-12-28.csv.gz"
)
H41_BUILD_MANIFEST = (
    "data/federal_reserve_h41_net_liquidity_2018_2023/build_manifest.json"
)
H41_SOURCE_MANIFEST = (
    "data/federal_reserve_h41_net_liquidity_2018_2023/source_manifest.json"
)
GDELT_SOURCE = "data/gdelt_bitcoin_narrative_daily_2020_2023.csv.gz"
GDELT_MANIFEST = "results/gdelt_bitcoin_narrative_source_manifest_2026-07-20.json"
GDELT_ACCESS_SEAL = "results/gdelt_gnrc_source_access_seal_2026-07-22.json"


@dataclass(frozen=True)
class Policy:
    policy_id: str = "FLNSR-2016"
    liquidity_delta_releases: int = 1
    liquidity_rank_lookback_releases: int = 104
    liquidity_lower_rank_numerator: int = 83
    liquidity_upper_rank_numerator: int = 125
    narrative_recent_days: int = 7
    narrative_baseline_days: int = 21
    narrative_pseudocount: float = 0.5
    entry_delay_minutes: int = 10
    hold_bars: int = 2_016
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
        "protocol_version": "federal_liquidity_narrative_sponsorship_relay_v1",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "policy": asdict(Policy()),
        "research_history_boundary": {
            "repo_wide_btc_history_already_seen": True,
            "h41_source_values_already_audited": True,
            "gdelt_source_values_already_seen": True,
            "flcc_train_2020_2022_outcomes_seen_and_rejected": True,
            "flcc_2023_outcomes_seen": False,
            "gnrc_train_and_2023_outcomes_seen_and_rejected": True,
            "exact_flnsr2016_clock_or_outcomes_seen": False,
            "claim_scope": (
                "source- and ancestor-outcome-seen cross-axis candidate, not a pristine "
                "discovery or globally untouched market holdout"
            ),
            "llm_used_in_this_stage": False,
        },
        "mechanism": {
            "name": "Federal liquidity narrative sponsorship relay",
            "economic_object": (
                "a one-release change in archived Federal Reserve net liquidity supplies a "
                "weekly USD-liquidity impulse; a simultaneous rotation of Bitcoin news toward "
                "adoption or toward failure/constraint determines whether that impulse has "
                "crypto-specific narrative sponsorship"
            ),
            "weak_signal_conjunction": [
                "H.4.1 net-liquidity impulse is public-macro and individually regime-dependent",
                "GDELT category rotation is public narrative composition and individually weak",
                "only same-direction agreement creates a candidate",
            ],
            "new_geometry": [
                "event clock fixed by exogenous H.4.1 release availability",
                "one-release liquidity change rather than FLCC's four/eight-release impulse",
                "no H.4.1 component-breadth gate",
                "threshold-free sign of 7-day versus preceding-21-day narrative quality",
                "seven-day consequence horizon matching the weekly release clock",
            ],
            "explicitly_not": [
                "a threshold or hold repair of FLCC-1",
                "a threshold/window repair of rejected GNRC",
                "a BTC price, trend, funding, premium, OI, FX, or kimchi gate",
                "a family grid selected from source incidence",
                "an LLM-generated side, event, entry, exit, or hold",
            ],
        },
        "h41_source_contract": {
            "source": H41_SOURCE,
            "source_sha256": (
                "224883dad01b9d7f17d52eb87f3d7ef9890c8dd055a6c36577a534d2afe69621"
            ),
            "build_manifest": H41_BUILD_MANIFEST,
            "build_manifest_sha256": (
                "1ec212a85de0e49c5a0c2d35b8b22be86eb7d62989f7a0098be1bb1274b2a99b"
            ),
            "source_manifest": H41_SOURCE_MANIFEST,
            "source_manifest_sha256": (
                "61dca0ae9e29c2c96307a3442037e43aedae15e21d3aedc9ee209c7ebbcac271"
            ),
            "rows": 313,
            "release_interval": ["2018-01-04", "2023-12-28"],
            "required_columns": [
                "release_date",
                "observation_date",
                "available_at_utc",
                "total_assets_usd_millions",
                "treasury_general_account_usd_millions",
                "reverse_repurchase_agreements_usd_millions",
                "net_liquidity_usd_millions",
            ],
            "availability": (
                "archived release date at 16:35 America/New_York, already five minutes "
                "after the Federal Reserve's stated 16:30 release time"
            ),
            "identity": "net_liquidity = total_assets - TGA - reverse_repurchase_agreements",
            "missing_or_invalid_release_policy": "fail closed; no fill or skipped-rank repair",
        },
        "gdelt_source_contract": {
            "source": GDELT_SOURCE,
            "source_sha256": (
                "52d98ee9d63049ca9b12a70f7728a56dfad520f6379feef4e173140e7581347b"
            ),
            "manifest": GDELT_MANIFEST,
            "manifest_file_sha256": (
                "b6e413cca8ba62ca614c5343c81c59e08c04b9819b6f061d123cfa2f0dbc0c68"
            ),
            "manifest_hash": (
                "ec8721f42bc8efc19251f7ee7bb526fe70df6c6426b9c63688850bfb0c142a17"
            ),
            "access_seal": GDELT_ACCESS_SEAL,
            "access_seal_sha256": (
                "267cbc8c1edd3bbfbbb290f39536a79ff90d51a26eb942c1dabcab5113cc81e8"
            ),
            "rows": 1_461,
            "source_interval": ["2020-01-01", "2024-01-01"],
            "required_columns": [
                "date",
                "available_at",
                "global_article_count",
                "broad_article_count",
                "failure_article_count",
                "constraint_article_count",
                "adoption_article_count",
            ],
            "availability": "source date UTC midnight + 48 hours + 15 minutes",
            "known_zero_outage_days": ["2020-10-20", "2023-03-23"],
            "zero_outage_policy": (
                "retain the audited all-zero daily rows exactly; pseudocount keeps quality finite"
            ),
            "missing_or_invalid_day_policy": "fail closed; exact daily grid required",
        },
        "causal_feature_contract": {
            "liquidity_impulse": (
                "current net_liquidity_usd_millions minus the immediately previous release; "
                "holiday spacing does not change the one-release definition"
            ),
            "liquidity_reference": (
                "exactly 104 prior one-release impulses excluding current; exact integer "
                "midrank numerator = 2*count(prior<current)+count(prior==current), denominator 208"
            ),
            "liquidity_side": (
                "LONG when numerator>=125, SHORT when numerator<=83, otherwise neutral"
            ),
            "latest_narrative_day": (
                "maximum GDELT source date whose audited available_at is <= current H.4.1 "
                "available_at_utc"
            ),
            "narrative_quality": (
                "q[d]=log((adoption_article_count[d]+0.5)/"
                "(failure_article_count[d]+constraint_article_count[d]+1.0))"
            ),
            "narrative_rotation": (
                "arithmetic mean q over the latest 7 consecutive source days minus arithmetic "
                "mean q over the immediately preceding 21 consecutive source days"
            ),
            "narrative_side": (
                "LONG when narrative_rotation>0, SHORT when <0, neutral when exactly zero"
            ),
            "primary": (
                "candidate only when liquidity_side and narrative_side are the same non-neutral side"
            ),
            "future_source_row_used": False,
            "btc_market_field_used": False,
        },
        "execution_contract": {
            "signal_time": "current H.4.1 available_at_utc",
            "decision": "signal_time plus ten minutes",
            "entry": "the exact UTC five-minute open at signal_time plus ten minutes",
            "exit": "entry plus exactly 2,016 five-minute bars / seven calendar days",
            "side": "fixed to the agreeing liquidity/narrative side",
            "maximum_one_candidate_per_h41_release": True,
            "chronological_nonoverlap": (
                "accept when entry >= previous accepted exit; suppress rather than queue overlap"
            ),
            "split_containment": (
                "signal, decision, entry, full hold, and exit must remain inside one half-open split"
            ),
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
            "train_events_min": 24,
            "each_train_year_events_min": 6,
            "selection_events_min": 8,
            "each_selection_half_events_min": 3,
            "each_side_share_range_all_train_selection": [0.25, 0.75],
            "active_months_min": 20,
            "maximum_single_month_share": 0.15,
            "maximum_single_quarter_share": 0.30,
            "maximum_calendar_gap_days": 120,
            "maximum_same_side_run": 6,
            "gate_domain": (
                "chronologically accepted split-contained primary clock; window attribution by "
                "entry_time with exit_time <= window end"
            ),
            "controls_affect_primary_pass": False,
            "failure_action": "reject without calculating a BTC post-entry return",
        },
        "source_only_novelty_gate": {
            "comparator": (
                "results/federal_liquidity_component_concordance_"
                "preregistered_clock_2026-07-17.csv.gz"
            ),
            "comparator_sha256": (
                "7ebb0450422d9265e46c596e0b6415b6a8816c66f5e0cbb9ccda14ca6cb4c67c"
            ),
            "comparator_clock": "primary for each frozen FLCC candidate",
            "comparison_window": "train plus selection accepted entries",
            "matching": (
                "chronological deterministic one-to-one nearest match within plus/minus "
                "15 minutes; ties choose earlier comparator entry"
            ),
            "jaccard_max_each_flcc_candidate": 0.50,
            "flnsr_containment_max_each_flcc_candidate": 0.70,
            "same_side_flnsr_containment_max_each_flcc_candidate": 0.75,
            "comparator_outcomes_read": False,
            "failure_action": (
                "reject as an H.4.1 clock re-expression without opening BTC outcomes"
            ),
        },
        "source_only_controls": {
            "liquidity_only": (
                "every non-neutral H.4.1 liquidity tail, same side, scheduler, and execution"
            ),
            "narrative_only": (
                "every non-neutral narrative rotation sampled at H.4.1 releases, same side"
            ),
            "disagreement": (
                "when both sides are non-neutral and opposite, trade the liquidity side"
            ),
            "exact_side_flip": "same primary timestamps with opposite side",
            "one_release_stale_narrative": (
                "current liquidity side combined with narrative_side sampled at the prior H.4.1 release"
            ),
            "deterministic_random_side": (
                "primary timestamps with NumPy default_rng seed from SHA256('FLNSR-2016|side')"
            ),
            "shared_control_contract": (
                "controls are report-only and cannot rescue or reject primary; every control uses "
                "the same availability delay, hold, nonoverlap, split containment, and clock-only "
                "schema; malformed controls fail the source build"
            ),
        },
        "strict_sequence": {
            "phase_1": "source support and controls only",
            "phase_2": "hash-freeze strict evaluator before any FLNSR BTC outcome",
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
            "mean_gross_underlying_bp_min": 30.0,
            "monthly_cluster_signflip_p_max": 0.05,
            "train_each_calendar_year_positive": True,
            "selection_h1_and_h2_positive": True,
            "primary_mean_gross_margin_over_each_component_bp_min": 5.0,
            "component_controls": ["liquidity_only", "narrative_only", "disagreement"],
            "multiplicity_scope": (
                "one frozen FLNSR singleton; ancestor FLCC/GNRC outcome exposure is disclosed "
                "and prevents a global first-discovery claim"
            ),
        },
        "llm_boundary": {
            "allowed_only_after_standalone_train_and_selection_pass": True,
            "requires_separate_overlay_preregistration": True,
            "later_action_space": ["TRADE_FIXED_SIDE", "ABSTAIN"],
            "preferred_symbolic_context": [
                "liquidity shock side and causal percentile bucket",
                "asset/TGA/RRP contribution-sign relation",
                "narrative adoption-versus-stress rotation relation",
                "source evidence and recency tokens",
                "current position and time-to-exit",
            ],
            "llm_may_not_change": [
                "candidate clock",
                "side",
                "entry",
                "hold",
                "leverage",
                "cost model",
            ],
            "reason": (
                "an LLM can arbitrate symbolic macro/narrative relations only after the causal "
                "base conjunction demonstrates economic edge; it cannot rescue a failed event"
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
        raise RuntimeError("FLNSR-2016 preregistration hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("FLNSR-2016 preregistration cannot open outcomes")
    if payload.get("source_incidence_opened") is not False:
        raise RuntimeError("FLNSR-2016 source incidence was opened before freeze")
    if payload.get("policy") != asdict(Policy()):
        raise RuntimeError("FLNSR-2016 policy differs from code")


def write_once(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text())
        validate_manifest(existing)
        if existing["manifest_hash"] != payload["manifest_hash"]:
            raise RuntimeError("refusing to overwrite frozen FLNSR-2016 preregistration")
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
