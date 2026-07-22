"""Freeze IFDA-72 before raw-trade incidence or market outcomes are opened."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training import preregister_quantity_lattice_cohort_disagreement as qlcd


DEFAULT_OUTPUT = (
    "results/individual_fill_dispersion_absorption_preregistration_2026-07-20.json"
)
SOURCE_AXIS_DECISION = (
    "docs/individual-fill-dispersion-absorption-source-axis-decision-2026-07-20.md"
)
SOURCE_AXIS_DECISION_SHA256 = (
    "43f2f65412751993e90a85fb8edea0d0365f110186a4063e04fc5f8bfb9c4da3"
)
PRIOR_COMPARATOR_BUNDLE = (
    "results/prior_microstructure_comparator_clock_bundle_2026-07-20.json"
)
PRIOR_COMPARATOR_BUNDLE_SHA256 = (
    "c5584256140799b380973f9f376e5751ad754a81c9683473467b9d05af0bb9f0"
)
OWN_CLOCK_CONTROL_IDS = (
    "aggtrade_equalization",
    "flow_only_fade",
    "remove_cross_side_asymmetry",
    "all_fill_equalization",
)


@dataclass(frozen=True)
class Policy:
    policy_id: str = "IFDA-72"
    baseline_bars: int = 8_640
    baseline_min_periods: int = 2_016
    score_quantile: float = 0.995
    minimum_total_fill_count: int = 128
    minimum_each_side_fill_count: int = 32
    execution_delay_bars: int = 2
    hold_bars: int = 72
    post_gap_quarantine_bars: int = 24
    exposure: float = 0.5
    disk_used_abort_gib: int = 300


def explicit_comparator_clocks() -> list[dict[str, Any]]:
    """Return an independent copy of every frozen sparse comparator contract."""
    return [
        *copy.deepcopy(qlcd.COMPARATOR_REGISTRY),
        {
            "family": "VTMS",
            "path": "results/venue_ticket_migration_shock_clock_2026-07-17.csv",
            "sha256": (
                "7baf6f7de33e66417061dbea6f51efc6ea4993b2b5f2b9e0c09627a68adc57e2"
            ),
            "members": ["VTMS-288"],
            "member_column": None,
            "entry_column": "entry_date",
            "coverage": ["2020-01-01", "2024-01-01"],
        },
        {
            "family": "QLCD",
            "path": (
                "data/quantity_lattice_cohort_disagreement_clock_2020_2023.csv.gz"
            ),
            "sha256": (
                "ed882ac8a28f1f0b2b7ad7bf3d2de1f37b175cde63b20d4d1c7a290f3eb89bec"
            ),
            "members": ["QLCD-288"],
            "member_column": None,
            "entry_column": "entry_time",
            "coverage": ["2020-01-01", "2024-01-01"],
        },
    ]


def canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_manifest() -> dict[str, Any]:
    core: dict[str, Any] = {
        "protocol_version": "individual_fill_dispersion_absorption_v1",
        "outcomes_opened": False,
        "source_incidence_opened": False,
        "policy": asdict(Policy()),
        "source_axis_decision": {
            "path": SOURCE_AXIS_DECISION,
            "sha256": SOURCE_AXIS_DECISION_SHA256,
            "tcda_rejected_before_preregistration": True,
        },
        "source_contract": {
            "source": "official Binance USD-M BTCUSDT daily trades archives",
            "official_specification": "https://github.com/binance/binance-public-data",
            "archive_url_template": (
                "https://data.binance.vision/data/futures/um/daily/trades/"
                "BTCUSDT/BTCUSDT-trades-{YYYY-MM-DD}.zip"
            ),
            "checksum_suffix": ".CHECKSUM",
            "range": ["2020-01-01", "2024-01-01"],
            "end_is_exclusive": True,
            "raw_columns": [
                "trade_id",
                "price",
                "qty",
                "quote_qty",
                "time",
                "is_buyer_maker",
            ],
            "raw_archives_persisted": False,
            "download_unit": "one UTC daily ZIP at a time",
            "checksum_verified_before_parse": True,
            "trade_id_contract": (
                "strictly increasing unique IDs with exact +1 continuity across "
                "rows and archive boundaries"
            ),
            "notional_reconciliation": (
                "positive quote_qty must agree with positive price*qty under the "
                "builder's frozen decimal tolerance"
            ),
            "missing_archive_policy": "reject candidate; do not bridge or impute",
            "source_gap_policy": (
                "verified complete zero-trade buckets are allowed; any unverified gap "
                "and the following 24 five-minute bars are incomplete"
            ),
            "disk_guard": (
                "abort before each download when filesystem used bytes are at least "
                "300 GiB; raw ZIP bytes are discarded after each day"
            ),
            "builder": "training.build_binance_individual_fill_dispersion",
            "transform": "preprocessing.individual_fill_dispersion",
            "forbidden_reads": [
                "future OHLC",
                "funding PnL",
                "post-entry return",
                "strategy label",
                "existing alpha outcome",
            ],
        },
        "feature_contract": {
            "candidate_count": 1,
            "bucket": "completed UTC five-minute bar t",
            "fill_notional": "official quote_qty, reconciled to price*qty",
            "aggressive_side": "+1 when is_buyer_maker=false, else -1",
            "buy_notional": "sum(fill_notional where aggressive_side=+1)",
            "sell_notional": "sum(fill_notional where aggressive_side=-1)",
            "dominant_side": "sign(buy_notional-sell_notional)",
            "flow_coherence": (
                "abs(buy_notional-sell_notional)/(buy_notional+sell_notional)"
            ),
            "side_hhi": "sum(q_i^2)/(sum(q_i)^2) within each aggressive side",
            "side_normalized_effective_count": "(1/side_hhi)/side_fill_count",
            "dominant_equalization": (
                "side_normalized_effective_count of dominant_side"
            ),
            "opposing_equalization": (
                "side_normalized_effective_count of -dominant_side"
            ),
            "equalization_gap": (
                "max(dominant_equalization-opposing_equalization,0)"
            ),
            "score": (
                "flow_coherence*dominant_equalization*equalization_gap"
            ),
            "direction": "-dominant_side (fade the coherent aggressor wave)",
            "price_path_fields_used": False,
            "aggregate_trade_event_fields_used_by_primary": False,
            "participant_or_parent_order_identity_inferred": False,
            "threshold_grid": False,
        },
        "support_schedule_contract": {
            "stable_universe": (
                "source_complete; total fills>=128; buy fills>=32; sell fills>=32; "
                "dominant_side nonzero; equalization_gap>0; score>0"
            ),
            "threshold": (
                "mask score to the stable universe, shift one calendar row, then "
                "rolling(8640,min_periods=2016).quantile(0.995)"
            ),
            "activation": (
                "current stable-universe score>=strictly-prior threshold and the "
                "immediately previous calendar bar was not active"
            ),
            "decision_time": "after completed bar t closes",
            "entry_time": "bucket date at t+2",
            "exit_time": "bucket date at entry_position+72",
            "nonoverlap": (
                "accept activations chronologically; skip while entry precedes the "
                "last scheduled exit; re-entry at exit timestamp is allowed"
            ),
            "future_source_rule": (
                "source completeness after decision t never cancels a selected event"
            ),
            "pre2024_containment": True,
            "severity_rationale": (
                "q99.5 is a single frozen upper-tail threshold on a bounded joint "
                "coherence/equalization/asymmetry score; no incidence grid is allowed"
            ),
            "hold_rationale": (
                "72 five-minute bars is six hours, testing cross-session inventory "
                "absorption without reusing the prior 12h/24h fragmentation holds"
            ),
        },
        "support_gates": {
            "total_2020_2023_min": 250,
            "total_2020_2023_max": 1_100,
            "each_calendar_year_min": 50,
            "each_2023_half_min": 20,
            "each_side_share_min": 0.25,
            "each_side_share_max": 0.75,
            "maximum_single_month_share": 0.15,
            "mechanism_control_total_min": 125,
            "mechanism_control_each_year_min": 20,
        },
        "novelty_gates": {
            "prior_bundle": {
                "path": PRIOR_COMPARATOR_BUNDLE,
                "sha256": PRIOR_COMPARATOR_BUNDLE_SHA256,
                "members": [
                    "cbfr72",
                    "mfic_fast",
                    "mfic_slow",
                    "netf_fast",
                    "netf_slow",
                    "wfrs_l288_q90_h144",
                    "terminal_absorption_wait72_h72",
                ],
                "entry_column": "signal_date",
            },
            "explicit_clocks": explicit_comparator_clocks(),
            "exact_entry_jaccard_max": 0.05,
            "tolerant_window_five_minute_bars": 12,
            "tolerant_one_to_one_jaccard_max": 0.15,
            "primary_containment_max": 0.30,
            "matching_algorithm": (
                "within common coverage, sort unique UTC entries; visit primary "
                "entries chronologically and match the unused comparator entry with "
                "minimum absolute distance, ties toward the earlier comparator"
            ),
            "dense_bafr": {
                "path": "results/binance_aggressor_frustration_clock_2026-07-20.csv",
                "sha256": (
                    "f3b816a76decce31136ed23d22f043eb8e80ef1b8697b869241b060062f01747"
                ),
                "entry_column": "entry_date",
                "coverage": ["2020-01-01", "2024-01-01"],
                "report_only": True,
            },
            "missing_or_malformed_comparator": (
                "all sparse clocks fail closed; dense BAFR remains report-only"
            ),
        },
        "falsification_controls": {
            "exact_side_flip": "primary clock with side multiplied by -1",
            "aggtrade_equalization": (
                "same formula on aggregate-event notionals from the frozen aggTrades "
                "source, own q99.5 prior-only onset clock"
            ),
            "flow_only_fade": (
                "score=flow_coherence, direction=-dominant_side, own q99.5 clock"
            ),
            "remove_cross_side_asymmetry": (
                "score=flow_coherence*dominant_equalization, own q99.5 clock"
            ),
            "all_fill_equalization": (
                "score=flow_coherence*all-fill normalized effective count, direction "
                "=-dominant_side, own q99.5 clock"
            ),
            "stale_one_hour": "primary signal, side, entry, and exit shifted +12 bars",
            "stale_twenty_four_hours": (
                "primary signal, side, entry, and exit shifted +288 bars"
            ),
        },
        "support_artifact_contract": {
            "source": (
                "data/binance_um_individual_fill_dispersion_btc_2020_2023/"
                "BTCUSDT_individual_fill_dispersion_5m_2020-01-01_2023-12-31.csv.gz"
            ),
            "source_manifest": (
                "data/binance_um_individual_fill_dispersion_btc_2020_2023/"
                "build_manifest.json"
            ),
            "clock": (
                "data/individual_fill_dispersion_absorption_clock_2020_2023.csv.gz"
            ),
            "clock_columns": [
                "decision_time",
                "entry_time",
                "exit_time",
                "side",
                "score",
                "threshold",
            ],
            "control_clock_bundle": (
                "data/individual_fill_dispersion_absorption_control_clocks_"
                "2020_2023.csv.gz"
            ),
            "control_clock_columns": [
                "control_id",
                "decision_time",
                "entry_time",
                "exit_time",
                "side",
                "score",
                "threshold",
            ],
            "required_own_clock_controls": list(OWN_CLOCK_CONTROL_IDS),
            "result": (
                "results/individual_fill_dispersion_absorption_support_2026-07-20.json"
            ),
            "evaluator": (
                "training.evaluate_individual_fill_dispersion_absorption_support"
            ),
            "write_once": True,
            "deterministic_gzip_mtime": 0,
            "outcomes_opened": False,
            "decision_values": ["PASS_SUPPORT", "REJECT_NO_REPAIR"],
            "required_result_hash_fields": [
                "preregistration_file_sha256",
                "source_file_sha256",
                "source_manifest_sha256",
                "primary_clock_sha256",
                "control_clock_bundle_sha256",
                "comparator_bundle_sha256",
            ],
            "missing_control_effect": "REJECT_NO_REPAIR",
        },
        "later_economic_protocol": {
            "sequential_stages": [
                ["train", "2020-01-01", "2023-01-01"],
                ["selection", "2023-01-01", "2024-01-01"],
                ["test", "2024-01-01", "2025-01-01"],
                ["eval", "2025-01-01", "2026-01-01"],
                ["recent_report", "2026-01-01", None],
            ],
            "base_cost_bp_per_side": 6.0,
            "stress_cost_bp_per_side": 10.0,
            "full_calendar_cagr": True,
            "strict_held_path_mdd": True,
            "train": {
                "absolute_return_positive": True,
                "cagr_to_strict_mdd_min": 1.5,
                "strict_mdd_pct_max": 20.0,
                "each_year_positive": True,
                "weekly_cluster_signflip_p_max": 0.10,
                "mean_gross_underlying_bp_min": 24.0,
                "stress_absolute_return_positive": True,
            },
            "selection_and_oos": {
                "absolute_return_positive": True,
                "cagr_to_strict_mdd_min": 3.0,
                "strict_mdd_pct_max": 15.0,
                "weekly_cluster_signflip_p_max": 0.10,
                "stress_absolute_return_positive": True,
                "minimum_trades_per_year": 40,
                "selection_each_half_positive": True,
            },
            "ratio_definition": {
                "formula": "cagr_pct/strict_mdd_pct",
                "domain": (
                    "strict_mdd_pct must be strictly positive and CAGR, MDD, and ratio "
                    "must all be finite; otherwise the comparison fails closed"
                ),
            },
            "granularity_increment_gate": {
                "control_id": "aggtrade_equalization",
                "required_stages": ["train", "selection", "test", "eval"],
                "absolute_return_rule": (
                    "primary_absolute_return_pct > control_absolute_return_pct"
                ),
                "ratio_rule": (
                    "primary_cagr_to_strict_mdd - "
                    "control_cagr_to_strict_mdd >= 0.50"
                ),
                "ratio_margin_min": 0.50,
                "all_stages_must_pass_independently": True,
                "required_result_fields": [
                    "primary_absolute_return_pct",
                    "control_absolute_return_pct",
                    "primary_cagr_to_strict_mdd",
                    "control_cagr_to_strict_mdd",
                    "absolute_return_pass",
                    "ratio_margin",
                    "ratio_margin_pass",
                    "stage_pass",
                ],
            },
            "mechanism_control_dominance_gate": {
                "control_ids": list(OWN_CLOCK_CONTROL_IDS),
                "required_stages": ["train", "selection", "test", "eval"],
                "ratio_margin_min": 0.25,
                "rule": (
                    "primary_cagr_to_strict_mdd - strongest_finite_control_"
                    "cagr_to_strict_mdd >= 0.25"
                ),
                "missing_or_nonfinite_control_effect": "stage fails closed",
            },
            "stop_on_first_failure": True,
        },
        "live_parity_gate": {
            "recent_endpoint": "/fapi/v1/trades",
            "catchup_endpoint": "/fapi/v1/historicalTrades",
            "raw_futures_websocket_assumed": False,
            "official_rest_reference": (
                "https://developers.binance.com/en/docs/catalog/"
                "core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/"
                "market-data#recent-trades-list"
            ),
            "schema_normalization": {
                "archive_trade_id": "trade Id -> trade_id",
                "rest_trade_id": "id -> trade_id",
                "archive_and_rest": {
                    "price": "price",
                    "qty": "qty",
                    "quoteQty": "quote_qty",
                    "time": "time",
                    "isBuyerMaker": "is_buyer_maker",
                },
                "isRPITrade": (
                    "REST-only boolean metadata introduced after the research window; "
                    "validate and count when present but exclude from IFDA feature math"
                ),
                "unknown_fields": "reject until a new schema version is frozen",
            },
            "request_contract": {
                "recent_limit": 1_000,
                "recent_ip_weight": 5,
                "historical_limit": 500,
                "historical_ip_weight": 20,
                "historical_api_key_required": True,
                "historical_lookback_max": "one month",
                "normal_poll_interval_seconds_max": 1.0,
                "catchup_from_id_inclusive": True,
                "unresolved_gap_wall_clock_minutes_max": 15,
                "rate_limit_behavior": (
                    "track response weight headers and obey 429 Retry-After; never "
                    "continue polling through a required backoff"
                ),
            },
            "requirements": [
                "strictly contiguous trade IDs",
                "no unrecovered five-minute bucket",
                "historical/live feature byte-equivalence on replay fixture",
                "documented request-weight compliance with backoff",
            ],
            "gap_behavior": (
                "suppress new orders immediately; fetch sequentially from the first "
                "missing trade ID; resume only after exact continuity and bar replay"
            ),
            "lookback_or_timeout_failure": (
                "halt IFDA, flatten only under the separately frozen portfolio safety "
                "policy, and rebuild from checksum archives; never bridge the gap"
            ),
            "failure_effect": "research-only; forbidden from live portfolio",
        },
        "claim_boundary": {
            "claim": (
                "a coherent aggressor wave distributed across unusually equal-sized "
                "individual matches relative to the opposing side may indicate broad "
                "passive absorption and subsequent reversion"
            ),
            "not_claimed": [
                "participant identity",
                "maker ownership",
                "one parent order",
                "resting-book depth",
                "profitability before sequential evaluation",
            ],
        },
        "rejection_contract": (
            "any source, disk, support, control-support, novelty, train, sequential "
            "stage, granularity-increment, or live-parity failure retires IFDA-72 "
            "without changing formula, fade direction, quantile, minimum counts, "
            "delay, hold, costs, or gates"
        ),
    }
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
        raise RuntimeError("IFDA-72 preregistration hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise RuntimeError("IFDA-72 preregistration cannot open outcomes")
    if payload.get("source_incidence_opened") is not False:
        raise RuntimeError("IFDA-72 preregistration must precede source incidence")
    if payload.get("policy") != asdict(Policy()):
        raise RuntimeError("IFDA-72 frozen policy differs from code")
    if payload.get("feature_contract", {}).get("candidate_count") != 1:
        raise RuntimeError("IFDA-72 must remain a singleton policy")
    expected = build_manifest()
    expected_core = {
        key: value
        for key, value in expected.items()
        if key not in {"manifest_hash", "created_at"}
    }
    if core != expected_core:
        raise RuntimeError("IFDA-72 frozen contract differs from code")


def write_manifest_once(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        validate_manifest(existing)
        if existing["manifest_hash"] != payload["manifest_hash"]:
            raise RuntimeError("refusing to overwrite frozen IFDA-72 preregistration")
        return "verified_existing"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_manifest()
    status = write_manifest_once(args.output, payload)
    print(
        json.dumps(
            {
                "status": status,
                "policy_id": payload["policy"]["policy_id"],
                "manifest_hash": payload["manifest_hash"],
                "outcomes_opened": False,
                "source_incidence_opened": False,
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
