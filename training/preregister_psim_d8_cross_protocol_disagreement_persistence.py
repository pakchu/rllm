#!/usr/bin/env python3
"""Freeze a source-structural PSIM-D8 alpha before opening source values.

This registration intentionally does not read the D8 cards/events or the
market/funding payloads.  It binds their already-frozen identities and defines
an alpha family that is independent of the terminal PSIM-D8 RLLM2 family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_ID = "PSIM-D8-CDP1"
PROTOCOL_VERSION = "psim_d8_cross_protocol_disagreement_persistence_v1"
AS_OF_DATE = "2026-07-29"
DEFAULT_OUTPUT = Path(
    "results/psim_d8_cross_protocol_disagreement_persistence_"
    "preregistration_2026-07-29.json"
)

D8_RESULT = Path(
    "results/protocol_specification_intent_maturity_d8_"
    "source_support_2026-07-27.json"
)
D8_RESULT_SHA256 = (
    "0b92b476b654cd76f0cf9dc004690cbcb78e7a5e73917b5d66611c0460d00204"
)
D8_CONTROLS = Path(
    "results/protocol_specification_intent_maturity_d8_"
    "source_controls_2026-07-27.json"
)
D8_CONTROLS_SHA256 = (
    "6c24b5d6ea693e19a90972a31ae96a24ac28a1f1a6b20be63418d0b5881551b1"
)
D8_EXECUTION_SEAL = Path(
    "results/psim_d8_source_support_execution_seal_2026-07-27.json"
)
D8_EXECUTION_SEAL_SHA256 = (
    "c63951fddbae7aabf0eaa51edaacfdfc67203b004580d080189eb8635648f9df"
)
D8_EVENTS = Path(
    "data/protocol_specification_intent_maturity_d8_events_2020_2023.jsonl.gz"
)
D8_EVENTS_SHA256 = (
    "d7308789176af4bfe1bb2f5f13c89d6811bc7f938f3ecec08b1bf8acc5f7e2b2"
)
D8_EVENTS_ROWS_SHA256 = (
    "b6f1e1733d423fd0fd88f7008d1e505d3a513c0d2bec692446c6e2cf32196ac0"
)
D8_CARDS = Path(
    "data/protocol_specification_intent_maturity_d8_cards_2020_2024q1.jsonl.gz"
)
D8_CARDS_SHA256 = (
    "ce1bd1bd9a24068e6e223efca323db805781e912eadb0d2a8b7d63610fab96c1"
)
D8_CARDS_ROWS_SHA256 = (
    "cd73cd6f7f82a02b8662ef4689a721fa32698f73f37aebc1f1041dbfab3fb071"
)

RLLM2_TERMINAL_RESULT = Path(
    "results/psim_d8_rllm2_s7_2021_report_only_transfer_"
    "result_2026-07-27.json"
)
RLLM2_TERMINAL_RESULT_SHA256 = (
    "c061b82438a5b207801b321b864a252564fcd754f3ce09a4ff4d427c3327480a"
)
RLLM2_TERMINAL_RESULT_HASH = (
    "545b58bd5346d6fa5c87195e06692432a0b0447cbc31a56a917830b62da59e71"
)
RLLM2_TERMINAL_REPORT = Path(
    "docs/psim-d8-rllm2-s7-2021-report-only-transfer-"
    "rejection-2026-07-27.md"
)
RLLM2_TERMINAL_REPORT_SHA256 = (
    "d171edfda4518c80815d2404553ab10ffe394f83a78d38f514aa0e315e4ccfde"
)

MARKET = Path(
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)
MARKET_SHA256 = (
    "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
)
FUNDING = Path("data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz")
FUNDING_SHA256 = (
    "3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6"
)

UNIT_COMPONENTS = (
    "event_type_exact_mismatch",
    "window_revision_count_bucket_exact_mismatch",
    "window_age_bucket_exact_mismatch",
    "update_gap_bucket_exact_mismatch",
    "dependency_delta_state_exact_mismatch",
    "dependency_edge_delta_count_bucket_exact_mismatch",
    "line_change_count_bucket_exact_mismatch",
    "changed_section_count_bucket_exact_mismatch",
    "changed_sections_jaccard_distance",
)
SLOW_FLOORS = (0.35, 0.50, 0.65)
FAST_SLOW_GAPS = (0.05, 0.10, 0.15)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with (REPO_ROOT / path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_metadata_authority() -> None:
    expected = {
        D8_RESULT: D8_RESULT_SHA256,
        D8_CONTROLS: D8_CONTROLS_SHA256,
        D8_EXECUTION_SEAL: D8_EXECUTION_SEAL_SHA256,
        RLLM2_TERMINAL_RESULT: RLLM2_TERMINAL_RESULT_SHA256,
        RLLM2_TERMINAL_REPORT: RLLM2_TERMINAL_REPORT_SHA256,
    }
    for path, expected_hash in expected.items():
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"frozen authority drift: {path.as_posix()}")


def _candidate_ids() -> list[str]:
    return [
        f"CDP_S{int(round(slow * 100)):02d}_G{int(round(gap * 100)):02d}"
        for slow in SLOW_FLOORS
        for gap in FAST_SLOW_GAPS
    ]


def build_preregistration() -> dict[str, Any]:
    _validate_metadata_authority()
    core: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "as_of_date": AS_OF_DATE,
        "candidate": {
            "id": POLICY_ID,
            "name": "PSIM-D8 cross-protocol disagreement persistence",
            "stage": "source_structural_family_preregistration",
            "source_candidate": "PSIM-D8",
            "profitability_claim": False,
            "source_values_opened": False,
            "outcomes_opened": False,
        },
        "source_authority": {
            "source_support_result": {
                "path": D8_RESULT.as_posix(),
                "sha256": D8_RESULT_SHA256,
            },
            "source_controls": {
                "path": D8_CONTROLS.as_posix(),
                "sha256": D8_CONTROLS_SHA256,
            },
            "execution_seal": {
                "path": D8_EXECUTION_SEAL.as_posix(),
                "sha256": D8_EXECUTION_SEAL_SHA256,
            },
            "events": {
                "path": D8_EVENTS.as_posix(),
                "sha256": D8_EVENTS_SHA256,
                "canonical_rows_sha256": D8_EVENTS_ROWS_SHA256,
                "payload_opened_or_hashed_at_preregistration": False,
            },
            "cards": {
                "path": D8_CARDS.as_posix(),
                "sha256": D8_CARDS_SHA256,
                "canonical_rows_sha256": D8_CARDS_ROWS_SHA256,
                "payload_opened_or_hashed_at_preregistration": False,
            },
            "source_rerun_repair_or_d9_allowed": False,
        },
        "independence_contract": {
            "rllm2_terminal_result": {
                "path": RLLM2_TERMINAL_RESULT.as_posix(),
                "sha256": RLLM2_TERMINAL_RESULT_SHA256,
                "result_hash": RLLM2_TERMINAL_RESULT_HASH,
                "decision": "reject",
                "terminal_action": (
                    "RETIRE_UNCHANGED_S6R1_HYPOTHESIS_NO_2021_REPAIR"
                ),
                "2022_or_later_outcomes_opened": False,
            },
            "rllm2_terminal_report": {
                "path": RLLM2_TERMINAL_REPORT.as_posix(),
                "sha256": RLLM2_TERMINAL_REPORT_SHA256,
            },
            "repair_or_successor_of_rllm2": False,
            "forbidden_inputs": [
                "gemma_or_any_other_language_model_output",
                "teacher_relation_labels_or_logits",
                "text_embeddings_or_generated_text",
                "selected_subcard_or_subcard_selector",
                "ridge_fqi_q_values_or_residual_rewards",
                "2020_or_2021_market_funding_returns_or_policy_metrics",
            ],
            "2021_terminal_metrics_used_for_direction_threshold_or_selection": False,
        },
        "feature_contract": {
            "primary_schedule": "ARCHIVE_D90",
            "card_clock": "daily_card_decision_at_12:05:00_utc",
            "relation_scope": "complete_logical_day_relation_roster",
            "eligible_unit": (
                "ethereum and bitcoin are mapping payloads; counterpart_state "
                "is SAME_DAY_CARTESIAN or TRAILING_90D; "
                "memorization_excluded is false"
            ),
            "unit_components": list(UNIT_COMPONENTS),
            "exact_mismatch_definition": (
                "0.0 when the two protocol payload values are exactly equal; "
                "1.0 otherwise"
            ),
            "changed_sections_jaccard_definition": (
                "1 - |set(ethereum.changed_sections) intersect "
                "set(bitcoin.changed_sections)| / "
                "|set(ethereum.changed_sections) union "
                "set(bitcoin.changed_sections)|; both empty gives 0"
            ),
            "unit_score_formula": (
                "mean(component_disagreement_j for j in the 9 frozen components)"
            ),
            "daily_aggregate": (
                "arithmetic_mean(unit_score for all eligible relation units)"
            ),
            "empty_day_action": (
                "daily_score_missing; leave both EWMAs unchanged; emit no signal"
            ),
            "fast_ewma_half_life_cards": 3,
            "slow_ewma_half_life_cards": 30,
            "ewma_formula": (
                "ewma_t = alpha*x_t + (1-alpha)*ewma_previous, "
                "alpha = 1 - exp(log(0.5)/half_life)"
            ),
            "ewma_initialization": "first_nonmissing_daily_score",
            "minimum_nonmissing_cards_before_signal": 30,
            "normalization_or_fitted_parameters": False,
            "market_funding_or_return_inputs": False,
        },
        "frozen_family": {
            "slow_floor_grid": list(SLOW_FLOORS),
            "fast_slow_gap_grid": list(FAST_SLOW_GAPS),
            "cartesian_product_order": "slow_floor_outer_gap_inner",
            "candidate_ids": _candidate_ids(),
            "candidate_count": len(SLOW_FLOORS) * len(FAST_SLOW_GAPS),
            "rank2_or_post_result_threshold_repair_allowed": False,
        },
        "action_contract": {
            "short_rule": (
                "slow_ewma >= slow_floor AND "
                "fast_ewma - slow_ewma >= fast_slow_gap"
            ),
            "long_rule": (
                "slow_ewma >= slow_floor AND "
                "slow_ewma - fast_ewma >= fast_slow_gap"
            ),
            "flat_rule": "otherwise",
            "simultaneous_long_short_action": "impossible_by_positive_gap",
            "signal_time": "card decision_at after causal source availability",
            "entry_time": (
                "open of the first complete 5m interval beginning at least "
                "one full bar after decision_at"
            ),
            "entry_lag_bars_5m": 1,
            "holding_bars_5m": 288,
            "exit_time": "open exactly 288 5m bars after entry",
            "overlap": "forbidden_first_signal_wins",
            "position_size": "one_unit_notional_standalone",
        },
        "source_support_gate": {
            "evaluated_before_market_or_funding_open": True,
            "minimum_eligible_signal_days_2022": 24,
            "minimum_eligible_signal_days_2023": 24,
            "minimum_active_quarters_each_year": 3,
            "maximum_top_calendar_month_share": 0.50,
            "minimum_long_candidates_with_incidence": 1,
            "minimum_short_candidates_with_incidence": 1,
            "failure_action": (
                "TERMINAL_REJECT_CDP1_SOURCE_SUPPORT_NO_THRESHOLD_REPAIR"
            ),
        },
        "outcome_authority": {
            "market": {
                "path": MARKET.as_posix(),
                "sha256": MARKET_SHA256,
                "payload_opened_or_hashed_at_preregistration": False,
            },
            "funding": {
                "path": FUNDING.as_posix(),
                "sha256": FUNDING_SHA256,
                "payload_opened_or_hashed_at_preregistration": False,
            },
        },
        "evaluation_contract": {
            "selection_period": ["2022-01-01", "2022-12-31"],
            "future_veto_period": ["2023-01-01", "2023-12-31"],
            "future_veto_is_untouched": True,
            "2020_2021_outcomes_excluded_from_all_decisions": True,
            "base_cost_per_side": 0.0006,
            "stress_cost_per_side": 0.0010,
            "funding": "exact_mark_funding_cashflows",
            "drawdown": "strict_intra_position_adverse_excursion",
            "selection_eligibility": {
                "minimum_closed_trades": 20,
                "minimum_long_trade_share": 0.20,
                "minimum_short_trade_share": 0.20,
                "base_net_return_strictly_positive": True,
                "stress_net_return_strictly_positive": True,
            },
            "selection_order": [
                "base_net_return_over_strict_mdd_descending",
                "stress_net_return_descending",
                "closed_trades_descending",
                "candidate_id_ascending",
            ],
            "single_selected_top1": True,
            "future_veto_gate": {
                "base_net_return_strictly_positive": True,
                "stress_net_return_not_negative": True,
                "base_return_over_strict_mdd_minimum": 0.50,
                "strict_mdd_not_more_than_selection_multiple": 2.0,
                "minimum_closed_trades": 15,
            },
            "post_veto_repair_rank2_or_threshold_change_allowed": False,
            "same_gross_portfolio_promotion": (
                "separate preregistration required after standalone future "
                "veto and restoration of exact frozen Gross9 inputs"
            ),
        },
        "access_boundary": {
            "source_payload_paths_opened_or_read": [],
            "source_payload_rows_parsed": 0,
            "source_values_or_incidence_computed": False,
            "market_or_funding_paths_opened_or_read": [],
            "market_rows_parsed": 0,
            "funding_rows_parsed": 0,
            "economic_metrics_computed": 0,
        },
        "execution_contract": {
            "preregistration_committed_before_source_payload_open": True,
            "source_support_result_committed_before_outcome_payload_open": True,
            "selection_result_committed_before_2023_future_veto_open": True,
            "failure_is_terminal_for_cdp1": True,
            "new_family_after_failure_requires_new_preregistration": True,
        },
    }
    return {**core, "manifest_hash": canonical_hash(core)}


def write_preregistration(path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    payload = build_preregistration()
    output = path if path.is_absolute() else REPO_ROOT / path
    rendered = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"preregistration drift: {output}")
        return payload
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = write_preregistration(args.output)
    print(
        json.dumps(
            {
                "candidate": payload["candidate"]["id"],
                "family_count": payload["frozen_family"]["candidate_count"],
                "manifest_hash": payload["manifest_hash"],
                "output": args.output.as_posix(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
