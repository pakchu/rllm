#!/usr/bin/env python3
"""Preregister the frozen ExtraTrees rank-7 leverage battery.

This module fixes the complete leverage family, the pre-2025 selection rule,
and the later report-only gates before any levered replay is computed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


OUTPUT = Path(
    "results/"
    "expanding_extratrees_rank7_leverage_battery_preregistration_2026-07-27.json"
)
SUMMARY = Path(
    "docs/"
    "expanding-extratrees-rank7-leverage-battery-preregistration-2026-07-27.md"
)

PROTOCOL_VERSION = "expanding_extratrees_rank7_leverage_battery_v1"
LEVERAGE_GRID = (0.50, 0.75, 1.00, 1.25, 1.50)
SELECTION_WINDOWS = ("2023", "2024", "selection")
REPORT_ONLY_WINDOWS = ("2025", "2026h1", "future", "all")

FROZEN_DEPENDENCIES = {
    "results/expanding_extratrees_rank7_hardened_strict_audit_2026-07-19.json": (
        "90d1d1add44a119d2e2083944fc5e483ce1e221b814015d233899cd87adf4e57"
    ),
    "results/expanding_extratrees_rank7_stability_2026-07-15.json": (
        "ffdde5450ef73e0f6099a49ff75a6e16a272dc0c55eb68857ed0f6e168a8ccb2"
    ),
    "results/expanding_extratrees_top10_oos_2026-07-15.json": (
        "41a06a4f492a8625fc3023dd8a7e03a0e229182b7325fc9deb13c100cf381c15"
    ),
    "results/expanding_extratrees_top10_pre2025_manifest_2026-07-15.json": (
        "5901c4cdd66f7894e4712c1d3d5f490af39b8557170968442c84981d34b90976"
    ),
    "training/audit_expanding_extratrees_rank7_hardened_strict.py": (
        "b7d239352ff090726cac1fef5bbb318329f01d4800b17a4816ada69c792c0c6e"
    ),
    "training/audit_expanding_extratrees_rank7_stability.py": (
        "a5280fda3e9ac6267b82914b3e1a094f1bf5716656554d4e0e5ca32306ae0540"
    ),
    "training/audit_weak_feature_responsibility_stability.py": (
        "154352ab3af889716e06cfb6fa3ddd3242a478ded3bdd93b5bcadd54f15e25f9"
    ),
    "training/evaluate_packet_churn_persistence_pre2024.py": (
        "a3d8d56ed9cc6a49603206253bef3d19b42854f9bb58b3aca76b803b2f7258e2"
    ),
    "training/search_inventory_purge_reclaim_alpha.py": (
        "5d8d4df7ea79790afb919bbb481d11de33ecba5768f6e26feb1f7667cd947d65"
    ),
}


def canonical_hash(value: Any) -> str:
    blob = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_frozen_dependencies() -> None:
    for path, expected in FROZEN_DEPENDENCIES.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"frozen dependency drifted: {path}")


def build_preregistration() -> dict[str, Any]:
    validate_frozen_dependencies()
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": {
            "id": "expanding_extratrees_rank7_annual_refit",
            "policy_changes_allowed": False,
            "schedule_changes_allowed": False,
            "only_variable": "absolute account leverage",
            "base_leverage": 0.50,
            "leverage_grid": list(LEVERAGE_GRID),
            "leverage_cap_rationale": (
                "three-times the frozen 0.50x exposure; fixed before levered "
                "metrics to test the user target without an unbounded sizing search"
            ),
        },
        "research_status": {
            "pristine_discovery_oos": False,
            "reason": (
                "2025+ rank-7 outcomes were already viewed before this sizing audit"
            ),
            "protocol_isolation": (
                "leverage is selected only from 2023-2024; 2025+ is report-only "
                "and cannot alter, repair, or reselect the leverage"
            ),
        },
        "immutable_policy": {
            "direction": "long_only",
            "annual_expanding_refits": True,
            "predictor_delay_bars_5m": 12,
            "entry": "next_5m_open",
            "exit": "source_owned_tp_sl_or_time",
            "non_overlap": True,
            "base_cost_per_side": 0.0006,
            "stress_cost_per_side": 0.0010,
            "funding_boundary": (
                "interior symmetric; exact entry/exit credits dropped and debits retained"
            ),
            "strict_mdd": (
                "global/pre-entry HWM plus entry cost, favorable-before-adverse "
                "held 5m OHLC, realized funding debit, virtual adverse-mark "
                "liquidation cost, and actual exit cost"
            ),
            "full_calendar_cagr_includes_idle_time": True,
            "selected_position_hash": (
                "8ffbd55f07ceda0e82c270fe4b370fffba44bb3fcfc807368c4385d2ba97f531"
            ),
        },
        "selection": {
            "windows": list(SELECTION_WINDOWS),
            "calendar": ["2023-01-01", "2025-01-01"],
            "rule": (
                "select the highest leverage in the fixed grid that passes every "
                "base gate and the fixed 10bp-per-side stress profitability gate"
            ),
            "base_gates_each_window": {
                "absolute_return_pct_strictly_above": 0.0,
                "cagr_to_strict_mdd_at_least": 3.0,
                "strict_mdd_pct_at_most": 12.0,
            },
            "minimum_trades": {"2023": 12, "2024": 12, "selection": 24},
            "stress_gates_each_window": {
                "absolute_return_pct_strictly_above": 0.0,
                "strict_mdd_pct_at_most": 15.0,
            },
            "no_tie_breaker_needed": (
                "the leverage ordering is strict and the highest passing value wins"
            ),
            "no_passing_cell_action": "reject_scaling_and_keep_frozen_0_50x_research_baseline",
        },
        "report_only_evaluation": {
            "windows": list(REPORT_ONLY_WINDOWS),
            "calendar": ["2025-01-01", "2026-06-02"],
            "no_repair_after_open": True,
            "base_gates_each_annual_window": {
                "absolute_return_pct_strictly_above": 0.0,
                "cagr_to_strict_mdd_at_least": 3.0,
                "strict_mdd_pct_at_most": 15.0,
                "minimum_trades": {"2025": 12, "2026h1": 6},
            },
            "base_gates_each_combined_window": {
                "absolute_return_pct_strictly_above": 0.0,
                "cagr_to_strict_mdd_at_least": 3.0,
                "strict_mdd_pct_at_most": 15.0,
                "minimum_trades": {"future": 18, "all": 42},
            },
            "stress_gates_combined_windows": {
                "absolute_return_pct_strictly_above": 0.0,
            },
            "significance_gates_combined_windows": {
                "weekly_cluster_sign_flip_one_sided_p_at_most": 0.10,
                "stationary_trade_bootstrap_one_sided_p_at_most": 0.10,
                "draws": 50_000,
                "seed": 20_260_727,
            },
        },
        "user_target": {
            "required_windows": ["future", "all"],
            "absolute_return_always_reported": True,
            "cagr_pct_at_least": 50.0,
            "strict_mdd_pct_at_most": 15.0,
            "cagr_to_strict_mdd_at_least": 3.0,
            "classification": (
                "target_hit only if both fixed future and all windows pass all "
                "three thresholds; otherwise alpha may survive but target is not met"
            ),
        },
        "integrity": {
            "frozen_dependencies_sha256": FROZEN_DEPENDENCIES,
            "all_leverage_cells_must_share_exact_trade_clock_hashes": True,
            "all_grid_cells_reported": True,
            "nonfinite_metrics_fail_closed": True,
            "future_metrics_must_not_enter_selection_payload": True,
        },
        "next_authorized_step": (
            "IMPLEMENT_REVIEW_COMMIT_AND_PUSH_FIXED_BATTERY_THEN_EXECUTE_ONCE"
        ),
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def render_docs(payload: dict[str, Any]) -> str:
    grid = ", ".join(f"{value:.2f}x" for value in LEVERAGE_GRID)
    return "\n".join(
        [
            "# ExtraTrees rank-7 leverage battery preregistration — 2026-07-27",
            "",
            "Status: **PREREGISTERED — levered grid metrics not computed**",
            "",
            "## Fixed question",
            "",
            "Can the already-frozen annual ExtraTrees rank-7 alpha be scaled to the "
            "user target without changing its features, learner, thresholds, direction, "
            "exits, refit cadence, or trade clocks?",
            "",
            f"- fixed leverage grid: `{grid}`;",
            "- selection period: `2023-01-01`–`2025-01-01` only;",
            "- selection rule: highest cell passing every 2023, 2024, combined, and "
            "10 bp stress gate;",
            "- 2025–2026H1: report-only; no repair or reselection;",
            "- full-calendar CAGR includes idle time; absolute return is always shown;",
            "- hardened strict MDD includes entry cost, favorable-before-adverse 5m "
            "path, conservative funding, virtual adverse liquidation cost, and exit cost.",
            "",
            "## Fixed target",
            "",
            "Both `future` (2025–2026H1) and `all` (2023–2026H1) must independently "
            "reach `CAGR >= 50%`, `strict MDD <= 15%`, and `CAGR/MDD >= 3`. A policy "
            "can survive the robustness gates without satisfying this stronger target.",
            "",
            "## Research boundary",
            "",
            "This is not globally pristine OOS: rank-7's 2025+ outcomes were already "
            "viewed. The useful protection is narrower: the new sizing choice is fixed "
            "from pre-2025 windows and later outcomes cannot change it.",
            "",
            f"Manifest hash: `{payload['manifest_hash']}`",
            "",
            "Next authorized action: "
            "`IMPLEMENT_REVIEW_COMMIT_AND_PUSH_FIXED_BATTERY_THEN_EXECUTE_ONCE`.",
            "",
        ]
    )


def write_preregistration(
    output: Path = OUTPUT,
    summary: Path = SUMMARY,
) -> dict[str, Any]:
    payload = build_preregistration()
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != encoded:
        raise RuntimeError(f"preregistration drift: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded, encoding="utf-8")
    rendered = render_docs(payload)
    if summary.exists() and summary.read_text(encoding="utf-8") != rendered:
        raise RuntimeError(f"preregistration summary drift: {summary}")
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(rendered, encoding="utf-8")
    return payload


def main() -> None:
    payload = write_preregistration()
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "summary": str(SUMMARY),
                "manifest_hash": payload["manifest_hash"],
                "leverage_grid": list(LEVERAGE_GRID),
                "next_authorized_step": payload["next_authorized_step"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
