"""Professor-critic gate for the conditional PPOSM residual-router result."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ACTIONS = ("SKIP", "TP4", "TP12")
OOS_WINDOWS = ("test_2024", "eval_2025", "holdout_2026")
COMBINED_WINDOW = "combined_2024_2026_06_02"
REQUIRED_ARTIFACTS = {
    "preregistration",
    "data_summary",
    "sft_summary",
    "sft_adapter",
    "rlvr_config",
    "rlvr_reward_diagnostics",
    "rlvr_gradient_diagnostics",
    "rlvr_adapter",
    "train_scores",
    "threshold",
    "oos_scores",
    "route_predictions",
    "route_report",
    "backtest",
    "replay_snapshot",
}


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    observed: Any
    required: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "observed": self.observed,
            "required": self.required,
        }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _resolve_artifact(raw: str, *, result_path: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    candidates = (
        Path.cwd() / path,
        result_path.parent / path,
        result_path.parent.parent / path,
    )
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _stats(window: dict[str, Any], actor: str, cost: str) -> dict[str, Any]:
    value = (
        window.get("economics", {})
        .get(actor, {})
        .get(cost, {})
        .get("equity_stats", {})
    )
    return value if isinstance(value, dict) else {}


def _metric(window: dict[str, Any], actor: str, cost: str, key: str) -> float | None:
    return _number(_stats(window, actor, cost).get(key))


def _artifact_checks(
    result: dict[str, Any], *, result_path: Path
) -> tuple[list[Check], dict[str, Path]]:
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict):
        return [Check("artifact_manifest", False, None, "artifact manifest present")], {}
    missing = sorted(REQUIRED_ARTIFACTS - set(artifacts))
    resolved: dict[str, Path] = {}
    mismatches: dict[str, Any] = {}
    for name, item in artifacts.items():
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            mismatches[name] = "invalid artifact entry"
            continue
        path = _resolve_artifact(item["path"], result_path=result_path)
        resolved[name] = path
        if not path.is_file():
            mismatches[name] = "missing file"
            continue
        observed = _sha256(path)
        if observed != item.get("sha256"):
            mismatches[name] = {"recorded": item.get("sha256"), "observed": observed}
    return [
        Check("required_artifacts_present", not missing, missing, "all required artifacts"),
        Check("artifact_hashes_exact", not mismatches, mismatches, "all recorded hashes match files"),
    ], resolved


def evaluate(result: dict[str, Any], *, result_path: Path) -> dict[str, Any]:
    checks, artifact_paths = _artifact_checks(result, result_path=result_path)
    backtest_path = artifact_paths.get("backtest")
    backtest = _load_object(backtest_path) if backtest_path and backtest_path.is_file() else {}
    windows = backtest.get("windows") if isinstance(backtest.get("windows"), dict) else {}
    combined = windows.get(COMBINED_WINDOW) if isinstance(windows, dict) else None
    combined = combined if isinstance(combined, dict) else {}

    result_core = {key: value for key, value in result.items() if key != "result_hash"}
    expected_result_hash = hashlib.sha256(
        _canonical_json(result_core).encode("utf-8")
    ).hexdigest()
    checks.append(
        Check(
            "result_hash_exact",
            result.get("result_hash") == expected_result_hash,
            result.get("result_hash"),
            expected_result_hash,
        )
    )

    boundary = result.get("research_boundary")
    boundary_text = _canonical_json(boundary).lower() if isinstance(boundary, dict) else ""
    checks.append(
        Check(
            "split_contamination_no_repair_boundary",
            all(token in boundary_text for token in ("pre-2024", "2024", "2025", "2026", "contamination"))
            and "report/veto only" in boundary_text
            and result.get("checks", {}).get("oos_rerank_or_repair") is False,
            boundary,
            "pre-2024 train; 2024/2025/2026 report-veto; contamination; no repair",
        )
    )

    invariants = backtest.get("invariants") if isinstance(backtest.get("invariants"), dict) else {}
    checks.append(
        Check(
            "exact_execution_contract",
            invariants.get("baseline") == "ALWAYS_TP4"
            and invariants.get("entry_rule") == "exact_next_5m_open"
            and invariants.get("lifecycle") == "TP_or_48h_cap"
            and invariants.get("funding_applied") is True
            and invariants.get("non_overlapping") is True
            and invariants.get("future_return_used_for_route") is False,
            invariants,
            "always-TP4, exact next-open, TP/48h, funding, non-overlap, causal route",
        )
    )

    prereg_path = artifact_paths.get("preregistration")
    prereg = _load_object(prereg_path) if prereg_path and prereg_path.is_file() else {}
    data_summary_path = artifact_paths.get("data_summary")
    data_summary = (
        _load_object(data_summary_path)
        if data_summary_path and data_summary_path.is_file()
        else {}
    )

    threshold_path = artifact_paths.get("threshold")
    train_scores_path = artifact_paths.get("train_scores")
    threshold = _load_object(threshold_path) if threshold_path and threshold_path.is_file() else {}
    selection_train = threshold.get("selection_inputs", {}).get("train_scores", {})
    threshold_train_only = (
        "train_only" in str(threshold.get("protocol", ""))
        and set(threshold.get("selection_inputs", {})) == {"train_scores"}
        and threshold.get("future_can_rank_repair_or_reselect") is False
        and train_scores_path is not None
        and train_scores_path.is_file()
        and selection_train.get("sha256") == _sha256(train_scores_path)
        and threshold.get("train_pair_identity_sha256")
        == data_summary.get("identity_sha256_by_split", {}).get("train")
    )
    checks.append(
        Check(
            "threshold_frozen_from_train_only",
            threshold_train_only,
            threshold,
            "threshold selection input is only the exact pre-2024 score artifact",
        )
    )

    frozen_hash = prereg.get("source", {}).get("manifest_freeze_hash")
    checks.append(
        Check(
            "frozen_source_hash_bound_end_to_end",
            isinstance(frozen_hash, str)
            and data_summary.get("manifest_freeze_hash") == frozen_hash
            and backtest.get("manifest_freeze_hash") == frozen_hash,
            {
                "preregistration": frozen_hash,
                "data": data_summary.get("manifest_freeze_hash"),
                "backtest": backtest.get("manifest_freeze_hash"),
            },
            "one manifest freeze hash across preregistration, data, and backtest",
        )
    )
    rlvr_config = result.get("model", {}).get("rlvr", {}).get("config", {})
    rlvr_schema = rlvr_config.get("config", {}).get("label_schema")
    checks.append(
        Check(
            "preregistered_sft_plus_rlvr_residual_model",
            prereg.get("status") == "preregistered_before_model_scoring"
            and result.get("architecture", {}).get("name") == "pairwise_residual_action_router"
            and rlvr_schema == "pposm_residual_utility",
            {"prereg_status": prereg.get("status"), "rlvr_schema": rlvr_schema},
            "frozen preregistration plus SFT and residual-utility RLVR",
        )
    )

    required_windows_present = all(name in windows for name in (*OOS_WINDOWS, COMBINED_WINDOW))
    checks.append(
        Check(
            "required_windows_present",
            required_windows_present,
            sorted(windows) if isinstance(windows, dict) else None,
            "2024, 2025, 2026H1, combined",
        )
    )
    expected_costs = {"base_6bp": 0.0006, "stress_10bp": 0.0010}
    exact_costs = required_windows_present and all(
        _number(
            windows[name]
            .get("economics", {})
            .get(actor, {})
            .get(cost, {})
            .get("one_side_cost_rate")
        )
        == expected
        for name in (*OOS_WINDOWS, COMBINED_WINDOW)
        for actor in ("baseline", "predicted")
        for cost, expected in expected_costs.items()
    )
    checks.append(
        Check(
            "exact_base_and_stress_costs",
            exact_costs,
            expected_costs,
            "6bp/side base and 10bp/side stress in every report",
        )
    )

    route_counts_raw = combined.get("route_counts", {}).get("predicted", {})
    route_counts = {
        action: int(route_counts_raw.get(action, 0))
        for action in ACTIONS
    } if isinstance(route_counts_raw, dict) else {action: 0 for action in ACTIONS}
    total = sum(route_counts.values())
    used = {action: count for action, count in route_counts.items() if count > 0}
    non_default = {
        action: count
        for action, count in route_counts.items()
        if action != "TP4" and count > 0
    }
    agreement = _number(combined.get("agreement", {}).get("decision_agreement_rate"))
    difference_rate = None if agreement is None else 1.0 - agreement
    max_share = max(route_counts.values()) / total if total else None
    checks.extend(
        (
            Check("at_least_two_routes", len(used) >= 2, used, ">=2 actions OOS"),
            Check("difference_rate_ge_10pct", difference_rate is not None and difference_rate >= 0.10, difference_rate, ">=0.10"),
            Check("used_non_default_count_ge_10", bool(non_default) and all(count >= 10 for count in non_default.values()), non_default, "every used non-default route >=10"),
            Check("single_action_share_le_90pct", max_share is not None and max_share <= 0.90, max_share, "<=0.90"),
        )
    )

    all_windows_positive = required_windows_present and all(
        (_metric(windows[name], "predicted", cost, "absolute_return_pct") or 0.0) > 0.0
        for name in OOS_WINDOWS
        for cost in ("base_6bp", "stress_10bp")
    )
    checks.append(
        Check(
            "all_windows_positive_base_stress",
            all_windows_positive,
            {
                name: {
                    cost: _metric(windows.get(name, {}), "predicted", cost, "absolute_return_pct")
                    for cost in ("base_6bp", "stress_10bp")
                }
                for name in OOS_WINDOWS
            },
            "every OOS window positive at 6bp and 10bp per side",
        )
    )

    base_return = _metric(combined, "predicted", "base_6bp", "absolute_return_pct")
    control_return = _metric(combined, "baseline", "base_6bp", "absolute_return_pct")
    base_ratio = _metric(combined, "predicted", "base_6bp", "cagr_to_strict_mdd")
    control_ratio = _metric(combined, "baseline", "base_6bp", "cagr_to_strict_mdd")
    base_mdd = _metric(combined, "predicted", "base_6bp", "strict_mdd_pct")
    stress_return = _metric(combined, "predicted", "stress_10bp", "absolute_return_pct")
    control_stress_return = _metric(combined, "baseline", "stress_10bp", "absolute_return_pct")
    stress_mdd = _metric(combined, "predicted", "stress_10bp", "strict_mdd_pct")
    control_stress_mdd = _metric(combined, "baseline", "stress_10bp", "strict_mdd_pct")
    trades = _number(_stats(combined, "predicted", "base_6bp").get("trades"))
    weekly_p = _number(
        combined.get("economics", {})
        .get("predicted", {})
        .get("base_6bp", {})
        .get("one_sided_utc_week_sign_flip", {})
        .get("p_value_one_sided")
    )
    checks.extend(
        (
            Check("base_return_not_lower", base_return is not None and control_return is not None and base_return >= control_return, {"model": base_return, "always_tp4": control_return}, "model >= control"),
            Check("base_ratio_lift_ge_0_05", base_ratio is not None and control_ratio is not None and base_ratio >= control_ratio + 0.05, {"model": base_ratio, "always_tp4": control_ratio}, ">=+0.05"),
            Check("stress_return_strictly_higher", stress_return is not None and control_stress_return is not None and stress_return > control_stress_return, {"model": stress_return, "always_tp4": control_stress_return}, "model > control"),
            Check("stress_mdd_worsening_le_0_01pp", stress_mdd is not None and control_stress_mdd is not None and stress_mdd <= control_stress_mdd + 0.01, {"model": stress_mdd, "always_tp4": control_stress_mdd}, "<=+0.01pp"),
            Check("combined_mdd_le_15", base_mdd is not None and base_mdd <= 15.0, base_mdd, "<=15%"),
            Check("combined_ratio_ge_3", base_ratio is not None and base_ratio >= 3.0, base_ratio, ">=3.0"),
            Check("combined_trades_ge_40", trades is not None and trades >= 40, trades, ">=40"),
            Check("weekly_signflip_p_le_0_20", weekly_p is not None and weekly_p <= 0.20, weekly_p, "<=0.20"),
        )
    )

    replay_path = artifact_paths.get("replay_snapshot")
    replay_identical = (
        backtest_path is not None
        and replay_path is not None
        and backtest_path.is_file()
        and replay_path.is_file()
        and backtest_path.read_bytes() == replay_path.read_bytes()
        and result.get("checks", {}).get("byte_replay_identical") is True
    )
    checks.append(
        Check(
            "byte_replay_identical",
            replay_identical,
            {
                "backtest": _sha256(backtest_path) if backtest_path and backtest_path.is_file() else None,
                "replay": _sha256(replay_path) if replay_path and replay_path.is_file() else None,
            },
            "exact bytes equal across two backtest runs",
        )
    )

    passed = all(check.passed for check in checks)
    return {
        "protocol": "pposm_conditional_residual_professor_critic_v1",
        "decision": "PASS" if passed else "FAIL",
        "passed": passed,
        "checks": [check.as_dict() for check in checks],
        "failed_checks": [check.name for check in checks if not check.passed],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate(_load_object(args.result), result_path=args.result)
    rendered = json.dumps(
        report, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False
    ) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
