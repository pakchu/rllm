"""Frozen pre-Gemma learnability evaluator for CARTA v1.

This file is committed before opening CARTA returns.  It compares causal cheap
policies on 2023 while preserving 2024+ as sealed windows.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from training.causal_adaptive_relational_bandit import (
    ACTION_NAMES,
    INTERACTION_PAIRS,
    BanditConfig,
    build_window,
    constant_predictions,
    evaluate_predictions,
    fit_naive_bayes,
    fit_ridge_policy,
    fit_signature_memory,
    load_state_frame,
    predict_naive_bayes,
    predict_ridge,
    predict_signature_memory,
)


DEFAULT_OUTPUT = (
    "results/causal_adaptive_relational_baseline_selection_2026-07-14.json"
)
WINDOWS: dict[str, tuple[str, str]] = {
    "train_in_sample": ("2020-01-01", "2023-01-01"),
    "select2023": ("2023-01-01", "2024-01-01"),
    "select2023_h1": ("2023-01-01", "2023-07-01"),
    "select2023_h2": ("2023-07-01", "2024-01-01"),
}
LEARNED_POLICIES = ("relational_ridge", "naive_bayes")
CONTROL_POLICIES = (
    "always_abstain",
    "always_follow",
    "always_fade",
    "signature_memory",
    "shuffled_relational_ridge",
)


def _outcome_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["oracle_best_action"]) for row in rows)
    return {
        "rows": len(rows),
        "oracle_best_action_counts": {
            action: int(counts[action]) for action in ACTION_NAMES
        },
        "mean_action_utility": {
            action: float(
                np.mean(
                    [
                        float(row["action_outcomes"][action]["utility"])
                        for row in rows
                    ]
                )
            )
            if rows
            else 0.0
            for action in ACTION_NAMES
        },
    }


def _quarterly_label_release(
    rows: list[dict[str, Any]],
    frame: pd.DataFrame,
) -> dict[str, int]:
    boundaries = (
        ("initial_through_2021", "2020-01-01", "2022-01-01"),
        ("2022_q1", "2022-01-01", "2022-04-01"),
        ("2022_q2", "2022-04-01", "2022-07-01"),
        ("2022_q3", "2022-07-01", "2022-10-01"),
        ("2022_q4", "2022-10-01", "2023-01-01"),
    )
    exit_dates = [pd.Timestamp(frame.loc[row["exit_position"], "date"]) for row in rows]
    return {
        name: int(
            sum(pd.Timestamp(start) <= date < pd.Timestamp(end) for date in exit_dates)
        )
        for name, start, end in boundaries
    }


def _predictor_map(
    train_rows: list[dict[str, Any]],
    cfg: BanditConfig,
) -> tuple[
    dict[str, Callable[[list[dict[str, Any]]], list[str]]],
    dict[str, Any],
]:
    ridge = fit_ridge_policy(train_rows, cfg)
    shuffled = fit_ridge_policy(train_rows, cfg, shuffle_targets=True)
    naive_bayes = fit_naive_bayes(train_rows, cfg)
    memory = fit_signature_memory(train_rows)
    predictors: dict[str, Callable[[list[dict[str, Any]]], list[str]]] = {
        "always_abstain": constant_predictions("ABSTAIN"),
        "always_follow": constant_predictions("FOLLOW"),
        "always_fade": constant_predictions("FADE"),
        "signature_memory": lambda rows: predict_signature_memory(memory, rows),
        "relational_ridge": lambda rows: predict_ridge(ridge, rows),
        "shuffled_relational_ridge": lambda rows: predict_ridge(shuffled, rows),
        "naive_bayes": lambda rows: predict_naive_bayes(naive_bayes, rows),
    }
    model_audit = {
        "ridge_vocabulary_size": len(ridge.vocabulary),
        "ridge_alpha": cfg.ridge_alpha,
        "ridge_minimum_feature_count": cfg.minimum_feature_count,
        "ridge_interaction_pairs": [list(pair) for pair in INTERACTION_PAIRS],
        "ridge_utility_floor": cfg.prediction_utility_floor,
        "shuffled_seed": cfg.seed,
        "naive_bayes_alpha": cfg.naive_bayes_alpha,
        "signature_memory_cells": len(memory),
    }
    return predictors, model_audit


def _control_floor(policies: dict[str, dict[str, Any]]) -> dict[str, float | str]:
    selected = max(
        CONTROL_POLICIES,
        key=lambda name: (
            float(policies[name]["windows"]["select2023"]["cagr_to_strict_mdd"]),
            float(policies[name]["windows"]["select2023"]["absolute_return_pct"]),
            name,
        ),
    )
    metrics = policies[selected]["windows"]["select2023"]
    return {
        "policy": selected,
        "absolute_return_pct": float(metrics["absolute_return_pct"]),
        "cagr_to_strict_mdd": float(metrics["cagr_to_strict_mdd"]),
    }


def baseline_qualification(
    item: dict[str, Any],
    control_floor: dict[str, float | str],
) -> dict[str, Any]:
    selection = item["windows"]["select2023"]
    h1 = item["windows"]["select2023_h1"]
    h2 = item["windows"]["select2023_h2"]
    failures: list[str] = []
    if selection["absolute_return_pct"] <= 0.0:
        failures.append("select2023: non-positive absolute return")
    if selection["cagr_to_strict_mdd"] < 1.0:
        failures.append("select2023: learnability CAGR/strict-MDD below 1")
    if selection["strict_mdd_pct"] > 20.0:
        failures.append("select2023: strict MDD above 20%")
    if selection["trade_count"] < 40:
        failures.append("select2023: fewer than 40 trades")
    if selection["weekly_cluster_sign_flip"]["p_value_one_sided"] >= 0.20:
        failures.append("select2023: weekly-cluster p-value not below 0.20")
    for name, metrics in (("select2023_h1", h1), ("select2023_h2", h2)):
        if metrics["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}: non-positive absolute return")
        if metrics["trade_count"] < 10:
            failures.append(f"{name}: fewer than 10 trades")
    counts = selection["action_counts"]
    if counts["FOLLOW"] < 10 or counts["FADE"] < 10:
        failures.append("select2023: learned actions collapsed below 10 per trade side")
    if selection["long_count"] < 10 or selection["short_count"] < 10:
        failures.append("select2023: executed direction collapsed below 10 per side")
    if selection["absolute_return_pct"] <= float(
        control_floor["absolute_return_pct"]
    ):
        failures.append("select2023: absolute return did not beat strongest control")
    if selection["cagr_to_strict_mdd"] <= float(
        control_floor["cagr_to_strict_mdd"]
    ):
        failures.append("select2023: CAGR/strict-MDD did not beat strongest control")
    return {"qualifies": not failures, "failures": failures}


def select_learnable_policy(
    policies: dict[str, dict[str, Any]],
    control_floor: dict[str, float | str],
) -> dict[str, Any]:
    for name in LEARNED_POLICIES:
        policies[name]["qualification"] = baseline_qualification(
            policies[name], control_floor
        )
    qualified = [
        name
        for name in LEARNED_POLICIES
        if policies[name]["qualification"]["qualifies"]
    ]
    if not qualified:
        return {
            "selected_policy": None,
            "rejected": True,
            "gemma_stage_allowed": False,
            "reason": "no cheap causal CARTA baseline passed the frozen learnability gate",
        }
    selected = max(
        qualified,
        key=lambda name: (
            policies[name]["windows"]["select2023"]["cagr_to_strict_mdd"],
            -policies[name]["windows"]["select2023"]["strict_mdd_pct"],
            name,
        ),
    )
    return {
        "selected_policy": selected,
        "rejected": False,
        "gemma_stage_allowed": True,
        "reason": "causal token family passed the frozen pre-Gemma learnability gate",
    }


def run_evaluation(
    output: str = DEFAULT_OUTPUT,
    cfg: BanditConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or BanditConfig()
    frame, state, source = load_state_frame()
    window_data: dict[str, tuple[pd.DataFrame, list[dict[str, Any]]]] = {
        name: build_window(frame, state, start=start, end=end, cfg=cfg)
        for name, (start, end) in WINDOWS.items()
    }
    train_rows = window_data["train_in_sample"][1]
    predictors, model_audit = _predictor_map(train_rows, cfg)
    policies: dict[str, dict[str, Any]] = {}
    for policy_name, predictor in predictors.items():
        windows: dict[str, Any] = {}
        for window_name, (start, end) in WINDOWS.items():
            schedule, rows = window_data[window_name]
            predictions = predictor(rows)
            windows[window_name] = evaluate_predictions(
                frame,
                schedule,
                predictions,
                start=start,
                end=end,
                cfg=cfg,
            )
        policies[policy_name] = {"windows": windows}

    control_floor = _control_floor(policies)
    selection = select_learnable_policy(policies, control_floor)
    result = {
        "protocol": {
            "name": "CARTA frozen pre-Gemma learnability evaluation",
            "preregistration_commit": "1f8439b",
            "outcomes_opened_for_carta": True,
            "opened_windows": list(WINDOWS),
            "sealed_windows": ["test2024", "eval2025", "ytd2026"],
            "policy_parameters_mutable": False,
            "target_echo_allowed": False,
            "candidate_clock_released_by_abstention": False,
            "entry": "next 5m open",
            "exit": "scheduled open after 72 bars",
            "strict_mdd": "complete held path, favorable extreme first then adverse extreme",
            "cagr": "full wall-clock split including idle cash",
        },
        "bandit_config": asdict(cfg),
        "source": source,
        "model_audit": model_audit,
        "label_release": _quarterly_label_release(train_rows, frame),
        "outcome_audit": {
            name: _outcome_audit(rows) for name, (_, rows) in window_data.items()
        },
        "control_floor": control_floor,
        "policies": policies,
        "selection": selection,
        "output": output,
    }
    return result


def _metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics[key]
        for key in (
            "absolute_return_pct",
            "cagr_pct",
            "strict_mdd_pct",
            "cagr_to_strict_mdd",
            "trade_count",
            "long_count",
            "short_count",
            "action_counts",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_evaluation(args.output)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "selection": result["selection"],
                "control_floor": result["control_floor"],
                "policies": {
                    name: {
                        window: _metric_summary(metrics)
                        for window, metrics in item["windows"].items()
                    }
                    for name, item in result["policies"].items()
                },
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
