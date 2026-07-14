"""Post-hoc CARTA v1 temporal-transfer failure decomposition.

This diagnostic cannot repair or promote CARTA.  It uses only the already
opened 2020-2023 windows and keeps every 2024+ window sealed.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from training.causal_adaptive_relational_bandit import (
    ACTION_NAMES,
    BanditConfig,
    build_window,
    evaluate_predictions,
    fit_naive_bayes,
    fit_ridge_policy,
    load_state_frame,
    predict_naive_bayes,
    predict_ridge,
)
from training.preregister_causal_adaptive_relational_tokens import TOKEN_COLUMNS


SELECTION_RESULT = Path(
    "results/causal_adaptive_relational_baseline_selection_2026-07-14.json"
)
SELECTION_RESULT_SHA256 = (
    "b17ef30fd97bc8054a49e42c84d406439c547b97fbd8fb94f0baf59625c55a75"
)
OUTPUT = Path("results/carta_transfer_failure_diagnostic_2026-07-14.json")
PERIODS = {
    "2020": ("2020-01-01", "2021-01-01"),
    "2021": ("2021-01-01", "2022-01-01"),
    "2022": ("2022-01-01", "2023-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
}
TRAIN_SPECS = {
    "2020_only": ("2020",),
    "2021_only": ("2021",),
    "2022_only": ("2022",),
    "2020_2021": ("2020", "2021"),
    "2021_2022": ("2021", "2022"),
    "2020_2022": ("2020", "2021", "2022"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_rejection() -> dict[str, Any]:
    if _sha256(SELECTION_RESULT) != SELECTION_RESULT_SHA256:
        raise ValueError("CARTA selection result changed after rejection")
    result = json.loads(SELECTION_RESULT.read_text())
    if result.get("selection", {}).get("rejected") is not True:
        raise ValueError("CARTA result is not a frozen rejection")
    if result.get("selection", {}).get("gemma_stage_allowed") is not False:
        raise ValueError("CARTA rejection unexpectedly permits Gemma")
    if result.get("protocol", {}).get("sealed_windows") != [
        "test2024",
        "eval2025",
        "ytd2026",
    ]:
        raise ValueError("CARTA rejection did not preserve sealed windows")
    return result


def _annual_label_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        action: sum(row["oracle_best_action"] == action for row in rows)
        for action in ACTION_NAMES
    }
    return {
        "candidate_count": len(rows),
        "oracle_best_action_counts": counts,
        "mean_utility": {
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


def token_action_effects(
    rows: list[dict[str, Any]],
    *,
    minimum_count: int = 5,
) -> dict[str, dict[str, float | int]]:
    annual_mean = {
        action: float(
            np.mean(
                [float(row["action_outcomes"][action]["utility"]) for row in rows]
            )
        )
        for action in ("FOLLOW", "FADE")
    }
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for action in ("FOLLOW", "FADE"):
            utility = float(row["action_outcomes"][action]["utility"])
            for field in TOKEN_COLUMNS:
                key = f"{action}|{field}={row['tokens'][field]}"
                values[key].append(utility)
    return {
        key: {
            "count": len(cell),
            "mean_utility": float(np.mean(cell)),
            "effect_vs_action_mean": float(
                np.mean(cell) - annual_mean[key.split("|", 1)[0]]
            ),
        }
        for key, cell in values.items()
        if len(cell) >= minimum_count
    }


def effect_transfer(
    source: dict[str, dict[str, float | int]],
    target: dict[str, dict[str, float | int]],
) -> dict[str, Any]:
    shared = sorted(set(source) & set(target))
    if len(shared) < 2:
        correlation = 0.0
    else:
        left = np.asarray(
            [float(source[key]["effect_vs_action_mean"]) for key in shared]
        )
        right = np.asarray(
            [float(target[key]["effect_vs_action_mean"]) for key in shared]
        )
        correlation = (
            float(np.corrcoef(left, right)[0, 1])
            if np.std(left) > 0.0 and np.std(right) > 0.0
            else 0.0
        )
    sign_agreement = (
        float(
            np.mean(
                [
                    np.sign(float(source[key]["effect_vs_action_mean"]))
                    == np.sign(float(target[key]["effect_vs_action_mean"]))
                    for key in shared
                ]
            )
        )
        if shared
        else 0.0
    )
    return {
        "shared_supported_token_action_cells": len(shared),
        "pearson_effect_correlation": correlation,
        "effect_sign_agreement": sign_agreement,
    }


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


def run_diagnostic() -> dict[str, Any]:
    selection = _verify_rejection()
    cfg = BanditConfig()
    frame, state, source = load_state_frame()
    annual = {
        year: build_window(frame, state, start=start, end=end, cfg=cfg)
        for year, (start, end) in PERIODS.items()
    }
    annual_rows = {year: rows for year, (_, rows) in annual.items()}
    effects = {
        year: token_action_effects(rows) for year, rows in annual_rows.items()
    }
    pooled_train_rows = [
        row for year in ("2020", "2021", "2022") for row in annual_rows[year]
    ]
    pooled_effects = token_action_effects(pooled_train_rows)

    schedule_2023, rows_2023 = annual["2023"]
    transfer_models: dict[str, Any] = {}
    for spec, years in TRAIN_SPECS.items():
        train_rows = [row for year in years for row in annual_rows[year]]
        ridge = fit_ridge_policy(train_rows, cfg)
        naive_bayes = fit_naive_bayes(train_rows, cfg)
        ridge_predictions = predict_ridge(ridge, rows_2023)
        nb_predictions = predict_naive_bayes(naive_bayes, rows_2023)
        transfer_models[spec] = {
            "train_rows": len(train_rows),
            "relational_ridge": _metric_summary(
                evaluate_predictions(
                    frame,
                    schedule_2023,
                    ridge_predictions,
                    start="2023-01-01",
                    end="2024-01-01",
                    cfg=cfg,
                )
            ),
            "naive_bayes": _metric_summary(
                evaluate_predictions(
                    frame,
                    schedule_2023,
                    nb_predictions,
                    start="2023-01-01",
                    end="2024-01-01",
                    cfg=cfg,
                )
            ),
        }

    return {
        "protocol": {
            "name": "CARTA v1 post-hoc temporal-transfer diagnostic",
            "selection_result_sha256": SELECTION_RESULT_SHA256,
            "opened_windows_only": list(PERIODS),
            "sealed_windows_still_unopened": [
                "test2024",
                "eval2025",
                "ytd2026",
            ],
            "may_repair_or_promote_carta": False,
            "gemma_stage_allowed": False,
        },
        "selection_verdict": selection["selection"],
        "source": source,
        "annual_label_audit": {
            year: _annual_label_audit(rows) for year, rows in annual_rows.items()
        },
        "token_effect_transfer": {
            "2020_to_2021": effect_transfer(effects["2020"], effects["2021"]),
            "2021_to_2022": effect_transfer(effects["2021"], effects["2022"]),
            "2022_to_2023": effect_transfer(effects["2022"], effects["2023"]),
            "pooled_2020_2022_to_2023": effect_transfer(
                pooled_effects, effects["2023"]
            ),
        },
        "recent_history_model_transfer_to_2023": transfer_models,
    }


def main() -> None:
    result = run_diagnostic()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "annual_label_audit": result["annual_label_audit"],
                "token_effect_transfer": result["token_effect_transfer"],
                "recent_history_model_transfer_to_2023": result[
                    "recent_history_model_transfer_to_2023"
                ],
                "output": str(OUTPUT),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
