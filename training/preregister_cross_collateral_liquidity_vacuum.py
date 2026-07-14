"""Support-only preregistration for cross-collateral liquidity vacuum (CLV).

CLV keeps only the continuation mechanism from the outcome-blind CLVR support
study: a flow-confirmed shock whose coin-margined stress-side liquidity voids
relative to USD-M. This module contains no future-return or risk calculation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from training import preregister_cross_collateral_liquidity_void_refill as clvr
from training.preregister_metaorder_fragmentation_impact_curvature import (
    nonoverlapping_schedule,
)


SUPPORT_CALIBRATION_GRID = (0.90, 0.925, 0.95, 0.975, 0.99, 0.995)
PREREGISTRATION_SOURCE = Path(
    "training/preregister_cross_collateral_liquidity_vacuum.py"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/cross-collateral-liquidity-vacuum-preregistration-2026-07-14.md"
)
FEATURE_SOURCE = Path(
    "training/preregister_cross_collateral_liquidity_void_refill.py"
)
SCHEDULER_SOURCE = Path(
    "training/preregister_metaorder_fragmentation_impact_curvature.py"
)


@dataclass(frozen=True)
class Config(clvr.Config):
    output: str = (
        "results/cross_collateral_liquidity_vacuum_support_2026-07-14.json"
    )
    score_quantile: float = 0.975
    minimum_branch_share: float = 0.0


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classify_vacuum(
    features: pd.DataFrame,
    cfg: Config,
    *,
    score_quantile: float | None = None,
) -> pd.DataFrame:
    """Retain only the preregistered positive liquidity-response branch."""
    signal = clvr.classify_features(
        features,
        cfg,
        score_quantile=score_quantile,
    )
    keep = signal["branch"].eq("void")
    signal.loc[~keep, "candidate"] = False
    signal.loc[~keep, "side"] = 0
    signal.loc[~keep, "branch"] = "none"
    signal.loc[~keep, "hold_bars"] = 0
    signal.loc[keep, "branch"] = "vacuum"
    return signal


def support_summary(
    signal: pd.DataFrame,
    market: pd.DataFrame,
    cfg: Config,
) -> dict[str, Any]:
    summary = clvr.support_summary(signal, market, cfg)
    summary.pop("void_share")
    summary.pop("refill_share")
    summary["vacuum_share"] = (
        1.0 if summary["nonoverlap_total"] > 0 else 0.0
    )
    return summary


def run_support(cfg: Config) -> dict[str, Any]:
    market, source = clvr.load_sources(cfg)
    features = clvr.build_features(market, cfg)
    trials: list[dict[str, Any]] = []
    selected_signal: pd.DataFrame | None = None
    selected_support: dict[str, Any] | None = None
    for quantile in SUPPORT_CALIBRATION_GRID:
        signal = classify_vacuum(features, cfg, score_quantile=quantile)
        support = support_summary(signal, market, cfg)
        trials.append(
            {
                "score_quantile": quantile,
                "raw_candidate_count": int(signal["candidate"].sum()),
                **support,
            }
        )
        if quantile == cfg.score_quantile:
            selected_signal = signal
            selected_support = support

    selected = clvr._selected_support_quantile(trials)
    if selected is not None and selected != cfg.score_quantile:
        raise ValueError("configured CLV quantile violates support stopping rule")
    if selected is not None and (
        selected_signal is None or selected_support is None
    ):
        raise AssertionError("configured CLV support was not evaluated")

    if selected is None:
        schedule = pd.DataFrame(columns=["side"])
    else:
        schedule = pd.concat(
            [
                nonoverlapping_schedule(
                    selected_signal,
                    market,
                    start=start,
                    end=end,
                )
                for start, end in (
                    ("2023-01-01", "2023-04-01"),
                    ("2023-04-01", "2023-07-01"),
                    ("2023-07-01", "2023-10-01"),
                    ("2023-10-01", "2024-01-01"),
                )
            ],
            ignore_index=True,
        )

    return {
        "protocol": {
            "name": "CLV — Cross-Collateral Liquidity Vacuum",
            "support_only": True,
            "outcomes_opened_for_clv": False,
            "support_rejected": selected is None,
            "selection_end_exclusive": "2024-01-01 00:00:00",
            "event_clock": (
                "completed 5m USD-M/COIN-M depth bar after a flow-confirmed "
                "30m shock"
            ),
            "signal_availability": (
                "depth medians and market bar complete; enter next 5m open"
            ),
            "action_rule": (
                "coin-margined stress-side liquidity depletes relative to "
                "USD-M; follow completed shock"
            ),
            "candidate_clock": "fixed before any outcome",
            "holding_rule": "12 completed 5m bars; scheduled-open exit",
            "source_gap_policy": (
                "current and prior six depth bars must be complete; future "
                "depth gaps do not cancel an already entered trade"
            ),
            "sealed_windows": ["test2024", "eval2025", "ytd2026"],
        },
        "config": asdict(cfg),
        "frozen_artifacts": {
            "preregistration_source": str(PREREGISTRATION_SOURCE),
            "preregistration_source_sha256": _sha256(PREREGISTRATION_SOURCE),
            "preregistration_document": str(PREREGISTRATION_DOCUMENT),
            "preregistration_document_sha256": _sha256(
                PREREGISTRATION_DOCUMENT
            ),
            "feature_source": str(FEATURE_SOURCE),
            "feature_source_sha256": _sha256(FEATURE_SOURCE),
            "scheduler_source": str(SCHEDULER_SOURCE),
            "scheduler_source_sha256": _sha256(SCHEDULER_SOURCE),
            "depth_manifest_sha256": _sha256(cfg.depth_manifest),
            "market_manifest_sha256": _sha256(cfg.market_manifest),
        },
        "source": source,
        "feature": {
            "response_window_bars": cfg.response_bars,
            "clean_feature_rows": int(features["clean"].sum()),
            "flow_aligned_rows": int(
                (features["clean"] & features["flow_aligned"]).sum()
            ),
            "standardization": (
                "strictly lagged rolling median and recursive MAD; "
                "clip [-12, 12]"
            ),
        },
        "support_calibration": {
            "outcomes_opened_for_clv": False,
            "tested_score_quantiles": list(SUPPORT_CALIBRATION_GRID),
            "all_other_parameters_fixed": True,
            "stopping_rule": (
                "highest tested quantile passing every frozen support floor"
            ),
            "selected_score_quantile": selected,
            "further_support_repairs_allowed": False,
            "trials": trials,
        },
        "raw_candidate_count": (
            int(selected_signal["candidate"].sum())
            if selected_signal is not None and selected is not None
            else None
        ),
        "scheduled_side_counts": {
            "long": int(schedule["side"].gt(0).sum()),
            "short": int(schedule["side"].lt(0).sum()),
        },
        "support": selected_support if selected is not None else None,
        "all_support_gates_pass": bool(
            selected_support is not None
            and selected_support["passes_support"]
            and selected is not None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=Config.output)
    args = parser.parse_args()
    result = run_support(Config(output=args.output))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "outcomes_opened_for_clv": False,
                "selected_score_quantile": result["support_calibration"][
                    "selected_score_quantile"
                ],
                "support_rejected": result["protocol"]["support_rejected"],
                "support": result["support"],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
