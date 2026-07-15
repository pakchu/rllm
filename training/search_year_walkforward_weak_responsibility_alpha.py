"""Search a deterministic walk-forward weak-signal responsibility alpha.

The model has one bounded job: predict whether a fixed funding/premium event is
worth executing with its source-owned exit. Every evaluated year is scored by a
model fitted only on fully purged earlier events. Market features are prior-bar
live features, BOCPD inputs are completed hours, and all sources are physically
truncated before 2024. This search has no path that opens 2024+ data.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.audit_causal_bocpd_pullback_overlay import (
    completed_hour_features,
    exact_hour_map,
)
from training.audit_confirmed_pullback_squeeze_live_parity import (
    PRE2024_WINDOWS,
    _execution_config,
    selection_passes,
)
from training.audit_weak_feature_responsibility_stability import (
    CANDIDATE_SPEC,
    FEATURE_COLUMNS,
    OTHER_COLUMNS,
    PA_COLUMNS,
    Config as ResponsibilityConfig,
    build_design,
)
from training.search_bocpd_state_gated_alpha import _model_output
from training.search_causal_bocpd_pullback_exit_router_alpha import trade_utility
from training.search_causal_weak_tensor_exit_router_alpha import (
    BOCPD_COLUMNS,
    tensor_design,
)
from training.search_funding_premium_external_state_gate_alpha import _frame_hash
from training.search_inventory_purge_reclaim_alpha import (
    ExecutionEngine,
    Trade,
    _schedule_hash,
    equity_stats,
)
from training.search_liveparity_state_feature_interactions import immutable_anchors, slim
from training.search_minimal_stress_weak_action_expert_alpha import (
    event_weights,
    fit_weighted_action_ridge,
)

SELECTION_END = "2024-01-01"
TRAIN_SOURCE_START = "2020-01-01"
DEFAULT_OUTPUT = "results/year_walkforward_weak_responsibility_pre2024_2026-07-15.json"
DEFAULT_DOCS = "docs/year-walkforward-weak-responsibility-pre2024-2026-07-15.md"
NO_BARRIER_BPS = 1_000_000

BASE_RESPONSIBILITY_COLUMNS = [
    "rex_2016_range_width_pct",
    "htf_1w_return_1",
    "funding_rate",
    "htf_1d_range_pos",
    "rex_576_range_pos",
    "htf_1d_return_4",
    "htf_1w_range_pos",
    "volume_zscore",
    "rex_144_range_pos",
    "taker_imbalance",
    "htf_4h_return_4",
    "braid_recent_48h_side",
    "rex_8640_range_pos",
    "premium_index_change",
    "rex_2016_range_pos",
    "kimchi_premium_change",
    "s_age",
    "funding_zscore",
    "nested_recent_24h_side",
]
EVENT_SOURCE_FEATURE = "event_source_signed"
RESPONSIBILITY_COLUMNS = BASE_RESPONSIBILITY_COLUMNS + [EVENT_SOURCE_FEATURE]
HAZARD_GRID = (168, 336)
FORM_GRID = ("linear", "tensor")
RIDGE_GRID = (10.0, 100.0)
MARGIN_GRID = (0.0, 0.001, 0.002)
RISK_LAMBDA = 0.5
WEIGHT_MODE = "year_source"
FOLDS = (
    {
        "name": "predict_2020h2",
        "fit_end": "2020-07-01",
        "predict_start": "2020-07-01",
        "predict_end": "2021-01-01",
    },
    {
        "name": "predict_2021",
        "fit_end": "2021-01-01",
        "predict_start": "2021-01-01",
        "predict_end": "2022-01-01",
    },
    {
        "name": "predict_2022",
        "fit_end": "2022-01-01",
        "predict_start": "2022-01-01",
        "predict_end": "2023-01-01",
    },
    {
        "name": "predict_2023",
        "fit_end": "2023-01-01",
        "predict_start": "2023-01-01",
        "predict_end": "2024-01-01",
    },
)

SEARCH_SPEC: dict[str, Any] = {
    "name": "year_walkforward_weak_responsibility",
    "responsibility": "execute or abstain only",
    "features": RESPONSIBILITY_COLUMNS,
    "bocpd_columns": BOCPD_COLUMNS,
    "hazard_hours": list(HAZARD_GRID),
    "forms": list(FORM_GRID),
    "ridge": list(RIDGE_GRID),
    "margins": list(MARGIN_GRID),
    "risk_lambda": RISK_LAMBDA,
    "weight_mode": WEIGHT_MODE,
    "folds": list(FOLDS),
    "target": "risk-adjusted utility of the exact source-owned executable trade",
    "label_purge": "maximum source-owned path must end before each fit cutoff",
    "grid_cells": len(HAZARD_GRID) * len(FORM_GRID) * len(RIDGE_GRID) * len(MARGIN_GRID),
    "selection_rule": "strict absolute gate plus one adjacent passing margin/ridge cell",
}


@dataclass(frozen=True)
class Config(ResponsibilityConfig):
    output: str = DEFAULT_OUTPUT
    docs_output: str = DEFAULT_DOCS


def _array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _spec_hash() -> str:
    encoded = json.dumps(SEARCH_SPEC, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_fold_plan(folds: tuple[dict[str, str], ...] = FOLDS) -> None:
    previous_fit_end = pd.Timestamp(TRAIN_SOURCE_START)
    for fold in folds:
        fit_end = pd.Timestamp(fold["fit_end"])
        predict_start = pd.Timestamp(fold["predict_start"])
        predict_end = pd.Timestamp(fold["predict_end"])
        if fit_end != predict_start or predict_start >= predict_end:
            raise ValueError(f"non-causal fold boundary: {fold}")
        if fit_end <= previous_fit_end:
            raise ValueError(f"fit cutoffs must expand: {fold}")
        previous_fit_end = fit_end
    if pd.Timestamp(folds[-1]["predict_end"]) > pd.Timestamp(SELECTION_END):
        raise ValueError("fold plan opens 2024+")


def source_action(funding_leg: bool) -> tuple[int, int, int]:
    spec = CANDIDATE_SPEC["funding_exit"] if funding_leg else CANDIDATE_SPEC["premium_exit"]
    return int(spec["hold_bars"]), int(spec["take_bps"]), int(spec["stop_bps"])


def candidate_events(context: dict[str, Any]) -> dict[str, Any]:
    anchors = immutable_anchors(context["base"], int(CANDIDATE_SPEC["anchor_cooldown_bars"]))
    signals = np.flatnonzero(anchors).astype(np.int64)
    funding = np.asarray(context["funding_leg"], dtype=bool)[signals]
    holds = np.asarray([source_action(bool(value))[0] for value in funding], dtype=np.int64)
    return {
        "signals": signals,
        "funding_leg": funding,
        "source_signed": np.where(funding, 1.0, -1.0),
        "max_path_end": signals + 1 + holds,
    }


def event_feature_values(context: dict[str, Any], events: dict[str, Any]) -> np.ndarray:
    frame = pd.DataFrame(context["matrix"], columns=FEATURE_COLUMNS)
    # Restore original NaNs for externally sourced columns so each fold owns its
    # imputation/scaling statistics. State and shifted witness columns are finite.
    for column in set(PA_COLUMNS + OTHER_COLUMNS).intersection(BASE_RESPONSIBILITY_COLUMNS):
        frame[column] = pd.to_numeric(context["features"][column], errors="coerce")
    values = frame[BASE_RESPONSIBILITY_COLUMNS].iloc[events["signals"]].to_numpy(float)
    return np.column_stack([values, events["source_signed"]])


def fit_event_mask(
    dates: pd.Series,
    events: dict[str, Any],
    *,
    fit_end: str,
) -> np.ndarray:
    signals = events["signals"]
    signal_dates = dates.iloc[signals].reset_index(drop=True)
    end_position = np.asarray(events["max_path_end"], dtype=np.int64)
    in_bounds = end_position < len(dates)
    end_dates = np.full(len(signals), np.datetime64("NaT"), dtype="datetime64[ns]")
    end_dates[in_bounds] = dates.iloc[end_position[in_bounds]].to_numpy(dtype="datetime64[ns]")
    return np.asarray(
        (signal_dates >= pd.Timestamp(TRAIN_SOURCE_START)).to_numpy(bool)
        & in_bounds
        & (end_dates < np.datetime64(fit_end)),
        dtype=bool,
    )


def _bocpd_features(
    context: dict[str, Any],
    *,
    hazard_hours: int,
    fit_end: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    cache = context.setdefault("walkforward_bocpd_cache", {})
    key = (int(hazard_hours), str(fit_end))
    if key in cache:
        return cache[key]
    hourly = context.get("completed_hourly")
    if hourly is None:
        hourly = completed_hour_features(context["market"])
        context["completed_hourly"] = hourly
    fit_mask = np.asarray(
        (hourly.index >= pd.Timestamp(TRAIN_SOURCE_START))
        & (hourly.index < pd.Timestamp(fit_end)),
        dtype=bool,
    )
    output, metadata = _model_output(
        hourly,
        fit_mask,
        columns=("ret1", "flow24"),
        secondary_index=1,
        hazard_lambda=int(hazard_hours),
    )
    mapped = exact_hour_map(context["dates"], output)
    values = mapped[BOCPD_COLUMNS].to_numpy(float)
    cache[key] = (values, metadata)
    return values, metadata


def _fit_labels(
    context: dict[str, Any],
    cfg: Config,
    events: dict[str, Any],
    fit_mask: np.ndarray,
) -> np.ndarray:
    engine: ExecutionEngine = context["walkforward_engine"]
    labels: list[float] = []
    for row in np.flatnonzero(fit_mask):
        signal = int(events["signals"][row])
        hold, take, stop = source_action(bool(events["funding_leg"][row]))
        trade = engine.trade_at(signal, 1, hold, take, stop)
        if trade is None:
            raise RuntimeError("purged source-owned trade is not executable")
        labels.append(
            trade_utility(
                trade,
                RISK_LAMBDA,
                leverage=float(cfg.leverage),
                cost=float(cfg.fee_rate + cfg.slippage_rate),
            )
        )
    return np.asarray(labels, dtype=float)[:, None]


def walkforward_predictions(
    context: dict[str, Any],
    cfg: Config,
    events: dict[str, Any],
    raw_features: np.ndarray,
    *,
    hazard_hours: int,
    form: str,
    ridge: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    signal_dates = context["dates"].iloc[events["signals"]].reset_index(drop=True)
    predictions = np.full(len(events["signals"]), np.nan, dtype=float)
    metadata: list[dict[str, Any]] = []
    for fold in FOLDS:
        fit_mask = fit_event_mask(context["dates"], events, fit_end=fold["fit_end"])
        predict_mask = np.asarray(
            (signal_dates >= pd.Timestamp(fold["predict_start"]))
            & (signal_dates < pd.Timestamp(fold["predict_end"])),
            dtype=bool,
        )
        if int(fit_mask.sum()) < 30:
            raise RuntimeError(f"insufficient purged events for {fold['name']}")
        bocpd, bocpd_model = _bocpd_features(
            context,
            hazard_hours=int(hazard_hours),
            fit_end=fold["fit_end"],
        )
        design, scaler = tensor_design(
            raw_features,
            bocpd[events["signals"]],
            fit_mask,
            form=str(form),
        )
        labels = _fit_labels(context, cfg, events, fit_mask)
        weights = event_weights(
            signal_dates,
            events["source_signed"],
            fit_mask,
            WEIGHT_MODE,
        )
        prediction, ridge_model = fit_weighted_action_ridge(
            design,
            labels,
            fit_mask,
            weights,
            ridge=float(ridge),
        )
        predictions[predict_mask] = prediction[predict_mask, 0]
        metadata.append(
            {
                "fold": fold,
                "fit_events": int(fit_mask.sum()),
                "predict_events": int(predict_mask.sum()),
                "dimensions": int(scaler["dimensions"]),
                "coefficient_l2": float(ridge_model["coefficient_l2"]),
                "bocpd_standardization": {
                    "mean": bocpd_model["train_standardization_mean"],
                    "std": bocpd_model["train_standardization_std"],
                },
            }
        )
    return predictions, metadata


def schedule_window(
    context: dict[str, Any],
    events: dict[str, Any],
    selected_events: np.ndarray,
    *,
    start: str,
    end: str,
) -> list[Trade]:
    dates = context["dates"]
    signals = events["signals"]
    signal_dates = dates.iloc[signals].reset_index(drop=True)
    period = np.asarray(
        (signal_dates >= pd.Timestamp(start)) & (signal_dates < pd.Timestamp(end)),
        dtype=bool,
    )
    trades: list[Trade] = []
    next_allowed = 0
    engine: ExecutionEngine = context["walkforward_engine"]
    for row in np.flatnonzero(np.asarray(selected_events, dtype=bool) & period):
        signal = int(signals[row])
        if signal < next_allowed:
            continue
        hold, take, stop = source_action(bool(events["funding_leg"][row]))
        trade = engine.trade_at(signal, 1, hold, take, stop)
        if trade is None or not (pd.Timestamp(start) <= dates.iloc[trade.exit_position] < pd.Timestamp(end)):
            continue
        trades.append(trade)
        next_allowed = trade.exit_position + 1
    return trades


def schedules_and_stats(
    context: dict[str, Any],
    cfg: Config,
    events: dict[str, Any],
    selected_events: np.ndarray,
) -> tuple[dict[str, list[Trade]], dict[str, dict[str, Any]]]:
    schedules = {
        name: schedule_window(context, events, selected_events, start=start, end=end)
        for name, (start, end) in PRE2024_WINDOWS.items()
    }
    execution_cfg = _execution_config(cfg, cfg.leverage)
    stats = {
        name: slim(equity_stats(schedules[name], start=start, end=end, cfg=execution_cfg))
        for name, (start, end) in PRE2024_WINDOWS.items()
    }
    return schedules, stats


def rank_stats(stats: dict[str, dict[str, Any]]) -> list[float]:
    ratios = [float(stats[name]["cagr_to_strict_mdd"]) for name in ("train", "select_2023", "pre_2024")]
    return [
        float(min(ratios)),
        float(np.median(ratios)),
        float(stats["pre_2024"]["trades"]),
        float(stats["pre_2024"]["cagr_pct"]),
    ]


def mark_adjacent_stability(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        spec = row["spec"]
        neighbors = []
        for other in rows:
            other_spec = other["spec"]
            if other is row or other_spec["hazard_hours"] != spec["hazard_hours"] or other_spec["form"] != spec["form"]:
                continue
            margin_neighbor = other_spec["ridge"] == spec["ridge"] and abs(MARGIN_GRID.index(other_spec["margin"]) - MARGIN_GRID.index(spec["margin"])) == 1
            ridge_neighbor = other_spec["margin"] == spec["margin"] and other_spec["ridge"] != spec["ridge"]
            if margin_neighbor or ridge_neighbor:
                neighbors.append(other)
        row["adjacent_passing_cells"] = sum(int(other["selection_passed"]) for other in neighbors)
        row["accepted"] = bool(row["selection_passed"] and row["adjacent_passing_cells"] > 0)


def search(context: dict[str, Any], cfg: Config) -> list[dict[str, Any]]:
    events = candidate_events(context)
    raw_features = event_feature_values(context, events)
    rows: list[dict[str, Any]] = []
    for hazard, form, ridge in itertools.product(HAZARD_GRID, FORM_GRID, RIDGE_GRID):
        prediction, model = walkforward_predictions(
            context,
            cfg,
            events,
            raw_features,
            hazard_hours=int(hazard),
            form=str(form),
            ridge=float(ridge),
        )
        for margin in MARGIN_GRID:
            selected = np.isfinite(prediction) & (prediction >= float(margin))
            schedules, stats = schedules_and_stats(context, cfg, events, selected)
            rows.append(
                {
                    "spec": {
                        "hazard_hours": int(hazard),
                        "form": str(form),
                        "ridge": float(ridge),
                        "margin": float(margin),
                    },
                    "model": model,
                    "prediction_hash": _array_hash(np.nan_to_num(prediction, nan=-999.0)),
                    "selected_events": int(selected.sum()),
                    "selection_passed": bool(selection_passes(stats)),
                    "selection_schedule_hashes": {
                        name: _schedule_hash(trades) for name, trades in schedules.items()
                    },
                    "rank": rank_stats(stats),
                    "stats": stats,
                }
            )
    mark_adjacent_stability(rows)
    rows.sort(
        key=lambda row: (
            row["accepted"],
            row["selection_passed"],
            *row["rank"],
        ),
        reverse=True,
    )
    return rows


def _implementation_hash() -> str:
    functions = (
        validate_fold_plan,
        source_action,
        candidate_events,
        event_feature_values,
        fit_event_mask,
        _bocpd_features,
        _fit_labels,
        walkforward_predictions,
        schedule_window,
        schedules_and_stats,
        rank_stats,
        mark_adjacent_stability,
        search,
        build_design,
        completed_hour_features,
        exact_hour_map,
        _model_output,
        tensor_design,
        event_weights,
        fit_weighted_action_ridge,
        ExecutionEngine.trade_at,
        equity_stats,
    )
    return hashlib.sha256("\n\n".join(inspect.getsource(fn) for fn in functions).encode()).hexdigest()


def _metric(row: dict[str, Any]) -> str:
    return f"{row['absolute_return_pct']:.2f}% / {row['cagr_pct']:.2f}% / {row['strict_mdd_pct']:.2f}% / {row['cagr_to_strict_mdd']:.2f} / {row['trades']}"


def _atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(target)


def _write_docs(path: str | Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Year-walkforward weak responsibility search — 2026-07-15",
        "",
        f"**Decision: {payload['decision']}**. 2024+ remains sealed.",
        "",
        "Metric: absolute return / CAGR / strict MDD / CAGR-to-strict-MDD / trades.",
        "",
        "- Each scored year uses only fully purged earlier labels; no held-out year trains an earlier year.",
        "- Ridge is deterministic and weighted by year/source. The model only executes or abstains.",
        "- Funding events keep 48h/TP4/no-stop; premium events keep 12h/no-TP/SL3.",
        "- A candidate needs the strict absolute gate and one adjacent passing ridge/margin cell.",
        "- Market/funding/premium/OI/spot-premium sources are physically truncated before 2024.",
        "",
        f"Passing cells: `{payload['passing_cells']}/{payload['grid_cells']}`; adjacent-stable candidates: `{payload['accepted_cells']}/{payload['grid_cells']}`.",
        "",
        "| Rank | Spec | Train | 2023 | 2023 H2 | Pre-2024 | Pass | Adjacent |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(payload["rows"][:10], start=1):
        stats = row["stats"]
        lines.append(
            f"| {index} | `{json.dumps(row['spec'], sort_keys=True)}` | {_metric(stats['train'])} | {_metric(stats['select_2023'])} | {_metric(stats['select_2023_h2'])} | {_metric(stats['pre_2024'])} | {row['selection_passed']} | {row['adjacent_passing_cells']} |"
        )
    lines += ["", f"Implementation hash: `{payload['implementation_hash']}`", ""]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")


def run(cfg: Config) -> dict[str, Any]:
    validate_fold_plan()
    context = build_design(cfg)
    if len(context["dates"]) and context["dates"].max() >= pd.Timestamp(SELECTION_END):
        raise RuntimeError("selection source was not physically truncated")
    context["walkforward_engine"] = ExecutionEngine(
        context["market"],
        context["funding"],
        _execution_config(cfg, cfg.leverage),
    )
    rows = search(context, cfg)
    accepted = [row for row in rows if row["accepted"]]
    payload: dict[str, Any] = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "phase": "pre_2024_search",
        "oos_opened": False,
        "sealed_windows": ["2024+"],
        "selection_end_exclusive": SELECTION_END,
        "config": asdict(cfg),
        "search_spec": SEARCH_SPEC,
        "spec_hash": _spec_hash(),
        "implementation_hash": _implementation_hash(),
        "source_prefix_hashes": context["source_hashes"],
        "selected_feature_prefix_hash": _frame_hash(
            pd.DataFrame(context["matrix"], columns=FEATURE_COLUMNS)[BASE_RESPONSIBILITY_COLUMNS].assign(
                date=context["dates"].to_numpy()
            )
        ),
        "grid_cells": len(rows),
        "passing_cells": sum(int(row["selection_passed"]) for row in rows),
        "accepted_cells": len(accepted),
        "decision": "candidate_found_requires_stability_audit" if accepted else "reject",
        "candidate": accepted[0] if accepted else None,
        "rows": rows,
    }
    _atomic_write_json(cfg.output, payload)
    _write_docs(cfg.docs_output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=Config.input_csv)
    parser.add_argument("--funding-csv", default=Config.funding_csv)
    parser.add_argument("--premium-csv", default=Config.premium_csv)
    parser.add_argument("--spot-premium-csv", default=Config.spot_premium_csv)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--docs-output", default=Config.docs_output)
    return parser.parse_args()


def main() -> None:
    payload = run(Config(**vars(parse_args())))
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "grid_cells": payload["grid_cells"],
                "passing_cells": payload["passing_cells"],
                "accepted_cells": payload["accepted_cells"],
                "top": payload["rows"][0] if payload["rows"] else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
