"""Leak-safe sparse REX event critic selected before 2024.

The REX candidate generator is fixed on 2021-2022 feature history.  This search
fits compact numeric critics on completed 2021-2022 paths, selects a Top-10 on
2023 full/H1/H2 performance, writes the manifest, and only then opens the
2024-2026 candidate files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import MISSING, asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from training.economic_action_backtest import EconomicActionBacktestConfig, strict_backtest_actions
from training.strict_bar_backtest import load_market_bars


FIT_END = pd.Timestamp("2023-01-01")
SELECT_END = pd.Timestamp("2024-01-01")
WINDOWS = {
    "select2023": (pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")),
    "select2023_h1": (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-07-01")),
    "select2023_h2": (pd.Timestamp("2023-07-01"), pd.Timestamp("2024-01-01")),
    "test2024": (pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")),
    "eval2025": (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")),
    "ytd2026": (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-06-02")),
}


@dataclass(frozen=True)
class RexPre2024MlConfig:
    train_jsonl: str
    test_jsonl: str
    eval_jsonl: str
    market_csv: str
    output: str
    manifest_output: str
    top_n: int = 10
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    hold_bars: int = 144
    random_state: int = 42
    selection_strategy: str = "global"


MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {"name": "ridge_u_a10", "kind": "ridge", "target": "utility", "alpha": 10.0},
    {"name": "ridge_u_a100", "kind": "ridge", "target": "utility", "alpha": 100.0},
    {"name": "hgb_u_d2_l40", "kind": "hgb", "target": "utility", "max_depth": 2, "min_samples_leaf": 40, "l2": 1.0},
    {"name": "hgb_u_d3_l60", "kind": "hgb", "target": "utility", "max_depth": 3, "min_samples_leaf": 60, "l2": 3.0},
    {"name": "extra_u_d4_l20", "kind": "extra", "target": "utility", "max_depth": 4, "min_samples_leaf": 20},
    {"name": "extra_u_d6_l30", "kind": "extra", "target": "utility", "max_depth": 6, "min_samples_leaf": 30},
    {"name": "hgb_take_d2_l40", "kind": "hgb", "target": "take", "max_depth": 2, "min_samples_leaf": 40, "l2": 1.0},
    {"name": "extra_take_d4_l20", "kind": "extra", "target": "take", "max_depth": 4, "min_samples_leaf": 20},
)


def read_jsonl(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        for name, value in (row.get("feature_snapshot") or {}).items():
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                names.add(str(name))
    return sorted(names)


def fit_medians(rows: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    raw = np.asarray(
        [[float((row.get("feature_snapshot") or {}).get(name, np.nan)) for name in names] for row in rows],
        dtype=np.float64,
    )
    medians = np.nanmedian(raw, axis=0)
    return np.where(np.isfinite(medians), medians, 0.0)


def feature_matrix(rows: list[dict[str, Any]], names: list[str], medians: np.ndarray) -> np.ndarray:
    raw = np.asarray(
        [[float((row.get("feature_snapshot") or {}).get(name, np.nan)) for name in names] for row in rows],
        dtype=np.float64,
    )
    bad = ~np.isfinite(raw)
    if np.any(bad):
        raw[bad] = np.broadcast_to(medians, raw.shape)[bad]
    return raw


def target_vector(rows: list[dict[str, Any]], target: str) -> np.ndarray:
    if target == "take":
        return np.asarray([1.0 if str(row.get("target")) == "TAKE" else 0.0 for row in rows], dtype=np.float64)
    return np.asarray([float((row.get("reward") or {}).get("utility", 0.0) or 0.0) for row in rows], dtype=np.float64)


def completed_before(row: dict[str, Any], market_dates: pd.Series, cutoff: pd.Timestamp, hold_bars: int) -> bool:
    exit_pos = int(row["signal_pos"]) + 1 + int(hold_bars)
    return 0 <= exit_pos < len(market_dates) and pd.Timestamp(market_dates.iloc[exit_pos]) < cutoff


def select_highest_per_signal(rows: list[dict[str, Any]], scores: np.ndarray) -> list[dict[str, Any]]:
    best: dict[int, tuple[float, dict[str, Any]]] = {}
    for row, score in zip(rows, scores):
        pos = int(row["signal_pos"])
        previous = best.get(pos)
        if previous is None or float(score) > previous[0]:
            best[pos] = (float(score), row)
    return [best[pos][1] for pos in sorted(best)]


def prediction_rows(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    *,
    threshold: float,
    side: str,
    family: str,
    market_dates: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    hold_bars: int,
) -> list[dict[str, Any]]:
    eligible_rows: list[dict[str, Any]] = []
    eligible_scores: list[float] = []
    for row, score in zip(rows, scores):
        date = pd.Timestamp(str(row["date"]))
        if not (start <= date < end) or float(score) < float(threshold):
            continue
        row_side = str(row.get("side", "")).lower()
        row_family = str(row.get("family", ""))
        if side != "both" and row_side != side:
            continue
        if family != "both" and row_family != family:
            continue
        if not completed_before(row, market_dates, end, hold_bars):
            continue
        eligible_rows.append(row)
        eligible_scores.append(float(score))
    chosen = select_highest_per_signal(eligible_rows, np.asarray(eligible_scores, dtype=np.float64))
    return [
        {
            "date": str(row["date"]),
            "signal_pos": int(row["signal_pos"]),
            "prediction": {
                "gate": "TRADE",
                "side": str(row["side"]).upper(),
                "hold_bars": int(hold_bars),
            },
        }
        for row in chosen
    ]


def empty_metrics() -> dict[str, Any]:
    return {
        "return_pct": 0.0,
        "cagr_pct": 0.0,
        "strict_mdd_pct": 0.0,
        "ratio": 0.0,
        "trades": 0,
        "p_value": 1.0,
    }


def score_window(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    policy: dict[str, Any],
    market: pd.DataFrame,
    cfg: RexPre2024MlConfig,
    window: tuple[pd.Timestamp, pd.Timestamp],
) -> dict[str, Any]:
    start, end = window
    selected = prediction_rows(
        rows,
        scores,
        threshold=float(policy["score_threshold"]),
        side=str(policy["side"]),
        family=str(policy["family"]),
        market_dates=market["date"],
        start=start,
        end=end,
        hold_bars=int(cfg.hold_bars),
    )
    if not selected:
        return empty_metrics()
    report = strict_backtest_actions(
        selected,
        market,
        EconomicActionBacktestConfig(
            annualization_start=str(start),
            annualization_end=str(end),
            leverage=float(cfg.leverage),
            fee_rate=float(cfg.fee_rate),
            slippage_rate=float(cfg.slippage_rate),
            entry_delay_bars=1,
            max_hold_bars=int(cfg.hold_bars),
        ),
    )
    sim = report["sim"]
    stats = report["trade_stats"]
    return {
        "return_pct": float(sim["ret_pct"]),
        "cagr_pct": float(sim["cagr_pct"]),
        "strict_mdd_pct": float(sim["strict_mdd_pct"]),
        "ratio": float(sim["cagr_to_strict_mdd"]),
        "trades": int(sim["trade_entries"]),
        "p_value": float(stats["p_value_mean_ret_approx"]),
    }


def build_model(spec: dict[str, Any], random_state: int):
    if spec["kind"] == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=float(spec["alpha"])))
    if spec["kind"] == "hgb":
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=160,
            max_depth=int(spec["max_depth"]),
            min_samples_leaf=int(spec["min_samples_leaf"]),
            l2_regularization=float(spec["l2"]),
            random_state=int(random_state),
        )
    return ExtraTreesRegressor(
        n_estimators=300,
        max_depth=int(spec["max_depth"]),
        min_samples_leaf=int(spec["min_samples_leaf"]),
        max_features=0.7,
        n_jobs=-1,
        random_state=int(random_state),
    )


def prefix_hash(rows: list[dict[str, Any]], names: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(names, separators=(",", ":")).encode())
    for row in rows:
        payload = {
            "date": str(row["date"]),
            "signal_pos": int(row["signal_pos"]),
            "features": {name: (row.get("feature_snapshot") or {}).get(name) for name in names},
        }
        digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True).encode())
    return digest.hexdigest()


def manifest_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def select_manifest_candidates(
    candidates: list[dict[str, Any]],
    *,
    top_n: int,
    strategy: str,
    rank_key,
) -> list[dict[str, Any]]:
    ranked = sorted(candidates, key=rank_key, reverse=True)
    if strategy == "global":
        return ranked[: int(top_n)]
    if strategy != "side_balanced":
        raise ValueError(f"unknown selection strategy: {strategy}")
    long_slots = (int(top_n) + 1) // 2
    short_slots = int(top_n) - long_slots
    longs = [row for row in ranked if row["side"] == "long"][:long_slots]
    shorts = [row for row in ranked if row["side"] == "short"][:short_slots]
    return longs + shorts


def run(cfg: RexPre2024MlConfig) -> dict[str, Any]:
    market = load_market_bars(cfg.market_csv)

    # Physical phase 1: only the pre-2024 JSONL is opened here.
    pre_rows = read_jsonl(cfg.train_jsonl)
    pre_dates = np.asarray([np.datetime64(str(row["date"])) for row in pre_rows])
    fit_mask = pre_dates < np.datetime64(FIT_END)
    select_mask = (pre_dates >= np.datetime64(FIT_END)) & (pre_dates < np.datetime64(SELECT_END))
    fit_rows = [row for row, keep in zip(pre_rows, fit_mask) if keep and completed_before(row, market["date"], FIT_END, cfg.hold_bars)]
    select_rows = [row for row, keep in zip(pre_rows, select_mask) if keep]
    names = feature_names(fit_rows)
    medians = fit_medians(fit_rows, names)
    x_fit = feature_matrix(fit_rows, names, medians)
    x_select = feature_matrix(select_rows, names, medians)

    models: dict[str, Any] = {}
    select_scores: dict[str, np.ndarray] = {}
    specs = {str(spec["name"]): dict(spec) for spec in MODEL_SPECS}
    for spec in MODEL_SPECS:
        model = build_model(spec, cfg.random_state)
        model.fit(x_fit, target_vector(fit_rows, str(spec["target"])))
        name = str(spec["name"])
        models[name] = model
        select_scores[name] = np.asarray(model.predict(x_select), dtype=np.float64)

    candidates: list[dict[str, Any]] = []
    for model_name, scores in select_scores.items():
        for side in ("both", "long", "short"):
            for family in ("both", "rex_htf_pullback_reclaim", "rex_htf_pullback_resume"):
                eligible = np.asarray(
                    [
                        (side == "both" or str(row.get("side", "")).lower() == side)
                        and (family == "both" or str(row.get("family", "")) == family)
                        for row in select_rows
                    ],
                    dtype=bool,
                )
                if int(eligible.sum()) < 30:
                    continue
                for quantile in (0.70, 0.80, 0.90):
                    threshold = float(np.quantile(scores[eligible], quantile))
                    policy = {
                        "model": model_name,
                        "model_spec": specs[model_name],
                        "score_quantile": quantile,
                        "score_threshold": threshold,
                        "side": side,
                        "family": family,
                    }
                    full = score_window(select_rows, scores, policy, market, cfg, WINDOWS["select2023"])
                    h1 = score_window(select_rows, scores, policy, market, cfg, WINDOWS["select2023_h1"])
                    h2 = score_window(select_rows, scores, policy, market, cfg, WINDOWS["select2023_h2"])
                    positive_halves = int(h1["cagr_pct"] > 0.0) + int(h2["cagr_pct"] > 0.0)
                    candidate = {
                        **policy,
                        "select2023": full,
                        "select2023_halves": [h1, h2],
                        "selection_score": {
                            "positive_halves": positive_halves,
                            "min_half_ratio": min(float(h1["ratio"]), float(h2["ratio"])),
                            "full_ratio": float(full["ratio"]),
                            "min_half_trades": min(int(h1["trades"]), int(h2["trades"])),
                        },
                    }
                    candidates.append(candidate)

    def rank_key(row: dict[str, Any]) -> tuple[float, ...]:
        score = row["selection_score"]
        full = row["select2023"]
        return (
            float(score["positive_halves"]),
            float(score["min_half_trades"] >= 8),
            float(score["min_half_ratio"]),
            float(score["full_ratio"]),
            -float(full["p_value"]),
            float(full["trades"]),
        )

    selected = select_manifest_candidates(
        candidates,
        top_n=int(cfg.top_n),
        strategy=str(cfg.selection_strategy),
        rank_key=rank_key,
    )
    phase1_hash = prefix_hash(pre_rows, names)
    manifest_core = {
        "phase": "pre_future_manifest",
        "fit_window": ["2021-01-01", "2023-01-01"],
        "selection_window": ["2023-01-01", "2024-01-01"],
        "selection_strategy": str(cfg.selection_strategy),
        "feature_hash": phase1_hash,
        "feature_names": names,
        "policies": [
            {key: row[key] for key in ("model", "model_spec", "score_quantile", "score_threshold", "side", "family")}
            for row in selected
        ],
    }
    frozen_hash = manifest_hash(manifest_core)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **manifest_core,
        "manifest_hash": frozen_hash,
    }
    manifest_path = Path(cfg.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    # Physical phase 2: future files are opened only after the manifest exists.
    future_sets = {
        "test2024": read_jsonl(cfg.test_jsonl),
        "eval2025_2026": read_jsonl(cfg.eval_jsonl),
    }
    for row in selected:
        model_name = str(row["model"])
        model = models[model_name]
        test_rows = future_sets["test2024"]
        eval_rows = future_sets["eval2025_2026"]
        test_scores = np.asarray(model.predict(feature_matrix(test_rows, names, medians)), dtype=np.float64)
        eval_scores = np.asarray(model.predict(feature_matrix(eval_rows, names, medians)), dtype=np.float64)
        row["test2024"] = score_window(test_rows, test_scores, row, market, cfg, WINDOWS["test2024"])
        row["eval2025"] = score_window(eval_rows, eval_scores, row, market, cfg, WINDOWS["eval2025"])
        row["ytd2026"] = score_window(eval_rows, eval_scores, row, market, cfg, WINDOWS["ytd2026"])
        row["passes_alpha_target"] = all(
            row[name]["cagr_pct"] > 0.0
            and row[name]["ratio"] >= 3.0
            and row[name]["strict_mdd_pct"] <= 15.0
            and row[name]["trades"] >= 30
            for name in ("test2024", "eval2025")
        )
        row["passes_live_target"] = bool(
            row["passes_alpha_target"]
            and row["ytd2026"]["cagr_pct"] > 0.0
            and row["ytd2026"]["ratio"] >= 3.0
            and row["ytd2026"]["strict_mdd_pct"] <= 15.0
            and row["ytd2026"]["trades"] >= 10
        )

    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "protocol": {
            "candidate_threshold_fit": "2021-2022 only in source dataset",
            "model_fit": "completed paths strictly before 2023-01-01",
            "selection": "2023 full/H1/H2 Top-10",
            "manifest_written_before_future": True,
            "future_windows": {name: [str(a), str(b)] for name, (a, b) in WINDOWS.items() if name in {"test2024", "eval2025", "ytd2026"}},
            "cagr": "full configured calendar window including idle time",
            "strict_mdd": "worst-order favorable-to-adverse OHLC high-water path drawdown",
        },
        "input": {
            "fit_rows": len(fit_rows),
            "select_rows": len(select_rows),
            "features": len(names),
            "tested": len(candidates),
        },
        "manifest_hash": frozen_hash,
        "selected": selected,
        "alpha_qualifiers": [row for row in selected if row["passes_alpha_target"]],
        "live_qualifiers": [row for row in selected if row["passes_live_target"]],
    }
    output_path = Path(cfg.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> RexPre2024MlConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in RexPre2024MlConfig.__dataclass_fields__.values():
        required = field.default is MISSING and field.default_factory is MISSING
        parser.add_argument("--" + field.name.replace("_", "-"), default=None if required else field.default, required=required)
    ns = parser.parse_args()
    ns.top_n = int(ns.top_n)
    ns.leverage = float(ns.leverage)
    ns.fee_rate = float(ns.fee_rate)
    ns.slippage_rate = float(ns.slippage_rate)
    ns.hold_bars = int(ns.hold_bars)
    ns.random_state = int(ns.random_state)
    return RexPre2024MlConfig(**vars(ns))


def main() -> None:
    report = run(parse_args())
    print(
        json.dumps(
            {
                "manifest_hash": report["manifest_hash"],
                "tested": report["input"]["tested"],
                "alpha_qualifiers": len(report["alpha_qualifiers"]),
                "live_qualifiers": len(report["live_qualifiers"]),
                "selected": report["selected"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
