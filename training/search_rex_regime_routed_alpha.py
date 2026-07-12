"""Route frozen pre-2024 REX long/short critics with slow price state."""

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

from training.economic_action_backtest import EconomicActionBacktestConfig, strict_backtest_actions
from training.search_rex_pre2024_ml_alpha import (
    FIT_END,
    SELECT_END,
    WINDOWS,
    build_model,
    completed_before,
    feature_matrix,
    feature_names,
    fit_medians,
    load_market_before,
    manifest_hash,
    read_jsonl,
    select_highest_per_signal,
    target_vector,
)
from training.strict_bar_backtest import load_market_bars


ROUTERS: tuple[dict[str, Any], ...] = (
    {"name": "weekly_return_sign", "features": ["htf_1w_return_4"], "mode": "all_sign"},
    {"name": "monthly_range_sign", "features": ["rex_8640_range_pos"], "mode": "all_sign"},
    {"name": "daily_return_sign", "features": ["htf_1d_return_4"], "mode": "all_sign"},
    {"name": "medium_trend_sign", "features": ["trend_96"], "mode": "all_sign"},
    {
        "name": "weekly_and_monthly_sign",
        "features": ["htf_1w_return_4", "rex_8640_range_pos"],
        "mode": "all_sign",
    },
    {
        "name": "daily_and_weekly_sign",
        "features": ["htf_1d_return_4", "htf_1w_return_4"],
        "mode": "all_sign",
    },
)


@dataclass(frozen=True)
class RexRegimeRoutedConfig:
    train_jsonl: str
    test_jsonl: str
    eval_jsonl: str
    market_csv: str
    specialist_manifest: str
    output: str
    manifest_output: str
    top_n: int = 10
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    hold_bars: int = 144
    random_state: int = 42


def router_allows(row: dict[str, Any], side: str, router: dict[str, Any]) -> bool:
    snapshot = row.get("feature_snapshot") or {}
    values = [float(snapshot.get(name, np.nan)) for name in router["features"]]
    if not values or not all(np.isfinite(value) for value in values):
        return False
    if side == "long":
        return all(value >= 0.0 for value in values)
    if side == "short":
        return all(value < 0.0 for value in values)
    return False


def policy_match(row: dict[str, Any], score: float, policy: dict[str, Any]) -> bool:
    if str(row.get("side", "")).lower() != str(policy["side"]):
        return False
    if str(policy["family"]) != "both" and str(row.get("family", "")) != str(policy["family"]):
        return False
    return float(score) >= float(policy["score_threshold"])


def routed_prediction_rows(
    rows: list[dict[str, Any]],
    score_bank: dict[str, np.ndarray],
    pair: dict[str, Any],
    *,
    market_dates: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    hold_bars: int,
) -> list[dict[str, Any]]:
    long_policy = pair["long_policy"]
    short_policy = pair["short_policy"]
    router = pair["router"]
    eligible_rows: list[dict[str, Any]] = []
    margins: list[float] = []
    for i, row in enumerate(rows):
        date = pd.Timestamp(str(row["date"]))
        if not (start <= date < end) or not completed_before(row, market_dates, end, hold_bars):
            continue
        side = str(row.get("side", "")).lower()
        policy = long_policy if side == "long" else short_policy if side == "short" else None
        if policy is None or not router_allows(row, side, router):
            continue
        score = float(score_bank[str(policy["model"])][i])
        if not policy_match(row, score, policy):
            continue
        scale = max(float(policy["score_scale"]), 1e-12)
        eligible_rows.append(row)
        margins.append((score - float(policy["score_threshold"])) / scale)
    chosen = select_highest_per_signal(eligible_rows, np.asarray(margins, dtype=np.float64))
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
    score_bank: dict[str, np.ndarray],
    pair: dict[str, Any],
    market: pd.DataFrame,
    cfg: RexRegimeRoutedConfig,
    window: tuple[pd.Timestamp, pd.Timestamp],
) -> dict[str, Any]:
    start, end = window
    predictions = routed_prediction_rows(
        rows,
        score_bank,
        pair,
        market_dates=market["date"],
        start=start,
        end=end,
        hold_bars=int(cfg.hold_bars),
    )
    if not predictions:
        return empty_metrics()
    report = strict_backtest_actions(
        predictions,
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


def _policy_copy(policy: dict[str, Any], scores: np.ndarray) -> dict[str, Any]:
    return {
        **policy,
        "score_scale": float(max(np.std(scores), 1e-12)),
    }


def run(cfg: RexRegimeRoutedConfig) -> dict[str, Any]:
    specialist_manifest = json.loads(Path(cfg.specialist_manifest).read_text())
    if specialist_manifest.get("selection_strategy") != "side_balanced":
        raise ValueError("specialist manifest must be side_balanced")
    policies = specialist_manifest["policies"]
    long_policies = [policy for policy in policies if policy["side"] == "long"]
    short_policies = [policy for policy in policies if policy["side"] == "short"]
    if not long_policies or not short_policies:
        raise ValueError("specialist manifest lacks long or short policies")

    phase_market = load_market_before(cfg.market_csv, str(SELECT_END))
    pre_rows = read_jsonl(cfg.train_jsonl)
    pre_dates = np.asarray([np.datetime64(str(row["date"])) for row in pre_rows])
    fit_rows = [
        row
        for row, date in zip(pre_rows, pre_dates)
        if date < np.datetime64(FIT_END) and completed_before(row, phase_market["date"], FIT_END, cfg.hold_bars)
    ]
    select_rows = [row for row, date in zip(pre_rows, pre_dates) if np.datetime64(FIT_END) <= date < np.datetime64(SELECT_END)]
    names = feature_names(fit_rows)
    if names != specialist_manifest["feature_names"]:
        raise RuntimeError("specialist feature names changed")
    medians = fit_medians(fit_rows, names)
    x_fit = feature_matrix(fit_rows, names, medians)
    x_select = feature_matrix(select_rows, names, medians)

    model_specs = {policy["model"]: policy["model_spec"] for policy in policies}
    models: dict[str, Any] = {}
    select_scores: dict[str, np.ndarray] = {}
    for name, spec in model_specs.items():
        model = build_model(spec, cfg.random_state)
        model.fit(x_fit, target_vector(fit_rows, str(spec["target"])))
        models[name] = model
        select_scores[name] = np.asarray(model.predict(x_select), dtype=np.float64)

    enriched_longs = [_policy_copy(policy, select_scores[str(policy["model"])]) for policy in long_policies]
    enriched_shorts = [_policy_copy(policy, select_scores[str(policy["model"])]) for policy in short_policies]
    candidates: list[dict[str, Any]] = []
    for long_policy in enriched_longs:
        for short_policy in enriched_shorts:
            for router in ROUTERS:
                pair = {
                    "long_policy": long_policy,
                    "short_policy": short_policy,
                    "router": dict(router),
                }
                full = score_window(select_rows, select_scores, pair, phase_market, cfg, WINDOWS["select2023"])
                h1 = score_window(select_rows, select_scores, pair, phase_market, cfg, WINDOWS["select2023_h1"])
                h2 = score_window(select_rows, select_scores, pair, phase_market, cfg, WINDOWS["select2023_h2"])
                pair["select2023"] = full
                pair["select2023_halves"] = [h1, h2]
                pair["selection_score"] = {
                    "positive_halves": int(h1["cagr_pct"] > 0) + int(h2["cagr_pct"] > 0),
                    "min_half_ratio": min(float(h1["ratio"]), float(h2["ratio"])),
                    "full_ratio": float(full["ratio"]),
                    "min_half_trades": min(int(h1["trades"]), int(h2["trades"])),
                }
                candidates.append(pair)

    def rank_key(row: dict[str, Any]) -> tuple[float, ...]:
        score = row["selection_score"]
        return (
            float(score["positive_halves"]),
            float(score["min_half_trades"] >= 8),
            float(score["min_half_ratio"]),
            float(score["full_ratio"]),
            -float(row["select2023"]["p_value"]),
        )

    selected = sorted(candidates, key=rank_key, reverse=True)[: int(cfg.top_n)]
    manifest_core = {
        "phase": "pre_future_manifest",
        "base_specialist_manifest_hash": specialist_manifest["manifest_hash"],
        "fit_window": ["2021-01-01", "2023-01-01"],
        "selection_window": ["2023-01-01", "2024-01-01"],
        "selection": "Top-10 long/short specialist pairs with fixed zero-sign routers",
        "phase_market_end": str(phase_market["date"].iloc[-1]),
        "pairs": [
            {key: row[key] for key in ("long_policy", "short_policy", "router")}
            for row in selected
        ],
    }
    frozen_hash = manifest_hash(manifest_core)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), **manifest_core, "manifest_hash": frozen_hash}
    manifest_path = Path(cfg.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    market = load_market_bars(cfg.market_csv)
    prefix = market.iloc[: len(phase_market)].reset_index(drop=True)
    if not prefix.equals(phase_market.reset_index(drop=True)):
        raise RuntimeError("pre-2024 OHLC prefix changed after full load")
    future_sets = {"test2024": read_jsonl(cfg.test_jsonl), "eval2025_2026": read_jsonl(cfg.eval_jsonl)}
    future_scores: dict[str, dict[str, np.ndarray]] = {}
    for split, rows in future_sets.items():
        x = feature_matrix(rows, names, medians)
        future_scores[split] = {name: np.asarray(model.predict(x), dtype=np.float64) for name, model in models.items()}

    for pair in selected:
        pair["test2024"] = score_window(
            future_sets["test2024"], future_scores["test2024"], pair, market, cfg, WINDOWS["test2024"]
        )
        pair["eval2025"] = score_window(
            future_sets["eval2025_2026"], future_scores["eval2025_2026"], pair, market, cfg, WINDOWS["eval2025"]
        )
        pair["ytd2026"] = score_window(
            future_sets["eval2025_2026"], future_scores["eval2025_2026"], pair, market, cfg, WINDOWS["ytd2026"]
        )
        pair["passes_alpha_target"] = all(
            pair[name]["cagr_pct"] > 0
            and pair[name]["ratio"] >= 3
            and pair[name]["strict_mdd_pct"] <= 15
            and pair[name]["trades"] >= 30
            for name in ("test2024", "eval2025")
        )
        pair["passes_live_target"] = bool(
            pair["passes_alpha_target"]
            and pair["ytd2026"]["cagr_pct"] > 0
            and pair["ytd2026"]["ratio"] >= 3
            and pair["ytd2026"]["strict_mdd_pct"] <= 15
            and pair["ytd2026"]["trades"] >= 10
        )

    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "protocol": {
            "specialists": "frozen side-balanced pre-2024 manifest",
            "routers": "fixed zero-sign rules over signal-time slow price state",
            "manifest_written_before_future": True,
            "full_market_loaded_after_manifest": True,
            "cagr": "full configured calendar window including idle time",
            "strict_mdd": "worst-order favorable-to-adverse OHLC high-water path drawdown",
        },
        "tested": len(candidates),
        "manifest_hash": frozen_hash,
        "selected": selected,
        "alpha_qualifiers": [row for row in selected if row["passes_alpha_target"]],
        "live_qualifiers": [row for row in selected if row["passes_live_target"]],
    }
    output_path = Path(cfg.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> RexRegimeRoutedConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in RexRegimeRoutedConfig.__dataclass_fields__.values():
        required = field.default is MISSING and field.default_factory is MISSING
        parser.add_argument("--" + field.name.replace("_", "-"), default=None if required else field.default, required=required)
    ns = parser.parse_args()
    for name in ("top_n", "hold_bars", "random_state"):
        setattr(ns, name, int(getattr(ns, name)))
    for name in ("leverage", "fee_rate", "slippage_rate"):
        setattr(ns, name, float(getattr(ns, name)))
    return RexRegimeRoutedConfig(**vars(ns))


def main() -> None:
    report = run(parse_args())
    print(
        json.dumps(
            {
                "manifest_hash": report["manifest_hash"],
                "tested": report["tested"],
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
