"""Select fixed stop/take-profit exits for frozen pre-2025 REX ML policies."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import MISSING, asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.audit_rex8640_usdkrw_gate import gate_match
from training.search_rex_pre2024_ml_alpha import (
    build_model,
    completed_before,
    feature_matrix,
    feature_names,
    fit_medians,
    load_market_before,
    manifest_hash,
    prediction_rows,
    read_jsonl,
    target_vector,
)
from training.strict_bar_backtest import _trade_stats, load_market_bars


FIT_END = pd.Timestamp("2024-01-01")
SELECT_END = pd.Timestamp("2025-01-01")
WINDOWS = {
    "selection2024": (pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")),
    "selection2024_h1": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-07-01")),
    "selection2024_h2": (pd.Timestamp("2024-07-01"), pd.Timestamp("2025-01-01")),
    "eval2025": (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")),
    "holdout2026": (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-06-02")),
    "eval2025_2026": (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-06-02")),
}


@dataclass(frozen=True)
class ExitOverlayConfig:
    train_jsonl: str
    selection_jsonl: str
    eval_jsonl: str
    market_csv: str
    base_manifest: str
    output: str
    manifest_output: str
    top_n: int = 10
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    hold_bars: int = 144
    random_state: int = 42
    stop_losses: str = "0,0.005,0.01,0.015,0.02,0.03"
    take_profits: str = "0,0.01,0.02,0.03,0.04,0.06"


def parse_fractions(raw: str) -> tuple[float, ...]:
    values = tuple(sorted({float(value.strip()) for value in str(raw).split(",") if value.strip()}))
    if not values or min(values) < 0:
        raise ValueError("exit fractions must be non-negative")
    return values


def simulate_exit_overlay(
    predictions: list[dict[str, Any]],
    market: pd.DataFrame,
    *,
    window: tuple[pd.Timestamp, pd.Timestamp],
    hold_bars: int,
    stop_loss: float,
    take_profit: float,
    leverage: float,
    fee_rate: float,
    slippage_rate: float,
) -> dict[str, Any]:
    start, end = window
    dates = pd.to_datetime(market["date"])
    opens = market["open"].to_numpy(float)
    highs = market["high"].to_numpy(float)
    lows = market["low"].to_numpy(float)
    cost = (float(fee_rate) + float(slippage_rate)) * float(leverage)
    equity = peak = 1.0
    max_dd = 0.0
    next_allowed = 0
    returns: list[float] = []
    exits: Counter[str] = Counter()
    for row in sorted(predictions, key=lambda item: int(item["signal_pos"])):
        signal_pos = int(row["signal_pos"])
        if signal_pos < next_allowed:
            continue
        entry = signal_pos + 1
        scheduled_exit = entry + int(hold_bars)
        if scheduled_exit >= len(market):
            continue
        if not (start <= pd.Timestamp(dates.iloc[entry]) < end) or pd.Timestamp(dates.iloc[scheduled_exit]) >= end:
            continue
        side = 1 if row["prediction"]["side"] == "LONG" else -1
        entry_price = float(opens[entry])
        entry_equity = equity
        equity *= 1.0 - cost
        max_dd = max(max_dd, 1.0 - equity / peak)
        position_equity = equity
        exit_position = scheduled_exit
        exit_reason = "time"
        for bar in range(entry, scheduled_exit):
            high = float(highs[bar])
            low = float(lows[bar])
            if side > 0:
                stop_hit = stop_loss > 0 and low <= entry_price * (1.0 - stop_loss)
                take_hit = take_profit > 0 and high >= entry_price * (1.0 + take_profit)
                favorable_price = high
                adverse_price = low
            else:
                stop_hit = stop_loss > 0 and high >= entry_price * (1.0 + stop_loss)
                take_hit = take_profit > 0 and low <= entry_price * (1.0 - take_profit)
                favorable_price = low
                adverse_price = high

            # Same-bar ambiguity is resolved against the strategy: stop first.
            if stop_hit:
                if not take_hit:
                    favorable_eq = max(
                        0.0,
                        position_equity * (1.0 + leverage * side * (favorable_price / entry_price - 1.0)),
                    )
                    peak = max(peak, favorable_eq)
                equity = position_equity * max(0.0, 1.0 - leverage * stop_loss)
                max_dd = max(max_dd, 1.0 - equity / peak)
                exit_position = bar + 1
                exit_reason = "stop"
                break
            if take_hit:
                equity = position_equity * (1.0 + leverage * take_profit)
                peak = max(peak, equity)
                exit_position = bar + 1
                exit_reason = "take"
                break

            open_bar = float(opens[bar])
            if open_bar <= 0:
                continue
            favorable_eq = max(
                0.0,
                equity * (1.0 + leverage * side * (favorable_price / open_bar - 1.0)),
            )
            intrabar_peak = max(peak, favorable_eq)
            adverse_eq = max(
                0.0,
                equity * (1.0 + leverage * side * (adverse_price / open_bar - 1.0)),
            )
            max_dd = max(max_dd, 1.0 - adverse_eq / intrabar_peak)
            peak = max(peak, intrabar_peak)
            close_return = side * (float(opens[bar + 1]) / open_bar - 1.0)
            equity *= max(0.0, 1.0 + leverage * close_return)
            peak = max(peak, equity)
        equity *= 1.0 - cost
        max_dd = max(max_dd, 1.0 - equity / peak)
        peak = max(peak, equity)
        returns.append(equity / entry_equity - 1.0)
        exits[exit_reason] += 1
        next_allowed = exit_position + 1
    years = (end - start).total_seconds() / (365.25 * 86400.0)
    absolute_return = (equity - 1.0) * 100.0
    cagr = (equity ** (1.0 / years) - 1.0) * 100.0 if equity > 0 else -100.0
    mdd = max_dd * 100.0
    stats = _trade_stats(returns)
    return {
        "return_pct": float(absolute_return),
        "cagr_pct": float(cagr),
        "strict_mdd_pct": float(mdd),
        "ratio": float(cagr / mdd) if mdd > 1e-12 else 0.0,
        "trades": len(returns),
        "exit_counts": dict(exits),
        "p_value": float(stats["p_value_mean_ret_approx"]),
    }


def _fit_models(
    cfg: ExitOverlayConfig,
    base_manifest: dict[str, Any],
    phase_market: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    gates = base_manifest["fixed_gates"]
    train_rows = [row for row in read_jsonl(cfg.train_jsonl) if gate_match(row, gates)]
    selection_rows = [row for row in read_jsonl(cfg.selection_jsonl) if gate_match(row, gates)]
    fit_rows = [row for row in train_rows if completed_before(row, phase_market["date"], FIT_END, cfg.hold_bars)]
    names = feature_names(fit_rows)
    if names != base_manifest["feature_names"]:
        raise RuntimeError("base manifest feature names changed")
    medians = fit_medians(fit_rows, names)
    x_fit = feature_matrix(fit_rows, names, medians)
    x_selection = feature_matrix(selection_rows, names, medians)
    specs = {policy["model"]: policy["model_spec"] for policy in base_manifest["policies"]}
    models: dict[str, Any] = {}
    scores: dict[str, np.ndarray] = {}
    for name, spec in specs.items():
        model = build_model(spec, cfg.random_state)
        model.fit(x_fit, target_vector(fit_rows, str(spec["target"])))
        models[name] = model
        scores[name] = np.asarray(model.predict(x_selection), dtype=np.float64)
    return fit_rows, selection_rows, names, medians, models, scores


def run(cfg: ExitOverlayConfig) -> dict[str, Any]:
    base_manifest = json.loads(Path(cfg.base_manifest).read_text())
    phase_market = load_market_before(cfg.market_csv, str(SELECT_END))
    _, selection_rows, names, medians, models, selection_score_bank = _fit_models(cfg, base_manifest, phase_market)
    stops = parse_fractions(cfg.stop_losses)
    takes = parse_fractions(cfg.take_profits)
    candidates: list[dict[str, Any]] = []
    for policy_index, policy in enumerate(base_manifest["policies"]):
        scores = selection_score_bank[str(policy["model"])]
        prediction_bank = {
            name: prediction_rows(
                selection_rows,
                scores,
                threshold=float(policy["score_threshold"]),
                side=str(policy["side"]),
                family=str(policy["family"]),
                market_dates=phase_market["date"],
                start=window[0],
                end=window[1],
                hold_bars=cfg.hold_bars,
            )
            for name, window in WINDOWS.items()
            if name.startswith("selection")
        }
        for stop_loss in stops:
            for take_profit in takes:
                metrics = {
                    name: simulate_exit_overlay(
                        prediction_bank[name],
                        phase_market,
                        window=WINDOWS[name],
                        hold_bars=cfg.hold_bars,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        leverage=cfg.leverage,
                        fee_rate=cfg.fee_rate,
                        slippage_rate=cfg.slippage_rate,
                    )
                    for name in ("selection2024", "selection2024_h1", "selection2024_h2")
                }
                full, h1, h2 = metrics["selection2024"], metrics["selection2024_h1"], metrics["selection2024_h2"]
                candidates.append(
                    {
                        "policy_index": policy_index,
                        "policy": policy,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        **metrics,
                        "selection_score": {
                            "positive_halves": int(h1["cagr_pct"] > 0) + int(h2["cagr_pct"] > 0),
                            "min_half_ratio": min(float(h1["ratio"]), float(h2["ratio"])),
                            "full_ratio": float(full["ratio"]),
                            "min_half_trades": min(int(h1["trades"]), int(h2["trades"])),
                        },
                    }
                )

    def rank_key(row: dict[str, Any]) -> tuple[float, ...]:
        score = row["selection_score"]
        return (
            float(score["positive_halves"]),
            float(score["min_half_trades"] >= 6),
            float(score["min_half_ratio"]),
            float(score["full_ratio"]),
            -float(row["selection2024"]["p_value"]),
        )

    selected = sorted(candidates, key=rank_key, reverse=True)[: int(cfg.top_n)]
    manifest_core = {
        "phase": "pre_future_manifest",
        "base_manifest_hash": base_manifest["manifest_hash"],
        "selection_window": ["2024-01-01", "2025-01-01"],
        "exit_units": "unlevered price fractions",
        "overlays": [
            {key: row[key] for key in ("policy_index", "policy", "stop_loss", "take_profit")}
            for row in selected
        ],
    }
    frozen_hash = manifest_hash(manifest_core)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), **manifest_core, "manifest_hash": frozen_hash}
    manifest_path = Path(cfg.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    market = load_market_bars(cfg.market_csv)
    if not market.iloc[: len(phase_market)].reset_index(drop=True).equals(phase_market.reset_index(drop=True)):
        raise RuntimeError("pre-2025 OHLC prefix changed")
    gates = base_manifest["fixed_gates"]
    eval_rows = [row for row in read_jsonl(cfg.eval_jsonl) if gate_match(row, gates)]
    x_eval = feature_matrix(eval_rows, names, medians)
    eval_scores = {name: np.asarray(model.predict(x_eval), dtype=np.float64) for name, model in models.items()}
    for row in selected:
        policy = row["policy"]
        scores = eval_scores[str(policy["model"])]
        for name in ("eval2025", "holdout2026", "eval2025_2026"):
            predictions = prediction_rows(
                eval_rows,
                scores,
                threshold=float(policy["score_threshold"]),
                side=str(policy["side"]),
                family=str(policy["family"]),
                market_dates=market["date"],
                start=WINDOWS[name][0],
                end=WINDOWS[name][1],
                hold_bars=cfg.hold_bars,
            )
            row[name] = simulate_exit_overlay(
                predictions,
                market,
                window=WINDOWS[name],
                hold_bars=cfg.hold_bars,
                stop_loss=float(row["stop_loss"]),
                take_profit=float(row["take_profit"]),
                leverage=cfg.leverage,
                fee_rate=cfg.fee_rate,
                slippage_rate=cfg.slippage_rate,
            )
        row["passes_alpha_target"] = bool(
            row["eval2025"]["cagr_pct"] > 0
            and row["eval2025"]["ratio"] >= 3
            and row["eval2025"]["trades"] >= 16
            and row["holdout2026"]["cagr_pct"] > 0
            and row["holdout2026"]["ratio"] >= 3
            and row["holdout2026"]["trades"] >= 10
            and row["eval2025_2026"]["ratio"] >= 3
            and row["eval2025_2026"]["trades"] >= 30
        )
        row["passes_live_target"] = bool(row["passes_alpha_target"] and row["eval2025_2026"]["p_value"] <= 0.10)

    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "protocol": {
            "base_policies_frozen": True,
            "overlay_selection": "2024 full/H1/H2",
            "manifest_written_before_future": True,
            "full_market_loaded_after_manifest": True,
            "same_bar_stop_before_take": True,
            "strict_mdd": "favorable-to-adverse OHLC high-water while position remains open",
            "cagr": "full configured calendar window including idle time",
        },
        "tested": len(candidates),
        "manifest_hash": frozen_hash,
        "selected": selected,
        "alpha_qualifiers": [row for row in selected if row["passes_alpha_target"]],
        "live_qualifiers": [row for row in selected if row["passes_live_target"]],
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> ExitOverlayConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ExitOverlayConfig.__dataclass_fields__.values():
        required = field.default is MISSING and field.default_factory is MISSING
        parser.add_argument("--" + field.name.replace("_", "-"), default=None if required else field.default, required=required)
    ns = parser.parse_args()
    for name in ("top_n", "hold_bars", "random_state"):
        setattr(ns, name, int(getattr(ns, name)))
    for name in ("leverage", "fee_rate", "slippage_rate"):
        setattr(ns, name, float(getattr(ns, name)))
    return ExitOverlayConfig(**vars(ns))


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
