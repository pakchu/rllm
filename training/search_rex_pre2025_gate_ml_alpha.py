"""Refine the pre-2025 taker/range REX gate with a frozen sparse ML critic."""

from __future__ import annotations

import argparse
import json
from dataclasses import MISSING, asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.audit_rex8640_usdkrw_gate import gate_match, parse_gates
from training.search_rex_pre2024_ml_alpha import (
    MODEL_SPECS,
    build_model,
    completed_before,
    feature_matrix,
    feature_names,
    fit_medians,
    load_market_before,
    manifest_hash,
    prefix_hash,
    read_jsonl,
    score_window,
    target_vector,
)
from training.strict_bar_backtest import load_market_bars


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
DEFAULT_GATES = json.dumps(
    [
        {"feature": "taker_imbalance", "op": "<=", "threshold": -0.07073595391836504},
        {"feature": "rex_2016_range_pos", "op": "<=", "threshold": 0.6865011402825759},
    ],
    separators=(",", ":"),
)


@dataclass(frozen=True)
class Pre2025GateMlConfig:
    train_jsonl: str
    selection_jsonl: str
    eval_jsonl: str
    market_csv: str
    output: str
    manifest_output: str
    gates_json: str = DEFAULT_GATES
    top_n: int = 10
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    hold_bars: int = 144
    random_state: int = 42


def filter_gate_rows(rows: list[dict[str, Any]], gates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if gate_match(row, gates)]


def run(cfg: Pre2025GateMlConfig) -> dict[str, Any]:
    gates = parse_gates(cfg.gates_json)
    phase_market = load_market_before(cfg.market_csv, str(SELECT_END))
    train_rows = filter_gate_rows(read_jsonl(cfg.train_jsonl), gates)
    selection_rows = filter_gate_rows(read_jsonl(cfg.selection_jsonl), gates)
    fit_rows = [row for row in train_rows if completed_before(row, phase_market["date"], FIT_END, cfg.hold_bars)]
    names = feature_names(fit_rows)
    medians = fit_medians(fit_rows, names)
    x_fit = feature_matrix(fit_rows, names, medians)
    x_selection = feature_matrix(selection_rows, names, medians)

    models: dict[str, Any] = {}
    selection_scores: dict[str, np.ndarray] = {}
    specs = {str(spec["name"]): dict(spec) for spec in MODEL_SPECS}
    for spec in MODEL_SPECS:
        model = build_model(spec, cfg.random_state)
        model.fit(x_fit, target_vector(fit_rows, str(spec["target"])))
        name = str(spec["name"])
        models[name] = model
        selection_scores[name] = np.asarray(model.predict(x_selection), dtype=np.float64)

    candidates: list[dict[str, Any]] = []
    for model_name, scores in selection_scores.items():
        for side in ("both", "long", "short"):
            eligible = np.asarray(
                [side == "both" or str(row.get("side", "")).lower() == side for row in selection_rows],
                dtype=bool,
            )
            if int(eligible.sum()) < 20:
                continue
            for quantile in (0.30, 0.50, 0.70, 0.80, 0.90):
                policy = {
                    "model": model_name,
                    "model_spec": specs[model_name],
                    "score_quantile": quantile,
                    "score_threshold": float(np.quantile(scores[eligible], quantile)),
                    "side": side,
                    "family": "both",
                }
                full = score_window(selection_rows, scores, policy, phase_market, cfg, WINDOWS["selection2024"])
                h1 = score_window(selection_rows, scores, policy, phase_market, cfg, WINDOWS["selection2024_h1"])
                h2 = score_window(selection_rows, scores, policy, phase_market, cfg, WINDOWS["selection2024_h2"])
                if full["trades"] < 16:
                    continue
                candidates.append(
                    {
                        **policy,
                        "selection2024": full,
                        "selection2024_halves": [h1, h2],
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
        full = row["selection2024"]
        return (
            float(score["positive_halves"]),
            float(score["min_half_trades"] >= 6),
            float(score["min_half_ratio"]),
            float(score["full_ratio"]),
            -float(full["p_value"]),
            float(full["trades"]),
        )

    selected = sorted(candidates, key=rank_key, reverse=True)[: int(cfg.top_n)]
    phase_rows = sorted(fit_rows + selection_rows, key=lambda row: (str(row["date"]), int(row["signal_pos"])))
    manifest_core = {
        "phase": "pre_future_manifest",
        "gate_provenance": "Top-10 rank 8 from 2021-2023 train + 2024 selection corrected-MDD scan",
        "fixed_gates": gates,
        "fit_window": ["2021-01-01", "2024-01-01"],
        "selection_window": ["2024-01-01", "2025-01-01"],
        "feature_hash": prefix_hash(phase_rows, names),
        "feature_names": names,
        "phase_market_end": str(phase_market["date"].iloc[-1]),
        "policies": [
            {key: row[key] for key in ("model", "model_spec", "score_quantile", "score_threshold", "side", "family")}
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
        raise RuntimeError("pre-2025 OHLC prefix changed after full load")
    eval_rows = filter_gate_rows(read_jsonl(cfg.eval_jsonl), gates)
    x_eval = feature_matrix(eval_rows, names, medians)
    eval_score_bank = {name: np.asarray(model.predict(x_eval), dtype=np.float64) for name, model in models.items()}
    for row in selected:
        scores = eval_score_bank[str(row["model"])]
        for name in ("eval2025", "holdout2026", "eval2025_2026"):
            row[name] = score_window(eval_rows, scores, row, market, cfg, WINDOWS[name])
        row["passes_alpha_target"] = bool(
            row["eval2025"]["cagr_pct"] > 0
            and row["eval2025"]["ratio"] >= 3
            and row["eval2025"]["strict_mdd_pct"] <= 15
            and row["eval2025"]["trades"] >= 16
            and row["holdout2026"]["cagr_pct"] > 0
            and row["holdout2026"]["ratio"] >= 3
            and row["holdout2026"]["strict_mdd_pct"] <= 15
            and row["holdout2026"]["trades"] >= 10
            and row["eval2025_2026"]["ratio"] >= 3
            and row["eval2025_2026"]["trades"] >= 30
        )
        row["passes_live_target"] = bool(row["passes_alpha_target"] and row["eval2025_2026"]["p_value"] <= 0.10)

    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "protocol": {
            "gate_selected_by_top10_rule": True,
            "model_fit": "2021-2023 completed paths",
            "selection": "2024 full/H1/H2",
            "manifest_written_before_future": True,
            "full_market_loaded_after_manifest": True,
            "eval2025_and_holdout2026_not_used_for_selection": True,
            "cagr": "full configured calendar window including idle time",
            "strict_mdd": "worst-order favorable-to-adverse OHLC high-water path drawdown",
        },
        "input": {
            "fit_rows": len(fit_rows),
            "selection_rows": len(selection_rows),
            "features": len(names),
            "tested": len(candidates),
        },
        "manifest_hash": frozen_hash,
        "selected": selected,
        "alpha_qualifiers": [row for row in selected if row["passes_alpha_target"]],
        "live_qualifiers": [row for row in selected if row["passes_live_target"]],
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> Pre2025GateMlConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in Pre2025GateMlConfig.__dataclass_fields__.values():
        required = field.default is MISSING and field.default_factory is MISSING
        parser.add_argument("--" + field.name.replace("_", "-"), default=None if required else field.default, required=required)
    ns = parser.parse_args()
    for name in ("top_n", "hold_bars", "random_state"):
        setattr(ns, name, int(getattr(ns, name)))
    for name in ("leverage", "fee_rate", "slippage_rate"):
        setattr(ns, name, float(getattr(ns, name)))
    return Pre2025GateMlConfig(**vars(ns))


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
