"""Backtest a score-direction router against selector scoreboards.

The selector report contains fold-safe pre-fold family scoreboards.  This script
uses a router dataset or prediction report to decide whether a fold should take
the highest-scored family (HIGH) or invert to the lowest finite scored family
among the visible options (LOW), then recomputes the selected family events and
simulates them without using target-fold results for selection.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from preprocessing.market_features import build_market_feature_frame
from training.event_candidate_pool_probe import _candidate_rows_for_family, _feature_candidates, _load_market, _simulate_rows, _split_mask
from training.event_candidate_regime_family_selector import RegimeFamilySelectorConfig


@dataclass(frozen=True)
class ScoreDirectionRouterBacktestConfig:
    selector_report: str
    router_input: str
    input_csv: str
    output: str
    route_source: str = "target"  # target, prediction, always_high, always_low
    fold_start: str = ""
    fold_end: str = ""
    max_options: int = 5
    train_start: str = "2020-01-01"
    hold_bars: int = 288
    entry_delay_bars: int = 1
    window_size: int = 144
    stride_bars: int = 24
    quantile: float = 0.80
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    family_include: str = ""


def _safe_json(x: Any) -> Any:
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x


def _fold_key(fold: dict[str, Any] | None) -> str:
    fold = fold or {}
    return str(fold.get("name") or f"{fold.get('start','')}:{fold.get('end','')}")


def _route_value(row: dict[str, Any], source: str) -> str:
    source = source.lower()
    if source == "always_high":
        return "HIGH"
    if source == "always_low":
        return "LOW"
    if source == "prediction":
        return str(row.get("prediction", "")).upper()
    target = _safe_json(row.get("target"))
    if isinstance(target, dict):
        if "trust_score_rank" in target:
            return str(target.get("trust_score_rank", "")).upper()
        if "direction_regime" in target:
            return "HIGH" if target.get("direction_regime") == "HIGH_SCORE_WINS" else "LOW" if target.get("direction_regime") == "LOW_SCORE_WINS" else "ABSTAIN"
    return str(target).upper()


def _load_routes(path: str, source: str) -> dict[str, str]:
    if source in {"always_high", "always_low"}:
        return {}
    p = Path(path)
    rows: list[dict[str, Any]]
    if p.suffix == ".json":
        obj = json.loads(p.read_text())
        rows = obj.get("predictions", []) if isinstance(obj, dict) else obj
    else:
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    out: dict[str, str] = {}
    for row in rows:
        key = _fold_key(row.get("fold"))
        # eval reports may not retain fold; fall back to index only when caller
        # uses the same ordered selector folds, handled later by index.
        if not key.strip(":") and "index" in row:
            key = f"index:{row['index']}"
        route = _route_value(row, source)
        if route in {"HIGH", "LOW"}:
            out[key] = route
    return out


def _in_range(fold: dict[str, Any], cfg: ScoreDirectionRouterBacktestConfig) -> bool:
    start = str(fold.get("start", ""))
    if cfg.fold_start and start < cfg.fold_start:
        return False
    if cfg.fold_end and start >= cfg.fold_end:
        return False
    return True


def _finite_options(scoreboard: list[dict[str, Any]], max_options: int) -> list[dict[str, Any]]:
    opts = []
    for row in scoreboard[: int(max_options)]:
        try:
            score = float(row.get("score", 0.0) or 0.0)
        except Exception:
            score = 0.0
        if score <= -1e8 or not np.isfinite(score):
            continue
        opts.append({**row, "score": score})
    return opts


def _choose_family(scoreboard: list[dict[str, Any]], route: str, max_options: int) -> dict[str, Any] | None:
    opts = _finite_options(scoreboard, max_options)
    if not opts:
        return None
    return max(opts, key=lambda r: r["score"]) if route == "HIGH" else min(opts, key=lambda r: r["score"])


def run(cfg: ScoreDirectionRouterBacktestConfig) -> dict[str, Any]:
    report = json.loads(Path(cfg.selector_report).read_text())
    routes = _load_routes(cfg.router_input, cfg.route_source)
    market = _load_market(cfg.input_csv)
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    dates = pd.to_datetime(market["date"])
    families = _feature_candidates(features)
    if cfg.family_include:
        needles = [x.strip() for x in cfg.family_include.split(",") if x.strip()]
        families = {name: value for name, value in families.items() if any(needle in name for needle in needles)}

    sim_cfg = RegimeFamilySelectorConfig(
        input_csv=cfg.input_csv,
        output=cfg.output,
        train_start=cfg.train_start,
        hold_bars=cfg.hold_bars,
        entry_delay_bars=cfg.entry_delay_bars,
        window_size=cfg.window_size,
        stride_bars=cfg.stride_bars,
        quantile=cfg.quantile,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        family_include=cfg.family_include,
    )
    selected_events: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    eligible_idx = 0
    for fold_row in report.get("folds", []):
        fold = fold_row.get("fold") or {}
        if not _in_range(fold, cfg):
            continue
        key = _fold_key(fold)
        route = "HIGH" if cfg.route_source == "always_high" else "LOW" if cfg.route_source == "always_low" else routes.get(key, routes.get(f"index:{eligible_idx}"))
        eligible_idx += 1
        if route not in {"HIGH", "LOW"}:
            fold_rows.append({"fold": fold, "route": route, "skip": "no_binary_route"})
            continue
        chosen = _choose_family(fold_row.get("pre_fold_scoreboard") or [], route, cfg.max_options)
        if not chosen:
            fold_rows.append({"fold": fold, "route": route, "skip": "no_finite_option"})
            continue
        fam = str(chosen.get("family"))
        if fam not in families:
            fold_rows.append({"fold": fold, "route": route, "selected_family": fam, "skip": "family_not_available"})
            continue
        strength, direction = families[fam]
        threshold = float(chosen.get("threshold"))
        fold_mask = _split_mask(dates, fold["start"], fold["end"])
        rows = _candidate_rows_for_family(market, strength, direction, family=fam, threshold=threshold, mask=fold_mask, cfg=sim_cfg)  # type: ignore[arg-type]
        selected_events.extend(rows)
        fold_rows.append({"fold": fold, "route": route, "selected_family": fam, "score": chosen.get("score"), "threshold": threshold, "events": len(rows)})
    selected_events.sort(key=lambda r: (str(r.get("entry_date")), str(r.get("family"))))
    sim = _simulate_rows(selected_events, market, sim_cfg)  # type: ignore[arg-type]
    out = {"config": asdict(cfg), "folds": fold_rows, "event_count": len(selected_events), "sim": sim["sim"], "trade_stats": sim["trade_stats"], "leakage_guard": {"routes_supplied_externally": True, "target_fold_metrics_not_used_by_backtester": True}}
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name, field in ScoreDirectionRouterBacktestConfig.__dataclass_fields__.items():
        arg = "--" + name.replace("_", "-")
        kwargs: dict[str, Any] = {"default": field.default}
        if field.default == "" and name in {"selector_report", "router_input", "input_csv", "output"}:
            kwargs["required"] = True
        if isinstance(field.default, int):
            kwargs["type"] = int
        elif isinstance(field.default, float):
            kwargs["type"] = float
        p.add_argument(arg, **kwargs)
    rep = run(ScoreDirectionRouterBacktestConfig(**vars(p.parse_args())))
    print(json.dumps({"event_count": rep["event_count"], "sim": rep["sim"], "trade_stats": rep["trade_stats"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
