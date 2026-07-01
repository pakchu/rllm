"""Expanding-window rolling validation for fixed REX candidate parameters."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.market_features import build_market_feature_frame
from training.event_candidate_pool_probe import EventPoolConfig, _feature_candidates, _load_market, _simulate_rows, _split_mask
from training.rex_horizon_sweep import _fast_candidate_rows, _parse_floats, _parse_ints


@dataclass(frozen=True)
class RexRollingValidationConfig:
    input_csv: str
    output: str
    family: str = "rex_htf_pullback_resume"
    quantile_grid: tuple[float, ...] = (0.80, 0.85)
    hold_bars_grid: tuple[int, ...] = (288,)
    stride_bars_grid: tuple[int, ...] = (24,)
    window_size: int = 144
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    folds: tuple[str, ...] = (
        "2020-01-01:2023-01-01:2023-01-01:2024-01-01",
        "2020-01-01:2024-01-01:2024-01-01:2025-01-01",
        "2020-01-01:2025-01-01:2025-01-01:2026-01-01",
        "2020-01-01:2026-01-01:2026-01-01:2026-06-01",
    )


def _parse_fold(raw: str) -> tuple[str, str, str, str]:
    parts = [x.strip() for x in str(raw).split(":")]
    if len(parts) != 4:
        raise ValueError(f"fold must be train_start:train_end:val_start:val_end, got {raw!r}")
    return tuple(parts)  # type: ignore[return-value]


def _mk_cfg(base: RexRollingValidationConfig, *, hold: int, stride: int) -> EventPoolConfig:
    return EventPoolConfig(
        input_csv=base.input_csv,
        output=base.output,
        hold_bars=int(hold),
        entry_delay_bars=int(base.entry_delay_bars),
        window_size=int(base.window_size),
        stride_bars=int(stride),
        leverage=float(base.leverage),
        fee_rate=float(base.fee_rate),
        slippage_rate=float(base.slippage_rate),
    )


def _fold_score(folds: list[dict[str, Any]]) -> float:
    vals = []
    total_trades = 0
    for f in folds:
        sim = f["validation"]["sim"]
        cagr = float(sim.get("cagr_pct", 0.0) or 0.0)
        mdd = float(sim.get("strict_mdd_pct", 0.0) or 0.0)
        trades = int(sim.get("trade_entries", 0) or 0)
        total_trades += trades
        vals.append((cagr, mdd, trades, float(sim.get("cagr_to_strict_mdd", 0.0) or 0.0)))
    positive = sum(1 for c, _m, t, _r in vals if c > 0 and t >= 20)
    mean_ratio = sum(r for _c, _m, _t, r in vals) / max(1, len(vals))
    worst_cagr = min((c for c, _m, _t, _r in vals), default=-100.0)
    worst_mdd = max((m for _c, m, _t, _r in vals), default=100.0)
    return positive + 0.25 * mean_ratio + min(total_trades, 400) / 400.0 - 0.05 * max(0.0, worst_mdd - 15.0) + 0.02 * min(0.0, worst_cagr)


def run(cfg: RexRollingValidationConfig) -> dict[str, Any]:
    market = _load_market(cfg.input_csv)
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    families = _feature_candidates(features)
    if cfg.family not in families:
        raise ValueError(f"family not found: {cfg.family}")
    strength, direction = families[cfg.family]
    dates = pd.to_datetime(market["date"])
    date_strings = [str(x) for x in market["date"].tolist()]
    trials: list[dict[str, Any]] = []
    for q in cfg.quantile_grid:
        for hold in cfg.hold_bars_grid:
            for stride in cfg.stride_bars_grid:
                ecfg = _mk_cfg(cfg, hold=int(hold), stride=int(stride))
                fold_rows: list[dict[str, Any]] = []
                for raw_fold in cfg.folds:
                    train_start, train_end, val_start, val_end = _parse_fold(raw_fold)
                    train_mask = _split_mask(dates, train_start, train_end)
                    val_mask = _split_mask(dates, val_start, val_end)
                    train_x = strength[train_mask & np.isfinite(strength)]
                    if train_x.size < 100:
                        continue
                    threshold = float(np.quantile(train_x, float(q)))
                    val_rows = _fast_candidate_rows(
                        date_strings,
                        strength,
                        direction,
                        family=cfg.family,
                        threshold=threshold,
                        mask=val_mask,
                        hold_bars=int(hold),
                        entry_delay_bars=int(cfg.entry_delay_bars),
                        stride_bars=int(stride),
                        window_size=int(cfg.window_size),
                    )
                    val_result = _simulate_rows(val_rows, market, ecfg)
                    fold_rows.append({
                        "train_start": train_start,
                        "train_end": train_end,
                        "validation_start": val_start,
                        "validation_end": val_end,
                        "threshold": threshold,
                        "validation": {"sim": val_result["sim"], "trade_stats": val_result["trade_stats"], "candidate_count": len(val_rows)},
                    })
                trial = {"family": cfg.family, "quantile": float(q), "hold_bars": int(hold), "stride_bars": int(stride), "folds": fold_rows}
                trial["rolling_score"] = _fold_score(fold_rows)
                trials.append(trial)
    trials.sort(key=lambda r: float(r["rolling_score"]), reverse=True)
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg) | {"quantile_grid": list(cfg.quantile_grid), "hold_bars_grid": list(cfg.hold_bars_grid), "stride_bars_grid": list(cfg.stride_bars_grid), "folds": list(cfg.folds)},
        "trial_count": len(trials),
        "top": trials,
        "leakage_guard": {"each_fold_threshold_fit_on_fold_train_only": True, "validation_not_used_for_threshold": True, "features_use_rows_at_or_before_signal": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--family", default=RexRollingValidationConfig.family)
    p.add_argument("--quantile-grid", default=",".join(map(str, RexRollingValidationConfig.quantile_grid)))
    p.add_argument("--hold-bars-grid", default=",".join(map(str, RexRollingValidationConfig.hold_bars_grid)))
    p.add_argument("--stride-bars-grid", default=",".join(map(str, RexRollingValidationConfig.stride_bars_grid)))
    p.add_argument("--folds", default=",".join(RexRollingValidationConfig.folds))
    return p.parse_args()


def main() -> None:
    a = parse_args()
    cfg = RexRollingValidationConfig(
        input_csv=a.input_csv,
        output=a.output,
        family=a.family,
        quantile_grid=_parse_floats(a.quantile_grid),
        hold_bars_grid=_parse_ints(a.hold_bars_grid),
        stride_bars_grid=_parse_ints(a.stride_bars_grid),
        folds=tuple(x.strip() for x in str(a.folds).split(",") if x.strip()),
    )
    out = run(cfg)
    print(json.dumps({"trial_count": out["trial_count"], "top": out["top"][:10]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
