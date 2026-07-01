"""Rolling validation for fixed combinations of REX candidate families.

This tests whether weak but repeatable REX sub-families improve as a combined
non-overlapping trade pool. Thresholds are fitted only on each fold's train
period; validation folds are report-only for that fold.
"""
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
from training.rex_horizon_sweep import _fast_candidate_rows
from training.rex_rolling_validation import _fold_score, _parse_fold


@dataclass(frozen=True)
class RexComboSpec:
    family: str
    quantile: float


@dataclass(frozen=True)
class RexComboRollingConfig:
    input_csv: str
    output: str
    combo: str = "rex_htf_pullback_resume:0.80,rex_htf_pullback_reclaim:0.85"
    hold_bars: int = 288
    stride_bars: int = 24
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


def _parse_combo(raw: str) -> list[RexComboSpec]:
    specs: list[RexComboSpec] = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        family, q = item.rsplit(":", 1)
        specs.append(RexComboSpec(family=family.strip(), quantile=float(q)))
    if not specs:
        raise ValueError("combo must contain at least one family:quantile spec")
    return specs


def _mk_cfg(cfg: RexComboRollingConfig) -> EventPoolConfig:
    return EventPoolConfig(
        input_csv=cfg.input_csv,
        output=cfg.output,
        hold_bars=int(cfg.hold_bars),
        entry_delay_bars=int(cfg.entry_delay_bars),
        window_size=int(cfg.window_size),
        stride_bars=int(cfg.stride_bars),
        leverage=float(cfg.leverage),
        fee_rate=float(cfg.fee_rate),
        slippage_rate=float(cfg.slippage_rate),
    )


def _dedupe_sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["signal_date"]), str(row["side"]))
        old = best.get(key)
        if old is None or float(row.get("score_mean", 0.0)) > float(old.get("score_mean", 0.0)):
            best[key] = row
    return sorted(best.values(), key=lambda r: (str(r["signal_date"]), -float(r.get("score_mean", 0.0))))


def run(cfg: RexComboRollingConfig) -> dict[str, Any]:
    market = _load_market(cfg.input_csv)
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    families = _feature_candidates(features)
    specs = _parse_combo(cfg.combo)
    missing = [s.family for s in specs if s.family not in families]
    if missing:
        raise ValueError(f"family not found: {missing}")

    dates = pd.to_datetime(market["date"])
    date_strings = [str(x) for x in market["date"].tolist()]
    sim_cfg = _mk_cfg(cfg)
    fold_rows: list[dict[str, Any]] = []
    for raw_fold in cfg.folds:
        train_start, train_end, val_start, val_end = _parse_fold(raw_fold)
        train_mask = _split_mask(dates, train_start, train_end)
        val_mask = _split_mask(dates, val_start, val_end)
        candidates: list[dict[str, Any]] = []
        thresholds: dict[str, float] = {}
        candidate_counts: dict[str, int] = {}
        for spec in specs:
            strength, direction = families[spec.family]
            train_x = strength[train_mask & np.isfinite(strength)]
            if train_x.size < 100:
                continue
            threshold = float(np.quantile(train_x, float(spec.quantile)))
            thresholds[f"{spec.family}:{spec.quantile:.4f}"] = threshold
            rows = _fast_candidate_rows(
                date_strings,
                strength,
                direction,
                family=spec.family,
                threshold=threshold,
                mask=val_mask,
                hold_bars=int(cfg.hold_bars),
                entry_delay_bars=int(cfg.entry_delay_bars),
                stride_bars=int(cfg.stride_bars),
                window_size=int(cfg.window_size),
            )
            for row in rows:
                # Priority is only for same-bar duplicate ordering; it is based
                # on fold-train threshold excess, not future outcome.
                row["score_mean"] = float(row["strength"]) / max(abs(threshold), 1e-12)
            candidate_counts[spec.family] = len(rows)
            candidates.extend(rows)
        merged = _dedupe_sort_rows(candidates)
        result = _simulate_rows(merged, market, sim_cfg)
        fold_rows.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "validation_start": val_start,
                "validation_end": val_end,
                "thresholds": thresholds,
                "candidate_counts_by_family": candidate_counts,
                "merged_candidate_count": len(merged),
                "validation": {"sim": result["sim"], "trade_stats": result["trade_stats"]},
            }
        )

    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg) | {"folds": list(cfg.folds), "combo_specs": [asdict(s) for s in specs]},
        "fold_score": _fold_score(fold_rows),
        "folds": fold_rows,
        "leakage_guard": {
            "each_family_threshold_fit_on_fold_train_only": True,
            "validation_not_used_for_threshold": True,
            "same_bar_priority_uses_threshold_excess_only": True,
            "features_use_rows_at_or_before_signal": True,
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--combo", default=RexComboRollingConfig.combo)
    p.add_argument("--hold-bars", type=int, default=RexComboRollingConfig.hold_bars)
    p.add_argument("--stride-bars", type=int, default=RexComboRollingConfig.stride_bars)
    p.add_argument("--window-size", type=int, default=RexComboRollingConfig.window_size)
    p.add_argument("--entry-delay-bars", type=int, default=RexComboRollingConfig.entry_delay_bars)
    p.add_argument("--folds", default=",".join(RexComboRollingConfig.folds))
    return p.parse_args()


def main() -> None:
    a = parse_args()
    cfg = RexComboRollingConfig(
        input_csv=a.input_csv,
        output=a.output,
        combo=a.combo,
        hold_bars=a.hold_bars,
        stride_bars=a.stride_bars,
        window_size=a.window_size,
        entry_delay_bars=a.entry_delay_bars,
        folds=tuple(x.strip() for x in str(a.folds).split(",") if x.strip()),
    )
    out = run(cfg)
    print(json.dumps({"fold_score": out["fold_score"], "folds": out["folds"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
