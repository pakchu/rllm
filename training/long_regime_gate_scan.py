"""Long-allowed regime gate scan for REX long entries.

This searches whether an interpretable regime filter can rescue the long-side
REX entry edge.  Entry thresholds and gate thresholds are fit on train only,
selection uses train+validation only, and 2025 eval is report-only.
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
from training.long_regime_combo_scan import LongComboScanConfig, _load_market, _split_mask, _strict_long_sim
from training.strict_bar_backtest import _trade_stats


@dataclass(frozen=True)
class LongRegimeGateConfig(LongComboScanConfig):
    rules: str = "pb30,pb30_funding"
    entry_quantiles: str = "0.75,0.80,0.85"
    gate_quantiles: str = "0.20,0.30,0.70,0.80"
    gate_features: str = (
        "htf_1d_return_4,htf_3d_return_4,htf_1w_return_4,"
        "rex_8640_range_width_pct,rex_8640_range_pos,range_vol,trend_96,window_drawdown,"
        "funding_zscore,premium_index_zscore,dxy_zscore,kimchi_premium_zscore,usdkrw_zscore"
    )
    hold_bars: str = "144,288"
    stride_bars: str = "12,24"
    leverage: float = 0.5
    max_train_mdd: float = 32.0
    max_val_mdd: float = 18.0


def _parse_list(raw: str, cast: Any) -> list[Any]:
    return [cast(x.strip()) for x in str(raw).split(",") if x.strip()]


def _arr(features: pd.DataFrame, col: str) -> np.ndarray:
    return features.get(col, pd.Series(0.0, index=features.index)).to_numpy(dtype=float)


def _entry_active(
    features: pd.DataFrame,
    *,
    rule: str,
    train_mask: np.ndarray,
    quantile: float,
) -> tuple[np.ndarray, list[dict[str, Any]]] | None:
    pb30 = _arr(features, "rex_8640_max_to_cur_pct")
    funding = _arr(features, "funding_zscore")
    if rule == "pb30":
        components = [("pb30", pb30, "ge")]
    elif rule == "pb30_funding":
        components = [("pb30", pb30, "ge"), ("funding_low", funding, "le")]
    else:
        raise ValueError(f"unknown rule: {rule}")
    active = np.ones(len(features), dtype=bool)
    spec: list[dict[str, Any]] = []
    for name, values, op in components:
        ref = values[train_mask & np.isfinite(values)]
        if ref.size < 200:
            return None
        thr = float(np.quantile(ref, float(quantile)))
        active &= (values >= thr if op == "ge" else values <= thr) & np.isfinite(values)
        spec.append({"name": name, "op": op, "quantile": float(quantile), "threshold": thr})
    return active, spec


def _gate_active(
    features: pd.DataFrame,
    *,
    feature: str,
    op: str,
    train_mask: np.ndarray,
    quantile: float,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    values = _arr(features, feature)
    ref = values[train_mask & np.isfinite(values)]
    if ref.size < 200 or float(np.nanstd(ref)) <= 1e-12:
        return None
    thr = float(np.quantile(ref, float(quantile)))
    active = (values >= thr if op == "ge" else values <= thr) & np.isfinite(values)
    return active, {"feature": feature, "op": op, "quantile": float(quantile), "threshold": thr}


def _score(row: dict[str, Any], cfg: LongRegimeGateConfig) -> float:
    train = row["train"]["sim"]
    val = row["val"]["sim"]
    if int(train["trade_entries"]) < int(cfg.min_train_trades) or int(val["trade_entries"]) < int(cfg.min_val_trades):
        return -1e9
    if float(train["cagr_pct"]) <= 0.0 or float(val["cagr_pct"]) <= 0.0:
        return -1e9
    if float(train["strict_mdd_pct"]) > float(cfg.max_train_mdd) or float(val["strict_mdd_pct"]) > float(cfg.max_val_mdd):
        return -1e9
    tr = float(train["cagr_to_strict_mdd"])
    va = float(val["cagr_to_strict_mdd"])
    val_p = float(row["val"]["trade_stats"].get("p_value_mean_ret_approx", 1.0))
    return va + 0.7 * tr + min(1.0, float(val["trade_entries"]) / 120.0) - 0.25 * abs(va - tr) - 0.2 * val_p


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    out = {k: row[k] for k in ("rule", "entry_spec", "gate", "hold_bars", "hold_hours", "stride_bars", "selection_score")}
    for split in ("train", "val", "eval"):
        sim = row[split]["sim"]
        out[split] = {
            "cagr_pct": sim["cagr_pct"],
            "strict_mdd_pct": sim["strict_mdd_pct"],
            "cagr_to_strict_mdd": sim["cagr_to_strict_mdd"],
            "trade_entries": sim["trade_entries"],
            "win_rate": sim["win_rate"],
            "p_value": row[split]["trade_stats"].get("p_value_mean_ret_approx"),
        }
    return out


def run_scan(cfg: LongRegimeGateConfig) -> dict[str, Any]:
    market = _load_market(cfg)
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    dates = pd.to_datetime(market["date"])
    split_masks = {
        "train": _split_mask(dates, cfg.train_start, cfg.train_end),
        "val": _split_mask(dates, cfg.val_start, cfg.val_end),
        "eval": _split_mask(dates, cfg.eval_start, cfg.eval_end),
    }
    split_bounds = {
        "train": (cfg.train_start, cfg.train_end),
        "val": (cfg.val_start, cfg.val_end),
        "eval": (cfg.eval_start, cfg.eval_end),
    }
    max_hold = max(_parse_list(cfg.hold_bars, int))
    positions_by_stride = {
        s: np.arange(max(0, int(cfg.window_size) - 1), max(0, len(market) - max_hold - int(cfg.entry_delay_bars) - 1), s, dtype=np.int64)
        for s in _parse_list(cfg.stride_bars, int)
    }
    rows: list[dict[str, Any]] = []
    for rule in _parse_list(cfg.rules, str):
        for entry_q in _parse_list(cfg.entry_quantiles, float):
            entry = _entry_active(features, rule=rule, train_mask=split_masks["train"], quantile=entry_q)
            if entry is None:
                continue
            entry_mask, entry_spec = entry
            for gate_feature in _parse_list(cfg.gate_features, str):
                if gate_feature not in features.columns:
                    continue
                for gate_q in _parse_list(cfg.gate_quantiles, float):
                    op = "le" if gate_q < 0.5 else "ge"
                    gate = _gate_active(features, feature=gate_feature, op=op, train_mask=split_masks["train"], quantile=gate_q)
                    if gate is None:
                        continue
                    gate_mask, gate_spec = gate
                    active = entry_mask & gate_mask
                    if int((active & split_masks["train"]).sum()) < 200:
                        continue
                    for hold in _parse_list(cfg.hold_bars, int):
                        for stride, base_positions in positions_by_stride.items():
                            row: dict[str, Any] = {
                                "rule": rule,
                                "entry_spec": entry_spec,
                                "gate": gate_spec,
                                "hold_bars": int(hold),
                                "hold_hours": float(hold) * 5.0 / 60.0,
                                "stride_bars": int(stride),
                            }
                            for split, mask in split_masks.items():
                                positions = base_positions[active[base_positions] & mask[base_positions]]
                                sim, returns = _strict_long_sim(
                                    positions,
                                    market=market,
                                    hold_bars=int(hold),
                                    entry_delay_bars=int(cfg.entry_delay_bars),
                                    leverage=float(cfg.leverage),
                                    fee_rate=float(cfg.fee_rate),
                                    slippage_rate=float(cfg.slippage_rate),
                                    annualization_start=split_bounds[split][0],
                                    annualization_end=split_bounds[split][1],
                                )
                                row[split] = {"sim": sim, "trade_stats": _trade_stats(returns)}
                            row["selection_score"] = _score(row, cfg)
                            rows.append(row)
    ranked = sorted(rows, key=lambda r: float(r["selection_score"]), reverse=True)
    robust = sorted(
        [
            r
            for r in rows
            if int(r["train"]["sim"]["trade_entries"]) >= int(cfg.min_train_trades)
            and int(r["val"]["sim"]["trade_entries"]) >= int(cfg.min_val_trades)
            and int(r["eval"]["sim"]["trade_entries"]) >= int(cfg.min_eval_trades_for_robust)
            and all(float(r[s]["sim"]["cagr_pct"]) > 0.0 for s in ("train", "val", "eval"))
        ],
        key=lambda r: min(float(r[s]["sim"]["cagr_to_strict_mdd"]) for s in ("train", "val", "eval")),
        reverse=True,
    )
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "input": {"rows": len(market), "start": str(market["date"].iloc[0]), "end": str(market["date"].iloc[-1])},
        "selection_protocol": "entry and regime-gate thresholds fit on train only; selection uses train+val only; eval is report-only",
        "top_selection": [_compact(r) for r in ranked[:50]],
        "top_robust": [_compact(r) for r in robust[:50]],
        "all_count": len(rows),
        "leakage_guard": {
            "market_rows_after_exclude_from_removed_before_feature_build": True,
            "entry_thresholds_fit_train_only": True,
            "gate_thresholds_fit_train_only": True,
            "eval_not_used_for_selection": True,
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--funding-csv", default="")
    p.add_argument("--premium-csv", default="")
    for field in (
        "rules",
        "entry-quantiles",
        "gate-quantiles",
        "gate-features",
        "train-start",
        "train-end",
        "val-start",
        "val-end",
        "eval-start",
        "eval-end",
        "exclude-from",
        "hold-bars",
        "stride-bars",
    ):
        p.add_argument(f"--{field}", default=getattr(LongRegimeGateConfig, field.replace("-", "_")))
    p.add_argument("--window-size", type=int, default=LongRegimeGateConfig.window_size)
    p.add_argument("--entry-delay-bars", type=int, default=LongRegimeGateConfig.entry_delay_bars)
    p.add_argument("--leverage", type=float, default=LongRegimeGateConfig.leverage)
    p.add_argument("--fee-rate", type=float, default=LongRegimeGateConfig.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=LongRegimeGateConfig.slippage_rate)
    p.add_argument("--min-train-trades", type=int, default=LongRegimeGateConfig.min_train_trades)
    p.add_argument("--min-val-trades", type=int, default=LongRegimeGateConfig.min_val_trades)
    p.add_argument("--min-eval-trades-for-robust", type=int, default=LongRegimeGateConfig.min_eval_trades_for_robust)
    p.add_argument("--max-train-mdd", type=float, default=LongRegimeGateConfig.max_train_mdd)
    p.add_argument("--max-val-mdd", type=float, default=LongRegimeGateConfig.max_val_mdd)
    return p.parse_args()


def main() -> None:
    report = run_scan(LongRegimeGateConfig(**vars(parse_args())))
    print(
        json.dumps(
            {
                "output": report["config"]["output"],
                "input": report["input"],
                "all_count": report["all_count"],
                "top_selection": report["top_selection"][:10],
                "top_robust": report["top_robust"][:10],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
