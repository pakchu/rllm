"""Test beta short features as selectors/boosters on non-REX short alphas.

Base sleeves:
- FX-stress short: htf_3d_return_1 weak + USDKRW stress.
- Premium-discount short: htf_3d_range_pos low + premium_index_zscore low.

Overlay features:
- kimchi unwind: htf_3d_return_1 weak + kimchi_premium_change falling.
- sell-flow liquidation: weak 3d + taker sell imbalance + quote volume.
- upper-shadow bounce rejection: upper shadow + strong 4h rebound.

The scan tests intersection filters, OR unions, and simple size boosters using a
bar-level portfolio simulator with strict short adverse-high MDD.  It is meant
to answer whether the beta features add marginal value rather than serve as
standalone entries.
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
from training.long_regime_combo_scan import LongComboScanConfig, _load_market, _split_mask
from training.long_regime_interest_gate_validation import build_interest_features


@dataclass(frozen=True)
class SelectorOverlayConfig(LongComboScanConfig):
    exclude_from: str = "2026-06-02"
    leverage: float = 0.5
    max_gross: float = 1.0


WINDOWS = {
    "test2024": ("2024-01-01", "2025-01-01"),
    "eval2025": ("2025-01-01", "2026-01-01"),
    "ytd2026": ("2026-01-01", "2026-06-02"),
    "bear_combined_2025_2026h1": ("2025-01-01", "2026-06-02"),
}

# Fixed thresholds from the previously recorded candidate artifacts.
COMPONENTS: dict[str, list[tuple[str, str, float]]] = {
    "fx_q20": [("htf_3d_return_1", "le", -0.03252949727545951), ("usdkrw_zscore", "ge", 1.3870063774765273)],
    "fx_q30": [("htf_3d_return_1", "le", -0.019505961757087298), ("usdkrw_zscore", "ge", 1.3870063774765273)],
    "premium_panic": [("htf_3d_range_pos", "le", -0.5114186851211089), ("premium_index_zscore", "le", -1.472093119977103)],
    "kimchi_unwind": [("htf_3d_return_1", "le", -0.030319683322027502), ("kimchi_premium_change", "le", -0.00461237517905572)],
    "sellflow_liq": [("htf_3d_return_1", "le", -0.030319683322027502), ("taker_imbalance", "le", -0.18814067006213206), ("quote_vol_z_1d", "ge", 0.4294945086445027)],
    "upper_reject": [("upper_shadow", "ge", 0.0015885139737276228), ("htf_4h_return_4", "ge", 0.017497690522213727)],
    "dxy_weak3d": [("htf_3d_return_1", "le", -0.030319683322027502), ("dxy_momentum", "ge", 0.0010202304677308)],
}

BASE_SPECS: dict[str, dict[str, Any]] = {
    "fx_q20_tp4_sl25": {"components": ["fx_q20"], "hold": 288, "tp": 0.04, "sl": 0.025, "weight": 1.0},
    "premium_tp25_sl15": {"components": ["premium_panic"], "hold": 288, "tp": 0.025, "sl": 0.015, "weight": 1.0},
    "premium_tp4_sl25": {"components": ["premium_panic"], "hold": 288, "tp": 0.04, "sl": 0.025, "weight": 1.0},
}


def _mask_for(features: pd.DataFrame, component: str) -> np.ndarray:
    active = np.ones(len(features), dtype=bool)
    for feature, op, threshold in COMPONENTS[component]:
        values = features[feature].to_numpy(float)
        active &= ((values <= threshold) if op == "le" else (values >= threshold)) & np.isfinite(values)
    return active


def _component_masks(features: pd.DataFrame) -> dict[str, np.ndarray]:
    return {name: _mask_for(features, name) for name in COMPONENTS}


def _events_from_mask(
    active: np.ndarray,
    *,
    dates: pd.Series,
    start: str,
    end: str,
    hold: int,
    stride: int,
    weight: float,
) -> list[tuple[int, int, float]]:
    wmask = _split_mask(dates, start, end)
    positions = np.arange(143, max(0, len(active) - hold - 2), stride, dtype=np.int64)
    signals = positions[active[positions] & wmask[positions]]
    return [(int(pos), int(hold), float(weight)) for pos in signals]


def _simulate_portfolio(
    market: pd.DataFrame,
    events: list[tuple[int, int, float]],
    *,
    start: str,
    end: str,
    leverage: float,
    max_gross: float,
    fee_rate: float,
    slippage_rate: float,
    take_profit: float | None,
    stop_loss: float | None,
) -> dict[str, Any]:
    opens = market["open"].to_numpy(float)
    highs = market["high"].to_numpy(float)
    lows = market["low"].to_numpy(float)
    events = sorted(events, key=lambda x: x[0])
    eq = peak = 1.0
    max_dd = 0.0
    returns: list[float] = []
    # Conservative sequential non-overlap at portfolio level. If a booster and base fire same bar, their weights combine up to max_gross.
    merged: dict[int, tuple[int, float]] = {}
    for pos, hold, weight in events:
        if pos not in merged:
            merged[pos] = (hold, 0.0)
        old_hold, old_weight = merged[pos]
        merged[pos] = (max(old_hold, hold), min(float(max_gross), old_weight + weight))
    next_allowed = 0
    for pos, (hold, weight) in sorted(merged.items()):
        if pos < next_allowed or weight <= 0:
            continue
        entry_pos = int(pos) + 1
        exit_pos = entry_pos + int(hold)
        if exit_pos >= len(market):
            continue
        entry = float(opens[entry_pos])
        if entry <= 0:
            continue
        lev = float(leverage) * float(weight)
        cost = (float(fee_rate) + float(slippage_rate)) * lev
        entry_eq = eq
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, 1.0 - eq / peak)
        exit_ret = -(float(opens[exit_pos]) - entry) / entry
        actual_exit = exit_pos
        for j in range(entry_pos, exit_pos):
            adverse = (float(highs[j]) - entry) / entry
            max_dd = max(max_dd, 1.0 - max(0.0, eq * (1.0 - lev * adverse)) / peak)
            if stop_loss is not None and float(highs[j]) >= entry * (1.0 + float(stop_loss)):
                exit_ret = -float(stop_loss)
                actual_exit = j
                break
            if take_profit is not None and float(lows[j]) <= entry * (1.0 - float(take_profit)):
                exit_ret = float(take_profit)
                actual_exit = j
                break
        eq *= max(0.0, 1.0 + lev * exit_ret)
        peak = max(peak, eq)
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, 1.0 - eq / peak)
        peak = max(peak, eq)
        returns.append(eq / entry_eq - 1.0)
        next_allowed = actual_exit + 1
    years = max(1.0 / 365.25, (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / (365.25 * 24 * 3600))
    ret_pct = (eq - 1.0) * 100.0
    cagr_pct = ((eq ** (1.0 / years) - 1.0) * 100.0) if eq > 0 else -100.0
    mdd_pct = max_dd * 100.0
    return {
        "ret_pct": ret_pct,
        "cagr_pct": cagr_pct,
        "mdd_pct": mdd_pct,
        "ratio": cagr_pct / mdd_pct if mdd_pct > 1e-12 else (float("inf") if cagr_pct > 0 else 0.0),
        "trades": len(returns),
        "win_rate": sum(r > 0 for r in returns) / len(returns) if returns else 0.0,
        "signals": len(merged),
    }


def _build_variants(masks: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    overlays = ["kimchi_unwind", "sellflow_liq", "upper_reject", "dxy_weak3d"]
    for base_name, base in BASE_SPECS.items():
        base_mask = np.logical_or.reduce([masks[c] for c in base["components"]])
        variants.append({"name": base_name, "kind": "base", "mask": base_mask, "hold": base["hold"], "tp": base["tp"], "sl": base["sl"], "weight": base["weight"], "components": base["components"]})
        for overlay in overlays:
            variants.append({"name": f"{base_name}__filter_{overlay}", "kind": "filter", "mask": base_mask & masks[overlay], "hold": base["hold"], "tp": base["tp"], "sl": base["sl"], "weight": base["weight"], "components": base["components"] + [overlay]})
            variants.append({"name": f"{base_name}__union_{overlay}", "kind": "union", "mask": base_mask | masks[overlay], "hold": base["hold"], "tp": base["tp"], "sl": base["sl"], "weight": base["weight"], "components": base["components"] + [overlay]})
    # Simple portfolio unions of the two primary independent alphas.
    variants.append({"name": "fx_q20_plus_premium_tp25", "kind": "portfolio_union", "mask": masks["fx_q20"] | masks["premium_panic"], "hold": 288, "tp": 0.025, "sl": 0.015, "weight": 1.0, "components": ["fx_q20", "premium_panic"]})
    variants.append({"name": "fx_q20_plus_premium_tp4", "kind": "portfolio_union", "mask": masks["fx_q20"] | masks["premium_panic"], "hold": 288, "tp": 0.04, "sl": 0.025, "weight": 1.0, "components": ["fx_q20", "premium_panic"]})
    variants.append({"name": "fx_q20_premium_kimchi_union", "kind": "portfolio_union", "mask": masks["fx_q20"] | masks["premium_panic"] | masks["kimchi_unwind"], "hold": 288, "tp": 0.04, "sl": 0.025, "weight": 1.0, "components": ["fx_q20", "premium_panic", "kimchi_unwind"]})
    return variants


def run(cfg: SelectorOverlayConfig) -> dict[str, Any]:
    market = _load_market(cfg)
    base_features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    features = pd.concat([base_features, build_interest_features(market, base_features)], axis=1)
    dates = pd.to_datetime(market["date"])
    masks = _component_masks(features)
    variants = _build_variants(masks)
    rows = []
    for variant in variants:
        stats = {}
        for window_name, (start, end) in WINDOWS.items():
            events = _events_from_mask(variant["mask"], dates=dates, start=start, end=end, hold=int(variant["hold"]), stride=12, weight=float(variant["weight"]))
            stats[window_name] = _simulate_portfolio(
                market,
                events,
                start=start,
                end=end,
                leverage=float(cfg.leverage),
                max_gross=float(cfg.max_gross),
                fee_rate=float(cfg.fee_rate),
                slippage_rate=float(cfg.slippage_rate),
                take_profit=variant["tp"],
                stop_loss=variant["sl"],
            )
        rows.append({k: v for k, v in variant.items() if k != "mask"} | {"stats": stats})
    rows.sort(key=lambda r: (float(r["stats"]["bear_combined_2025_2026h1"]["ratio"]), float(r["stats"]["bear_combined_2025_2026h1"]["ret_pct"])), reverse=True)
    report = {"as_of": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg), "components": COMPONENTS, "top": rows, "all_count": len(rows)}
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--funding-csv", default="")
    p.add_argument("--premium-csv", default="")
    p.add_argument("--exclude-from", default=SelectorOverlayConfig.exclude_from)
    p.add_argument("--leverage", type=float, default=SelectorOverlayConfig.leverage)
    p.add_argument("--max-gross", type=float, default=SelectorOverlayConfig.max_gross)
    p.add_argument("--window-size", type=int, default=SelectorOverlayConfig.window_size)
    p.add_argument("--fee-rate", type=float, default=SelectorOverlayConfig.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=SelectorOverlayConfig.slippage_rate)
    return p.parse_args()


def main() -> None:
    report = run(SelectorOverlayConfig(**vars(parse_args())))
    print(json.dumps({"output": report["config"]["output"], "all_count": report["all_count"], "top": report["top"][:20]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
