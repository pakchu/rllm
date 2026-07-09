"""Validate union ensembles of discovered long-regime alpha components.

This keeps the alpha components explicit instead of fitting a black-box gate:
- range expansion breakout
- HTF momentum continuation
- low-funding squeeze with trend confirmation
- volatility compression breakout
- weak-USDKRW macro relief momentum

The component thresholds are fixed from train-window quantiles discovered by the
family scan.  Evaluation CAGR is annualized over the full calendar window,
including idle/cash periods, and strict MDD includes intrabar adverse excursion.
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
from training.long_regime_interest_gate_validation import build_interest_features
from training.strict_bar_backtest import _trade_stats


@dataclass(frozen=True)
class UnionValidateConfig(LongComboScanConfig):
    exclude_from: str = "2026-06-02"
    leverage: float = 0.5


COMPONENTS: dict[str, list[tuple[str, str, float]]] = {
    "range_z70": [("rex_576_range_width_pct", "ge", 0.12959816105499766), ("close_zscore_48", "ge", 0.8418236952912183)],
    "range_bb90": [("rex_576_range_width_pct", "ge", 0.12959816105499766), ("bb_z", "ge", 1.6850824973528202)],
    "mom85_pos50": [("htf_1d_return_4", "ge", 0.06914164671210155), ("rex_8640_range_pos", "ge", 0.11270125172457102)],
    "funding10_trend70": [("funding_rate", "le", -0.0000167), ("trend_96", "ge", 0.007485218212390219)],
    "macro_usdkrw10_mom70": [("usdkrw_momentum", "le", -0.00260933911622682), ("htf_1d_return_1", "ge", 0.013123763526011079)],
    "compress05_trend80": [("rex_2016_range_width_pct", "le", 0.05074314472814484), ("trend_24", "ge", 0.004797228904277088)],
}

CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "combo_A_range_mom_funding_compress",
        "components": ["range_bb90", "mom85_pos50", "funding10_trend70", "compress05_trend80"],
        "hold_bars": 576,
        "stride_bars": 12,
    },
    {
        "name": "combo_B_range_compress_sparse",
        "components": ["range_z70", "range_bb90", "compress05_trend80"],
        "hold_bars": 576,
        "stride_bars": 12,
    },
    {
        "name": "combo_C_range_mom_macro",
        "components": ["range_z70", "range_bb90", "mom85_pos50", "macro_usdkrw10_mom70"],
        "hold_bars": 216,
        "stride_bars": 24,
    },
    {
        "name": "combo_D_range_compress_432",
        "components": ["range_z70", "range_bb90", "compress05_trend80"],
        "hold_bars": 432,
        "stride_bars": 12,
    },
]

WINDOWS: list[tuple[str, str, str]] = [
    ("test_2024", "2024-01-01", "2025-01-01"),
    ("eval_2025", "2025-01-01", "2026-01-01"),
    ("eval_2026h1", "2026-01-01", "2026-06-02"),
    ("combined_2024_2026h1", "2024-01-01", "2026-06-02"),
]


def _component_mask(features: pd.DataFrame, name: str) -> np.ndarray:
    mask = np.ones(len(features), dtype=bool)
    for feature, op, threshold in COMPONENTS[name]:
        values = features[feature].to_numpy(float)
        mask &= ((values <= threshold) if op == "le" else (values >= threshold)) & np.isfinite(values)
    return mask


def _union_mask(features: pd.DataFrame, component_names: list[str]) -> np.ndarray:
    active = np.zeros(len(features), dtype=bool)
    for name in component_names:
        active |= _component_mask(features, name)
    return active


def _score_window(
    *,
    market: pd.DataFrame,
    dates: pd.Series,
    active: np.ndarray,
    cfg: UnionValidateConfig,
    start: str,
    end: str,
    hold_bars: int,
    stride_bars: int,
) -> dict[str, Any]:
    wmask = _split_mask(dates, start, end)
    positions = np.arange(max(0, int(cfg.window_size) - 1), max(0, len(market) - hold_bars - int(cfg.entry_delay_bars) - 1), stride_bars, dtype=np.int64)
    p = positions[active[positions] & wmask[positions]]
    sim, returns = _strict_long_sim(
        p,
        market=market,
        hold_bars=hold_bars,
        entry_delay_bars=int(cfg.entry_delay_bars),
        leverage=float(cfg.leverage),
        fee_rate=float(cfg.fee_rate),
        slippage_rate=float(cfg.slippage_rate),
        annualization_start=start,
        annualization_end=end,
    )
    return {"signals": int(len(p)), "sim": sim, "trade_stats": _trade_stats(returns)}


def run(cfg: UnionValidateConfig) -> dict[str, Any]:
    market = _load_market(cfg)
    base_features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    features = pd.concat([base_features, build_interest_features(market, base_features)], axis=1)
    dates = pd.to_datetime(market["date"])
    rows = []
    for candidate in CANDIDATES:
        active = _union_mask(features, list(candidate["components"]))
        windows = {
            name: _score_window(
                market=market,
                dates=dates,
                active=active,
                cfg=cfg,
                start=start,
                end=end,
                hold_bars=int(candidate["hold_bars"]),
                stride_bars=int(candidate["stride_bars"]),
            )
            for name, start, end in WINDOWS
        }
        rows.append({"candidate": candidate, "windows": windows})
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "components": COMPONENTS,
        "protocol": "Fixed union candidates; component thresholds from train-window quantiles; full-window CAGR; strict OHLC MDD.",
        "input": {"rows": len(market), "start": str(market["date"].iloc[0]), "end": str(market["date"].iloc[-1])},
        "rows": rows,
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
    p.add_argument("--exclude-from", default=UnionValidateConfig.exclude_from)
    p.add_argument("--leverage", type=float, default=UnionValidateConfig.leverage)
    p.add_argument("--window-size", type=int, default=UnionValidateConfig.window_size)
    p.add_argument("--entry-delay-bars", type=int, default=UnionValidateConfig.entry_delay_bars)
    p.add_argument("--fee-rate", type=float, default=UnionValidateConfig.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=UnionValidateConfig.slippage_rate)
    return p.parse_args()


def main() -> None:
    report = run(UnionValidateConfig(**vars(parse_args())))
    summary = []
    for row in report["rows"]:
        item = {"name": row["candidate"]["name"], "components": row["candidate"]["components"], "hold_bars": row["candidate"]["hold_bars"], "stride_bars": row["candidate"]["stride_bars"]}
        for window_name, payload in row["windows"].items():
            sim = payload["sim"]
            item[window_name] = {
                "signals": payload["signals"],
                "ret_pct": sim["total_return_pct"],
                "cagr_pct": sim["cagr_pct"],
                "mdd_pct": sim["strict_mdd_pct"],
                "ratio": sim["cagr_to_strict_mdd"],
                "trades": sim["trade_entries"],
                "p": payload["trade_stats"].get("p_value_mean_ret_approx"),
            }
        summary.append(item)
    print(json.dumps({"output": report["config"]["output"], "summary": summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
