"""Validate REX-derived short predictions for manually declared bearish regimes.

This is not a universal all-market short alpha.  It validates the exported
REX/LLM dual-regime SHORT predictions under the intended use case: operator or
higher-level regime detector declares a persistent bearish regime, and only
SHORT TRADE predictions are executed.
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

from training.long_regime_combo_scan import LongComboScanConfig, _load_market, _split_mask
from training.short_regime_alpha_family_scan import _strict_short_sim
from training.strict_bar_backtest import _trade_stats


@dataclass(frozen=True)
class RexShortValidateConfig(LongComboScanConfig):
    predictions_jsonl: str = "results/rex_dual_regime_eval_2025_2026h1_predictions_2026-07-03.jsonl"
    exclude_from: str = "2026-06-02"
    leverage: float = 0.5
    fallback_hold_bars: int = 144


def _load_short_predictions(path: str) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pred = row.get("prediction", {})
        if pred.get("gate") == "TRADE" and pred.get("side") == "SHORT":
            rows.append(row)
    return rows


def _score_positions(market: pd.DataFrame, positions: np.ndarray, hold_bars: int, cfg: RexShortValidateConfig, start: str, end: str) -> dict[str, Any]:
    sim, returns = _strict_short_sim(
        positions,
        market=market,
        hold_bars=hold_bars,
        entry_delay_bars=int(cfg.entry_delay_bars),
        leverage=float(cfg.leverage),
        fee_rate=float(cfg.fee_rate),
        slippage_rate=float(cfg.slippage_rate),
        annualization_start=start,
        annualization_end=end,
    )
    return {"signals": int(len(positions)), "sim": sim, "trade_stats": _trade_stats(returns)}


def run(cfg: RexShortValidateConfig) -> dict[str, Any]:
    market = _load_market(cfg)
    dates = pd.to_datetime(market["date"])
    pred_rows = _load_short_predictions(cfg.predictions_jsonl)
    by_hold: dict[int, list[int]] = {}
    families = set()
    for row in pred_rows:
        pred = row.get("prediction", {})
        hold = int(pred.get("hold_bars") or row.get("source_action", {}).get("hold_bars") or cfg.fallback_hold_bars)
        by_hold.setdefault(hold, []).append(int(row["signal_pos"]))
        families.add(str(pred.get("family") or row.get("source_action", {}).get("family") or "unknown"))
    windows = [
        ("test_2024", "2024-01-01", "2025-01-01"),
        ("bear_eval_2025", "2025-01-01", "2026-01-01"),
        ("bear_ytd_2026", "2026-01-01", "2026-06-02"),
        ("bear_combined_2025_2026h1", "2025-01-01", "2026-06-02"),
        ("all_2024_2026h1", "2024-01-01", "2026-06-02"),
    ]
    results = {}
    for name, start, end in windows:
        # Current exported candidate uses one hold; still aggregate by hold defensively.
        if not by_hold:
            results[name] = _score_positions(market, np.array([], dtype=np.int64), int(cfg.fallback_hold_bars), cfg, start, end)
            continue
        if len(by_hold) == 1:
            hold, positions = next(iter(by_hold.items()))
            mask = _split_mask(dates, start, end)
            p = np.asarray([pos for pos in positions if 0 <= pos < len(mask) and mask[pos]], dtype=np.int64)
            results[name] = _score_positions(market, p, int(hold), cfg, start, end)
        else:
            # Conservative summary for unexpected variable holds: score each hold sleeve separately and sum metadata.
            sleeves = {}
            for hold, positions in by_hold.items():
                mask = _split_mask(dates, start, end)
                p = np.asarray([pos for pos in positions if 0 <= pos < len(mask) and mask[pos]], dtype=np.int64)
                sleeves[str(hold)] = _score_positions(market, p, int(hold), cfg, start, end)
            results[name] = {"variable_hold_sleeves": sleeves}
    monthly = []
    for start in pd.date_range("2025-01-01", "2026-06-01", freq="MS"):
        end = start + pd.offsets.MonthBegin(1)
        if start >= pd.Timestamp("2026-06-01"):
            end = pd.Timestamp("2026-06-02")
        if start >= pd.Timestamp("2026-06-02"):
            break
        if len(by_hold) != 1:
            continue
        hold, positions = next(iter(by_hold.items()))
        mask = _split_mask(dates, str(start.date()), str(end.date()))
        p = np.asarray([pos for pos in positions if 0 <= pos < len(mask) and mask[pos]], dtype=np.int64)
        scored = _score_positions(market, p, int(hold), cfg, str(start.date()), str(end.date()))
        monthly.append({"month": str(start.date())[:7], **scored})
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "input": {"rows": len(market), "start": str(market["date"].iloc[0]), "end": str(market["date"].iloc[-1])},
        "prediction_summary": {"short_signals": len(pred_rows), "families": sorted(families), "holds": {str(k): len(v) for k, v in by_hold.items()}},
        "protocol": "Execute only prediction.gate=TRADE and prediction.side=SHORT; intended for manually declared bearish regime; CAGR uses full calendar window including idle time; strict MDD uses intrabar highs for shorts.",
        "windows": results,
        "monthly_2025_2026h1": monthly,
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--predictions-jsonl", default=RexShortValidateConfig.predictions_jsonl)
    p.add_argument("--funding-csv", default="")
    p.add_argument("--premium-csv", default="")
    p.add_argument("--exclude-from", default=RexShortValidateConfig.exclude_from)
    p.add_argument("--leverage", type=float, default=RexShortValidateConfig.leverage)
    p.add_argument("--fallback-hold-bars", type=int, default=RexShortValidateConfig.fallback_hold_bars)
    p.add_argument("--entry-delay-bars", type=int, default=RexShortValidateConfig.entry_delay_bars)
    p.add_argument("--fee-rate", type=float, default=RexShortValidateConfig.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=RexShortValidateConfig.slippage_rate)
    return p.parse_args()


def _compact_window(payload: dict[str, Any]) -> dict[str, Any]:
    sim = payload["sim"]
    return {
        "signals": payload["signals"],
        "ret_pct": sim["total_return_pct"],
        "cagr_pct": sim["cagr_pct"],
        "mdd_pct": sim["strict_mdd_pct"],
        "ratio": sim["cagr_to_strict_mdd"],
        "trades": sim["trade_entries"],
        "p": payload["trade_stats"].get("p_value_mean_ret_approx"),
    }


def main() -> None:
    report = run(RexShortValidateConfig(**vars(parse_args())))
    print(json.dumps({"output": report["config"]["output"], "prediction_summary": report["prediction_summary"], "windows": {k: _compact_window(v) for k, v in report["windows"].items() if "sim" in v}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
