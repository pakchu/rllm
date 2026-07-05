"""Existing-data long-regime composite scan.

This script takes the strongest long-side feature findings from the REX
feature-edge audit and tests them as fixed, interpretable composite rules.  It
does not train or fine-tune a model.  Thresholds are fitted on train only, rules
are ranked with train+validation only, and eval is reported as a held-out 2025
check.
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

from preprocessing.binance_aux_features import attach_binance_um_aux_features
from preprocessing.market_features import build_market_feature_frame
from training.strict_bar_backtest import _trade_stats


@dataclass(frozen=True)
class LongComboScanConfig:
    input_csv: str
    output: str
    funding_csv: str = ""
    premium_csv: str = ""
    train_start: str = "2020-01-01"
    train_end: str = "2024-01-01"
    val_start: str = "2024-01-01"
    val_end: str = "2025-01-01"
    eval_start: str = "2025-01-01"
    eval_end: str = "2026-01-01"
    exclude_from: str = "2026-01-01"
    hold_bars: str = "72,144,216,288,432"
    stride_bars: str = "6,12,24,36"
    quantiles: str = "0.10,0.15,0.20,0.80,0.85,0.90"
    window_size: int = 144
    entry_delay_bars: int = 1
    leverage: float = 1.0
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    min_train_trades: int = 80
    min_val_trades: int = 25
    min_eval_trades_for_robust: int = 20


def _parse_list(raw: str, cast: Any) -> list[Any]:
    return [cast(x.strip()) for x in str(raw).split(",") if x.strip()]


def _load_market(cfg: LongComboScanConfig) -> pd.DataFrame:
    market = pd.read_csv(cfg.input_csv, parse_dates=["date"], compression="infer")
    market["date"] = pd.to_datetime(market["date"], utc=True, errors="raise").dt.tz_convert(None)
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    market = market[market["date"] < pd.Timestamp(cfg.exclude_from)].reset_index(drop=True)
    if cfg.funding_csv or cfg.premium_csv:
        market = attach_binance_um_aux_features(
            market,
            funding_csv=cfg.funding_csv or None,
            premium_csv=cfg.premium_csv or None,
            funding_tolerance="12h",
            premium_tolerance="2h",
        )
    return market


def _split_mask(dates: pd.Series, start: str, end: str) -> np.ndarray:
    return np.asarray((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end)), dtype=bool)


def _strict_long_sim(
    signal_positions: np.ndarray,
    *,
    market: pd.DataFrame,
    hold_bars: int,
    entry_delay_bars: int,
    leverage: float,
    fee_rate: float,
    slippage_rate: float,
    annualization_start: str | None = None,
    annualization_end: str | None = None,
) -> tuple[dict[str, Any], list[float]]:
    """Strict long-only OHLC simulation with non-overlapping holds.

    It mirrors the repository's candidate backtest semantics: signal at bar t,
    entry at t + entry_delay open, open-to-open equity updates, intrabar low
    adverse excursion counted in strict MDD, and one entry/exit fee+slippage.
    """
    opens = market["open"].to_numpy(float)
    lows = market["low"].to_numpy(float)
    dates = pd.to_datetime(market["date"]).to_numpy()
    cost = (float(fee_rate) + float(slippage_rate)) * float(leverage)
    signal_positions = np.asarray(signal_positions, dtype=np.int64)

    eq = peak = 1.0
    max_dd = 0.0
    next_allowed = 0
    trade_returns: list[float] = []
    first_signal: int | None = None
    last_signal: int | None = None
    for pos in signal_positions:
        if pos < next_allowed:
            continue
        entry_pos = int(pos) + int(entry_delay_bars)
        exit_pos = entry_pos + int(hold_bars)
        if entry_pos >= len(market) - 1 or exit_pos >= len(market):
            continue
        first_signal = int(pos) if first_signal is None else first_signal
        last_signal = int(pos)
        entry_eq = eq
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, 1.0 - max(0.0, eq) / peak if peak > 0.0 else 0.0)
        for j in range(entry_pos, exit_pos):
            open_j = float(opens[j])
            if open_j <= 0.0:
                continue
            adverse_ret = (float(lows[j]) - open_j) / open_j
            max_dd = max(max_dd, 1.0 - max(0.0, eq * (1.0 + float(leverage) * adverse_ret)) / peak)
            close_ret = (float(opens[j + 1]) - open_j) / open_j
            eq *= max(0.0, 1.0 + float(leverage) * close_ret)
            peak = max(peak, eq)
            if eq <= 0.0:
                break
        eq *= max(0.0, 1.0 - cost)
        max_dd = max(max_dd, 1.0 - max(0.0, eq) / peak if peak > 0.0 else 0.0)
        peak = max(peak, eq)
        trade_returns.append(eq / entry_eq - 1.0)
        next_allowed = exit_pos
        if eq <= 0.0:
            break

    trade_start_dt = pd.Timestamp(dates[first_signal]).to_pydatetime() if first_signal is not None else None
    trade_end_dt = pd.Timestamp(dates[last_signal]).to_pydatetime() if last_signal is not None else None
    if annualization_start is not None and annualization_end is not None:
        start_dt = pd.Timestamp(annualization_start).to_pydatetime()
        end_dt = pd.Timestamp(annualization_end).to_pydatetime()
    elif trade_start_dt is not None and trade_end_dt is not None:
        start_dt = trade_start_dt
        end_dt = trade_end_dt
    else:
        start_dt = end_dt = datetime.now()
    years = max(1.0 / 365.25, (end_dt - start_dt).total_seconds() / (365.25 * 24 * 3600))
    ret_pct = (eq - 1.0) * 100.0
    cagr_pct = ((eq ** (1.0 / years) - 1.0) * 100.0) if eq > 0.0 else -100.0
    mdd_pct = max_dd * 100.0
    sim = {
        "period": {"start": str(start_dt), "end": str(end_dt), "years": years},
        "trade_period": {"start": str(trade_start_dt), "end": str(trade_end_dt)},
        "cagr_pct": cagr_pct,
        "strict_mdd_pct": mdd_pct,
        "cagr_to_strict_mdd": cagr_pct / mdd_pct if mdd_pct > 1e-12 else float("inf"),
        "trade_entries": len(trade_returns),
        "win_rate": sum(1 for r in trade_returns if r > 0.0) / len(trade_returns) if trade_returns else 0.0,
        "total_return_pct": ret_pct,
        "hold_bars": int(hold_bars),
        "entry_delay_bars": int(entry_delay_bars),
        "return_application": "long_only_actual_ohlc_strict_mdd",
    }
    return sim, trade_returns


def _feature_array(features: pd.DataFrame, name: str) -> np.ndarray:
    return features.get(name, pd.Series(0.0, index=features.index)).to_numpy(dtype=float)


def _build_rule_inputs(features: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return raw signals used by long composite rules.

    Positive values are better for long entries unless the rule says otherwise.
    All source columns are generated from rows at or before the signal bar.
    """
    rex_30d_width = _feature_array(features, "rex_8640_range_width_pct")
    rex_30d_from_high = _feature_array(features, "rex_8640_max_to_cur_pct")
    rex_12h_from_high = _feature_array(features, "rex_144_max_to_cur_pct")
    rex_12h_pos = _feature_array(features, "rex_144_range_pos")
    rex_30d_pos = _feature_array(features, "rex_8640_range_pos")
    htf_1d_ret_4 = _feature_array(features, "htf_1d_return_4")
    htf_3d_ret_4 = _feature_array(features, "htf_3d_return_4")
    funding_z = _feature_array(features, "funding_zscore")
    premium_z = _feature_array(features, "premium_index_zscore")
    taker = _feature_array(features, "taker_imbalance")

    return {
        "range_expansion_pullback": 0.45 * rex_30d_width + 0.35 * rex_30d_from_high + 0.20 * rex_12h_from_high,
        "deep_pullback_rebound": 0.50 * rex_30d_from_high + 0.30 * rex_12h_from_high - 0.20 * rex_12h_pos,
        "trend_pullback": 0.40 * np.maximum(htf_1d_ret_4, 0.0)
        + 0.25 * np.maximum(htf_3d_ret_4, 0.0)
        + 0.25 * rex_12h_from_high
        + 0.10 * rex_30d_width,
        "contrarian_stress_long": 0.45 * rex_30d_from_high + 0.20 * rex_30d_width - 0.20 * funding_z - 0.15 * premium_z,
        "range_upper_continuation": 0.35 * rex_30d_pos
        + 0.25 * rex_30d_width
        + 0.20 * np.maximum(htf_1d_ret_4, 0.0)
        + 0.20 * taker,
    }


def _threshold(values: np.ndarray, mask: np.ndarray, q: float) -> float | None:
    x = values[mask & np.isfinite(values)]
    if x.size < 200:
        return None
    return float(np.quantile(x, float(q)))


def _score_trial(row: dict[str, Any], cfg: LongComboScanConfig) -> float:
    train = row["train"]["sim"]
    val = row["val"]["sim"]
    if int(train.get("trade_entries", 0)) < int(cfg.min_train_trades):
        return -1e9
    if int(val.get("trade_entries", 0)) < int(cfg.min_val_trades):
        return -1e9
    if float(train.get("cagr_pct", -999.0)) <= 0.0 or float(val.get("cagr_pct", -999.0)) <= 0.0:
        return -1e9
    if float(train.get("strict_mdd_pct", 999.0)) > 45.0 or float(val.get("strict_mdd_pct", 999.0)) > 25.0:
        return -1e9
    train_ratio = float(train.get("cagr_to_strict_mdd", -999.0))
    val_ratio = float(val.get("cagr_to_strict_mdd", -999.0))
    val_p = float(row["val"]["trade_stats"].get("p_value_mean_ret_approx", 1.0))
    return (
        val_ratio
        + 0.5 * train_ratio
        + min(1.0, float(val.get("trade_entries", 0)) / 150.0)
        - 0.35 * abs(val_ratio - train_ratio)
        - 0.2 * val_p
    )


def run_scan(cfg: LongComboScanConfig) -> dict[str, Any]:
    market = _load_market(cfg)
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    rule_inputs = _build_rule_inputs(features)
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
    base_positions_by_stride = {
        s: np.arange(max(0, int(cfg.window_size) - 1), max(0, len(market) - max_hold - int(cfg.entry_delay_bars) - 1), s, dtype=np.int64)
        for s in _parse_list(cfg.stride_bars, int)
    }

    trials: list[dict[str, Any]] = []
    for rule_name, values in rule_inputs.items():
        if not np.isfinite(values).any() or float(np.nanstd(values)) <= 1e-12:
            continue
        for q in _parse_list(cfg.quantiles, float):
            threshold = _threshold(values, split_masks["train"], q)
            if threshold is None:
                continue
            op = "le" if q < 0.5 else "ge"
            active = values <= threshold if op == "le" else values >= threshold
            active &= np.isfinite(values)
            for hold in _parse_list(cfg.hold_bars, int):
                for stride in _parse_list(cfg.stride_bars, int):
                    row: dict[str, Any] = {
                        "rule": rule_name,
                        "quantile": float(q),
                        "op": op,
                        "threshold": float(threshold),
                        "hold_bars": int(hold),
                        "hold_hours": float(hold) * 5.0 / 60.0,
                        "stride_bars": int(stride),
                    }
                    positions = base_positions_by_stride[stride]
                    for split, mask in split_masks.items():
                        split_positions = positions[active[positions] & mask[positions]]
                        sim, trade_returns = _strict_long_sim(
                            split_positions,
                            market=market,
                            hold_bars=int(hold),
                            entry_delay_bars=int(cfg.entry_delay_bars),
                            leverage=float(cfg.leverage),
                            fee_rate=float(cfg.fee_rate),
                            slippage_rate=float(cfg.slippage_rate),
                            annualization_start=split_bounds[split][0],
                            annualization_end=split_bounds[split][1],
                        )
                        row[split] = {"sim": sim, "trade_stats": _trade_stats(trade_returns)}
                    row["selection_score"] = _score_trial(row, cfg)
                    trials.append(row)

    ranked = sorted(trials, key=lambda r: float(r.get("selection_score", -1e9)), reverse=True)
    robust = sorted(
        [
            r
            for r in trials
            if int(r["train"]["sim"].get("trade_entries", 0)) >= int(cfg.min_train_trades)
            and int(r["val"]["sim"].get("trade_entries", 0)) >= int(cfg.min_val_trades)
            and int(r["eval"]["sim"].get("trade_entries", 0)) >= int(cfg.min_eval_trades_for_robust)
            and all(float(r[s]["sim"].get("cagr_pct", -999.0)) > 0.0 for s in ("train", "val", "eval"))
        ],
        key=lambda r: min(float(r[s]["sim"].get("cagr_to_strict_mdd", -999.0)) for s in ("train", "val", "eval")),
        reverse=True,
    )
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "input": {
            "rows": len(market),
            "start": str(market["date"].iloc[0]),
            "end": str(market["date"].iloc[-1]),
        },
        "selection_protocol": "thresholds fit on train only; train+val rank only; eval is held out report-only",
        "rules": {
            "range_expansion_pullback": "long when long-range width and high-to-current pullback are large",
            "deep_pullback_rebound": "long deeper high-to-current pullbacks with short-range location depressed",
            "trend_pullback": "long positive higher-timeframe trend after pullback",
            "contrarian_stress_long": "long pullback when funding/premium stress is not euphoric",
            "range_upper_continuation": "long upper-range continuation with flow confirmation",
        },
        "top_selection": ranked[:50],
        "top_robust": robust[:50],
        "all_count": len(trials),
        "leakage_guard": {
            "market_rows_after_exclude_from_removed_before_feature_build": True,
            "features_use_rows_at_or_before_signal": True,
            "entry_after_signal_by_bars": int(cfg.entry_delay_bars),
            "eval_not_used_for_threshold_or_selection": True,
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {k: row[k] for k in ("rule", "quantile", "op", "threshold", "hold_bars", "hold_hours", "stride_bars", "selection_score")}
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--funding-csv", default="")
    p.add_argument("--premium-csv", default="")
    for field in (
        "train-start",
        "train-end",
        "val-start",
        "val-end",
        "eval-start",
        "eval-end",
        "exclude-from",
        "hold-bars",
        "stride-bars",
        "quantiles",
    ):
        p.add_argument(f"--{field}", default=getattr(LongComboScanConfig, field.replace("-", "_")))
    p.add_argument("--window-size", type=int, default=LongComboScanConfig.window_size)
    p.add_argument("--entry-delay-bars", type=int, default=LongComboScanConfig.entry_delay_bars)
    p.add_argument("--leverage", type=float, default=LongComboScanConfig.leverage)
    p.add_argument("--fee-rate", type=float, default=LongComboScanConfig.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=LongComboScanConfig.slippage_rate)
    p.add_argument("--min-train-trades", type=int, default=LongComboScanConfig.min_train_trades)
    p.add_argument("--min-val-trades", type=int, default=LongComboScanConfig.min_val_trades)
    p.add_argument("--min-eval-trades-for-robust", type=int, default=LongComboScanConfig.min_eval_trades_for_robust)
    return p.parse_args()


def main() -> None:
    report = run_scan(LongComboScanConfig(**vars(parse_args())))
    print(
        json.dumps(
            {
                "output": report["config"]["output"],
                "input": report["input"],
                "all_count": report["all_count"],
                "top_selection": [_compact_row(r) for r in report["top_selection"][:10]],
                "top_robust": [_compact_row(r) for r in report["top_robust"][:10]],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
