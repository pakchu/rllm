"""Probe a wider event/candidate pool before more LLM/RL training.

The Kimchi-flow pool is too narrow and regime-fragile.  This script generates
past-only candidate events directly from the extended market feature frame:
momentum, reversal, volatility breakout, path-stress, kimchi, macro, and higher
timeframe families.  It then selects candidate-family thresholds on train,
chooses a family on validation, and reports the fixed choice on eval.
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

from preprocessing.external_features import attach_wave_trading_external_features
from preprocessing.market_features import build_market_feature_frame
from training.eval_pairwise_candidate_backtest import CandidateBacktestConfig, simulate_candidates
from training.strict_bar_backtest import _trade_stats


@dataclass(frozen=True)
class EventPoolConfig:
    input_csv: str
    output: str
    train_start: str = "2020-01-01"
    train_end: str = "2023-01-01"
    val_start: str = "2023-01-01"
    val_end: str = "2024-01-01"
    eval_start: str = "2024-01-01"
    eval_end: str = "2025-01-01"
    hold_bars: int = 288
    entry_delay_bars: int = 1
    window_size: int = 144
    stride_bars: int = 12
    quantile: float = 0.8
    min_train_trades: int = 50
    min_val_trades: int = 50
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    wave_trading_root: str = ""
    external_tolerance: str = "30min"


def _load_market(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], compression="infer")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="raise").dt.tz_convert(None)
    return df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _split_mask(dates: pd.Series, start: str, end: str) -> np.ndarray:
    return np.asarray((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end)), dtype=bool)


def _feature_candidates(features: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return family -> (strength, direction), direction +1 LONG / -1 SHORT."""
    f = features
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def arr(name: str) -> np.ndarray:
        return f.get(name, pd.Series(0.0, index=f.index)).to_numpy(dtype=float)

    trend = 0.40 * arr("trend_96") + 0.25 * arr("trend_24") + 0.20 * arr("htf_4h_return_4") + 0.15 * arr("htf_1d_return_1")
    out["momentum_trend"] = (np.abs(trend), np.sign(trend))

    stretch = 0.45 * arr("range_pos") + 0.25 * arr("bb_z") + 0.20 * arr("rsi_norm") + 0.10 * arr("htf_4h_range_pos")
    out["mean_reversion_stretch"] = (np.abs(stretch), -np.sign(stretch))

    breakout = np.maximum(0.0, arr("range_vol")) * (np.abs(arr("trend_24")) + np.abs(arr("htf_4h_return_1")))
    breakout_dir = np.sign(arr("trend_24") + arr("htf_4h_return_1"))
    out["vol_breakout"] = (breakout, breakout_dir)

    stress = arr("window_drawdown") + arr("htf_4h_drawdown_4") + arr("htf_1d_drawdown_4")
    stress_dir = np.where(arr("trend_24") >= 0.0, 1.0, -1.0)
    out["drawdown_continuation"] = (stress, stress_dir)
    out["drawdown_reversal"] = (stress, -stress_dir)

    flow = arr("taker_imbalance") + 0.15 * arr("volume_zscore")
    out["orderflow_follow"] = (np.abs(flow), np.sign(flow))
    out["orderflow_fade"] = (np.abs(flow), -np.sign(flow))

    kimchi = arr("kimchi_premium_zscore") + 3.0 * arr("kimchi_premium_change")
    out["kimchi_extreme_fade"] = (np.abs(kimchi), -np.sign(kimchi))
    out["kimchi_flow_follow"] = (np.abs(arr("kimchi_premium_change")), np.sign(arr("kimchi_premium_change")))

    macro = arr("dxy_zscore") + arr("usdkrw_zscore") + 2.0 * arr("dxy_momentum")
    # Risk-off macro pressure often hurts BTC; positive pressure => SHORT.
    out["macro_pressure"] = (np.abs(macro), -np.sign(macro))

    htf = arr("htf_1d_return_4") + arr("htf_3d_return_4") + arr("htf_1w_return_4")
    out["higher_tf_momentum"] = (np.abs(htf), np.sign(htf))
    out["higher_tf_fade"] = (np.abs(htf), -np.sign(htf))

    candle_shock = arr("candle_range") * (1.0 + np.maximum(0.0, arr("volume_zscore")))
    candle_dir = np.sign(arr("body_ratio"))
    out["candle_shock_follow"] = (candle_shock, candle_dir)
    out["candle_shock_fade"] = (candle_shock, -candle_dir)

    # New pool-expansion families for the symbolic ridge stage.  These are all
    # computed from completed/current-bar history-only features and expose more
    # diverse hypotheses than the original momentum/reversion families.
    compression = np.maximum(0.0, 0.035 - arr("range_vol")) * (np.abs(arr("return_zscore_48")) + np.abs(arr("trend_12")) * 20.0)
    compression_dir = np.sign(arr("trend_12") + 0.5 * arr("body_ratio"))
    out["vol_compression_breakout"] = (compression, compression_dir)
    out["vol_compression_fakeout"] = (compression, -compression_dir)

    micro_reversal = np.abs(arr("return_zscore_48")) + np.maximum(0.0, np.abs(arr("range_pos")) - 0.55) + np.maximum(0.0, np.abs(arr("rsi_norm")) - 0.35)
    micro_reversal_dir = -np.sign(arr("return_zscore_48") + arr("range_pos") + arr("rsi_norm"))
    out["micro_exhaustion_reversal"] = (micro_reversal, micro_reversal_dir)

    htf_pullback = np.abs(arr("htf_1w_return_4")) + np.abs(arr("htf_1d_return_4")) + np.maximum(0.0, -arr("htf_4h_return_1") * np.sign(arr("htf_1d_return_4")))
    htf_pullback_dir = np.sign(arr("htf_1d_return_4") + arr("htf_1w_return_4"))
    out["htf_pullback_resume"] = (htf_pullback, htf_pullback_dir)

    htf_break = arr("htf_1d_drawdown_4") + arr("htf_3d_drawdown_4") + np.maximum(0.0, -arr("trend_96") * np.sign(arr("htf_1d_return_4")))
    out["htf_structure_break"] = (htf_break, -np.sign(arr("htf_1d_return_4") + arr("htf_1w_return_4")))

    macro_kimchi_div = np.abs(arr("dxy_zscore") - arr("kimchi_premium_zscore")) + np.abs(arr("usdkrw_momentum"))
    macro_kimchi_dir = -np.sign(arr("dxy_zscore") + arr("usdkrw_zscore") - arr("kimchi_premium_zscore"))
    out["macro_kimchi_divergence"] = (macro_kimchi_div, macro_kimchi_dir)

    funding_stress = np.abs(arr("funding_zscore")) + np.abs(arr("oi_zscore")) + np.abs(arr("oi_change"))
    funding_dir = -np.sign(arr("funding_zscore") + 0.5 * arr("oi_change"))
    out["derivatives_stress_fade"] = (funding_stress, funding_dir)

    return out


def _candidate_rows_for_family(
    market: pd.DataFrame,
    strength: np.ndarray,
    direction: np.ndarray,
    *,
    family: str,
    threshold: float,
    mask: np.ndarray,
    cfg: EventPoolConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    last_pos = len(market) - int(cfg.entry_delay_bars) - int(cfg.hold_bars) - 1
    for pos in range(max(0, int(cfg.window_size) - 1), max(0, last_pos) + 1, max(1, int(cfg.stride_bars))):
        if not mask[pos]:
            continue
        if not np.isfinite(strength[pos]) or float(strength[pos]) < float(threshold):
            continue
        side_dir = float(direction[pos])
        if side_dir == 0.0 or not np.isfinite(side_dir):
            continue
        entry_pos = pos + int(cfg.entry_delay_bars)
        exit_pos = entry_pos + int(cfg.hold_bars)
        side = "LONG" if side_dir > 0 else "SHORT"
        rows.append(
            {
                "date": str(market.iloc[pos]["date"]),
                "signal_date": str(market.iloc[pos]["date"]),
                "entry_date": str(market.iloc[entry_pos]["date"]),
                "exit_date": str(market.iloc[exit_pos]["date"]),
                "side": side,
                "family": family,
                "strength": float(strength[pos]),
                "score_mean": 1.0,
            }
        )
    return rows


def _simulate_rows(rows: list[dict[str, Any]], market: pd.DataFrame, cfg: EventPoolConfig) -> dict[str, Any]:
    sim_cfg = CandidateBacktestConfig(
        market_csv=cfg.input_csv,
        pairwise_jsonl="",
        predictions_jsonl="",
        output="",
        score_threshold=0.0,
        hold_bars=int(cfg.hold_bars),
        entry_delay_bars=int(cfg.entry_delay_bars),
        leverage=float(cfg.leverage),
        fee_rate=float(cfg.fee_rate),
        slippage_rate=float(cfg.slippage_rate),
    )
    return simulate_candidates(rows, market[["date", "open", "high", "low", "close"]].copy(), sim_cfg)


def _choose_family(train_trials: list[dict[str, Any]], val_trials: list[dict[str, Any]], cfg: EventPoolConfig) -> dict[str, Any]:
    train_ok = {
        t["family"]
        for t in train_trials
        if int(t["train"]["sim"]["trade_entries"]) >= int(cfg.min_train_trades)
        and float(t["train"]["sim"]["cagr_to_strict_mdd"]) > 0.0
    }
    eligible = [
        v for v in val_trials
        if v["family"] in train_ok
        and int(v["val"]["sim"]["trade_entries"]) >= int(cfg.min_val_trades)
    ]
    pool = eligible or val_trials
    return max(
        pool,
        key=lambda r: (
            float(r["val"]["sim"].get("cagr_to_strict_mdd", -1e9)),
            -float(r["val"]["trade_stats"].get("p_value_mean_ret_approx", 1.0)),
            int(r["val"]["sim"].get("trade_entries", 0)),
        ),
    )


def run_event_pool_probe(cfg: EventPoolConfig) -> dict[str, Any]:
    market = _load_market(cfg.input_csv)
    if cfg.wave_trading_root:
        market = attach_wave_trading_external_features(market, wave_trading_root=cfg.wave_trading_root, tolerance=cfg.external_tolerance)
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    dates = pd.to_datetime(market["date"])
    train_mask = _split_mask(dates, cfg.train_start, cfg.train_end)
    val_mask = _split_mask(dates, cfg.val_start, cfg.val_end)
    eval_mask = _split_mask(dates, cfg.eval_start, cfg.eval_end)
    families = _feature_candidates(features)
    q = float(np.clip(cfg.quantile, 0.5, 0.99))

    train_trials: list[dict[str, Any]] = []
    val_trials: list[dict[str, Any]] = []
    eval_by_family: dict[str, dict[str, Any]] = {}
    for family, (strength, direction) in families.items():
        x = strength[train_mask & np.isfinite(strength)]
        if x.size < 100:
            continue
        threshold = float(np.quantile(x, q))
        train_rows = _candidate_rows_for_family(market, strength, direction, family=family, threshold=threshold, mask=train_mask, cfg=cfg)
        val_rows = _candidate_rows_for_family(market, strength, direction, family=family, threshold=threshold, mask=val_mask, cfg=cfg)
        eval_rows = _candidate_rows_for_family(market, strength, direction, family=family, threshold=threshold, mask=eval_mask, cfg=cfg)
        train_result = _simulate_rows(train_rows, market, cfg)
        val_result = _simulate_rows(val_rows, market, cfg)
        eval_result = _simulate_rows(eval_rows, market, cfg)
        train_trials.append({"family": family, "threshold": threshold, "train": {"sim": train_result["sim"], "trade_stats": train_result["trade_stats"], "candidate_count": len(train_rows)}})
        val_trials.append({"family": family, "threshold": threshold, "val": {"sim": val_result["sim"], "trade_stats": val_result["trade_stats"], "candidate_count": len(val_rows)}})
        eval_by_family[family] = {"threshold": threshold, "eval": {"sim": eval_result["sim"], "trade_stats": eval_result["trade_stats"], "candidate_count": len(eval_rows)}}

    selected = _choose_family(train_trials, val_trials, cfg)
    selected_eval = eval_by_family.get(selected["family"], {})
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "inputs": {"rows": len(market), "start": str(market["date"].iloc[0]), "end": str(market["date"].iloc[-1])},
        "selected_family": selected["family"],
        "selected_validation": selected,
        "selected_eval": selected_eval,
        "top_val": sorted(val_trials, key=lambda r: float(r["val"]["sim"].get("cagr_to_strict_mdd", -1e9)), reverse=True)[:20],
        "train_trials": train_trials,
        "eval_by_family": eval_by_family,
        "leakage_guard": {
            "thresholds_fit_on_train_only": True,
            "family_selection_uses_val_only_after_train_positive_filter": True,
            "eval_not_used_for_selection": True,
            "features_use_rows_at_or_before_signal": True,
            "entry_after_signal_by_bars": int(cfg.entry_delay_bars),
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe wider no-leak event candidate families")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--train-start", default="2020-01-01")
    p.add_argument("--train-end", default="2023-01-01")
    p.add_argument("--val-start", default="2023-01-01")
    p.add_argument("--val-end", default="2024-01-01")
    p.add_argument("--eval-start", default="2024-01-01")
    p.add_argument("--eval-end", default="2025-01-01")
    p.add_argument("--hold-bars", type=int, default=288)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--window-size", type=int, default=144)
    p.add_argument("--stride-bars", type=int, default=12)
    p.add_argument("--quantile", type=float, default=0.8)
    p.add_argument("--min-train-trades", type=int, default=50)
    p.add_argument("--min-val-trades", type=int, default=50)
    p.add_argument("--leverage", type=float, default=0.5)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--external-tolerance", default="30min")
    return p.parse_args()


def main() -> None:
    out = run_event_pool_probe(EventPoolConfig(**vars(parse_args())))
    print(json.dumps({"selected_family": out["selected_family"], "selected_validation": out["selected_validation"], "selected_eval": out["selected_eval"], "top_val": out["top_val"][:10]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
