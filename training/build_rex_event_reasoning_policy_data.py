"""Build symbolic REX event reasoning policy rows for a single LLM.

This dataset is meant to exploit LLM strengths: categorical price-action/regime
reasoning instead of raw numeric regression.  Prompts contain only signal-time
bucketed facts.  Targets use future executable path utility for offline SFT only.
"""
from __future__ import annotations

import argparse, json, math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.event_candidate_pool_probe import EventPoolConfig, _candidate_rows_for_family, _load_market, _simulate_rows, _split_mask
from training.path_outcome_dataset import PathOutcomeConfig, compute_trade_path_outcome


@dataclass(frozen=True)
class RexEventReasoningCfg:
    input_csv: str
    output_jsonl: str
    summary_output: str = ""
    family: str = "rex_htf_pullback_reclaim"
    hold_bars: int = 144
    quantile: float = 0.75
    stride_bars: int = 24
    threshold_train_start: str = "2020-01-01"
    threshold_train_end: str = "2025-01-01"
    dataset_start: str = "2020-01-01"
    dataset_end: str = "2026-06-01"
    train_end: str = "2025-01-01"
    test_end: str = "2026-01-01"
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    mae_penalty: float = 1.0
    no_trade_utility: float = 0.001
    min_trade_net_return: float = 0.001
    max_trade_mae: float = 0.035
    skip_oracle: bool = False


def _clean(x: pd.Series, clip: float | None = None) -> pd.Series:
    out = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if clip is not None:
        out = out.clip(-float(clip), float(clip))
    return out


def _rolling_z(x: pd.Series, window: int) -> pd.Series:
    mean = x.rolling(window, min_periods=max(5, window // 3)).mean()
    std = x.rolling(window, min_periods=max(5, window // 3)).std(ddof=0)
    return _clean((x - mean) / std.replace(0.0, np.nan), clip=5.0)


def _rsi_norm(close: pd.Series, length: int = 14) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0.0).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    loss = (-diff.clip(upper=0.0)).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = gain / loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return _clean((rsi - 50.0) / 50.0)


def _build_light_rex_features(market: pd.DataFrame) -> pd.DataFrame:
    close = market["close"].astype(float)
    high = market["high"].astype(float)
    low = market["low"].astype(float)
    volume = market["volume"].astype(float)
    out: dict[str, pd.Series] = {}
    roll_high = high.rolling(144, min_periods=144).max()
    roll_low = low.rolling(144, min_periods=144).min()
    span = (roll_high - roll_low).replace(0.0, np.nan)
    mid = (roll_high + roll_low) / 2.0
    out["range_vol"] = _clean((roll_high - roll_low) / mid.replace(0.0, np.nan))
    out["range_pos"] = _clean(((close - roll_low) / span) * 2.0 - 1.0)
    out["trend_24"] = _clean(close / close.shift(23).replace(0.0, np.nan) - 1.0)
    out["trend_96"] = _clean(close / close.shift(143).replace(0.0, np.nan) - 1.0)
    logret = np.log(close / close.shift(1).replace(0.0, np.nan))
    out["return_zscore_48"] = _rolling_z(logret, 48)
    out["rsi_norm"] = _rsi_norm(close)
    bb_mean = close.rolling(20, min_periods=1).mean()
    bb_std = close.rolling(20, min_periods=1).std(ddof=0)
    out["bb_z"] = _clean((close - bb_mean) / bb_std.replace(0.0, np.nan), clip=5.0)
    vol_mean = volume.rolling(48, min_periods=16).mean()
    vol_std = volume.rolling(48, min_periods=16).std(ddof=0)
    out["volume_zscore"] = _clean((volume - vol_mean) / vol_std.replace(0.0, np.nan), clip=5.0)
    if "taker_buy_base" in market.columns:
        out["taker_imbalance"] = _clean((market["taker_buy_base"].astype(float) / volume.replace(0.0, np.nan)).fillna(0.5) * 2.0 - 1.0)
    else:
        out["taker_imbalance"] = pd.Series(0.0, index=market.index)
    peak = close.rolling(144, min_periods=1).max()
    out["window_drawdown"] = _clean(1.0 - close / peak.replace(0.0, np.nan))
    # Fast completed-HTF approximations using past bar returns.  These are causal and sufficient for symbolic REX context.
    out["htf_4h_return_1"] = _clean(close / close.shift(48).replace(0.0, np.nan) - 1.0)
    out["htf_4h_return_4"] = _clean(close / close.shift(192).replace(0.0, np.nan) - 1.0)
    out["htf_1d_return_1"] = _clean(close / close.shift(288).replace(0.0, np.nan) - 1.0)
    out["htf_1d_return_4"] = _clean(close / close.shift(1152).replace(0.0, np.nan) - 1.0)
    out["htf_3d_return_4"] = _clean(close / close.shift(3456).replace(0.0, np.nan) - 1.0)
    out["htf_1w_return_4"] = _clean(close / close.shift(8064).replace(0.0, np.nan) - 1.0)
    for col in ("dxy_zscore", "dxy_momentum", "usdkrw_zscore", "usdkrw_momentum", "kimchi_premium_zscore", "kimchi_premium_change"):
        out[col] = _clean(market[col], clip=5.0) if col in market.columns else pd.Series(0.0, index=market.index)
    if "open_interest" in market.columns:
        oi = market["open_interest"].astype(float)
        out["oi_zscore"] = _rolling_z(oi, 48)
    else:
        out["oi_zscore"] = pd.Series(0.0, index=market.index)
    for w in (144, 576, 2016, 8640):
        minp = min(w, max(12, w // 4))
        rh = high.rolling(w, min_periods=minp).max()
        rl = low.rolling(w, min_periods=minp).min()
        rs = (rh - rl).replace(0.0, np.nan)
        prefix = f"rex_{w}"
        out[f"{prefix}_range_width_pct"] = _clean(rs / close.replace(0.0, np.nan))
        out[f"{prefix}_range_pos"] = _clean(((close - rl) / rs) * 2.0 - 1.0)
        out[f"{prefix}_max_to_cur_pct"] = _clean(rh / close.replace(0.0, np.nan) - 1.0)
        out[f"{prefix}_cur_to_min_pct"] = _clean(close / rl.replace(0.0, np.nan) - 1.0)
    return pd.DataFrame(out, index=market.index).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _rex_pullback_reclaim_arrays(feat: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    def a(k: str) -> np.ndarray:
        return feat[k].to_numpy(float) if k in feat.columns else np.zeros(len(feat), dtype=float)
    rex_windows = (144, 576, 2016, 8640)
    rex_pos_stack = np.vstack([a(f"rex_{w}_range_pos") for w in rex_windows])
    rex_max_gap_stack = np.vstack([a(f"rex_{w}_max_to_cur_pct") for w in rex_windows])
    rex_min_gap_stack = np.vstack([a(f"rex_{w}_cur_to_min_pct") for w in rex_windows])
    rex_loc = np.nanmean(rex_pos_stack, axis=0)
    rex_max_gap = np.nanmean(rex_max_gap_stack, axis=0)
    rex_min_gap = np.nanmean(rex_min_gap_stack, axis=0)
    local_trend = a("trend_24") + 0.5 * a("htf_4h_return_1")
    higher_trend = a("htf_1d_return_4") + a("htf_3d_return_4") + a("htf_1w_return_4")
    vol_confirm = np.maximum(0.0, a("volume_zscore")) + 0.5 * np.abs(a("taker_imbalance"))
    htf_dir = np.sign(higher_trend)
    pullback_alignment = -np.sign(rex_loc) * htf_dir
    rex_pullback = np.maximum(0.0, pullback_alignment) * (np.abs(higher_trend) + 0.25 * np.abs(rex_max_gap - rex_min_gap))
    local_reclaim = np.maximum(0.0, np.sign(local_trend) * htf_dir)
    strength = rex_pullback * (0.5 + local_reclaim) * (1.0 + 0.25 * vol_confirm)
    direction = htf_dir
    return np.nan_to_num(strength, nan=0.0, posinf=0.0, neginf=0.0), np.nan_to_num(direction, nan=0.0, posinf=0.0, neginf=0.0)


def _bucket(v: float, cuts: list[float], labels: list[str]) -> str:
    if not math.isfinite(float(v)):
        return "missing"
    for c, lab in zip(cuts, labels):
        if float(v) <= float(c):
            return lab
    return labels[-1]


def _sign_bucket(v: float, small: float = 1e-9) -> str:
    if not math.isfinite(float(v)):
        return "missing"
    if v > small:
        return "positive"
    if v < -small:
        return "negative"
    return "flat"


def _state_tokens(feat: pd.DataFrame, pos: int, base_side: str, strength: float, threshold: float) -> dict[str, str]:
    row = feat.iloc[int(pos)]
    def f(k: str) -> float:
        try:
            return float(row.get(k, 0.0) or 0.0)
        except Exception:
            return 0.0
    toks: dict[str, str] = {
        "base_event_side": base_side,
        "event_strength_vs_threshold": _bucket(strength / max(abs(threshold), 1e-12), [1.05, 1.25, 1.75], ["barely_active", "active", "strong", "extreme"]),
        "local_trend_24": _sign_bucket(f("trend_24"), 0.001),
        "swing_trend_96": _sign_bucket(f("trend_96"), 0.003),
        "ret48_state": _bucket(f("return_zscore_48"), [-1.0, -0.4, 0.4, 1.0], ["selloff_extreme", "pullback", "neutral", "rally", "rally_extreme"]),
        "rsi_state": _bucket(f("rsi_norm"), [-0.45, -0.20, 0.20, 0.45], ["oversold", "soft_oversold", "neutral", "soft_overbought", "overbought"]),
        "bb_state": _bucket(f("bb_z"), [-1.5, -0.5, 0.5, 1.5], ["below_band", "lower_half", "middle", "upper_half", "above_band"]),
        "short_range_location": _bucket(f("rex_144_range_pos"), [-0.70, -0.30, 0.30, 0.70], ["near_short_low", "lower_short_range", "mid_short_range", "upper_short_range", "near_short_high"]),
        "mid_range_location": _bucket(f("rex_576_range_pos"), [-0.70, -0.30, 0.30, 0.70], ["near_mid_low", "lower_mid_range", "mid_range", "upper_mid_range", "near_mid_high"]),
        "long_range_location": _bucket(f("rex_2016_range_pos"), [-0.70, -0.30, 0.30, 0.70], ["near_long_low", "lower_long_range", "mid_long_range", "upper_long_range", "near_long_high"]),
        "weekly_range_location": _bucket(f("rex_8640_range_pos"), [-0.70, -0.30, 0.30, 0.70], ["near_week_low", "lower_week_range", "mid_week_range", "upper_week_range", "near_week_high"]),
        "range_vol_state": _bucket(f("range_vol"), [0.01, 0.02, 0.035], ["compressed", "normal", "expanded", "stress_expanded"]),
        "drawdown_state": _bucket(f("window_drawdown"), [0.02, 0.05, 0.10], ["shallow", "moderate", "deep", "crash_like"]),
        "taker_flow": _bucket(f("taker_imbalance"), [-0.20, -0.05, 0.05, 0.20], ["strong_sell_flow", "sell_flow", "balanced", "buy_flow", "strong_buy_flow"]),
        "volume_state": _bucket(f("volume_zscore"), [-0.5, 0.5, 1.5], ["quiet", "normal", "active", "surge"]),
        "htf_4h": _sign_bucket(f("htf_4h_return_4"), 0.005),
        "htf_1d": _sign_bucket(f("htf_1d_return_4"), 0.015),
        "htf_1w": _sign_bucket(f("htf_1w_return_4"), 0.03),
        "usdkrw_stress": _bucket(f("usdkrw_zscore"), [-1.0, 0.5, 1.5], ["low", "normal", "elevated", "extreme"]),
        "dxy_pressure": _bucket(f("dxy_zscore"), [-1.0, 0.5, 1.5], ["weak_dollar", "neutral", "strong_dollar", "extreme_dollar"]),
        "kimchi_state": _bucket(f("kimchi_premium_zscore"), [-1.0, 0.5, 1.5], ["discount", "normal", "premium", "extreme_premium"]),
        "oi_state": _bucket(f("oi_zscore"), [-1.0, 0.5, 1.5], ["low_oi", "normal_oi", "high_oi", "crowded_oi"]),
    }
    # Deductive hints from the side-controller scan; these are causal facts, not decisions.
    toks["long_context_hint"] = "deep_pullback_ok" if f("rsi_norm") <= -0.3052479582547623 else "long_not_deep_pullback"
    toks["short_context_hint"] = "range_stress_ok" if f("range_vol") >= 0.0215245646710851 else "short_range_not_stressed"
    toks["macro_context_hint"] = "fx_stress_low_enough" if f("usdkrw_zscore") <= 0.4804156485137074 else "fx_stress_high"
    return toks


def _prompt(date: str, tokens: dict[str, str], cfg: RexEventReasoningCfg) -> str:
    lines = [
        "You are a single Gemma REX event policy for BTCUSDT perpetual futures.",
        "Use symbolic past-only facts. Do not do arithmetic on raw prices.",
        "Task: choose one action for the current REX pullback/reclaim event.",
        "Allowed actions: LONG, SHORT, NO_TRADE.",
        "Prefer NO_TRADE when event direction conflicts with side context or risk is unclear.",
        "Return compact JSON: {\"action\":...,\"hold_bars\":...,\"confidence\":...,\"reason\":...}.",
        f"date: {date}",
        f"fixed_hold_bars: {int(cfg.hold_bars)}",
        "past_only_symbolic_context:",
    ]
    for k in sorted(tokens):
        lines.append(f"- {k}: {tokens[k]}")
    return "\n".join(lines)


def _precompute_path_arrays(market: pd.DataFrame, cfg: RexEventReasoningCfg) -> dict[str, np.ndarray]:
    n = len(market)
    hold = int(cfg.hold_bars)
    delay = int(cfg.entry_delay_bars)
    opens = market["open"].to_numpy(float)
    highs = market["high"].to_numpy(float)
    lows = market["low"].to_numpy(float)
    entry = np.full(n, np.nan); exitp = np.full(n, np.nan)
    maxh = np.full(n, np.nan); minl = np.full(n, np.nan)
    last = n - delay - hold - 1
    for pos in range(0, max(0, last) + 1):
        ep = pos + delay; xp = ep + hold
        entry[pos] = opens[ep]; exitp[pos] = opens[xp]
        maxh[pos] = np.max(highs[ep:xp]); minl[pos] = np.min(lows[ep:xp])
    return {"entry": entry, "exit": exitp, "maxh": maxh, "minl": minl}


def _target_fast(path: dict[str, np.ndarray], signal_pos: int, cfg: RexEventReasoningCfg) -> tuple[dict[str, Any], dict[str, Any]] | None:
    pos = int(signal_pos)
    e = float(path["entry"][pos]); x = float(path["exit"][pos]); h = float(path["maxh"][pos]); l = float(path["minl"][pos])
    if not all(math.isfinite(v) and v > 0.0 for v in (e, x, h, l)):
        return None
    lev = float(cfg.leverage); cost = 2.0 * (float(cfg.fee_rate) + float(cfg.slippage_rate)) * lev
    long_net = lev * ((x - e) / e) - cost
    long_mae = max(0.0, (e - l) / e); long_mfe = max(0.0, (h - e) / e); long_util = long_net - lev * float(cfg.mae_penalty) * long_mae
    short_net = lev * ((e - x) / e) - cost
    short_mae = max(0.0, (h - e) / e); short_mfe = max(0.0, (e - l) / e); short_util = short_net - lev * float(cfg.mae_penalty) * short_mae
    if long_util >= short_util:
        side, net, mae, mfe, util = "LONG", long_net, long_mae, long_mfe, long_util
    else:
        side, net, mae, mfe, util = "SHORT", short_net, short_mae, short_mfe, short_util
    action = side
    if util <= cfg.no_trade_utility or net <= cfg.min_trade_net_return or mae > cfg.max_trade_mae:
        action = "NO_TRADE"
    conf = "HIGH" if util >= 0.01 and mae <= 0.015 else ("MID" if util >= 0.004 else "LOW")
    target = {"action": action, "hold_bars": int(cfg.hold_bars) if action != "NO_TRADE" else 0, "confidence": conf, "reason": "offline_path_utility_label"}
    audit = {
        "best_side": side, "best_utility": util, "best_net_return": net, "best_mae": mae, "best_mfe": mfe,
        "long": {"side":"LONG","net_return":long_net,"mae":long_mae,"mfe":long_mfe,"utility":long_util},
        "short": {"side":"SHORT","net_return":short_net,"mae":short_mae,"mfe":short_mfe,"utility":short_util},
    }
    return target, audit


def _target(market: pd.DataFrame, signal_pos: int, cfg: RexEventReasoningCfg) -> tuple[dict[str, Any], dict[str, Any]] | None:
    pcfg = PathOutcomeConfig(hold_bars=cfg.hold_bars, entry_delay_bars=cfg.entry_delay_bars, fee_rate=cfg.fee_rate, slippage_rate=cfg.slippage_rate, leverage=cfg.leverage, mae_penalty=cfg.mae_penalty)
    long = compute_trade_path_outcome(market, signal_pos, "LONG", pcfg)
    short = compute_trade_path_outcome(market, signal_pos, "SHORT", pcfg)
    if long is None or short is None:
        return None
    best = long if long.utility >= short.utility else short
    action = best.side
    if best.utility <= cfg.no_trade_utility or best.net_return <= cfg.min_trade_net_return or best.mae > cfg.max_trade_mae:
        action = "NO_TRADE"
    conf = "HIGH" if best.utility >= 0.01 and best.mae <= 0.015 else ("MID" if best.utility >= 0.004 else "LOW")
    target = {"action": action, "hold_bars": int(cfg.hold_bars) if action != "NO_TRADE" else 0, "confidence": conf, "reason": "offline_path_utility_label"}
    audit = {
        "best_side": best.side,
        "best_utility": best.utility,
        "best_net_return": best.net_return,
        "best_mae": best.mae,
        "best_mfe": best.mfe,
        "long": asdict(long),
        "short": asdict(short),
    }
    return target, audit


def _split(date: str, cfg: RexEventReasoningCfg) -> str:
    ts = pd.Timestamp(date)
    if ts < pd.Timestamp(cfg.train_end):
        return "train"
    if ts < pd.Timestamp(cfg.test_end):
        return "test"
    return "eval"


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _oracle_stats(rows: list[dict[str, Any]], market: pd.DataFrame, cfg: RexEventReasoningCfg) -> dict[str, Any]:
    out = {}
    ecfg = EventPoolConfig(input_csv=cfg.input_csv, output="", hold_bars=cfg.hold_bars, entry_delay_bars=cfg.entry_delay_bars, stride_bars=cfg.stride_bars, leverage=cfg.leverage, fee_rate=cfg.fee_rate, slippage_rate=cfg.slippage_rate)
    for split in ("train", "test", "eval"):
        trade_rows=[]
        for r in rows:
            if r["split"] != split:
                continue
            tgt = json.loads(r["target"])
            if tgt["action"] in {"LONG", "SHORT"}:
                trade_rows.append({"date": r["date"], "signal_date": r["date"], "entry_date": r["date"], "exit_date": r["date"], "side": tgt["action"], "family": "rex_event_reasoning_oracle", "strength": 1.0, "score_mean": 1.0})
        sim = _simulate_rows(trade_rows, market, ecfg)
        out[split] = {"rows": len([r for r in rows if r["split"]==split]), "oracle_trade_rows": len(trade_rows), "sim": sim.get("sim", {}), "trade_stats": sim.get("trade_stats", {})}
    return out


def build(cfg: RexEventReasoningCfg) -> dict[str, Any]:
    market = _load_market(cfg.input_csv)
    feat = _build_light_rex_features(market)
    dates = pd.to_datetime(market["date"])
    if cfg.family != "rex_htf_pullback_reclaim":
        raise ValueError("fast builder currently supports rex_htf_pullback_reclaim only")
    strength, direction = _rex_pullback_reclaim_arrays(feat)
    train_mask = _split_mask(dates, cfg.threshold_train_start, cfg.threshold_train_end)
    x = strength[train_mask & np.isfinite(strength) & (strength > 0.0)]
    threshold = float(np.quantile(x, float(cfg.quantile)))
    data_mask = _split_mask(dates, cfg.dataset_start, cfg.dataset_end)
    ecfg = EventPoolConfig(input_csv=cfg.input_csv, output="", hold_bars=cfg.hold_bars, entry_delay_bars=cfg.entry_delay_bars, stride_bars=cfg.stride_bars, quantile=cfg.quantile)
    events = _candidate_rows_for_family(market, strength, direction, family=cfg.family, threshold=threshold, mask=data_mask, cfg=ecfg)
    date_to_pos = {str(x): i for i, x in enumerate(market["date"].tolist())}
    path = _precompute_path_arrays(market, cfg)
    rows: list[dict[str, Any]] = []
    for ev in events:
        date = str(ev["signal_date"])
        pos = date_to_pos.get(date)
        if pos is None:
            continue
        t = _target_fast(path, pos, cfg)
        if t is None:
            continue
        target, audit = t
        toks = _state_tokens(feat, pos, str(ev["side"]), float(ev["strength"]), threshold)
        rows.append({
            "task": "rex_event_reasoning_policy_sft",
            "date": date,
            "signal_pos": int(pos),
            "split": _split(date, cfg),
            "prompt": _prompt(date, toks, cfg),
            "target": json.dumps(target, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            "state_tokens": toks,
            "target_action_audit": audit,
            "base_event": {"family": cfg.family, "base_side": ev["side"], "strength": ev["strength"], "threshold": threshold},
            "leakage_guard": {"prompt_uses_future_path": False, "state_tokens_are_signal_time_only": True, "target_uses_future_path_for_offline_training_only": True},
        })
    _write_jsonl(cfg.output_jsonl, rows)
    counts: dict[str, Any] = {}
    for split in ("train", "test", "eval"):
        part=[r for r in rows if r["split"]==split]
        acts=Counter(json.loads(r["target"])["action"] for r in part)
        base=Counter(r["base_event"]["base_side"] for r in part)
        counts[split]={"rows": len(part), "target_actions": dict(acts), "base_sides": dict(base), "prompt_chars_mean": sum(len(r["prompt"]) for r in part)/max(1,len(part))}
    report={
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "threshold": threshold,
        "rows": len(rows),
        "split_summary": counts,
        "oracle_upper_bound": ({} if cfg.skip_oracle else _oracle_stats(rows, market, cfg)),
        "leakage_guard": {"threshold_fit_on_train_only": True, "prompts_are_symbolic_signal_time_only": True, "oracle_upper_bound_is_not_deployable": True},
    }
    out = Path(cfg.summary_output or f"{cfg.output_jsonl}.summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description="Build symbolic REX event reasoning policy SFT rows")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--family", default=RexEventReasoningCfg.family)
    p.add_argument("--hold-bars", type=int, default=RexEventReasoningCfg.hold_bars)
    p.add_argument("--quantile", type=float, default=RexEventReasoningCfg.quantile)
    p.add_argument("--stride-bars", type=int, default=RexEventReasoningCfg.stride_bars)
    p.add_argument("--threshold-train-start", default=RexEventReasoningCfg.threshold_train_start)
    p.add_argument("--threshold-train-end", default=RexEventReasoningCfg.threshold_train_end)
    p.add_argument("--dataset-start", default=RexEventReasoningCfg.dataset_start)
    p.add_argument("--dataset-end", default=RexEventReasoningCfg.dataset_end)
    p.add_argument("--train-end", default=RexEventReasoningCfg.train_end)
    p.add_argument("--test-end", default=RexEventReasoningCfg.test_end)
    p.add_argument("--no-trade-utility", type=float, default=RexEventReasoningCfg.no_trade_utility)
    p.add_argument("--min-trade-net-return", type=float, default=RexEventReasoningCfg.min_trade_net_return)
    p.add_argument("--max-trade-mae", type=float, default=RexEventReasoningCfg.max_trade_mae)
    p.add_argument("--skip-oracle", action="store_true")
    return p.parse_args()


def main() -> None:
    print(json.dumps(build(RexEventReasoningCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
