"""Build categorical text labels for exact action verification.

This is the post-ranker verifier pivot after generic regime-safety labels failed.
Instead of asking whether a signal has *any* safe candidate, each row asks whether
one exact executable action should be allowed.  The prompt avoids raw decimal
feature dumps and expresses the past-only state as categorical/rule text so a
small LLM can use pattern language rather than numeric interpolation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from preprocessing.external_features import attach_wave_trading_external_features
from preprocessing.market_features import build_market_feature_frame
from training.event_action_policy_data import _action_utility, _candidate_book, _date_mask, _load_market, _state_summary, EventActionPolicyConfig
from training.event_candidate_pool_probe import _feature_candidates


@dataclass(frozen=True)
class ActionVerifierTextConfig:
    market_csv: str
    output: str
    summary_output: str = ""
    start_date: str = "2020-01-01"
    end_date: str = "2024-01-01"
    window_size: int = 144
    stride_bars: int = 72
    hold_bars_list: tuple[int, ...] = (72, 144, 288, 432)
    top_k_families: int = 5
    min_rank_utility: float = 0.012
    min_net_return: float = 0.0075
    max_mae: float = 0.014
    min_mfe_to_mae: float = 1.2
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    mae_penalty: float = 1.0
    wave_trading_root: str = ""
    external_tolerance: str = "30min"


def _parse_ints(csv: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(csv).split(",") if x.strip())


def _policy_cfg(cfg: ActionVerifierTextConfig) -> EventActionPolicyConfig:
    return EventActionPolicyConfig(
        market_csv=cfg.market_csv,
        output=cfg.output,
        summary_output=cfg.summary_output,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        window_size=cfg.window_size,
        stride_bars=cfg.stride_bars,
        hold_bars_list=cfg.hold_bars_list,
        top_k_families=cfg.top_k_families,
        entry_delay_bars=cfg.entry_delay_bars,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        mae_penalty=cfg.mae_penalty,
        wave_trading_root=cfg.wave_trading_root,
        external_tolerance=cfg.external_tolerance,
    )


def _bucket(x: float, cuts: tuple[float, ...], labels: tuple[str, ...]) -> str:
    v = float(x)
    for cut, label in zip(cuts, labels):
        if v < cut:
            return label
    return labels[-1]


def _signed_bucket(x: float, scale: float = 1.0) -> str:
    v = float(x) * float(scale)
    if v <= -0.025:
        return "strong_down"
    if v <= -0.0075:
        return "down"
    if v < 0.0075:
        return "flat"
    if v < 0.025:
        return "up"
    return "strong_up"


def _state_text(state: dict[str, Any]) -> list[str]:
    trend24 = _signed_bucket(float(state.get("trend_24", 0.0)))
    trend96 = _signed_bucket(float(state.get("trend_96", 0.0)))
    h4 = _signed_bucket(float(state.get("htf_4h_return_4", 0.0)))
    d1 = _signed_bucket(float(state.get("htf_1d_return_1", 0.0)))
    d3 = _signed_bucket(float(state.get("htf_3d_return_1", 0.0)))
    w4 = _signed_bucket(float(state.get("htf_1w_return_4", 0.0)))
    rsi = _bucket(float(state.get("rsi_norm", 0.0)), (-0.35, -0.12, 0.12, 0.35), ("oversold", "soft_oversold", "neutral", "soft_overbought", "overbought"))
    range_pos = _bucket(float(state.get("range_pos", 0.5)), (0.15, 0.35, 0.65, 0.85), ("near_range_low", "lower_range", "middle_range", "upper_range", "near_range_high"))
    weekly_pos = _bucket(float(state.get("htf_1w_range_pos", 0.0)), (-0.65, -0.25, 0.25, 0.65), ("weekly_near_low", "weekly_lower", "weekly_middle", "weekly_upper", "weekly_near_high"))
    drawdown = _bucket(float(state.get("window_drawdown", 0.0)), (0.005, 0.015, 0.035), ("no_recent_drawdown", "small_drawdown", "medium_drawdown", "large_drawdown"))
    htf_dd = _bucket(float(state.get("htf_1d_drawdown_4", 0.0)), (0.01, 0.03, 0.06), ("daily_dd_low", "daily_dd_medium", "daily_dd_high", "daily_dd_extreme"))
    weekly_dd = _bucket(float(state.get("htf_1w_drawdown_4", 0.0)), (0.03, 0.08, 0.16), ("weekly_dd_low", "weekly_dd_medium", "weekly_dd_high", "weekly_dd_extreme"))
    volume = _bucket(float(state.get("volume_zscore", 0.0)), (-1.0, -0.25, 0.75, 1.5), ("volume_very_low", "volume_low", "volume_normal", "volume_high", "volume_extreme"))
    taker = _bucket(float(state.get("taker_imbalance", 0.0)), (-0.08, -0.02, 0.02, 0.08), ("sell_aggression_strong", "sell_aggression", "flow_balanced", "buy_aggression", "buy_aggression_strong"))
    funding = _bucket(float(state.get("funding_zscore", 0.0)), (-1.0, -0.25, 0.25, 1.0), ("funding_discount", "funding_soft_discount", "funding_neutral", "funding_soft_premium", "funding_premium"))
    oi = _bucket(float(state.get("oi_zscore", 0.0)), (-1.0, -0.25, 0.25, 1.0), ("oi_low", "oi_soft_low", "oi_neutral", "oi_soft_high", "oi_high"))
    oi_change = _bucket(float(state.get("oi_change", 0.0)), (-0.02, -0.005, 0.005, 0.02), ("oi_flush", "oi_falling", "oi_stable", "oi_building", "oi_squeeze"))
    macro = _bucket(float(state.get("dxy_zscore", 0.0)), (-1.0, -0.25, 0.25, 1.0), ("dxy_low", "dxy_soft_low", "dxy_neutral", "dxy_soft_high", "dxy_high"))
    kimchi = _bucket(float(state.get("kimchi_premium_zscore", 0.0)), (-1.0, -0.25, 0.25, 1.0), ("kimchi_discount", "kimchi_soft_discount", "kimchi_neutral", "kimchi_soft_premium", "kimchi_premium"))
    return [
        f"short_trend={trend24}",
        f"medium_trend={trend96}",
        f"four_hour_context={h4}",
        f"daily_context={d1}",
        f"three_day_context={d3}",
        f"weekly_context={w4}",
        f"oscillator={rsi}",
        f"location={range_pos}",
        f"weekly_location={weekly_pos}",
        f"recent_drawdown={drawdown}",
        f"higher_tf_drawdown={htf_dd}",
        f"weekly_drawdown={weekly_dd}",
        f"volume={volume}",
        f"orderflow={taker}",
        f"funding_context={funding}",
        f"open_interest_level={oi}",
        f"open_interest_change={oi_change}",
        f"dollar_pressure={macro}",
        f"kimchi_context={kimchi}",
    ]


def _candidate_text(book: list[dict[str, Any]]) -> list[str]:
    out = []
    for c in book:
        strength = _bucket(float(c.get("strength", 0.0)), (0.05, 0.15, 0.35, 0.7), ("tiny", "small", "medium", "large", "extreme"))
        out.append(f"{c['family']}:{str(c['side']).upper()}:{strength}")
    return out


def _hold_text(hold: int) -> str:
    h = int(hold)
    if h <= 72:
        return "intraday_6h"
    if h <= 144:
        return "intraday_12h"
    if h <= 288:
        return "swing_24h"
    return "multi_day_36h"


def _label(util: dict[str, Any], cfg: ActionVerifierTextConfig) -> str:
    rank_utility = float(util.get("rank_utility", -1e9) or -1e9)
    net_return = float(util.get("net_return", -1e9) or -1e9)
    mae = float(util.get("mae", 1e9) or 1e9)
    mfe_to_mae = float(util.get("mfe_to_mae", 0.0) or 0.0)
    if rank_utility >= cfg.min_rank_utility and net_return >= cfg.min_net_return and mae <= cfg.max_mae and mfe_to_mae >= cfg.min_mfe_to_mae:
        return "ALLOW"
    return "BLOCK"


def _prompt(date: str, state: dict[str, Any], book: list[dict[str, Any]], action: dict[str, Any]) -> str:
    action_strength = _bucket(float(action.get("strength", 0.0)), (0.05, 0.15, 0.35, 0.7), ("tiny", "small", "medium", "large", "extreme"))
    lines = [
        "You are a post-ranker action verifier for BTCUSDT futures.",
        "Decide whether this exact executable action should be allowed after the ranker selected it.",
        "Use the categorical past-only regime text, action book, and selected action only.",
        "Output exactly one label: ALLOW or BLOCK.",
        "ALLOW requires clear edge, controlled adverse excursion, and reward/risk asymmetry for this exact side and horizon.",
        "BLOCK if edge is weak, regime is contradictory, horizon is too risky, or the action has break-risk.",
        "Do not output JSON or extra words.",
        "",
        f"Date: {date}",
        "Regime tokens: " + "; ".join(_state_text(state)),
        "Candidate book tokens: " + "; ".join(_candidate_text(book)),
        "Selected action tokens: " + "; ".join([
            f"family={action['family']}",
            f"side={str(action['side']).upper()}",
            f"horizon={_hold_text(int(action['hold_bars']))}",
            f"strength={action_strength}",
        ]),
    ]
    return "\n".join(lines)


def _load_market_with_external(cfg: ActionVerifierTextConfig) -> pd.DataFrame:
    market = _load_market(cfg.market_csv)
    if cfg.wave_trading_root:
        market = attach_wave_trading_external_features(market, wave_trading_root=cfg.wave_trading_root, tolerance=cfg.external_tolerance)
    return market


def build_rows(market: pd.DataFrame, cfg: ActionVerifierTextConfig) -> list[dict[str, Any]]:
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    families = _feature_candidates(features)
    dates = pd.to_datetime(market["date"])
    mask = _date_mask(dates, cfg.start_date, cfg.end_date)
    last_pos = len(market) - int(cfg.entry_delay_bars) - max(cfg.hold_bars_list) - 1
    pcfg = _policy_cfg(cfg)
    rows: list[dict[str, Any]] = []
    for pos in range(max(0, int(cfg.window_size) - 1), max(0, last_pos) + 1, max(1, int(cfg.stride_bars))):
        if not mask[pos]:
            continue
        date = str(market.iloc[pos]["date"])
        state = _state_summary(features, pos)
        book = _candidate_book(families, pos, top_k=int(cfg.top_k_families))
        for cand in book:
            for hold in cfg.hold_bars_list:
                util = _action_utility(market, pos, str(cand["side"]), int(hold), pcfg)
                if util is None:
                    continue
                action = {**cand, "hold_bars": int(hold)}
                rows.append({
                    "task": "event_action_verifier_text",
                    "date": date,
                    "signal_pos": int(pos),
                    "action": {"family": cand["family"], "side": cand["side"], "hold_bars": int(hold), "strength": cand["strength"]},
                    "prompt": _prompt(date, state, book, action),
                    "target": _label(util, cfg),
                    "action_audit": {"family": cand["family"], "side": cand["side"], "hold_bars": int(hold), **{k: util.get(k) for k in ("rank_utility", "net_return", "mae", "mfe", "mfe_to_mae", "utility")}},
                    "leakage_guard": {"prompt_uses_future_path": False, "target_uses_future_audit_for_training_only": True, "candidate_book_uses_past_only_features": True, "categorical_prompt_uses_past_only_state": True},
                })
    return rows


def run(cfg: ActionVerifierTextConfig) -> dict[str, Any]:
    rows = build_rows(_load_market_with_external(cfg), cfg)
    out = Path(cfg.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n")
    counts = Counter(str(r["target"]) for r in rows)
    summary = {"config": asdict(cfg), "rows": len(rows), "target_counts": dict(sorted(counts.items())), "allow_rate": counts.get("ALLOW", 0) / max(1, len(rows)), "leakage_guard": {"prompts_are_past_only": True, "labels_use_future_outcomes": True, "not_a_backtest": True}}
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build categorical exact-action verifier labels")
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--start-date", default="2020-01-01")
    p.add_argument("--end-date", default="2024-01-01")
    p.add_argument("--window-size", type=int, default=144)
    p.add_argument("--stride-bars", type=int, default=72)
    p.add_argument("--hold-bars-list", default="72,144,288,432")
    p.add_argument("--top-k-families", type=int, default=5)
    p.add_argument("--min-rank-utility", type=float, default=0.012)
    p.add_argument("--min-net-return", type=float, default=0.0075)
    p.add_argument("--max-mae", type=float, default=0.014)
    p.add_argument("--min-mfe-to-mae", type=float, default=1.2)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--external-tolerance", default="30min")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    cfg = ActionVerifierTextConfig(
        market_csv=a.market_csv,
        output=a.output,
        summary_output=a.summary_output,
        start_date=a.start_date,
        end_date=a.end_date,
        window_size=a.window_size,
        stride_bars=a.stride_bars,
        hold_bars_list=_parse_ints(a.hold_bars_list),
        top_k_families=a.top_k_families,
        min_rank_utility=a.min_rank_utility,
        min_net_return=a.min_net_return,
        max_mae=a.max_mae,
        min_mfe_to_mae=a.min_mfe_to_mae,
        wave_trading_root=a.wave_trading_root,
        external_tolerance=a.external_tolerance,
    )
    print(json.dumps(run(cfg), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
