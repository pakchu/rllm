"""Build single-LLM policy rows with a richer event-action book.

Instead of choosing one global event family, each timestamp presents multiple
past-only candidate actions (family, side, hold profile) and the target selects
the best action by strict future path utility for training only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.external_features import attach_wave_trading_external_features
from preprocessing.market_features import build_market_feature_frame
from training.event_candidate_pool_probe import _feature_candidates, _load_market
from training.path_outcome_dataset import PathOutcomeConfig, compute_trade_path_outcome


@dataclass(frozen=True)
class EventActionPolicyConfig:
    market_csv: str
    output: str
    summary_output: str = ""
    start_date: str = "2020-01-01"
    end_date: str = "2024-01-01"
    window_size: int = 144
    stride_bars: int = 72
    hold_bars_list: tuple[int, ...] = (72, 144, 288, 432)
    top_k_families: int = 5
    no_trade_utility: float = 0.001
    min_trade_net_return: float = -1.0
    max_trade_mae: float = 1.0
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    mae_penalty: float = 1.0
    wave_trading_root: str = ""
    external_tolerance: str = "30min"


def _parse_ints(csv: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(csv).split(",") if x.strip())


def _date_mask(dates: pd.Series, start: str, end: str) -> np.ndarray:
    return np.asarray((dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end)), dtype=bool)


def _state_summary(features: pd.DataFrame, pos: int) -> dict[str, Any]:
    row = features.iloc[int(pos)]
    keys = [
        "trend_24",
        "trend_96",
        "range_pos",
        "bb_z",
        "rsi_norm",
        "volume_zscore",
        "taker_imbalance",
        "kimchi_premium_zscore",
        "kimchi_premium_change",
        "dxy_zscore",
        "usdkrw_zscore",
        "window_drawdown",
        "htf_4h_return_4",
        "htf_1d_return_1",
        "htf_1d_drawdown_4",
        "htf_1w_return_4",
    ]
    return {k: round(float(row.get(k, 0.0)), 6) for k in keys}


def _candidate_book(families: dict[str, tuple[np.ndarray, np.ndarray]], pos: int, *, top_k: int) -> list[dict[str, Any]]:
    scored = []
    for family, (strength, direction) in families.items():
        s = float(strength[pos]) if np.isfinite(strength[pos]) else 0.0
        d = float(direction[pos]) if np.isfinite(direction[pos]) else 0.0
        if s <= 0.0 or d == 0.0:
            continue
        scored.append(
            {
                "family": family,
                "side": "LONG" if d > 0 else "SHORT",
                "strength": round(s, 6),
            }
        )
    scored.sort(key=lambda r: float(r["strength"]), reverse=True)
    return scored[: max(1, int(top_k))]


def _action_utility(market: pd.DataFrame, signal_pos: int, side: str, hold: int, cfg: EventActionPolicyConfig) -> dict[str, Any] | None:
    pcfg = PathOutcomeConfig(
        hold_bars=int(hold),
        entry_delay_bars=int(cfg.entry_delay_bars),
        fee_rate=float(cfg.fee_rate),
        slippage_rate=float(cfg.slippage_rate),
        leverage=float(cfg.leverage),
        mae_penalty=float(cfg.mae_penalty),
    )
    out = compute_trade_path_outcome(market, signal_pos, side, pcfg)
    if out is None:
        return None
    rank_utility = float(out.utility)
    if float(out.net_return) < float(cfg.min_trade_net_return) or float(out.mae) > float(cfg.max_trade_mae):
        rank_utility = float("-inf")
    return {
        "net_return": float(out.net_return),
        "mae": float(out.mae),
        "mfe": float(out.mfe),
        "utility": float(out.utility),
        "rank_utility": rank_utility,
    }


def _prompt(date: str, state: dict[str, Any], candidates: list[dict[str, Any]], cfg: EventActionPolicyConfig) -> str:
    visible = [
        {
            "family": c["family"],
            "side": c["side"],
            "strength": c["strength"],
            "allowed_holds": list(cfg.hold_bars_list),
        }
        for c in candidates
    ]
    return "\n".join(
        [
            "You are a single Gemma policy for BTCUSDT futures.",
            "Use only the past-only state and candidate action book below.",
            "Choose one action. If edge is weak or risk is high, choose NO_TRADE.",
            "Return JSON with keys: gate, family, side, hold_bars, confidence.",
            "Allowed gate: TRADE, NO_TRADE. If NO_TRADE, family NONE, side NONE, hold_bars 0.",
            "",
            f"Date: {date}",
            f"Past-only state: {json.dumps(state, sort_keys=True, separators=(',', ':'))}",
            f"Candidate action book: {json.dumps(visible, sort_keys=True, separators=(',', ':'))}",
        ]
    )


def _target(best: dict[str, Any], cfg: EventActionPolicyConfig) -> dict[str, Any]:
    if str(best["gate"]) == "NO_TRADE":
        return {"gate": "NO_TRADE", "family": "NONE", "side": "NONE", "hold_bars": 0, "confidence": "HIGH"}
    util = float(best.get("rank_utility", 0.0))
    confidence = "HIGH" if util >= 0.01 else ("MID" if util >= 0.004 else "LOW")
    return {
        "gate": "TRADE",
        "family": best["family"],
        "side": best["side"],
        "hold_bars": int(best["hold_bars"]),
        "confidence": confidence,
    }


def build_event_action_policy_rows(market: pd.DataFrame, cfg: EventActionPolicyConfig) -> list[dict[str, Any]]:
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    families = _feature_candidates(features)
    dates = pd.to_datetime(market["date"])
    mask = _date_mask(dates, cfg.start_date, cfg.end_date)
    last_pos = len(market) - int(cfg.entry_delay_bars) - max(cfg.hold_bars_list) - 1
    rows: list[dict[str, Any]] = []
    for pos in range(max(0, int(cfg.window_size) - 1), max(0, last_pos) + 1, max(1, int(cfg.stride_bars))):
        if not mask[pos]:
            continue
        date = str(market.iloc[pos]["date"])
        state = _state_summary(features, pos)
        book = _candidate_book(families, pos, top_k=int(cfg.top_k_families))
        actions: list[dict[str, Any]] = [
            {
                "gate": "NO_TRADE",
                "family": "NONE",
                "side": "NONE",
                "hold_bars": 0,
                "rank_utility": float(cfg.no_trade_utility),
                "net_return": 0.0,
                "mae": 0.0,
            }
        ]
        for cand in book:
            for hold in cfg.hold_bars_list:
                util = _action_utility(market, pos, str(cand["side"]), int(hold), cfg)
                if util is None:
                    continue
                actions.append({**cand, "gate": "TRADE", "hold_bars": int(hold), **util})
        actions.sort(key=lambda a: (float(a.get("rank_utility", -1e9)), float(a.get("net_return", -1e9))), reverse=True)
        best = actions[0]
        rows.append(
            {
                "task": "event_action_policy_sft",
                "date": date,
                "signal_pos": int(pos),
                "prompt": _prompt(date, state, book, cfg),
                "target": json.dumps(_target(best, cfg), sort_keys=True, separators=(",", ":"), ensure_ascii=False),
                "target_action_audit": {k: best.get(k) for k in ("gate", "family", "side", "hold_bars", "rank_utility", "net_return", "mae", "mfe", "utility")},
                "candidate_count": len(actions),
                "leakage_guard": {
                    "prompt_uses_future_path": False,
                    "target_uses_future_ohlc_utility_for_training_only": True,
                    "candidate_book_uses_past_only_features": True,
                },
            }
        )
    return rows


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summary(rows: list[dict[str, Any]], cfg: EventActionPolicyConfig) -> dict[str, Any]:
    counts = Counter()
    family_counts = Counter()
    hold_counts = Counter()
    utilities = []
    prompt_lens = []
    for row in rows:
        target = json.loads(row["target"])
        counts[str(target["gate"])] += 1
        family_counts[str(target["family"])] += 1
        hold_counts[str(target["hold_bars"])] += 1
        utilities.append(float(row.get("target_action_audit", {}).get("rank_utility", 0.0) or 0.0))
        prompt_lens.append(len(str(row.get("prompt", ""))))
    return {
        "rows": len(rows),
        "period": {"start": rows[0]["date"] if rows else None, "end": rows[-1]["date"] if rows else None},
        "gate_counts": dict(counts),
        "family_counts": dict(family_counts.most_common(20)),
        "hold_counts": dict(hold_counts),
        "rank_utility_mean_pct": (sum(utilities) / max(1, len(utilities))) * 100.0,
        "prompt_chars": {"min": min(prompt_lens) if prompt_lens else 0, "max": max(prompt_lens) if prompt_lens else 0, "mean": sum(prompt_lens) / max(1, len(prompt_lens))},
        "config": asdict(cfg) | {"hold_bars_list": list(cfg.hold_bars_list)},
        "leakage_guard": {
            "prompts_are_past_only": True,
            "targets_use_future_ohlc_utility": True,
            "not_a_backtest_result": True,
        },
    }


def build_event_action_policy_jsonl(cfg: EventActionPolicyConfig) -> dict[str, Any]:
    market = _load_market(cfg.market_csv)
    if cfg.wave_trading_root:
        market = attach_wave_trading_external_features(market, wave_trading_root=cfg.wave_trading_root, tolerance=cfg.external_tolerance)
    rows = build_event_action_policy_rows(market, cfg)
    _write_jsonl(cfg.output, rows)
    summary = {"as_of": datetime.now(timezone.utc).isoformat(), "output": cfg.output, "summary": _summary(rows, cfg)}
    summary_output = cfg.summary_output or f"{cfg.output}.summary.json"
    Path(summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build event-action single policy SFT rows")
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--start-date", default="2020-01-01")
    p.add_argument("--end-date", default="2024-01-01")
    p.add_argument("--window-size", type=int, default=144)
    p.add_argument("--stride-bars", type=int, default=72)
    p.add_argument("--hold-bars-list", default="72,144,288,432")
    p.add_argument("--top-k-families", type=int, default=5)
    p.add_argument("--no-trade-utility", type=float, default=0.001)
    p.add_argument("--min-trade-net-return", type=float, default=-1.0)
    p.add_argument("--max-trade-mae", type=float, default=1.0)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--leverage", type=float, default=0.5)
    p.add_argument("--fee-rate", type=float, default=0.0004)
    p.add_argument("--slippage-rate", type=float, default=0.0001)
    p.add_argument("--mae-penalty", type=float, default=1.0)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--external-tolerance", default="30min")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EventActionPolicyConfig(
        market_csv=args.market_csv,
        output=args.output,
        summary_output=args.summary_output,
        start_date=args.start_date,
        end_date=args.end_date,
        window_size=int(args.window_size),
        stride_bars=int(args.stride_bars),
        hold_bars_list=_parse_ints(args.hold_bars_list),
        top_k_families=int(args.top_k_families),
        no_trade_utility=float(args.no_trade_utility),
        min_trade_net_return=float(args.min_trade_net_return),
        max_trade_mae=float(args.max_trade_mae),
        entry_delay_bars=int(args.entry_delay_bars),
        leverage=float(args.leverage),
        fee_rate=float(args.fee_rate),
        slippage_rate=float(args.slippage_rate),
        mae_penalty=float(args.mae_penalty),
        wave_trading_root=args.wave_trading_root,
        external_tolerance=args.external_tolerance,
    )
    print(json.dumps(build_event_action_policy_jsonl(cfg), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
