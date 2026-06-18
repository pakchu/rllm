"""Build DPO-style preferences from the event-action book."""

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
from training.event_action_policy_data import (
    _action_utility,
    _candidate_book,
    _date_mask,
    _parse_ints,
    _prompt,
    _state_summary,
    _target,
    EventActionPolicyConfig,
)
from training.event_candidate_pool_probe import _feature_candidates, _load_market


@dataclass(frozen=True)
class EventActionPreferenceConfig(EventActionPolicyConfig):
    preference_output: str = "data/event_action_preferences.jsonl"
    max_pairs_per_row: int = 3
    min_utility_gap: float = 0.003


def _action_json(action: dict[str, Any], cfg: EventActionPolicyConfig) -> str:
    return json.dumps(_target(action, cfg), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _finite(value: Any, default: float = -1.0) -> float:
    try:
        x = float(value)
    except Exception:
        return float(default)
    return x if np.isfinite(x) else float(default)


def _audit(action: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k in ("gate", "family", "side", "hold_bars"):
        out[k] = action.get(k)
    for k in ("rank_utility", "net_return", "mae", "mfe", "mfe_to_mae", "utility"):
        out[k] = _finite(action.get(k), default=-1.0)
    return out


def _build_actions(market: pd.DataFrame, features: pd.DataFrame, families: dict[str, tuple[np.ndarray, np.ndarray]], pos: int, cfg: EventActionPreferenceConfig) -> tuple[str, list[dict[str, Any]]]:
    date = str(market.iloc[pos]["date"])
    state = _state_summary(features, pos)
    book = _candidate_book(families, pos, top_k=int(cfg.top_k_families))
    prompt = _prompt(date, state, book, cfg)
    actions: list[dict[str, Any]] = [
        {
            "gate": "NO_TRADE",
            "family": "NONE",
            "side": "NONE",
            "hold_bars": 0,
            "rank_utility": float(cfg.no_trade_utility),
            "net_return": 0.0,
            "mae": 0.0,
            "mfe": 0.0,
            "utility": float(cfg.no_trade_utility),
        }
    ]
    for cand in book:
        for hold in cfg.hold_bars_list:
            util = _action_utility(market, pos, str(cand["side"]), int(hold), cfg)
            if util is None:
                continue
            actions.append({**cand, "gate": "TRADE", "hold_bars": int(hold), **util})
    actions.sort(key=lambda a: (float(a.get("rank_utility", -1e9)), float(a.get("net_return", -1e9))), reverse=True)
    return prompt, actions


def build_preferences(market: pd.DataFrame, cfg: EventActionPreferenceConfig) -> list[dict[str, Any]]:
    features = build_market_feature_frame(market, window_size=int(cfg.window_size))
    families = _feature_candidates(features)
    dates = pd.to_datetime(market["date"])
    mask = _date_mask(dates, cfg.start_date, cfg.end_date)
    last_pos = len(market) - int(cfg.entry_delay_bars) - max(cfg.hold_bars_list) - 1
    pairs: list[dict[str, Any]] = []
    for pos in range(max(0, int(cfg.window_size) - 1), max(0, last_pos) + 1, max(1, int(cfg.stride_bars))):
        if not mask[pos]:
            continue
        prompt, actions = _build_actions(market, features, families, pos, cfg)
        if len(actions) < 2:
            continue
        chosen = actions[0]
        rejected_pool = []
        # Prefer hard negatives: high MAE or negative utility trades, plus no-trade
        # when a trade is chosen and the gap is meaningful.
        for a in actions[1:]:
            gap = _finite(chosen.get("rank_utility"), default=-1.0) - _finite(a.get("rank_utility"), default=-1.0)
            if gap < float(cfg.min_utility_gap):
                continue
            hard = (
                str(a.get("gate")) == "NO_TRADE"
                or _finite(a.get("rank_utility"), default=-1.0) < 0.0
                or float(a.get("mae", 0.0) or 0.0) > float(cfg.max_trade_mae)
            )
            rejected_pool.append((0 if hard else 1, -gap, a))
        rejected_pool.sort(key=lambda x: (x[0], x[1]))
        for _, neg_gap, rejected in rejected_pool[: max(1, int(cfg.max_pairs_per_row))]:
            gap = -float(neg_gap)
            pairs.append(
                {
                    "task": "event_action_policy_preference",
                    "date": str(market.iloc[pos]["date"]),
                    "signal_pos": int(pos),
                    "prompt": prompt,
                    "chosen": _action_json(chosen, cfg),
                    "rejected": _action_json(rejected, cfg),
                    "chosen_action_audit": _audit(chosen),
                    "rejected_action_audit": _audit(rejected),
                    "utility_gap": gap,
                    "leakage_guard": {
                        "prompt_uses_future_path": False,
                        "chosen_rejected_use_future_ohlc_utility_for_training_only": True,
                        "candidate_book_uses_past_only_features": True,
                    },
                }
            )
    return pairs


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summary(pairs: list[dict[str, Any]], cfg: EventActionPreferenceConfig) -> dict[str, Any]:
    chosen = Counter()
    rejected = Counter()
    gaps = []
    for p in pairs:
        c = json.loads(p["chosen"])
        r = json.loads(p["rejected"])
        chosen[f"{c['gate']}/{c['family']}/{c['side']}/{c['hold_bars']}"] += 1
        rejected[f"{r['gate']}/{r['family']}/{r['side']}/{r['hold_bars']}"] += 1
        gaps.append(float(p.get("utility_gap", 0.0)))
    return {
        "pairs": len(pairs),
        "period": {"start": pairs[0]["date"] if pairs else None, "end": pairs[-1]["date"] if pairs else None},
        "chosen_top": dict(chosen.most_common(20)),
        "rejected_top": dict(rejected.most_common(20)),
        "mean_utility_gap_pct": (sum(gaps) / max(1, len(gaps))) * 100.0,
        "config": asdict(cfg) | {"hold_bars_list": list(cfg.hold_bars_list)},
        "leakage_guard": {
            "prompts_are_past_only": True,
            "preferences_use_future_ohlc_utility": True,
            "not_a_backtest_result": True,
        },
    }


def build_event_action_preferences(cfg: EventActionPreferenceConfig) -> dict[str, Any]:
    market = _load_market(cfg.market_csv)
    if cfg.wave_trading_root:
        market = attach_wave_trading_external_features(market, wave_trading_root=cfg.wave_trading_root, tolerance=cfg.external_tolerance)
    pairs = build_preferences(market, cfg)
    _write_jsonl(cfg.preference_output, pairs)
    summary = {"as_of": datetime.now(timezone.utc).isoformat(), "output": cfg.preference_output, "summary": _summary(pairs, cfg)}
    out = cfg.summary_output or f"{cfg.preference_output}.summary.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build event-action DPO preference rows")
    p.add_argument("--market-csv", required=True)
    p.add_argument("--preference-output", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--start-date", default="2020-01-01")
    p.add_argument("--end-date", default="2024-01-01")
    p.add_argument("--window-size", type=int, default=144)
    p.add_argument("--stride-bars", type=int, default=72)
    p.add_argument("--hold-bars-list", default="72,144,288,432")
    p.add_argument("--top-k-families", type=int, default=5)
    p.add_argument("--no-trade-utility", type=float, default=0.002)
    p.add_argument("--min-trade-net-return", type=float, default=0.001)
    p.add_argument("--max-trade-mae", type=float, default=0.015)
    p.add_argument("--min-trade-utility", type=float, default=0.002)
    p.add_argument("--min-trade-mfe-to-mae", type=float, default=1.0)
    p.add_argument("--min-utility-gap", type=float, default=0.003)
    p.add_argument("--max-pairs-per-row", type=int, default=3)
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
    cfg = EventActionPreferenceConfig(
        market_csv=args.market_csv,
        output="",
        preference_output=args.preference_output,
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
        min_trade_utility=float(args.min_trade_utility),
        min_trade_mfe_to_mae=float(args.min_trade_mfe_to_mae),
        min_utility_gap=float(args.min_utility_gap),
        max_pairs_per_row=int(args.max_pairs_per_row),
        entry_delay_bars=int(args.entry_delay_bars),
        leverage=float(args.leverage),
        fee_rate=float(args.fee_rate),
        slippage_rate=float(args.slippage_rate),
        mae_penalty=float(args.mae_penalty),
        wave_trading_root=args.wave_trading_root,
        external_tolerance=args.external_tolerance,
    )
    print(json.dumps(build_event_action_preferences(cfg), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
