"""Build pre-entry regime safety labels for LLM abstention.

The value-ranker failed because it kept ranking actions during break regimes.
This dataset changes the abstraction: before choosing an action, the model must
classify the current prompt-visible context as SAFE_TRADE, UNSAFE_NO_EDGE, or
BREAK_RISK.  Prompts use only past state and the candidate action book; labels
use future executable path outcomes for training only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.external_features import attach_wave_trading_external_features
from preprocessing.market_features import build_market_feature_frame
from training.event_action_policy_data import _action_utility, _candidate_book, _date_mask, _load_market, _state_summary, EventActionPolicyConfig
from training.event_candidate_pool_probe import _feature_candidates


@dataclass(frozen=True)
class RegimeSafetyConfig:
    market_csv: str
    output: str
    summary_output: str = ""
    start_date: str = "2020-01-01"
    end_date: str = "2024-01-01"
    window_size: int = 144
    stride_bars: int = 72
    hold_bars_list: tuple[int, ...] = (72, 144, 288, 432)
    top_k_families: int = 5
    safe_min_rank_utility: float = 0.01
    safe_min_net_return: float = 0.005
    safe_max_mae: float = 0.018
    break_max_mae: float = 0.035
    break_max_loss: float = -0.012
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    mae_penalty: float = 1.0
    wave_trading_root: str = ""
    external_tolerance: str = "30min"


def _parse_ints(csv: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(csv).split(",") if x.strip())


def _policy_cfg(cfg: RegimeSafetyConfig) -> EventActionPolicyConfig:
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


def _prompt(date: str, state: dict[str, Any], book: list[dict[str, Any]]) -> str:
    visible_book = [{"family": c["family"], "side": c["side"], "strength": c["strength"]} for c in book]
    return "\n".join(
        [
            "You are a pre-trade regime safety classifier for BTCUSDT futures.",
            "Use only the past-only state and prompt-visible candidate action book.",
            "Output exactly one label: SAFE_TRADE, UNSAFE_NO_EDGE, or BREAK_RISK.",
            "SAFE_TRADE means at least one candidate has enough edge after path risk.",
            "UNSAFE_NO_EDGE means no candidate has enough edge but path risk is not extreme.",
            "BREAK_RISK means candidate actions face severe adverse excursion or broad negative path risk.",
            "Do not output JSON or extra words.",
            "",
            f"Date: {date}",
            f"Past-only state: {json.dumps(state, sort_keys=True, separators=(',', ':'))}",
            f"Candidate action book: {json.dumps(visible_book, sort_keys=True, separators=(',', ':'))}",
        ]
    )


def _label(actions: list[dict[str, Any]], cfg: RegimeSafetyConfig) -> tuple[str, dict[str, Any]]:
    if not actions:
        return "UNSAFE_NO_EDGE", {"reason": "empty_action_book"}
    best = max(actions, key=lambda a: (float(a.get("rank_utility", -1e9)), float(a.get("net_return", -1e9))))
    best_rank = float(best.get("rank_utility", -1e9))
    best_net = float(best.get("net_return", -1e9))
    best_mae = float(best.get("mae", 1e9))
    worst_net = min(float(a.get("net_return", 0.0)) for a in actions)
    worst_mae = max(float(a.get("mae", 0.0)) for a in actions)
    losing_rate = sum(1 for a in actions if float(a.get("net_return", 0.0)) <= 0.0) / max(1, len(actions))
    if best_rank >= cfg.safe_min_rank_utility and best_net >= cfg.safe_min_net_return and best_mae <= cfg.safe_max_mae:
        label = "SAFE_TRADE"
    elif worst_mae >= cfg.break_max_mae or worst_net <= cfg.break_max_loss or (losing_rate >= 0.75 and best_net <= 0.0):
        label = "BREAK_RISK"
    else:
        label = "UNSAFE_NO_EDGE"
    audit = {
        "best_family": best.get("family"),
        "best_side": best.get("side"),
        "best_hold_bars": best.get("hold_bars"),
        "best_rank_utility": best_rank,
        "best_net_return": best_net,
        "best_mae": best_mae,
        "worst_net_return": worst_net,
        "worst_mae": worst_mae,
        "losing_action_rate": losing_rate,
        "candidate_actions": len(actions),
    }
    return label, audit


def build_rows(market: pd.DataFrame, cfg: RegimeSafetyConfig) -> list[dict[str, Any]]:
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
        actions: list[dict[str, Any]] = []
        for cand in book:
            for hold in cfg.hold_bars_list:
                util = _action_utility(market, pos, str(cand["side"]), int(hold), pcfg)
                if util is None:
                    continue
                actions.append({**cand, "hold_bars": int(hold), **util})
        label, audit = _label(actions, cfg)
        rows.append(
            {
                "task": "event_regime_safety",
                "date": date,
                "signal_pos": int(pos),
                "prompt": _prompt(date, state, book),
                "target": label,
                "safety_audit": audit,
                "leakage_guard": {
                    "prompt_uses_future_path": False,
                    "target_uses_future_action_outcomes_for_training_only": True,
                    "candidate_book_uses_past_only_features": True,
                },
            }
        )
    return rows


def _load_market_with_external(cfg: RegimeSafetyConfig) -> pd.DataFrame:
    market = _load_market(cfg.market_csv)
    if cfg.wave_trading_root:
        market = attach_wave_trading_external_features(market, wave_trading_root=cfg.wave_trading_root, tolerance=cfg.external_tolerance)
    return market


def run(cfg: RegimeSafetyConfig) -> dict[str, Any]:
    rows = build_rows(_load_market_with_external(cfg), cfg)
    out = Path(cfg.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))
    counts = Counter(str(r["target"]) for r in rows)
    audit = [r["safety_audit"] for r in rows]
    summary = {
        "config": asdict(cfg) | {"hold_bars_list": list(cfg.hold_bars_list)},
        "rows": len(rows),
        "period": {"start": rows[0]["date"] if rows else None, "end": rows[-1]["date"] if rows else None},
        "target_counts": dict(sorted(counts.items())),
        "safe_rate": counts.get("SAFE_TRADE", 0) / max(1, len(rows)),
        "mean_best_net_return_pct": float(np.mean([float(a.get("best_net_return", 0.0)) for a in audit]) * 100.0) if audit else 0.0,
        "mean_worst_mae_pct": float(np.mean([float(a.get("worst_mae", 0.0)) for a in audit]) * 100.0) if audit else 0.0,
        "leakage_guard": {"prompts_are_past_only": True, "labels_use_future_outcomes": True, "not_a_backtest": True},
    }
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build pre-entry regime safety labels")
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--start-date", default="2020-01-01")
    p.add_argument("--end-date", default="2024-01-01")
    p.add_argument("--window-size", type=int, default=144)
    p.add_argument("--stride-bars", type=int, default=72)
    p.add_argument("--hold-bars-list", default="72,144,288,432")
    p.add_argument("--top-k-families", type=int, default=5)
    p.add_argument("--safe-min-rank-utility", type=float, default=0.01)
    p.add_argument("--safe-min-net-return", type=float, default=0.005)
    p.add_argument("--safe-max-mae", type=float, default=0.018)
    p.add_argument("--break-max-mae", type=float, default=0.035)
    p.add_argument("--break-max-loss", type=float, default=-0.012)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--external-tolerance", default="30min")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    cfg = RegimeSafetyConfig(
        market_csv=a.market_csv,
        output=a.output,
        summary_output=a.summary_output,
        start_date=a.start_date,
        end_date=a.end_date,
        window_size=a.window_size,
        stride_bars=a.stride_bars,
        hold_bars_list=_parse_ints(a.hold_bars_list),
        top_k_families=a.top_k_families,
        safe_min_rank_utility=a.safe_min_rank_utility,
        safe_min_net_return=a.safe_min_net_return,
        safe_max_mae=a.safe_max_mae,
        break_max_mae=a.break_max_mae,
        break_max_loss=a.break_max_loss,
        wave_trading_root=a.wave_trading_root,
        external_tolerance=a.external_tolerance,
    )
    print(json.dumps(run(cfg), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
