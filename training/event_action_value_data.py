"""Build candidate-level value labels for prompt-visible event actions.

Each row asks whether one concrete candidate action is worth taking.  This is a
better match for action selection than scoring a whole JSON action by likelihood:
the model learns TAKE/SKIP value for each candidate, then a live selector can pick
the highest TAKE-margin action or abstain.
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
class EventActionValueConfig:
    market_csv: str
    output: str
    summary_output: str = ""
    start_date: str = "2020-01-01"
    end_date: str = "2024-01-01"
    window_size: int = 144
    stride_bars: int = 72
    hold_bars_list: tuple[int, ...] = (72, 144, 288, 432)
    top_k_families: int = 5
    min_rank_utility: float = 0.01
    min_net_return: float = 0.005
    max_mae: float = 0.018
    min_mfe_to_mae: float = 0.0
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    mae_penalty: float = 1.0
    wave_trading_root: str = ""
    external_tolerance: str = "30min"


def _parse_ints(csv: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(csv).split(",") if x.strip())


def _policy_cfg(cfg: EventActionValueConfig) -> EventActionPolicyConfig:
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


def _label(util: dict[str, Any], cfg: EventActionValueConfig) -> str:
    rank_utility = float(util.get("rank_utility", -1e9) or -1e9)
    net_return = float(util.get("net_return", -1e9) or -1e9)
    mae = float(util.get("mae", 1e9) or 1e9)
    mfe_to_mae = float(util.get("mfe_to_mae", 0.0) or 0.0)
    return "TAKE" if rank_utility >= cfg.min_rank_utility and net_return >= cfg.min_net_return and mae <= cfg.max_mae and mfe_to_mae >= cfg.min_mfe_to_mae else "SKIP"


def _prompt(date: str, state: dict[str, Any], book: list[dict[str, Any]], action: dict[str, Any]) -> str:
    visible_book = [
        {"family": c["family"], "side": c["side"], "strength": c["strength"]}
        for c in book
    ]
    candidate = {"family": action["family"], "side": action["side"], "hold_bars": int(action["hold_bars"]), "strength": action.get("strength")}
    return "\n".join(
        [
            "You are an action value judge for BTCUSDT futures.",
            "Use only the past-only state, the prompt-visible action book, and the candidate action.",
            "Output exactly one label: TAKE or SKIP.",
            "TAKE only if the candidate has enough expected edge after path-risk; otherwise SKIP.",
            "Do not output JSON or extra words.",
            "",
            f"Date: {date}",
            f"Past-only state: {json.dumps(state, sort_keys=True, separators=(',', ':'))}",
            f"Action book: {json.dumps(visible_book, sort_keys=True, separators=(',', ':'))}",
            f"Candidate action: {json.dumps(candidate, sort_keys=True, separators=(',', ':'))}",
        ]
    )


def build_value_rows(market: pd.DataFrame, cfg: EventActionValueConfig) -> list[dict[str, Any]]:
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
                action = {**cand, "hold_bars": int(hold), **util}
                target = _label(util, cfg)
                rows.append(
                    {
                        "task": "event_action_value",
                        "date": date,
                        "signal_pos": int(pos),
                        "action": {"family": cand["family"], "side": cand["side"], "hold_bars": int(hold), "strength": cand["strength"]},
                        "prompt": _prompt(date, state, book, action),
                        "target": target,
                        "action_audit": {k: action.get(k) for k in ("family", "side", "hold_bars", "rank_utility", "net_return", "mae", "mfe", "mfe_to_mae", "utility")},
                        "leakage_guard": {"prompt_uses_future_path": False, "target_uses_future_audit_for_training_only": True, "candidate_book_uses_past_only_features": True},
                    }
                )
    return rows


def _load_market_with_external(cfg: EventActionValueConfig) -> pd.DataFrame:
    market = _load_market(cfg.market_csv)
    if cfg.wave_trading_root:
        market = attach_wave_trading_external_features(market, wave_trading_root=cfg.wave_trading_root, tolerance=cfg.external_tolerance)
    return market


def run(cfg: EventActionValueConfig) -> dict[str, Any]:
    rows = build_value_rows(_load_market_with_external(cfg), cfg)
    out = Path(cfg.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n")
    counts = Counter(str(r["target"]) for r in rows)
    summary = {"config": asdict(cfg), "rows": len(rows), "target_counts": dict(sorted(counts.items()))}
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build candidate-level event-action value labels")
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--start-date", default="2020-01-01")
    p.add_argument("--end-date", default="2024-01-01")
    p.add_argument("--window-size", type=int, default=144)
    p.add_argument("--stride-bars", type=int, default=72)
    p.add_argument("--hold-bars-list", default="72,144,288,432")
    p.add_argument("--top-k-families", type=int, default=5)
    p.add_argument("--min-rank-utility", type=float, default=0.01)
    p.add_argument("--min-net-return", type=float, default=0.005)
    p.add_argument("--max-mae", type=float, default=0.018)
    p.add_argument("--min-mfe-to-mae", type=float, default=0.0)
    p.add_argument("--wave-trading-root", default="")
    p.add_argument("--external-tolerance", default="30min")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    cfg = EventActionValueConfig(
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
