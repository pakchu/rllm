"""Build same-prompt event signal preference pairs for DPO.

Side-specific TAKE labels made the LLM learn priors instead of ranking actions.
This builder groups LONG/SHORT candidates for the same signal and creates a
single prompt with action alternatives. Chosen/rejected responses are compact
trade actions ranked by future reward utility for training only.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EventSignalPreferenceCfg:
    train_candidates: str
    eval_candidates: str
    train_output: str
    eval_output: str
    summary_output: str
    min_trade_net_pct: float = 0.25
    min_trade_utility: float = 0.0
    full_net_pct: float = 1.2
    full_utility: float = 0.5
    max_pairs_per_signal: int = 2
    min_rank_margin: float = 0.0
    require_chosen_positive: bool = False


def _load(path: str) -> list[dict[str, Any]]:
    return [json.loads(l) for l in open(path) if l.strip()]


def _group(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    g: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        g[int(r.get("signal_pos"))].append(r)
    return g


def _f(row: dict[str, Any], key: str) -> float:
    try:
        return float((row.get("feature_snapshot", {}) or {}).get(key, 0.0) or 0.0)
    except Exception:
        return 0.0


def _prompt(rows: list[dict[str, Any]]) -> str:
    base = rows[0]
    toks = base.get("state_tokens", {}) if isinstance(base.get("state_tokens"), dict) else {}
    snap = base.get("feature_snapshot", {}) if isinstance(base.get("feature_snapshot"), dict) else {}
    keep = ["trend_96", "htf_4h_return_4", "range_pos", "rsi_norm", "bb_z", "range_vol", "window_drawdown", "taker_buy_ratio", "dxy_zscore", "kimchi_premium_zscore", "usdkrw_zscore"]
    lines = [
        "You are a BTCUSDT futures trader.",
        "Use only signal-time evidence. Choose exactly one action: NO_TRADE, LONG, or SHORT.",
        "Return compact JSON with keys: gate, side, position_scale, hold_bars, rationale_class.",
        f"date: {base.get('date')}",
        f"hold_bars: {int(base.get('candidate', {}).get('hold_bars', 288) or 288)}",
        f"event_triggers: {', '.join(map(str, base.get('event_triggers', []))) if base.get('event_triggers') else 'none'}",
        "state_tokens:",
    ]
    for k in sorted(toks):
        lines.append(f"- {k}: {toks[k]}")
    lines.append("numeric_features:")
    for k in keep:
        if k in snap:
            lines.append(f"- {k}: {float(snap.get(k, 0.0) or 0.0):+.6f}")
    lines.append("candidate_actions: NO_TRADE, LONG, SHORT")
    return "\n".join(lines)


def _action(gate: str, side: str = "NONE", scale: float = 0.0, hold: int = 0, rationale: str = "avoid_low_edge") -> str:
    return json.dumps({"gate": gate, "side": side, "position_scale": scale, "hold_bars": hold, "rationale_class": rationale}, ensure_ascii=False, sort_keys=True)


def _action_for_candidate(row: dict[str, Any], cfg: EventSignalPreferenceCfg) -> dict[str, Any]:
    reward = row.get("reward", {}) if isinstance(row.get("reward"), dict) else {}
    net = float(reward.get("net_return_pct", 0.0) or 0.0)
    util = float(reward.get("utility", net) or 0.0)
    side = str(row.get("side"))
    hold = int(row.get("candidate", {}).get("hold_bars", 288) or 288)
    if net >= cfg.full_net_pct and util >= cfg.full_utility:
        scale = 1.0
        rationale = "strong_reward_after_path_risk"
    elif net >= cfg.min_trade_net_pct and util >= cfg.min_trade_utility:
        scale = 0.5
        rationale = "positive_but_thin_edge"
    else:
        # Keep losing/weak side actions as explicit rejected alternatives so DPO
        # can learn to rank NO_TRADE or the opposite side above them.
        scale = 0.5
        rationale = "weak_or_negative_edge"
    text = _action("TRADE", side, scale, hold, rationale)
    rank = util + 0.05 * net
    return {"text": text, "rank": float(rank), "side": side, "net": net, "utility": util}


def _pairs_for_signal(rows: list[dict[str, Any]], cfg: EventSignalPreferenceCfg) -> list[dict[str, Any]]:
    if not rows:
        return []
    prompt = _prompt(rows)
    hold = int(rows[0].get("candidate", {}).get("hold_bars", 288) or 288)
    actions = [_action_for_candidate(r, cfg) for r in rows]
    # Include an explicit no-trade baseline with zero utility so DPO can learn abstention.
    actions.append({"text": _action("NO_TRADE", "NONE", 0.0, 0, "no_trade_baseline"), "rank": 0.0, "side": "NONE", "net": 0.0, "utility": 0.0})
    # Deduplicate identical action texts, keep best rank.
    best: dict[str, dict[str, Any]] = {}
    for a in actions:
        cur = best.get(a["text"])
        if cur is None or float(a["rank"]) > float(cur["rank"]):
            best[a["text"]] = a
    ranked = sorted(best.values(), key=lambda a: float(a["rank"]), reverse=True)
    if len(ranked) < 2:
        return []
    chosen = ranked[0]
    out = []
    if cfg.require_chosen_positive and float(chosen.get("rank", 0.0)) <= 0.0:
        return []
    for rejected in ranked[1 : 1 + max(1, int(cfg.max_pairs_per_signal))]:
        if chosen["text"] == rejected["text"]:
            continue
        margin = float(chosen["rank"]) - float(rejected["rank"])
        if margin < float(cfg.min_rank_margin):
            continue
        out.append({
            "task": "event_signal_preference_dpo",
            "date": rows[0].get("date"),
            "signal_pos": rows[0].get("signal_pos"),
            "prompt": prompt,
            "chosen": chosen["text"],
            "rejected": rejected["text"],
            "chosen_action": chosen,
            "rejected_action": rejected,
            "rank_margin": margin,
            "leakage_guard": {"prompt_uses_future_reward": False, "preference_uses_future_reward_for_training_only": True, "same_prompt_action_ranking": True},
        })
    return out


def _build(rows: list[dict[str, Any]], cfg: EventSignalPreferenceCfg) -> list[dict[str, Any]]:
    out=[]
    for _, group in sorted(_group(rows).items()):
        out.extend(_pairs_for_signal(group, cfg))
    return out


def _write(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    chosen = Counter(json.loads(r["chosen"])["gate"] + ":" + json.loads(r["chosen"])["side"] for r in rows)
    rejected = Counter(json.loads(r["rejected"])["gate"] + ":" + json.loads(r["rejected"])["side"] for r in rows)
    prompt_lens = [len(r["prompt"]) for r in rows]
    margins = [float(r["chosen_action"]["rank"]) - float(r["rejected_action"]["rank"]) for r in rows]
    return {"rows": len(rows), "chosen_counts": dict(chosen), "rejected_counts": dict(rejected), "prompt_chars": {"min": min(prompt_lens) if prompt_lens else 0, "max": max(prompt_lens) if prompt_lens else 0, "mean": sum(prompt_lens)/max(1,len(prompt_lens))}, "rank_margin": {"mean": float(np.mean(margins)) if margins else 0.0, "min": float(np.min(margins)) if margins else 0.0}}


def run(cfg: EventSignalPreferenceCfg) -> dict[str, Any]:
    train = _build(_load(cfg.train_candidates), cfg)
    ev = _build(_load(cfg.eval_candidates), cfg)
    _write(cfg.train_output, train)
    _write(cfg.eval_output, ev)
    report = {"config": cfg.__dict__, "train": _summary(train), "eval": _summary(ev), "contract": "same-prompt action preference pairs; prompt signal-time only; preference future-reward training label only"}
    Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build event signal DPO preference pairs")
    p.add_argument("--train-candidates", required=True)
    p.add_argument("--eval-candidates", required=True)
    p.add_argument("--train-output", required=True)
    p.add_argument("--eval-output", required=True)
    p.add_argument("--summary-output", required=True)
    p.add_argument("--min-trade-net-pct", type=float, default=EventSignalPreferenceCfg.min_trade_net_pct)
    p.add_argument("--min-trade-utility", type=float, default=EventSignalPreferenceCfg.min_trade_utility)
    p.add_argument("--full-net-pct", type=float, default=EventSignalPreferenceCfg.full_net_pct)
    p.add_argument("--full-utility", type=float, default=EventSignalPreferenceCfg.full_utility)
    p.add_argument("--max-pairs-per-signal", type=int, default=EventSignalPreferenceCfg.max_pairs_per_signal)
    p.add_argument("--min-rank-margin", type=float, default=EventSignalPreferenceCfg.min_rank_margin)
    p.add_argument("--require-chosen-positive", action="store_true")
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(EventSignalPreferenceCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
