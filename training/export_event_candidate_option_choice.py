"""Export same-signal A/B/C option-choice rows for compact LLM action ranking."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EventCandidateOptionChoiceCfg:
    train_candidates_jsonl: str
    eval_candidates_jsonl: str
    train_output: str
    eval_output: str
    summary_output: str
    min_trade_utility: float = 0.25
    numeric_keys: str = "trend_24,trend_96,range_pos,rsi_norm,mfi_norm,range_vol,window_drawdown,volume_zscore,taker_imbalance,dxy_momentum,dxy_zscore,kimchi_premium_zscore,kimchi_premium_change,usdkrw_momentum,mp_range_pos_24,mp_range_pos_96,mp_ret_24,mp_ret_96,mp_realized_vol_24,mp_taker_imbalance_mean_24,htf_4h_range_pos,htf_1d_range_pos,htf_4h_return_4,htf_1d_return_4"


def _load(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _groups(rows: list[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    out: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault((str(row.get("date")), int(row.get("signal_pos", -1) or -1)), []).append(row)
    return out


def _utility(row: dict[str, Any] | None) -> float:
    if not row:
        return -999.0
    reward = row.get("reward") if isinstance(row.get("reward"), dict) else {}
    return float(reward.get("utility", reward.get("rank_utility", reward.get("net_return_pct", 0.0))) or 0.0)


def _reward_summary(row: dict[str, Any] | None) -> dict[str, float]:
    if not row:
        return {"utility": -999.0, "net_return_pct": -999.0, "mae_pct": 999.0, "mfe_pct": 0.0}
    reward = row.get("reward") if isinstance(row.get("reward"), dict) else {}
    return {
        "utility": float(reward.get("utility", reward.get("net_return_pct", 0.0)) or 0.0),
        "net_return_pct": float(reward.get("net_return_pct", 0.0) or 0.0),
        "mae_pct": float(reward.get("mae_pct", 999.0) or 999.0),
        "mfe_pct": float(reward.get("mfe_pct", 0.0) or 0.0),
    }


def _hold(row: dict[str, Any] | None) -> int:
    if not row:
        return 0
    cand = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
    return int(cand.get("hold_bars", cand.get("horizon", 288)) or 288)


def _choice(by_side: dict[str, dict[str, Any]], min_trade_utility: float) -> tuple[str, dict[str, float]]:
    long_u = _utility(by_side.get("LONG"))
    short_u = _utility(by_side.get("SHORT"))
    no_u = 0.0
    scores = {"A": long_u, "B": short_u, "C": no_u}
    best_trade_label = "A" if long_u >= short_u else "B"
    if scores[best_trade_label] >= float(min_trade_utility):
        return best_trade_label, scores
    return "C", scores


def _prompt(group: list[dict[str, Any]], cfg: EventCandidateOptionChoiceCfg) -> str:
    first = group[0]
    by_side = {str(r.get("side", "")).upper(): r for r in group}
    tokens = first.get("state_tokens", {}) if isinstance(first.get("state_tokens"), dict) else {}
    snap = first.get("feature_snapshot", {}) if isinstance(first.get("feature_snapshot"), dict) else {}
    triggers = first.get("event_triggers", []) if isinstance(first.get("event_triggers"), list) else []
    numeric_keys = [k.strip() for k in str(cfg.numeric_keys).split(",") if k.strip()]
    lines = [
        "Task: choose the best action for this BTCUSDT futures signal.",
        "Use only signal-time information. Answer with exactly one letter: A, B, or C.",
        "A = LONG trade",
        "B = SHORT trade",
        "C = NO_TRADE",
        f"Date: {first.get('date')}",
        f"Signal position: {first.get('signal_pos')}",
        "Event triggers: " + (", ".join(map(str, triggers)) if triggers else "none"),
        f"LONG hold_bars: {_hold(by_side.get('LONG'))}",
        f"SHORT hold_bars: {_hold(by_side.get('SHORT'))}",
        "State buckets:",
    ]
    for key in sorted(tokens):
        lines.append(f"- {key}: {tokens[key]}")
    lines.append("Numeric evidence:")
    for key in numeric_keys:
        if key in snap:
            try:
                lines.append(f"- {key}: {float(snap[key]):+.5f}")
            except Exception:
                pass
    return "\n".join(lines)


def _rows(rows: list[dict[str, Any]], cfg: EventCandidateOptionChoiceCfg) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, group in sorted(_groups(rows).items()):
        by_side = {str(r.get("side", "")).upper(): r for r in group}
        if "LONG" not in by_side or "SHORT" not in by_side:
            continue
        choice, utility_by_choice = _choice(by_side, cfg.min_trade_utility)
        out.append(
            {
                "task": "event_candidate_option_choice",
                "date": key[0],
                "signal_pos": key[1],
                "prompt": _prompt(group, cfg),
                "target": choice,
                "choice_utility": utility_by_choice,
                "actions": {
                    "A": {"gate": "TRADE", "side": "LONG", "hold_bars": _hold(by_side.get("LONG"))},
                    "B": {"gate": "TRADE", "side": "SHORT", "hold_bars": _hold(by_side.get("SHORT"))},
                    "C": {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0},
                },
                "reward_audit": {"LONG": _reward_summary(by_side.get("LONG")), "SHORT": _reward_summary(by_side.get("SHORT")), "NO_TRADE": {"utility": 0.0}},
                "leakage_guard": {
                    "prompt_uses_future_reward": False,
                    "target_uses_future_reward_for_training_only": True,
                    "same_signal_candidates_only": True,
                },
            }
        )
    return out


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(r["target"]) for r in rows)
    lens = [len(str(r.get("prompt", ""))) for r in rows]
    return {
        "rows": len(rows),
        "target_counts": dict(sorted(counts.items())),
        "prompt_chars": {"min": min(lens) if lens else 0, "mean": sum(lens) / max(1, len(lens)), "max": max(lens) if lens else 0},
    }


def run(cfg: EventCandidateOptionChoiceCfg) -> dict[str, Any]:
    train = _rows(_load(cfg.train_candidates_jsonl), cfg)
    eval_rows = _rows(_load(cfg.eval_candidates_jsonl), cfg)
    _write(cfg.train_output, train)
    _write(cfg.eval_output, eval_rows)
    report = {
        "config": asdict(cfg),
        "outputs": {"train": cfg.train_output, "eval": cfg.eval_output},
        "train": _summary(train),
        "eval": _summary(eval_rows),
        "contract": "A/B/C same-signal option choice; prompt is signal-time only; target is label-only",
    }
    Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-candidates-jsonl", required=True)
    p.add_argument("--eval-candidates-jsonl", required=True)
    p.add_argument("--train-output", required=True)
    p.add_argument("--eval-output", required=True)
    p.add_argument("--summary-output", required=True)
    p.add_argument("--min-trade-utility", type=float, default=EventCandidateOptionChoiceCfg.min_trade_utility)
    p.add_argument("--numeric-keys", default=EventCandidateOptionChoiceCfg.numeric_keys)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(EventCandidateOptionChoiceCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
