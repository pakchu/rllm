"""Export same-signal LONG/SHORT/NO_TRADE listwise preference rows for RLLM training."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EventCandidateListwisePreferenceCfg:
    train_candidates_jsonl: str
    eval_candidates_jsonl: str
    train_output: str
    eval_output: str
    summary_output: str
    min_trade_utility: float = 0.25
    min_pair_gap: float = 0.25
    max_pairs_per_signal: int = 2
    numeric_keys: str = "trend_24,trend_96,range_pos,rsi_norm,mfi_norm,range_vol,window_drawdown,volume_zscore,taker_imbalance,dxy_momentum,dxy_zscore,kimchi_premium_zscore,kimchi_premium_change,usdkrw_momentum,mp_range_pos_24,mp_range_pos_96,mp_ret_24,mp_ret_96,mp_realized_vol_24,mp_taker_imbalance_mean_24,htf_4h_range_pos,htf_1d_range_pos,htf_4h_return_4,htf_1d_return_4"


def _load(path: str) -> list[dict[str, Any]]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _groups(rows: list[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    out: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("date")), int(row.get("signal_pos", -1) or -1))
        out.setdefault(key, []).append(row)
    return out


def _utility(row: dict[str, Any]) -> float:
    reward = row.get("reward") if isinstance(row.get("reward"), dict) else {}
    return float(reward.get("utility", reward.get("rank_utility", reward.get("net_return_pct", 0.0))) or 0.0)


def _hold(row: dict[str, Any] | None) -> int:
    if not row:
        return 0
    cand = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
    return int(cand.get("hold_bars", cand.get("horizon", 288)) or 288)


def _action(side: str, hold_bars: int, *, confidence: str = "MEDIUM") -> dict[str, Any]:
    side = str(side).upper()
    if side not in {"LONG", "SHORT"}:
        return {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0, "confidence": "LOW"}
    return {"gate": "TRADE", "side": side, "hold_bars": int(hold_bars), "confidence": confidence}


def _prompt(rows: list[dict[str, Any]], cfg: EventCandidateListwisePreferenceCfg) -> str:
    first = rows[0]
    tokens = first.get("state_tokens", {}) if isinstance(first.get("state_tokens"), dict) else {}
    snap = first.get("feature_snapshot", {}) if isinstance(first.get("feature_snapshot"), dict) else {}
    triggers = first.get("event_triggers", []) if isinstance(first.get("event_triggers"), list) else []
    numeric_keys = [k.strip() for k in str(cfg.numeric_keys).split(",") if k.strip()]
    by_side = {str(r.get("side", "")).upper(): r for r in rows}
    lines = [
        "Task: choose the best BTCUSDT futures action for this signal.",
        "Use only signal-time state. Do not infer or use future reward.",
        "Choose exactly one JSON action: LONG trade, SHORT trade, or NO_TRADE.",
        f"Date: {first.get('date')}",
        f"Signal position: {first.get('signal_pos')}",
        "Available actions: LONG, SHORT, NO_TRADE",
        "Event triggers: " + (", ".join(map(str, triggers)) if triggers else "none"),
        "State buckets:",
    ]
    for key in sorted(tokens):
        lines.append(f"- {key}: {tokens[key]}")
    lines.append("Signal numeric evidence:")
    for key in numeric_keys:
        if key in snap:
            try:
                lines.append(f"- {key}: {float(snap[key]):+.5f}")
            except Exception:
                pass
    lines.append("Candidate metadata:")
    for side in ("LONG", "SHORT"):
        row = by_side.get(side)
        lines.append(f"- {side}: hold_bars={_hold(row)}")
    return "\n".join(lines)


def _action_utility(action: dict[str, Any], by_side: dict[str, dict[str, Any]], no_trade_utility: float = 0.0) -> float:
    if action["gate"] == "NO_TRADE":
        return float(no_trade_utility)
    row = by_side.get(str(action["side"]).upper())
    return _utility(row) if row else -999.0


def _preference_rows(rows: list[dict[str, Any]], cfg: EventCandidateListwisePreferenceCfg) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, group in sorted(_groups(rows).items()):
        by_side = {str(r.get("side", "")).upper(): r for r in group}
        actions = [_action("NO_TRADE", 0)]
        for side in ("LONG", "SHORT"):
            if side in by_side:
                actions.append(_action(side, _hold(by_side[side]), confidence="HIGH"))
        scored = [(a, _action_utility(a, by_side)) for a in actions]
        # Trade only if the best trade clears the no-trade utility hurdle; otherwise choose NO_TRADE.
        best_trade = max((x for x in scored if x[0]["gate"] == "TRADE"), key=lambda x: x[1], default=None)
        if best_trade is not None and best_trade[1] >= float(cfg.min_trade_utility):
            chosen_action, chosen_u = best_trade
        else:
            chosen_action, chosen_u = _action("NO_TRADE", 0), 0.0
        rejected = [x for x in scored if x[0] != chosen_action and chosen_u - x[1] >= float(cfg.min_pair_gap)]
        rejected.sort(key=lambda x: x[1])
        if not rejected:
            continue
        prompt = _prompt(group, cfg)
        for rejected_action, rejected_u in rejected[: max(1, int(cfg.max_pairs_per_signal))]:
            out.append(
                {
                    "task": "event_candidate_listwise_preference",
                    "date": key[0],
                    "signal_pos": key[1],
                    "prompt": prompt,
                    "chosen": json.dumps(chosen_action, sort_keys=True),
                    "rejected": json.dumps(rejected_action, sort_keys=True),
                    "chosen_action": chosen_action,
                    "rejected_action": rejected_action,
                    "chosen_utility": chosen_u,
                    "rejected_utility": rejected_u,
                    "utility_gap": chosen_u - rejected_u,
                    "leakage_guard": {
                        "prompt_uses_future_reward": False,
                        "chosen_rejected_use_future_reward_for_training_only": True,
                        "same_signal_candidates_only": True,
                    },
                }
            )
    return out


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    chosen = Counter(json.loads(r["chosen"])["gate"] + ":" + json.loads(r["chosen"])["side"] for r in rows)
    rejected = Counter(json.loads(r["rejected"])["gate"] + ":" + json.loads(r["rejected"])["side"] for r in rows)
    gaps = [float(r.get("utility_gap", 0.0)) for r in rows]
    lens = [len(str(r.get("prompt", ""))) for r in rows]
    return {
        "rows": len(rows),
        "signals": len({(r["date"], r["signal_pos"]) for r in rows}),
        "chosen_counts": dict(sorted(chosen.items())),
        "rejected_counts": dict(sorted(rejected.items())),
        "utility_gap": {
            "min": min(gaps) if gaps else 0.0,
            "mean": sum(gaps) / max(1, len(gaps)),
            "max": max(gaps) if gaps else 0.0,
        },
        "prompt_chars": {
            "min": min(lens) if lens else 0,
            "mean": sum(lens) / max(1, len(lens)),
            "max": max(lens) if lens else 0,
        },
    }


def run(cfg: EventCandidateListwisePreferenceCfg) -> dict[str, Any]:
    train = _preference_rows(_load(cfg.train_candidates_jsonl), cfg)
    eval_rows = _preference_rows(_load(cfg.eval_candidates_jsonl), cfg)
    _write_jsonl(cfg.train_output, train)
    _write_jsonl(cfg.eval_output, eval_rows)
    report = {
        "config": asdict(cfg),
        "outputs": {"train": cfg.train_output, "eval": cfg.eval_output},
        "train": _summary(train),
        "eval": _summary(eval_rows),
        "contract": "same-signal listwise LONG/SHORT/NO_TRADE preference; prompt is signal-time only",
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
    p.add_argument("--min-trade-utility", type=float, default=EventCandidateListwisePreferenceCfg.min_trade_utility)
    p.add_argument("--min-pair-gap", type=float, default=EventCandidateListwisePreferenceCfg.min_pair_gap)
    p.add_argument("--max-pairs-per-signal", type=int, default=EventCandidateListwisePreferenceCfg.max_pairs_per_signal)
    p.add_argument("--numeric-keys", default=EventCandidateListwisePreferenceCfg.numeric_keys)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(EventCandidateListwisePreferenceCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
