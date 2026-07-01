"""Export wave text-state rows as A/B option-choice take/skip policy rows."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WaveStateOptionChoiceCfg:
    train_jsonl: str
    eval_jsonl: str
    train_output: str
    eval_output: str
    summary_output: str
    take_full_only: bool = False


def _read(path: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _reward_pct(row: dict[str, Any]) -> float:
    return float(dict(row.get("reward") or {}).get("trade_ret_pct", 0.0) or 0.0)


def _decision(row: dict[str, Any]) -> str:
    target = row.get("target") if isinstance(row.get("target"), dict) else {}
    return str(target.get("decision", "ABSTAIN"))


def _target(row: dict[str, Any], *, take_full_only: bool) -> str:
    dec = _decision(row)
    if take_full_only:
        return "A" if dec == "TAKE_FULL" else "B"
    return "A" if dec in {"TAKE_FULL", "TAKE_SMALL"} else "B"


def _prompt(row: dict[str, Any]) -> str:
    base = str(row.get("prompt", ""))
    # Preserve the original state card but replace the JSON instruction with a
    # single-token option-choice contract for stable next-token logprob scoring.
    marker = "Return JSON with decision in {TAKE_FULL, TAKE_SMALL, ABSTAIN} and a short risk reason."
    if marker in base:
        base = base.replace(marker, "")
    lines = [
        "Task: decide whether this BTCUSDT futures candidate should be traded.",
        "Use only the state card below. Answer exactly one letter: A or B.",
        "A = TAKE_TRADE",
        "B = SKIP_TRADE",
        "",
        base.strip(),
    ]
    return "\n".join(x for x in lines if x != "")


def _row(row: dict[str, Any], cfg: WaveStateOptionChoiceCfg) -> dict[str, Any]:
    target = _target(row, take_full_only=bool(cfg.take_full_only))
    reward_pct = _reward_pct(row)
    return {
        "task": "wave_state_take_skip_option_choice",
        "date": row.get("date"),
        "signal_pos": row.get("signal_pos"),
        "side": row.get("side"),
        "prompt": _prompt(row),
        "target": target,
        "choice_utility": {"A": reward_pct, "B": 0.0},
        "source": {
            "target_decision": _decision(row),
            "reward": row.get("reward"),
            "candidate": row.get("candidate"),
            "state_tokens": row.get("state_tokens"),
        },
        "leakage_guard": {
            "prompt_uses_future_reward": False,
            "target_uses_future_reward_for_training_only": True,
            "option_A_is_take_trade": True,
            "option_B_is_skip_trade": True,
        },
    }


def _convert(rows: list[dict[str, Any]], cfg: WaveStateOptionChoiceCfg) -> list[dict[str, Any]]:
    return [_row(r, cfg) for r in rows]


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(r.get("target")) for r in rows)
    rewards = [float(r["choice_utility"]["A"]) for r in rows]
    return {
        "rows": len(rows),
        "target_counts": dict(sorted(counts.items())),
        "take_mean_reward_pct": sum(rewards) / max(1, len(rewards)),
        "positive_reward_rate": sum(1 for r in rewards if r > 0.0) / max(1, len(rewards)),
    }


def run(cfg: WaveStateOptionChoiceCfg) -> dict[str, Any]:
    train = _convert(_read(cfg.train_jsonl), cfg)
    ev = _convert(_read(cfg.eval_jsonl), cfg)
    _write(cfg.train_output, train)
    _write(cfg.eval_output, ev)
    report = {
        "config": asdict(cfg),
        "outputs": {"train": cfg.train_output, "eval": cfg.eval_output},
        "train": _summary(train),
        "eval": _summary(ev),
        "contract": "A/B take-skip option choice; prompt is signal-time only; reward is label/audit only",
    }
    Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--train-output", required=True)
    p.add_argument("--eval-output", required=True)
    p.add_argument("--summary-output", required=True)
    p.add_argument("--take-full-only", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(WaveStateOptionChoiceCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
