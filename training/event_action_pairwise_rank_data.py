"""Build pairwise ranking rows from event-action value candidates.

The previous TAKE/SKIP value SFT is dominated by SKIP labels and failed full
2026 validation.  This builder reframes the same leak-safe candidate universe as
within-signal pairwise ranking: choose the candidate with higher realized strict
future utility.  Prompts keep only past-visible context plus two candidate action
specs; future utility is stored only in training labels/audits.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EventActionPairwiseRankConfig:
    input_jsonl: str
    output_jsonl: str
    summary_output: str = ""
    min_utility_gap: float = 0.002
    max_pairs_per_signal: int = 6
    include_same_side_pairs: bool = True
    include_cross_side_pairs: bool = True
    emit_swapped_duplicates: bool = False


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row.get("date")), int(row.get("signal_pos", -1) or -1))


def _utility(row: dict[str, Any]) -> float:
    audit = row.get("action_audit", {}) if isinstance(row.get("action_audit"), dict) else {}
    return float(audit.get("rank_utility", audit.get("utility", -1e9)) or -1e9)


def _net_return(row: dict[str, Any]) -> float:
    audit = row.get("action_audit", {}) if isinstance(row.get("action_audit"), dict) else {}
    return float(audit.get("net_return", 0.0) or 0.0)


def _candidate(row: dict[str, Any]) -> dict[str, Any]:
    action = row.get("action", {}) if isinstance(row.get("action"), dict) else {}
    return {
        "family": str(action.get("family", "UNKNOWN")),
        "side": str(action.get("side", "NONE")).upper(),
        "hold_bars": int(action.get("hold_bars", 0) or 0),
        "strength": action.get("strength"),
    }


def _base_context(prompt: str) -> str:
    lines = []
    for line in str(prompt).splitlines():
        if line.startswith("Output exactly one label:"):
            continue
        if line.startswith("TAKE only if"):
            continue
        if line.startswith("Do not output JSON"):
            continue
        if line.startswith("Candidate action:"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _pair_prompt(base_prompt: str, cand_a: dict[str, Any], cand_b: dict[str, Any]) -> str:
    return "\n".join(
        [
            "You are an action ranking judge for BTCUSDT futures.",
            "Use only the past-only state, the prompt-visible action book, and the two candidate actions.",
            "Choose which candidate has better expected net utility after path risk.",
            "Output exactly one label: A or B.",
            "",
            _base_context(base_prompt),
            f"Candidate A: {json.dumps(cand_a, sort_keys=True, separators=(',', ':'))}",
            f"Candidate B: {json.dumps(cand_b, sort_keys=True, separators=(',', ':'))}",
        ]
    )


def _allowed_pair(chosen: dict[str, Any], rejected: dict[str, Any], cfg: EventActionPairwiseRankConfig) -> bool:
    same_side = str(_candidate(chosen)["side"]) == str(_candidate(rejected)["side"])
    return (same_side and bool(cfg.include_same_side_pairs)) or ((not same_side) and bool(cfg.include_cross_side_pairs))


def build_pairwise_rows(rows: list[dict[str, Any]], cfg: EventActionPairwiseRankConfig) -> list[dict[str, Any]]:
    by_signal: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_signal[_key(row)].append(row)

    pairs: list[dict[str, Any]] = []
    for key, group in sorted(by_signal.items(), key=lambda kv: kv[0]):
        ranked = sorted(group, key=lambda r: (_utility(r), _net_return(r)), reverse=True)
        made = 0
        for i, chosen in enumerate(ranked):
            if made >= int(cfg.max_pairs_per_signal):
                break
            chosen_u = _utility(chosen)
            for rejected in reversed(ranked[i + 1 :]):
                if made >= int(cfg.max_pairs_per_signal):
                    break
                rejected_u = _utility(rejected)
                gap = chosen_u - rejected_u
                if gap < float(cfg.min_utility_gap):
                    continue
                if not _allowed_pair(chosen, rejected, cfg):
                    continue
                cand_chosen = _candidate(chosen)
                cand_rejected = _candidate(rejected)
                orientations = [(True, "deterministic_balanced")]
                if bool(cfg.emit_swapped_duplicates):
                    orientations = [(True, "swapped_duplicate"), (False, "swapped_duplicate")]
                elif made % 2 != 0:
                    orientations = [(False, "deterministic_balanced")]
                for chosen_is_a, orientation_mode in orientations:
                    cand_a = cand_chosen if chosen_is_a else cand_rejected
                    cand_b = cand_rejected if chosen_is_a else cand_chosen
                    pairs.append(
                        {
                            "task": "event_action_pairwise_rank",
                            "date": key[0],
                            "signal_pos": key[1],
                            "prompt": _pair_prompt(str(chosen.get("prompt", "")), cand_a, cand_b),
                            "target": "A" if chosen_is_a else "B",
                            "chosen_action": cand_chosen,
                            "rejected_action": cand_rejected,
                            "chosen_utility": chosen_u,
                            "rejected_utility": rejected_u,
                            "utility_gap": gap,
                            "orientation_mode": orientation_mode,
                            "chosen_action_audit": chosen.get("action_audit", {}),
                            "rejected_action_audit": rejected.get("action_audit", {}),
                            "leakage_guard": {
                                "prompt_uses_future_path": False,
                                "chosen_rejected_use_future_utility_for_training_only": True,
                                "candidate_book_uses_past_only_features": True,
                                "swapped_duplicate_pairing": bool(cfg.emit_swapped_duplicates),
                            },
                        }
                    )
                made += 1
    return pairs


def summarize_pairs(pairs: list[dict[str, Any]], cfg: EventActionPairwiseRankConfig) -> dict[str, Any]:
    targets = Counter(str(p["target"]) for p in pairs)
    orientations = Counter(str(p.get("orientation_mode", "")) for p in pairs)
    chosen_sides = Counter(str(p.get("chosen_action", {}).get("side")) for p in pairs)
    rejected_sides = Counter(str(p.get("rejected_action", {}).get("side")) for p in pairs)
    gaps = [float(p.get("utility_gap", 0.0) or 0.0) for p in pairs]
    signals = {(p.get("date"), p.get("signal_pos")) for p in pairs}
    prompt_lens = [len(str(p.get("prompt", ""))) for p in pairs]
    return {
        "input_jsonl": str(Path(cfg.input_jsonl).resolve()),
        "output_jsonl": cfg.output_jsonl,
        "pairs": len(pairs),
        "signals": len(signals),
        "target_counts": dict(sorted(targets.items())),
        "orientation_counts": dict(sorted(orientations.items())),
        "chosen_side_counts": dict(sorted(chosen_sides.items())),
        "rejected_side_counts": dict(sorted(rejected_sides.items())),
        "utility_gap": {
            "min": min(gaps) if gaps else 0.0,
            "max": max(gaps) if gaps else 0.0,
            "mean": sum(gaps) / max(1, len(gaps)),
        },
        "prompt_chars": {
            "min": min(prompt_lens) if prompt_lens else 0,
            "max": max(prompt_lens) if prompt_lens else 0,
            "mean": sum(prompt_lens) / max(1, len(prompt_lens)),
        },
        "config": asdict(cfg),
        "leakage_guard": {
            "prompts_are_past_only": True,
            "future_utility_only_in_targets_and_audits": True,
            "not_a_backtest_result": True,
        },
    }


def build_pairwise_jsonl(**kwargs: Any) -> dict[str, Any]:
    cfg = EventActionPairwiseRankConfig(**kwargs)
    pairs = build_pairwise_rows(read_jsonl(cfg.input_jsonl), cfg)
    write_jsonl(cfg.output_jsonl, pairs)
    summary = summarize_pairs(pairs, cfg)
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build event-action pairwise ranking rows from value candidates")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--summary-output", default="")
    p.add_argument("--min-utility-gap", type=float, default=0.002)
    p.add_argument("--max-pairs-per-signal", type=int, default=6)
    p.add_argument("--same-side-only", action="store_true")
    p.add_argument("--cross-side-only", action="store_true")
    p.add_argument("--emit-swapped-duplicates", action="store_true", help="Emit both A/B orientations for each selected semantic pair")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if a.same_side_only and a.cross_side_only:
        raise SystemExit("choose at most one of --same-side-only or --cross-side-only")
    print(
        json.dumps(
            build_pairwise_jsonl(
                input_jsonl=a.input_jsonl,
                output_jsonl=a.output_jsonl,
                summary_output=a.summary_output,
                min_utility_gap=a.min_utility_gap,
                max_pairs_per_signal=a.max_pairs_per_signal,
                include_same_side_pairs=not a.cross_side_only,
                include_cross_side_pairs=not a.same_side_only,
                emit_swapped_duplicates=bool(a.emit_swapped_duplicates),
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
