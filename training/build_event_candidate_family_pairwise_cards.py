"""Build pairwise chosen/rejected records from family state cards.

This converts listwise state-card options into contrastive examples so an LLM can
learn comparative family validity without relying on fixed option positions.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PairwiseFamilyCardConfig:
    input_jsonl: str
    output_jsonl: str
    max_rejected_per_row: int = 3
    include_abstain_pairs: bool = True
    min_score_gap: float = 0.0


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _option_summary(opt: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": opt.get("id"),
        "family": opt.get("family"),
        "pre_fold_score": opt.get("pre_fold_score"),
        "threshold": opt.get("threshold"),
        "evidence_count": opt.get("evidence_count"),
        "latest_evidence": opt.get("latest_evidence", {}),
    }


def _prompt(row: dict[str, Any], chosen: dict[str, Any], rejected: dict[str, Any]) -> str:
    card = {
        "fold": row.get("fold"),
        "position_state": row.get("position_state"),
        "option_a": _option_summary(chosen),
        "option_b": _option_summary(rejected),
    }
    return "\n".join([
        "Choose which family option is more valid for the next chronological fold.",
        "Use only pre-fold evidence and current position state. Do not infer future outcomes.",
        "Answer exactly A or B.",
        json.dumps(card, ensure_ascii=False, sort_keys=True),
    ])


def _score(opt: dict[str, Any]) -> float:
    try:
        return float(opt.get("pre_fold_score", 0.0) or 0.0)
    except Exception:
        return 0.0


def build_records(cfg: PairwiseFamilyCardConfig) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in _load_jsonl(cfg.input_jsonl):
        target = row.get("target", {})
        target_id = target.get("choice_id")
        options = list(row.get("options") or [])
        chosen = next((o for o in options if o.get("id") == target_id), None)
        if chosen is None:
            continue
        rejected = [o for o in options if o.get("id") != target_id]
        if not cfg.include_abstain_pairs:
            rejected = [o for o in rejected if o.get("family") != "ABSTAIN"]
        rejected.sort(key=_score, reverse=True)
        kept = []
        for opt in rejected:
            if _score(chosen) - _score(opt) < float(cfg.min_score_gap) and target_id != "ABSTAIN":
                continue
            kept.append(opt)
            if len(kept) >= int(cfg.max_rejected_per_row):
                break
        for idx, neg in enumerate(kept):
            prompt = _prompt(row, chosen, neg)
            out.append({
                "split": row.get("split"),
                "fold": row.get("fold"),
                "position_state": row.get("position_state"),
                "chosen": _option_summary(chosen),
                "rejected": _option_summary(neg),
                "target_family": target.get("family"),
                "target_reason": target.get("reason"),
                "pair_index": idx,
                "prompt": prompt,
                "completion": "A",
                "chosen_response": "A",
                "rejected_response": "B",
                "leakage_guard": {
                    "source_options_from_pre_fold_scoreboard": True,
                    "position_state_included": True,
                    "target_fold_metrics_not_in_prompt": True,
                },
            })
    return out


def run(cfg: PairwiseFamilyCardConfig) -> dict[str, Any]:
    rows = build_records(cfg)
    out = Path(cfg.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    fams = Counter(row["target_family"] for row in rows)
    return {"config": asdict(cfg), "rows": len(rows), "output_jsonl": str(out), "target_families": dict(fams)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--max-rejected-per-row", type=int, default=PairwiseFamilyCardConfig.max_rejected_per_row)
    p.add_argument("--exclude-abstain-pairs", action="store_true")
    p.add_argument("--min-score-gap", type=float, default=PairwiseFamilyCardConfig.min_score_gap)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    print(json.dumps(run(PairwiseFamilyCardConfig(input_jsonl=a.input_jsonl, output_jsonl=a.output_jsonl, max_rejected_per_row=a.max_rejected_per_row, include_abstain_pairs=not a.exclude_abstain_pairs, min_score_gap=a.min_score_gap)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
