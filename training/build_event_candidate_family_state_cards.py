"""Build LLM state-card records from pre-fold family scoreboards.

The model should reason over compact family-validity evidence, not raw OHLC
numbers.  Position state is explicit even when historical fold training is flat,
so the same schema can be filled by live Binance/wave_trading execution later.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FamilyStateCardConfig:
    selector_report: str
    output_jsonl: str
    max_options: int = 5
    include_abstain: bool = True
    split_name: str = "all"
    default_position_mode: str = "FLAT"
    fold_start: str = ""
    fold_end: str = ""
    randomize_options: bool = False
    random_seed: int = 17


def _num(x: Any, ndigits: int = 3) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return round(v, ndigits)


def _option_from_score(row: dict[str, Any], idx: int) -> dict[str, Any]:
    ev = row.get("evidence") or []
    metrics = ev[0].get("metrics", {}) if ev and isinstance(ev[0], dict) else {}
    return {
        "id": chr(ord("A") + idx),
        "family": row.get("family"),
        "pre_fold_score": _num(row.get("score")),
        "threshold": _num(row.get("threshold"), 6),
        "evidence_count": len(ev),
        "latest_evidence": {
            "fold": ev[0].get("fold") if ev else None,
            "distance": _num(ev[0].get("distance")) if ev else None,
            "raw_score": _num(ev[0].get("raw_score")) if ev else None,
            "weighted_score": _num(ev[0].get("weighted_score")) if ev else None,
            "cagr_to_mdd": _num(metrics.get("cagr_to_strict_mdd")),
            "trades": metrics.get("trade_entries"),
            "p_value": _num(metrics.get("p_value_mean_ret_approx")),
        },
    }


def _position_state(mode: str) -> dict[str, Any]:
    mode = str(mode or "FLAT").upper()
    return {
        "mode": mode,
        "side": "NONE" if mode == "FLAT" else "UNKNOWN",
        "size_pct": 0.0,
        "entry_price": None,
        "entry_time": None,
        "age_bars": 0,
        "unrealized_pnl_pct": 0.0,
        "source": "historical_fold_default; live runner must overwrite from exchange/wave_trading",
    }


def _assign_option_ids(options: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    letter_idx = 0
    for opt in options:
        opt = dict(opt)
        if opt.get("family") == "ABSTAIN":
            opt["id"] = "ABSTAIN"
        else:
            opt["id"] = chr(ord("A") + letter_idx)
            letter_idx += 1
        out.append(opt)
    return out


def _target_for_fold(fold: dict[str, Any], options: list[dict[str, Any]]) -> dict[str, Any]:
    if fold.get("abstained"):
        return {"choice_id": "ABSTAIN", "family": "ABSTAIN", "reason": "selector_abstained_pre_fold"}
    fam = fold.get("selected_family")
    for opt in options:
        if opt.get("family") == fam:
            return {"choice_id": opt["id"], "family": fam, "reason": "selector_selected_pre_fold"}
    return {"choice_id": "ABSTAIN", "family": "ABSTAIN", "reason": "selected_family_not_in_options"}


def _prompt(card: dict[str, Any]) -> str:
    lines = [
        "You are choosing a trading family for the next chronological fold.",
        "Use only the pre-fold evidence and current position state. Do not infer future outcomes.",
        "Return exactly one option id.",
        f"Fold: {card['fold']['name']} from {card['fold']['start']} to {card['fold']['end']}",
        f"Current position: {json.dumps(card['position_state'], ensure_ascii=False, sort_keys=True)}",
        "Options:",
    ]
    for opt in card["options"]:
        lines.append(json.dumps(opt, ensure_ascii=False, sort_keys=True))
    return "\n".join(lines)


def _in_fold_range(fold: dict[str, Any], cfg: FamilyStateCardConfig) -> bool:
    start = str((fold.get("fold") or {}).get("start", ""))
    if cfg.fold_start and start < str(cfg.fold_start):
        return False
    if cfg.fold_end and start >= str(cfg.fold_end):
        return False
    return True


def build_records(cfg: FamilyStateCardConfig) -> list[dict[str, Any]]:
    report = json.loads(Path(cfg.selector_report).read_text())
    records: list[dict[str, Any]] = []
    for fold in report.get("folds", []):
        if not _in_fold_range(fold, cfg):
            continue
        scoreboard = list(fold.get("pre_fold_scoreboard") or [])[: int(cfg.max_options)]
        options = [_option_from_score(row, i) for i, row in enumerate(scoreboard)]
        if cfg.include_abstain:
            options.append({"id": "ABSTAIN", "family": "ABSTAIN", "pre_fold_score": 0.0, "threshold": None, "evidence_count": 0, "latest_evidence": {}})
        if cfg.randomize_options:
            rng = random.Random(int(cfg.random_seed) + len(records))
            trade_options = [opt for opt in options if opt.get("family") != "ABSTAIN"]
            abstain_options = [opt for opt in options if opt.get("family") == "ABSTAIN"]
            rng.shuffle(trade_options)
            options = trade_options + abstain_options
        options = _assign_option_ids(options)
        card = {
            "split": cfg.split_name,
            "fold": fold.get("fold"),
            "selector_mode": fold.get("selector_mode"),
            "position_state": _position_state(cfg.default_position_mode),
            "options": options,
            "target": _target_for_fold(fold, options),
            "diagnostic_selected_metrics": fold.get("selected_metrics"),
            "leakage_guard": {
                "options_from_pre_fold_scoreboard": True,
                "position_state_is_signal_time_input": True,
                "target_fold_metrics_diagnostic_only": True,
                "option_order_randomized": bool(cfg.randomize_options),
            },
        }
        records.append({**card, "prompt": _prompt(card), "completion": card["target"]["choice_id"]})
    return records


def run(cfg: FamilyStateCardConfig) -> dict[str, Any]:
    rows = build_records(cfg)
    out = Path(cfg.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {"config": asdict(cfg), "rows": len(rows), "output_jsonl": str(out), "targets": {}}
    for row in rows:
        key = row["target"]["choice_id"]
        summary["targets"][key] = summary["targets"].get(key, 0) + 1
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selector-report", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--max-options", type=int, default=FamilyStateCardConfig.max_options)
    p.add_argument("--no-abstain", action="store_true")
    p.add_argument("--split-name", default=FamilyStateCardConfig.split_name)
    p.add_argument("--default-position-mode", default=FamilyStateCardConfig.default_position_mode)
    p.add_argument("--fold-start", default=FamilyStateCardConfig.fold_start, help="Inclusive fold start date filter, e.g. 2025-01-01")
    p.add_argument("--fold-end", default=FamilyStateCardConfig.fold_end, help="Exclusive fold start date filter, e.g. 2026-01-01")
    p.add_argument("--randomize-options", action="store_true", help="Shuffle non-ABSTAIN options and reassign letter ids to reduce position bias")
    p.add_argument("--random-seed", type=int, default=FamilyStateCardConfig.random_seed)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    print(json.dumps(run(FamilyStateCardConfig(selector_report=a.selector_report, output_jsonl=a.output_jsonl, max_options=a.max_options, include_abstain=not a.no_abstain, split_name=a.split_name, default_position_mode=a.default_position_mode, fold_start=a.fold_start, fold_end=a.fold_end, randomize_options=a.randomize_options, random_seed=a.random_seed)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
