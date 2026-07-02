"""Build cleaner pairwise family cards from pre-fold options and target-fold diagnostics.

This is a supervised-data builder, not a live selector.  Prompts are restricted to
pre-fold scoreboards plus position state; target-fold diagnostics are used only to
choose training/evaluation labels and are kept out of the prompt.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from training.build_event_candidate_family_state_cards import _assign_option_ids, _num, _option_from_score, _position_state


@dataclass(frozen=True)
class CleanPairwiseFamilyCardConfig:
    selector_report: str
    output_jsonl: str
    split_name: str = "all"
    max_options: int = 5
    max_rejected_per_row: int = 3
    fold_start: str = ""
    fold_end: str = ""
    randomize_options: bool = True
    random_seed: int = 31
    default_position_mode: str = "FLAT"
    min_diagnostic_trades: int = 12
    min_diagnostic_ratio: float = 0.25
    min_diagnostic_cagr_pct: float = 0.0
    max_diagnostic_mdd_pct: float = 25.0
    trade_count_cap: int = 30
    include_abstain_pairs: bool = True
    augment_reverse_pairs: bool = True
    label_source: str = "target_fold_diagnostic_not_for_prompt"



def _pair_option_summary(opt: dict[str, Any], pair_id: str) -> dict[str, Any]:
    return {
        "id": pair_id,
        "source_option_id": opt.get("id"),
        "family": opt.get("family"),
        "pre_fold_score": opt.get("pre_fold_score"),
        "threshold": opt.get("threshold"),
        "evidence_count": opt.get("evidence_count"),
        "latest_evidence": opt.get("latest_evidence", {}),
    }

def _load_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _in_fold_range(fold: dict[str, Any], cfg: CleanPairwiseFamilyCardConfig) -> bool:
    start = str((fold.get("fold") or {}).get("start", ""))
    if cfg.fold_start and start < str(cfg.fold_start):
        return False
    if cfg.fold_end and start >= str(cfg.fold_end):
        return False
    return True


def _metric(diag: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        v = float((diag.get("metrics") or {}).get(key, default) or default)
    except Exception:
        return default
    if not math.isfinite(v):
        return default
    return v


def _trades(diag: dict[str, Any]) -> int:
    try:
        return int((diag.get("metrics") or {}).get("trade_entries", 0) or 0)
    except Exception:
        return 0


def _target_score(diag: dict[str, Any], cfg: CleanPairwiseFamilyCardConfig) -> float:
    ratio = _metric(diag, "cagr_to_strict_mdd")
    trades = _trades(diag)
    cagr = _metric(diag, "cagr_pct")
    mdd = _metric(diag, "strict_mdd_pct", 999.0)
    if trades < int(cfg.min_diagnostic_trades):
        return float("-inf")
    if ratio < float(cfg.min_diagnostic_ratio):
        return float("-inf")
    if cagr < float(cfg.min_diagnostic_cagr_pct):
        return float("-inf")
    if mdd > float(cfg.max_diagnostic_mdd_pct):
        return float("-inf")
    trade_weight = math.sqrt(min(trades, int(cfg.trade_count_cap)) / max(1, int(cfg.trade_count_cap)))
    # Prefer statistically usable, repeatable winners over one-trade cagr/mdd spikes.
    return ratio * trade_weight


def _diagnostic_summary(diag: dict[str, Any] | None) -> dict[str, Any] | None:
    if not diag:
        return None
    m = diag.get("metrics") or {}
    return {
        "family": diag.get("family"),
        "score_used_for_label": _num(diag.get("clean_label_score"), 6),
        "cagr_to_strict_mdd": _num(m.get("cagr_to_strict_mdd"), 6),
        "cagr_pct": _num(m.get("cagr_pct"), 3),
        "strict_mdd_pct": _num(m.get("strict_mdd_pct"), 3),
        "trade_entries": m.get("trade_entries"),
        "p_value_mean_ret_approx": _num(m.get("p_value_mean_ret_approx"), 6),
    }


def choose_clean_target(fold: dict[str, Any], options: list[dict[str, Any]], cfg: CleanPairwiseFamilyCardConfig) -> dict[str, Any]:
    option_by_family = {opt.get("family"): opt for opt in options}
    best: dict[str, Any] | None = None
    best_score = float("-inf")
    for diag in fold.get("top_fold_diagnostic_not_for_selection") or []:
        family = diag.get("family")
        if family not in option_by_family:
            continue
        score = _target_score(diag, cfg)
        if score > best_score:
            best_score = score
            best = dict(diag)
            best["clean_label_score"] = score
    if best is None or not math.isfinite(best_score):
        return {
            "choice_id": "ABSTAIN",
            "family": "ABSTAIN",
            "reason": "no_diagnostic_family_passed_clean_target_filters",
            "label_source": cfg.label_source,
            "diagnostic_target": None,
        }
    opt = option_by_family[best.get("family")]
    return {
        "choice_id": opt.get("id"),
        "family": best.get("family"),
        "reason": "best_target_fold_diagnostic_family_with_prefold_option",
        "label_source": cfg.label_source,
        "diagnostic_target": _diagnostic_summary(best),
    }


def _prompt(row: dict[str, Any], option_a: dict[str, Any], option_b: dict[str, Any]) -> str:
    payload = {
        "fold": row.get("fold"),
        "position_state": row.get("position_state"),
        "option_a": _pair_option_summary(option_a, "A"),
        "option_b": _pair_option_summary(option_b, "B"),
    }
    return "\n".join([
        "Choose which family option is more valid for the next chronological fold.",
        "Use only pre-fold evidence and current position state. Do not infer future outcomes.",
        "Target-fold outcomes are intentionally not shown.",
        "Answer exactly A or B.",
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
    ])


def _build_row(fold: dict[str, Any], cfg: CleanPairwiseFamilyCardConfig, row_idx: int) -> dict[str, Any]:
    scoreboard = list(fold.get("pre_fold_scoreboard") or [])[: int(cfg.max_options)]
    options = [_option_from_score(row, i) for i, row in enumerate(scoreboard)]
    options.append({"id": "ABSTAIN", "family": "ABSTAIN", "pre_fold_score": 0.0, "threshold": None, "evidence_count": 0, "latest_evidence": {}})
    if cfg.randomize_options:
        rng = random.Random(int(cfg.random_seed) + row_idx)
        trade_options = [opt for opt in options if opt.get("family") != "ABSTAIN"]
        rng.shuffle(trade_options)
        options = trade_options + [opt for opt in options if opt.get("family") == "ABSTAIN"]
    options = _assign_option_ids(options)
    target = choose_clean_target(fold, options, cfg)
    return {
        "split": cfg.split_name,
        "fold": fold.get("fold"),
        "selector_mode": fold.get("selector_mode"),
        "position_state": _position_state(cfg.default_position_mode),
        "options": options,
        "target": target,
        "leakage_guard": {
            "options_from_pre_fold_scoreboard": True,
            "position_state_is_signal_time_input": True,
            "target_fold_metrics_not_in_prompt": True,
            "target_fold_metrics_used_only_for_labeling": True,
            "option_order_randomized": bool(cfg.randomize_options),
            "eval_labels_must_not_be_used_for_model_selection": cfg.split_name.lower() in {"eval", "holdout"},
        },
    }


def build_state_rows(cfg: CleanPairwiseFamilyCardConfig) -> list[dict[str, Any]]:
    report = _load_report(cfg.selector_report)
    rows: list[dict[str, Any]] = []
    for fold in report.get("folds", []):
        if _in_fold_range(fold, cfg):
            rows.append(_build_row(fold, cfg, len(rows)))
    return rows


def build_records(cfg: CleanPairwiseFamilyCardConfig) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in build_state_rows(cfg):
        target_id = (row.get("target") or {}).get("choice_id")
        options = list(row.get("options") or [])
        chosen = next((o for o in options if o.get("id") == target_id), None)
        if chosen is None:
            continue
        rejected = [o for o in options if o.get("id") != target_id]
        if not cfg.include_abstain_pairs:
            rejected = [o for o in rejected if o.get("family") != "ABSTAIN"]
        rejected.sort(key=lambda o: float(o.get("pre_fold_score") or 0.0), reverse=True)
        for idx, neg in enumerate(rejected[: int(cfg.max_rejected_per_row)]):
            variants = [(chosen, neg, "A", "B", "chosen_as_a")]
            if cfg.augment_reverse_pairs:
                variants.append((neg, chosen, "B", "A", "chosen_as_b"))
            for option_a, option_b, chosen_response, rejected_response, order_variant in variants:
                records.append({
                    "split": row.get("split"),
                    "fold": row.get("fold"),
                    "position_state": row.get("position_state"),
                    "chosen": chosen_response,
                    "rejected": rejected_response,
                    "chosen_option": _pair_option_summary(chosen, chosen_response),
                    "rejected_option": _pair_option_summary(neg, rejected_response),
                    "option_a_family": option_a.get("family"),
                    "option_b_family": option_b.get("family"),
                    "target_family": (row.get("target") or {}).get("family"),
                    "target_reason": (row.get("target") or {}).get("reason"),
                    "label_source": (row.get("target") or {}).get("label_source"),
                    "diagnostic_target": (row.get("target") or {}).get("diagnostic_target"),
                    "pair_index": idx,
                    "order_variant": order_variant,
                    "prompt": _prompt(row, option_a, option_b),
                    "target": chosen_response,
                    "completion": chosen_response,
                    "chosen_response": chosen_response,
                    "rejected_response": rejected_response,
                    "leakage_guard": {**(row.get("leakage_guard") or {}), "order_augmented": bool(cfg.augment_reverse_pairs)},
                })
    return records


def run(cfg: CleanPairwiseFamilyCardConfig) -> dict[str, Any]:
    rows = build_records(cfg)
    out = Path(cfg.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    fams = Counter(row["target_family"] for row in rows)
    folds = Counter((row.get("fold") or {}).get("name") for row in rows)
    return {
        "config": asdict(cfg),
        "rows": len(rows),
        "folds": len(folds),
        "output_jsonl": str(out),
        "target_families": dict(fams),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selector-report", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--split-name", default=CleanPairwiseFamilyCardConfig.split_name)
    p.add_argument("--max-options", type=int, default=CleanPairwiseFamilyCardConfig.max_options)
    p.add_argument("--max-rejected-per-row", type=int, default=CleanPairwiseFamilyCardConfig.max_rejected_per_row)
    p.add_argument("--fold-start", default=CleanPairwiseFamilyCardConfig.fold_start)
    p.add_argument("--fold-end", default=CleanPairwiseFamilyCardConfig.fold_end)
    p.add_argument("--no-randomize-options", action="store_true")
    p.add_argument("--random-seed", type=int, default=CleanPairwiseFamilyCardConfig.random_seed)
    p.add_argument("--default-position-mode", default=CleanPairwiseFamilyCardConfig.default_position_mode)
    p.add_argument("--min-diagnostic-trades", type=int, default=CleanPairwiseFamilyCardConfig.min_diagnostic_trades)
    p.add_argument("--min-diagnostic-ratio", type=float, default=CleanPairwiseFamilyCardConfig.min_diagnostic_ratio)
    p.add_argument("--min-diagnostic-cagr-pct", type=float, default=CleanPairwiseFamilyCardConfig.min_diagnostic_cagr_pct)
    p.add_argument("--max-diagnostic-mdd-pct", type=float, default=CleanPairwiseFamilyCardConfig.max_diagnostic_mdd_pct)
    p.add_argument("--trade-count-cap", type=int, default=CleanPairwiseFamilyCardConfig.trade_count_cap)
    p.add_argument("--exclude-abstain-pairs", action="store_true")
    p.add_argument("--no-reverse-pairs", action="store_true", help="Disable A/B order augmentation")
    p.add_argument("--label-source", default=CleanPairwiseFamilyCardConfig.label_source)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    cfg = CleanPairwiseFamilyCardConfig(
        selector_report=a.selector_report,
        output_jsonl=a.output_jsonl,
        split_name=a.split_name,
        max_options=a.max_options,
        max_rejected_per_row=a.max_rejected_per_row,
        fold_start=a.fold_start,
        fold_end=a.fold_end,
        randomize_options=not a.no_randomize_options,
        random_seed=a.random_seed,
        default_position_mode=a.default_position_mode,
        min_diagnostic_trades=a.min_diagnostic_trades,
        min_diagnostic_ratio=a.min_diagnostic_ratio,
        min_diagnostic_cagr_pct=a.min_diagnostic_cagr_pct,
        max_diagnostic_mdd_pct=a.max_diagnostic_mdd_pct,
        trade_count_cap=a.trade_count_cap,
        include_abstain_pairs=not a.exclude_abstain_pairs,
        augment_reverse_pairs=not a.no_reverse_pairs,
        label_source=a.label_source,
    )
    print(json.dumps(run(cfg), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
