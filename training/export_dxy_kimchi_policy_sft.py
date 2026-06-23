"""Export DXY/Kimchi prior policy rows to balanced chat SFT splits."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExportDxyKimchiSftCfg:
    input_jsonl: str
    train_output: str
    test_output: str
    eval_output: str
    summary_output: str
    system_prompt: str = "You are a compact BTCUSDT futures RLLM policy. Return exactly one valid compact JSON object."
    no_trade_per_activate: float = 3.0
    balance_mode: str = "standard"
    seed: int = 42


def _load(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _target_obj(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(str(row.get("target", "{}")))


def _bucket(row: dict[str, Any]) -> str:
    obj = _target_obj(row)
    if bool(obj.get("activate")):
        return f"activate_{obj.get('action')}"
    return str(obj.get("reason_code", "no_trade"))


def _prior_side(row: dict[str, Any]) -> str:
    signal = row.get("prior_signal", {})
    if isinstance(signal, dict):
        side = str(signal.get("side", "NONE")).upper()
    else:
        side = "NONE"
    return side if side in {"LONG", "SHORT"} else "NONE"


def _message_row(row: dict[str, Any], cfg: ExportDxyKimchiSftCfg) -> dict[str, Any]:
    target = str(row["target"])
    prompt = str(row["prompt"])
    return {
        "task": row.get("task"),
        "split": row.get("split"),
        "date": row.get("date"),
        "signal_pos": row.get("signal_pos"),
        "prompt": prompt,
        "target": target,
        "messages": [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ],
        "metadata": {
            "task": row.get("task"),
            "split": row.get("split"),
            "date": row.get("date"),
            "signal_pos": row.get("signal_pos"),
            "prior_signal": row.get("prior_signal", {}),
            "target": _target_obj(row),
            "leakage_guard": row.get("leakage_guard", {}),
        },
    }


def _balanced_train(rows: list[dict[str, Any]], *, no_trade_per_activate: float, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(int(seed))
    active = [r for r in rows if bool(_target_obj(r).get("activate"))]
    inactive_by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        obj = _target_obj(row)
        if not bool(obj.get("activate")):
            inactive_by_reason[str(obj.get("reason_code", "no_trade"))].append(row)
    selected = list(active)
    inactive_budget = int(round(len(active) * float(no_trade_per_activate))) if active else len(rows)
    reasons = sorted(inactive_by_reason)
    per_reason = max(1, inactive_budget // max(1, len(reasons))) if inactive_budget else 0
    for reason in reasons:
        pool = list(inactive_by_reason[reason])
        rng.shuffle(pool)
        selected.extend(pool[: min(per_reason, len(pool))])
    if len(selected) < len(active) + inactive_budget:
        used = {id(r) for r in selected}
        leftovers = [r for rows_for_reason in inactive_by_reason.values() for r in rows_for_reason if id(r) not in used]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: len(active) + inactive_budget - len(selected)])
    return sorted(selected, key=lambda r: (str(r.get("date")), int(r.get("signal_pos", 0) or 0)))


def _side_contrast_train(rows: list[dict[str, Any]], *, no_prior_per_prior_row: float, seed: int) -> list[dict[str, Any]]:
    """Keep side-specific prior examples and only subsample no-prior abstentions.

    Plain NO_TRADE oversampling collapsed the policy into majority-class
    abstention.  This selector preserves all rows where the train-fitted prior
    fired (both accepted and rejected LONG/SHORT) so the model can learn
    side-specific rejection boundaries.
    """
    rng = random.Random(int(seed))
    prior_rows = [r for r in rows if _prior_side(r) in {"LONG", "SHORT"}]
    no_prior_rows = [r for r in rows if _prior_side(r) == "NONE"]
    rng.shuffle(no_prior_rows)
    no_prior_budget = int(round(len(prior_rows) * max(0.0, float(no_prior_per_prior_row))))
    selected = list(prior_rows) + no_prior_rows[: min(no_prior_budget, len(no_prior_rows))]
    return sorted(selected, key=lambda r: (str(r.get("date")), int(r.get("signal_pos", 0) or 0)))



def _side_bucket(row: dict[str, Any]) -> str:
    target = _target_obj(row)
    side = _prior_side(row)
    if bool(target.get("activate")) and str(target.get("action")) in {"LONG", "SHORT"}:
        return f"active_{target['action']}"
    if side in {"LONG", "SHORT"} and str(target.get("reason_code")) == "prior_signal_path_reward_rejected":
        return f"rejected_{side}"
    return "no_prior"


def _side_contrast_oversample_train(rows: list[dict[str, Any]], *, no_prior_per_bucket: float, seed: int) -> list[dict[str, Any]]:
    """Balance accepted/rejected prior-side buckets with replacement.

    The train split has fewer active SHORT rows than LONG/rejected rows. Without
    replacement the adapter learned a conservative LONG-only policy. This mode
    intentionally duplicates scarce train rows, while test/eval remain untouched.
    """
    rng = random.Random(int(seed))
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[_side_bucket(row)].append(row)
    side_keys = ["active_LONG", "active_SHORT", "rejected_LONG", "rejected_SHORT"]
    target_n = max((len(buckets[k]) for k in side_keys), default=0)
    selected: list[dict[str, Any]] = []
    for key in side_keys:
        pool = list(buckets.get(key, []))
        if not pool:
            continue
        selected.extend(pool)
        selected.extend(rng.choice(pool) for _ in range(max(0, target_n - len(pool))))
    no_prior_pool = list(buckets.get("no_prior", []))
    rng.shuffle(no_prior_pool)
    no_prior_n = min(len(no_prior_pool), int(round(target_n * max(0.0, float(no_prior_per_bucket)))))
    selected.extend(no_prior_pool[:no_prior_n])
    return sorted(selected, key=lambda r: (str(r.get("date")), int(r.get("signal_pos", 0) or 0), json.dumps(_target_obj(r), sort_keys=True)))

def _write(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    targets = [_target_obj(r["metadata"] if "messages" in r else r) if False else r.get("metadata", {}).get("target", _target_obj(r)) for r in rows]
    action_counts = Counter(str(t.get("action")) for t in targets)
    activate_counts = Counter(str(bool(t.get("activate"))) for t in targets)
    reason_counts = Counter(str(t.get("reason_code")) for t in targets)
    lens = [sum(len(m["content"]) for m in r.get("messages", [])) for r in rows]
    return {
        "rows": len(rows),
        "action_counts": dict(sorted(action_counts.items())),
        "activate_counts": dict(sorted(activate_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "chars": {"min": min(lens) if lens else 0, "max": max(lens) if lens else 0, "mean": sum(lens) / max(1, len(lens))},
    }


def run(cfg: ExportDxyKimchiSftCfg) -> dict[str, Any]:
    rows = [r for r in _load(cfg.input_jsonl) if str(r.get("task")) == "dxy_kimchi_regime_policy_sft"]
    train_raw = [r for r in rows if r.get("split") == "train"]
    test_raw = [r for r in rows if r.get("split") == "test"]
    eval_raw = [r for r in rows if r.get("split") == "eval"]
    mode = str(cfg.balance_mode).lower()
    if mode == "standard":
        train_balanced = _balanced_train(train_raw, no_trade_per_activate=float(cfg.no_trade_per_activate), seed=int(cfg.seed))
    elif mode == "side_contrast":
        train_balanced = _side_contrast_train(train_raw, no_prior_per_prior_row=float(cfg.no_trade_per_activate), seed=int(cfg.seed))
    elif mode == "side_contrast_oversample":
        train_balanced = _side_contrast_oversample_train(train_raw, no_prior_per_bucket=float(cfg.no_trade_per_activate), seed=int(cfg.seed))
    else:
        raise ValueError("balance_mode must be one of {'standard','side_contrast','side_contrast_oversample'}")
    train = [_message_row(r, cfg) for r in train_balanced]
    test = [_message_row(r, cfg) for r in test_raw]
    eval_rows = [_message_row(r, cfg) for r in eval_raw]
    _write(cfg.train_output, train)
    _write(cfg.test_output, test)
    _write(cfg.eval_output, eval_rows)
    report = {
        "config": asdict(cfg),
        "raw_counts": {"train": len(train_raw), "test": len(test_raw), "eval": len(eval_raw)},
        "balance_mode": mode,
        "train": _summary(train),
        "test": _summary(test),
        "eval": _summary(eval_rows),
        "outputs": {"train": cfg.train_output, "test": cfg.test_output, "eval": cfg.eval_output},
        "leakage_guard": {
            "train_balancing_uses_train_split_only": True,
            "test_eval_are_untouched_chronological_splits": True,
            "assistant_content_is_target_json_only": True,
        },
    }
    Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export DXY/Kimchi policy rows to balanced chat SFT splits")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--train-output", required=True)
    p.add_argument("--test-output", required=True)
    p.add_argument("--eval-output", required=True)
    p.add_argument("--summary-output", required=True)
    p.add_argument("--system-prompt", default=ExportDxyKimchiSftCfg.system_prompt)
    p.add_argument("--no-trade-per-activate", type=float, default=ExportDxyKimchiSftCfg.no_trade_per_activate)
    p.add_argument("--balance-mode", choices=["standard", "side_contrast", "side_contrast_oversample"], default=ExportDxyKimchiSftCfg.balance_mode)
    p.add_argument("--seed", type=int, default=ExportDxyKimchiSftCfg.seed)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(ExportDxyKimchiSftCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
