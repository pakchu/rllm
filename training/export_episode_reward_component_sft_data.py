"""Export reward-component SFT rows from episode survival candidates.

The direct TRADE/NO_TRADE and pairwise A/B targets were too brittle for Gemma.
This exporter keeps causal clause prompts but asks the model to predict decomposed
future path components.  Component bucket thresholds are fit on train only.
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from training.alpha_linear_combo_scan import _load_market
from training.export_episode_survival_pairwise_data import _candidate_clauses, _market_clauses


@dataclass(frozen=True)
class RewardComponentSftCfg:
    train_jsonl: str
    test_jsonl: str
    eval_jsonl: str
    market_csv: str
    output_dir: str
    gzip_output: bool = True


def _open(path: str, mode: str = "rt"):
    return gzip.open(path, mode, encoding="utf-8") if str(path).endswith(".gz") else open(path, mode, encoding="utf-8")


def _load(path: str) -> list[dict[str, Any]]:
    with _open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write(path: Path, rows: list[dict[str, Any]], gzip_output: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if gzip_output else open
    with opener(path, "wt", encoding="utf-8") as f:  # type: ignore[arg-type]
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _extract_json_line(prompt: str, key: str) -> dict[str, Any]:
    prefix = f"{key}: "
    for line in str(prompt).splitlines():
        if line.startswith(prefix):
            return json.loads(line[len(prefix):])
    return {}


def _candidate_view(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate": row.get("candidate") or _extract_json_line(str(row.get("prompt", "")), "candidate"),
        "setup_quality": _extract_json_line(str(row.get("prompt", "")), "setup_quality"),
        "macro_context": _extract_json_line(str(row.get("prompt", "")), "macro_context"),
    }


def _train_cuts(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    cuts: dict[str, list[float]] = {}
    for key in ("net_pct", "mae_pct", "mfe_pct", "utility_pct", "mfe_to_mae"):
        vals = np.asarray([float((r.get("target_audit") or {}).get(key, 0.0) or 0.0) for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        cuts[key] = [float(x) for x in np.quantile(vals, [1/3, 2/3])] if len(vals) else [0.0, 0.0]
    return cuts


def _bucket(value: float, cuts: list[float], labels: tuple[str, str, str]) -> str:
    if not np.isfinite(value):
        return "UNKNOWN"
    if value <= cuts[0]:
        return labels[0]
    if value <= cuts[1]:
        return labels[1]
    return labels[2]


def _target(audit: dict[str, Any], cuts: dict[str, list[float]]) -> dict[str, str]:
    net = float(audit.get("net_pct", 0.0) or 0.0)
    mae = float(audit.get("mae_pct", 0.0) or 0.0)
    mfe = float(audit.get("mfe_pct", 0.0) or 0.0)
    util = float(audit.get("utility_pct", 0.0) or 0.0)
    ratio = float(audit.get("mfe_to_mae", 0.0) or 0.0)
    net_b = _bucket(net, cuts["net_pct"], ("NET_WEAK", "NET_MID", "NET_STRONG"))
    mae_b = _bucket(mae, cuts["mae_pct"], ("ADVERSE_LOW", "ADVERSE_MID", "ADVERSE_HIGH"))
    mfe_b = _bucket(mfe, cuts["mfe_pct"], ("FAVORABLE_LOW", "FAVORABLE_MID", "FAVORABLE_HIGH"))
    util_b = _bucket(util, cuts["utility_pct"], ("UTILITY_LOW", "UTILITY_MID", "UTILITY_HIGH"))
    ratio_b = _bucket(ratio, cuts["mfe_to_mae"], ("PAYOFF_POOR", "PAYOFF_MID", "PAYOFF_GOOD"))
    if mae_b == "ADVERSE_HIGH":
        shape = "HIGH_ADVERSE_PATH"
    elif net_b == "NET_STRONG" and util_b == "UTILITY_HIGH" and mae_b != "ADVERSE_HIGH":
        shape = "CLEAN_WIN_PATH"
    elif mfe_b == "FAVORABLE_HIGH" and net_b == "NET_WEAK":
        shape = "FAILED_FOLLOW_THROUGH"
    elif net_b == "NET_WEAK" and util_b == "UTILITY_LOW":
        shape = "LOW_EDGE_PATH"
    else:
        shape = "MIXED_PATH"
    return {
        "net_bucket": net_b,
        "mae_bucket": mae_b,
        "mfe_bucket": mfe_b,
        "mfe_to_mae_bucket": ratio_b,
        "utility_bucket": util_b,
        "path_shape": shape,
    }


def _prompt(date: str, history: dict[str, Any], view: dict[str, Any]) -> str:
    lines = [
        "You are a BTCUSDT futures reward-component analyst.",
        "Use only causal price-action/setup/macro clauses.",
        "Predict future path component buckets, not a direct trade/no-trade decision.",
        "Return JSON with keys: net_bucket, mae_bucket, mfe_bucket, mfe_to_mae_bucket, utility_bucket, path_shape.",
        f"date: {date}",
        "market_regime:",
    ]
    lines.extend(f"- {clause}" for clause in _market_clauses(history))
    lines.append("candidate:")
    lines.extend(f"- {clause}" for clause in _candidate_clauses("candidate", view))
    return "\n".join(lines)


def _history_context(market, pos: int) -> dict[str, Any]:
    # Import lazily to reuse the tested causal history implementation without
    # making this exporter depend on pairwise row construction.
    from training.export_episode_survival_pairwise_data import _history_context as history_context

    return history_context(market, pos)


def _convert(rows: list[dict[str, Any]], market, cuts: dict[str, list[float]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        pos = int(r.get("signal_pos", -1))
        if pos < 0 or pos >= len(market):
            continue
        audit = dict(r.get("target_audit") or {})
        tgt = _target(audit, cuts)
        view = _candidate_view(r)
        date = str(r.get("date") or market.iloc[pos]["date"])
        out.append({
            "task": "episode_reward_component_sft",
            "date": date,
            "signal_pos": pos,
            "candidate": view.get("candidate") or {},
            "prompt": _prompt(date, _history_context(market, pos), view),
            "target": json.dumps(tgt, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
            "target_audit": audit,
            "leakage_guard": {
                "prompt_uses_future_path": False,
                "target_uses_future_path_for_training_only": True,
                "component_thresholds_fit_on_train_only": True,
            },
        })
    out.sort(key=lambda x: (str(x["date"]), int(x["signal_pos"]), str(x["target"])))
    return out


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("net_bucket", "mae_bucket", "mfe_bucket", "mfe_to_mae_bucket", "utility_bucket", "path_shape")
    counts = {k: Counter() for k in keys}
    prompt_lens = []
    for r in rows:
        t = json.loads(r["target"])
        prompt_lens.append(len(r["prompt"]))
        for k in keys:
            counts[k][str(t.get(k))] += 1
    return {
        "rows": len(rows),
        "target_counts": {k: dict(v) for k, v in counts.items()},
        "prompt_chars": {
            "min": int(min(prompt_lens)) if prompt_lens else 0,
            "mean": float(np.mean(prompt_lens)) if prompt_lens else 0.0,
            "max": int(max(prompt_lens)) if prompt_lens else 0,
        },
    }


def run(cfg: RewardComponentSftCfg) -> dict[str, Any]:
    market = _load_market(cfg.market_csv)
    loaded = {"train": _load(cfg.train_jsonl), "test": _load(cfg.test_jsonl), "eval": _load(cfg.eval_jsonl)}
    cuts = _train_cuts(loaded["train"])
    out_dir = Path(cfg.output_dir)
    suffix = ".jsonl.gz" if cfg.gzip_output else ".jsonl"
    report: dict[str, Any] = {"config": asdict(cfg), "train_component_cuts": cuts, "splits": {}}
    for split, rows in loaded.items():
        converted = _convert(rows, market, cuts)
        path = out_dir / f"episode_reward_components_{split}{suffix}"
        _write(path, converted, bool(cfg.gzip_output))
        report["splits"][split] = {**_summary(converted), "source_rows": len(rows), "output": str(path)}
    sp = out_dir / "episode_reward_components_summary.json"
    sp.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--test-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--no-gzip-output", dest="gzip_output", action="store_false")
    p.set_defaults(gzip_output=RewardComponentSftCfg.gzip_output)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(RewardComponentSftCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
