"""Relabel episode survival SFT rows with train-fitted utility thresholds.

The v1 binary survival label was too noisy for ranking.  This converter keeps
prompts unchanged but makes TRADE a high-utility class based on train quantiles.
"""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RelabelCfg:
    train_jsonl: str
    test_jsonl: str
    eval_jsonl: str
    output_dir: str
    utility_quantile: float = 0.70
    min_net_pct: float = 0.25
    max_mae_pct: float = 2.0
    min_mfe_to_mae: float = 1.25
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


def _threshold(train: list[dict[str, Any]], cfg: RelabelCfg) -> float:
    vals = [float((r.get("target_audit") or {}).get("utility_pct", -1e9) or -1e9) for r in train]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.quantile(vals, float(cfg.utility_quantile))) if vals else 0.0


def _decision(a: dict[str, Any], threshold: float, cfg: RelabelCfg) -> tuple[str, str, str]:
    util = float(a.get("utility_pct", -1e9) or -1e9)
    net = float(a.get("net_pct", -1e9) or -1e9)
    mae = float(a.get("mae_pct", 1e9) or 1e9)
    ratio = float(a.get("mfe_to_mae", 0.0) or 0.0)
    if util >= threshold and net >= float(cfg.min_net_pct) and mae <= float(cfg.max_mae_pct) and ratio >= float(cfg.min_mfe_to_mae):
        return "TRADE", "HIGH" if util >= threshold * 1.5 else "MID", "top_train_utility_survival"
    if mae > float(cfg.max_mae_pct):
        return "NO_TRADE", "LOW", "adverse_excursion_too_large"
    if net < float(cfg.min_net_pct):
        return "NO_TRADE", "LOW", "net_edge_too_small"
    if ratio < float(cfg.min_mfe_to_mae):
        return "NO_TRADE", "LOW", "favorable_excursion_not_enough"
    return "NO_TRADE", "LOW", "utility_below_train_top_quantile"


def _convert(rows: list[dict[str, Any]], threshold: float, cfg: RelabelCfg) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        r = dict(row)
        a = dict(r.get("target_audit") or {})
        dec, conf, reason = _decision(a, threshold, cfg)
        a["utility_threshold_pct"] = threshold
        a["decision"] = dec
        a["confidence"] = conf
        a["reason"] = reason
        r["target_audit"] = a
        r["target"] = json.dumps({"decision": dec, "confidence": conf, "reason": reason}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        r["task"] = "episode_survival_utility_rank_sft"
        out.append(r)
    return out


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dec = {"TRADE": 0, "NO_TRADE": 0}
    util = []
    for r in rows:
        d = json.loads(r["target"])["decision"]
        dec[d] = dec.get(d, 0) + 1
        if d == "TRADE":
            util.append(float((r.get("target_audit") or {}).get("utility_pct", 0.0) or 0.0))
    return {"rows": len(rows), "decisions": dec, "trade_mean_utility_pct": float(np.mean(util)) if util else 0.0}


def run(cfg: RelabelCfg) -> dict[str, Any]:
    train = _load(cfg.train_jsonl)
    test = _load(cfg.test_jsonl)
    ev = _load(cfg.eval_jsonl)
    th = _threshold(train, cfg)
    outs = {"train": _convert(train, th, cfg), "test": _convert(test, th, cfg), "eval": _convert(ev, th, cfg)}
    out_dir = Path(cfg.output_dir)
    suffix = ".jsonl.gz" if cfg.gzip_output else ".jsonl"
    report = {"config": asdict(cfg), "utility_threshold_pct": th, "splits": {}}
    for split, rows in outs.items():
        path = out_dir / f"episode_survival_utility_{split}{suffix}"
        _write(path, rows, bool(cfg.gzip_output))
        report["splits"][split] = {**_summary(rows), "output": str(path)}
    sp = out_dir / "episode_survival_utility_summary.json"
    sp.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--test-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--utility-quantile", type=float, default=RelabelCfg.utility_quantile)
    p.add_argument("--min-net-pct", type=float, default=RelabelCfg.min_net_pct)
    p.add_argument("--max-mae-pct", type=float, default=RelabelCfg.max_mae_pct)
    p.add_argument("--min-mfe-to-mae", type=float, default=RelabelCfg.min_mfe_to_mae)
    p.add_argument("--no-gzip-output", dest="gzip_output", action="store_false")
    p.set_defaults(gzip_output=RelabelCfg.gzip_output)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(RelabelCfg(**vars(parse_args()))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
