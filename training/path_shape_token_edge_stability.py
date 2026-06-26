"""Audit split-to-split stability of path-shape prompt tokens.

This is a leakage-safe diagnostic: each split's future-derived labels are only
used to summarize that split after the split already exists. It does not select
a deployable policy from eval.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training.path_shape_token_policy_tte import _load, tokens_from_row


@dataclass(frozen=True)
class EdgeStabilityCfg:
    train_jsonl: str
    val_jsonl: str
    eval_jsonl: str
    output: str
    min_count: int = 12
    unit_mode: str = "token"  # token | semantic
    exclude_regex: str = ""
    top_n: int = 50


def _semantic_unit(tok: str) -> str:
    if "=" not in tok:
        return tok
    key, val = tok.split("=", 1)
    key = re.sub(r"\.w\d+\.", ".", key)
    return f"{key}={val}"


def _target_side(row: dict[str, Any]) -> str:
    target = row.get("target")
    if isinstance(target, str):
        try:
            target = json.loads(target)
        except json.JSONDecodeError:
            return "UNKNOWN"
    if not isinstance(target, dict):
        return "UNKNOWN"
    gate = str(target.get("gate", ""))
    side = str(target.get("side", ""))
    if gate != "TRADE":
        return "NO_TRADE"
    if side in {"LONG", "SHORT"}:
        return side
    return "UNKNOWN"


def _units(row: dict[str, Any], cfg: EdgeStabilityCfg) -> set[str]:
    pat = re.compile(cfg.exclude_regex) if cfg.exclude_regex.strip() else None
    out: set[str] = set()
    for tok in tokens_from_row(row):
        if pat and pat.search(tok):
            continue
        out.add(_semantic_unit(tok) if cfg.unit_mode == "semantic" else tok)
    return out


def _split_stats(rows: list[dict[str, Any]], cfg: EdgeStabilityCfg) -> tuple[dict[str, Counter[str]], Counter[str]]:
    by_tok: dict[str, Counter[str]] = defaultdict(Counter)
    totals: Counter[str] = Counter()
    for row in rows:
        side = _target_side(row)
        totals[side] += 1
        for tok in _units(row, cfg):
            by_tok[tok][side] += 1
    return by_tok, totals


def _edge(c: Counter[str]) -> dict[str, Any]:
    n = int(sum(c.values()))
    long = int(c.get("LONG", 0))
    short = int(c.get("SHORT", 0))
    no = int(c.get("NO_TRADE", 0))
    trade = long + short
    directional_edge = (long - short) / trade if trade else 0.0
    trade_rate = trade / n if n else 0.0
    long_rate = long / n if n else 0.0
    short_rate = short / n if n else 0.0
    no_trade_rate = no / n if n else 0.0
    # Binomial z-like directional imbalance among trade labels only.
    z = (long - short) / math.sqrt(trade) if trade else 0.0
    return {
        "n": n,
        "long": long,
        "short": short,
        "no_trade": no,
        "directional_edge": directional_edge,
        "trade_rate": trade_rate,
        "long_rate": long_rate,
        "short_rate": short_rate,
        "no_trade_rate": no_trade_rate,
        "dir_z_like": z,
    }


def _sign(x: float, eps: float = 1e-12) -> int:
    return 1 if x > eps else -1 if x < -eps else 0


def _agreement(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    common = set(a) & set(b)
    if not common:
        return {"common": 0, "same_sign_rate": None, "opposite_sign_rate": None}
    same = opp = nonzero = 0
    for tok in common:
        sa = _sign(a[tok]["directional_edge"])
        sb = _sign(b[tok]["directional_edge"])
        if sa == 0 or sb == 0:
            continue
        nonzero += 1
        same += int(sa == sb)
        opp += int(sa == -sb)
    return {
        "common": len(common),
        "nonzero_common": nonzero,
        "same_sign_rate": same / nonzero if nonzero else None,
        "opposite_sign_rate": opp / nonzero if nonzero else None,
    }


def run(cfg: EdgeStabilityCfg) -> dict[str, Any]:
    splits = {
        "train": _load(cfg.train_jsonl),
        "val": _load(cfg.val_jsonl),
        "eval": _load(cfg.eval_jsonl),
    }
    raw: dict[str, dict[str, Counter[str]]] = {}
    totals: dict[str, Counter[str]] = {}
    split_edges: dict[str, dict[str, dict[str, Any]]] = {}
    for name, rows in splits.items():
        raw[name], totals[name] = _split_stats(rows, cfg)
        split_edges[name] = {
            tok: _edge(c)
            for tok, c in raw[name].items()
            if sum(c.values()) >= cfg.min_count
        }

    train_rank = sorted(
        split_edges["train"].items(),
        key=lambda kv: (abs(kv[1]["dir_z_like"]), kv[1]["n"]),
        reverse=True,
    )
    rows = []
    for tok, tr in train_rank[: max(cfg.top_n, 0)]:
        val = split_edges["val"].get(tok)
        ev = split_edges["eval"].get(tok)
        rows.append({
            "token": tok,
            "train": tr,
            "val": val,
            "eval": ev,
            "train_val_same_sign": None if val is None else _sign(tr["directional_edge"]) == _sign(val["directional_edge"]),
            "train_eval_same_sign": None if ev is None else _sign(tr["directional_edge"]) == _sign(ev["directional_edge"]),
            "val_eval_same_sign": None if val is None or ev is None else _sign(val["directional_edge"]) == _sign(ev["directional_edge"]),
        })

    report = {
        "config": asdict(cfg),
        "split_totals": {k: dict(v) for k, v in totals.items()},
        "eligible_token_counts": {k: len(v) for k, v in split_edges.items()},
        "agreement": {
            "train_val": _agreement(split_edges["train"], split_edges["val"]),
            "train_eval": _agreement(split_edges["train"], split_edges["eval"]),
            "val_eval": _agreement(split_edges["val"], split_edges["eval"]),
        },
        "top_train_directional_tokens": rows,
        "leakage_guard": {
            "split_labels_summarized_independently": True,
            "eval_used_for_diagnostic_only_not_selection": True,
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> EdgeStabilityCfg:
    p = argparse.ArgumentParser(description="Audit split-to-split token directional-edge stability")
    for field in EdgeStabilityCfg.__dataclass_fields__.values():
        name = "--" + field.name.replace("_", "-")
        required = field.default.__class__.__name__ == "_MISSING_TYPE"
        p.add_argument(name, default=None if required else field.default, required=required)
    ns = vars(p.parse_args())
    ns["min_count"] = int(ns["min_count"])
    ns["top_n"] = int(ns["top_n"])
    if ns["unit_mode"] not in {"token", "semantic"}:
        raise ValueError("unit_mode must be token or semantic")
    return EdgeStabilityCfg(**ns)


def main() -> None:
    rep = run(parse_args())
    print(json.dumps({
        "split_totals": rep["split_totals"],
        "eligible_token_counts": rep["eligible_token_counts"],
        "agreement": rep["agreement"],
        "top5": rep["top_train_directional_tokens"][:5],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
