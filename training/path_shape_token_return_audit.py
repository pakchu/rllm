"""Audit token-conditioned strict trade returns for path-shape token policies."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training.path_shape_token_policy_tte import PathShapeTokenPolicyCfg, _fit, _load, _maybe_invert, _predict, tokens_from_row


@dataclass(frozen=True)
class TokenReturnAuditCfg:
    train_jsonl: str
    eval_jsonl: str
    executed_json: str
    output: str
    min_count: int = 3
    top_k_tokens: int = 24
    side_mode: str = "normal"
    smoothing: float = 2.0
    min_token_trades: int = 8


def _dummy_policy_cfg(cfg: TokenReturnAuditCfg) -> PathShapeTokenPolicyCfg:
    return PathShapeTokenPolicyCfg(
        train_jsonl=cfg.train_jsonl,
        val_jsonl=cfg.eval_jsonl,
        eval_jsonl=cfg.eval_jsonl,
        market_csv="",
        work_dir="",
        output="",
        min_count=int(cfg.min_count),
        smoothing=float(cfg.smoothing),
        top_k_tokens=int(cfg.top_k_tokens),
    )


def _trade_key(row: dict[str, Any]) -> tuple[str, int, str, int]:
    return (str(row.get("date")), int(row.get("signal_pos", -1) or -1), str(row.get("side", "")), int(row.get("hold_bars", 0) or 0))


def _summ(vals: list[float]) -> dict[str, Any]:
    n = len(vals)
    if not vals:
        return {"n": 0, "sum_ret_pct": 0.0, "mean_ret_pct": 0.0, "win_rate": 0.0}
    return {
        "n": n,
        "sum_ret_pct": float(sum(vals)),
        "mean_ret_pct": float(sum(vals) / n),
        "win_rate": float(sum(1 for x in vals if x > 0.0) / n),
    }


def _group(tok: str) -> str:
    raw = str(tok)
    if raw.startswith("aug.micro."):
        return "micro"
    if raw.startswith("aug.pa.") or raw.startswith("augnum.pa."):
        return "price_action"
    if raw.startswith("aug.macro.") or raw.startswith("augnum.macro."):
        return "macro"
    if raw.startswith("recent="):
        return "recent_bar_sequence"
    if raw.startswith("ev."):
        return "evidence"
    if raw.startswith("seq."):
        return "sequence_stats"
    if raw.startswith("sym.") or raw.startswith("tag="):
        return "symbolic"
    return raw.split("=", 1)[0].split(".", 1)[0]


def run(cfg: TokenReturnAuditCfg) -> dict[str, Any]:
    policy_cfg = _dummy_policy_cfg(cfg)
    train = _load(cfg.train_jsonl)
    rows = _load(cfg.eval_jsonl)
    model = _fit(train, policy_cfg)
    preds = [_predict(r, model, policy_cfg) for r in rows]
    executed_payload = json.loads(Path(cfg.executed_json).read_text())
    executed = executed_payload.get("executed", [])
    by_key = {_trade_key(r): float(r.get("trade_ret_pct", 0.0) or 0.0) for r in executed}
    by_token: dict[str, list[float]] = defaultdict(list)
    by_group: dict[str, list[float]] = defaultdict(list)
    matched = 0
    misses = 0
    for row, pred in zip(rows, preds):
        label = _maybe_invert(str(pred["label"]), cfg.side_mode)
        if label == "NO_TRADE":
            continue
        key = (str(row.get("date")), int(row.get("signal_pos", -1) or -1), label, 144)
        if key not in by_key:
            misses += 1
            continue
        matched += 1
        ret = by_key[key]
        toks = tokens_from_row(row)
        for tok in set(toks):
            by_token[tok].append(ret)
            by_group[_group(tok)].append(ret)
    token_rows = [{"token": tok, "group": _group(tok), **_summ(vals)} for tok, vals in by_token.items() if len(vals) >= int(cfg.min_token_trades)]
    group_rows = [{"group": g, **_summ(vals)} for g, vals in by_group.items() if len(vals) >= int(cfg.min_token_trades)]
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "matched_trades": matched,
        "misses": misses,
        "groups": sorted(group_rows, key=lambda r: r["mean_ret_pct"]),
        "worst_tokens": sorted(token_rows, key=lambda r: (r["mean_ret_pct"], -r["n"]))[:80],
        "best_tokens": sorted(token_rows, key=lambda r: (r["mean_ret_pct"], r["n"]), reverse=True)[:80],
        "leakage_guard": {"model_fit_on_train_only": True, "audit_uses_realized_eval_trades_for_diagnostics_only": True},
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit token-conditioned returns for path-shape token policy")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--executed-json", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-count", type=int, default=3)
    p.add_argument("--top-k-tokens", type=int, default=24)
    p.add_argument("--side-mode", choices=["normal", "invert"], default="normal")
    p.add_argument("--smoothing", type=float, default=2.0)
    p.add_argument("--min-token-trades", type=int, default=8)
    return p.parse_args()


def main() -> None:
    rep = run(TokenReturnAuditCfg(**vars(parse_args())))
    print(json.dumps({"matched_trades": rep["matched_trades"], "groups": rep["groups"], "worst_tokens": rep["worst_tokens"][:10], "best_tokens": rep["best_tokens"][:10]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
