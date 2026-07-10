"""Validate a fixed event gate across calendar blocks.

This is for post-selection robustness checks. It does not fit thresholds or
rank gates. A caller supplies fixed gates and the script reports strict
bar-by-bar performance for train/test/eval and calendar sub-periods.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import MISSING, asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from training.sweep_conjunctive_event_gates import Gate, backtest, filter_rows, load_rows
from training.strict_bar_backtest import load_market_bars


@dataclass(frozen=True)
class FixedGatePeriodValidationCfg:
    train_jsonl: str
    test_jsonl: str
    eval_jsonl: str
    market_csv: str
    output: str
    gates_json: str
    leverage: float = 0.5
    side_filter: str = ""
    family_filter: str = ""
    train_start: str = "2021-01-01"
    train_end: str = "2024-12-31 23:59:59"
    test_start: str = "2025-01-01"
    test_end: str = "2025-12-31 23:59:59"
    eval_start: str = "2026-01-01"
    eval_end: str = "2026-06-01 00:00:00"
    block_freq: str = "Q"


def _parse_gates(raw: str) -> tuple[Gate, ...]:
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError("gates_json must be a JSON list")
    return tuple(Gate(feature=str(g["feature"]), op=str(g["op"]), threshold=float(g["threshold"])) for g in payload)


def _load_sets(cfg: FixedGatePeriodValidationCfg) -> dict[str, list[dict[str, Any]]]:
    sets = {"train": load_rows(cfg.train_jsonl), "test": load_rows(cfg.test_jsonl), "eval": load_rows(cfg.eval_jsonl)}
    side = str(cfg.side_filter).strip().upper()
    family = str(cfg.family_filter).strip()
    if side:
        sets = {k: [r for r in rows if str(r.get("action", {}).get("side", "")).upper() == side] for k, rows in sets.items()}
    if family:
        sets = {k: [r for r in rows if str(r.get("action", {}).get("family", "")) == family] for k, rows in sets.items()}
    return sets


def _period_rows(rows: list[dict[str, Any]], start: str, end: str) -> list[dict[str, Any]]:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    out = []
    for row in rows:
        ts = pd.Timestamp(str(row["date"]))
        if s <= ts <= e:
            out.append(row)
    return out


def _score_period(name: str, rows: list[dict[str, Any]], gates: tuple[Gate, ...], market: Any, cfg: FixedGatePeriodValidationCfg, start: str, end: str) -> dict[str, Any]:
    selected = filter_rows(_period_rows(rows, start, end), gates)
    result = backtest(selected, market, float(cfg.leverage), annualization_start=start, annualization_end=end)
    if result is None:
        return {"name": name, "start": start, "end": end, "n_rows": 0, "sim": {}, "trade_stats": {}}
    return {"name": name, "start": start, "end": end, "n_rows": len(selected), "sim": result["sim"], "trade_stats": result["trade_stats"]}


def _calendar_blocks(start: str, end: str, freq: str) -> list[tuple[str, str]]:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    if str(freq).upper().startswith("M"):
        starts = pd.date_range(s.normalize().replace(day=1), e, freq="MS")
        ends = [min((x + pd.offsets.MonthEnd(1)).normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59), e) for x in starts]
    else:
        starts = pd.date_range(s.to_period("Q").start_time, e, freq="QS")
        ends = [min(x.to_period("Q").end_time, e) for x in starts]
    return [
        (max(st, s).strftime("%Y-%m-%d %H:%M:%S"), en.floor("s").strftime("%Y-%m-%d %H:%M:%S"))
        for st, en in zip(starts, ends)
        if en >= s and st <= e
    ]


def run(cfg: FixedGatePeriodValidationCfg) -> dict[str, Any]:
    gates = _parse_gates(cfg.gates_json)
    market = load_market_bars(cfg.market_csv)
    sets = _load_sets(cfg)
    all_rows = sorted(sets["train"] + sets["test"] + sets["eval"], key=lambda r: str(r["date"]))
    periods = {
        "train": _score_period("train", sets["train"], gates, market, cfg, cfg.train_start, cfg.train_end),
        "test": _score_period("test", sets["test"], gates, market, cfg, cfg.test_start, cfg.test_end),
        "eval": _score_period("eval", sets["eval"], gates, market, cfg, cfg.eval_start, cfg.eval_end),
    }
    blocks = [
        _score_period(f"{start[:10]}..{end[:10]}", all_rows, gates, market, cfg, start, end)
        for start, end in _calendar_blocks(cfg.test_start, cfg.eval_end, cfg.block_freq)
    ]
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "gates": [g.as_dict() for g in gates],
        "periods": periods,
        "blocks": blocks,
        "leakage_guard": {
            "fixed_gates_only_no_fit": True,
            "calendar_blocks_report_only": True,
            "cagr_uses_full_period_including_idle_time": True,
            "strict_mdd_uses_bar_by_bar_adverse_excursion": True,
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    for field in FixedGatePeriodValidationCfg.__dataclass_fields__.values():
        required = field.default is MISSING and field.default_factory is MISSING
        p.add_argument("--" + field.name.replace("_", "-"), default=None if required else field.default, required=required)
    ns = p.parse_args()
    ns.leverage = float(ns.leverage)
    return ns


def main() -> None:
    ns = parse_args()
    cfg = FixedGatePeriodValidationCfg(**vars(ns))
    out = run(cfg)
    print(json.dumps({"periods": {k: v["sim"] for k, v in out["periods"].items()}, "blocks": out["blocks"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
