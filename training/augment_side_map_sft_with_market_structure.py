"""Augment side-map reliability SFT prompts with prior-month market structure."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AugmentSideMapSftWithMarketStructureCfg:
    input_jsonl: str
    market_csv: str
    output_jsonl: str
    summary_output: str = ""


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + ("\n" if rows else ""))


def _prev_month(month: str) -> str:
    ts = pd.Timestamp(f"{month}-01") - pd.offsets.MonthBegin(1)
    return f"{ts.year:04d}-{ts.month:02d}"


def _bucket_signed(x: float | None, small: float, large: float) -> str:
    if x is None or pd.isna(x):
        return "unknown"
    x = float(x)
    if x >= large:
        return "strong_up"
    if x >= small:
        return "up"
    if x <= -large:
        return "strong_down"
    if x <= -small:
        return "down"
    return "flat"


def _bucket_unsigned(x: float | None, small: float, large: float) -> str:
    if x is None or pd.isna(x):
        return "unknown"
    x = float(x)
    if x >= large:
        return "high"
    if x >= small:
        return "medium"
    return "low"


def _monthly_market(market_csv: str) -> dict[str, dict[str, float]]:
    df = pd.read_csv(market_csv)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.strftime("%Y-%m")
    out: dict[str, dict[str, float]] = {}
    for month, g in df.groupby("month", sort=True):
        opens = g["open"].to_numpy(dtype=float)
        highs = g["high"].to_numpy(dtype=float)
        lows = g["low"].to_numpy(dtype=float)
        if len(opens) < 2 or opens[0] <= 0:
            continue
        ret = opens[-1] / opens[0] - 1.0
        logret = np.diff(np.log(np.maximum(opens, 1e-12)))
        vol = float(np.std(logret) * np.sqrt(max(1, len(logret))))
        rng = float(np.max(highs) / max(1e-12, np.min(lows)) - 1.0)
        eq = opens / opens[0]
        peak = np.maximum.accumulate(eq)
        dd = float(np.max(1.0 - eq / np.maximum(peak, 1e-12)))
        close_pos = float((opens[-1] - np.min(lows)) / max(1e-12, np.max(highs) - np.min(lows)))
        out[str(month)] = {"return": float(ret), "vol": vol, "range": rng, "drawdown": dd, "close_pos": close_pos}
    return out


def _tokens_for(m: dict[str, float] | None) -> dict[str, str]:
    m = m or {}
    return {
        "prior_market_return": _bucket_signed(m.get("return"), 0.03, 0.10),
        "prior_market_vol": _bucket_unsigned(m.get("vol"), 0.15, 0.35),
        "prior_market_range": _bucket_unsigned(m.get("range"), 0.15, 0.35),
        "prior_market_drawdown": _bucket_unsigned(m.get("drawdown"), 0.08, 0.20),
        "prior_market_close_pos": _close_pos_bucket(m.get("close_pos")),
    }


def _close_pos_bucket(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "unknown"
    x = float(x)
    if x >= 0.8:
        return "near_high"
    if x <= 0.2:
        return "near_low"
    return "mid_range"


def _inject_prompt(prompt: str, tokens: dict[str, str]) -> str:
    lines = str(prompt).splitlines()
    rendered = ["prior_market_structure:"] + [f"- {k}: {v}" for k, v in sorted(tokens.items())]
    for i, line in enumerate(lines):
        if line.startswith("Policy intent:"):
            return "\n".join(lines[:i] + rendered + lines[i:])
    return "\n".join(lines + rendered)


def augment(cfg: AugmentSideMapSftWithMarketStructureCfg) -> dict[str, Any]:
    rows = _read_jsonl(cfg.input_jsonl)
    monthly = _monthly_market(cfg.market_csv)
    out = []
    counts: dict[str, int] = {}
    for row in rows:
        month = str(row.get("month"))
        pm = _prev_month(month)
        toks = _tokens_for(monthly.get(pm))
        nr = dict(row)
        nr["prompt"] = _inject_prompt(str(row.get("prompt", "")), toks)
        src = dict(nr.get("source", {}) if isinstance(nr.get("source"), dict) else {})
        src["prior_market_structure_month"] = pm
        src["prior_market_structure_tokens"] = toks
        nr["source"] = src
        guard = dict(nr.get("leakage_guard", {}) if isinstance(nr.get("leakage_guard"), dict) else {})
        guard["market_structure_uses_only_prior_month"] = True
        nr["leakage_guard"] = guard
        for k, v in toks.items():
            counts[f"{k}={v}"] = counts.get(f"{k}={v}", 0) + 1
        out.append(nr)
    _write_jsonl(cfg.output_jsonl, out)
    report = {"as_of": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg), "rows": len(out), "token_counts": counts, "leakage_guard": {"market_month_is_prior_to_target_month": True}}
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augment side-map SFT rows with prior market structure tokens")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--market-csv", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--summary-output", default="")
    return p.parse_args()


def main() -> None:
    report = augment(AugmentSideMapSftWithMarketStructureCfg(**vars(parse_args())))
    print(json.dumps({"output": report["config"]["output_jsonl"], "rows": report["rows"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
