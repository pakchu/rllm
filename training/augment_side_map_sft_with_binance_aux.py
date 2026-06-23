"""Augment side-map reliability SFT prompts with prior-month Binance aux state."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class AugmentSideMapSftWithBinanceAuxCfg:
    input_jsonl: str
    premium_csv: str
    funding_csv: str
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


def _bucket(x: float | None, small: float, large: float) -> str:
    if x is None or pd.isna(x):
        return "unknown"
    x = float(x)
    if x >= large:
        return "high_positive"
    if x >= small:
        return "positive"
    if x <= -large:
        return "high_negative"
    if x <= -small:
        return "negative"
    return "neutral"


def _load_monthly_aux(premium_csv: str, funding_csv: str) -> dict[str, dict[str, Any]]:
    prem = pd.read_csv(premium_csv)
    prem["date"] = pd.to_datetime(prem["date"])
    prem["month"] = prem["date"].dt.strftime("%Y-%m")
    prem_m = prem.groupby("month").agg(premium_mean=("close", "mean"), premium_abs_mean=("close", lambda s: float(s.abs().mean()))).reset_index()

    fund = pd.read_csv(funding_csv)
    fund["date"] = pd.to_datetime(fund["date"])
    fund["month"] = fund["date"].dt.strftime("%Y-%m")
    fund_m = fund.groupby("month").agg(funding_mean=("funding_rate", "mean"), funding_abs_mean=("funding_rate", lambda s: float(s.abs().mean()))).reset_index()

    joined = prem_m.merge(fund_m, on="month", how="outer")
    out: dict[str, dict[str, Any]] = {}
    for _, r in joined.iterrows():
        out[str(r["month"])] = {
            "premium_mean": None if pd.isna(r.get("premium_mean")) else float(r.get("premium_mean")),
            "premium_abs_mean": None if pd.isna(r.get("premium_abs_mean")) else float(r.get("premium_abs_mean")),
            "funding_mean": None if pd.isna(r.get("funding_mean")) else float(r.get("funding_mean")),
            "funding_abs_mean": None if pd.isna(r.get("funding_abs_mean")) else float(r.get("funding_abs_mean")),
        }
    return out


def _tokens_for(aux: dict[str, Any] | None) -> dict[str, str]:
    aux = aux or {}
    return {
        "prior_btc_premium_mean": _bucket(aux.get("premium_mean"), 0.0001, 0.0005),
        "prior_btc_premium_abs": _bucket(aux.get("premium_abs_mean"), 0.0002, 0.0008).replace("negative", "positive"),
        "prior_btc_funding_mean": _bucket(aux.get("funding_mean"), 0.00003, 0.0001),
        "prior_btc_funding_abs": _bucket(aux.get("funding_abs_mean"), 0.00003, 0.0001).replace("negative", "positive"),
    }


def _inject_prompt(prompt: str, tokens: dict[str, str]) -> str:
    lines = str(prompt).splitlines()
    rendered = ["prior_binance_aux_state:"] + [f"- {k}: {v}" for k, v in sorted(tokens.items())]
    for i, line in enumerate(lines):
        if line.startswith("Policy intent:"):
            return "\n".join(lines[:i] + rendered + lines[i:])
    return "\n".join(lines + rendered)


def augment(cfg: AugmentSideMapSftWithBinanceAuxCfg) -> dict[str, Any]:
    rows = _read_jsonl(cfg.input_jsonl)
    monthly = _load_monthly_aux(cfg.premium_csv, cfg.funding_csv)
    out = []
    counts: dict[str, int] = {}
    for row in rows:
        month = str(row.get("month"))
        pm = _prev_month(month)
        toks = _tokens_for(monthly.get(pm))
        nr = dict(row)
        nr["prompt"] = _inject_prompt(str(row.get("prompt", "")), toks)
        src = dict(nr.get("source", {}) if isinstance(nr.get("source"), dict) else {})
        src["prior_binance_aux_month"] = pm
        src["prior_binance_aux_tokens"] = toks
        nr["source"] = src
        guard = dict(nr.get("leakage_guard", {}) if isinstance(nr.get("leakage_guard"), dict) else {})
        guard["binance_aux_uses_only_prior_month"] = True
        nr["leakage_guard"] = guard
        for k, v in toks.items():
            counts[f"{k}={v}"] = counts.get(f"{k}={v}", 0) + 1
        out.append(nr)
    _write_jsonl(cfg.output_jsonl, out)
    report = {"as_of": datetime.now(timezone.utc).isoformat(), "config": asdict(cfg), "rows": len(out), "token_counts": counts, "leakage_guard": {"aux_month_is_prior_to_target_month": True}}
    if cfg.summary_output:
        Path(cfg.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.summary_output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augment side-map SFT rows with prior Binance aux tokens")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--premium-csv", required=True)
    p.add_argument("--funding-csv", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--summary-output", default="")
    return p.parse_args()


def main() -> None:
    report = augment(AugmentSideMapSftWithBinanceAuxCfg(**vars(parse_args())))
    print(json.dumps({"output": report["config"]["output_jsonl"], "rows": report["rows"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
