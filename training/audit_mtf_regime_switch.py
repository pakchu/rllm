"""Audit higher-timeframe regime buckets for broad-on vs selective opportunity."""
from __future__ import annotations

import argparse, json, re
from collections import defaultdict
from pathlib import Path
from typing import Any

SYMBOL_RE = re.compile(r"^(4H|1D|3D|1W) (Regime|Location):\s*(\S+)\s*$")


def load_jsonl(path: str) -> list[dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def parse_symbols(prompt: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in str(prompt).splitlines():
        m = SYMBOL_RE.match(line.strip())
        if m:
            out[f"{m.group(1)}_{m.group(2).lower()}"] = m.group(3)
        elif line.startswith("MTF Activation Mode:"):
            out["mtf_mode"] = line.split(":", 1)[1].strip()
    return out


def summarize(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sym = parse_symbols(row.get("prompt", ""))
        by[sym.get(key, "MISSING")].append(row)
    out = []
    for bucket, rs in sorted(by.items()):
        rets = [float(r.get("trade_ret_pct", 0.0)) for r in rs]
        targets = [json.loads(r["target"])["decision"] == "ACTIVATE" for r in rs]
        all_ret = sum(rets)
        oracle_ret = sum(r for r, ok in zip(rets, targets) if ok)
        out.append({
            "bucket": bucket,
            "rows": len(rs),
            "all_activate_ret_pct": all_ret,
            "oracle_ret_pct": oracle_ret,
            "oracle_activations": sum(targets),
            "bad_all_activate": all_ret < 0.0,
            "oracle_gap_pct": oracle_ret - all_ret,
        })
    return out


def run(splits: list[str], output: str) -> dict[str, Any]:
    report: dict[str, Any] = {"splits": {}}
    keys = ["4H_regime", "4H_location", "1D_regime", "1D_location", "3D_regime", "3D_location", "1W_regime", "1W_location", "mtf_mode"]
    for path in splits:
        rows = load_jsonl(path)
        report["splits"][path] = {key: summarize(rows, key) for key in keys}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    print(json.dumps(run(args.splits, args.output), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
