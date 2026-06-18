"""Audit executed-trade distribution for candidate backtest result files."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _load_executed(path: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obj = json.loads(Path(path).read_text())
    if "executed" in obj:
        return obj, list(obj.get("executed", []))
    if "trials" in obj and obj["trials"]:
        trial = obj["trials"][0]
        return trial, list(trial.get("executed", []))
    raise ValueError(f"no executed trades found in {path}; run detailed evaluator first")


def audit(path: str, *, top_n: int = 10) -> dict[str, Any]:
    summary, executed = _load_executed(path)
    rows: list[dict[str, Any]] = []
    for row in executed:
        dt = pd.to_datetime(row["signal_date"])
        rows.append(
            {
                "month": dt.strftime("%Y-%m"),
                "date": str(row["signal_date"]),
                "side": str(row.get("side", "")),
                "ret_pct": float(row.get("executed_ret_pct", 0.0)),
                "score": float(row.get("score_mean", 0.0)),
                "path_mae_pct": float(row.get("path_mae_pct", 0.0)),
                "path_net_pct": float(row.get("path_net_pct", 0.0)),
                "target_value": float(row.get("eval_target_value", 0.0)),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return {"as_of": datetime.now(timezone.utc).isoformat(), "input": path, "n_trades": 0}
    monthly = []
    for month, g in df.groupby("month", sort=True):
        monthly.append(
            {
                "month": str(month),
                "n": int(len(g)),
                "sum_ret_pct": float(g["ret_pct"].sum()),
                "mean_ret_pct": float(g["ret_pct"].mean()),
                "min_ret_pct": float(g["ret_pct"].min()),
                "max_ret_pct": float(g["ret_pct"].max()),
                "win_rate": float((g["ret_pct"] > 0).mean()),
                "mean_path_mae_pct": float(g["path_mae_pct"].mean()),
            }
        )
    best = df.sort_values("ret_pct", ascending=False).head(top_n).to_dict(orient="records")
    worst = df.sort_values("ret_pct", ascending=True).head(top_n).to_dict(orient="records")
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "input": path,
        "sim": summary.get("sim", {}),
        "trade_stats": summary.get("trade_stats", {}),
        "n_trades": int(len(df)),
        "overall": {
            "sum_trade_ret_pct": float(df["ret_pct"].sum()),
            "mean_trade_ret_pct": float(df["ret_pct"].mean()),
            "win_rate": float((df["ret_pct"] > 0).mean()),
            "top5_gain_sum_pct": float(df.sort_values("ret_pct", ascending=False).head(5)["ret_pct"].sum()),
            "top10_gain_sum_pct": float(df.sort_values("ret_pct", ascending=False).head(10)["ret_pct"].sum()),
            "bottom5_loss_sum_pct": float(df.sort_values("ret_pct", ascending=True).head(5)["ret_pct"].sum()),
            "bottom10_loss_sum_pct": float(df.sort_values("ret_pct", ascending=True).head(10)["ret_pct"].sum()),
        },
        "monthly": monthly,
        "best_trades": best,
        "worst_trades": worst,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit executed trade distribution from detailed candidate backtest JSON")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--top-n", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = audit(args.input, top_n=args.top_n)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({"n_trades": out["n_trades"], "overall": out.get("overall"), "monthly": out.get("monthly")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
