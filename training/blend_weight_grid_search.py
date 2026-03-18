"""Grid-search non-leaky blend weights for exp6/exp4 on 5m score-band mode."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from evaluation.backtest import run_backtest

EXP6 = "checkpoints/ppo_option_a_real_w384_t786k_exp6_balanced.zip"
EXP4 = "checkpoints/ppo_option_a_real_exp4_nosym.zip"


@dataclass
class Candidate:
    name: str
    params: dict


def candidates() -> list[Candidate]:
    out: list[Candidate] = []
    for w in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        out.append(
            Candidate(
                name=f"static_w{int(w*100):02d}",
                params={
                    "blend_weight_mode": "static",
                    "blend_weight_a": w,
                    "blend_weight_up": 0.90,
                    "blend_weight_down": 0.35,
                    "blend_trend_threshold": 0.002,
                },
            )
        )

    trend_configs = [
        (0.90, 0.25, 0.0015),
        (0.90, 0.35, 0.0015),
        (0.90, 0.35, 0.0020),
        (0.85, 0.35, 0.0020),
        (0.85, 0.45, 0.0020),
        (0.80, 0.45, 0.0030),
        (0.95, 0.30, 0.0020),
        (0.95, 0.25, 0.0015),
    ]
    for up, down, th in trend_configs:
        out.append(
            Candidate(
                name=f"trend_up{int(up*100)}_dn{int(down*100)}_th{str(th).replace('.','p')}",
                params={
                    "blend_weight_mode": "trend",
                    "blend_weight_a": 0.75,
                    "blend_weight_up": up,
                    "blend_weight_down": down,
                    "blend_trend_threshold": th,
                },
            )
        )
    return out


def score(rep: dict, dd_coef: float = 0.05, sh_coef: float = 0.5) -> float:
    return float(rep["cumulative_return_pct"]) - dd_coef * float(rep["max_drawdown_pct"]) + sh_coef * float(rep["sharpe_ratio"])


def run(output: str, topn: int) -> dict:
    periods = {
        "recent_3m": ("2025-12-01", "2026-02-28"),
        "strict_oos": ("2025-04-01", "2026-02-28"),
    }
    followup_1y = ("2025-03-01", "2026-02-28")

    common = dict(
        source="binance",
        symbol="BTCUSDT",
        timeframe="5m",
        market_type="futures",
        window_size=384,
        deterministic=True,
        decision_mode="blend_score_band",
        model_path=EXP6,
        blend_model_a=EXP6,
        blend_model_b=EXP4,
        debiased_action="mirror_scalar",
        flat_start_policy="as_is",
        score_centering="ema",
        score_center_alpha=0.02,
        score_entry_threshold=0.005,
        score_flip_threshold=0.02,
        score_neutral_band=0.001,
    )

    out = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "periods": periods,
        "followup_1y": {"start": followup_1y[0], "end": followup_1y[1]},
        "candidates": {},
    }

    cands = candidates()
    total = len(cands) * len(periods)
    k = 0
    for c in cands:
        row = {"params": c.params, "period_reports": {}, "scores": {}}
        for pname, (s, e) in periods.items():
            k += 1
            print(f"[{k}/{total}] {c.name} {pname}", flush=True)
            rep = run_backtest(start_date=s, end_date=e, **common, **c.params)
            row["period_reports"][pname] = rep
            row["scores"][pname] = score(rep)
        row["combined_score"] = 0.5 * row["scores"]["recent_3m"] + 0.5 * row["scores"]["strict_oos"]
        out["candidates"][c.name] = row
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(out, indent=2))

    ranking = sorted(
        [
            {
                "name": name,
                "combined_score": float(data["combined_score"]),
                "recent_score": float(data["scores"]["recent_3m"]),
                "strict_oos_score": float(data["scores"]["strict_oos"]),
                "recent_return_pct": float(data["period_reports"]["recent_3m"]["cumulative_return_pct"]),
                "strict_oos_return_pct": float(data["period_reports"]["strict_oos"]["cumulative_return_pct"]),
                "strict_oos_mdd_pct": float(data["period_reports"]["strict_oos"]["max_drawdown_pct"]),
            }
            for name, data in out["candidates"].items()
        ],
        key=lambda x: (x["combined_score"], x["strict_oos_score"]),
        reverse=True,
    )
    out["ranking"] = ranking

    top = ranking[: max(1, int(topn))]
    out["followup_last1y_reports"] = {}
    for item in top:
        name = item["name"]
        params = out["candidates"][name]["params"]
        print(f"[followup1y] {name}", flush=True)
        rep = run_backtest(start_date=followup_1y[0], end_date=followup_1y[1], **common, **params)
        out["followup_last1y_reports"][name] = rep
    out["followup_last1y_ranking"] = sorted(
        [
            {
                "name": name,
                "return_pct": float(rep["cumulative_return_pct"]),
                "mdd_pct": float(rep["max_drawdown_pct"]),
                "sharpe": float(rep["sharpe_ratio"]),
                "score": score(rep),
            }
            for name, rep in out["followup_last1y_reports"].items()
        ],
        key=lambda x: (x["score"], x["return_pct"]),
        reverse=True,
    )

    Path(output).write_text(json.dumps(out, indent=2))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid search non-leaky blend weights")
    p.add_argument("--topn", type=int, default=4)
    p.add_argument("--output", type=str, default="results/blend_weight_grid_search_v1.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = run(output=args.output, topn=args.topn)
    print("[saved]", args.output)
    print("top:", out["ranking"][:5])
    print("followup:", out.get("followup_last1y_ranking", []))


if __name__ == "__main__":
    main()
