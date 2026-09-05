"""Reporting-unit-only successor for the fixed macro/OI five-minute audit."""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from training import audit_macro_oi_fresh_portfolio as v1
from training import evaluate_macro_flow_fixed_fresh as macro_fresh
from training import evaluate_oi_divergence_fresh as oi_v1
from training import search_macro_flow_alpha_combinations as macro_search
from training import search_meaningful_alpha_combinations as base

OUT = base.ROOT / "research/macro_oi_fresh_portfolio_v2"
FAILURE = base.ROOT / "research/macro_oi_fresh_portfolio/runtime_failure.json"
DESIGN = {
    **v1.DESIGN,
    "version": 2,
    "correction": "annualize five-minute returns with 12 bars/hour and CAGR with actual elapsed time",
    "unchanged": "sources, signals, targets, weights, netting, costs, funding, absolute returns and MDD",
}


def register() -> dict:
    payload = {
        "design": DESIGN,
        "code_sha256": base.sha(__file__),
        "v1_sha256": base.sha(v1.__file__),
        "v1_design_sha256": base.sha(v1.OUT / "design.json"),
        "failure_sha256": base.sha(FAILURE),
        "macro_report_sha256": base.sha(v1.MACRO_REPORT),
        "oi_report_sha256": base.sha(v1.OI_REPORT),
    }
    path = OUT / "design.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise RuntimeError("Frozen portfolio v2 audit drift")
    base.write_json(path, payload)
    return payload


def corrected_row(stats: dict, index: int, start: str, end: str) -> dict:
    row = base.stats_row(stats, index)
    years = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / (365.25 * 86400)
    equity = float(stats["equity"][index])
    cagr = (max(equity, 0) ** (1 / years) - 1) * 100 if equity > 0 else -100.0
    returns = stats["returns"][:, index]
    standard_deviation = float(returns.std())
    sharpe = float(returns.mean() * np.sqrt(365.25 * 24 * 12) / standard_deviation) if standard_deviation > 1e-12 else 0.0
    row["cagr_pct"] = float(cagr)
    row["calmar"] = float(cagr / row["mdd_pct"]) if row["mdd_pct"] > 1e-12 else 0.0
    row["sharpe"] = sharpe
    return row


def run() -> None:
    registration = json.loads((OUT / "design.json").read_text())
    if registration != register():
        raise RuntimeError("Registration changed")
    source_config = json.loads(oi_v1.CONFIG.read_text())["signal"]
    candidate = {
        **source_config,
        "hold_bars": int(source_config["hold_bars_5m"]),
        "stride_bars": int(source_config["stride_bars_5m"]),
    }
    market, funding, oi_source, receipt = oi_v1.load_context()
    trades, _, _, _ = oi_v1.schedule(market, funding, candidate)
    features = base.features(market, funding)
    features, hourly, engine_receipt = base.execution_blocks(market, funding, features)
    raw_macro = pd.DataFrame(
        {
            "date": market.date,
            "dxy": market.dxy,
            "usdkrw": market.usdkrw,
            "kimchi_premium": market.kimchi_premium,
            "dxy_available": market.dxy_available,
            "usdkrw_available": market.usdkrw_available,
            "kimchi_available": market.kimchi_available,
        }
    )
    features = pd.concat([features, macro_search.macro_features(raw_macro, features.index)], axis=1)
    macro_positions, _ = macro_fresh.fixed_positions(features)
    indices, blocks = v1.five_minute_blocks(market, funding)
    block_dates = pd.Series(pd.to_datetime(market.date).to_numpy()[indices])
    macro_target = v1.target_from_updates(
        block_dates,
        hourly["date"],
        macro_positions["dollar_flow_plus_regime_switch"],
    )
    oi_target_full = np.zeros(len(market))
    for trade in trades:
        oi_target_full[trade.entry_position : trade.exit_position] = trade.side
    oi_target = oi_target_full[indices]
    position_matrix = np.column_stack(
        [weight * macro_target + (1 - weight) * oi_target for weight in v1.WEIGHTS]
    )
    reports = {}
    isolated_returns = None
    for cost in DESIGN["costs_per_side"]:
        stats = base.simulate(blocks, position_matrix, cost=cost, fine=False)
        reports[str(cost)] = {
            f"macro_{weight:g}_oi_{1-weight:g}": corrected_row(
                stats, i, oi_v1.START, oi_v1.EVAL_END
            )
            for i, weight in enumerate(v1.WEIGHTS)
        }
        if cost == 0.0006:
            isolated_returns = base.simulate(
                blocks,
                np.column_stack([macro_target, oi_target]),
                cost=cost,
                fine=False,
            )["returns"]
    result = {
        "registration": registration,
        "source_receipt": receipt,
        "engine_receipt": engine_receipt,
        "oi_source": {
            "rows": len(oi_source),
            "first": str(oi_source.date.min()),
            "last": str(oi_source.date.max()),
        },
        "targets_sha256": {
            "macro": hashlib.sha256(macro_target.tobytes()).hexdigest(),
            "oi": hashlib.sha256(oi_target.tobytes()).hexdigest(),
        },
        "oi_trades": len(trades),
        "base_cost_return_correlation": float(np.corrcoef(isolated_returns.T)[0, 1]),
        "reports": reports,
        "live_enabled": False,
        "limitations": [
            "Both sleeves and the common window were exposed before this audit.",
            "No weight is an OOS optimized winner; all five predeclared weights are reported.",
            "Eight OI trades dominate inference and the OI source ends on 2026-08-03.",
        ],
    }
    base.write_json(OUT / "report.json", result)
    print(json.dumps(reports["0.0006"], indent=2), flush=True)
    print("return correlation", result["base_cost_return_correlation"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    register() if args.freeze else run()
