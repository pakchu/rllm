"""Native-event rebalance successor for the fixed macro/OI portfolio audit."""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from training import audit_macro_oi_fresh_portfolio as v1
from training import audit_macro_oi_fresh_portfolio_v2 as v2
from training import evaluate_macro_flow_fixed_fresh as macro_fresh
from training import evaluate_oi_divergence_fresh as oi_v1
from training import search_macro_flow_alpha_combinations as macro_search
from training import search_meaningful_alpha_combinations as base

OUT = base.ROOT / "research/macro_oi_fresh_portfolio_v3"
FAILURE = base.ROOT / "research/macro_oi_fresh_portfolio_v2/runtime_failure.json"
DESIGN = {
    **v2.DESIGN,
    "version": 3,
    "correction": "native event rebalancing: macro hourly updates; OI entries/exits only",
    "unchanged": "sources, signals, targets, weights, netting, cost rates, funding and price bars",
}


def register() -> dict:
    payload = {
        "design": DESIGN,
        "code_sha256": base.sha(__file__),
        "v1_sha256": base.sha(v1.__file__),
        "v2_sha256": base.sha(v2.__file__),
        "v2_failure_sha256": base.sha(FAILURE),
        "macro_report_sha256": base.sha(v1.MACRO_REPORT),
        "oi_report_sha256": base.sha(v1.OI_REPORT),
    }
    path = OUT / "design.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise RuntimeError("Frozen portfolio v3 audit drift")
    base.write_json(path, payload)
    return payload


def simulate_events(blocks, positions, rebalance, *, cost, start, end):
    positions = np.asarray(positions, dtype=float)
    rebalance = np.asarray(rebalance, dtype=bool)
    if positions.ndim == 1:
        positions = positions[:, None]
    if rebalance.ndim == 1:
        rebalance = rebalance[:, None]
    if positions.shape != rebalance.shape or np.max(np.abs(positions), initial=0) > 1.0000001:
        raise ValueError("Invalid event-driven targets")
    rows, candidates = positions.shape
    equity = np.ones(candidates)
    peak = equity.copy()
    mdd = np.zeros(candidates)
    units = np.zeros(candidates)
    turnover = np.zeros(candidates)
    fees = np.zeros(candidates)
    funding_paid = np.zeros(candidates)
    entries = np.zeros(candidates, dtype=int)
    changes = np.zeros(candidates, dtype=int)
    returns = np.zeros((rows, candidates))
    previous_close = float(blocks["open"][0])
    for i in range(rows):
        opening = blocks["open"][i]
        equity += units * (opening - previous_close)
        prior = equity.copy()
        update = rebalance[i]
        target_units = positions[i] * equity / opening
        traded = np.where(update, target_units - units, 0.0)
        entries += (update & (target_units * units <= 0) & (np.abs(target_units) > 1e-15)).astype(int)
        changes += (np.abs(traded) > 1e-10).astype(int)
        turnover += np.abs(traded) * opening / np.maximum(prior, 1e-12)
        charge = np.abs(traded) * opening * cost
        fees += charge
        equity -= charge
        units += traded
        transfer = units * blocks["funding"][i]
        high_equity = equity + units * (
            np.where(units >= 0, blocks["high"][i], blocks["low"][i]) - opening
        ) + np.where(transfer < 0, -transfer, 0)
        low_equity = equity + units * (
            np.where(units >= 0, blocks["low"][i], blocks["high"][i]) - opening
        ) - np.where(transfer > 0, transfer, 0)
        peak = np.maximum(peak, high_equity)
        mdd = np.maximum(mdd, 1 - low_equity / np.maximum(peak, 1e-12))
        funding_paid += transfer
        equity -= transfer
        closing = blocks["end"][i]
        equity += units * (closing - opening)
        previous_close = closing
        if i == rows - 1:
            final_charge = np.abs(units) * closing * cost
            fees += final_charge
            equity -= final_charge
            units[:] = 0
        peak = np.maximum(peak, equity)
        mdd = np.maximum(mdd, 1 - equity / np.maximum(peak, 1e-12))
        returns[i] = equity / np.maximum(prior, 1e-12) - 1
    years = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / (365.25 * 86400)
    cagr = np.where(equity > 0, equity ** (1 / years) - 1, -1)
    standard_deviation = returns.std(axis=0)
    sharpe = np.divide(
        returns.mean(axis=0) * np.sqrt(365.25 * 24 * 12),
        standard_deviation,
        out=np.zeros(candidates),
        where=standard_deviation > 1e-12,
    )
    return {
        "equity": equity,
        "return_pct": (equity - 1) * 100,
        "cagr_pct": cagr * 100,
        "mdd_pct": mdd * 100,
        "calmar": np.divide(cagr, mdd, out=np.zeros(candidates), where=mdd > 1e-12),
        "sharpe": sharpe,
        "entry_episodes": entries,
        "rebalance_orders": changes,
        "turnover": turnover,
        "fees_pct_initial": fees * 100,
        "funding_pct_initial": funding_paid * 100,
        "returns": returns,
    }


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
        block_dates, hourly["date"], macro_positions["dollar_flow_plus_regime_switch"]
    )
    macro_event = block_dates.isin(pd.to_datetime(hourly["date"])).to_numpy()
    oi_target_full = np.zeros(len(market))
    for trade in trades:
        oi_target_full[trade.entry_position : trade.exit_position] = trade.side
    oi_target = oi_target_full[indices]
    oi_event = np.r_[oi_target[0] != 0, oi_target[1:] != oi_target[:-1]]
    position_matrix = np.column_stack(
        [weight * macro_target + (1 - weight) * oi_target for weight in v1.WEIGHTS]
    )
    rebalance_matrix = np.column_stack(
        [
            (macro_event if weight > 0 else False) | (oi_event if weight < 1 else False)
            for weight in v1.WEIGHTS
        ]
    )
    reports = {}
    isolated_returns = None
    for cost in DESIGN["costs_per_side"]:
        stats = simulate_events(
            blocks,
            position_matrix,
            rebalance_matrix,
            cost=cost,
            start=oi_v1.START,
            end=oi_v1.EVAL_END,
        )
        reports[str(cost)] = {
            f"macro_{weight:g}_oi_{1-weight:g}": base.stats_row(stats, i)
            for i, weight in enumerate(v1.WEIGHTS)
        }
        if cost == 0.0006:
            isolated_returns = simulate_events(
                blocks,
                np.column_stack([macro_target, oi_target]),
                np.column_stack([macro_event, oi_event]),
                cost=cost,
                start=oi_v1.START,
                end=oi_v1.EVAL_END,
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
        "events": {
            "macro_updates": int(macro_event.sum()),
            "oi_entries_exits": int(oi_event.sum()),
            "oi_trades": len(trades),
        },
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
    print("events", result["events"], "correlation", result["base_cost_return_correlation"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    register() if args.freeze else run()
