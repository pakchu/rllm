"""Audit fixed macro-flow and OI-pullback sleeves on their common fresh window."""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from training import evaluate_macro_flow_fixed_fresh as macro_fresh
from training import evaluate_oi_divergence_fresh as oi_v1
from training import evaluate_oi_divergence_fresh_v2 as oi_v2
from training import search_macro_flow_alpha_combinations as macro_search
from training import search_meaningful_alpha_combinations as base

OUT = base.ROOT / "research/macro_oi_fresh_portfolio"
MACRO_REPORT = base.ROOT / "research/macro_flow_fresh/report.json"
OI_REPORT = base.ROOT / "research/oi_divergence_fresh_v2/report.json"
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
DESIGN = {
    "version": 1,
    "window": [oi_v1.START, oi_v1.EVAL_END],
    "sleeves": {
        "macro": "dollar_flow_plus_regime_switch, fixed 75/25 formula",
        "oi": "fixed four-gate OI divergence pullback, non-overlapping 8h longs",
    },
    "weights": WEIGHTS,
    "weight_meaning": "macro weight; OI weight is one minus macro weight",
    "execution": "five-minute target fractions; same-symbol signed positions net before cost and risk",
    "risk": "weighted sleeves sum to one and absolute net exposure is capped at 1x",
    "costs_per_side": [0.0, 0.0006, 0.001],
    "selection": "none; report every predeclared weight and do not name an optimized winner",
    "status": "descriptive exposed-period portfolio audit; no live authorization",
}


def register() -> dict:
    payload = {
        "design": DESIGN,
        "code_sha256": base.sha(__file__),
        "macro_engine_sha256": base.sha(macro_fresh.__file__),
        "oi_engine_sha256": base.sha(oi_v2.__file__),
        "macro_report_sha256": base.sha(MACRO_REPORT),
        "oi_report_sha256": base.sha(OI_REPORT),
    }
    path = OUT / "design.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise RuntimeError("Frozen portfolio audit drift")
    base.write_json(path, payload)
    return payload


def target_from_updates(dates: pd.Series, update_dates, values) -> np.ndarray:
    updates = pd.Series(np.asarray(values, dtype=float), index=pd.DatetimeIndex(update_dates))
    target = updates.reindex(pd.DatetimeIndex(dates)).ffill().fillna(0.0)
    return target.to_numpy(float)


def five_minute_blocks(market: pd.DataFrame, funding: pd.DataFrame):
    dates = pd.to_datetime(market.date)
    use = (dates >= pd.Timestamp(oi_v1.START).tz_localize(None)) & (
        dates.shift(-1) <= pd.Timestamp(oi_v1.EVAL_END).tz_localize(None)
    )
    indices = np.flatnonzero(use.to_numpy())
    indices = indices[indices + 1 < len(market)]
    transfers = np.zeros(len(market))
    funding_positions = np.searchsorted(dates.to_numpy(), funding.date.to_numpy(), side="right") - 1
    valid = (funding_positions >= 0) & (funding_positions < len(market))
    marks = pd.to_numeric(funding.loc[valid, "mark_price"], errors="coerce").to_numpy(float)
    fallback = ~np.isfinite(marks) | (marks <= 0)
    opens = market.open.to_numpy(float)
    marks[fallback] = opens[funding_positions[valid][fallback]]
    np.add.at(
        transfers,
        funding_positions[valid],
        funding.loc[valid, "funding_rate"].to_numpy(float) * marks,
    )
    return indices, {
        "date": dates.to_numpy()[indices],
        "end_date": dates.to_numpy()[indices + 1],
        "open": opens[indices],
        "end": opens[indices + 1],
        "high": market.high.to_numpy(float)[indices],
        "low": market.low.to_numpy(float)[indices],
        "funding": transfers[indices],
        "debit": np.maximum(transfers[indices], 0),
        "credit": np.minimum(transfers[indices], 0),
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
    indices, blocks = five_minute_blocks(market, funding)
    block_dates = pd.Series(pd.to_datetime(market.date).to_numpy()[indices])
    macro_target = target_from_updates(
        block_dates,
        hourly["date"],
        macro_positions["dollar_flow_plus_regime_switch"],
    )
    oi_target_full = np.zeros(len(market))
    for trade in trades:
        oi_target_full[trade.entry_position : trade.exit_position] = trade.side
    oi_target = oi_target_full[indices]

    position_matrix = np.column_stack(
        [weight * macro_target + (1 - weight) * oi_target for weight in WEIGHTS]
    )
    if np.max(np.abs(position_matrix)) > 1.0000001:
        raise RuntimeError("Portfolio net cap exceeded")
    reports = {}
    isolated_returns = None
    for cost in DESIGN["costs_per_side"]:
        stats = base.simulate(blocks, position_matrix, cost=cost, fine=False)
        reports[str(cost)] = {
            f"macro_{weight:g}_oi_{1-weight:g}": base.stats_row(stats, i)
            for i, weight in enumerate(WEIGHTS)
        }
        if cost == 0.0006:
            isolated = base.simulate(
                blocks,
                np.column_stack([macro_target, oi_target]),
                cost=cost,
                fine=False,
            )
            isolated_returns = isolated["returns"]
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
            "Both sleeves and this common window were already exposed before the portfolio audit.",
            "Weight results are descriptive and cannot be used to claim an optimized OOS portfolio.",
            "Five-minute bar MDD uses a conservative high-before-low envelope.",
        ],
    }
    base.write_json(OUT / "report.json", result)
    print(json.dumps(result["reports"]["0.0006"], indent=2), flush=True)
    print("return correlation", result["base_cost_return_correlation"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    register() if args.freeze else run()
