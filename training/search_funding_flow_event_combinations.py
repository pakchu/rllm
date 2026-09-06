"""Search funding-event and aggressive-flow interactions with fixed robust ranking.

Signals use the funding event observed by completed UTC hour T and execute at
T+5 minutes.  Selection is confined to six half-years in 2021--2023; all
2024+ output is report-only and cannot replace a frozen finalist.
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np

from training import search_meaningful_alpha_combinations as base
from training import search_regime_diverse_alpha_combinations as regime

OUT = base.ROOT / "research/funding_flow_event_combinations"
DESIGN = {
    "version": 1,
    "selection": "2021--2023 six half-years using inherited robust rank",
    "reports": "2024, 2025, 2026H1 and combined; report-only exposed history",
    "source": "BTCUSDT 5m market plus realized Binance funding events",
    "causality": "funding event known by completed UTC hour T; execute exact T+5m open",
    "mechanisms": [
        "crowded funding plus same-direction price/flow reversal",
        "funding shock contradicted by aggressive flow",
        "funding sign flip confirmed by aggressive flow",
        "adverse funding relief within established trend",
        "funding acceleration exhaustion at price displacement",
    ],
    "level_thresholds": [0.0001, 0.0002, 0.0004],
    "delta_thresholds": [0.00008, 0.00015, 0.0003],
    "flow_thresholds": [0.01, 0.02, 0.04],
    "hold_hours": [8, 16, 24],
    "side_modes": ["both", "long", "short"],
    "sizing": ["raw", "20pct annual volatility target clipped 0.1--1.0"],
    "portfolio": "two distinct paths per family; cross-family pairs at 25/50/75 percent",
    "costs_per_side": [0.0, 0.0006, 0.001],
    "risk": "overlap allowed; signed same-symbol sleeves net before 1x cap, costs and funding",
    "gates_removed": ["fee ratio", "trade frequency", "position overlap"],
    "limitations": [
        "2024+ history already exposed",
        "hourly event visibility is a conservative settlement-time proxy, not exchange message latency",
        "no capacity, liquidation or tick-order model",
    ],
}


def register() -> dict:
    payload = {
        "design": DESIGN,
        "code_sha256": base.sha(__file__),
        "engine_sha256": base.sha(base.__file__),
        "rank_sha256": base.sha(regime.__file__),
        "market_sha256": base.sha(base.MARKET),
        "funding_sha256": base.sha(base.FUNDING),
    }
    path = OUT / "design.json"
    if path.exists() and json.loads(path.read_text()) != payload:
        raise RuntimeError("Frozen funding-flow study drift")
    base.write_json(path, payload)
    return payload


def event_hold(raw: np.ndarray, hours: int) -> np.ndarray:
    """Hold a nonzero event pulse for a fixed number of following hourly rows."""
    raw = np.nan_to_num(np.asarray(raw, dtype=float))
    out = np.zeros_like(raw)
    active = 0.0
    remaining = 0
    for i, value in enumerate(raw):
        if value:
            active = float(value)
            remaining = hours
        if remaining:
            out[i] = active
            remaining -= 1
    return out


def candidates(x):
    funding = x.funding.to_numpy(float)
    delta = x.funding.diff().to_numpy(float)
    previous = x.funding.shift(1).to_numpy(float)
    event = x.funding.ne(x.funding.shift()).to_numpy() & (x.funding_available.to_numpy() > 0.5)
    flow = x.flow6.to_numpy(float)
    trend = np.sign(x.mom168.to_numpy(float))
    strong = np.abs(x.mom168.to_numpy(float)) > 0.75
    displacement = np.sign(x.z24.to_numpy(float))
    displaced = np.abs(x.z24.to_numpy(float)) > 1.0
    vol_size = np.clip(
        np.divide(
            0.20,
            x.vol24.to_numpy(float) * np.sqrt(8766),
            out=np.ones(len(x)),
            where=x.vol24.to_numpy(float) > 0,
        ),
        0.1,
        1.0,
    )
    signals: dict[str, np.ndarray] = {}
    specs: dict[str, dict] = {}

    def add(name, family, pulse, rationale):
        pulse = np.where(event, pulse, 0.0)
        for side in DESIGN["side_modes"]:
            sided = pulse if side == "both" else np.maximum(pulse, 0) if side == "long" else np.minimum(pulse, 0)
            for hold in DESIGN["hold_hours"]:
                held = event_hold(sided, hold)
                for sizing, multiplier in (("raw", 1.0), ("vol20", vol_size)):
                    key = f"{name}__{side}__h{hold}__{sizing}"
                    signals[key] = np.clip(held * multiplier, -1, 1)
                    specs[key] = {
                        "family": family,
                        "side": side,
                        "hold_hours": hold,
                        "sizing": sizing,
                        "rationale": rationale,
                    }

    for level in DESIGN["level_thresholds"]:
        for flow_threshold in DESIGN["flow_thresholds"]:
            crowded = np.abs(funding) >= level
            moving_with_crowd = displacement == np.sign(funding)
            add(
                f"crowd_reversal_l{level:g}_f{flow_threshold:g}",
                "crowd_reversal",
                np.where(crowded & displaced & moving_with_crowd & (np.sign(funding) * flow < -flow_threshold), -np.sign(funding), 0),
                "fade a funding-aligned displacement only after aggressive flow contradicts the crowd",
            )
            add(
                f"level_flow_contradiction_l{level:g}_f{flow_threshold:g}",
                "level_flow_contradiction",
                np.where(crowded & (np.abs(flow) > flow_threshold) & (np.sign(flow) * funding < 0), np.sign(flow), 0),
                "follow aggressive flow when it opposes the sign of an extreme funding level",
            )
            add(
                f"level_flow_exhaustion_l{level:g}_f{flow_threshold:g}",
                "level_flow_exhaustion",
                np.where(crowded & displaced & (displacement == np.sign(funding)) & (np.sign(flow) == displacement), -displacement, 0),
                "fade a price, flow and funding consensus as leveraged exhaustion",
            )

    for change in DESIGN["delta_thresholds"]:
        for flow_threshold in DESIGN["flow_thresholds"]:
            shocked = np.abs(delta) >= change
            flip = np.sign(funding) != np.sign(previous)
            add(
                f"shock_flow_contradiction_d{change:g}_f{flow_threshold:g}",
                "shock_flow_contradiction",
                np.where(shocked & (np.abs(flow) > flow_threshold) & (np.sign(flow) * delta < 0), np.sign(flow), 0),
                "follow aggressive flow when a large funding change points the other way",
            )
            add(
                f"funding_flip_flow_d{change:g}_f{flow_threshold:g}",
                "funding_flip_flow",
                np.where(shocked & flip & (np.abs(flow) > flow_threshold), np.sign(flow), 0),
                "use aggressive flow to route a large funding sign transition",
            )
            relief = shocked & strong & (trend * delta < 0)
            add(
                f"trend_funding_relief_d{change:g}_f{flow_threshold:g}",
                "trend_funding_relief",
                np.where(relief & (trend * flow > flow_threshold), trend, 0),
                "continue a flow-confirmed trend when its adverse funding pressure falls",
            )
            add(
                f"shock_displacement_fade_d{change:g}_f{flow_threshold:g}",
                "shock_displacement_fade",
                np.where(shocked & displaced & (displacement == np.sign(delta)) & (np.sign(delta) * flow < flow_threshold), -displacement, 0),
                "fade displacement when funding accelerates with price but aggressive flow does not confirm",
            )
    return signals, specs


def run() -> None:
    frozen = json.loads((OUT / "design.json").read_text())
    if frozen != register():
        raise RuntimeError("Registration changed")
    market, funding = base.load_sources()
    x = base.features(market, funding)
    x, data, receipt = base.execution_blocks(market, funding, x)
    signals, specs = candidates(x)
    names = list(signals)
    positions = np.column_stack([signals[name] for name in names])
    scores, selection_stats, halves = regime.rank(data, positions)
    selection = base.window_mask(data, "2021-01-01", "2024-01-01")

    seen: set[str] = set()
    family_counts: dict[str, int] = {}
    champions: list[int] = []
    for index in np.argsort(-scores, kind="stable"):
        family = specs[names[index]]["family"]
        digest = hashlib.sha256(positions[selection, index].tobytes()).hexdigest()
        if family_counts.get(family, 0) >= 2 or digest in seen:
            continue
        seen.add(digest)
        family_counts[family] = family_counts.get(family, 0) + 1
        champions.append(int(index))

    original_positions = positions
    for left_offset, left in enumerate(champions):
        for right in champions[left_offset + 1 :]:
            if specs[names[left]]["family"] == specs[names[right]]["family"]:
                continue
            for weight in (0.25, 0.5, 0.75):
                key = f"mix_{left}_{right}_{weight}"
                signals[key] = weight * original_positions[:, left] + (1 - weight) * original_positions[:, right]
                specs[key] = {
                    "family": "portfolio",
                    "components": {names[left]: weight, names[right]: 1 - weight},
                }

    names = list(signals)
    positions = np.column_stack([signals[name] for name in names])
    scores, selection_stats, halves = regime.rank(data, positions)
    order = np.argsort(-scores, kind="stable")
    finalists = list(map(int, order[:5]))
    pure = next(int(i) for i in order if specs[names[i]]["family"] != "portfolio")
    if pure not in finalists:
        finalists.append(pure)
    freeze = {
        "selection": "six halves 2021--2023",
        "report_reranking": False,
        "candidates": len(names),
        "family_champions": len(champions),
        "top": [
            {
                "name": names[i],
                "spec": specs[names[i]],
                "score": float(scores[i]),
                "selection": base.stats_row(selection_stats, i),
                "half_sharpes": halves[:, i].tolist(),
            }
            for i in finalists
        ],
    }
    base.write_json(OUT / "selection_freeze.json", freeze)

    final_positions = np.column_stack([positions[:, finalists], np.ones(len(x)), np.zeros(len(x))])
    final_names = [names[i] for i in finalists] + ["control_long", "control_cash"]
    reports = {}
    windows = (
        ("report2024", "2024-01-01", "2025-01-01"),
        ("report2025", "2025-01-01", "2026-01-01"),
        ("report2026", "2026-01-01", "2026-06-01"),
        ("combined", "2024-01-01", "2026-06-01"),
    )
    for window, start, end in windows:
        mask = base.window_mask(data, start, end)
        reports[window] = {}
        for cost in DESIGN["costs_per_side"]:
            stats = base.simulate(base.subset(data, mask), final_positions[mask], cost=cost, fine=True)
            reports[window][str(cost)] = {
                name: base.stats_row(stats, i) for i, name in enumerate(final_names)
            }
    result = {
        "registration": frozen,
        "source_receipt": receipt,
        "freeze": freeze,
        "reports": reports,
        "inventory": [
            {
                "name": names[i],
                "spec": specs[names[i]],
                "score": float(scores[i]),
                "selection": base.stats_row(selection_stats, i),
            }
            for i in order
        ],
        "live_enabled": False,
    }
    base.write_json(OUT / "report.json", result)
    base.write_json(
        OUT / "research_config.json",
        {
            "research_only": True,
            "live_enabled": False,
            "winner": freeze["top"][0],
            "overlap_allowed": True,
            "long_short_offset": True,
            "net_exposure_cap": 1.0,
            "fee_ratio_gate": False,
            "frequency_gate": False,
        },
    )
    print(f"candidates {len(names)}; family champions {len(champions)}", flush=True)
    for row in freeze["top"]:
        name = row["name"]
        print(name, json.dumps(row["spec"]), flush=True)
        for window in reports:
            print(window, json.dumps(reports[window]["0.0006"][name]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    register() if args.freeze else run()
