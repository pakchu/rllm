"""Search a frozen absorbed-basis recoil continuation after a pullback setup.

This family tests a specific two-stage mechanism rather than another weighted
feature blend:

1. the already-audited confirmed pullback setup occurs at an hourly decision;
2. one to six hours later, premium has recovered from the setup level while
   taker flow remains seller-dominant, price has stopped falling, and dollar
   participation is present; and
3. the continuation is entered at the next five-minute open.

The twelve combinations and every threshold are fixed on physically truncated
pre-2024 data.  A failed pre-2024 family must not open 2024+ observations.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.audit_confirmed_pullback_squeeze_live_parity import (
    AuditConfig,
    PRE2024_WINDOWS,
    SELECTION_END,
    _evaluate,
    _fit_active,
    _load_bundle,
    decision_mask,
    live_decision_features,
)


LAG_HOURS = (1, 3, 6)
SELL_QUANTILES = (0.30, 0.40)
PARTICIPATION_QUANTILES = (0.50, 0.70)
GRID = tuple(itertools.product(LAG_HOURS, SELL_QUANTILES, PARTICIPATION_QUANTILES))
FIT_START = pd.Timestamp("2020-07-01")
FIT_END = pd.Timestamp("2023-01-01")


@dataclass(frozen=True)
class AbsorbedBasisRecoilConfig(AuditConfig):
    output: str = "results/absorbed_basis_recoil_pullback_alpha_selection_2026-07-15.json"
    hold_bars: int = 576
    take_bps: int = 1_000
    stop_bps: int = 1_000_000


def grid_specs() -> tuple[dict[str, float | int | str], ...]:
    """Return the pre-registered twelve-rule family."""

    return tuple(
        {
            "name": f"abr_L{lag}_sell{sell}_part{part}",
            "lag_hours": int(lag),
            "sell_quantile": float(sell),
            "participation_quantile": float(part),
        }
        for lag, sell, part in GRID
    )


def _numeric(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame:
        raise ValueError(f"missing absorbed-basis feature: {name}")
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(float)


def _fit_quantile(values: np.ndarray, fit_clock: np.ndarray, quantile: float) -> float:
    sample = values[fit_clock & np.isfinite(values)]
    if len(sample) < 100:
        raise ValueError(f"insufficient fit observations: {len(sample)}")
    return float(np.quantile(sample, quantile))


def build_transition_inputs(
    market: pd.DataFrame,
    features: pd.DataFrame,
    dates: pd.Series,
    decisions: np.ndarray,
    *,
    fit_start: pd.Timestamp = FIT_START,
    fit_end: pd.Timestamp = FIT_END,
) -> dict[str, Any]:
    """Build causal transition inputs and fit all thresholds on train only."""

    if not (len(market) == len(features) == len(dates) == len(decisions)):
        raise ValueError("market, features, dates and decisions must have equal length")
    close = pd.to_numeric(market["close"], errors="coerce")
    quote = pd.to_numeric(market["quote_asset_volume"], errors="coerce")
    buy = pd.to_numeric(market["taker_buy_quote"], errors="coerce")

    # Market bar t is unavailable at its opening timestamp.  Shift by one bar
    # so these inputs agree with live_decision_features.
    log_close = np.log(close.where(close > 0.0))
    price_return_12 = (log_close - log_close.shift(12)).shift(1).to_numpy(float)
    taker_imbalance_12 = (
        (2.0 * buy / quote.replace(0.0, np.nan) - 1.0)
        .rolling(12, min_periods=12)
        .mean()
        .shift(1)
        .to_numpy(float)
    )
    participation = _numeric(features, "dollar_flow_rel_4h_30d")
    premium = _numeric(features, "premium_index")
    premium_available = _numeric(features, "premium_available") > 0.5

    parsed = pd.to_datetime(dates)
    fit_clock = (
        decisions
        & (parsed >= fit_start).to_numpy(bool)
        & (parsed < fit_end).to_numpy(bool)
    )
    thresholds = {
        "price_return_12_q40": _fit_quantile(price_return_12, fit_clock, 0.40),
        "taker_imbalance_12": {
            str(q): _fit_quantile(taker_imbalance_12, fit_clock, q)
            for q in SELL_QUANTILES
        },
        "dollar_flow_rel_4h_30d": {
            str(q): _fit_quantile(participation, fit_clock, q)
            for q in PARTICIPATION_QUANTILES
        },
    }
    return {
        "price_return_12": price_return_12,
        "taker_imbalance_12": taker_imbalance_12,
        "participation": participation,
        "premium": premium,
        "premium_available": premium_available,
        "thresholds": thresholds,
    }


def build_recoil_trigger(
    setup: np.ndarray,
    decisions: np.ndarray,
    *,
    premium: np.ndarray,
    premium_available: np.ndarray,
    taker_imbalance_12: np.ndarray,
    price_return_12: np.ndarray,
    participation: np.ndarray,
    lag_hours: int,
    taker_threshold: float,
    price_threshold: float,
    participation_threshold: float,
) -> np.ndarray:
    """Emit only a later hourly recoupling event after the latest setup.

    The current row is evaluated before it can become a setup anchor, which
    prevents a same-row setup/trigger and makes the one-hour minimum explicit.
    """

    arrays = (
        setup,
        decisions,
        premium,
        premium_available,
        taker_imbalance_12,
        price_return_12,
        participation,
    )
    if len({len(values) for values in arrays}) != 1:
        raise ValueError("all trigger inputs must have equal length")
    if int(lag_hours) not in LAG_HOURS:
        raise ValueError(f"unsupported lag_hours: {lag_hours}")

    active = np.zeros(len(setup), dtype=bool)
    latest_setup = -10**9
    maximum_gap = int(lag_hours) * 12
    for position in np.flatnonzero(decisions):
        position = int(position)
        gap = position - latest_setup
        if 12 <= gap <= maximum_gap:
            recoupled = (
                bool(premium_available[latest_setup])
                and bool(premium_available[position])
                and np.isfinite(premium[latest_setup])
                and np.isfinite(premium[position])
                and premium[position] > premium[latest_setup]
            )
            absorbed = (
                np.isfinite(taker_imbalance_12[position])
                and taker_imbalance_12[position] <= taker_threshold
                and np.isfinite(price_return_12[position])
                and price_return_12[position] >= price_threshold
            )
            participated = (
                np.isfinite(participation[position])
                and participation[position] >= participation_threshold
            )
            active[position] = bool(recoupled and absorbed and participated)
        if setup[position]:
            latest_setup = position
    return active


def selection_passes(stats: dict[str, dict[str, Any]]) -> bool:
    """Apply the frozen count, stability, target, and risk contract."""

    count_ok = (
        stats["train"]["trades"] >= 50
        and stats["select_2023"]["trades"] >= 10
        and stats["select_2023_h1"]["trades"] >= 4
        and stats["select_2023_h2"]["trades"] >= 4
    )
    stable = all(
        stats[name]["absolute_return_pct"] > 0.0
        for name in (
            "train_2020h2",
            "train_2021",
            "train_2022",
            "select_2023_h1",
            "select_2023_h2",
        )
    )
    target = all(
        stats[name]["cagr_to_strict_mdd"] >= 3.0
        for name in ("train", "select_2023", "pre_2024")
    )
    risk = all(
        stats[name]["strict_mdd_pct"] <= 15.0
        for name in ("train", "select_2023", "pre_2024")
    )
    return bool(count_ok and stable and target and risk)


def run(cfg: AbsorbedBasisRecoilConfig) -> dict[str, Any]:
    market, raw_features, funding, source_hashes = _load_bundle(
        cfg,
        cutoff=SELECTION_END,
        premium_tolerance=cfg.live_premium_tolerance,
    )
    dates = pd.to_datetime(market["date"])
    if len(dates) and dates.max() >= pd.Timestamp(SELECTION_END):
        raise RuntimeError("future data entered the pre-2024 selection process")
    decisions = decision_mask(
        dates,
        "live_hour_signal_bar",
        window_size=cfg.window_size,
    )
    features = live_decision_features(raw_features)
    setup, setup_thresholds = _fit_active(features, dates, decisions)
    inputs = build_transition_inputs(market, features, dates, decisions)
    thresholds = inputs["thresholds"]

    rows: list[dict[str, Any]] = []
    for spec in grid_specs():
        sell_q = str(spec["sell_quantile"])
        participation_q = str(spec["participation_quantile"])
        active = build_recoil_trigger(
            setup,
            decisions,
            premium=inputs["premium"],
            premium_available=inputs["premium_available"],
            taker_imbalance_12=inputs["taker_imbalance_12"],
            price_return_12=inputs["price_return_12"],
            participation=inputs["participation"],
            lag_hours=int(spec["lag_hours"]),
            taker_threshold=float(thresholds["taker_imbalance_12"][sell_q]),
            price_threshold=float(thresholds["price_return_12_q40"]),
            participation_threshold=float(
                thresholds["dollar_flow_rel_4h_30d"][participation_q]
            ),
        )
        stats = _evaluate(
            market,
            funding,
            active,
            cfg,
            leverage=cfg.leverage,
            windows=PRE2024_WINDOWS,
            hold_bars=cfg.hold_bars,
            take_bps=cfg.take_bps,
            stop_bps=cfg.stop_bps,
        )
        ratios = [
            stats[name]["cagr_to_strict_mdd"]
            for name in ("train", "select_2023", "pre_2024")
        ]
        rows.append(
            {
                **spec,
                "activation_count": int(active.sum()),
                "selection_passed": selection_passes(stats),
                "score": [float(min(ratios)), float(np.median(ratios))],
                "stats": stats,
            }
        )
    rows.sort(
        key=lambda row: (row["selection_passed"], *row["score"]),
        reverse=True,
    )
    passed = [row for row in rows if row["selection_passed"]]
    result = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "verdict": "CANDIDATE_PRE_OOS" if passed else "REJECTED_PRE_OOS",
        "future_opened": False,
        "protocol": {
            "physical_cutoff": SELECTION_END,
            "threshold_fit": "2020-07-01 through 2022-12-31 only",
            "selection": "2023 with both halves required",
            "family_size": len(GRID),
            "entry": "completed hourly t signal, next five-minute open",
            "exit": {
                "hold_bars": cfg.hold_bars,
                "take_bps": cfg.take_bps,
                "stop_bps": cfg.stop_bps,
            },
            "cost_per_side": cfg.fee_rate + cfg.slippage_rate,
            "funding": "realized funding cash flows",
            "strict_mdd": "global/pre-entry HWM plus position-wide favorable then adverse envelope",
            "stop_rule": "stop before take-profit when both touch",
            "selection_target": "train, 2023, and combined pre-2024 CAGR/strict-MDD >= 3; strict MDD <= 15%",
        },
        "source_prefix_hashes": source_hashes,
        "setup": {
            "confirmed_pullback_count": int(setup.sum()),
            "thresholds": setup_thresholds,
        },
        "transition_thresholds": thresholds,
        "grid": rows,
        "selected": passed[0] if passed else None,
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def parse_args() -> AbsorbedBasisRecoilConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=AbsorbedBasisRecoilConfig.input_csv)
    parser.add_argument("--funding-csv", default=AbsorbedBasisRecoilConfig.funding_csv)
    parser.add_argument("--premium-csv", default=AbsorbedBasisRecoilConfig.premium_csv)
    parser.add_argument("--output", default=AbsorbedBasisRecoilConfig.output)
    args = parser.parse_args()
    return AbsorbedBasisRecoilConfig(
        input_csv=args.input_csv,
        funding_csv=args.funding_csv,
        premium_csv=args.premium_csv,
        output=args.output,
    )


def main() -> None:
    result = run(parse_args())
    best = result["grid"][0]
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "future_opened": result["future_opened"],
                "best": {
                    "name": best["name"],
                    "selection_passed": best["selection_passed"],
                    "stats": {
                        name: best["stats"][name]
                        for name in ("train", "select_2023", "pre_2024")
                    },
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
