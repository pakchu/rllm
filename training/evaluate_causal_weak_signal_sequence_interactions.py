"""Evaluate primary-only causal interactions between frozen weak signals.

The experiment is preregistered in
``docs/causal-weak-signal-sequence-interactions-preregistration-2026-07-15.md``.
It never adds two sleeves.  A frozen primary antecedent only admits or rejects a
trade from another frozen primary trigger schedule.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import evaluate_cash_sponsored_perp_rejection as cspr_ev
from training import preregister_cash_sponsored_perp_rejection as cspr
from training.evaluate_metaorder_fragmentation_impact_curvature import (
    weekly_cluster_sign_flip,
)
from training.evaluate_weak_signal_feature_ensemble import (
    build_family_schedules,
    load_execution_market_and_funding,
)
from training.strict_bar_backtest import _trade_stats


PREREGISTRATION_DOCUMENT = Path(
    "docs/causal-weak-signal-sequence-interactions-preregistration-2026-07-15.md"
)
WINDOWS: dict[str, tuple[str, str]] = {
    "train": ("2020-01-01", "2023-01-01"),
    "train2020": ("2020-01-01", "2021-01-01"),
    "train2021": ("2021-01-01", "2022-01-01"),
    "train2022": ("2022-01-01", "2023-01-01"),
    "dev2023": ("2023-01-01", "2024-01-01"),
    "dev2023_h1": ("2023-01-01", "2023-07-01"),
    "dev2023_h2": ("2023-07-01", "2024-01-01"),
}
CASH_FAMILIES = ("cspr", "catch", "clasp")
DERIVATIVE_FAMILIES = ("umfr", "luri")


@dataclass(frozen=True)
class Config:
    output: str = (
        "results/causal_weak_signal_sequence_interactions_pre2024_2026-07-15.json"
    )
    docs_output: str = (
        "docs/causal-weak-signal-sequence-interactions-result-2026-07-15.md"
    )
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    stress_fee_plus_slippage_rate: float = 0.0008
    cluster_permutations: int = 100_000
    cluster_seed: int = 20_260_715
    minimum_train_trades: int = 80
    minimum_train_year_trades: int = 15
    minimum_dev_trades: int = 40
    minimum_dev_half_trades: int = 12
    minimum_gross_underlying_bp: float = 12.0
    maximum_strict_mdd_pct: float = 15.0
    minimum_cagr_to_strict_mdd: float = 3.0


@dataclass(frozen=True)
class InteractionSpec:
    operator: str
    trigger: str
    antecedent: str
    lookback_bars: int
    long_only: bool = False
    veto_opposite_derivative: bool = False

    @property
    def name(self) -> str:
        suffix = "_long" if self.long_only else ""
        veto = "_veto" if self.veto_opposite_derivative else ""
        return (
            f"{self.operator}_{self.antecedent}_to_{self.trigger}_"
            f"lb{self.lookback_bars}{suffix}{veto}"
        )


def candidate_specs() -> list[InteractionSpec]:
    specs: list[InteractionSpec] = []
    for trigger in DERIVATIVE_FAMILIES:
        for antecedent in CASH_FAMILIES:
            for lookback in (12, 36):
                specs.append(InteractionSpec("o1", trigger, antecedent, lookback))
    for trigger in CASH_FAMILIES:
        for antecedent in CASH_FAMILIES:
            if trigger == antecedent:
                continue
            for lookback in (12, 36):
                specs.append(InteractionSpec("o2", trigger, antecedent, lookback))
    for antecedent in CASH_FAMILIES:
        for lookback in (12, 36):
            specs.append(
                InteractionSpec(
                    "o3",
                    "rift",
                    antecedent,
                    lookback,
                    long_only=True,
                    veto_opposite_derivative=True,
                )
            )
    for trigger in DERIVATIVE_FAMILIES:
        for lookback in (12, 36):
            specs.append(
                InteractionSpec(
                    "o4", trigger, "rift", lookback, long_only=True
                )
            )
    return specs


def load_primary_schedules() -> dict[str, pd.DataFrame]:
    support = json.loads(cspr_ev.PREREGISTRATION_RESULT.read_text())
    cspr_cfg = cspr.Config()
    cspr_frame, source = cspr.load_causal_frame(cspr_cfg)
    controls, primary = cspr_ev.verify_signal_replay(
        cspr_frame, cspr_cfg, support, source
    )
    schedules = {
        "cspr": cspr_ev.build_control_schedules(
            cspr_frame, controls, primary, cspr_cfg
        )["primary"]
    }
    for family in ("catch", "clasp", "luri", "rift", "umfr"):
        schedules[family] = build_family_schedules(family)["primary"]
    return schedules


def _last_signal_state(
    schedule: pd.DataFrame, frame_length: int
) -> tuple[np.ndarray, np.ndarray]:
    side_at = np.zeros(frame_length, dtype=np.int8)
    for row in schedule.itertuples(index=False):
        position = int(row.signal_position)
        if not 0 <= position < frame_length:
            raise ValueError("signal position is outside the market frame")
        side_at[position] = int(row.side)
    positions = np.arange(frame_length, dtype=np.int64)
    last_position = np.maximum.accumulate(
        np.where(side_at != 0, positions, -1)
    )
    last_side = np.zeros(frame_length, dtype=np.int8)
    available = last_position >= 0
    last_side[available] = side_at[last_position[available]]
    return last_position, last_side


def _recent_relation_mask(
    trigger: pd.DataFrame,
    antecedent_state: tuple[np.ndarray, np.ndarray],
    *,
    minimum_age_bars: int,
    maximum_age_bars: int,
    same_side: bool,
) -> np.ndarray:
    if minimum_age_bars < 0 or maximum_age_bars < minimum_age_bars:
        raise ValueError("invalid causal age interval")
    trigger_position = trigger["signal_position"].to_numpy(np.int64)
    trigger_side = trigger["side"].to_numpy(np.int8)
    last_position, last_side = antecedent_state
    antecedent_position = last_position[trigger_position]
    age = trigger_position - antecedent_position
    expected_product = 1 if same_side else -1
    return (
        (antecedent_position >= 0)
        & (age >= minimum_age_bars)
        & (age <= maximum_age_bars)
        & ((trigger_side * last_side[trigger_position]) == expected_product)
    )


def build_interaction_schedule(
    spec: InteractionSpec,
    schedules: dict[str, pd.DataFrame],
    *,
    frame_length: int,
) -> pd.DataFrame:
    trigger = schedules[spec.trigger].copy().reset_index(drop=True)
    states = {
        family: _last_signal_state(schedule, frame_length)
        for family, schedule in schedules.items()
    }
    admitted = _recent_relation_mask(
        trigger,
        states[spec.antecedent],
        minimum_age_bars=0,
        maximum_age_bars=spec.lookback_bars,
        same_side=True,
    )
    if spec.long_only:
        admitted &= trigger["side"].to_numpy(np.int8) == 1
    if spec.veto_opposite_derivative:
        for family in DERIVATIVE_FAMILIES:
            admitted &= ~_recent_relation_mask(
                trigger,
                states[family],
                minimum_age_bars=0,
                maximum_age_bars=12,
                same_side=False,
            )
    selected = trigger.loc[admitted].copy().reset_index(drop=True)
    if not selected.empty:
        selected["branch"] = spec.name
    return selected


def _slice_schedule(
    schedule: pd.DataFrame, *, start: str, end: str
) -> pd.DataFrame:
    if schedule.empty:
        return schedule.copy()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    signal = pd.to_datetime(schedule["signal_date"], errors="raise")
    entry = pd.to_datetime(schedule["entry_date"], errors="raise")
    exit_ = pd.to_datetime(schedule["exit_date"], errors="raise")
    inside = (
        signal.ge(start_ts)
        & signal.lt(end_ts)
        & entry.ge(start_ts)
        & entry.lt(end_ts)
        & exit_.ge(start_ts)
        & exit_.lt(end_ts)
    )
    return schedule.loc[inside].reset_index(drop=True)


def simulate_schedule(
    frame: pd.DataFrame,
    funding: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    start: str,
    end: str,
    cfg: Config,
    fee_plus_slippage_rate: float | None = None,
    cluster_permutations: int = 0,
) -> dict[str, Any]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    schedule = _slice_schedule(schedule, start=start, end=end)
    cost_rate = (
        cfg.fee_rate + cfg.slippage_rate
        if fee_plus_slippage_rate is None
        else fee_plus_slippage_rate
    )
    per_side_cost = cost_rate * cfg.leverage
    opens = frame["open"].to_numpy(float)
    highs = frame["high"].to_numpy(float)
    lows = frame["low"].to_numpy(float)
    dates = pd.to_datetime(frame["date"])
    funding_times = funding["funding_time_ms"].to_numpy(np.int64)
    funding_rates = funding["funding_rate"].to_numpy(float)
    equity = 1.0
    peak = 1.0
    strict_mdd = 0.0
    previous_exit = -1
    trade_returns: list[float] = []
    gross_returns: list[float] = []
    entry_dates: list[str] = []
    sides: list[int] = []
    for row in schedule.itertuples(index=False):
        signal = int(row.signal_position)
        entry = int(row.entry_position)
        exit_ = int(row.exit_position)
        side = int(row.side)
        hold = int(row.hold_bars)
        if side not in (-1, 1):
            raise ValueError("trade side must be long or short")
        if entry != signal + 1 or exit_ != entry + hold:
            raise ValueError("trade does not preserve next-open entry and frozen hold")
        if entry < previous_exit:
            raise ValueError("filtered trigger trades overlap")
        if not (
            start_ts <= dates.iloc[signal] < end_ts
            and start_ts <= dates.iloc[entry] < end_ts
            and start_ts <= dates.iloc[exit_] < end_ts
        ):
            raise ValueError("trade crosses the requested split")
        entry_price = float(opens[entry])
        exit_price = float(opens[exit_])
        held_high = float(np.max(highs[entry:exit_]))
        held_low = float(np.min(lows[entry:exit_]))
        entry_ms = int(dates.iloc[entry].value // 1_000_000)
        exit_ms = int(dates.iloc[exit_].value // 1_000_000)
        left = int(np.searchsorted(funding_times, entry_ms, side="left"))
        right = int(np.searchsorted(funding_times, exit_ms, side="right"))
        factors = 1.0 - cfg.leverage * side * funding_rates[left:right]
        if not np.isfinite(factors).all() or (factors <= 0.0).any():
            raise ValueError("invalid realized funding factor")
        funding_factor = float(np.prod(factors, dtype=float))
        funding_debit_factor = float(np.prod(np.minimum(factors, 1.0), dtype=float))
        entry_equity = equity
        equity *= 1.0 - per_side_cost
        strict_mdd = max(strict_mdd, 1.0 - equity / peak)
        post_entry_equity = equity
        favorable_price = held_high if side > 0 else held_low
        adverse_price = held_low if side > 0 else held_high
        favorable_equity = max(
            0.0,
            post_entry_equity
            * (1.0 + cfg.leverage * side * (favorable_price / entry_price - 1.0)),
        )
        intratrade_peak = max(peak, favorable_equity)
        adverse_equity = max(
            0.0,
            post_entry_equity
            * funding_debit_factor
            * (1.0 + cfg.leverage * side * (adverse_price / entry_price - 1.0)),
        )
        strict_mdd = max(strict_mdd, 1.0 - adverse_equity / intratrade_peak)
        peak = max(peak, intratrade_peak)
        raw_return = side * (exit_price / entry_price - 1.0)
        equity = post_entry_equity * max(0.0, 1.0 + cfg.leverage * raw_return)
        equity *= funding_factor * (1.0 - per_side_cost)
        strict_mdd = max(strict_mdd, 1.0 - equity / peak)
        peak = max(peak, equity)
        trade_returns.append(equity / entry_equity - 1.0)
        gross_returns.append(raw_return)
        entry_dates.append(str(dates.iloc[entry]))
        sides.append(side)
        previous_exit = exit_
    years = (end_ts - start_ts).total_seconds() / (365.25 * 86_400.0)
    absolute_return = (equity - 1.0) * 100.0
    cagr = (equity ** (1.0 / years) - 1.0) * 100.0 if equity > 0.0 else -100.0
    strict_mdd_pct = strict_mdd * 100.0
    cluster = (
        weekly_cluster_sign_flip(
            trade_returns,
            entry_dates,
            permutations=cluster_permutations,
            seed=cfg.cluster_seed,
        )
        if cluster_permutations
        else None
    )
    return {
        "absolute_return_pct": float(absolute_return),
        "cagr_pct": float(cagr),
        "strict_mdd_pct": float(strict_mdd_pct),
        "cagr_to_strict_mdd": (
            float(cagr / strict_mdd_pct) if strict_mdd_pct > 1e-12 else 0.0
        ),
        "trade_count": len(trade_returns),
        "long_count": int(sum(side > 0 for side in sides)),
        "short_count": int(sum(side < 0 for side in sides)),
        "mean_gross_underlying_move_bp": (
            float(np.mean(gross_returns) * 10_000.0) if gross_returns else 0.0
        ),
        "trade_statistics": _trade_stats(trade_returns),
        "weekly_cluster_sign_flip": cluster,
    }


def _basic_train_failures(
    metrics: dict[str, Any], yearly: dict[str, dict[str, Any]], cfg: Config
) -> list[str]:
    failures: list[str] = []
    if metrics["absolute_return_pct"] <= 0.0:
        failures.append("non-positive train absolute return")
    if metrics["cagr_to_strict_mdd"] < cfg.minimum_cagr_to_strict_mdd:
        failures.append("train CAGR/strict-MDD below 3")
    if metrics["strict_mdd_pct"] > cfg.maximum_strict_mdd_pct:
        failures.append("train strict MDD above 15%")
    if metrics["trade_count"] < cfg.minimum_train_trades:
        failures.append("fewer than 80 train trades")
    if (
        metrics["mean_gross_underlying_move_bp"]
        <= cfg.minimum_gross_underlying_bp
    ):
        failures.append("train mean gross move not above 12 bp")
    for year in ("train2020", "train2021", "train2022"):
        if yearly[year]["trade_count"] < cfg.minimum_train_year_trades:
            failures.append(f"{year}: fewer than 15 trades")
        if yearly[year]["trade_statistics"]["mean_trade_ret_pct"] <= 0.0:
            failures.append(f"{year}: non-positive mean net trade return")
    return failures


def run(cfg: Config) -> dict[str, Any]:
    if not PREREGISTRATION_DOCUMENT.is_file():
        raise ValueError("interaction preregistration document is missing")
    frame, funding = load_execution_market_and_funding()
    schedules = load_primary_schedules()
    rows: list[dict[str, Any]] = []
    for spec in candidate_specs():
        schedule = build_interaction_schedule(spec, schedules, frame_length=len(frame))
        train = simulate_schedule(
            frame,
            funding,
            schedule,
            start=WINDOWS["train"][0],
            end=WINDOWS["train"][1],
            cfg=cfg,
        )
        yearly = {
            name: simulate_schedule(
                frame, funding, schedule, start=start, end=end, cfg=cfg
            )
            for name, (start, end) in WINDOWS.items()
            if name.startswith("train20")
        }
        failures = _basic_train_failures(train, yearly, cfg)
        if not failures:
            train = simulate_schedule(
                frame,
                funding,
                schedule,
                start=WINDOWS["train"][0],
                end=WINDOWS["train"][1],
                cfg=cfg,
                cluster_permutations=cfg.cluster_permutations,
            )
            cluster = train["weekly_cluster_sign_flip"]
            if cluster is None or cluster["p_value_one_sided"] >= 0.10:
                failures.append("train weekly-cluster p-value not below 0.10")
        rows.append(
            {
                "spec": asdict(spec),
                "name": spec.name,
                "global_trade_count": int(len(schedule)),
                "train": train,
                "train_years": yearly,
                "train_failures": failures,
                "passes_train": not failures,
            }
        )
    rows.sort(
        key=lambda row: (
            row["passes_train"],
            min(
                row["train_years"][year]["trade_statistics"][
                    "mean_trade_ret_pct"
                ]
                for year in ("train2020", "train2021", "train2022")
            ),
            row["train"]["cagr_to_strict_mdd"],
            row["train"]["trade_count"],
        ),
        reverse=True,
    )
    passing = [row for row in rows if row["passes_train"]]
    result: dict[str, Any] = {
        "protocol": {
            "name": "primary-only causal weak-signal sequence interactions",
            "fit_window": "2020-2022 only",
            "2023_role": "development confirmation; already exposed globally",
            "sealed_windows": ["2024", "2025", "2026"],
            "2023_opened": False,
            "simple_sleeve_blending": False,
            "controls_promotable": False,
        },
        "config": asdict(cfg),
        "tested_candidates": len(rows),
        "train_qualifier_count": len(passing),
        "train_champion": passing[0] if passing else None,
        "candidates": rows,
    }
    if passing:
        champion = passing[0]
        spec = InteractionSpec(**champion["spec"])
        schedule = build_interaction_schedule(spec, schedules, frame_length=len(frame))
        dev = {
            name: simulate_schedule(
                frame,
                funding,
                schedule,
                start=start,
                end=end,
                cfg=cfg,
                cluster_permutations=(
                    cfg.cluster_permutations if name == "dev2023" else 0
                ),
            )
            for name, (start, end) in WINDOWS.items()
            if name.startswith("dev2023")
        }
        stress = simulate_schedule(
            frame,
            funding,
            schedule,
            start=WINDOWS["dev2023"][0],
            end=WINDOWS["dev2023"][1],
            cfg=cfg,
            fee_plus_slippage_rate=cfg.stress_fee_plus_slippage_rate,
        )
        result["protocol"]["2023_opened"] = True
        result["development_confirmation"] = {"metrics": dev, "stress": stress}
    return result


def write_doc(result: dict[str, Any], path: Path) -> None:
    champion = result["train_champion"]
    lines = [
        "# Causal weak-signal sequence interactions — result (2026-07-15)",
        "",
        "This experiment uses primary causal clocks only and filters one trigger trade; it does not blend sleeves.",
        "",
        f"- Tested candidates: {result['tested_candidates']}",
        f"- Train qualifiers: {result['train_qualifier_count']}",
        f"- 2023 opened by the formal evaluator: {result['protocol']['2023_opened']}",
        "- 2024+ opened: no",
        "",
    ]
    if champion is None:
        lines += [
            "## Decision",
            "",
            "**Reject this primary-only sequence family at the train gate.** No candidate met the preregistered 2020–2022 return, CAGR/MDD, strict-MDD, gross-edge, trade-count, and yearly-stability requirements. The evaluator therefore did not use 2023 to rescue a failed train candidate.",
            "",
            "The result implies that full primary events are too sparse and/or too weak after costs. The next experiment should combine causal component states around a denser, independently frozen trigger rather than promote stale or component-removal controls.",
        ]
    else:
        train = champion["train"]
        lines += [
            "## Train champion",
            "",
            f"- Rule: `{champion['name']}`",
            f"- Absolute return: {train['absolute_return_pct']:.2f}%",
            f"- CAGR / strict MDD: {train['cagr_to_strict_mdd']:.2f}",
            f"- Strict MDD: {train['strict_mdd_pct']:.2f}%",
            f"- Trades: {train['trade_count']}",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--docs-output", default=Config.docs_output)
    args = parser.parse_args()
    cfg = Config(output=args.output, docs_output=args.docs_output)
    result = run(cfg)
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_doc(result, Path(cfg.docs_output))
    print(
        json.dumps(
            {
                "tested_candidates": result["tested_candidates"],
                "train_qualifier_count": result["train_qualifier_count"],
                "train_champion": (
                    result["train_champion"]["name"]
                    if result["train_champion"]
                    else None
                ),
                "2023_opened": result["protocol"]["2023_opened"],
                "output": cfg.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
