"""Evaluate ensembles of previously meaningful weak alpha-feature sleeves.

This intentionally tests the user's thesis that profitability may come from a
combination of weak signals rather than one strong sparse signal.  It uses only
frozen/pre-2024 families and fits non-negative sleeve weights on 2020-2022, then
reports 2023/H1/H2 as holdout.  2024+ is not opened here.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import evaluate_cash_auction_transfer_catchup_handoff as catch_ev
from training import evaluate_cash_late_arrival_spillover_propagation as clasp_ev
from training import evaluate_cash_sponsored_perp_rejection as cspr_ev
from training import evaluate_leveraged_um_inventory_release_handoff as luri_ev
from training import evaluate_refill_inference_flow_topology as rift_ev
from training import evaluate_um_forced_flow_reversion as umfr_ev
from training import preregister_cash_auction_transfer_catchup_handoff as catch
from training import preregister_cash_late_arrival_spillover_propagation as clasp
from training import preregister_cash_sponsored_perp_rejection as cspr
from training import preregister_leveraged_um_inventory_release_handoff as luri
from training import preregister_refill_inference_flow_topology as rift
from training import preregister_um_forced_flow_reversion as umfr


WINDOWS: dict[str, tuple[str, str]] = {
    "train": ("2020-01-01", "2023-01-01"),
    "select2023": ("2023-01-01", "2024-01-01"),
    "select2023_h1": ("2023-01-01", "2023-07-01"),
    "select2023_h2": ("2023-07-01", "2024-01-01"),
}
FAMILY_MODULES: dict[str, Any] = {
    "cspr": cspr_ev,
    "rift": rift_ev,
    "catch": catch_ev,
    "luri": luri_ev,
    "clasp": clasp_ev,
    "umfr": umfr_ev,
}
SELECTION_RESULTS: dict[str, Path] = {
    "cspr": Path("results/cash_sponsored_perp_rejection_selection_2026-07-14.json"),
    "rift": Path("results/refill_inference_flow_topology_selection_2026-07-14.json"),
    "catch": Path(
        "results/cash_auction_transfer_catchup_handoff_selection_2026-07-14.json"
    ),
    "luri": Path(
        "results/leveraged_um_inventory_release_handoff_selection_2026-07-14.json"
    ),
    "clasp": Path(
        "results/cash_late_arrival_spillover_propagation_selection_2026-07-14.json"
    ),
    "umfr": Path("results/um_forced_flow_reversion_selection_2026-07-14.json"),
}


@dataclass(frozen=True)
class Config:
    output: str = "results/weak_signal_feature_ensemble_pre2024_2026-07-15.json"
    docs_output: str = "docs/weak-signal-feature-ensemble-pre2024-2026-07-15.md"
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    min_train_trades: int = 300
    min_train_gross_bp: float = 0.0
    max_candidates: int = 24
    gross_cap: float = 4.0
    min_nonzero_weight: float = 0.10
    weight_step: float = 0.05
    random_samples: int = 20_000
    seed: int = 20_260_715


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected object JSON: {path}")
    return value


def _slice_schedule(schedule: pd.DataFrame, *, start: str, end: str) -> pd.DataFrame:
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


def select_candidate_policies(cfg: Config) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, path in SELECTION_RESULTS.items():
        result = _read_json(path)
        train = result.get("windows", {}).get("train", {})
        select = result.get("windows", {}).get("select2023", {})
        for policy, metrics in train.items():
            if not isinstance(metrics, dict):
                continue
            gross = float(metrics.get("mean_gross_underlying_move_bp", 0.0))
            trades = int(metrics.get("trade_count", 0))
            if gross <= cfg.min_train_gross_bp or trades < cfg.min_train_trades:
                continue
            select_metrics = select.get(policy, {}) if isinstance(select, dict) else {}
            rows.append(
                {
                    "name": f"{family}:{policy}",
                    "family": family,
                    "policy": policy,
                    "train_gross_bp": gross,
                    "train_cagr_pct": float(metrics.get("cagr_pct", 0.0)),
                    "train_strict_mdd_pct": float(metrics.get("strict_mdd_pct", 0.0)),
                    "train_trades": trades,
                    "select2023_gross_bp": float(
                        select_metrics.get("mean_gross_underlying_move_bp", 0.0)
                    ),
                    "select2023_cagr_pct": float(select_metrics.get("cagr_pct", 0.0)),
                    "select2023_strict_mdd_pct": float(
                        select_metrics.get("strict_mdd_pct", 0.0)
                    ),
                    "select2023_trades": int(select_metrics.get("trade_count", 0)),
                }
            )
    rows.sort(
        key=lambda x: (
            x["train_gross_bp"],
            -x["train_strict_mdd_pct"],
            x["select2023_gross_bp"],
        ),
        reverse=True,
    )
    return rows[: cfg.max_candidates]


def load_execution_market_and_funding() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Reuse UMFR's frozen loaders; they check market/funding hashes and 2024 seal.
    _, market_manifest, funding_manifest = umfr_ev.verify_preregistration()
    signal_frame, _ = umfr.load_causal_frame()
    frame, _ = umfr_ev.load_execution_market(signal_frame, market_manifest)
    funding, _ = umfr_ev.load_realized_funding(funding_manifest)
    return frame, funding


def build_family_schedules(family: str) -> dict[str, pd.DataFrame]:
    if family == "cspr":
        support = _read_json(cspr_ev.SUPPORT_RESULT)
        cfg = cspr.Config()
        frame, source = cspr.load_causal_frame(cfg)
        controls, primary = cspr_ev.verify_signal_replay(frame, cfg, support, source)
        return cspr_ev.build_control_schedules(frame, controls, primary, cfg)
    if family == "rift":
        support = _read_json(rift_ev.PREREGISTRATION_RESULT)
        cfg = rift.Config()
        frame, source = rift.load_causal_frame(cfg)
        controls, primary = rift_ev.verify_signal_replay(frame, cfg, support, source)
        return rift_ev.build_control_schedules(frame, controls, primary, cfg)
    if family == "catch":
        support = _read_json(catch_ev.SUPPORT_RESULT)
        cfg = catch.Config()
        frame, source = catch.load_causal_frame(cfg)
        controls, sides, primary = catch_ev.verify_signal_replay(
            frame, cfg, support, source
        )
        return catch_ev.build_control_schedules(
            frame, controls, sides, primary, cfg, support
        )
    if family == "luri":
        support = _read_json(luri_ev.SUPPORT_RESULT)
        cfg = luri.Config()
        frame, source = luri.load_causal_frame()
        controls, sides, primary = luri_ev.verify_signal_replay(
            frame, cfg, support, source
        )
        return luri_ev.build_control_schedules(
            frame, controls, sides, primary, cfg, support
        )
    if family == "clasp":
        support = _read_json(clasp_ev.SUPPORT_RESULT)
        cfg = clasp.Config()
        frame, source = clasp.load_causal_frame()
        controls, sides, primary = clasp_ev.verify_signal_replay(
            frame, cfg, support, source
        )
        return clasp_ev.build_control_schedules(
            frame, controls, sides, primary, cfg, support
        )
    if family == "umfr":
        support = _read_json(umfr_ev.SUPPORT_RESULT)
        cfg = umfr.Config()
        frame, source = umfr.load_causal_frame()
        controls, sides, primary = umfr_ev.verify_signal_replay(
            frame, cfg, support, source
        )
        return umfr_ev.build_control_schedules(
            frame, controls, sides, primary, cfg, support
        )
    raise KeyError(family)


def schedule_to_stream(
    frame: pd.DataFrame,
    funding: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    start: str,
    end: str,
    cfg: Config,
) -> dict[str, Any]:
    schedule = _slice_schedule(schedule, start=start, end=end)
    idx = np.flatnonzero(
        (frame["date"] >= pd.Timestamp(start)).to_numpy()
        & (frame["date"] < pd.Timestamp(end)).to_numpy()
    )
    st, en = int(idx[0]), int(idx[-1]) + 1
    returns = np.zeros(en - st, dtype=np.float64)
    adverse = np.zeros(en - st, dtype=np.float64)
    opens = frame["open"].to_numpy(float)
    highs = frame["high"].to_numpy(float)
    lows = frame["low"].to_numpy(float)
    dates = frame["date"]
    funding_times = funding["funding_time_ms"].to_numpy(np.int64)
    funding_rates = funding["funding_rate"].to_numpy(float)
    per_side_cost = (cfg.fee_rate + cfg.slippage_rate) * cfg.leverage
    gross_returns: list[float] = []
    wins = 0
    for row in schedule.itertuples(index=False):
        entry = int(row.entry_position)
        exit_ = int(row.exit_position)
        signal = int(row.signal_position)
        side = int(row.side)
        if not signal < entry < exit_ or entry != signal + 1:
            raise ValueError(f"invalid schedule order for {row}")
        if exit_ >= len(frame) or entry < st or exit_ >= en:
            continue
        entry_date = pd.Timestamp(dates.iloc[entry])
        exit_date = pd.Timestamp(dates.iloc[exit_])
        if str(entry_date) != str(pd.Timestamp(row.entry_date)):
            raise ValueError("schedule entry date does not align with market frame")
        entry_price = float(opens[entry])
        exit_price = float(opens[exit_])
        held_high = float(np.max(highs[entry:exit_]))
        held_low = float(np.min(lows[entry:exit_]))
        raw_return = side * (exit_price / entry_price - 1.0)
        entry_ms = int(entry_date.value // 1_000_000)
        exit_ms = int(exit_date.value // 1_000_000)
        left = int(np.searchsorted(funding_times, entry_ms, side="left"))
        right = int(np.searchsorted(funding_times, exit_ms, side="right"))
        factors = 1.0 - cfg.leverage * side * funding_rates[left:right]
        if not np.isfinite(factors).all() or (factors <= 0.0).any():
            raise ValueError("invalid funding factor")
        funding_factor = float(np.prod(factors, dtype=float))
        funding_debit_factor = float(np.prod(np.minimum(factors, 1.0), dtype=float))
        price_factor = max(0.0, 1.0 + cfg.leverage * raw_return)
        realized_factor = (
            (1.0 - per_side_cost)
            * price_factor
            * funding_factor
            * (1.0 - per_side_cost)
        )
        realized = realized_factor - 1.0
        favorable_price = held_high if side > 0 else held_low
        adverse_price = held_low if side > 0 else held_high
        favorable = cfg.leverage * side * (favorable_price / entry_price - 1.0)
        adverse_move = cfg.leverage * side * (adverse_price / entry_price - 1.0)
        # Store returns at entry/exit and a conservative intratrade adverse shock.
        returns[entry - st] += -per_side_cost
        returns[exit_ - st] += realized - (-per_side_cost)
        adverse[entry - st] += -per_side_cost
        adverse[exit_ - st] += min(
            -per_side_cost,
            (1.0 - per_side_cost)
            * funding_debit_factor
            * (1.0 + adverse_move)
            / max(1.0, 1.0 + favorable)
            - 1.0,
        )
        gross_returns.append(raw_return)
        wins += int(realized > 0.0)
    return {
        "return_stream": returns,
        "adverse_stream": adverse,
        "trade_count": int(len(schedule)),
        "win_count": int(wins),
        "mean_gross_underlying_move_bp": float(np.mean(gross_returns) * 10_000.0)
        if gross_returns
        else 0.0,
        "start": start,
        "end": end,
    }


def pack_streams(streams: dict[str, dict[str, Any]]) -> dict[str, Any]:
    names = list(streams)
    return {
        "names": names,
        "R": np.vstack([streams[name]["return_stream"] for name in names])
        if names
        else np.zeros((0, 0)),
        "A": np.vstack([streams[name]["adverse_stream"] for name in names])
        if names
        else np.zeros((0, 0)),
        "trade_counts": np.array(
            [streams[name]["trade_count"] for name in names], dtype=np.int64
        ),
        "win_counts": np.array(
            [streams[name]["win_count"] for name in names], dtype=np.int64
        ),
        "gross_bp": np.array(
            [streams[name]["mean_gross_underlying_move_bp"] for name in names],
            dtype=np.float64,
        ),
        "start": next(iter(streams.values()))["start"] if streams else "2020-01-01",
        "end": next(iter(streams.values()))["end"] if streams else "2020-01-01",
    }


def metric_from_pack(pack: dict[str, Any], weights: dict[str, float]) -> dict[str, Any]:
    names = pack["names"]
    if not names:
        return {}
    wv = np.array([float(weights.get(name, 0.0)) for name in names], dtype=np.float64)
    r = wv @ pack["R"]
    a = wv @ pack["A"]
    start = pd.Timestamp(pack["start"])
    end = pd.Timestamp(pack["end"])
    years = (end - start).total_seconds() / (365.25 * 86_400.0)
    if len(r):
        eq_path = np.cumprod(np.maximum(0.0, 1.0 + r))
        eq_before = np.r_[1.0, eq_path[:-1]]
        peak_after = np.maximum.accumulate(eq_path)
        peak_before = np.maximum.accumulate(eq_before)
        dd_after = float(np.nanmax(1.0 - eq_path / np.maximum(peak_after, 1e-12)))
        dd_adv = float(
            np.nanmax(
                1.0
                - (eq_before * np.maximum(0.0, 1.0 + a))
                / np.maximum(peak_before, 1e-12)
            )
        )
        equity = float(eq_path[-1])
        strict_mdd = max(dd_after, dd_adv) * 100.0
    else:
        equity = 1.0
        strict_mdd = 0.0
    absolute = (equity - 1.0) * 100.0
    cagr = ((equity ** (1.0 / years) - 1.0) * 100.0) if equity > 0 else -100.0
    active = wv > 1e-12
    trades = int(pack["trade_counts"][active].sum())
    wins = int(pack["win_counts"][active].sum())
    weighted_trade = pack["trade_counts"] * wv
    weighted_gross_den = float(weighted_trade[active].sum())
    weighted_gross_num = float((pack["gross_bp"] * weighted_trade)[active].sum())
    return {
        "absolute_return_pct": float(absolute),
        "cagr_pct": float(cagr),
        "strict_mdd_pct": float(strict_mdd),
        "cagr_to_strict_mdd": float(cagr / strict_mdd) if strict_mdd > 1e-12 else 0.0,
        "trade_count_sum": trades,
        "win_rate_unweighted": float(wins / trades) if trades else 0.0,
        "weighted_mean_gross_underlying_bp": float(
            weighted_gross_num / weighted_gross_den
        )
        if weighted_gross_den
        else 0.0,
        "nonzero_sleeves": [name for name, value in zip(names, wv) if value > 1e-12],
    }


def metric_from_streams(
    streams: dict[str, dict[str, Any]], weights: dict[str, float]
) -> dict[str, Any]:
    names = list(streams)
    if not names:
        return {}
    r = sum(
        float(weights.get(name, 0.0)) * streams[name]["return_stream"] for name in names
    )
    a = sum(
        float(weights.get(name, 0.0)) * streams[name]["adverse_stream"]
        for name in names
    )
    start = pd.Timestamp(next(iter(streams.values()))["start"])
    end = pd.Timestamp(next(iter(streams.values()))["end"])
    years = (end - start).total_seconds() / (365.25 * 86_400.0)
    if len(r):
        eq_path = np.cumprod(np.maximum(0.0, 1.0 + r))
        eq_before = np.r_[1.0, eq_path[:-1]]
        peak_after = np.maximum.accumulate(eq_path)
        peak_before = np.maximum.accumulate(eq_before)
        dd_after = float(np.nanmax(1.0 - eq_path / np.maximum(peak_after, 1e-12)))
        dd_adv = float(
            np.nanmax(
                1.0
                - (eq_before * np.maximum(0.0, 1.0 + a))
                / np.maximum(peak_before, 1e-12)
            )
        )
        equity = float(eq_path[-1])
        strict_mdd = max(dd_after, dd_adv) * 100.0
    else:
        equity = 1.0
        strict_mdd = 0.0
    absolute = (equity - 1.0) * 100.0
    cagr = ((equity ** (1.0 / years) - 1.0) * 100.0) if equity > 0 else -100.0
    nonzero = [name for name in names if weights.get(name, 0.0) > 1e-12]
    trades = int(sum(streams[name]["trade_count"] for name in nonzero))
    wins = int(sum(streams[name]["win_count"] for name in nonzero))
    weighted_gross_num = sum(
        streams[name]["mean_gross_underlying_move_bp"]
        * streams[name]["trade_count"]
        * float(weights.get(name, 0.0))
        for name in nonzero
    )
    weighted_gross_den = sum(
        streams[name]["trade_count"] * float(weights.get(name, 0.0)) for name in nonzero
    )
    return {
        "absolute_return_pct": float(absolute),
        "cagr_pct": float(cagr),
        "strict_mdd_pct": float(strict_mdd),
        "cagr_to_strict_mdd": float(cagr / strict_mdd) if strict_mdd > 1e-12 else 0.0,
        "trade_count_sum": trades,
        "win_rate_unweighted": float(wins / trades) if trades else 0.0,
        "weighted_mean_gross_underlying_bp": float(
            weighted_gross_num / weighted_gross_den
        )
        if weighted_gross_den
        else 0.0,
        "nonzero_sleeves": nonzero,
    }


def _random_weights(
    names: list[str], cfg: Config, rng: random.Random
) -> dict[str, float]:
    k = rng.randint(1, min(8, len(names)))
    picked = rng.sample(names, k)
    remaining = cfg.gross_cap
    weights: dict[str, float] = {}
    for name in picked:
        max_units = int((remaining - cfg.min_nonzero_weight) / cfg.weight_step)
        if max_units < 0:
            break
        units = rng.randint(
            int(cfg.min_nonzero_weight / cfg.weight_step),
            max_units + int(cfg.min_nonzero_weight / cfg.weight_step),
        )
        value = min(remaining, round(units * cfg.weight_step, 10))
        if value >= cfg.min_nonzero_weight:
            weights[name] = value
            remaining -= value
        if remaining < cfg.min_nonzero_weight:
            break
    return {k: round(v, 10) for k, v in weights.items() if v >= cfg.min_nonzero_weight}


def fit_weights(train_pack: dict[str, Any], cfg: Config) -> list[dict[str, Any]]:
    names = list(train_pack["names"])
    rng = random.Random(cfg.seed)
    candidates: list[dict[str, float]] = []
    # Single sleeves and equal-weight family seeds.
    for name in names:
        candidates.append({name: min(cfg.gross_cap, 1.0)})
    candidates.append({name: cfg.gross_cap / len(names) for name in names})
    for _ in range(cfg.random_samples):
        w = _random_weights(names, cfg, rng)
        if w:
            candidates.append(w)
    seen: set[tuple[tuple[str, float], ...]] = set()
    rows: list[dict[str, Any]] = []
    for weights in candidates:
        key = tuple(sorted(weights.items()))
        if key in seen:
            continue
        seen.add(key)
        m = metric_from_pack(train_pack, weights)
        if not m:
            continue
        score = m["cagr_to_strict_mdd"]
        if m["strict_mdd_pct"] > 25.0:
            score -= (m["strict_mdd_pct"] - 25.0) / 10.0
        if m["absolute_return_pct"] <= 0:
            score -= 5.0
        rows.append({"weights": weights, "train": m, "fit_score": float(score)})
    rows.sort(
        key=lambda row: (
            row["fit_score"],
            row["train"]["cagr_to_strict_mdd"],
            row["train"]["absolute_return_pct"],
        ),
        reverse=True,
    )
    return rows[:50]


def run(cfg: Config) -> dict[str, Any]:
    candidates = select_candidate_policies(cfg)
    frame, funding = load_execution_market_and_funding()
    family_schedules = {
        family: build_family_schedules(family)
        for family in sorted({c["family"] for c in candidates})
    }
    streams: dict[str, dict[str, dict[str, Any]]] = {window: {} for window in WINDOWS}
    candidate_meta: dict[str, Any] = {}
    for cand in candidates:
        name = cand["name"]
        schedule = family_schedules[cand["family"]][cand["policy"]]
        candidate_meta[name] = {**cand, "global_clock_count": int(len(schedule))}
        for window, (start, end) in WINDOWS.items():
            streams[window][name] = schedule_to_stream(
                frame, funding, schedule, start=start, end=end, cfg=cfg
            )
    packs = {window: pack_streams(streams[window]) for window in WINDOWS}
    fitted = fit_weights(packs["train"], cfg)
    evaluated = []
    for row in fitted[:30]:
        weights = row["weights"]
        metrics = {
            window: metric_from_streams(streams[window], weights) for window in WINDOWS
        }
        evaluated.append(
            {"weights": weights, "metrics": metrics, "fit_score": row["fit_score"]}
        )
    evaluated.sort(
        key=lambda row: (
            min(
                row["metrics"]["train"]["cagr_to_strict_mdd"],
                row["metrics"]["select2023"]["cagr_to_strict_mdd"],
            ),
            row["metrics"]["select2023"]["absolute_return_pct"],
            row["metrics"]["train"]["absolute_return_pct"],
        ),
        reverse=True,
    )
    return {
        "protocol": {
            "name": "weak signal feature ensemble pre-2024 test",
            "opened_windows": list(WINDOWS),
            "sealed_windows": ["test2024", "eval2025", "ytd2026"],
            "fit_window": "train 2020-2022 only",
            "selection_holdout": "2023/H1/H2 report-only",
            "candidate_rule": "train gross underlying bp > threshold and train trades >= threshold",
            "no_2024_plus_opened": True,
        },
        "config": asdict(cfg),
        "candidate_count": len(candidates),
        "candidates": candidate_meta,
        "top": evaluated[:20],
    }


def write_doc(result: dict[str, Any], path: Path) -> None:
    top = result["top"][:5]
    lines = [
        "# Weak signal feature ensemble pre-2024 test — 2026-07-15",
        "",
        "This tests combinations of previously meaningful weak signal policies. Weights are fit on 2020–2022 only; 2023 is holdout. 2024+ remains sealed.",
        "",
        f"Candidate sleeves: {result['candidate_count']}",
        "",
        "## Top combinations",
        "",
        "| Rank | Train abs | Train CAGR/MDD | Train MDD | 2023 abs | 2023 CAGR/MDD | 2023 MDD | 2023 trades | Weights |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for i, row in enumerate(top, 1):
        tr = row["metrics"]["train"]
        se = row["metrics"]["select2023"]
        if "weights" in row:
            descriptor = ", ".join(f"{k}={v:.2f}" for k, v in row["weights"].items())
        else:
            descriptor = (
                f"{row.get('weight_set', 'unknown')} "
                f"thr={row.get('threshold')} hold={row.get('hold_bars')} "
                f"counts={row.get('counts', {})}"
            )
        lines.append(
            f"| {i} | {tr['absolute_return_pct']:.2f}% | {tr['cagr_to_strict_mdd']:.2f} | {tr['strict_mdd_pct']:.2f}% | "
            f"{se['absolute_return_pct']:.2f}% | {se['cagr_to_strict_mdd']:.2f} | {se['strict_mdd_pct']:.2f}% | {se['trade_count_sum']} | {descriptor} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "This is a weak-signal ensemble test, not a deployment decision. Passing requires the 2023 holdout to remain positive with acceptable strict MDD before any 2024+ evaluation is justified.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--docs-output", default=Config.docs_output)
    parser.add_argument(
        "--mode", choices=["allocation", "consensus"], default="allocation"
    )
    args = parser.parse_args()
    cfg = Config(output=args.output, docs_output=args.docs_output)
    result = run_consensus(cfg) if args.mode == "consensus" else run(cfg)
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_doc(result, Path(cfg.docs_output))
    print(
        json.dumps(
            {
                "candidate_count": result["candidate_count"],
                "top": [
                    {
                        "descriptor": row.get("weights")
                        or {
                            "weight_set": row.get("weight_set"),
                            "threshold": row.get("threshold"),
                            "hold_bars": row.get("hold_bars"),
                            "counts": row.get("counts"),
                        },
                        "train": row["metrics"]["train"],
                        "select2023": row["metrics"]["select2023"],
                    }
                    for row in result["top"][:5]
                ],
                "output": cfg.output,
                "docs_output": cfg.docs_output,
            },
            indent=2,
        )
    )


# --- Consensus-vote mode -------------------------------------------------


def build_vote_arrays(
    candidates: list[dict[str, Any]],
    family_schedules: dict[str, dict[str, pd.DataFrame]],
    n: int,
) -> tuple[list[str], np.ndarray]:
    names = [c["name"] for c in candidates]
    votes = np.zeros((len(names), n), dtype=np.float32)
    for i, cand in enumerate(candidates):
        schedule = family_schedules[cand["family"]][cand["policy"]]
        for row in schedule.itertuples(index=False):
            pos = int(row.signal_position)
            if 0 <= pos < n:
                votes[i, pos] = float(row.side)
    return names, votes


def consensus_schedule(
    frame: pd.DataFrame,
    names: list[str],
    votes: np.ndarray,
    weights: dict[str, float],
    *,
    threshold: float,
    hold_bars: int,
    start: str,
    end: str,
) -> pd.DataFrame:
    wv = np.array([weights.get(name, 0.0) for name in names], dtype=np.float32)
    score = wv @ votes
    dates = pd.to_datetime(frame["date"])
    split = (dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))
    active = split.to_numpy() & (np.abs(score) >= float(threshold))
    side = np.sign(score).astype(np.int8)
    rows = []
    next_allowed = 0
    last_signal = len(frame) - int(hold_bars) - 2
    for pos in np.flatnonzero(active):
        pos = int(pos)
        if pos < next_allowed or pos > last_signal:
            continue
        entry = pos + 1
        exit_ = entry + int(hold_bars)
        if not (split.iloc[entry] and split.iloc[exit_]):
            continue
        rows.append(
            {
                "signal_position": pos,
                "entry_position": entry,
                "exit_position": exit_,
                "signal_date": str(dates.iloc[pos]),
                "entry_date": str(dates.iloc[entry]),
                "exit_date": str(dates.iloc[exit_]),
                "side": int(side[pos]),
                "branch": "weak_consensus",
                "hold_bars": int(hold_bars),
                "score": float(score[pos]),
            }
        )
        next_allowed = exit_
    return pd.DataFrame(rows)


def consensus_metric(
    frame: pd.DataFrame,
    funding: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    start: str,
    end: str,
    cfg: Config,
) -> dict[str, Any]:
    if schedule.empty:
        return {
            "absolute_return_pct": 0.0,
            "cagr_pct": 0.0,
            "strict_mdd_pct": 0.0,
            "cagr_to_strict_mdd": 0.0,
            "trade_count_sum": 0,
            "win_rate_unweighted": 0.0,
            "weighted_mean_gross_underlying_bp": 0.0,
            "nonzero_sleeves": [],
        }
    stream = schedule_to_stream(frame, funding, schedule, start=start, end=end, cfg=cfg)
    return metric_from_streams({"consensus": stream}, {"consensus": 1.0})


def run_consensus(cfg: Config) -> dict[str, Any]:
    candidates = select_candidate_policies(cfg)
    frame, funding = load_execution_market_and_funding()
    family_schedules = {
        family: build_family_schedules(family)
        for family in sorted({c["family"] for c in candidates})
    }
    names, votes = build_vote_arrays(candidates, family_schedules, len(frame))
    gross = np.array(
        [max(0.0, c["train_gross_bp"]) for c in candidates], dtype=np.float32
    )
    base_weights = {
        name: float(value / gross.sum())
        for name, value in zip(names, gross)
        if gross.sum() > 0 and value > 0
    }
    equal_weights = {name: 1.0 / len(names) for name in names}
    weight_sets = {"gross_weighted": base_weights, "equal": equal_weights}
    rows = []
    for weight_name, weights in weight_sets.items():
        for threshold in (0.05, 0.08, 0.10, 0.12, 0.15, 0.20):
            for hold in (12, 24, 36, 48, 72, 96):
                metrics = {}
                counts = {}
                for window, (start, end) in WINDOWS.items():
                    sched = consensus_schedule(
                        frame,
                        names,
                        votes,
                        weights,
                        threshold=threshold,
                        hold_bars=hold,
                        start=start,
                        end=end,
                    )
                    counts[window] = int(len(sched))
                    metrics[window] = consensus_metric(
                        frame, funding, sched, start=start, end=end, cfg=cfg
                    )
                tr = metrics["train"]
                score = tr["cagr_to_strict_mdd"]
                if tr["absolute_return_pct"] <= 0 or tr["trade_count_sum"] < 80:
                    score -= 10.0
                rows.append(
                    {
                        "weight_set": weight_name,
                        "threshold": threshold,
                        "hold_bars": hold,
                        "counts": counts,
                        "metrics": metrics,
                        "fit_score": score,
                    }
                )
    rows.sort(
        key=lambda r: (r["fit_score"], r["metrics"]["train"]["absolute_return_pct"]),
        reverse=True,
    )
    return {
        "protocol": {
            "name": "weak signal consensus vote pre-2024 test",
            "fit_window": "train only",
            "sealed_windows": ["test2024", "eval2025", "ytd2026"],
        },
        "config": asdict(cfg),
        "candidate_count": len(candidates),
        "top": rows[:20],
    }


if __name__ == "__main__":
    main()
