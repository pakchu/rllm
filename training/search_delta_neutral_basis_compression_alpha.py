"""Preregister and test a market-neutral spot/perpetual basis-compression alpha.

Unlike the rejected always/conditional funding carry sleeve, this strategy is
short-lived and must remain profitable with funding cash removed.  It enters
equal-BTC long spot / short perpetual only after a completed one-minute basis
observation is abnormally high, then exits on causal compression, a fixed
adverse basis stop, or a fixed maximum hold.  All search inputs end before
2024; later windows are not opened here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_delta_neutral_funding_carry_alpha import (
    DEFAULT_FUNDING,
    DEFAULT_FUTURES,
    DEFAULT_SOURCE_MANIFEST,
    DEFAULT_SPOT,
    SELECTION_END,
    Config as CarryConfig,
    CostModel,
    Policy as CarryPolicy,
    Sources,
    _json_hash,
    block_bootstrap,
    gate_actions,
    load_sources,
    simulate_window,
)


DEFAULT_OUTPUT = "results/delta_neutral_basis_compression_pre2024_2026-07-16.json"
DEFAULT_MANIFEST = "results/delta_neutral_basis_compression_frozen_policy_2026-07-16.json"
DEFAULT_DOCS = "docs/delta-neutral-basis-compression-pre2024-2026-07-16.md"
DEFAULT_CARRY_RESULT = "results/delta_neutral_funding_carry_pre2024_2026-07-16.json"

WINDOWS: dict[str, tuple[str, str]] = {
    "fit_2020_2022": ("2020-01-01", "2023-01-01"),
    "select_2023h1": ("2023-01-01", "2023-07-01"),
    "select_2023h2": ("2023-07-01", SELECTION_END),
    "select_2023": ("2023-01-01", SELECTION_END),
}


@dataclass(frozen=True)
class Config:
    futures_csv: str = DEFAULT_FUTURES
    spot_csv: str = DEFAULT_SPOT
    funding_csv: str = DEFAULT_FUNDING
    source_manifest: str = DEFAULT_SOURCE_MANIFEST
    carry_result: str = DEFAULT_CARRY_RESULT
    output: str = DEFAULT_OUTPUT
    policy_manifest: str = DEFAULT_MANIFEST
    docs_output: str = DEFAULT_DOCS
    gross_exposure: float = 1.0
    spot_fee_rate: float = 0.0010
    perp_fee_rate: float = 0.0005
    spot_slippage_rate: float = 0.0001
    perp_slippage_rate: float = 0.0001
    incomplete_spot_cushion: float = 0.0025
    exit_z: float = 0.5
    adverse_stop_bps: float = 25.0
    minimum_expected_compression_bps: float = 40.0
    bootstrap_samples: int = 5_000
    bootstrap_seed: int = 314_159


@dataclass(frozen=True, order=True)
class BasisPolicy:
    lookback_minutes: int
    entry_z: float
    max_hold_minutes: int


def policy_grid() -> list[BasisPolicy]:
    return [
        BasisPolicy(lookback, entry_z, hold)
        for lookback in (10_080, 43_200)
        for entry_z in (2.0, 2.5, 3.0)
        for hold in (360, 1_440)
    ]


def _carry_config(cfg: Config) -> CarryConfig:
    return CarryConfig(
        futures_csv=cfg.futures_csv,
        spot_csv=cfg.spot_csv,
        funding_csv=cfg.funding_csv,
        source_manifest=cfg.source_manifest,
        gross_exposure=cfg.gross_exposure,
        spot_fee_rate=cfg.spot_fee_rate,
        perp_fee_rate=cfg.perp_fee_rate,
        spot_slippage_rate=cfg.spot_slippage_rate,
        perp_slippage_rate=cfg.perp_slippage_rate,
        incomplete_spot_cushion=cfg.incomplete_spot_cushion,
        bootstrap_samples=cfg.bootstrap_samples,
        bootstrap_seed=cfg.bootstrap_seed,
        search_workers=1,
    )


def basis_feature(
    sources: Sources,
    lookback_minutes: int,
) -> pd.DataFrame:
    if lookback_minutes < 2:
        raise ValueError("basis lookback must be at least two minutes")
    market = sources.market
    basis = np.log(market["perp_close"].to_numpy(float) / market["spot_close"].to_numpy(float))
    series = pd.Series(basis, index=market.index)
    prior = series.shift(1)
    mean = prior.rolling(lookback_minutes, min_periods=lookback_minutes).mean()
    std = prior.rolling(lookback_minutes, min_periods=lookback_minutes).std(ddof=0)
    std = std.where(std > 1e-12)
    proxy = market["spot_proxy"].astype(bool)
    clean = proxy.rolling(lookback_minutes + 1, min_periods=lookback_minutes + 1).sum().eq(0)
    return pd.DataFrame(
        {
            "basis": series,
            "prior_mean": mean,
            "prior_std": std,
            "z": (series - mean) / std,
            "clean_window": clean,
        }
    )


def basis_actions(
    sources: Sources,
    policy: BasisPolicy,
    cfg: Config,
    *,
    signal_delay_minutes: int = 0,
    invert: bool = False,
) -> tuple[dict[int, bool], list[dict[str, Any]]]:
    if signal_delay_minutes < 0:
        raise ValueError("basis signal delay cannot be negative")
    feature = basis_feature(sources, policy.lookback_minutes)
    dates = sources.market["date"]
    execution = np.flatnonzero(
        (dates.dt.minute.to_numpy() % 5 == 0) & (dates.dt.second.to_numpy() == 0)
    )
    active = False
    entry_basis = math.nan
    entry_index = -1
    actions: dict[int, bool] = {}
    trace: list[dict[str, Any]] = []
    for index in execution:
        signal = int(index) - 1 - int(signal_delay_minutes)
        if signal < 0:
            continue
        row = feature.iloc[signal]
        finite = all(
            math.isfinite(float(row[name])) for name in ("basis", "prior_mean", "prior_std", "z")
        )
        clean = bool(row["clean_window"])
        if not active:
            if not finite or not clean:
                continue
            z = float(row["z"])
            if invert:
                edge = float(row["prior_mean"] - cfg.exit_z * row["prior_std"] - row["basis"])
                enter = z <= -policy.entry_z
            else:
                edge = float(row["basis"] - row["prior_mean"] - cfg.exit_z * row["prior_std"])
                enter = z >= policy.entry_z
            if not enter or edge * 10_000.0 < cfg.minimum_expected_compression_bps:
                continue
            active = True
            entry_basis = float(row["basis"])
            entry_index = int(index)
            actions[int(index)] = True
            trace.append(
                {
                    "execution_index": int(index),
                    "signal_index": signal,
                    "target_active": True,
                    "basis": entry_basis,
                    "z": z,
                    "expected_compression_bps": edge * 10_000.0,
                }
            )
            continue

        held_minutes = int(index) - entry_index
        data_available = finite and clean
        current_basis = float(row["basis"]) if finite else math.nan
        current_z = float(row["z"]) if finite else math.nan
        normalized = (
            data_available and (current_z >= -cfg.exit_z if invert else current_z <= cfg.exit_z)
        )
        adverse = (
            data_available
            and (current_basis - entry_basis) * 10_000.0 >= cfg.adverse_stop_bps
        )
        timed_out = held_minutes >= policy.max_hold_minutes
        if not (normalized or adverse or timed_out):
            continue
        active = False
        actions[int(index)] = False
        trace.append(
            {
                "execution_index": int(index),
                "signal_index": signal,
                "target_active": False,
                "basis": current_basis if finite else None,
                "z": current_z if finite else None,
                "reason": (
                    "normalized"
                    if normalized
                    else "adverse_stop"
                    if adverse
                    else "max_hold_data_unavailable"
                    if not data_available
                    else "max_hold"
                ),
            }
        )
    return actions, trace


def delay_actions(actions: dict[int, bool], minutes: int, market_rows: int) -> dict[int, bool]:
    delayed: dict[int, bool] = {}
    for index, target in sorted(actions.items()):
        shifted = int(index) + int(minutes)
        if shifted >= market_rows:
            continue
        if shifted in delayed and delayed[shifted] != target:
            raise RuntimeError("delayed basis actions collide")
        delayed[shifted] = bool(target)
    return delayed


def schedule_hash(actions: dict[int, bool]) -> str:
    payload = [[int(index), bool(target)] for index, target in sorted(actions.items())]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def _simulate(
    sources: Sources,
    actions: dict[int, bool],
    cfg: Config,
    name: str,
    *,
    costs: CostModel | None = None,
    include_funding: bool = True,
) -> dict[str, Any]:
    carry = _carry_config(cfg)
    start, end = WINDOWS[name]
    return simulate_window(
        sources,
        actions,
        start=start,
        end=end,
        cfg=carry,
        costs=costs,
        include_funding=include_funding,
        force_initial_active=False,
        daily_rebalance=False,
    )


def window_stats(
    sources: Sources,
    actions: dict[int, bool],
    cfg: Config,
    *,
    costs: CostModel | None = None,
    include_funding: bool = True,
) -> dict[str, dict[str, Any]]:
    return {
        name: _simulate(
            sources,
            actions,
            cfg,
            name,
            costs=costs,
            include_funding=include_funding,
        )["stats"]
        for name in WINDOWS
    }


def _eligibility(stats: dict[str, dict[str, Any]]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for name in WINDOWS:
        if stats[name]["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}:nonpositive")
    for name in ("fit_2020_2022", "select_2023"):
        if stats[name]["cagr_to_strict_mdd"] < 3.0:
            failures.append(f"{name}:ratio<3")
        if stats[name]["strict_mdd_pct"] > 15.0:
            failures.append(f"{name}:mdd>15")
    if stats["fit_2020_2022"]["episodes"] < 20:
        failures.append("fit_2020_2022:episodes<20")
    for name in ("select_2023h1", "select_2023h2"):
        if stats[name]["episodes"] < 8:
            failures.append(f"{name}:episodes<8")
    return not failures, failures


def _rank(row: dict[str, Any]) -> tuple[float, float, float]:
    stats = row["stats"]
    return (
        float(
            min(
                stats["fit_2020_2022"]["cagr_to_strict_mdd"],
                stats["select_2023"]["cagr_to_strict_mdd"],
            )
        ),
        float(
            min(
                stats["select_2023h1"]["absolute_return_pct"],
                stats["select_2023h2"]["absolute_return_pct"],
            )
        ),
        float(stats["fit_2020_2022"]["absolute_return_pct"]),
    )


def weekly_rademacher(daily: pd.Series, cfg: Config) -> dict[str, Any]:
    if len(daily) < 14:
        return {"valid": False, "days": int(len(daily))}
    weekly = daily.groupby(daily.index.to_period("W-SUN")).apply(
        lambda values: float(np.prod(1.0 + values.to_numpy(float)) - 1.0)
    )
    values = weekly.to_numpy(float)
    observed = float(values.mean())
    rng = np.random.default_rng(cfg.bootstrap_seed)
    simulated = np.empty(cfg.bootstrap_samples, dtype=float)
    for sample in range(cfg.bootstrap_samples):
        simulated[sample] = float((values * rng.choice((-1.0, 1.0), len(values))).mean())
    return {
        "valid": True,
        "weeks": int(len(values)),
        "observed_mean_weekly_bps": observed * 10_000.0,
        "one_sided_p": float((simulated >= observed).mean()),
        "bonferroni_12_p": float(min(1.0, 12.0 * (simulated >= observed).mean())),
        "samples": int(cfg.bootstrap_samples),
    }


def _fallback_only_sources(sources: Sources) -> Sources:
    funding = sources.funding.copy()
    indexes = funding["fallback_index"].to_numpy(int)
    closes = sources.market["perp_close"].to_numpy(float)
    marks = np.full(len(funding), np.nan, dtype=float)
    valid = indexes >= 0
    marks[valid] = closes[indexes[valid]]
    funding["settlement_mark"] = marks
    funding["settlement_mark_is_reported"] = False
    return Sources(sources.market, funding, sources.source_hashes, sources.diagnostics)


def _carry_comparator(
    sources: Sources,
    basis_actions_map: dict[int, bool],
    cfg: Config,
) -> dict[str, Any]:
    carry_result = json.loads(Path(cfg.carry_result).read_text())
    policy = CarryPolicy(**carry_result["selected"]["policy"])
    actions, _ = gate_actions(sources.funding, policy)
    output: dict[str, Any] = {"policy": asdict(policy), "windows": {}}
    carry_cfg = _carry_config(cfg)
    for name in ("fit_2020_2022", "select_2023"):
        start, end = WINDOWS[name]
        basis_sim = _simulate(sources, basis_actions_map, cfg, name)
        carry_sim = simulate_window(
            sources,
            actions,
            start=start,
            end=end,
            cfg=carry_cfg,
            force_initial_active=False,
            daily_rebalance=True,
        )
        joined = pd.concat(
            [
                basis_sim["daily_returns"].rename("basis"),
                carry_sim["daily_returns"].rename("carry"),
            ],
            axis=1,
        ).dropna()
        correlation = float(joined.corr().iloc[0, 1]) if len(joined) >= 10 else 0.0
        output["windows"][name] = {
            "daily_pnl_pearson": correlation,
            "basis_stats": basis_sim["stats"],
            "carry_stats": carry_sim["stats"],
            "overlap_days": int(len(joined)),
        }
    return output


def _control_min_ratio(stats: dict[str, dict[str, Any]]) -> float:
    return float(
        min(
            stats["fit_2020_2022"]["cagr_to_strict_mdd"],
            stats["select_2023"]["cagr_to_strict_mdd"],
        )
    )


def evaluate_controls(
    sources: Sources,
    policy: BasisPolicy,
    primary_actions: dict[int, bool],
    cfg: Config,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_cost = CostModel(
        cfg.spot_fee_rate + cfg.spot_slippage_rate,
        cfg.perp_fee_rate + cfg.perp_slippage_rate,
    )
    inverted, _ = basis_actions(sources, policy, cfg, invert=True)
    stale_1h, _ = basis_actions(sources, policy, cfg, signal_delay_minutes=60)
    stale_24h, _ = basis_actions(sources, policy, cfg, signal_delay_minutes=1_440)
    delayed_5m = delay_actions(primary_actions, 5, len(sources.market))
    fallback_sources = _fallback_only_sources(sources)
    controls = {
        "zero_funding": window_stats(
            sources, primary_actions, cfg, include_funding=False
        ),
        "double_execution_cost": window_stats(
            sources,
            primary_actions,
            cfg,
            costs=CostModel(base_cost.spot_rate * 2.0, base_cost.perp_rate * 2.0),
        ),
        "signal_stale_1h": window_stats(sources, stale_1h, cfg),
        "signal_stale_24h": window_stats(sources, stale_24h, cfg),
        "decision_delayed_5m": window_stats(sources, delayed_5m, cfg),
        "inverted_basis": window_stats(sources, inverted, cfg),
        "fallback_only_funding_mark": window_stats(
            fallback_sources, primary_actions, cfg
        ),
    }
    primary = window_stats(sources, primary_actions, cfg)
    primary_min = _control_min_ratio(primary)
    zero_positive = all(
        controls["zero_funding"][name]["absolute_return_pct"] > 0.0
        for name in ("fit_2020_2022", "select_2023")
    )
    double_positive = all(
        controls["double_execution_cost"][name]["absolute_return_pct"] > 0.0
        for name in ("fit_2020_2022", "select_2023")
    )
    stale_weaker = primary_min > max(
        _control_min_ratio(controls["signal_stale_1h"]),
        _control_min_ratio(controls["signal_stale_24h"]),
        _control_min_ratio(controls["decision_delayed_5m"]),
        _control_min_ratio(controls["inverted_basis"]),
    )
    significance = {}
    significance_pass = True
    for name in ("fit_2020_2022", "select_2023"):
        simulation = _simulate(sources, primary_actions, cfg, name)
        significance[name] = {
            "weekly_rademacher": weekly_rademacher(simulation["daily_returns"], cfg),
            "weekly_block_bootstrap": block_bootstrap(simulation["daily_returns"], _carry_config(cfg)),
        }
        significance_pass &= (
            significance[name]["weekly_rademacher"].get("bonferroni_12_p", 1.0) < 0.10
        )
    gates = {
        "zero_funding_positive": zero_positive,
        "double_cost_positive": double_positive,
        "primary_beats_stale_delay_inversion_controls": stale_weaker,
        "bonferroni_weekly_rademacher_p_below_0p10": significance_pass,
    }
    return {"stats": controls, "gates": gates}, significance


def _table(stats: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| window | absolute return | CAGR | strict MDD | CAGR/MDD | episodes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in stats.items():
        lines.append(
            f"| {name} | {row['absolute_return_pct']:.4f}% | {row['cagr_pct']:.4f}% | "
            f"{row['strict_mdd_pct']:.4f}% | {row['cagr_to_strict_mdd']:.4f} | {row['episodes']} |"
        )
    return lines


def write_docs(report: dict[str, Any], path: str) -> None:
    selected = report["selected"]
    lines = [
        "# Delta-neutral basis-compression alpha: pre-2024",
        "",
        f"- Decision: **{report['decision']['status']}**",
        "- 2024+ opened: **no**",
        "- All pre-2024 selected statistics and p-values are post-selection descriptive, not independent validation.",
        "- Equal BTC quantities are fixed from entry to exit; gross is fixed only at entry.",
        "- Proxy spot bars may mark strict MDD but cannot create or close a signal.",
        "",
        "## Frozen candidate",
        "",
        "```json",
        json.dumps(selected["policy"], indent=2, sort_keys=True),
        "```",
        "",
        "## Statistics",
        "",
        *_table(selected["stats"]),
        "",
        "## Promotion/control gates",
        "",
        "```json",
        json.dumps(selected["control_gates"], indent=2, sort_keys=True),
        "```",
        "",
        "## Carry orthogonality comparator",
        "",
        "```json",
        json.dumps(report["carry_comparator"], indent=2, sort_keys=True),
        "```",
        "",
        "## Constraints",
        "",
        "- Historical DB rows are backfilled/non-PIT; frozen OOS and live forward parity are still mandatory.",
        "- Unified margin or automatic collateral transfer/liquidation guard is required before live promotion.",
        "- The earlier directional spot-perp residual family failed OOS; this two-leg family must remain profitable with funding removed.",
        "",
    ]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def run(cfg: Config) -> dict[str, Any]:
    if cfg.exit_z != 0.5 or cfg.adverse_stop_bps != 25.0:
        raise ValueError("basis exit and stop are preregistered and immutable")
    sources = load_sources(_carry_config(cfg))
    searched: list[dict[str, Any]] = []
    seen: set[str] = set()
    for policy in policy_grid():
        actions, trace = basis_actions(sources, policy, cfg)
        path_hash = schedule_hash(actions)
        if path_hash in seen:
            continue
        seen.add(path_hash)
        stats = window_stats(sources, actions, cfg)
        eligible, failures = _eligibility(stats)
        searched.append(
            {
                "policy": asdict(policy),
                "schedule_hash": path_hash,
                "actions": len(actions),
                "trace": trace,
                "stats": stats,
                "eligible": eligible,
                "eligibility_failures": failures,
            }
        )
    searched.sort(key=lambda row: (row["eligible"], *_rank(row)), reverse=True)
    eligible = [row for row in searched if row["eligible"]]
    pool = eligible if eligible else [searched[0]]
    selected = pool[0]
    selected_controls: dict[str, Any] = {}
    selected_significance: dict[str, Any] = {}
    controls_pass = False
    for candidate in pool:
        policy = BasisPolicy(**candidate["policy"])
        actions, _ = basis_actions(sources, policy, cfg)
        controls, significance = evaluate_controls(sources, policy, actions, cfg)
        passes = all(bool(value) for value in controls["gates"].values())
        if not selected_controls:
            selected = candidate
            selected_controls = controls
            selected_significance = significance
            controls_pass = passes
        if passes or not eligible:
            selected = candidate
            selected_controls = controls
            selected_significance = significance
            controls_pass = passes
            break
    selected["controls"] = selected_controls["stats"]
    selected["control_gates"] = selected_controls["gates"]
    selected["significance"] = selected_significance
    selected_policy = BasisPolicy(**selected["policy"])
    selected_actions, selected_trace = basis_actions(sources, selected_policy, cfg)
    carry_comparator = _carry_comparator(sources, selected_actions, cfg)
    correlations = [
        abs(float(row["daily_pnl_pearson"]))
        for row in carry_comparator["windows"].values()
    ]
    orthogonal = bool(max(correlations, default=1.0) <= 0.30)
    promoted = bool(selected["eligible"] and controls_pass and orthogonal)
    decision = {
        "status": "freeze_for_one_shot_oos" if promoted else "reject_pre2024",
        "eligible": bool(selected["eligible"]),
        "controls_pass": controls_pass,
        "orthogonal_to_rejected_carry_daily_pnl": orthogonal,
        "live_promotion_blocked": True,
    }
    report = {
        "protocol": {
            "name": "exact-delta BTC spot/perp basis compression",
            "stage": "pre2024_preregistered_12_policy_falsification",
            "selection_end_exclusive": SELECTION_END,
            "oos_opened": False,
            "future_research_already_viewed_globally": True,
            "pre2024_statistics_are_post_selection_descriptive_only": True,
            "grid": {
                "lookback_minutes": [10_080, 43_200],
                "entry_z": [2.0, 2.5, 3.0],
                "max_hold_minutes": [360, 1_440],
                "exit_z": cfg.exit_z,
                "adverse_stop_bps": cfg.adverse_stop_bps,
                "minimum_expected_compression_bps": cfg.minimum_expected_compression_bps,
            },
            "entry": "completed prior 1m basis and prior-only rolling moments; next 5m open",
            "exit": "completed prior 1m normalization/stop/timeout; next 5m open",
            "strict_mdd": "spot-high/perp-low HWM then spot-low/perp-high adverse, all costs/funding",
            "proxy_policy": "proxy spot bars mark MDD only and force signal inactivity/exit",
        },
        "config": asdict(cfg),
        "sources": {"hashes": sources.source_hashes, "diagnostics": sources.diagnostics},
        "search": {
            "raw_policies": len(policy_grid()),
            "unique_schedules": len(searched),
            "eligible": len(eligible),
            "rows": searched,
        },
        "selected": selected,
        "selected_trace": selected_trace,
        "carry_comparator": carry_comparator,
        "decision": decision,
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest = {
        "protocol": report["protocol"],
        "decision": decision,
        "policy": selected["policy"],
        "schedule_hash": selected["schedule_hash"],
        "source_hashes": sources.source_hashes,
        "pre2024_stats": selected["stats"],
        "control_gates": selected["control_gates"],
        "manifest_hash_without_self": "",
    }
    manifest["manifest_hash_without_self"] = _json_hash(
        {**manifest, "manifest_hash_without_self": ""}
    )
    manifest_path = Path(cfg.policy_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if cfg.docs_output:
        write_docs(report, cfg.docs_output)
    return report


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--futures-csv", default=DEFAULT_FUTURES)
    parser.add_argument("--spot-csv", default=DEFAULT_SPOT)
    parser.add_argument("--funding-csv", default=DEFAULT_FUNDING)
    parser.add_argument("--source-manifest", default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--carry-result", default=DEFAULT_CARRY_RESULT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--policy-manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--docs-output", default=DEFAULT_DOCS)
    return Config(**vars(parser.parse_args()))


def main() -> None:
    report = run(parse_args())
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "policy": report["selected"]["policy"],
                "stats": report["selected"]["stats"],
                "control_gates": report["selected"]["control_gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
