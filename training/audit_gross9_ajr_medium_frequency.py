#!/usr/bin/env python3
"""Stage a frozen asymmetric-jump-rejection sleeve against Gross9.

Selection is physically limited to the authoritative Gross9 ``train`` split
and the candidate's already-frozen pre-2024 replay.  Evaluation consumes the
frozen top1 without reranking and opens test2024, eval2025, and ytd2026.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import training.audit_gross9_pullback_premium_overheat_marginal as gross9
import training.portfolio_opt_added_alpha_update as portfolio
import training.search_asymmetric_jump_rejection_alpha as ajr
from training.audit_rank7_fresh_kimchi_fixed_portfolio import subaccount_bar_path
from training.search_inventory_purge_reclaim_alpha import _schedule_hash, equity_stats


AS_OF = "2026-07-18"
CANDIDATE = "asymmetric_jump_rejection"
WEIGHTS = (0.10, 0.25, 0.50)
SELECTION_SPLITS = ("train",)
EVAL_SPLITS = ("test2024", "eval2025", "ytd2026")
BASELINE_WEIGHTS = dict(gross9.BASELINE_WEIGHTS)
BASELINE_GROSS = float(sum(BASELINE_WEIGHTS.values()))
NORMAL_COST = 0.0006
STRESS_COST = 0.0010
MIN_TRAIN_CANDIDATE_TRADES = 250
SELECTION_MAX_MDD_PCT = 40.0
EVAL_MAX_MDD_PCT = 20.0
MIN_COMPLETED_TRADES_PER_WEEK = 3.0

DEFAULT_SELECTION_OUTPUT = Path(
    "results/gross9_ajr_medium_frequency_selection_2026-07-18.json"
)
DEFAULT_EVAL_OUTPUT = Path(
    "results/gross9_ajr_medium_frequency_eval_2026-07-18.json"
)


@dataclass(frozen=True)
class Config:
    selection_output: str = str(DEFAULT_SELECTION_OUTPUT)
    eval_output: str = str(DEFAULT_EVAL_OUTPUT)
    market_csv: str = portfolio.Config.input_csv
    market_with_oi_csv: str = gross9.Config.market_with_oi_csv
    funding_csv: str = portfolio.Config.funding_csv
    premium_csv: str = portfolio.Config.premium_csv
    candidate_market_csv: str = (
        "/home/pakchu/rllm/data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
    )
    candidate_metrics_csv: str = (
        "/home/pakchu/rllm/data/binance_um_metrics_BTCUSDT_5m_2020-09-01_2026-06-01.csv.gz"
    )
    candidate_funding_csv: str = (
        "/home/pakchu/rllm/data/binance_um_aux_btc_2020_2026/"
        "BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
    )
    candidate_selection: str = (
        "results/asymmetric_jump_rejection_alpha_selection_2026-07-15.json"
    )
    candidate_oos: str = (
        "results/asymmetric_jump_rejection_alpha_oos_2026-07-15.json"
    )
    gross9_anchor: str = gross9.Config.gross9_anchor
    gross9_config: str = gross9.Config.gross9_config
    gross9_result: str = gross9.Config.gross9_result
    rank7_capacity_evidence: str = gross9.Config.rank7_capacity_evidence


def canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def json_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def finalize_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("result_hash", None)
    result["result_hash"] = json_hash(result)
    return result


def verify_result_hash(payload: Mapping[str, Any]) -> None:
    observed = str(payload.get("result_hash", ""))
    expected = json_hash({k: v for k, v in payload.items() if k != "result_hash"})
    if observed != expected:
        raise RuntimeError(f"result hash drifted: {observed} != {expected}")


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(canonical_json(payload) + b"\n")
    temporary.replace(destination)


def same_gross_weights(
    candidate_weight: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return candidate portfolio and pro-rata Gross9 at identical gross."""
    weight = float(candidate_weight)
    if weight not in WEIGHTS:
        raise ValueError(f"candidate weight is outside frozen grid: {weight}")
    combined = {**BASELINE_WEIGHTS, CANDIDATE: weight}
    scale = (BASELINE_GROSS + weight) / BASELINE_GROSS
    comparator = {name: value * scale for name, value in BASELINE_WEIGHTS.items()}
    if not np.isclose(sum(combined.values()), sum(comparator.values())):
        raise RuntimeError("same-gross comparator construction drifted")
    return combined, comparator


def completed_trades_per_week(trades: int, start: str, end: str) -> float:
    weeks = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / (7 * 86_400)
    if weeks <= 0.0:
        raise ValueError("frequency window must have positive duration")
    return float(trades) / weeks


def frequency_report(
    metric: Mapping[str, Any], *, start: str, end: str
) -> dict[str, Any]:
    per_sleeve = {
        str(name): int(count)
        for name, count in metric.get("trades_by_sleeve", {}).items()
    }
    total = int(metric["trades"])
    if sum(per_sleeve.values()) != total:
        raise RuntimeError("portfolio and per-sleeve completed trade counts differ")
    return {
        "completed_trades": total,
        "completed_trades_by_sleeve": per_sleeve,
        "calendar_weeks": float(
            (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds()
            / (7 * 86_400)
        ),
        "combined_completed_trades_per_week": completed_trades_per_week(
            total, start, end
        ),
    }


def _gross9_cfg(cfg: Config) -> gross9.Config:
    return gross9.Config(
        market_csv=cfg.market_csv,
        market_with_oi_csv=cfg.market_with_oi_csv,
        funding_csv=cfg.funding_csv,
        premium_csv=cfg.premium_csv,
        gross9_anchor=cfg.gross9_anchor,
        gross9_config=cfg.gross9_config,
        gross9_result=cfg.gross9_result,
        rank7_capacity_evidence=cfg.rank7_capacity_evidence,
    )


def _candidate_cfg(cfg: Config) -> ajr.Config:
    return ajr.Config(
        input_csv=cfg.candidate_market_csv,
        metrics_csv=cfg.candidate_metrics_csv,
        funding_csv=cfg.candidate_funding_csv,
        output="/tmp/no_write_gross9_ajr.json",
        manifest_output="/tmp/no_write_gross9_ajr_manifest.json",
        exclude_from="2026-06-02",
        leverage=0.5,
        fee_rate=0.0005,
        slippage_rate=0.0001,
    )


def validate_candidate_freeze(cfg: Config) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = json.loads(Path(cfg.candidate_selection).read_text(encoding="utf-8"))
    oos = json.loads(Path(cfg.candidate_oos).read_text(encoding="utf-8"))
    candidate_cfg = _candidate_cfg(cfg)
    if frozen.get("phase") != "pre_2024_freeze" or frozen.get("oos_opened") is not False:
        raise RuntimeError("AJR selection artifact is not the sealed pre-2024 freeze")
    ajr._validate_manifest(candidate_cfg, frozen)
    if frozen.get("freeze_hash") != oos.get("freeze_hash"):
        raise RuntimeError("AJR OOS replay is not bound to the selection freeze")
    if frozen.get("selection_schedule_hashes") != oos.get("selection_schedule_hashes"):
        raise RuntimeError("AJR selection schedule hashes differ across artifacts")
    if oos.get("phase") != "frozen_oos_replay":
        raise RuntimeError("AJR OOS artifact is not a frozen replay")
    return frozen, oos


def prepare_candidate(
    cfg: Config, *, cutoff: str, validate_oos_hashes: bool
) -> dict[str, Any]:
    frozen, known_oos = validate_candidate_freeze(cfg)
    candidate_cfg = _candidate_cfg(cfg)
    engine_cfg = ajr._engine_config(candidate_cfg)
    market, funding, prefix_hashes = ajr._load_sources(engine_cfg, cutoff=cutoff)
    dates = pd.to_datetime(market["date"])
    features = ajr.build_signal_features(market)
    prefix = (dates < pd.Timestamp(ajr.SELECTION_END)).to_numpy(bool)
    if prefix_hashes != frozen["source_prefix_hashes"] and cutoff == ajr.SELECTION_END:
        raise RuntimeError("AJR frozen source-prefix hashes drifted")
    ajr.validate_feature_prefix(frozen, features, dates)
    anchors = np.arange(
        int(ajr.SPEC["warmup_bars"]),
        len(market) - int(ajr.SPEC["hold_bars"]) - 2,
        int(ajr.SPEC["anchor_stride_bars"]),
        dtype=np.int64,
    )
    long_active, short_active = ajr.build_masks(features, anchors, frozen["thresholds"])
    prefix_anchors = np.arange(
        int(ajr.SPEC["warmup_bars"]),
        int(prefix.sum()) - int(ajr.SPEC["hold_bars"]) - 2,
        int(ajr.SPEC["anchor_stride_bars"]),
        dtype=np.int64,
    )
    prefix_long, prefix_short = ajr.build_masks(
        features, prefix_anchors, frozen["thresholds"]
    )
    if ajr.activation_hash(prefix_anchors, prefix_long, prefix_short) != frozen["activation_hash"]:
        raise RuntimeError("AJR activation hash drifted")
    engine = ajr.ExecutionEngine(market, funding, engine_cfg)
    selection_schedules = ajr._schedule_windows(
        engine, anchors, long_active, short_active, ajr.FIT_WINDOWS
    )
    observed_selection_hashes = {
        name: _schedule_hash(value) for name, value in selection_schedules.items()
    }
    if observed_selection_hashes != frozen["selection_schedule_hashes"]:
        raise RuntimeError("AJR frozen pre-2024 schedule hashes drifted")
    if validate_oos_hashes:
        schedules = ajr._schedule_windows(
            engine, anchors, long_active, short_active, ajr.OOS_WINDOWS
        )
        observed = {name: _schedule_hash(value) for name, value in schedules.items()}
        if observed != known_oos["oos_schedule_hashes"]:
            raise RuntimeError("AJR known-OOS schedule hashes drifted")
    return {
        "market": market,
        "funding": funding,
        "engine": engine,
        "engine_cfg": engine_cfg,
        "anchors": anchors,
        "long_active": long_active,
        "short_active": short_active,
        "freeze_hash": frozen["freeze_hash"],
        "selection_schedule_hashes": observed_selection_hashes,
    }


def install_candidate_sleeve() -> None:
    if CANDIDATE not in portfolio.SLEEVES:
        portfolio.SLEEVES = (*portfolio.SLEEVES, CANDIDATE)
    portfolio.FAMILIES[CANDIDATE] = "asymmetric_jump_rejection"


def _candidate_trades(candidate: Mapping[str, Any], start: str, end: str) -> list[Any]:
    return ajr._schedule_window(
        candidate["engine"], candidate["anchors"], candidate["long_active"],
        candidate["short_active"], _window_name(start, end),
    )


def _window_name(start: str, end: str) -> str:
    for name, bounds in ajr.WINDOWS.items():
        if tuple(bounds) == (start, end):
            return name
    # Gross9 train spans AJR fit plus select_2023 and is intentionally composed.
    if (start, end) == portfolio.SPLIT_BOUNDS["train"]:
        return "__gross9_train__"
    raise KeyError(f"AJR has no frozen window for {start}..{end}")


def candidate_trades(candidate: Mapping[str, Any], split: str) -> list[Any]:
    start, end = portfolio.SPLIT_BOUNDS[split]
    if split == "train":
        fit = ajr._schedule_window(
            candidate["engine"], candidate["anchors"], candidate["long_active"],
            candidate["short_active"], "fit",
        )
        select = ajr._schedule_window(
            candidate["engine"], candidate["anchors"], candidate["long_active"],
            candidate["short_active"], "select_2023",
        )
        return [*fit, *select]
    if split == "ytd2026":
        return ajr._schedule_window(
            candidate["engine"], candidate["anchors"], candidate["long_active"],
            candidate["short_active"], "holdout_2026",
        )
    return _candidate_trades(candidate, start, end)


def append_candidate_events(
    events: list[dict[str, Any]], gross9_market: pd.DataFrame,
    candidate: Mapping[str, Any], splits: Iterable[str], *, stress: bool,
) -> dict[str, Any]:
    execution_cfg = candidate["engine_cfg"]
    execution_cfg = replace(
        execution_cfg, fee_rate=0.0009 if stress else 0.0005, slippage_rate=0.0001
    )
    expected_cost = STRESS_COST if stress else NORMAL_COST
    if not np.isclose(execution_cfg.fee_rate + execution_cfg.slippage_rate, expected_cost):
        raise RuntimeError("AJR replay cost drifted")
    output: dict[str, Any] = {}
    for split in splits:
        start, end = portfolio.SPLIT_BOUNDS[split]
        trades = candidate_trades(candidate, split)
        path = subaccount_bar_path(
            candidate["market"], candidate["funding"], trades, execution_cfg,
            start=start, end=end, hold_bars=lambda _trade: int(ajr.SPEC["hold_bars"]),
        )
        events.append(portfolio.path_event(
            gross9_market, path, split=split, sleeve=CANDIDATE, trades=trades
        ))
        output[split] = {
            "completed_trades": len(trades),
            "schedule_hash": _schedule_hash(trades),
            "final_equity": float(path.final_equity),
        }
    return output


def _metric(arrays: Mapping[str, Mapping[str, Any]], split: str,
            weights: Mapping[str, float]) -> dict[str, Any]:
    return portfolio.strict_metric(arrays[split], portfolio.years_for(split), dict(weights))


def selection_cell(
    weight: float, normal_arrays: Mapping[str, Mapping[str, Any]],
    stress_arrays: Mapping[str, Mapping[str, Any]],
    baseline_arrays: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    combined_weights, comparator_weights = same_gross_weights(weight)
    combined = _metric(normal_arrays, "train", combined_weights)
    comparator = _metric(baseline_arrays, "train", comparator_weights)
    standalone = _metric(normal_arrays, "train", {CANDIDATE: 1.0})
    standalone_stress = _metric(stress_arrays, "train", {CANDIDATE: 1.0})
    improvement = float(
        combined["cagr_to_strict_mdd"] - comparator["cagr_to_strict_mdd"]
    )
    checks = {
        "train_same_gross_cagr_mdd_improves": improvement > 0.0,
        "candidate_standalone_positive": standalone["absolute_return_pct"] > 0.0,
        "candidate_stress_positive": standalone_stress["absolute_return_pct"] > 0.0,
        "combined_mdd_at_most_40": combined["strict_mdd_pct"] <= SELECTION_MAX_MDD_PCT,
        "candidate_at_least_250_train_trades": standalone["trades"] >= MIN_TRAIN_CANDIDATE_TRADES,
    }
    return {
        "candidate_weight": float(weight),
        "configured_gross": float(BASELINE_GROSS + weight),
        "combined": combined,
        "same_gross_comparator": comparator,
        "standalone": standalone,
        "standalone_stress": standalone_stress,
        "train_same_gross_cagr_mdd_improvement": improvement,
        "checks": checks,
        "passes": bool(all(checks.values())),
        "selection_key": [improvement, -float(weight)],
    }


def rank_selection_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: (bool(row["passes"]), *map(float, row["selection_key"])),
        reverse=True,
    )


def selection_payload(cfg: Config) -> dict[str, Any]:
    install_candidate_sleeve()
    original_splits = gross9.SELECTION_SPLITS
    gross9.SELECTION_SPLITS = SELECTION_SPLITS
    try:
        market, masks, base_events, source_meta = gross9.build_selection_context(
            _gross9_cfg(cfg)
        )
    finally:
        gross9.SELECTION_SPLITS = original_splits
    if tuple(masks) != SELECTION_SPLITS:
        raise RuntimeError("selection context opened a non-train split")
    candidate = prepare_candidate(cfg, cutoff=ajr.SELECTION_END, validate_oos_hashes=False)
    gross9.validate_shared_market(candidate["market"], market)
    base_arrays = portfolio.split_arrays(base_events, market, masks)
    authoritative = gross9.validate_authoritative_gross9(
        _gross9_cfg(cfg), base_arrays, SELECTION_SPLITS
    )
    normal_events, stress_events = list(base_events), list(base_events)
    normal_meta = append_candidate_events(
        normal_events, market, candidate, SELECTION_SPLITS, stress=False
    )
    stress_meta = append_candidate_events(
        stress_events, market, candidate, SELECTION_SPLITS, stress=True
    )
    normal_arrays = portfolio.split_arrays(normal_events, market, masks)
    stress_arrays = portfolio.split_arrays(stress_events, market, masks)
    rows = rank_selection_rows(
        selection_cell(weight, normal_arrays, stress_arrays, base_arrays)
        for weight in WEIGHTS
    )
    passing = [row for row in rows if row["passes"]]
    top1 = passing[0] if passing else None
    return finalize_payload({
        "as_of": AS_OF,
        "phase": "selection",
        "selection_windows": list(SELECTION_SPLITS),
        "candidate_source_end_exclusive": ajr.SELECTION_END,
        "weight_grid": list(WEIGHTS),
        "candidate_freeze_hash": candidate["freeze_hash"],
        "candidate_selection_schedule_hashes": candidate["selection_schedule_hashes"],
        "authoritative_gross9": authoritative,
        "source_meta": source_meta,
        "candidate_normal_meta": normal_meta,
        "candidate_stress_meta": stress_meta,
        "rows": rows,
        "frozen_top1": top1,
        "decision": "freeze_top1" if top1 else "reject",
        "future_opened": False,
        "future_can_rerank": False,
    })


def verify_selection_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    verify_result_hash(payload)
    if payload.get("phase") != "selection":
        raise RuntimeError("selection artifact phase drifted")
    if payload.get("selection_windows") != ["train"]:
        raise RuntimeError("selection artifact opened a non-train window")
    if payload.get("candidate_source_end_exclusive") != ajr.SELECTION_END:
        raise RuntimeError("selection artifact was not physically pre-2024")
    if payload.get("weight_grid") != list(WEIGHTS):
        raise RuntimeError("selection weight grid drifted")
    if payload.get("future_opened") is not False or payload.get("future_can_rerank") is not False:
        raise RuntimeError("selection artifact permits future reranking")
    top = payload.get("frozen_top1")
    if not isinstance(top, dict) or top.get("passes") is not True:
        raise RuntimeError("selection artifact has no passing frozen top1")
    if float(top["candidate_weight"]) not in WEIGHTS:
        raise RuntimeError("frozen top1 weight is outside the grid")
    return top


def evaluation_window(
    split: str, weight: float, normal_arrays: Mapping[str, Mapping[str, Any]],
    stress_arrays: Mapping[str, Mapping[str, Any]],
    baseline_arrays: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    combined_weights, comparator_weights = same_gross_weights(weight)
    combined = _metric(normal_arrays, split, combined_weights)
    stress = _metric(stress_arrays, split, combined_weights)
    comparator = _metric(baseline_arrays, split, comparator_weights)
    start, end = portfolio.SPLIT_BOUNDS[split]
    frequency = frequency_report(combined, start=start, end=end)
    checks = {
        "combined_base_positive": combined["absolute_return_pct"] > 0.0,
        "combined_candidate_stress_positive": stress["absolute_return_pct"] > 0.0,
        "combined_mdd_at_most_20": combined["strict_mdd_pct"] <= EVAL_MAX_MDD_PCT,
        "combined_frequency_at_least_3_per_week": (
            frequency["combined_completed_trades_per_week"]
            >= MIN_COMPLETED_TRADES_PER_WEEK
        ),
    }
    return {
        "window": split,
        "start": start,
        "end_exclusive": end,
        "combined": combined,
        "candidate_stress_combined": stress,
        "same_gross_comparator": comparator,
        "frequency": frequency,
        "checks": checks,
        "passes": bool(all(checks.values())),
    }


def eval_payload(cfg: Config) -> dict[str, Any]:
    # Deliberately no selection replay or rank call: eval consumes the seal.
    selection = json.loads(Path(cfg.selection_output).read_text(encoding="utf-8"))
    top = verify_selection_artifact(selection)
    frozen, _ = validate_candidate_freeze(cfg)
    if selection.get("candidate_freeze_hash") != frozen["freeze_hash"]:
        raise RuntimeError("selection is bound to a different AJR freeze")
    if selection.get("candidate_selection_schedule_hashes") != frozen["selection_schedule_hashes"]:
        raise RuntimeError("selection AJR schedule binding drifted")

    install_candidate_sleeve()
    market, masks, base_events, source_meta = gross9.build_full_context(_gross9_cfg(cfg))
    masks = {name: masks[name] for name in EVAL_SPLITS}
    base_events = [event for event in base_events if event["split"] in EVAL_SPLITS]
    candidate = prepare_candidate(cfg, cutoff="2026-06-02", validate_oos_hashes=True)
    gross9.validate_shared_market(candidate["market"], market)
    base_arrays = portfolio.split_arrays(base_events, market, masks)
    authoritative = gross9.validate_authoritative_gross9(
        _gross9_cfg(cfg), base_arrays, EVAL_SPLITS
    )
    normal_events, stress_events = list(base_events), list(base_events)
    normal_meta = append_candidate_events(
        normal_events, market, candidate, EVAL_SPLITS, stress=False
    )
    stress_meta = append_candidate_events(
        stress_events, market, candidate, EVAL_SPLITS, stress=True
    )
    normal_arrays = portfolio.split_arrays(normal_events, market, masks)
    stress_arrays = portfolio.split_arrays(stress_events, market, masks)
    weight = float(top["candidate_weight"])
    windows = {
        split: evaluation_window(split, weight, normal_arrays, stress_arrays, base_arrays)
        for split in EVAL_SPLITS
    }
    passed = all(row["passes"] for row in windows.values())
    return finalize_payload({
        "as_of": AS_OF,
        "phase": "evaluation",
        "selection_result_hash": selection["result_hash"],
        "candidate_freeze_hash": candidate["freeze_hash"],
        "frozen_candidate_weight": weight,
        "opened_windows": list(EVAL_SPLITS),
        "authoritative_gross9": authoritative,
        "source_meta": source_meta,
        "candidate_normal_meta": normal_meta,
        "candidate_stress_meta": stress_meta,
        "windows": windows,
        "passes": passed,
        "decision": "promote" if passed else "veto",
        "reranked": False,
    })


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("selection", "eval"))
    parser.add_argument("--selection-output", default=Config.selection_output)
    parser.add_argument("--eval-output", default=Config.eval_output)
    args = parser.parse_args(argv)
    cfg = Config(selection_output=args.selection_output, eval_output=args.eval_output)
    payload = selection_payload(cfg) if args.phase == "selection" else eval_payload(cfg)
    output = cfg.selection_output if args.phase == "selection" else cfg.eval_output
    atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
