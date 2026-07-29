#!/usr/bin/env python3
"""Audit fixed PPOSM as a same-gross Gross9 marginal sleeve.

The pullback premium-overheat state machine (PPOSM) was frozen before 2024
and has already been replayed standalone through 2026H1.  This audit does not
claim pristine alpha discovery.  It asks one still-unmeasured practical
question: does the exact frozen path improve the authoritative Gross9
portfolio against a pro-rata Gross9 comparator at identical configured gross?

Selection uses only the Gross9 train and 2024 clocks.  If a weight passes, the
single frozen top1 is replayed on 2025 and 2026H1 without reranking or repair.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import training.audit_gross9_fixed_candidate_state_substitution as gross9_context
import training.portfolio_opt_added_alpha_update as portfolio
import training.portfolio_opt_all_discovered_alpha_gross10 as legacy_all
import training.portfolio_opt_combined_rex_new_alpha as legacy_base
import training.search_pullback_premium_overheat_state_machine_alpha as pposm
from training.audit_confirmed_pullback_squeeze_live_parity import (
    PRE2024_WINDOWS,
    _activation_hash,
    _execution_config,
    _fit_active,
    _load_bundle,
    decision_mask,
    live_decision_features,
)
from training.audit_rank7_fresh_kimchi_fixed_portfolio import (
    subaccount_bar_path,
)
from training.search_inventory_purge_reclaim_alpha import (
    ExecutionEngine,
    _schedule_hash,
    equity_stats,
)


AS_OF = "2026-07-28"
CANDIDATE = "pullback_premium_overheat_state_machine"
PREREGISTRATION = Path(
    "results/gross9_pullback_premium_overheat_marginal_preregistration_2026-07-28.json"
)
SELECTION_OUTPUT = Path(
    "results/gross9_pullback_premium_overheat_marginal_selection_2026-07-28.json"
)
EVAL_OUTPUT = Path(
    "results/gross9_pullback_premium_overheat_marginal_eval_2026-07-28.json"
)
SELECTION_DOCS = Path(
    "docs/gross9-pullback-premium-overheat-marginal-selection-2026-07-28.md"
)
EVAL_DOCS = Path(
    "docs/gross9-pullback-premium-overheat-marginal-eval-2026-07-28.md"
)
EXPECTED_PREREGISTRATION_SHA256 = (
    "0fbb2680383899b867f6ff3d6c381be7796f3075bee4f2a9a2573da550ce1553"
)
SELECTION_SPLITS = ("train", "test2024")
EVAL_SPLITS = ("eval2025", "ytd2026")
FROZEN_REX_ROW_INDEX = 7
FROZEN_REX_SLEEVE = f"cand_rex_veto_{FROZEN_REX_ROW_INDEX}"
EXPECTED_FROZEN_REX_GATES_HASH = (
    "cd63c780bdb8848700476d585ae8e0cb95713aed856f7d9d681a7a0b5d5575fc"
)
WEIGHTS = (0.25, 0.50, 0.75, 1.00)
NORMAL_COST = 0.0006
STRESS_COST = 0.0010
UNIT_LEVERAGE = 0.50
BASELINE_WEIGHTS = {
    "cand_rex_veto_7": 1.6,
    "fresh_kimchi_fx": 2.0,
    "frozen_annual_rank7": 3.0,
    "markov_transition_long": 2.0,
    "rex_taker_low_range_position": 0.4,
}
BASELINE_GROSS = float(sum(BASELINE_WEIGHTS.values()))
if not np.isclose(BASELINE_GROSS, 9.0):
    raise RuntimeError("authoritative Gross9 weights no longer sum to 9")


@dataclass(frozen=True)
class Config:
    preregistration: str = str(PREREGISTRATION)
    selection_output: str = str(SELECTION_OUTPUT)
    eval_output: str = str(EVAL_OUTPUT)
    selection_docs: str = str(SELECTION_DOCS)
    eval_docs: str = str(EVAL_DOCS)
    market_csv: str = portfolio.Config.input_csv
    market_with_oi_csv: str = (
        "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz"
    )
    funding_csv: str = portfolio.Config.funding_csv
    premium_csv: str = portfolio.Config.premium_csv
    candidate_selection: str = (
        "results/pullback_premium_overheat_state_machine_selection_2026-07-15.json"
    )
    candidate_oos: str = (
        "results/pullback_premium_overheat_state_machine_oos_2026-07-15.json"
    )
    candidate_script: str = (
        "training/search_pullback_premium_overheat_state_machine_alpha.py"
    )
    gross9_anchor: str = (
        "results/gross9_pre2025_authoritative_anchor_2026-07-28.json"
    )
    gross9_config: str = (
        "configs/shadow/portfolio_rank7_capacity_candidate_2026-07-28.json"
    )
    gross9_result: str = (
        "results/portfolio_rank7_capacity_update_2026-07-28.json"
    )
    rank7_capacity_evidence: str = (
        "results/expanding_extratrees_rank7_leverage_battery_2026-07-27.json"
    )


def canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def json_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finalize_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("result_hash", None)
    result["result_hash"] = json_hash(result)
    return result


def verify_result_hash(payload: Mapping[str, Any]) -> None:
    observed = str(payload.get("result_hash", ""))
    expected = json_hash(
        {key: value for key, value in payload.items() if key != "result_hash"}
    )
    if observed != expected:
        raise RuntimeError(f"result hash drifted: {observed} != {expected}")


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(canonical_json(payload) + b"\n")
    temporary.replace(destination)


def configured_inputs(cfg: Config) -> dict[str, str]:
    return {
        "market": cfg.market_csv,
        "market_with_oi": cfg.market_with_oi_csv,
        "funding": cfg.funding_csv,
        "premium": cfg.premium_csv,
        "candidate_selection": cfg.candidate_selection,
        "candidate_oos": cfg.candidate_oos,
        "candidate_script": cfg.candidate_script,
        "gross9_anchor": cfg.gross9_anchor,
        "gross9_config": cfg.gross9_config,
        "gross9_result": cfg.gross9_result,
        "rank7_capacity_evidence": cfg.rank7_capacity_evidence,
        "gross9_context_builder": (
            "training/audit_gross9_fixed_candidate_state_substitution.py"
        ),
        "gross9_portfolio_engine": (
            "training/portfolio_opt_added_alpha_update.py"
        ),
        "candidate_path_engine": (
            "training/audit_rank7_fresh_kimchi_fixed_portfolio.py"
        ),
    }


def validate_preregistration_semantics(payload: Mapping[str, Any]) -> None:
    if payload.get("name") != (
        "gross9_fixed_pullback_premium_overheat_same_gross_marginal"
    ):
        raise RuntimeError("unexpected preregistration name")
    if payload.get("status") != (
        "preregistered_before_pposm_gross9_portfolio_interaction_scan"
    ):
        raise RuntimeError("preregistration status drifted")
    if payload.get("candidate_disclosure", {}).get(
        "standalone_future_already_exposed"
    ) is not True:
        raise RuntimeError("candidate contamination disclosure is missing")
    selection = payload["selection_contract"]
    if tuple(map(float, selection["candidate_weight_grid"])) != WEIGHTS:
        raise RuntimeError("candidate weight grid drifted")
    if selection.get("selection_windows") != ["train", "test2024"]:
        raise RuntimeError("selection windows drifted")
    if selection.get("top1_only") is not True:
        raise RuntimeError("top1-only selection drifted")
    if selection.get("gross_cap") != 10.0:
        raise RuntimeError("gross cap drifted")
    future = payload["future_veto_contract"]
    if future.get("windows") != ["eval2025", "ytd2026"]:
        raise RuntimeError("future windows drifted")
    if future.get("future_can_rerank") is not False:
        raise RuntimeError("future reranking was enabled")
    if future.get("future_can_repair") is not False:
        raise RuntimeError("future repair was enabled")
    formula = payload["same_gross_formula"]
    observed = {
        str(name): float(value)
        for name, value in formula["baseline_weights"].items()
    }
    if observed != BASELINE_WEIGHTS:
        raise RuntimeError("same-gross baseline weights drifted")
    if not np.isclose(
        float(formula["baseline_configured_gross_units"]), BASELINE_GROSS
    ):
        raise RuntimeError("same-gross baseline gross drifted")


def load_preregistration(path: str | Path) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed != EXPECTED_PREREGISTRATION_SHA256:
        raise RuntimeError(
            f"preregistration hash drifted: {observed} != "
            f"{EXPECTED_PREREGISTRATION_SHA256}"
        )
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_preregistration_semantics(payload)
    return payload


def validate_inputs(
    cfg: Config, preregistration: Mapping[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    expected = preregistration["input_provenance"]
    for name, path in configured_inputs(cfg).items():
        observed = sha256_file(path)
        wanted = str(expected[name]["sha256"])
        if observed != wanted:
            raise RuntimeError(
                f"input hash drifted for {name}: {observed} != {wanted}"
            )
        output[name] = {
            "path": path,
            "sha256": observed,
            "validated": True,
        }
    return output


def _candidate_cfg(cfg: Config) -> pposm.Config:
    return pposm.Config(
        input_csv=cfg.market_csv,
        funding_csv=cfg.funding_csv,
        premium_csv=cfg.premium_csv,
        output="/tmp/no_write_pposm_marginal.json",
        manifest_output="/tmp/no_write_pposm_marginal_manifest.json",
        docs_output="",
        exclude_from="2026-06-02",
        window_size=144,
        leverage=UNIT_LEVERAGE,
        fee_rate=0.0005,
        slippage_rate=0.0001,
        funding_tolerance="12h",
        live_premium_tolerance="10min",
    )


def _slim_stats(stats: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: stats[key]
        for key in (
            "absolute_return_pct",
            "cagr_pct",
            "strict_mdd_pct",
            "cagr_to_strict_mdd",
            "trades",
            "mean_net_bps",
            "win_rate",
        )
    }


def validate_candidate_freeze(
    cfg: Config,
) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = json.loads(Path(cfg.candidate_selection).read_text(encoding="utf-8"))
    known_oos = json.loads(Path(cfg.candidate_oos).read_text(encoding="utf-8"))
    if frozen.get("oos_opened") is not False:
        raise RuntimeError("candidate selection artifact is not the pre-OOS freeze")
    if frozen.get("spec_hash") != pposm._spec_hash():
        raise RuntimeError("candidate specification hash drifted")
    if frozen.get("implementation_hash") != pposm._implementation_hash():
        raise RuntimeError("candidate implementation hash drifted")
    if frozen.get("freeze_hash") != pposm._freeze_hash(frozen):
        raise RuntimeError("candidate freeze hash drifted")
    if frozen.get("freeze_hash") != known_oos.get("freeze_hash"):
        raise RuntimeError("candidate OOS artifact is not tied to the freeze")
    if frozen.get("selection_passed") is not True:
        raise RuntimeError("candidate did not pass its frozen pre-2024 selection")
    if pposm.FROZEN_CHAMPION != {
        "overheat": "premium_range_overheat",
        "action": "skip",
    }:
        raise RuntimeError("candidate champion drifted")
    frozen_execution = frozen["frozen_execution_config"]
    expected_execution = {
        "exclude_from": "2026-06-02",
        "window_size": 144,
        "leverage": UNIT_LEVERAGE,
        "fee_rate": 0.0005,
        "slippage_rate": 0.0001,
        "funding_tolerance": "12h",
        "live_premium_tolerance": "10min",
    }
    for key, value in expected_execution.items():
        if frozen_execution.get(key) != value:
            raise RuntimeError(f"candidate frozen execution drifted: {key}")

    candidate_cfg = _candidate_cfg(cfg)
    market, raw_features, funding, source_hashes = _load_bundle(
        candidate_cfg,
        cutoff=pposm.SELECTION_END,
        premium_tolerance=candidate_cfg.live_premium_tolerance,
    )
    if source_hashes != frozen["source_prefix_hashes"]:
        raise RuntimeError("candidate pre-2024 source prefix drifted")
    dates = pd.to_datetime(market["date"])
    decisions = decision_mask(
        dates, "live_hour_signal_bar", window_size=candidate_cfg.window_size
    )
    features = live_decision_features(raw_features)
    active, base_thresholds = _fit_active(
        features, dates, decisions
    )
    if base_thresholds != frozen["base_thresholds"]:
        raise RuntimeError("candidate base thresholds drifted")
    state_features = pposm.state_feature_frame(features)
    if pposm.feature_hash(state_features) != frozen["feature_prefix_hash"]:
        raise RuntimeError("candidate state feature prefix drifted")
    state_thresholds = pposm.fit_state_thresholds(
        state_features, dates, active
    )
    if state_thresholds != frozen["state_thresholds"]:
        raise RuntimeError("candidate state thresholds drifted")
    capitulation, overheat = pposm.build_state_masks(
        state_features,
        state_thresholds,
        pposm.FROZEN_CHAMPION["overheat"],
    )
    hashes = (
        _activation_hash(active, dates),
        _activation_hash(capitulation, dates),
        _activation_hash(overheat, dates),
    )
    expected_hashes = (
        frozen["activation_hash"],
        frozen["capitulation_hash"],
        frozen["overheat_hash"],
    )
    if hashes != expected_hashes:
        raise RuntimeError("candidate activation prefix drifted")
    execution_cfg = _execution_config(candidate_cfg, UNIT_LEVERAGE)
    engine = ExecutionEngine(market, funding, execution_cfg)
    schedules = {
        name: pposm.schedule_window(
            engine,
            active,
            capitulation,
            overheat,
            overheat_action=pposm.FROZEN_CHAMPION["action"],
            start=start,
            end=end,
        )
        for name, (start, end) in PRE2024_WINDOWS.items()
    }
    schedule_hashes = {
        name: _schedule_hash(schedules[name]) for name in PRE2024_WINDOWS
    }
    stats = {
        name: _slim_stats(
            equity_stats(
                schedules[name],
                start=start,
                end=end,
                cfg=execution_cfg,
            )
        )
        for name, (start, end) in PRE2024_WINDOWS.items()
    }
    if schedule_hashes != frozen["selection_schedule_hashes"]:
        raise RuntimeError("candidate pre-2024 schedule replay drifted")
    if stats != frozen["selection_stats"]:
        raise RuntimeError("candidate pre-2024 stats replay drifted")
    return frozen, known_oos


def prepare_candidate(
    cfg: Config,
    *,
    cutoff: str,
    validate_oos_hashes: bool,
) -> dict[str, Any]:
    frozen, known_oos = validate_candidate_freeze(cfg)
    candidate_cfg = _candidate_cfg(cfg)
    market, raw_features, funding, _ = _load_bundle(
        candidate_cfg,
        cutoff=cutoff,
        premium_tolerance=candidate_cfg.live_premium_tolerance,
    )
    dates = pd.to_datetime(market["date"])
    decisions = decision_mask(
        dates, "live_hour_signal_bar", window_size=candidate_cfg.window_size
    )
    features = live_decision_features(raw_features)
    active, base_thresholds = _fit_active(features, dates, decisions)
    if base_thresholds != frozen["base_thresholds"]:
        raise RuntimeError("full candidate replay changed frozen thresholds")
    state_features = pposm.state_feature_frame(features)
    prefix = (dates < pd.Timestamp(pposm.SELECTION_END)).to_numpy(bool)
    if pposm.feature_hash(state_features, prefix) != frozen["feature_prefix_hash"]:
        raise RuntimeError("full candidate replay changed state prefix")
    capitulation, overheat = pposm.build_state_masks(
        state_features,
        frozen["state_thresholds"],
        pposm.FROZEN_CHAMPION["overheat"],
    )
    prefix_dates = dates.loc[prefix].reset_index(drop=True)
    hashes = (
        _activation_hash(active[prefix], prefix_dates),
        _activation_hash(capitulation[prefix], prefix_dates),
        _activation_hash(overheat[prefix], prefix_dates),
    )
    expected_hashes = (
        frozen["activation_hash"],
        frozen["capitulation_hash"],
        frozen["overheat_hash"],
    )
    if hashes != expected_hashes:
        raise RuntimeError("full candidate replay changed activation prefix")
    execution_cfg = _execution_config(candidate_cfg, UNIT_LEVERAGE)
    engine = ExecutionEngine(market, funding, execution_cfg)
    if validate_oos_hashes:
        schedules = {
            name: pposm.schedule_window(
                engine,
                active,
                capitulation,
                overheat,
                overheat_action=pposm.FROZEN_CHAMPION["action"],
                start=start,
                end=end,
            )
            for name, (start, end) in pposm.FUTURE_WINDOWS.items()
        }
        observed = {
            name: _schedule_hash(schedule)
            for name, schedule in schedules.items()
        }
        if observed != known_oos["oos_schedule_hashes"]:
            raise RuntimeError("candidate known-OOS schedule replay drifted")
    return {
        "market": market,
        "funding": funding,
        "active": active,
        "capitulation": capitulation,
        "overheat": overheat,
        "execution_cfg": execution_cfg,
        "engine": engine,
        "freeze_hash": frozen["freeze_hash"],
        "known_oos_hash": sha256_file(cfg.candidate_oos),
    }


def install_candidate_sleeve() -> None:
    if CANDIDATE not in portfolio.SLEEVES:
        portfolio.SLEEVES = (*portfolio.SLEEVES, CANDIDATE)
    portfolio.FAMILIES[CANDIDATE] = "pullback_state_machine"


def validate_shared_market(
    candidate_market: pd.DataFrame, gross9_market: pd.DataFrame
) -> None:
    if len(candidate_market) > len(gross9_market):
        raise RuntimeError("candidate market exceeds Gross9 clock")
    gross_prefix = gross9_market.iloc[: len(candidate_market)]
    left_dates = pd.DatetimeIndex(pd.to_datetime(candidate_market["date"]))
    right_dates = pd.DatetimeIndex(pd.to_datetime(gross_prefix["date"]))
    if not left_dates.equals(right_dates):
        raise RuntimeError("candidate and Gross9 date clocks differ")
    for column in ("open", "high", "low", "close"):
        left = pd.to_numeric(
            candidate_market[column], errors="raise"
        ).to_numpy(float)
        right = pd.to_numeric(
            gross_prefix[column], errors="raise"
        ).to_numpy(float)
        if not np.array_equal(left, right):
            raise RuntimeError(f"candidate and Gross9 {column} differ")


def _candidate_trades(
    candidate: Mapping[str, Any],
    *,
    start: str,
    end: str,
) -> list[Any]:
    return pposm.schedule_window(
        candidate["engine"],
        candidate["active"],
        candidate["capitulation"],
        candidate["overheat"],
        overheat_action=pposm.FROZEN_CHAMPION["action"],
        start=start,
        end=end,
    )


def append_candidate_events(
    events: list[dict[str, Any]],
    gross9_market: pd.DataFrame,
    candidate: Mapping[str, Any],
    splits: Iterable[str],
    *,
    stress: bool,
) -> dict[str, Any]:
    execution_cfg = candidate["execution_cfg"]
    if stress:
        execution_cfg = replace(
            execution_cfg,
            fee_rate=0.0009,
            slippage_rate=0.0001,
        )
        if not np.isclose(
            execution_cfg.fee_rate + execution_cfg.slippage_rate,
            STRESS_COST,
        ):
            raise RuntimeError("candidate stress cost drifted")
    else:
        if not np.isclose(
            execution_cfg.fee_rate + execution_cfg.slippage_rate,
            NORMAL_COST,
        ):
            raise RuntimeError("candidate normal cost drifted")

    output: dict[str, Any] = {}
    for split in splits:
        start, end = portfolio.SPLIT_BOUNDS[split]
        trades = _candidate_trades(candidate, start=start, end=end)
        path = subaccount_bar_path(
            candidate["market"],
            candidate["funding"],
            trades,
            execution_cfg,
            start=start,
            end=end,
            hold_bars=lambda _trade: int(pposm.SPEC["hold_bars"]),
        )
        events.append(
            portfolio.path_event(
                gross9_market,
                path,
                split=split,
                sleeve=CANDIDATE,
                trades=trades,
            )
        )
        output[split] = {
            "trades": len(trades),
            "schedule_hash": _schedule_hash(trades),
            "final_equity": float(path.final_equity),
        }
    return output


def _unique_rex_rows(
    report: Mapping[str, Any],
    top_n: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for bucket in ("top", "tte_top"):
        for row in report.get(bucket, [])[: int(top_n)]:
            key = json.dumps(row.get("gates", []), sort_keys=True)
            if key not in seen:
                seen.add(key)
                rows.append(row)
    return rows


def validate_frozen_rex_identity(legacy_cfg: legacy_all.Config) -> str:
    report = legacy_all.load_json(legacy_all.SCAN_FILES["rex_veto"])
    bounded_rows = _unique_rex_rows(report, FROZEN_REX_ROW_INDEX + 1)
    legacy_rows = _unique_rex_rows(
        report,
        int(legacy_cfg.candidate_rex_top_n),
    )
    if min(len(bounded_rows), len(legacy_rows)) <= FROZEN_REX_ROW_INDEX:
        raise RuntimeError("frozen REX row is missing")
    bounded_hash = json_hash(
        bounded_rows[FROZEN_REX_ROW_INDEX].get("gates", [])
    )
    legacy_hash = json_hash(
        legacy_rows[FROZEN_REX_ROW_INDEX].get("gates", [])
    )
    if {
        bounded_hash,
        legacy_hash,
    } != {EXPECTED_FROZEN_REX_GATES_HASH}:
        raise RuntimeError(
            "bounded frozen REX identity drifted: "
            f"{bounded_hash} != {legacy_hash} != "
            f"{EXPECTED_FROZEN_REX_GATES_HASH}"
        )
    return bounded_hash


def build_frozen_rex_context(
    legacy_cfg: legacy_all.Config,
    splits: Sequence[str],
) -> tuple[
    pd.DataFrame,
    dict[str, np.ndarray],
    list[dict[str, Any]],
]:
    """Build only the frozen REX row instead of every legacy trade event."""
    market, _, all_masks, _ = legacy_base.vw.ep._prep()
    missing = [split for split in splits if split not in all_masks]
    if missing:
        raise RuntimeError(f"legacy context is missing splits: {missing}")
    masks = {split: all_masks[split] for split in splits}
    validate_frozen_rex_identity(legacy_cfg)
    bounded_cfg = replace(
        legacy_cfg,
        candidate_rex_top_n=FROZEN_REX_ROW_INDEX + 1,
    )
    source_events: list[dict[str, Any]] = []
    legacy_sleeves = list(legacy_base.SLEEVES)
    extra_sleeves = list(legacy_all.EXTRA_SLEEVES)
    try:
        legacy_all.add_rex_veto_candidates(
            source_events,
            market,
            masks,
            bounded_cfg,
        )
    finally:
        legacy_base.SLEEVES[:] = legacy_sleeves
        legacy_all.EXTRA_SLEEVES[:] = extra_sleeves
    events = [
        event
        for event in source_events
        if event["split"] in masks
        and event["sleeve"] == FROZEN_REX_SLEEVE
    ]
    counts = {
        split: sum(event["split"] == split for event in events)
        for split in masks
    }
    if any(count != 1 for count in counts.values()):
        raise RuntimeError(f"frozen REX event replay drifted: {counts}")
    return market, masks, events


def build_selection_context(
    cfg: Config,
) -> tuple[
    pd.DataFrame,
    dict[str, np.ndarray],
    list[dict[str, Any]],
    dict[str, Any],
]:
    base_cfg = gross9_context.Config(
        market_csv=cfg.market_csv,
        market_with_oi_csv=cfg.market_with_oi_csv,
        funding_csv=cfg.funding_csv,
        premium_csv=cfg.premium_csv,
        gross9_pre2025_anchor=cfg.gross9_anchor,
        rank7_capacity_evidence=cfg.rank7_capacity_evidence,
        cost_rate=NORMAL_COST,
    )
    oi_cache = gross9_context.materialize_frozen_oi_cache(
        cfg.market_with_oi_csv
    )
    portfolio.ensure_runtime_inputs()
    capacity_cfg = portfolio.Config(
        input_csv=cfg.market_csv,
        funding_csv=cfg.funding_csv,
        premium_csv=cfg.premium_csv,
        rank7_family_gross_cap=3.0,
        rank7_capacity_evidence=cfg.rank7_capacity_evidence,
        cost_rate=NORMAL_COST,
    )
    capacity = portfolio.validate_rank7_capacity_evidence(capacity_cfg)
    legacy_cfg = legacy_all.Config(
        random_samples=0,
        candidate_rex_top_n=50,
        train_mdd_cap=40.0,
        oos_mdd_cap=20.0,
        gross_cap=10.0,
        min_nonzero_weight=0.25,
        weight_step=0.05,
        cost_rate=NORMAL_COST,
    )
    market, masks, events = build_frozen_rex_context(
        legacy_cfg,
        SELECTION_SPLITS,
    )
    portfolio.attach_live_rex_ohlc(events, market, masks, capacity_cfg)
    portfolio.attach_default_favorable(events, market)

    features = gross9_context.build_candidate_feature_frame(market)
    markov = portfolio.markov_active(market, features)
    markov_counts = portfolio.append_mask_policy(
        events,
        market,
        masks,
        name="markov_transition_long",
        long_active=markov,
        short_active=np.zeros(len(market), dtype=bool),
        hold=576,
        stride=12,
        cost_rate=NORMAL_COST,
    )
    funding_active, funding_meta = portfolio.funding_lr_active(market)
    funding_counts = portfolio.append_mask_policy(
        events,
        market,
        masks,
        name="funding_premium_lr_impact_central",
        long_active=funding_active,
        short_active=np.zeros(len(market), dtype=bool),
        hold=576,
        stride=12,
        cost_rate=NORMAL_COST,
    )
    rex_counts = portfolio.append_rex_taker_policy(
        events,
        market,
        masks,
        cost_rate=NORMAL_COST,
    )
    path_counts = gross9_context._append_rank7_and_fresh_selection(
        events,
        base_cfg,
    )
    if tuple(masks) != SELECTION_SPLITS:
        raise RuntimeError("future masks entered selection")
    source_meta = {
        "rank7_capacity": capacity,
        "oi_cache": oi_cache,
        "feature_columns": int(features.shape[1]),
        "counts": {
            "markov_transition_long": markov_counts,
            "funding_premium_lr_impact_central": funding_counts,
            "rex_taker_low_range_position": rex_counts,
            **path_counts,
        },
        "funding_meta": funding_meta,
    }
    return market, masks, events, source_meta


def build_full_context(
    cfg: Config,
) -> tuple[
    pd.DataFrame,
    dict[str, np.ndarray],
    list[dict[str, Any]],
    dict[str, Any],
]:
    oi_cache = gross9_context.materialize_frozen_oi_cache(
        cfg.market_with_oi_csv
    )
    portfolio.ensure_runtime_inputs()
    capacity_cfg = portfolio.Config(
        input_csv=cfg.market_csv,
        funding_csv=cfg.funding_csv,
        premium_csv=cfg.premium_csv,
        rank7_family_gross_cap=3.0,
        rank7_capacity_evidence=cfg.rank7_capacity_evidence,
        cost_rate=NORMAL_COST,
    )
    capacity = portfolio.validate_rank7_capacity_evidence(capacity_cfg)
    legacy_cfg = legacy_all.Config(
        random_samples=0,
        candidate_rex_top_n=50,
        train_mdd_cap=40.0,
        oos_mdd_cap=20.0,
        gross_cap=10.0,
        min_nonzero_weight=0.25,
        weight_step=0.05,
        cost_rate=NORMAL_COST,
    )
    market, masks, events = build_frozen_rex_context(
        legacy_cfg,
        tuple(portfolio.SPLIT_BOUNDS),
    )
    portfolio.attach_live_rex_ohlc(events, market, masks, capacity_cfg)
    portfolio.attach_default_favorable(events, market)
    features = portfolio.feature_frame(market)
    markov = portfolio.markov_active(market, features)
    markov_counts = portfolio.append_mask_policy(
        events,
        market,
        masks,
        name="markov_transition_long",
        long_active=markov,
        short_active=np.zeros(len(market), dtype=bool),
        hold=576,
        stride=12,
        cost_rate=NORMAL_COST,
    )
    funding_active, funding_meta = portfolio.funding_lr_active(market)
    funding_counts = portfolio.append_mask_policy(
        events,
        market,
        masks,
        name="funding_premium_lr_impact_central",
        long_active=funding_active,
        short_active=np.zeros(len(market), dtype=bool),
        hold=576,
        stride=12,
        cost_rate=NORMAL_COST,
    )
    rex_counts = portfolio.append_rex_taker_policy(
        events, market, masks, cost_rate=NORMAL_COST
    )
    path_counts, path_meta = portfolio.append_rank7_and_fresh(
        events, capacity_cfg
    )
    return market, masks, events, {
        "oi_cache": oi_cache,
        "rank7_capacity": capacity,
        "markov_counts": markov_counts,
        "funding_counts": funding_counts,
        "funding_meta": funding_meta,
        "rex_counts": rex_counts,
        "path_counts": path_counts,
        "path_meta": path_meta,
    }


def _metric(
    arrays: Mapping[str, Mapping[str, Any]],
    split: str,
    weights: Mapping[str, float],
) -> dict[str, Any]:
    return portfolio.strict_metric(
        arrays[split],
        portfolio.years_for(split),
        dict(weights),
    )


def validate_authoritative_gross9(
    cfg: Config,
    arrays: Mapping[str, Mapping[str, Any]],
    splits: Iterable[str],
) -> dict[str, Any]:
    anchor = json.loads(Path(cfg.gross9_anchor).read_text(encoding="utf-8"))
    shadow = json.loads(Path(cfg.gross9_config).read_text(encoding="utf-8"))
    source = json.loads(Path(cfg.gross9_result).read_text(encoding="utf-8"))
    weights = {
        str(name): float(value) for name, value in anchor["weights"].items()
    }
    if weights != BASELINE_WEIGHTS:
        raise RuntimeError("authoritative Gross9 anchor weights drifted")
    if {
        str(name): float(value)
        for name, value in shadow["weights"].items()
    } != BASELINE_WEIGHTS:
        raise RuntimeError("Gross9 shadow weights drifted")
    top = source["frozen_pre2025_top1"]
    if {
        str(name): float(value)
        for name, value in top["weights"].items()
    } != BASELINE_WEIGHTS:
        raise RuntimeError("Gross9 source top1 weights drifted")
    observed: dict[str, Any] = {}
    for split in splits:
        observed[split] = _metric(arrays, split, BASELINE_WEIGHTS)
        expected = top["stats"][split]
        for key in (
            "absolute_return_pct",
            "cagr_pct",
            "strict_mdd_pct",
            "cagr_to_strict_mdd",
        ):
            if not np.isclose(
                float(observed[split][key]),
                float(expected[key]),
                rtol=0.0,
                atol=1e-9,
            ):
                raise RuntimeError(f"Gross9 drift in {split}/{key}")
        if int(observed[split]["trades"]) != int(expected["trades"]):
            raise RuntimeError(f"Gross9 trade count drift in {split}")
    return {
        "weights": weights,
        "stats": observed,
        "source_mode": source["mode"],
        "future_used_for_allocation_ranking": source[
            "future_used_for_allocation_ranking"
        ],
        "accounting_version": source["accounting_version"],
    }


def same_gross_weights(
    candidate_weight: float,
) -> tuple[dict[str, float], dict[str, float]]:
    weight = float(candidate_weight)
    combined = {**BASELINE_WEIGHTS, CANDIDATE: weight}
    comparator = {
        name: value * (BASELINE_GROSS + weight) / BASELINE_GROSS
        for name, value in BASELINE_WEIGHTS.items()
    }
    return combined, comparator


def _entry_jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    a = set(map(int, left))
    b = set(map(int, right))
    union = a | b
    return float(len(a & b) / len(union)) if union else 0.0


def entry_jaccards(
    arrays: Mapping[str, Mapping[str, Any]],
    splits: Iterable[str],
) -> dict[str, Any]:
    selected = tuple(splits)

    def pooled(sleeve: str) -> set[int]:
        values: set[int] = set()
        for split in selected:
            values.update(
                map(int, arrays[split]["entry_positions"][sleeve])
            )
        return values

    candidate = pooled(CANDIDATE)
    per_sleeve = {
        sleeve: _entry_jaccard(candidate, pooled(sleeve))
        for sleeve, weight in BASELINE_WEIGHTS.items()
        if weight > 0.0
    }
    return {
        "candidate_entries": len(candidate),
        "per_sleeve": per_sleeve,
        "max": max(per_sleeve.values(), default=0.0),
    }


def weighted_bar_returns(
    data: Mapping[str, Any], weights: Mapping[str, float]
) -> np.ndarray:
    vector = np.asarray(
        [float(weights.get(name, 0.0)) for name in portfolio.SLEEVES],
        dtype=float,
    )
    return vector @ np.asarray(data["R"], dtype=float)


def paired_weekly_effects(
    data: Mapping[str, Any],
    combined_weights: Mapping[str, float],
    comparator_weights: Mapping[str, float],
) -> np.ndarray:
    combined = weighted_bar_returns(data, combined_weights)
    comparator = weighted_bar_returns(data, comparator_weights)
    if np.any(1.0 + combined <= 0.0) or np.any(1.0 + comparator <= 0.0):
        raise RuntimeError("non-positive portfolio factor in paired test")
    difference = np.log1p(combined) - np.log1p(comparator)
    dates = pd.DatetimeIndex(data["dates"])
    iso = dates.isocalendar()
    frame = pd.DataFrame(
        {
            "year": iso["year"].to_numpy(int),
            "week": iso["week"].to_numpy(int),
            "effect": difference,
        }
    )
    weekly = frame.groupby(["year", "week"], sort=True)["effect"].sum()
    return weekly[np.abs(weekly) > 1e-15].to_numpy(float)


def paired_statistics(effects: np.ndarray) -> dict[str, Any]:
    values = np.asarray(effects, dtype=float)
    observed = float(values.sum())
    if len(values) == 0:
        return {
            "active_weeks": 0,
            "observed_total_log_effect": 0.0,
            "mean_weekly_log_effect": 0.0,
            "sign_flip_pvalue": 1.0,
            "bootstrap_90pct_lower_mean": 0.0,
        }
    rng = np.random.default_rng(20260728)
    exceed = 0
    remaining = 10_000
    while remaining:
        size = min(1_000, remaining)
        signs = rng.choice((-1.0, 1.0), size=(size, len(values)))
        simulated = signs @ values
        exceed += int(np.sum(simulated >= observed))
        remaining -= size
    bootstrap_rng = np.random.default_rng(20260729)
    means = np.empty(10_000, dtype=float)
    for start in range(0, 10_000, 1_000):
        size = min(1_000, 10_000 - start)
        indices = bootstrap_rng.integers(
            0, len(values), size=(size, len(values))
        )
        means[start : start + size] = values[indices].mean(axis=1)
    return {
        "active_weeks": int(len(values)),
        "observed_total_log_effect": observed,
        "mean_weekly_log_effect": float(values.mean()),
        "sign_flip_pvalue": float((1 + exceed) / 10_001),
        "bootstrap_90pct_lower_mean": float(
            np.quantile(means, 0.10, method="linear")
        ),
    }


def concat_array_data(
    parts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not parts:
        raise ValueError("at least one array part is required")
    return {
        "R": np.concatenate([np.asarray(part["R"]) for part in parts], axis=1),
        "A": np.concatenate([np.asarray(part["A"]) for part in parts], axis=1),
        "U": np.concatenate([np.asarray(part["U"]) for part in parts], axis=1),
        "L": np.concatenate([np.asarray(part["L"]) for part in parts], axis=1),
        "H": np.concatenate([np.asarray(part["H"]) for part in parts], axis=1),
        "counts": np.sum(
            [np.asarray(part["counts"]) for part in parts], axis=0
        ),
        "wins": np.sum(
            [np.asarray(part["wins"]) for part in parts], axis=0
        ),
        "dates": pd.DatetimeIndex(
            np.concatenate(
                [
                    pd.DatetimeIndex(part["dates"]).to_numpy()
                    for part in parts
                ]
            )
        ),
        "entry_positions": {
            sleeve: np.concatenate(
                [
                    np.asarray(
                        part["entry_positions"][sleeve], dtype=np.int64
                    )
                    for part in parts
                ]
            )
            for sleeve in portfolio.SLEEVES
        },
    }


def _calendar_years(start: str, end: str) -> float:
    return (
        (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds()
        / (365.25 * 86_400.0)
    )


def portfolio_cell(
    *,
    weight: float,
    normal_arrays: Mapping[str, Mapping[str, Any]],
    stress_arrays: Mapping[str, Mapping[str, Any]],
    baseline_arrays: Mapping[str, Mapping[str, Any]],
    splits: Sequence[str],
    requirements: Mapping[str, Any],
    jaccard: float,
) -> dict[str, Any]:
    combined_weights, comparator_weights = same_gross_weights(weight)
    standalone_weights = {CANDIDATE: 1.0}
    stats: dict[str, Any] = {}
    for split in splits:
        baseline = _metric(baseline_arrays, split, BASELINE_WEIGHTS)
        combined = _metric(normal_arrays, split, combined_weights)
        comparator = _metric(
            baseline_arrays, split, comparator_weights
        )
        stress = _metric(stress_arrays, split, combined_weights)
        standalone = _metric(
            normal_arrays, split, standalone_weights
        )
        standalone_stress = _metric(
            stress_arrays, split, standalone_weights
        )
        stats[split] = {
            "baseline": baseline,
            "combined": combined,
            "same_gross_comparator": comparator,
            "candidate_stress_combined": stress,
            "standalone": standalone,
            "standalone_stress": standalone_stress,
            "same_gross_ratio_improvement": float(
                combined["cagr_to_strict_mdd"]
                - comparator["cagr_to_strict_mdd"]
            ),
            "stress_same_gross_ratio_improvement": float(
                stress["cagr_to_strict_mdd"]
                - comparator["cagr_to_strict_mdd"]
            ),
            "absolute_return_retention": float(
                combined["absolute_return_pct"]
                / baseline["absolute_return_pct"]
            ),
            "strict_mdd_reduction": float(
                baseline["strict_mdd_pct"]
                - combined["strict_mdd_pct"]
            ),
        }

    checks: dict[str, bool] = {}
    for split in splits:
        row = stats[split]
        split_req = requirements["per_window"][split]
        checks[f"{split}_standalone_positive"] = bool(
            row["standalone"]["absolute_return_pct"] > 0.0
        )
        checks[f"{split}_standalone_stress_positive"] = bool(
            row["standalone_stress"]["absolute_return_pct"] > 0.0
        )
        checks[f"{split}_standalone_ratio"] = bool(
            row["standalone"]["cagr_to_strict_mdd"]
            >= float(split_req["minimum_standalone_ratio"])
        )
        checks[f"{split}_standalone_trades"] = bool(
            row["standalone"]["trades"]
            >= int(split_req["minimum_candidate_trades"])
        )
        checks[f"{split}_portfolio_mdd"] = bool(
            row["combined"]["strict_mdd_pct"]
            <= float(split_req["maximum_portfolio_mdd_pct"])
        )
        checks[f"{split}_return_retention"] = bool(
            row["absolute_return_retention"]
            >= float(split_req["minimum_return_retention"])
        )
        checks[f"{split}_same_gross_improvement"] = bool(
            row["same_gross_ratio_improvement"]
            >= float(split_req["minimum_same_gross_ratio_improvement"])
        )
        checks[f"{split}_stress_same_gross_improvement"] = bool(
            row["stress_same_gross_ratio_improvement"]
            >= float(
                split_req[
                    "minimum_stress_same_gross_ratio_improvement"
                ]
            )
        )
        checks[f"{split}_mdd_reduction"] = bool(
            row["strict_mdd_reduction"]
            >= float(split_req["minimum_strict_mdd_reduction_pct"])
        )
    checks["entry_jaccard"] = bool(
        jaccard <= float(requirements["maximum_entry_jaccard"])
    )
    selection_key = (
        min(
            stats[split]["same_gross_ratio_improvement"]
            for split in splits
        ),
        stats[splits[-1]]["same_gross_ratio_improvement"],
        sum(stats[split]["strict_mdd_reduction"] for split in splits),
        -float(weight),
    )
    return {
        "candidate_weight": float(weight),
        "configured_gross": float(BASELINE_GROSS + weight),
        "stats": stats,
        "checks": checks,
        "passes": bool(all(checks.values())),
        "selection_key": list(map(float, selection_key)),
    }


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            bool(row["passes"]),
            *map(float, row["selection_key"]),
        ),
        reverse=True,
    )


def selection_payload(cfg: Config) -> dict[str, Any]:
    preregistration = load_preregistration(cfg.preregistration)
    input_identity = validate_inputs(cfg, preregistration)
    install_candidate_sleeve()
    gross9_market, masks, base_events, source_meta = (
        build_selection_context(cfg)
    )
    candidate = prepare_candidate(
        cfg, cutoff="2025-01-01", validate_oos_hashes=False
    )
    validate_shared_market(candidate["market"], gross9_market)

    base_arrays = portfolio.split_arrays(base_events, gross9_market, masks)
    authoritative = validate_authoritative_gross9(
        cfg, base_arrays, SELECTION_SPLITS
    )
    normal_events = list(base_events)
    stress_events = list(base_events)
    normal_meta = append_candidate_events(
        normal_events,
        gross9_market,
        candidate,
        SELECTION_SPLITS,
        stress=False,
    )
    stress_meta = append_candidate_events(
        stress_events,
        gross9_market,
        candidate,
        SELECTION_SPLITS,
        stress=True,
    )
    normal_arrays = portfolio.split_arrays(
        normal_events, gross9_market, masks
    )
    stress_arrays = portfolio.split_arrays(
        stress_events, gross9_market, masks
    )
    for split in SELECTION_SPLITS:
        normal_entries = normal_arrays[split]["entry_positions"][CANDIDATE]
        stress_entries = stress_arrays[split]["entry_positions"][CANDIDATE]
        if not np.array_equal(normal_entries, stress_entries):
            raise RuntimeError("candidate stress replay changed entries")
    jaccards = entry_jaccards(normal_arrays, SELECTION_SPLITS)
    requirements = preregistration["selection_contract"]["requirements"]
    rows = _rank_rows(
        [
            portfolio_cell(
                weight=weight,
                normal_arrays=normal_arrays,
                stress_arrays=stress_arrays,
                baseline_arrays=base_arrays,
                splits=SELECTION_SPLITS,
                requirements=requirements,
                jaccard=float(jaccards["max"]),
            )
            for weight in WEIGHTS
        ]
    )
    passing = [row for row in rows if row["passes"]]
    top1 = passing[0] if passing else None
    statistics: dict[str, Any] = {}
    if top1 is not None:
        combined_weights, comparator_weights = same_gross_weights(
            top1["candidate_weight"]
        )
        for split in SELECTION_SPLITS:
            statistics[split] = paired_statistics(
                paired_weekly_effects(
                    normal_arrays[split],
                    combined_weights,
                    comparator_weights,
                )
            )
    payload = {
        "as_of": AS_OF,
        "phase": "selection",
        "preregistration_sha256": sha256_file(cfg.preregistration),
        "input_identity": input_identity,
        "candidate_freeze_hash": candidate["freeze_hash"],
        "authoritative_gross9": authoritative,
        "source_meta": source_meta,
        "candidate_normal_meta": normal_meta,
        "candidate_stress_meta": stress_meta,
        "entry_jaccards": jaccards,
        "paired_weekly_statistics_report_only": statistics,
        "tested_cells": len(rows),
        "passed_cells": len(passing),
        "rows": rows,
        "frozen_top1": top1,
        "decision": (
            "freeze_top1_for_future_veto"
            if top1 is not None
            else "reject_no_practical_same_gross_improvement"
        ),
        "future_opened": False,
        "future_can_rerank": False,
        "future_can_repair": False,
        "contamination_caveat": preregistration[
            "candidate_disclosure"
        ],
    }
    return finalize_payload(payload)


def verify_selection_artifact(
    selection: Mapping[str, Any],
    preregistration_sha256: str,
) -> dict[str, Any]:
    verify_result_hash(selection)
    if selection.get("phase") != "selection":
        raise RuntimeError("unexpected selection phase")
    if selection.get("preregistration_sha256") != preregistration_sha256:
        raise RuntimeError("selection preregistration hash drifted")
    if selection.get("future_opened") is not False:
        raise RuntimeError("selection artifact exposed future")
    if selection.get("future_can_rerank") is not False:
        raise RuntimeError("selection artifact enabled future reranking")
    top = selection.get("frozen_top1")
    if not isinstance(top, dict) or top.get("passes") is not True:
        raise RuntimeError("selection did not freeze a passing top1")
    if float(top["candidate_weight"]) not in WEIGHTS:
        raise RuntimeError("selection froze an invalid weight")
    return top


def _eval_combined_row(
    *,
    weight: float,
    normal_arrays: Mapping[str, Mapping[str, Any]],
    stress_arrays: Mapping[str, Mapping[str, Any]],
    base_arrays: Mapping[str, Mapping[str, Any]],
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    combined_weights, comparator_weights = same_gross_weights(weight)
    normal_combined = concat_array_data(
        [normal_arrays[split] for split in EVAL_SPLITS]
    )
    stress_combined = concat_array_data(
        [stress_arrays[split] for split in EVAL_SPLITS]
    )
    base_combined = concat_array_data(
        [base_arrays[split] for split in EVAL_SPLITS]
    )
    years = _calendar_years("2025-01-01", "2026-06-03")
    combined = portfolio.strict_metric(
        normal_combined, years, combined_weights
    )
    comparator = portfolio.strict_metric(
        base_combined, years, comparator_weights
    )
    baseline = portfolio.strict_metric(
        base_combined, years, BASELINE_WEIGHTS
    )
    stress = portfolio.strict_metric(
        stress_combined, years, combined_weights
    )
    standalone = portfolio.strict_metric(
        normal_combined, years, {CANDIDATE: 1.0}
    )
    standalone_stress = portfolio.strict_metric(
        stress_combined, years, {CANDIDATE: 1.0}
    )
    statistics = paired_statistics(
        paired_weekly_effects(
            normal_combined, combined_weights, comparator_weights
        )
    )
    req = preregistration["future_veto_contract"][
        "combined_requirements"
    ]
    checks = {
        "standalone_positive": standalone["absolute_return_pct"] > 0.0,
        "standalone_stress_positive": (
            standalone_stress["absolute_return_pct"] > 0.0
        ),
        "minimum_candidate_trades": (
            standalone["trades"] >= int(req["minimum_candidate_trades"])
        ),
        "portfolio_mdd": (
            combined["strict_mdd_pct"]
            <= float(req["maximum_portfolio_mdd_pct"])
        ),
        "return_retention": (
            combined["absolute_return_pct"]
            / baseline["absolute_return_pct"]
            >= float(req["minimum_return_retention"])
        ),
        "same_gross_improvement": (
            combined["cagr_to_strict_mdd"]
            - comparator["cagr_to_strict_mdd"]
            >= float(req["minimum_same_gross_ratio_improvement"])
        ),
        "stress_same_gross_improvement": (
            stress["cagr_to_strict_mdd"]
            - comparator["cagr_to_strict_mdd"]
            >= float(
                req["minimum_stress_same_gross_ratio_improvement"]
            )
        ),
        "mdd_reduction": (
            baseline["strict_mdd_pct"] - combined["strict_mdd_pct"]
            >= float(req["minimum_strict_mdd_reduction_pct"])
        ),
        "active_weeks": (
            statistics["active_weeks"]
            >= int(req["minimum_active_weeks"])
        ),
        "sign_flip": (
            statistics["sign_flip_pvalue"]
            <= float(req["maximum_sign_flip_pvalue"])
        ),
        "bootstrap_lower_mean": (
            statistics["bootstrap_90pct_lower_mean"] > 0.0
        ),
    }
    return {
        "window": "combined_2025_2026h1",
        "start": "2025-01-01",
        "end_exclusive": "2026-06-03",
        "baseline": baseline,
        "combined": combined,
        "same_gross_comparator": comparator,
        "candidate_stress_combined": stress,
        "standalone": standalone,
        "standalone_stress": standalone_stress,
        "same_gross_ratio_improvement": float(
            combined["cagr_to_strict_mdd"]
            - comparator["cagr_to_strict_mdd"]
        ),
        "stress_same_gross_ratio_improvement": float(
            stress["cagr_to_strict_mdd"]
            - comparator["cagr_to_strict_mdd"]
        ),
        "strict_mdd_reduction": float(
            baseline["strict_mdd_pct"] - combined["strict_mdd_pct"]
        ),
        "statistics": statistics,
        "checks": checks,
        "passes": bool(all(checks.values())),
    }


def eval_payload(cfg: Config) -> dict[str, Any]:
    preregistration = load_preregistration(cfg.preregistration)
    input_identity = validate_inputs(cfg, preregistration)
    selection = json.loads(
        Path(cfg.selection_output).read_text(encoding="utf-8")
    )
    top = verify_selection_artifact(
        selection, sha256_file(cfg.preregistration)
    )
    reproduced = selection_payload(cfg)
    if reproduced["result_hash"] != selection["result_hash"]:
        raise RuntimeError("selection replay changed before future opening")

    install_candidate_sleeve()
    gross9_market, masks, base_events, source_meta = build_full_context(cfg)
    candidate = prepare_candidate(
        cfg, cutoff="2026-06-02", validate_oos_hashes=True
    )
    validate_shared_market(candidate["market"], gross9_market)
    base_arrays = portfolio.split_arrays(base_events, gross9_market, masks)
    authoritative = validate_authoritative_gross9(
        cfg, base_arrays, EVAL_SPLITS
    )
    normal_events = list(base_events)
    stress_events = list(base_events)
    normal_meta = append_candidate_events(
        normal_events,
        gross9_market,
        candidate,
        EVAL_SPLITS,
        stress=False,
    )
    stress_meta = append_candidate_events(
        stress_events,
        gross9_market,
        candidate,
        EVAL_SPLITS,
        stress=True,
    )
    normal_arrays = portfolio.split_arrays(
        normal_events, gross9_market, masks
    )
    stress_arrays = portfolio.split_arrays(
        stress_events, gross9_market, masks
    )
    for split in EVAL_SPLITS:
        if not np.array_equal(
            normal_arrays[split]["entry_positions"][CANDIDATE],
            stress_arrays[split]["entry_positions"][CANDIDATE],
        ):
            raise RuntimeError("future stress replay changed entries")

    requirements = preregistration["future_veto_contract"][
        "per_window_requirements"
    ]
    row = portfolio_cell(
        weight=float(top["candidate_weight"]),
        normal_arrays=normal_arrays,
        stress_arrays=stress_arrays,
        baseline_arrays=base_arrays,
        splits=EVAL_SPLITS,
        requirements={
            "per_window": requirements,
            "maximum_entry_jaccard": 1.0,
        },
        jaccard=0.0,
    )
    combined = _eval_combined_row(
        weight=float(top["candidate_weight"]),
        normal_arrays=normal_arrays,
        stress_arrays=stress_arrays,
        base_arrays=base_arrays,
        preregistration=preregistration,
    )
    window_checks = {
        key: value
        for key, value in row["checks"].items()
        if key != "entry_jaccard"
    }
    passes = bool(all(window_checks.values()) and combined["passes"])
    payload = {
        "as_of": AS_OF,
        "phase": "future_veto",
        "preregistration_sha256": sha256_file(cfg.preregistration),
        "selection_result_hash": selection["result_hash"],
        "input_identity": input_identity,
        "candidate_freeze_hash": candidate["freeze_hash"],
        "authoritative_gross9": authoritative,
        "source_meta": source_meta,
        "candidate_normal_meta": normal_meta,
        "candidate_stress_meta": stress_meta,
        "frozen_candidate_weight": float(top["candidate_weight"]),
        "window_result": row,
        "window_checks_without_selection_jaccard": window_checks,
        "combined_result": combined,
        "passes": passes,
        "decision": (
            "promote_pposm_as_gross9_marginal_candidate"
            if passes
            else "terminal_veto_fixed_pposm_marginal"
        ),
        "reranked": False,
        "repaired": False,
        "rank2_opened": False,
        "contamination_caveat": preregistration[
            "candidate_disclosure"
        ],
    }
    return finalize_payload(payload)


def _fmt(metric: Mapping[str, Any]) -> str:
    return (
        f"{metric['absolute_return_pct']:.2f}% / "
        f"{metric['cagr_pct']:.2f}% / "
        f"{metric['strict_mdd_pct']:.2f}% / "
        f"{metric['cagr_to_strict_mdd']:.2f} / "
        f"{metric['trades']}"
    )


def render_selection(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Gross9 + frozen PPOSM same-gross marginal selection",
        "",
        "Metric: `absolute return / full-calendar CAGR / strict MDD / CAGR-MDD / trades`.",
        "",
        f"- decision: **{payload['decision']}**",
        f"- tested/pass: `{payload['tested_cells']}/{payload['passed_cells']}`",
        "- research boundary: the standalone candidate future was already exposed; only the exact Gross9 portfolio interaction was previously unmeasured.",
        "",
        "| weight | pass | train combined | 2024 combined | train Δ ratio | 2024 Δ ratio |",
        "|---:|:---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        train = row["stats"]["train"]
        test = row["stats"]["test2024"]
        lines.append(
            f"| {row['candidate_weight']:.2f} | "
            f"{'Y' if row['passes'] else 'N'} | "
            f"{_fmt(train['combined'])} | {_fmt(test['combined'])} | "
            f"{train['same_gross_ratio_improvement']:+.3f} | "
            f"{test['same_gross_ratio_improvement']:+.3f} |"
        )
    lines += [
        "",
        "## Integrity",
        "",
        f"- candidate freeze: `{payload['candidate_freeze_hash']}`",
        f"- max exact-entry Jaccard: `{payload['entry_jaccards']['max']:.4f}`",
        "- Weight selection saw only train and 2024 arrays.",
        "- 2025/2026 cannot rerank or repair the frozen top1.",
    ]
    return "\n".join(lines) + "\n"


def render_eval(payload: Mapping[str, Any]) -> str:
    row = payload["window_result"]
    combined = payload["combined_result"]
    lines = [
        "# Gross9 + frozen PPOSM future veto",
        "",
        "Metric: `absolute return / full-calendar CAGR / strict MDD / CAGR-MDD / trades`.",
        "",
        f"- decision: **{payload['decision']}**",
        f"- frozen weight: `{payload['frozen_candidate_weight']:.2f}`",
        "- reranked/repaired: `false/false`",
        "",
        "| window | combined | same-gross comparator | ratio Δ | MDD reduction |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in EVAL_SPLITS:
        stats = row["stats"][split]
        lines.append(
            f"| {split} | {_fmt(stats['combined'])} | "
            f"{_fmt(stats['same_gross_comparator'])} | "
            f"{stats['same_gross_ratio_improvement']:+.3f} | "
            f"{stats['strict_mdd_reduction']:+.3f}%p |"
        )
    lines += [
        (
            f"| combined 2025–2026H1 | {_fmt(combined['combined'])} | "
            f"{_fmt(combined['same_gross_comparator'])} | "
            f"{combined['same_gross_ratio_improvement']:+.3f} | "
            f"{combined['strict_mdd_reduction']:+.3f}%p |"
        ),
        "",
        "## Paired weekly evidence",
        "",
        f"- active weeks: `{combined['statistics']['active_weeks']}`",
        f"- sign-flip p: `{combined['statistics']['sign_flip_pvalue']:.4f}`",
        f"- bootstrap 90% lower mean log-effect: `{combined['statistics']['bootstrap_90pct_lower_mean']:.8f}`",
        "",
        "## Boundary",
        "",
        "- Standalone 2024–2026 results were known before this audit; this is a contamination-aware portfolio marginal audit, not pristine discovery OOS.",
        "- The PPOSM signal, state thresholds, skip rule, exits, leverage, costs, and schedules were replayed from the frozen artifacts without tuning.",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(
    payload: Mapping[str, Any], result_path: str | Path, docs_path: str | Path
) -> None:
    atomic_json(result_path, payload)
    destination = Path(docs_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    renderer = render_selection if payload["phase"] == "selection" else render_eval
    destination.write_text(renderer(payload), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("selection", "eval"))
    parser.add_argument("--preregistration", default=Config.preregistration)
    parser.add_argument("--selection-output", default=Config.selection_output)
    parser.add_argument("--eval-output", default=Config.eval_output)
    parser.add_argument("--selection-docs", default=Config.selection_docs)
    parser.add_argument("--eval-docs", default=Config.eval_docs)
    args = parser.parse_args(argv)
    cfg = Config(
        preregistration=args.preregistration,
        selection_output=args.selection_output,
        eval_output=args.eval_output,
        selection_docs=args.selection_docs,
        eval_docs=args.eval_docs,
    )
    payload = selection_payload(cfg) if args.phase == "selection" else eval_payload(cfg)
    if args.phase == "selection":
        write_outputs(payload, cfg.selection_output, cfg.selection_docs)
    else:
        write_outputs(payload, cfg.eval_output, cfg.eval_docs)
    print(canonical_json(payload).decode("utf-8"))


if __name__ == "__main__":
    main()
