"""Audit the frozen OI-pullback long as a missed Gross9 marginal sleeve.

The signal, thresholds, side, hold, and stride are read from the committed
paper-candidate config without modification.  Weight selection sees only train
and 2024 arrays.  Future windows require a separate, frozen-top1 phase.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import training.portfolio_opt_added_alpha_update as portfolio
import training.portfolio_opt_all_discovered_alpha_gross10 as legacy_all
import training.portfolio_opt_combined_rex_new_alpha as legacy_base
from training.audit_fresh_kimchi_orthogonal_alpha import (
    CANDIDATE_SPEC,
    Config as FreshAuditConfig,
    build_candidate_context,
    build_rank7_context,
    candidate_schedule,
    rank7_schedule,
)
from training.audit_rank7_fresh_kimchi_fixed_portfolio import (
    subaccount_bar_path,
)
from training.audit_weak_feature_responsibility_stability import (
    _action_spec as rank7_action_spec,
)
from training.evaluate_expanding_extratrees_top10_oos import FULL_CUTOFF
from training.evaluate_oi_llm_selector import (
    _feature_frame as oi_feature_frame,
)


PREREGISTRATION = Path(
    "results/gross9_oi_pullback_marginal_preregistration_2026-07-28.json"
)
OUTPUT = Path(
    "results/gross9_oi_pullback_pre2025_marginal_2026-07-28.json"
)
DOCS = Path(
    "docs/gross9-oi-pullback-pre2025-marginal-2026-07-28.md"
)
CANDIDATE = "oi_divergence_pullback"
SELECTION_SPLITS = ("train", "test2024")
OI_CACHE = Path("/tmp/btcusdt_open_interest_5m_2020_2026.csv")
EXPECTED_PREREGISTRATION_SHA256 = (
    "82a464c62a1652b5075f90664fd8d7cde9bfe34f737553c3fd590412ac29c4b8"
)


@dataclass(frozen=True)
class Config:
    preregistration: str = str(PREREGISTRATION)
    output: str = str(OUTPUT)
    docs_output: str = str(DOCS)
    candidate_config: str = (
        "configs/live/oi_divergence_pullback_range_rsi_h96_s6_candidate.json"
    )
    gross9_config: str = (
        "configs/shadow/portfolio_rank7_capacity_candidate_2026-07-28.json"
    )
    gross9_result: str = (
        "results/portfolio_rank7_capacity_update_2026-07-28.json"
    )
    market_csv: str = portfolio.Config.input_csv
    market_with_oi_csv: str = (
        "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz"
    )
    funding_csv: str = portfolio.Config.funding_csv
    premium_csv: str = portfolio.Config.premium_csv
    rank7_capacity_evidence: str = (
        "results/expanding_extratrees_rank7_leverage_battery_2026-07-27.json"
    )
    cost_rate: float = 0.0006


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_preregistration(path: str | Path) -> dict[str, Any]:
    actual_hash = _sha256(path)
    if actual_hash != EXPECTED_PREREGISTRATION_SHA256:
        raise RuntimeError(
            "preregistration hash drifted: "
            f"{actual_hash} != {EXPECTED_PREREGISTRATION_SHA256}"
        )
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("name") != "gross9_fixed_oi_divergence_pullback_marginal":
        raise RuntimeError("unexpected preregistration")
    if payload.get("status") != "preregistered_before_gross9_marginal_scan":
        raise RuntimeError("marginal audit was not preregistered")
    selection = payload["selection_contract"]
    if selection["candidate_weight_grid"] != [0.25, 0.5, 0.75, 1.0]:
        raise RuntimeError("candidate weight grid drifted")
    if selection.get("gross_cap") != 10.0:
        raise RuntimeError("gross cap drifted")
    if selection.get("top1_only") is not True:
        raise RuntimeError("top1-only contract drifted")
    if selection.get("selection_windows") != [
        "train through 2023",
        "test2024",
    ]:
        raise RuntimeError("selection windows drifted")
    if selection.get("future_windows_forbidden_during_ranking") != [
        "eval2025",
        "ytd2026",
        "July 2026",
    ]:
        raise RuntimeError("future ranking boundary drifted")
    future = payload["future_veto_contract"]
    if future.get("future_can_rerank") is not False:
        raise RuntimeError("future reranking is not disabled")
    if future.get("future_can_repair") is not False:
        raise RuntimeError("future repair is not disabled")
    return payload


def materialize_frozen_oi_cache(path: str | Path) -> dict[str, Any]:
    """Make the implicit Gross9 OI input identical to the frozen source file."""
    source = Path(path)
    frame = pd.read_csv(source, usecols=["date", "open_interest"])
    dates = pd.to_datetime(frame["date"], utc=True, errors="raise").dt.tz_convert(
        None
    )
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise RuntimeError("frozen OI source is not unique and monotonic")
    values = pd.to_numeric(frame["open_interest"], errors="coerce")
    if values.notna().sum() == 0:
        raise RuntimeError("frozen OI source contains no finite observations")
    normalized = pd.DataFrame({"date": dates, "open_interest": values})
    temporary = OI_CACHE.with_name(OI_CACHE.name + ".tmp")
    normalized.to_csv(temporary, index=False)
    temporary.replace(OI_CACHE)
    return {
        "path": str(OI_CACHE),
        "sha256": _sha256(OI_CACHE),
        "rows": int(len(normalized)),
        "first": str(dates.iloc[0]),
        "last": str(dates.iloc[-1]),
        "finite_open_interest": int(values.notna().sum()),
        "materialized_from_frozen_source": str(source),
    }


def validate_inputs(
    cfg: Config, preregistration: dict[str, Any]
) -> dict[str, Any]:
    configured = {
        "market": cfg.market_csv,
        "market_with_oi": cfg.market_with_oi_csv,
        "funding": cfg.funding_csv,
        "premium": cfg.premium_csv,
        "candidate_config": cfg.candidate_config,
        "gross9_config": cfg.gross9_config,
        "gross9_result": cfg.gross9_result,
    }
    records: dict[str, Any] = {}
    for name, path in configured.items():
        actual = _sha256(path)
        expected = str(preregistration["input_provenance"][name]["sha256"])
        if actual != expected:
            raise RuntimeError(
                f"input provenance mismatch for {name}: {actual} != {expected}"
            )
        records[name] = {
            "path": str(path),
            "sha256": actual,
            "validated_against_preregistration": True,
        }
    return records


def _install_candidate_sleeve() -> None:
    if CANDIDATE not in portfolio.SLEEVES:
        portfolio.SLEEVES = (*portfolio.SLEEVES, CANDIDATE)
    portfolio.FAMILIES[CANDIDATE] = "oi_divergence"


def _gate_mask(
    features: pd.DataFrame, gates: list[dict[str, Any]]
) -> np.ndarray:
    active = np.ones(len(features), dtype=bool)
    for gate in gates:
        values = pd.to_numeric(
            features[str(gate["feature"])], errors="coerce"
        ).to_numpy(float)
        finite = np.isfinite(values)
        op = str(gate["op"])
        threshold = float(gate["threshold"])
        if op in {">=", "ge"}:
            active &= finite & (values >= threshold)
        elif op in {"<=", "le"}:
            active &= finite & (values <= threshold)
        else:
            raise ValueError(f"unsupported candidate gate: {op}")
    return active


def validate_candidate_feature_contract(
    features: pd.DataFrame,
    gates: list[dict[str, Any]],
) -> None:
    missing = sorted(
        {
            str(gate["feature"])
            for gate in gates
            if str(gate["feature"]) not in features.columns
        }
    )
    if missing:
        raise RuntimeError(
            f"candidate features missing from shared market frame: {missing}"
        )


def build_candidate_feature_frame(market: pd.DataFrame) -> pd.DataFrame:
    """Preserve Gross9 features and add the frozen OI-divergence definitions."""
    shared = portfolio.feature_frame(market)
    exact_oi = oi_feature_frame(market, window_size=144)
    additions = exact_oi.loc[
        :, [column for column in exact_oi.columns if column not in shared.columns]
    ]
    return pd.concat([shared, additions], axis=1)


def _append_rank7_and_fresh_selection(
    events: list[dict[str, Any]],
    cfg: Config,
) -> dict[str, dict[str, int]]:
    audit_cfg = FreshAuditConfig(
        input_csv=str(portfolio.resolve_existing(cfg.market_csv)),
        funding_csv=str(portfolio.resolve_existing(cfg.funding_csv)),
        premium_csv=str(portfolio.resolve_existing(cfg.premium_csv)),
        output="/tmp/no_write_gross9_oi_pullback.json",
        docs_output="",
        exclude_from=FULL_CUTOFF,
    )
    fresh = build_candidate_context(audit_cfg)
    rank7 = build_rank7_context(audit_cfg)
    market = fresh["market"]
    rank7_market = rank7["base"]["context"]["market"]
    if not np.array_equal(
        pd.to_datetime(market["date"]),
        pd.to_datetime(rank7_market["date"]),
    ):
        raise RuntimeError("rank7/fresh market grids differ")
    funding_leg = np.asarray(
        rank7["base"]["context"]["funding_leg"], dtype=bool
    )
    counts = {"frozen_annual_rank7": {}, "fresh_kimchi_fx": {}}
    for split in SELECTION_SPLITS:
        start, end = portfolio.SPLIT_BOUNDS[split]
        fresh_trades = candidate_schedule(fresh, start=start, end=end)
        rank7_trades = rank7_schedule(rank7, start=start, end=end)
        fresh_path = subaccount_bar_path(
            market,
            fresh["funding"],
            fresh_trades,
            fresh["execution_cfg"],
            start=start,
            end=end,
            hold_bars=lambda _trade: int(CANDIDATE_SPEC["hold_bars"]),
        )
        rank7_path = subaccount_bar_path(
            rank7_market,
            rank7["base"]["context"]["funding"],
            rank7_trades,
            rank7["base"]["execution_cfg"],
            start=start,
            end=end,
            hold_bars=lambda trade: int(
                rank7_action_spec(bool(funding_leg[trade.signal_position]))[0]
            ),
        )
        events.append(
            portfolio.path_event(
                market,
                rank7_path,
                split=split,
                sleeve="frozen_annual_rank7",
                trades=rank7_trades,
            )
        )
        events.append(
            portfolio.path_event(
                market,
                fresh_path,
                split=split,
                sleeve="fresh_kimchi_fx",
                trades=fresh_trades,
            )
        )
        counts["frozen_annual_rank7"][split] = len(rank7_trades)
        counts["fresh_kimchi_fx"][split] = len(fresh_trades)
    return counts


def build_selection_arrays(
    cfg: Config, preregistration: dict[str, Any]
) -> tuple[
    pd.DataFrame,
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    _install_candidate_sleeve()
    oi_cache = materialize_frozen_oi_cache(cfg.market_with_oi_csv)
    portfolio.ensure_runtime_inputs()
    capacity_cfg = portfolio.Config(
        input_csv=cfg.market_csv,
        funding_csv=cfg.funding_csv,
        premium_csv=cfg.premium_csv,
        rank7_family_gross_cap=3.0,
        rank7_capacity_evidence=cfg.rank7_capacity_evidence,
        cost_rate=cfg.cost_rate,
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
        cost_rate=cfg.cost_rate,
    )
    market, _, all_masks, _, events, _ = legacy_base.build_combined_events(
        legacy_cfg
    )
    masks = {split: all_masks[split] for split in SELECTION_SPLITS}
    legacy_all.add_rex_veto_candidates(events, market, masks, legacy_cfg)
    events = [
        event
        for event in events
        if event["split"] in SELECTION_SPLITS
        and event["sleeve"] in portfolio.LIVE_WEIGHTS
    ]
    portfolio.attach_live_rex_ohlc(events, market, masks, capacity_cfg)
    portfolio.attach_default_favorable(events, market)
    features = build_candidate_feature_frame(market)
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
        cost_rate=cfg.cost_rate,
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
        cost_rate=cfg.cost_rate,
    )
    rex_counts = portfolio.append_rex_taker_policy(
        events, market, masks, cost_rate=cfg.cost_rate
    )
    path_counts = _append_rank7_and_fresh_selection(events, cfg)
    candidate_config = json.loads(
        Path(cfg.candidate_config).read_text(encoding="utf-8")
    )
    signal = candidate_config["signal"]
    if (
        str(signal["side"]).lower() != "long"
        or float(signal["leverage"]) != 1.0
        or int(signal["hold_bars_5m"])
        != int(preregistration["candidate_contract"]["hold_bars"])
        or int(signal["stride_bars_5m"])
        != int(preregistration["candidate_contract"]["stride_bars"])
        or int(signal["entry_delay_bars"])
        != int(preregistration["candidate_contract"]["entry_delay_bars"])
        or float(preregistration["candidate_contract"]["portfolio_unit_leverage"])
        != 0.5
        or float(preregistration["candidate_contract"]["cost_rate_each_side"])
        != float(cfg.cost_rate)
    ):
        raise RuntimeError("candidate execution contract drifted")
    validate_candidate_feature_contract(features, signal["gates"])
    candidate_active = _gate_mask(features, signal["gates"])
    candidate_counts = portfolio.append_mask_policy(
        events,
        market,
        masks,
        name=CANDIDATE,
        long_active=candidate_active,
        short_active=np.zeros(len(market), dtype=bool),
        hold=int(signal["hold_bars_5m"]),
        stride=int(signal["stride_bars_5m"]),
        cost_rate=cfg.cost_rate,
    )
    arrays = portfolio.split_arrays(events, market, masks)
    meta = {
        "rank7_capacity": capacity,
        "counts": {
            "markov_transition_long": markov_counts,
            "funding_premium_lr_impact_central": funding_counts,
            "rex_taker_low_range_position": rex_counts,
            **path_counts,
            CANDIDATE: candidate_counts,
        },
        "funding_meta": funding_meta,
        "oi_cache": oi_cache,
        "feature_columns": int(features.shape[1]),
        "candidate_raw_signal_count": {
            split: int(
                np.sum(
                    candidate_active
                    & np.asarray(masks[split], dtype=bool)
                )
            )
            for split in SELECTION_SPLITS
        },
    }
    return market, arrays, meta


def _baseline_weights(preregistration: dict[str, Any]) -> dict[str, float]:
    return {
        str(name): float(weight)
        for name, weight in preregistration["portfolio_baseline"][
            "weights"
        ].items()
    }


def _metric(
    arrays: dict[str, dict[str, Any]],
    split: str,
    weights: dict[str, float],
) -> dict[str, Any]:
    return portfolio.strict_metric(
        arrays[split],
        portfolio.years_for(split),
        weights,
    )


def validate_baseline(
    arrays: dict[str, dict[str, Any]],
    preregistration: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    weights = _baseline_weights(preregistration)
    actual = {
        split: _metric(arrays, split, weights)
        for split in SELECTION_SPLITS
    }
    expected = preregistration["portfolio_baseline"]["frozen_stats"]
    for split in SELECTION_SPLITS:
        for key in (
            "absolute_return_pct",
            "cagr_pct",
            "strict_mdd_pct",
            "cagr_to_strict_mdd",
        ):
            if not np.isclose(
                float(actual[split][key]),
                float(expected[split][key]),
                rtol=0.0,
                atol=1e-9,
            ):
                raise RuntimeError(
                    f"Gross9 baseline drifted in {split}/{key}: "
                    f"{actual[split][key]} != {expected[split][key]}"
                )
        if int(actual[split]["trades"]) != int(expected[split]["trades"]):
            raise RuntimeError(f"Gross9 trade count drifted in {split}")
    return actual


def validate_authoritative_gross9(
    cfg: Config,
    preregistration: dict[str, Any],
) -> dict[str, Any]:
    candidate = json.loads(Path(cfg.gross9_config).read_text(encoding="utf-8"))
    result = json.loads(Path(cfg.gross9_result).read_text(encoding="utf-8"))
    weights = _baseline_weights(preregistration)
    if {
        str(name): float(value)
        for name, value in candidate.get("weights", {}).items()
    } != weights:
        raise RuntimeError("Gross9 shadow-config weights drifted")
    if not np.isclose(float(candidate.get("gross_weight", -1.0)), 9.0):
        raise RuntimeError("Gross9 shadow-config gross drifted")
    if candidate.get("accounting_version") != portfolio.ACCOUNTING_VERSION:
        raise RuntimeError("Gross9 shadow-config accounting drifted")
    if result.get("accounting_version") != portfolio.ACCOUNTING_VERSION:
        raise RuntimeError("Gross9 source-result accounting drifted")
    if result.get("mode") != "frozen_pre2025_allocation_rank_future_veto_only":
        raise RuntimeError("Gross9 source-result selection mode drifted")
    if result.get("future_used_for_allocation_ranking") is not False:
        raise RuntimeError("Gross9 source result used future allocation ranking")
    if result.get("future_can_only_veto_frozen_rank1") is not True:
        raise RuntimeError("Gross9 source result did not freeze pre-2025 rank1")
    top1 = result.get("frozen_pre2025_top1", {})
    if {
        str(name): float(value)
        for name, value in top1.get("weights", {}).items()
    } != weights:
        raise RuntimeError("Gross9 authoritative top1 weights drifted")
    if not np.isclose(float(top1.get("gross", -1.0)), 9.0):
        raise RuntimeError("Gross9 authoritative top1 gross drifted")
    expected = preregistration["portfolio_baseline"]["frozen_stats"]
    for split in (*SELECTION_SPLITS, "eval2025", "ytd2026"):
        for key in (
            "absolute_return_pct",
            "cagr_pct",
            "strict_mdd_pct",
            "cagr_to_strict_mdd",
        ):
            if not np.isclose(
                float(top1["stats"][split][key]),
                float(expected[split][key]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(
                    f"Gross9 authoritative result drifted in {split}/{key}"
                )
    return {
        "config_name": candidate["name"],
        "config_accounting_version": candidate["accounting_version"],
        "source_mode": result["mode"],
        "source_future_used_for_allocation_ranking": False,
        "source_future_can_only_veto_frozen_rank1": True,
        "frozen_top1_weights": weights,
    }


def _occupied(data: dict[str, Any], sleeve: str) -> np.ndarray:
    index = portfolio.SLEEVES.index(sleeve)
    return (
        np.abs(data["L"][index]) > 1e-15
    ) | (np.abs(data["H"][index]) > 1e-15)


def _entries(occupied: np.ndarray) -> np.ndarray:
    previous = np.r_[False, occupied[:-1]]
    return np.flatnonzero(occupied & ~previous)


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    a = set(map(int, np.asarray(left, dtype=np.int64)))
    b = set(map(int, np.asarray(right, dtype=np.int64)))
    union = a | b
    return float(len(a & b) / len(union)) if union else 0.0


def _daily_returns(
    values: np.ndarray, dates: pd.DatetimeIndex
) -> pd.Series:
    frame = pd.DataFrame(
        {"date": pd.DatetimeIndex(dates).normalize(), "return": values}
    )
    return frame.groupby("date")["return"].apply(
        lambda rows: float(np.prod(1.0 + rows.to_numpy(float)) - 1.0)
    )


def marginal_diagnostics(
    arrays: dict[str, dict[str, Any]],
    baseline_weights: dict[str, float],
) -> dict[str, Any]:
    vector = np.asarray(
        [baseline_weights.get(name, 0.0) for name in portfolio.SLEEVES],
        dtype=float,
    )
    candidate_index = portfolio.SLEEVES.index(CANDIDATE)
    result: dict[str, Any] = {}
    all_jaccards: list[float] = []
    for split in SELECTION_SPLITS:
        data = arrays[split]
        candidate_occupied = _occupied(data, CANDIDATE)
        candidate_entries = np.asarray(
            data["entry_positions"][CANDIDATE], dtype=np.int64
        )
        candidate_count = int(data["counts"][candidate_index])
        if len(candidate_entries) != candidate_count:
            raise RuntimeError(
                f"{split} candidate entry count drifted: "
                f"{len(candidate_entries)} != {candidate_count}"
            )
        baseline_occupied = np.zeros(len(candidate_occupied), dtype=bool)
        per_sleeve: dict[str, Any] = {}
        for sleeve, weight in baseline_weights.items():
            if weight <= 0.0:
                continue
            occupied = _occupied(data, sleeve)
            baseline_occupied |= occupied
            sleeve_index = portfolio.SLEEVES.index(sleeve)
            sleeve_entries = np.asarray(
                data["entry_positions"][sleeve], dtype=np.int64
            )
            sleeve_count = int(data["counts"][sleeve_index])
            if len(sleeve_entries) != sleeve_count:
                raise RuntimeError(
                    f"{split}/{sleeve} entry count drifted: "
                    f"{len(sleeve_entries)} != {sleeve_count}"
                )
            entry_jaccard = _jaccard(
                candidate_entries, sleeve_entries
            )
            occupied_overlap = float(
                np.sum(candidate_occupied & occupied)
                / max(1, np.sum(candidate_occupied | occupied))
            )
            all_jaccards.append(entry_jaccard)
            per_sleeve[sleeve] = {
                "entry_jaccard": entry_jaccard,
                "occupied_bar_jaccard": occupied_overlap,
                "candidate_entries": int(len(candidate_entries)),
                "sleeve_entries": int(len(sleeve_entries)),
            }
        baseline_returns = vector @ data["R"]
        candidate_returns = data["R"][candidate_index]
        baseline_daily = _daily_returns(
            baseline_returns, data["dates"]
        )
        candidate_daily = _daily_returns(
            candidate_returns, data["dates"]
        )
        aligned = pd.concat(
            [baseline_daily, candidate_daily], axis=1
        ).fillna(0.0)
        aligned.columns = ["baseline", "candidate"]
        raw_pearson = float(aligned.corr("pearson").iloc[0, 1])
        raw_spearman = float(aligned.corr("spearman").iloc[0, 1])
        pearson = raw_pearson if np.isfinite(raw_pearson) else None
        spearman = raw_spearman if np.isfinite(raw_spearman) else None
        flat_returns = np.where(~baseline_occupied, candidate_returns, 0.0)
        flat_absolute = (
            float(np.prod(np.maximum(0.0, 1.0 + flat_returns)) - 1.0)
            * 100.0
        )
        worst = np.argsort(baseline_returns)[:20]
        result[split] = {
            "per_sleeve": per_sleeve,
            "daily_mtm_correlation": {
                "pearson": pearson,
                "spearman": spearman,
                "undefined_reason": (
                    "zero_variance_daily_series"
                    if pearson is None or spearman is None
                    else None
                ),
            },
            "candidate_absolute_return_while_gross9_flat_pct": flat_absolute,
            "candidate_return_sum_on_worst20_baseline_bars_pct": float(
                np.sum(candidate_returns[worst]) * 100.0
            ),
            "candidate_occupied_fraction": float(
                np.mean(candidate_occupied)
            ),
            "gross9_occupied_fraction": float(
                np.mean(baseline_occupied)
            ),
        }
    result["max_entry_jaccard"] = max(all_jaccards, default=0.0)
    return result


def selection_row(
    *,
    weight: float,
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    control: dict[str, dict[str, Any]],
    standalone: dict[str, dict[str, Any]],
    max_entry_jaccard: float,
) -> dict[str, Any]:
    ratio_delta = {
        split: float(candidate[split]["cagr_to_strict_mdd"])
        - float(baseline[split]["cagr_to_strict_mdd"])
        for split in SELECTION_SPLITS
    }
    control_delta = {
        split: float(control[split]["cagr_to_strict_mdd"])
        - float(baseline[split]["cagr_to_strict_mdd"])
        for split in SELECTION_SPLITS
    }
    excess = {
        split: ratio_delta[split] - control_delta[split]
        for split in SELECTION_SPLITS
    }
    passes = bool(
        candidate["train"]["strict_mdd_pct"] <= 40.0
        and candidate["test2024"]["strict_mdd_pct"] <= 20.0
        and all(
            candidate[split]["absolute_return_pct"]
            >= 0.97 * baseline[split]["absolute_return_pct"]
            for split in SELECTION_SPLITS
        )
        and all(ratio_delta[split] > 0.0 for split in SELECTION_SPLITS)
        and all(excess[split] > 0.0 for split in SELECTION_SPLITS)
        and any(
            candidate[split]["strict_mdd_pct"]
            < baseline[split]["strict_mdd_pct"]
            for split in SELECTION_SPLITS
        )
        and all(
            standalone[split]["absolute_return_pct"] > 0.0
            for split in SELECTION_SPLITS
        )
        and max_entry_jaccard <= 0.25
    )
    key = (
        min(excess.values()),
        math.sqrt(
            max(0.0, ratio_delta["train"])
            * max(0.0, ratio_delta["test2024"])
        ),
        baseline["test2024"]["strict_mdd_pct"]
        - candidate["test2024"]["strict_mdd_pct"],
        candidate["test2024"]["absolute_return_pct"]
        - baseline["test2024"]["absolute_return_pct"],
        -float(weight),
    )
    return {
        "candidate_weight": float(weight),
        "gross": 9.0 + float(weight),
        "passes": passes,
        "stats": candidate,
        "same_gross_control": control,
        "ratio_delta": ratio_delta,
        "control_ratio_delta": control_delta,
        "ratio_delta_over_control": excess,
        "selection_key": list(key),
    }


def _row_key(row: dict[str, Any]) -> tuple[float, ...]:
    return tuple(float(value) for value in row["selection_key"])


def _format(metric: dict[str, Any]) -> str:
    return (
        f"{metric['absolute_return_pct']:.2f}/"
        f"{metric['cagr_pct']:.2f}/"
        f"{metric['strict_mdd_pct']:.2f}/"
        f"{metric['cagr_to_strict_mdd']:.2f}/"
        f"{metric['trades']}"
    )


def _render(payload: dict[str, Any]) -> str:
    lines = [
        "# Gross9 + fixed OI-pullback pre-2025 marginal audit",
        "",
        "Metric: `absolute return / full-calendar CAGR / strict MDD / CAGR-MDD / trades`.",
        "",
        f"- evaluated weights: {payload['tested']}",
        f"- passers: {payload['passed']}",
        f"- decision: **{payload['decision']}**",
        "",
        "| weight | pass | train | 2024 | train ratio Δ / control Δ | 2024 ratio Δ / control Δ |",
        "|---:|:---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['candidate_weight']:.2f} | "
            f"{'Y' if row['passes'] else 'N'} | "
            f"{_format(row['stats']['train'])} | "
            f"{_format(row['stats']['test2024'])} | "
            f"{row['ratio_delta']['train']:+.3f} / "
            f"{row['control_ratio_delta']['train']:+.3f} | "
            f"{row['ratio_delta']['test2024']:+.3f} / "
            f"{row['control_ratio_delta']['test2024']:+.3f} |"
        )
    lines += [
        "",
        "## Fixed sleeve",
        "",
        f"- standalone train: `{_format(payload['standalone']['train'])}`",
        f"- standalone 2024: `{_format(payload['standalone']['test2024'])}`",
        f"- maximum entry Jaccard versus a Gross9 sleeve: `{payload['diagnostics']['max_entry_jaccard']:.4f}`",
        "",
        "## Boundary",
        "",
        "- The signal config, four thresholds, long side, hold, stride, and costs were not changed.",
        "- Only train and 2024 shared-clock arrays were passed to weight selection.",
        "- Same-gross control scales frozen Gross9 pro rata instead of adding the candidate.",
        "- 2025, 2026, and July results are absent from this artifact and cannot rerank it.",
    ]
    return "\n".join(lines) + "\n"


def _atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def run_pre2025(cfg: Config) -> dict[str, Any]:
    preregistration = load_preregistration(cfg.preregistration)
    input_identity = validate_inputs(cfg, preregistration)
    authoritative_gross9 = validate_authoritative_gross9(
        cfg, preregistration
    )
    market, arrays, source_meta = build_selection_arrays(
        cfg, preregistration
    )
    if set(arrays) != set(SELECTION_SPLITS):
        raise RuntimeError("future arrays entered pre-2025 selection")
    baseline_weights = _baseline_weights(preregistration)
    baseline = validate_baseline(arrays, preregistration)
    standalone_weights = {CANDIDATE: 1.0}
    standalone = {
        split: _metric(arrays, split, standalone_weights)
        for split in SELECTION_SPLITS
    }
    diagnostics = marginal_diagnostics(arrays, baseline_weights)
    rows: list[dict[str, Any]] = []
    for raw_weight in preregistration["selection_contract"][
        "candidate_weight_grid"
    ]:
        weight = float(raw_weight)
        candidate_weights = {**baseline_weights, CANDIDATE: weight}
        candidate = {
            split: _metric(arrays, split, candidate_weights)
            for split in SELECTION_SPLITS
        }
        scale = (9.0 + weight) / 9.0
        control_weights = {
            name: value * scale
            for name, value in baseline_weights.items()
        }
        control = {
            split: _metric(arrays, split, control_weights)
            for split in SELECTION_SPLITS
        }
        rows.append(
            selection_row(
                weight=weight,
                baseline=baseline,
                candidate=candidate,
                control=control,
                standalone=standalone,
                max_entry_jaccard=float(
                    diagnostics["max_entry_jaccard"]
                ),
            )
        )
    rows.sort(key=_row_key, reverse=True)
    passed = [row for row in rows if row["passes"]]
    frozen_top1 = passed[0] if passed else None
    freeze = {
        "preregistration_sha256": _sha256(cfg.preregistration),
        "candidate_config_sha256": _sha256(cfg.candidate_config),
        "gross9_config_sha256": _sha256(cfg.gross9_config),
        "selection_splits": list(SELECTION_SPLITS),
        "baseline_weights": baseline_weights,
        "candidate_schedule_counts": {
            split: int(
                arrays[split]["counts"][
                    portfolio.SLEEVES.index(CANDIDATE)
                ]
            )
            for split in SELECTION_SPLITS
        },
        "rows": rows,
        "frozen_top1": frozen_top1,
    }
    payload = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "phase": "pre2025_fixed_sleeve_marginal",
        "config": asdict(cfg),
        "input_identity": input_identity,
        "authoritative_gross9": authoritative_gross9,
        "source_meta": source_meta,
        "market": {
            "rows": int(len(market)),
            "first": str(pd.to_datetime(market["date"]).iloc[0]),
            "last": str(pd.to_datetime(market["date"]).iloc[-1]),
        },
        "preregistration": cfg.preregistration,
        "preregistration_sha256": freeze["preregistration_sha256"],
        "baseline": baseline,
        "standalone": standalone,
        "diagnostics": diagnostics,
        "tested": len(rows),
        "passed": len(passed),
        "rows": rows,
        "frozen_top1": frozen_top1,
        "decision": "open_future_veto" if frozen_top1 else "reject_marginal",
        "future_opened": False,
        "future_can_rerank": False,
        "freeze_hash": _json_hash(freeze),
    }
    _atomic_json(cfg.output, payload)
    docs = Path(cfg.docs_output)
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text(_render(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("pre2025",))
    parser.add_argument("--preregistration", default=Config.preregistration)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--docs-output", default=Config.docs_output)
    args = parser.parse_args()
    payload = run_pre2025(
        Config(
            preregistration=args.preregistration,
            output=args.output,
            docs_output=args.docs_output,
        )
    )
    print(
        json.dumps(
            {
                "phase": payload["phase"],
                "tested": payload["tested"],
                "passed": payload["passed"],
                "decision": payload["decision"],
                "future_opened": payload["future_opened"],
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
