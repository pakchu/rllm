"""Rank frozen Gross9 additions and state-family substitutions on pre-2025 only.

The candidate universe, grids, comparators, and acceptance rules are committed
in G9-FCSS-1.  This module deliberately exposes only the pre-2025 phase.
Future vetoes require a separate evaluator bound to this phase's freeze hash.
"""
from __future__ import annotations

import argparse
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
import training.state_model_top10_ensemble as state_top10
from training.audit_gross9_oi_pullback_marginal import (
    _append_rank7_and_fresh_selection,
    _atomic_json,
    _baseline_weights,
    _daily_returns,
    _gate_mask,
    _json_hash,
    _jaccard,
    _occupied,
    _sha256,
    build_candidate_feature_frame,
    materialize_frozen_oi_cache,
)
from training.evaluate_expanding_extratrees_top10_oos import FULL_CUTOFF
from training.state_model_top10_ensemble import (
    SLEEVE_NAMES,
    build_state_model_top10_ensembles,
)


PREREGISTRATION = Path(
    "results/"
    "gross9_fixed_candidate_state_substitution_preregistration_2026-07-28.json"
)
OUTPUT = Path(
    "results/"
    "gross9_fixed_candidate_state_substitution_pre2025_2026-07-28.json"
)
DOCS = Path(
    "docs/"
    "gross9-fixed-candidate-state-substitution-pre2025-2026-07-28.md"
)
EXPECTED_PREREGISTRATION_SHA256 = (
    "7d85fdf4f695781c5abc8f0a91ce5668158c2c9c5b9f7e1cd606987b74653d4e"
)
SELECTION_SPLITS = ("train", "test2024")
SELECTION_PROVENANCE_KEYS = (
    "market",
    "market_with_oi",
    "funding",
    "premium",
    "gross9_pre2025_anchor",
    "rank7_capacity_evidence",
    "nonpb30_config",
    "oi_highfreq_config",
    "kalman_scan",
    "bocpd_scan",
    "semimarkov_scan",
    "state_ensemble_builder",
)
FUTURE_ONLY_PROVENANCE_KEYS = (
    "gross9_config",
    "gross9_result",
    "prior_state_portfolio_result",
    "july_atomic_replay",
)
ADDITION_CANDIDATES = (
    "nonpb30_taker",
    "oi_divergence_highfreq",
)
STATE_CANDIDATES = tuple(SLEEVE_NAMES.values())
ALL_CANDIDATES = ADDITION_CANDIDATES + STATE_CANDIDATES


@dataclass(frozen=True)
class Config:
    preregistration: str = str(PREREGISTRATION)
    output: str = str(OUTPUT)
    docs_output: str = str(DOCS)
    market_csv: str = portfolio.Config.input_csv
    market_with_oi_csv: str = (
        "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz"
    )
    funding_csv: str = portfolio.Config.funding_csv
    premium_csv: str = portfolio.Config.premium_csv
    gross9_pre2025_anchor: str = (
        "results/gross9_pre2025_authoritative_anchor_2026-07-28.json"
    )
    nonpb30_config: str = (
        "configs/live/"
        "nonpb30_taker_returnz_rangevol_htf4hrange_h72_candidate.json"
    )
    oi_highfreq_config: str = (
        "configs/live/oi_divergence_sma24_highfreq_h30_s6_candidate.json"
    )
    rank7_capacity_evidence: str = (
        "results/expanding_extratrees_rank7_leverage_battery_2026-07-27.json"
    )
    cost_rate: float = 0.0006


def load_preregistration(path: str | Path) -> dict[str, Any]:
    actual = _sha256(path)
    if actual != EXPECTED_PREREGISTRATION_SHA256:
        raise RuntimeError(
            f"preregistration hash drifted: {actual} != "
            f"{EXPECTED_PREREGISTRATION_SHA256}"
        )
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("name") != (
        "gross9_fixed_candidate_and_state_substitution_battery"
    ):
        raise RuntimeError("unexpected preregistration")
    if payload.get("status") != (
        "preregistered_before_shared_clock_candidate_battery"
    ):
        raise RuntimeError("candidate battery was not preregistered")
    selection = payload["selection_contract"]
    if selection["addition_weight_grid"] != [0.25, 0.5, 0.75, 1.0]:
        raise RuntimeError("addition grid drifted")
    if selection["state_substitution_weight_grid"] != [
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
    ]:
        raise RuntimeError("state substitution grid drifted")
    if selection.get("top1_only") is not True:
        raise RuntimeError("top1-only contract drifted")
    if selection.get("selection_windows") != [
        "train through 2023",
        "test2024",
    ]:
        raise RuntimeError("selection windows drifted")
    future = payload["future_veto_contract"]
    if future.get("future_can_rerank") is not False:
        raise RuntimeError("future reranking is not disabled")
    if future.get("future_can_repair") is not False:
        raise RuntimeError("future repair is not disabled")
    return payload


def _configured_selection_inputs(cfg: Config) -> dict[str, str]:
    """Return the exact files the pre-2025 runner can read for selection."""
    return {
        "market": cfg.market_csv,
        "market_with_oi": cfg.market_with_oi_csv,
        "funding": cfg.funding_csv,
        "premium": cfg.premium_csv,
        "gross9_pre2025_anchor": cfg.gross9_pre2025_anchor,
        "rank7_capacity_evidence": cfg.rank7_capacity_evidence,
        "nonpb30_config": cfg.nonpb30_config,
        "oi_highfreq_config": cfg.oi_highfreq_config,
        "kalman_scan": str(state_top10.SCAN_PATHS["kalman"]),
        "bocpd_scan": str(state_top10.SCAN_PATHS["bocpd"]),
        "semimarkov_scan": str(state_top10.SCAN_PATHS["semimarkov"]),
        "state_ensemble_builder": str(Path(state_top10.__file__)),
    }


def validate_inputs(
    cfg: Config,
    preregistration: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    configured = _configured_selection_inputs(cfg)
    if tuple(configured) != SELECTION_PROVENANCE_KEYS:
        raise RuntimeError("selection input map drifted")
    provenance = preregistration["input_provenance"]
    if any(name not in provenance for name in SELECTION_PROVENANCE_KEYS):
        raise RuntimeError("preregistration is missing a selection input")
    if any(name not in provenance for name in FUTURE_ONLY_PROVENANCE_KEYS):
        raise RuntimeError("preregistration is missing future-veto provenance")
    records: dict[str, dict[str, Any]] = {}
    for name, configured_path in configured.items():
        record = provenance[name]
        preregistered_path = str(record["path"])
        configured_resolved = Path(configured_path).resolve(strict=True)
        preregistered_resolved = Path(preregistered_path).resolve(strict=True)
        if configured_resolved != preregistered_resolved:
            raise RuntimeError(
                f"configured path mismatch for {name}: "
                f"{configured_resolved} != {preregistered_resolved}"
            )
        actual = _sha256(configured_resolved)
        expected = str(record["sha256"])
        if actual != expected:
            raise RuntimeError(
                f"input provenance mismatch for {name}: "
                f"{actual} != {expected}"
            )
        records[name] = {
            "configured_path": str(configured_path),
            "preregistered_path": preregistered_path,
            "resolved_path": str(configured_resolved),
            "sha256": actual,
            "validated_against_preregistration": True,
        }
    return records


def validate_pre2025_gross9_anchor(
    cfg: Config,
    preregistration: dict[str, Any],
) -> dict[str, Any]:
    """Validate Gross9 authority without opening any future-bearing artifact."""
    anchor = json.loads(
        Path(cfg.gross9_pre2025_anchor).read_text(encoding="utf-8")
    )
    if anchor.get("name") != "gross9_pre2025_authoritative_anchor":
        raise RuntimeError("unexpected Gross9 pre-2025 anchor")
    if anchor.get("future_metrics_present") is not False:
        raise RuntimeError("Gross9 pre-2025 anchor contains future metrics")
    if set(anchor.get("selection_stats", {})) != set(SELECTION_SPLITS):
        raise RuntimeError("Gross9 anchor selection windows drifted")
    if anchor.get("accounting_version") != portfolio.ACCOUNTING_VERSION:
        raise RuntimeError("Gross9 anchor accounting drifted")
    if anchor.get("selection_mode") != (
        "frozen_pre2025_allocation_rank_future_veto_only"
    ):
        raise RuntimeError("Gross9 anchor selection mode drifted")
    if anchor.get("future_used_for_allocation_ranking") is not False:
        raise RuntimeError("Gross9 anchor used future allocation ranking")
    if anchor.get("future_can_only_veto_frozen_rank1") is not True:
        raise RuntimeError("Gross9 anchor does not freeze pre-2025 rank1")
    weights = _baseline_weights(preregistration)
    if {
        str(name): float(value)
        for name, value in anchor.get("weights", {}).items()
    } != weights:
        raise RuntimeError("Gross9 anchor weights drifted")
    if not np.isclose(float(anchor.get("gross", -1.0)), 9.0):
        raise RuntimeError("Gross9 anchor gross drifted")
    expected = preregistration["portfolio_baseline"]["frozen_stats"]
    for split in SELECTION_SPLITS:
        actual_metric = anchor["selection_stats"][split]
        for key in (
            "absolute_return_pct",
            "cagr_pct",
            "strict_mdd_pct",
            "cagr_to_strict_mdd",
        ):
            if not np.isclose(
                float(actual_metric[key]),
                float(expected[split][key]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(
                    f"Gross9 anchor drifted in {split}/{key}"
                )
        if int(actual_metric["trades"]) != int(expected[split]["trades"]):
            raise RuntimeError(f"Gross9 anchor trade count drifted in {split}")
    return {
        "anchor_name": anchor["name"],
        "accounting_version": anchor["accounting_version"],
        "selection_mode": anchor["selection_mode"],
        "selection_windows": list(SELECTION_SPLITS),
        "future_metrics_present": False,
        "future_bearing_source_opened": False,
        "weights": weights,
    }


def _install_candidate_universe() -> None:
    for candidate in ALL_CANDIDATES:
        if candidate not in portfolio.SLEEVES:
            portfolio.SLEEVES = (*portfolio.SLEEVES, candidate)
    portfolio.FAMILIES.update(
        {
            "nonpb30_taker": "orderflow_pullback",
            "oi_divergence_highfreq": "oi_divergence",
            **{candidate: "funding_premium" for candidate in STATE_CANDIDATES},
        }
    )


def _candidate_specs(cfg: Config) -> dict[str, dict[str, Any]]:
    nonpb = json.loads(Path(cfg.nonpb30_config).read_text(encoding="utf-8"))
    oi = json.loads(Path(cfg.oi_highfreq_config).read_text(encoding="utf-8"))
    nonpb_signal = nonpb["signal"]
    if (
        str(nonpb_signal["side"]).lower() != "long"
        or int(nonpb_signal["hold_bars_5m"]) != 72
        or int(nonpb_signal["stride_bars_5m"]) != 12
        or int(nonpb_signal["entry_delay_bars"]) != 1
    ):
        raise RuntimeError("nonpb30 execution contract drifted")
    if (
        str(oi["side"]).lower() != "long"
        or int(oi["hold_bars"]) != 30
        or int(oi["stride_bars"]) != 6
    ):
        raise RuntimeError("OI high-frequency execution contract drifted")
    return {
        "nonpb30_taker": {
            "gates": nonpb_signal["gates"],
            "hold": 72,
            "stride": 12,
        },
        "oi_divergence_highfreq": {
            "gates": oi["gates"],
            "hold": 30,
            "stride": 6,
        },
    }


def _validate_gate_features(
    features: pd.DataFrame,
    specs: dict[str, dict[str, Any]],
) -> None:
    for name, spec in specs.items():
        missing = sorted(
            {
                str(gate["feature"])
                for gate in spec["gates"]
                if str(gate["feature"]) not in features.columns
            }
        )
        if missing:
            raise RuntimeError(f"{name} features missing: {missing}")


def build_selection_arrays(
    cfg: Config,
    preregistration: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, Any]]:
    _install_candidate_universe()
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

    specs = _candidate_specs(cfg)
    _validate_gate_features(features, specs)
    addition_counts: dict[str, dict[str, int]] = {}
    for name, spec in specs.items():
        active = _gate_mask(features, spec["gates"])
        addition_counts[name] = portfolio.append_mask_policy(
            events,
            market,
            masks,
            name=name,
            long_active=active,
            short_active=np.zeros(len(market), dtype=bool),
            hold=int(spec["hold"]),
            stride=int(spec["stride"]),
            cost_rate=cfg.cost_rate,
        )

    state_signals, state_audit = build_state_model_top10_ensembles(
        market, features
    )
    state_counts: dict[str, dict[str, int]] = {}
    for name in STATE_CANDIDATES:
        state_counts[name] = portfolio.append_mask_policy(
            events,
            market,
            masks,
            name=name,
            long_active=state_signals[name],
            short_active=np.zeros(len(market), dtype=bool),
            hold=576,
            stride=12,
            cost_rate=cfg.cost_rate,
        )
    arrays = portfolio.split_arrays(events, market, masks)
    source_meta = {
        "rank7_capacity": capacity,
        "oi_cache": oi_cache,
        "feature_columns": int(features.shape[1]),
        "counts": {
            "markov_transition_long": markov_counts,
            "funding_premium_lr_impact_central": funding_counts,
            "rex_taker_low_range_position": rex_counts,
            **path_counts,
            **addition_counts,
            **state_counts,
        },
        "funding_meta": funding_meta,
        "state_model_top10_ensembles": state_audit,
    }
    return market, arrays, source_meta


def _metric(
    arrays: dict[str, dict[str, Any]],
    split: str,
    weights: dict[str, float],
) -> dict[str, Any]:
    return portfolio.strict_metric(
        arrays[split], portfolio.years_for(split), weights
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


def _candidate_diagnostics(
    arrays: dict[str, dict[str, Any]],
    baseline_weights: dict[str, float],
    candidate: str,
    *,
    exclude_markov_from_acceptance: bool,
) -> dict[str, Any]:
    baseline_vector = np.asarray(
        [baseline_weights.get(name, 0.0) for name in portfolio.SLEEVES],
        dtype=float,
    )
    candidate_index = portfolio.SLEEVES.index(candidate)
    result: dict[str, Any] = {}
    acceptance_jaccards: list[float] = []
    for split in SELECTION_SPLITS:
        data = arrays[split]
        candidate_entries = np.asarray(
            data["entry_positions"][candidate], dtype=np.int64
        )
        candidate_count = int(data["counts"][candidate_index])
        if len(candidate_entries) != candidate_count:
            raise RuntimeError(
                f"{split}/{candidate} exact entry count drifted: "
                f"{len(candidate_entries)} != {candidate_count}"
            )
        candidate_occupied = _occupied(data, candidate)
        baseline_occupied = np.zeros(len(candidate_occupied), dtype=bool)
        per_sleeve: dict[str, Any] = {}
        for sleeve, weight in baseline_weights.items():
            if weight <= 0.0:
                continue
            sleeve_index = portfolio.SLEEVES.index(sleeve)
            sleeve_entries = np.asarray(
                data["entry_positions"][sleeve], dtype=np.int64
            )
            sleeve_count = int(data["counts"][sleeve_index])
            if len(sleeve_entries) != sleeve_count:
                raise RuntimeError(
                    f"{split}/{sleeve} exact entry count drifted: "
                    f"{len(sleeve_entries)} != {sleeve_count}"
                )
            occupied = _occupied(data, sleeve)
            baseline_occupied |= occupied
            entry_jaccard = _jaccard(candidate_entries, sleeve_entries)
            acceptance_included = not (
                exclude_markov_from_acceptance
                and sleeve == "markov_transition_long"
            )
            if acceptance_included:
                acceptance_jaccards.append(entry_jaccard)
            per_sleeve[sleeve] = {
                "entry_jaccard": entry_jaccard,
                "occupied_bar_jaccard": float(
                    np.sum(candidate_occupied & occupied)
                    / max(1, np.sum(candidate_occupied | occupied))
                ),
                "acceptance_included": acceptance_included,
                "candidate_entries": candidate_count,
                "sleeve_entries": sleeve_count,
            }
        baseline_returns = baseline_vector @ data["R"]
        candidate_returns = data["R"][candidate_index]
        daily = pd.concat(
            [
                _daily_returns(baseline_returns, data["dates"]),
                _daily_returns(candidate_returns, data["dates"]),
            ],
            axis=1,
        ).fillna(0.0)
        daily.columns = ["baseline", "candidate"]
        raw_pearson = float(daily.corr("pearson").iloc[0, 1])
        raw_spearman = float(daily.corr("spearman").iloc[0, 1])
        worst = np.argsort(baseline_returns)[:20]
        flat_returns = np.where(
            ~baseline_occupied, candidate_returns, 0.0
        )
        result[split] = {
            "per_sleeve": per_sleeve,
            "daily_mtm_correlation": {
                "pearson": (
                    raw_pearson if np.isfinite(raw_pearson) else None
                ),
                "spearman": (
                    raw_spearman if np.isfinite(raw_spearman) else None
                ),
            },
            "candidate_absolute_return_while_gross9_flat_pct": float(
                (np.prod(np.maximum(0.0, 1.0 + flat_returns)) - 1.0)
                * 100.0
            ),
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
    result["max_acceptance_entry_jaccard"] = max(
        acceptance_jaccards, default=0.0
    )
    return result


def _cell_weights(
    *,
    mode: str,
    candidate: str,
    changed_weight: float,
    baseline_weights: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], float]:
    if mode == "addition":
        weights = {**baseline_weights, candidate: changed_weight}
        gross = 9.0 + changed_weight
        scale = gross / 9.0
        comparator = {
            name: weight * scale
            for name, weight in baseline_weights.items()
        }
        return weights, comparator, gross
    if mode == "state_substitution":
        weights = dict(baseline_weights)
        weights["markov_transition_long"] = 2.0 - changed_weight
        if weights["markov_transition_long"] <= 1e-12:
            del weights["markov_transition_long"]
        weights[candidate] = changed_weight
        return weights, dict(baseline_weights), 9.0
    raise ValueError(f"unknown candidate mode: {mode}")


def selection_row(
    *,
    mode: str,
    candidate_name: str,
    changed_weight: float,
    gross: float,
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    comparator: dict[str, dict[str, Any]],
    standalone: dict[str, dict[str, Any]],
    max_entry_jaccard: float,
    preregistration: dict[str, Any],
) -> dict[str, Any]:
    thresholds = preregistration["selection_contract"][
        "numeric_thresholds"
    ]
    improvement = {
        split: float(candidate[split]["cagr_to_strict_mdd"])
        - float(comparator[split]["cagr_to_strict_mdd"])
        for split in SELECTION_SPLITS
    }
    baseline_delta = {
        split: float(candidate[split]["cagr_to_strict_mdd"])
        - float(baseline[split]["cagr_to_strict_mdd"])
        for split in SELECTION_SPLITS
    }
    checks = {
        "train_mdd_cap": (
            candidate["train"]["strict_mdd_pct"]
            <= float(thresholds["train_strict_mdd_cap_pct"])
        ),
        "test2024_mdd_cap": (
            candidate["test2024"]["strict_mdd_pct"]
            <= float(thresholds["test2024_strict_mdd_cap_pct"])
        ),
        "absolute_return_retention": all(
            candidate[split]["absolute_return_pct"]
            >= float(thresholds["absolute_return_retention_floor"])
            * baseline[split]["absolute_return_pct"]
            for split in SELECTION_SPLITS
        ),
        "standalone_positive": all(
            standalone[split]["absolute_return_pct"] > 0.0
            for split in SELECTION_SPLITS
        ),
        "comparator_improvement": all(
            improvement[split]
            >= float(
                thresholds["min_ratio_improvement_over_comparator"]
            )
            for split in SELECTION_SPLITS
        ),
        "mdd_reduction": any(
            candidate[split]["strict_mdd_pct"]
            < baseline[split]["strict_mdd_pct"]
            for split in SELECTION_SPLITS
        ),
        "entry_jaccard": (
            max_entry_jaccard
            <= float(thresholds["entry_jaccard_cap"])
        ),
        "gross_cap": gross <= float(
            preregistration["selection_contract"]["gross_cap"]
        ),
    }
    passes = all(checks.values())
    key = (
        min(improvement.values()),
        math.sqrt(
            max(0.0, improvement["train"])
            * max(0.0, improvement["test2024"])
        ),
        baseline["test2024"]["strict_mdd_pct"]
        - candidate["test2024"]["strict_mdd_pct"],
        (
            candidate["test2024"]["absolute_return_pct"]
            / baseline["test2024"]["absolute_return_pct"]
        ),
        -changed_weight,
    )
    return {
        "mode": mode,
        "candidate_name": candidate_name,
        "changed_weight": float(changed_weight),
        "gross": float(gross),
        "passes": bool(passes),
        "checks": {name: bool(value) for name, value in checks.items()},
        "stats": candidate,
        "comparator_stats": comparator,
        "standalone": standalone,
        "ratio_improvement_over_comparator": improvement,
        "ratio_delta_vs_gross9": baseline_delta,
        "max_acceptance_entry_jaccard": float(max_entry_jaccard),
        "selection_key": list(key),
    }


def _numeric_key(row: dict[str, Any]) -> tuple[float, ...]:
    return tuple(float(value) for value in row["selection_key"])


def _metric_cell(metric: dict[str, Any]) -> str:
    return (
        f"{metric['absolute_return_pct']:.2f}/"
        f"{metric['cagr_pct']:.2f}/"
        f"{metric['strict_mdd_pct']:.2f}/"
        f"{metric['cagr_to_strict_mdd']:.2f}/"
        f"{metric['trades']}"
    )


def _render(payload: dict[str, Any]) -> str:
    lines = [
        "# Gross9 fixed-candidate and state-substitution pre-2025 battery",
        "",
        "Metric: `absolute return / full-calendar CAGR / strict MDD / CAGR-MDD / trades`.",
        "",
        f"- evaluated cells: {payload['tested']}",
        f"- passing cells: {payload['passed']}",
        f"- decision: **{payload['decision']}**",
        "- 2025, 2026, and July metrics are absent and cannot rerank this artifact.",
        "",
        "| candidate | mode | changed weight | pass | train | 2024 | min ratio improvement | max accepted entry Jaccard |",
        "|---|---|---:|:---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['candidate_name']} | {row['mode']} | "
            f"{row['changed_weight']:.2f} | "
            f"{'Y' if row['passes'] else 'N'} | "
            f"{_metric_cell(row['stats']['train'])} | "
            f"{_metric_cell(row['stats']['test2024'])} | "
            f"{min(row['ratio_improvement_over_comparator'].values()):+.3f} | "
            f"{row['max_acceptance_entry_jaccard']:.4f} |"
        )
    lines += [
        "",
        "## Frozen top 1",
        "",
    ]
    if payload["frozen_top1"]:
        top = payload["frozen_top1"]
        lines += [
            f"- candidate: `{top['candidate_name']}`",
            f"- mode / changed weight: `{top['mode']} / {top['changed_weight']:.2f}`",
            f"- train: `{_metric_cell(top['stats']['train'])}`",
            f"- 2024: `{_metric_cell(top['stats']['test2024'])}`",
            "- This row alone may enter the separately bound future veto.",
        ]
    else:
        lines.append("- No cell passed; future veto remains closed.")
    lines += [
        "",
        "## Boundaries",
        "",
        "- Addition cells beat a same-gross pro-rata leverage counterfactual; that comparator is diagnostic and non-deployable.",
        "- State cells keep Gross9 total gross and funding/premium family gross fixed by replacing Markov weight.",
        "- State acceptance Jaccard excludes Markov by contract, but exact Markov overlap is still reported.",
        "- Every candidate uses frozen signals, next-open execution, 0.5x unit leverage, and 6bp per side.",
        "- All source alphas and later windows are research-exposed; a survivor is forward-shadow only.",
    ]
    return "\n".join(lines) + "\n"


def run_pre2025(cfg: Config) -> dict[str, Any]:
    preregistration = load_preregistration(cfg.preregistration)
    input_identity = validate_inputs(cfg, preregistration)
    authoritative = validate_pre2025_gross9_anchor(
        cfg, preregistration
    )
    market, arrays, source_meta = build_selection_arrays(
        cfg, preregistration
    )
    if set(arrays) != set(SELECTION_SPLITS):
        raise RuntimeError("future arrays entered pre-2025 selection")
    baseline_weights = _baseline_weights(preregistration)
    baseline = validate_baseline(arrays, preregistration)
    diagnostics = {
        candidate: _candidate_diagnostics(
            arrays,
            baseline_weights,
            candidate,
            exclude_markov_from_acceptance=(
                candidate in STATE_CANDIDATES
            ),
        )
        for candidate in ALL_CANDIDATES
    }
    standalone = {
        candidate: {
            split: _metric(arrays, split, {candidate: 1.0})
            for split in SELECTION_SPLITS
        }
        for candidate in ALL_CANDIDATES
    }
    rows: list[dict[str, Any]] = []
    grids = preregistration["selection_contract"]
    candidate_modes = [
        *[
            (
                "addition",
                candidate,
                grids["addition_weight_grid"],
            )
            for candidate in ADDITION_CANDIDATES
        ],
        *[
            (
                "state_substitution",
                candidate,
                grids["state_substitution_weight_grid"],
            )
            for candidate in STATE_CANDIDATES
        ],
    ]
    for mode, candidate_name, grid in candidate_modes:
        for raw_weight in grid:
            changed_weight = float(raw_weight)
            weights, comparator_weights, gross = _cell_weights(
                mode=mode,
                candidate=candidate_name,
                changed_weight=changed_weight,
                baseline_weights=baseline_weights,
            )
            candidate_stats = {
                split: _metric(arrays, split, weights)
                for split in SELECTION_SPLITS
            }
            comparator_stats = {
                split: _metric(arrays, split, comparator_weights)
                for split in SELECTION_SPLITS
            }
            rows.append(
                selection_row(
                    mode=mode,
                    candidate_name=candidate_name,
                    changed_weight=changed_weight,
                    gross=gross,
                    baseline=baseline,
                    candidate=candidate_stats,
                    comparator=comparator_stats,
                    standalone=standalone[candidate_name],
                    max_entry_jaccard=float(
                        diagnostics[candidate_name][
                            "max_acceptance_entry_jaccard"
                        ]
                    ),
                    preregistration=preregistration,
                )
            )
    rows.sort(key=lambda row: row["candidate_name"])
    rows.sort(key=_numeric_key, reverse=True)
    passed = [row for row in rows if row["passes"]]
    frozen_top1 = passed[0] if passed else None
    freeze = {
        "preregistration_sha256": _sha256(cfg.preregistration),
        "selection_splits": list(SELECTION_SPLITS),
        "baseline_weights": baseline_weights,
        "candidate_names": list(ALL_CANDIDATES),
        "rows": rows,
        "frozen_top1": frozen_top1,
    }
    payload = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "phase": "pre2025_fixed_candidate_state_substitution",
        "config": asdict(cfg),
        "preregistration": cfg.preregistration,
        "preregistration_sha256": freeze["preregistration_sha256"],
        "input_identity": input_identity,
        "future_only_provenance_not_opened": {
            name: preregistration["input_provenance"][name]
            for name in FUTURE_ONLY_PROVENANCE_KEYS
        },
        "authoritative_gross9": authoritative,
        "market": {
            "rows": int(len(market)),
            "first": str(pd.to_datetime(market["date"]).iloc[0]),
            "last": str(pd.to_datetime(market["date"]).iloc[-1]),
        },
        "source_meta": source_meta,
        "baseline": baseline,
        "standalone": standalone,
        "diagnostics": diagnostics,
        "tested": len(rows),
        "passed": len(passed),
        "rows": rows,
        "frozen_top1": frozen_top1,
        "decision": "open_future_veto" if frozen_top1 else "reject_battery",
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
