"""Audit frozen invariant-ensemble Top10 sleeves as Gross9 additions.

G9-IEM-1 fixes ten pre-evaluation policies and four addition weights before
this runner opens any 2024 outcome.  Selection uses only train-through-2023 and
2024.  A later, separately bound runner may reconstruct future support for the
single frozen winner; this module has no future phase.
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

import training.audit_gross9_fixed_candidate_state_substitution as gross9_context
import training.portfolio_opt_added_alpha_update as portfolio
from training.audit_gross9_fixed_candidate_state_substitution import (
    build_gross9_selection_context,
)
from training.audit_gross9_oi_pullback_marginal import (
    _atomic_json,
    _json_hash,
    _sha256,
)


PREREGISTRATION = Path(
    "results/gross9_invariant_ensemble_top10_marginal_preregistration_2026-07-28.json"
)
OUTPUT = Path(
    "results/gross9_invariant_ensemble_top10_marginal_pre2025_2026-07-28.json"
)
DOCS = Path(
    "docs/gross9-invariant-ensemble-top10-marginal-pre2025-2026-07-28.md"
)
EXPECTED_PREREGISTRATION_SHA256 = (
    "57c8ca0e43d0e3b0c31db5957f094748bbbe48244da4554b40959f313307d58e"
)
SELECTION_SPLITS = ("train", "test2024")
CANDIDATE_NAMES = tuple(
    f"invariant_ensemble_pre_rank{rank:02d}" for rank in range(1, 11)
)
SELECTION_PROVENANCE_KEYS = (
    "market",
    "market_with_oi",
    "funding",
    "premium",
    "gross9_pre2025_anchor",
    "rank7_capacity_evidence",
    "invariant_support",
    "invariant_support_manifest",
    "invariant_groupdro_manifest",
    "invariant_ensemble_manifest",
    "invariant_support_builder",
    "gross9_context_builder",
    "gross9_portfolio_engine",
    "gross9_context_dependencies",
    "markov_candidate",
    "funding_lr_candidate",
    "funding_lr_manifest",
    "rank7_cadence_manifest",
    "rank7_cadence_selection",
    "rex_train_source",
    "rex_test2024_source",
)
FUTURE_ONLY_PROVENANCE_KEYS = (
    "gross9_config",
    "gross9_result",
    "rex_future_source",
    "invariant_full_result",
)
SELECTION_REX_PATHS = (
    "data/rex_pullback_reclaim_q075_h144_ranker_train_2021_2023.jsonl",
    "data/rex_pullback_reclaim_q075_h144_ranker_test_2024.jsonl",
)


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
    rank7_capacity_evidence: str = (
        "results/expanding_extratrees_rank7_leverage_battery_2026-07-27.json"
    )
    support_csv: str = (
        "data/invariant_ensemble_top10_pre2025_support_2026-07-28.csv.gz"
    )
    support_manifest: str = (
        "results/invariant_ensemble_top10_pre2025_support_manifest_2026-07-28.json"
    )
    cost_rate: float = 0.0006
    stress_cost_rate: float = 0.001


def _resolved(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate.resolve()
    fallback = Path("/home/pakchu/rllm") / candidate
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(path)


def load_preregistration(path: str | Path) -> dict[str, Any]:
    observed = _sha256(path)
    if observed != EXPECTED_PREREGISTRATION_SHA256:
        raise RuntimeError(
            f"preregistration hash drifted: {observed} != "
            f"{EXPECTED_PREREGISTRATION_SHA256}"
        )
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("name") != "gross9_invariant_ensemble_top10_marginal_battery":
        raise RuntimeError("unexpected preregistration")
    if payload.get("status") != (
        "preregistered_before_gross9_invariant_marginal_scan"
    ):
        raise RuntimeError("battery was not preregistered")
    selection = payload["selection_contract"]
    if selection.get("candidate_weight_grid") != [0.25, 0.5, 0.75, 1.0]:
        raise RuntimeError("candidate weight grid drifted")
    if selection.get("tested_cells") != 40:
        raise RuntimeError("tested cell count drifted")
    if selection.get("top1_only") is not True:
        raise RuntimeError("top1-only contract drifted")
    if selection.get("exact_selection_windows") != {
        "train": {
            "start_inclusive": "2020-09-01",
            "end_exclusive": "2024-01-01",
            "candidate_support_starts": "2023-01-01",
            "pre_support_candidate_return_policy": (
                "zero/flat while the Gross9 calendar remains active"
            ),
        },
        "test2024": {
            "start_inclusive": "2024-01-01",
            "end_exclusive": "2025-01-01",
        },
    }:
        raise RuntimeError("selection windows drifted")
    candidates = payload["candidate_universe"]["candidates"]
    if [row["candidate_name"] for row in candidates] != list(CANDIDATE_NAMES):
        raise RuntimeError("candidate ordering drifted")
    if [int(row["pre_evaluation_rank"]) for row in candidates] != list(
        range(1, 11)
    ):
        raise RuntimeError("candidate ranks drifted")
    future = payload["future_veto_contract"]
    if any(
        future.get(key) is not False
        for key in (
            "future_can_rerank",
            "future_can_repair",
            "future_can_select_another_rank",
        )
    ):
        raise RuntimeError("future ranking boundary drifted")
    if future["july_2026_scope"].split(":", 1)[0] != (
        "explicitly outside this battery"
    ):
        raise RuntimeError("July boundary drifted")
    return payload


def configured_selection_paths(cfg: Config) -> dict[str, str]:
    fixed = {
        key: str(value["path"])
        for key, value in load_preregistration(
            cfg.preregistration
        )["selection_input_provenance"].items()
    }
    fixed.update(
        {
            "market": cfg.market_csv,
            "market_with_oi": cfg.market_with_oi_csv,
            "funding": cfg.funding_csv,
            "premium": cfg.premium_csv,
            "gross9_pre2025_anchor": cfg.gross9_pre2025_anchor,
            "rank7_capacity_evidence": cfg.rank7_capacity_evidence,
            "invariant_support": cfg.support_csv,
            "invariant_support_manifest": cfg.support_manifest,
        }
    )
    return fixed


def validate_selection_inputs(
    cfg: Config,
    preregistration: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    provenance = preregistration["selection_input_provenance"]
    if tuple(provenance) != SELECTION_PROVENANCE_KEYS:
        raise RuntimeError("selection provenance key order drifted")
    future_only = preregistration[
        "future_only_provenance_not_openable_by_selection"
    ]
    if tuple(future_only) != FUTURE_ONLY_PROVENANCE_KEYS:
        raise RuntimeError("future-only provenance key order drifted")
    configured = configured_selection_paths(cfg)
    if tuple(configured) != SELECTION_PROVENANCE_KEYS:
        raise RuntimeError("configured selection path map drifted")

    records: dict[str, dict[str, Any]] = {}
    for name in SELECTION_PROVENANCE_KEYS:
        record = provenance[name]
        configured_path = str(configured[name])
        if configured_path != str(record["path"]):
            raise RuntimeError(
                f"configured path mismatch for {name}: "
                f"{configured_path} != {record['path']}"
            )
        resolved = _resolved(configured_path)
        observed_hash = _sha256(resolved)
        if observed_hash != str(record["sha256"]):
            raise RuntimeError(
                f"selection input hash drift for {name}: "
                f"{observed_hash} != {record['sha256']}"
            )
        if resolved.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"selection input size drift for {name}")
        records[name] = {
            "configured_path": configured_path,
            "resolved_path": str(resolved),
            "sha256": observed_hash,
            "bytes": resolved.stat().st_size,
            "validated_against_preregistration": True,
        }
    return records


def validate_pre2025_anchor(
    cfg: Config,
    preregistration: dict[str, Any],
) -> dict[str, Any]:
    anchor = json.loads(Path(cfg.gross9_pre2025_anchor).read_text(encoding="utf-8"))
    if anchor.get("name") != "gross9_pre2025_authoritative_anchor":
        raise RuntimeError("unexpected Gross9 pre-2025 anchor")
    if anchor.get("future_metrics_present") is not False:
        raise RuntimeError("Gross9 anchor contains future metrics")
    if anchor.get("future_used_for_allocation_ranking") is not False:
        raise RuntimeError("Gross9 anchor used future ranking")
    if set(anchor.get("selection_stats", {})) != set(SELECTION_SPLITS):
        raise RuntimeError("Gross9 anchor selection windows drifted")
    expected_weights = {
        str(name): float(weight)
        for name, weight in preregistration["portfolio_baseline"]["weights"].items()
    }
    actual_weights = {
        str(name): float(weight) for name, weight in anchor["weights"].items()
    }
    if actual_weights != expected_weights:
        raise RuntimeError("Gross9 anchor weights drifted")
    expected_stats = preregistration["portfolio_baseline"][
        "frozen_selection_stats"
    ]
    for split in SELECTION_SPLITS:
        for key in (
            "absolute_return_pct",
            "cagr_pct",
            "strict_mdd_pct",
            "cagr_to_strict_mdd",
        ):
            if not np.isclose(
                float(anchor["selection_stats"][split][key]),
                float(expected_stats[split][key]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(f"anchor drift in {split}/{key}")
        if int(anchor["selection_stats"][split]["trades"]) != int(
            expected_stats[split]["trades"]
        ):
            raise RuntimeError(f"anchor trade-count drift in {split}")
    return {
        "name": anchor["name"],
        "accounting_version": anchor["accounting_version"],
        "selection_mode": anchor["selection_mode"],
        "weights": actual_weights,
        "future_metrics_present": False,
        "future_bearing_artifacts_opened": False,
    }


def _install_candidate_universe() -> None:
    for name in CANDIDATE_NAMES:
        if name not in portfolio.SLEEVES:
            portfolio.SLEEVES = (*portfolio.SLEEVES, name)
        portfolio.FAMILIES[name] = "invariant_ml"


def load_support(
    cfg: Config,
    preregistration: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = json.loads(Path(cfg.support_manifest).read_text(encoding="utf-8"))
    contract = preregistration["support_contract"]
    if manifest.get("later_metrics_included") is not False:
        raise RuntimeError("support manifest contains later metrics")
    if manifest.get("no_future_rows") is not True:
        raise RuntimeError("support manifest permits future rows")
    if manifest.get("support_csv_sha256") != contract["support_csv_sha256"]:
        raise RuntimeError("support CSV hash contract drifted")
    if manifest.get("support_rows") != contract["support_rows"]:
        raise RuntimeError("support row count contract drifted")
    if manifest.get("support_columns") != contract["support_columns"]:
        raise RuntimeError("support schema contract drifted")
    if _sha256(cfg.support_csv) != str(contract["support_csv_sha256"]):
        raise RuntimeError("support CSV bytes drifted")
    frame = pd.read_csv(cfg.support_csv, compression="infer")
    expected_columns = [
        "pre_evaluation_rank",
        "signal_position",
        "signal_date",
        "side",
    ]
    if frame.columns.tolist() != expected_columns:
        raise RuntimeError("support columns drifted")
    if len(frame) != int(contract["support_rows"]):
        raise RuntimeError("support row count drifted")
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise")
    if not (
        (frame["signal_date"] >= pd.Timestamp("2023-01-01"))
        & (frame["signal_date"] < pd.Timestamp("2025-01-01"))
    ).all():
        raise RuntimeError("support crossed the pre-2025 boundary")
    if set(frame["side"]) != {"long"}:
        raise RuntimeError("support side drifted")
    if set(frame["pre_evaluation_rank"].astype(int)) != set(range(1, 11)):
        raise RuntimeError("support ranks drifted")
    if frame.duplicated(["pre_evaluation_rank", "signal_position"]).any():
        raise RuntimeError("support contains duplicate rank/position rows")

    prereg_rows = preregistration["candidate_universe"]["candidates"]
    manifest_rows = manifest["top10"]
    for prereg_row, manifest_row in zip(prereg_rows, manifest_rows, strict=True):
        rank = int(prereg_row["pre_evaluation_rank"])
        if rank != int(manifest_row["pre_evaluation_rank"]):
            raise RuntimeError("support manifest rank drifted")
        for key in (
            "stream_id",
            "feature_set",
            "transform",
            "rolling_score_window_anchors",
            "score_quantile",
            "side_policy",
            "hold_bars",
            "anchor_stride_bars",
            "support_rows_sha256",
            "support_signal_count_2023_2024",
        ):
            if prereg_row[key] != manifest_row[key]:
                raise RuntimeError(f"rank {rank} support contract drift in {key}")
        if prereg_row["frozen_2023_signal_hash"] != manifest_row[
            "verified_2023_signal_hash"
        ]:
            raise RuntimeError(f"rank {rank} frozen hash drifted")
    return frame, manifest


def support_masks(
    support: pd.DataFrame,
    market: pd.DataFrame,
) -> dict[str, np.ndarray]:
    dates = pd.to_datetime(market["date"]).reset_index(drop=True)
    output: dict[str, np.ndarray] = {}
    for rank, name in enumerate(CANDIDATE_NAMES, start=1):
        rows = support.loc[support["pre_evaluation_rank"].astype(int) == rank]
        positions = rows["signal_position"].astype(int).to_numpy()
        if (positions < 0).any() or (positions >= len(market)).any():
            raise RuntimeError(f"rank {rank} support position is out of bounds")
        observed_dates = dates.iloc[positions].to_numpy()
        expected_dates = rows["signal_date"].to_numpy()
        if not np.array_equal(observed_dates, expected_dates):
            raise RuntimeError(f"rank {rank} support date/position drifted")
        mask = np.zeros(len(market), dtype=bool)
        mask[positions] = True
        output[name] = mask
    return output


def _append_candidates(
    events: list[dict[str, Any]],
    market: pd.DataFrame,
    masks: dict[str, np.ndarray],
    active: dict[str, np.ndarray],
    *,
    cost_rate: float,
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for name in CANDIDATE_NAMES:
        counts[name] = portfolio.append_mask_policy(
            events,
            market,
            masks,
            name=name,
            long_active=active[name],
            short_active=np.zeros(len(market), dtype=bool),
            hold=576,
            stride=72,
            cost_rate=cost_rate,
        )
    return counts


def invariant_support_diagnostics(
    preregistration: dict[str, Any],
    support_manifest: dict[str, Any],
    executable_counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    prereg_rows = preregistration["candidate_universe"]["candidates"]
    manifest_rows = support_manifest["top10"]
    per_rank: list[dict[str, Any]] = []
    for prereg_row, manifest_row in zip(
        prereg_rows, manifest_rows, strict=True
    ):
        rank = int(prereg_row["pre_evaluation_rank"])
        name = str(prereg_row["candidate_name"])
        per_rank.append(
            {
                "pre_evaluation_rank": rank,
                "candidate_name": name,
                "frozen_2023_signal_hash": prereg_row[
                    "frozen_2023_signal_hash"
                ],
                "verified_2023_signal_hash": manifest_row[
                    "verified_2023_signal_hash"
                ],
                "emitted_2023_signal_hash": manifest_row[
                    "emitted_2023_signal_hash"
                ],
                "support_rows_sha256": manifest_row[
                    "support_rows_sha256"
                ],
                "support_signal_count_2023_2024": int(
                    manifest_row["support_signal_count_2023_2024"]
                ),
                "executable_counts": {
                    split: int(executable_counts[name][split])
                    for split in SELECTION_SPLITS
                },
                "date_position_parity": True,
            }
        )
    return {
        "support_csv_sha256": support_manifest["support_csv_sha256"],
        "support_manifest_builder": support_manifest["builder"],
        "processed_pre2025_market_frame_sha256": support_manifest[
            "processed_pre2025_market_frame_sha256"
        ],
        "support_rows": int(support_manifest["support_rows"]),
        "support_columns": list(support_manifest["support_columns"]),
        "later_metrics_included": False,
        "no_future_rows": True,
        "all_rank_date_position_parity": True,
        "per_rank": per_rank,
    }


def append_rex_taker_policy_selection_only(
    events: list[dict[str, Any]],
    market: pd.DataFrame,
    masks: dict[str, np.ndarray],
    *,
    cost_rate: float,
) -> dict[str, int]:
    """Rebuild the frozen REX sleeve without opening its 2025+ source file."""
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    for raw_path in SELECTION_REX_PATHS:
        for line in _resolved(raw_path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[(int(row["signal_pos"]), str(row["date"]))] = row
    ordered = sorted(rows.values(), key=lambda row: int(row["signal_pos"]))
    dates = pd.to_datetime(market["date"])
    counts: dict[str, int] = {}
    for split, split_mask in masks.items():
        returns = np.zeros(len(market), dtype=np.float64)
        adverse = np.zeros(len(market), dtype=np.float64)
        favorable = np.zeros(len(market), dtype=np.float64)
        market_low = np.zeros(len(market), dtype=np.float64)
        market_high = np.zeros(len(market), dtype=np.float64)
        entry_positions: list[int] = []
        next_allowed = 0
        trades = wins = 0
        first_signal: int | None = None
        for row in ordered:
            position = int(row["signal_pos"])
            if (
                position < next_allowed
                or position >= len(split_mask)
                or not split_mask[position]
            ):
                continue
            if pd.Timestamp(row["date"]) != pd.Timestamp(dates.iloc[position]):
                raise RuntimeError(
                    "REX source row is not aligned to the shared market grid"
                )
            if not portfolio.rex_gate_match(row, list(portfolio.REX_GATES)):
                continue
            side = str((row.get("action") or {}).get("side", "")).lower()
            if side not in {"long", "short"}:
                continue
            path = portfolio.new_alpha._event_path(
                market,
                position,
                side=side,
                hold=144,
                cost_rate=cost_rate,
                entry_delay=1,
                leverage=0.5,
            )
            if path is None:
                continue
            event_return, event_adverse, realized = path
            nonzero = np.flatnonzero(np.abs(event_return) > 1e-15)
            exit_position = (
                int(nonzero[-1]) if len(nonzero) else position + 145
            )
            if (
                exit_position >= len(split_mask)
                or not split_mask[exit_position]
            ):
                continue
            returns += event_return
            adverse += event_adverse
            event_favorable = portfolio.favorable_path(
                market,
                signal_position=position,
                exit_position=exit_position,
                side=side,
                leverage=0.5,
            )
            favorable += event_favorable
            if side == "long":
                market_low += event_adverse
                market_high += event_favorable
            else:
                market_low += event_favorable
                market_high += event_adverse
            trades += 1
            wins += int(float(realized) > 0.0)
            entry_positions.append(position + 1)
            first_signal = position if first_signal is None else first_signal
            next_allowed = exit_position + 1
        counts[split] = trades
        if trades:
            events.append(
                {
                    "split": split,
                    "sleeve": "rex_taker_low_range_position",
                    "side": "mixed",
                    "signal_pos": int(first_signal or 0),
                    "date": str(dates.iloc[int(first_signal or 0)]),
                    "ret": returns,
                    "adv": adverse,
                    "fav": favorable,
                    "low": market_low,
                    "high": market_high,
                    "trade_count": trades,
                    "win_count": wins,
                    "entry_positions": entry_positions,
                }
            )
    return counts


def build_selection_arrays(
    cfg: Config,
    preregistration: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    _install_candidate_universe()
    support, support_manifest = load_support(cfg, preregistration)
    base_cfg = gross9_context.Config(
        market_csv=cfg.market_csv,
        market_with_oi_csv=cfg.market_with_oi_csv,
        funding_csv=cfg.funding_csv,
        premium_csv=cfg.premium_csv,
        gross9_pre2025_anchor=cfg.gross9_pre2025_anchor,
        rank7_capacity_evidence=cfg.rank7_capacity_evidence,
        cost_rate=cfg.cost_rate,
    )
    original_rex_builder = portfolio.append_rex_taker_policy
    portfolio.append_rex_taker_policy = append_rex_taker_policy_selection_only
    try:
        market, masks, base_events, _features, source_meta = (
            build_gross9_selection_context(base_cfg)
        )
    finally:
        portfolio.append_rex_taker_policy = original_rex_builder
    if tuple(masks) != SELECTION_SPLITS:
        raise RuntimeError("future masks entered selection")
    active = support_masks(support, market)

    normal_events = list(base_events)
    normal_counts = _append_candidates(
        normal_events,
        market,
        masks,
        active,
        cost_rate=cfg.cost_rate,
    )
    arrays = portfolio.split_arrays(normal_events, market, masks)

    stress_events = list(base_events)
    stress_counts = _append_candidates(
        stress_events,
        market,
        masks,
        active,
        cost_rate=cfg.stress_cost_rate,
    )
    stress_arrays = portfolio.split_arrays(stress_events, market, masks)
    if set(arrays) != set(SELECTION_SPLITS) or set(stress_arrays) != set(
        SELECTION_SPLITS
    ):
        raise RuntimeError("future arrays entered selection")
    if normal_counts != stress_counts:
        raise RuntimeError("cost stress changed the candidate schedule")

    support_counts = support_manifest["support_counts_by_rank_and_year"]
    for rank, name in enumerate(CANDIDATE_NAMES, start=1):
        expected = support_counts[str(rank)]
        if normal_counts[name] != {
            "train": int(expected["2023"]),
            "test2024": int(expected["2024"]),
        }:
            raise RuntimeError(f"rank {rank} executable support count drifted")
    source_meta["invariant_support"] = {
        "rows": int(len(support)),
        "sha256": _sha256(cfg.support_csv),
        "normal_counts": normal_counts,
        "stress_counts": stress_counts,
        "stress_semantics": (
            "Gross9 remains at frozen 6bp/side; the added candidate is "
            "replayed exactly at 10bp/side"
        ),
    }
    source_meta["invariant_support_manifest"] = (
        invariant_support_diagnostics(
            preregistration,
            support_manifest,
            normal_counts,
        )
    )
    source_meta["selection_future_file_guard"] = {
        "rex_paths_opened": list(SELECTION_REX_PATHS),
        "rex_future_path_opened": False,
    }
    return market, arrays, stress_arrays, source_meta


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
    weights = {
        str(name): float(weight)
        for name, weight in preregistration["portfolio_baseline"]["weights"].items()
    }
    expected = preregistration["portfolio_baseline"][
        "frozen_selection_stats"
    ]
    observed = {
        split: _metric(arrays, split, weights) for split in SELECTION_SPLITS
    }
    for split in SELECTION_SPLITS:
        for key in (
            "absolute_return_pct",
            "cagr_pct",
            "strict_mdd_pct",
            "cagr_to_strict_mdd",
        ):
            if not np.isclose(
                float(observed[split][key]),
                float(expected[split][key]),
                rtol=0.0,
                atol=1e-10,
            ):
                raise RuntimeError(f"rebuilt Gross9 drift in {split}/{key}")
        if int(observed[split]["trades"]) != int(expected[split]["trades"]):
            raise RuntimeError(f"rebuilt Gross9 trade-count drift in {split}")
    return observed


def pooled_entry_jaccard(
    arrays: dict[str, dict[str, Any]],
    baseline_weights: dict[str, float],
    candidate: str,
) -> dict[str, float]:
    candidate_entries = np.concatenate(
        [
            np.asarray(arrays[split]["entry_positions"][candidate], dtype=np.int64)
            for split in SELECTION_SPLITS
        ]
    )
    output: dict[str, float] = {}
    for sleeve, weight in baseline_weights.items():
        if weight <= 0.0:
            continue
        entries = np.concatenate(
            [
                np.asarray(
                    arrays[split]["entry_positions"][sleeve], dtype=np.int64
                )
                for split in SELECTION_SPLITS
            ]
        )
        left, right = set(candidate_entries.tolist()), set(entries.tolist())
        union = left | right
        output[sleeve] = float(len(left & right) / len(union)) if union else 0.0
    return output


def candidate_diagnostics(
    arrays: dict[str, dict[str, Any]],
    baseline_weights: dict[str, float],
    candidate: str,
) -> dict[str, Any]:
    diagnostics = gross9_context._candidate_diagnostics(
        arrays,
        baseline_weights,
        candidate,
        exclude_markov_from_acceptance=False,
    )
    diagnostics["pooled_train_test2024_entry_jaccard"] = pooled_entry_jaccard(
        arrays, baseline_weights, candidate
    )
    return diagnostics


def cell_weights(
    baseline_weights: dict[str, float],
    candidate: str,
    weight: float,
) -> tuple[dict[str, float], dict[str, float], float]:
    weights = {**baseline_weights, candidate: float(weight)}
    gross = float(sum(weights.values()))
    comparator = {
        name: float(value) * gross / 9.0
        for name, value in baseline_weights.items()
    }
    return weights, comparator, gross


def signed_geometric_mean(left: float, right: float) -> float:
    product = float(left) * float(right)
    if product == 0.0:
        return 0.0
    sign = 1.0 if min(float(left), float(right)) > 0.0 else -1.0
    return sign * math.sqrt(abs(product))


def selection_row(
    *,
    preregistration: dict[str, Any],
    candidate_name: str,
    pre_evaluation_rank: int,
    weight: float,
    gross: float,
    baseline: dict[str, dict[str, Any]],
    stats: dict[str, dict[str, Any]],
    comparator: dict[str, dict[str, Any]],
    standalone: dict[str, dict[str, Any]],
    stressed_stats: dict[str, dict[str, Any]],
    stressed_standalone: dict[str, dict[str, Any]],
    max_entry_jaccard: float,
) -> dict[str, Any]:
    thresholds = preregistration["selection_contract"]["numeric_thresholds"]
    improvement = {
        split: float(stats[split]["cagr_to_strict_mdd"])
        - float(comparator[split]["cagr_to_strict_mdd"])
        for split in SELECTION_SPLITS
    }
    baseline_delta = {
        split: float(stats[split]["cagr_to_strict_mdd"])
        - float(baseline[split]["cagr_to_strict_mdd"])
        for split in SELECTION_SPLITS
    }
    return_retention = {
        split: float(stats[split]["absolute_return_pct"])
        / float(baseline[split]["absolute_return_pct"])
        for split in SELECTION_SPLITS
    }
    mdd_reduction = {
        split: float(baseline[split]["strict_mdd_pct"])
        - float(stats[split]["strict_mdd_pct"])
        for split in SELECTION_SPLITS
    }
    minimum_trades = int(
        thresholds["min_candidate_trades_each_selection_window"]
    )
    checks = {
        "train_mdd_cap": stats["train"]["strict_mdd_pct"]
        <= float(thresholds["train_strict_mdd_cap_pct"]),
        "test2024_mdd_cap": stats["test2024"]["strict_mdd_pct"]
        <= float(thresholds["test2024_strict_mdd_cap_pct"]),
        "absolute_return_retention": all(
            return_retention[split]
            >= float(thresholds["absolute_return_retention_floor"])
            for split in SELECTION_SPLITS
        ),
        "standalone_positive": all(
            standalone[split]["absolute_return_pct"] > 0.0
            for split in SELECTION_SPLITS
        ),
        "minimum_candidate_trades": all(
            int(standalone[split]["trades"]) >= minimum_trades
            for split in SELECTION_SPLITS
        ),
        "comparator_improvement": all(
            improvement[split]
            >= float(thresholds["min_ratio_improvement_over_comparator"])
            for split in SELECTION_SPLITS
        ),
        "mdd_reduction": any(
            mdd_reduction[split] > 0.0 for split in SELECTION_SPLITS
        ),
        "entry_jaccard": max_entry_jaccard
        <= float(thresholds["entry_jaccard_cap"]),
        "candidate_family_cap": weight
        <= float(
            preregistration["selection_contract"][
                "candidate_family_gross_cap"
            ]
        ),
        "gross_cap": gross
        <= float(preregistration["selection_contract"]["gross_cap"]),
        "stress_standalone_positive": all(
            stressed_standalone[split]["absolute_return_pct"] > 0.0
            for split in SELECTION_SPLITS
        ),
        "stress_portfolio_positive": all(
            stressed_stats[split]["absolute_return_pct"] > 0.0
            for split in SELECTION_SPLITS
        ),
    }
    key = (
        min(improvement.values()),
        signed_geometric_mean(improvement["train"], improvement["test2024"]),
        min(
            float(stressed_standalone[split]["absolute_return_pct"])
            for split in SELECTION_SPLITS
        ),
        mdd_reduction["test2024"],
        return_retention["test2024"],
        -float(weight),
        -float(pre_evaluation_rank),
    )
    return {
        "candidate_name": candidate_name,
        "pre_evaluation_rank": int(pre_evaluation_rank),
        "weight": float(weight),
        "gross": float(gross),
        "passes": bool(all(checks.values())),
        "checks": {name: bool(value) for name, value in checks.items()},
        "stats": stats,
        "comparator_stats": comparator,
        "standalone": standalone,
        "stressed_stats": stressed_stats,
        "stressed_standalone": stressed_standalone,
        "ratio_improvement_over_comparator": improvement,
        "ratio_delta_vs_gross9": baseline_delta,
        "absolute_return_retention": return_retention,
        "mdd_reduction_vs_gross9": mdd_reduction,
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


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Gross9 invariant-ensemble Top10 pre-2025 marginal battery",
        "",
        "Metric: `absolute return / full-calendar CAGR / strict MDD / CAGR-MDD / trades`.",
        "",
        f"- evaluated cells: {payload['tested']}",
        f"- passing cells: {payload['passed']}",
        f"- decision: **{payload['decision']}**",
        "- Future-bearing artifacts and future outcomes were not opened.",
        "",
        "| candidate | rank | weight | pass | train | 2024 | min ratio improvement | max entry Jaccard |",
        "|---|---:|---:|:---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['candidate_name']} | {row['pre_evaluation_rank']} | "
            f"{row['weight']:.2f} | {'Y' if row['passes'] else 'N'} | "
            f"{_metric_cell(row['stats']['train'])} | "
            f"{_metric_cell(row['stats']['test2024'])} | "
            f"{min(row['ratio_improvement_over_comparator'].values()):+.3f} | "
            f"{row['max_acceptance_entry_jaccard']:.4f} |"
        )
    lines += ["", "## Frozen top 1", ""]
    if payload["frozen_top1"] is None:
        lines.append("- No cell passed. Future support remains closed.")
    else:
        row = payload["frozen_top1"]
        lines += [
            f"- candidate: `{row['candidate_name']}`",
            f"- pre-evaluation rank / weight: `{row['pre_evaluation_rank']} / {row['weight']:.2f}`",
            f"- train: `{_metric_cell(row['stats']['train'])}`",
            f"- 2024: `{_metric_cell(row['stats']['test2024'])}`",
            "- This exact row alone may enter the separately bound future-support builder.",
        ]
    lines += [
        "",
        "## Boundaries",
        "",
        "- Candidate signals are frozen outcome-blind support rows; no feature, threshold, side, hold, stride, or exit was searched here.",
        "- Every cell is compared with pro-rata Gross9 at identical gross.",
        "- Gross9 remains at 6bp/side in stress; only the added candidate is exactly replayed at 10bp/side.",
        "- All candidates and later periods are research-exposed; even a future survivor is forward-shadow only.",
    ]
    return "\n".join(lines) + "\n"


def run_pre2025(cfg: Config) -> dict[str, Any]:
    preregistration = load_preregistration(cfg.preregistration)
    input_identity = validate_selection_inputs(cfg, preregistration)
    authoritative = validate_pre2025_anchor(cfg, preregistration)
    market, arrays, stress_arrays, source_meta = build_selection_arrays(
        cfg, preregistration
    )
    baseline_weights = {
        str(name): float(weight)
        for name, weight in preregistration["portfolio_baseline"]["weights"].items()
    }
    baseline = validate_baseline(arrays, preregistration)
    diagnostics = {
        name: candidate_diagnostics(arrays, baseline_weights, name)
        for name in CANDIDATE_NAMES
    }
    standalone = {
        name: {
            split: _metric(arrays, split, {name: 1.0})
            for split in SELECTION_SPLITS
        }
        for name in CANDIDATE_NAMES
    }
    stressed_standalone = {
        name: {
            split: _metric(stress_arrays, split, {name: 1.0})
            for split in SELECTION_SPLITS
        }
        for name in CANDIDATE_NAMES
    }

    rows: list[dict[str, Any]] = []
    grid = preregistration["selection_contract"]["candidate_weight_grid"]
    for pre_rank, name in enumerate(CANDIDATE_NAMES, start=1):
        for raw_weight in grid:
            weight = float(raw_weight)
            weights, comparator_weights, gross = cell_weights(
                baseline_weights, name, weight
            )
            stats = {
                split: _metric(arrays, split, weights)
                for split in SELECTION_SPLITS
            }
            comparator = {
                split: _metric(arrays, split, comparator_weights)
                for split in SELECTION_SPLITS
            }
            stressed_stats = {
                split: _metric(stress_arrays, split, weights)
                for split in SELECTION_SPLITS
            }
            rows.append(
                selection_row(
                    preregistration=preregistration,
                    candidate_name=name,
                    pre_evaluation_rank=pre_rank,
                    weight=weight,
                    gross=gross,
                    baseline=baseline,
                    stats=stats,
                    comparator=comparator,
                    standalone=standalone[name],
                    stressed_stats=stressed_stats,
                    stressed_standalone=stressed_standalone[name],
                    max_entry_jaccard=float(
                        diagnostics[name]["max_acceptance_entry_jaccard"]
                    ),
                )
            )
    if len(rows) != 40:
        raise RuntimeError(f"tested cell count drifted: {len(rows)}")
    rows.sort(key=lambda row: row["candidate_name"])
    rows.sort(key=_numeric_key, reverse=True)
    passed = [row for row in rows if row["passes"]]
    frozen_top1 = passed[0] if passed else None
    freeze = {
        "preregistration_sha256": _sha256(cfg.preregistration),
        "selection_splits": list(SELECTION_SPLITS),
        "baseline_weights": baseline_weights,
        "candidate_names": list(CANDIDATE_NAMES),
        "rows": rows,
        "frozen_top1": frozen_top1,
    }
    payload = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "phase": "pre2025_gross9_invariant_ensemble_top10_marginal",
        "config": asdict(cfg),
        "preregistration": cfg.preregistration,
        "preregistration_sha256": _sha256(cfg.preregistration),
        "input_identity": input_identity,
        "future_only_provenance_not_opened": preregistration[
            "future_only_provenance_not_openable_by_selection"
        ],
        "authoritative_gross9": authoritative,
        "market": {
            "rows": int(len(market)),
            "first": str(pd.to_datetime(market["date"]).iloc[0]),
            "last": str(pd.to_datetime(market["date"]).iloc[-1]),
        },
        "source_meta": source_meta,
        "baseline": baseline,
        "standalone": standalone,
        "stressed_standalone": stressed_standalone,
        "diagnostics": diagnostics,
        "tested": len(rows),
        "passed": len(passed),
        "rows": rows,
        "frozen_top1": frozen_top1,
        "decision": (
            "open_frozen_top1_future_support"
            if frozen_top1 is not None
            else "reject_battery"
        ),
        "future_opened": False,
        "future_can_rerank": False,
        "freeze_hash": _json_hash(freeze),
    }
    _atomic_json(cfg.output, payload)
    docs = Path(cfg.docs_output)
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text(render(payload), encoding="utf-8")
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
                "freeze_hash": payload["freeze_hash"],
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
