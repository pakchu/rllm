"""Diagnostic test2024/eval2025 revalidation for frozen Gross9 overlap candidates.

The protocol evaluates the frozen 78-candidate family from
``preregister_gross9_frequency_waived_revalidation``.  It explicitly overrides
the prior 2024 terminal stop for diagnostic measurement only: weights are fixed,
there is no repair/rerank/substitution, final2026 is not opened, and turnover /
trade-frequency limits are disclosures rather than rejection gates.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from training import build_gross9_overlap_net_position_config as net_config
from training import evaluate_gross9_overlap_net_position_eval2025_override as normalized_eval_source
from training import evaluate_gross9_overlap_net_position_portfolio as original_evaluator
from training import evaluate_gross9_qtr_distill_economics as fixed_ledger
from training import preregister_gross9_frequency_waived_revalidation as freeze_mod

POLICY_ID = freeze_mod.POLICY_ID
PROTOCOL_VERSION = "gross9_frequency_waived_revalidation_evaluator_v2"
FREEZE = freeze_mod.DEFAULT_OUTPUT
DEFAULT_OUTPUT = Path("results/gross9_frequency_waived_revalidation_test_eval_v2_2026-09-04.json")
BASE_COST = fixed_ledger.BASE_COST
STRESS_COST = fixed_ledger.STRESS_COST
MIN_INTERVALS = 8
MIN_ACTIVE_WEEKS = 4
MIN_AGGREGATE_NET_SIGNED_EPISODES = 12
STRICT_MDD_MAX = 15.0
CAGR_TO_MDD_MIN = 3.0
STRESS_CAGR_TO_MDD_MIN = 2.5
MEAN_GROSS_EDGE_MIN_BP = 20.0
MEAN_ABS_NET_POSITION_MAX = 0.85
MAX_ABS_NET_POSITION_MAX = 1.0


def canonical_hash(value: Any) -> str:
    return freeze_mod.canonical_hash(value)


def sha256_file(path: str | Path) -> str:
    return freeze_mod.sha256_file(path)


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _iso_z(value: Any) -> str:
    return _utc(value).isoformat().replace("+00:00", "Z")


def load_freeze(path: Path = FREEZE) -> dict[str, Any]:
    value = freeze_mod.load_hashed_json(path)
    freeze_mod.validate(value)
    built = freeze_mod.build()
    if value != built:
        raise RuntimeError(f"{POLICY_ID} freeze no longer matches current source bindings")
    return value


def freeze_receipt(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path), "manifest_hash": value["manifest_hash"]}


def verify_terminal_override_chain(freeze: Mapping[str, Any]) -> dict[str, Any]:
    inputs = freeze["frozen_inputs"]
    original_freeze = freeze_mod.load_hashed_json(inputs["original_validation_freeze"]["path"])
    holdout = freeze_mod.load_hashed_json(inputs["holdout_dec2023_artifact"]["path"])
    test2024 = freeze_mod.load_hashed_json(inputs["test2024_terminal_artifact"]["path"])
    freeze_mod.validate_original_terminal_chain(original_freeze, holdout, test2024)
    for label, artifact in (
        ("original_validation_freeze", original_freeze),
        ("holdout_dec2023_artifact", holdout),
        ("test2024_terminal_artifact", test2024),
    ):
        expected = inputs[label]
        observed = freeze_mod.receipt(expected["path"], artifact)
        if observed != expected:
            raise RuntimeError(f"{POLICY_ID} override chain receipt drift: {label}")
    return {
        "original_validation_freeze": dict(inputs["original_validation_freeze"]),
        "holdout_dec2023_artifact": dict(inputs["holdout_dec2023_artifact"]),
        "test2024_terminal_artifact": dict(inputs["test2024_terminal_artifact"]),
        "test2024_stop_explicitly_overridden": True,
    }


def load_all_stage_sources(freeze: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for stage, row in freeze["stages"].items():
        if stage not in freeze_mod.STAGES:
            raise RuntimeError(f"{POLICY_ID} unknown frozen stage: {stage}")
        start = _utc(row["start"])
        end = _utc(row["end"])
        if stage == "eval2025":
            market, funding, source = normalized_eval_source.load_normalized_eval2025_sources(
                start,
                end,
            )
        else:
            market, funding, source = original_evaluator.load_stage_sources(
                stage,
                start,
                end,
            )
        source = normalized_eval_source.public_source_receipt(source)
        sources[stage] = {"market": market, "funding": funding, "source": source, "start": start, "end": end, "split": row["split"]}
    if set(sources) != {"test2024", "eval2025"}:
        raise RuntimeError(f"{POLICY_ID} source stage drift")
    return sources


def load_sleeve_clocks_once(freeze: Mapping[str, Any]) -> dict[str, dict[str, pd.DataFrame]]:
    clocks: dict[str, dict[str, pd.DataFrame]] = {}
    stage_by_split = {row["split"]: stage for stage, row in freeze["stages"].items()}
    for record in freeze["frozen_inputs"]["selected_clocks"]:
        path = Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"{POLICY_ID} selected clock hash drift: {path}")
        stage_rows: dict[str, list[dict[str, Any]]] = {stage: [] for stage in freeze["stages"]}
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"split", "entry_time", "exit_time", "side"}
            if not required.issubset(reader.fieldnames or []):
                raise RuntimeError(f"{POLICY_ID} selected clock schema drift: {path}")
            for row in reader:
                stage = stage_by_split.get(row["split"])
                if stage is None:
                    continue
                bounds = freeze["stages"][stage]
                entry = _utc(row["entry_time"])
                exit_time = _utc(row["exit_time"])
                if entry >= _utc(bounds["start"]) and exit_time <= _utc(bounds["end"]):
                    stage_rows[stage].append({"entry_time": entry, "exit_time": exit_time, "side": row["side"]})
        clocks[record["sleeve_id"]] = {}
        for stage, rows in stage_rows.items():
            frame = pd.DataFrame(rows, columns=["entry_time", "exit_time", "side"])
            frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True, errors="raise")
            frame["exit_time"] = pd.to_datetime(frame["exit_time"], utc=True, errors="raise")
            frame["side"] = pd.to_numeric(frame["side"], errors="raise").astype(int)
            if not frame.empty:
                frame = frame.sort_values(["entry_time", "exit_time"]).reset_index(drop=True)
                if not frame["side"].isin([-1, 1]).all() or not (frame["entry_time"] < frame["exit_time"]).all():
                    raise RuntimeError(f"{POLICY_ID} selected clock value drift: {path}")
                if len(frame) > 1 and (frame["entry_time"].iloc[1:].to_numpy() < frame["exit_time"].iloc[:-1].to_numpy()).any():
                    raise RuntimeError(f"{POLICY_ID} intra-sleeve overlap: {path}")
            clocks[record["sleeve_id"]][stage] = frame
    return clocks


def candidate_clock(candidate: Mapping[str, Any], sleeve_clocks: Mapping[str, Mapping[str, pd.DataFrame]], stage: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for sleeve_id, weight in candidate["weights"].items():
        frame = sleeve_clocks[sleeve_id][stage]
        if frame.empty:
            continue
        part = frame.copy()
        part["sleeve"] = sleeve_id
        part["weight"] = float(weight)
        rows.append(part[["sleeve", "weight", "entry_time", "exit_time", "side"]])
    if not rows:
        return pd.DataFrame(columns=["sleeve", "weight", "entry_time", "exit_time", "side"])
    return fixed_ledger.normalize_portfolio_clock(pd.concat(rows, ignore_index=True), require_four_sleeves=False)


def weekly_effect_vector(effect_rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int], float]:
    weekly: dict[tuple[int, int], float] = {}
    for row in effect_rows:
        effect = float(row.get("log_effect", 0.0))
        if effect == 0.0:
            continue
        iso = _utc(row["time"]).isocalendar()
        key = (int(iso.year), int(iso.week))
        weekly[key] = weekly.get(key, 0.0) + effect
    return weekly


def aligned_week_max_t(candidate_weekly: Mapping[str, Mapping[tuple[int, int], float]], draws: int = freeze_mod.MAX_T_DRAWS, seed: int = freeze_mod.MAX_T_SEED) -> dict[str, Any]:
    candidate_ids = tuple(sorted(candidate_weekly))
    weeks = tuple(sorted({week for values in candidate_weekly.values() for week in values}))
    if not candidate_ids or not weeks:
        return {"method": "shared_aligned_week_signflip_max_t", "draws": draws, "seed": seed, "weeks": 0, "adjusted_pvalues": {cid: 1.0 for cid in candidate_ids}}
    matrix = np.zeros((len(candidate_ids), len(weeks)), dtype=float)
    for row_idx, cid in enumerate(candidate_ids):
        values = candidate_weekly[cid]
        for col_idx, week in enumerate(weeks):
            matrix[row_idx, col_idx] = float(values.get(week, 0.0))
    observed = matrix.sum(axis=1)
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(draws, len(weeks)))
    null_max = np.max(signs @ matrix.T, axis=1)
    adjusted = {cid: float((1 + int((null_max >= observed[idx]).sum())) / (draws + 1)) for idx, cid in enumerate(candidate_ids)}
    return {"method": "shared_aligned_week_signflip_max_t", "draws": draws, "seed": seed, "weeks": len(weeks), "candidate_count": len(candidate_ids), "adjusted_pvalues": adjusted}


def retained_checks(primary: Mapping[str, Any], net_risk: Mapping[str, float], intervals: int, active_weeks: int, signed_episodes: int, max_t_pvalue: float) -> dict[str, bool]:
    base = primary["base"]
    stress = primary["stress"]
    return {
        "absolute_return_positive": float(base["absolute_return_pct"]) > 0.0,
        "cagr_to_strict_mdd_min": float(base["cagr_to_strict_mdd"]) >= CAGR_TO_MDD_MIN,
        "strict_mdd_max": float(base["strict_mdd_pct"]) <= STRICT_MDD_MAX,
        "mean_exposure_weighted_gross_edge_min": float(base.get("mean_exposure_weighted_gross_edge_bp", 0.0)) >= MEAN_GROSS_EDGE_MIN_BP,
        "stress_absolute_return_positive": float(stress["absolute_return_pct"]) > 0.0,
        "stress_cagr_to_strict_mdd_min": float(stress["cagr_to_strict_mdd"]) >= STRESS_CAGR_TO_MDD_MIN,
        "each_calendar_half_positive": all(float(row["absolute_return_pct"]) > 0.0 for row in primary["calendar_halves"].values()),
        "max_t_aligned_week_p_max_0_10": float(max_t_pvalue) <= freeze_mod.MAX_T_P_MAX,
        "mean_abs_net_position_cap": float(net_risk["mean_abs_net_position"]) <= MEAN_ABS_NET_POSITION_MAX,
        "max_abs_net_position_cap": float(net_risk["max_abs_net_position"]) <= MAX_ABS_NET_POSITION_MAX,
        "minimum_intervals": int(intervals) >= MIN_INTERVALS,
        "minimum_active_iso_weeks": int(active_weeks) >= MIN_ACTIVE_WEEKS,
        "minimum_aggregate_net_signed_episodes": int(signed_episodes)
        >= MIN_AGGREGATE_NET_SIGNED_EPISODES,
    }


def log_return_from_primary(primary: Mapping[str, Any]) -> float:
    base = primary["base"]
    return float(np.log(float(base["final_equity"]) / float(base["initial_equity"])))


def evaluate_primary_without_candidate_mc(
    clock: pd.DataFrame,
    market: pd.DataFrame,
    funding: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_raw = fixed_ledger.simulate_portfolio(
        clock,
        market,
        funding,
        start,
        end,
        BASE_COST,
    )
    stress_raw = fixed_ledger.simulate_portfolio(
        clock,
        market,
        funding,
        start,
        end,
        STRESS_COST,
    )
    midpoint = start + (end - start) / 2
    halves = {
        name: fixed_ledger.public_metric(
            fixed_ledger.simulate_portfolio(
                clock.loc[clock["entry_time"].ge(left) & clock["exit_time"].le(right)],
                market,
                funding,
                left,
                right,
                BASE_COST,
            )
        )
        for name, left, right in (
            ("first", start, midpoint),
            ("second", midpoint, end),
        )
    }
    primary = {
        "base": fixed_ledger.public_metric(base_raw),
        "stress": fixed_ledger.public_metric(stress_raw),
        "calendar_halves": halves,
        "individual_cluster_signflip": "replaced_by_shared_aligned_week_max_t",
    }
    return primary, base_raw


def evaluate_candidate_stage(candidate: Mapping[str, Any], clock: pd.DataFrame, source_bundle: Mapping[str, Any]) -> tuple[dict[str, Any], Mapping[tuple[int, int], float]]:
    start = source_bundle["start"]
    end = source_bundle["end"]
    market = source_bundle["market"]
    funding = source_bundle["funding"]
    if clock.empty:
        empty_primary = {
            "base": {"initial_equity": fixed_ledger.INITIAL_EQUITY, "final_equity": fixed_ledger.INITIAL_EQUITY, "absolute_return_pct": 0.0, "cagr_pct": 0.0, "strict_mdd_pct": 0.0, "cagr_to_strict_mdd": 0.0, "intervals": 0, "long_intervals": 0, "short_intervals": 0, "mean_exposure_weighted_gross_edge_bp": 0.0, "transitions": 0, "total_fees": 0.0, "total_funding": 0.0},
            "stress": {"initial_equity": fixed_ledger.INITIAL_EQUITY, "final_equity": fixed_ledger.INITIAL_EQUITY, "absolute_return_pct": 0.0, "cagr_pct": 0.0, "strict_mdd_pct": 0.0, "cagr_to_strict_mdd": 0.0, "intervals": 0, "long_intervals": 0, "short_intervals": 0, "mean_exposure_weighted_gross_edge_bp": 0.0, "transitions": 0, "total_fees": 0.0, "total_funding": 0.0},
            "cluster_signflip": {"pvalue": 1.0, "clusters": 0},
            "calendar_halves": {"first": {"absolute_return_pct": 0.0}, "second": {"absolute_return_pct": 0.0}},
        }
        activity = {
            "intervals": 0,
            "active_iso_weeks": 0,
            "aggregate_net_signed_episodes": 0,
        }
        return {
            "primary": empty_primary,
            "net_position_risk": net_config.net_exposure_metrics(clock, start, end),
            "legacy_nonnet_risk_disclosure": {},
            "operating_cost_disclosure": {},
            "activity": activity,
            "retained_activity_sufficiency": {
                **activity,
                "applied_as_rejection": True,
            },
            "waived_cost_and_max_frequency_disclosures": {
                "applied_as_rejection": False,
            },
        }, {}
    primary, base_raw = evaluate_primary_without_candidate_mc(
        clock,
        market,
        funding,
        start,
        end,
    )
    legacy_risk = original_evaluator.optimizer.exposure_and_turnover(clock, start, end)
    net_risk = net_config.net_exposure_metrics(clock, start, end)
    activity = {
        "intervals": int(primary["base"]["intervals"]),
        "long_intervals": int(primary["base"]["long_intervals"]),
        "short_intervals": int(primary["base"]["short_intervals"]),
        "active_iso_weeks": int(original_evaluator.active_iso_weeks(clock)),
        "aggregate_net_signed_episodes": int(original_evaluator.aggregate_net_signed_episode_count(clock)),
    }
    operating_cost = original_evaluator.operating_cost_disclosure(
        clock,
        base_raw,
        primary,
        legacy_risk,
        start,
        end,
    )
    return {
        "primary": primary,
        "net_position_risk": net_risk,
        "legacy_nonnet_risk_disclosure": legacy_risk,
        "operating_cost_disclosure": operating_cost,
        "activity": activity,
        "retained_activity_sufficiency": {
            "aggregate_net_signed_episodes": activity["aggregate_net_signed_episodes"],
            "intervals": activity["intervals"],
            "active_iso_weeks": activity["active_iso_weeks"],
            "applied_as_rejection": True,
        },
        "waived_cost_and_max_frequency_disclosures": {
            "turnover": legacy_risk,
            "nonzero_net_execution_events_per_day": operating_cost[
                "nonzero_net_execution_events_per_day"
            ],
            "applied_as_rejection": False,
        },
    }, weekly_effect_vector(base_raw["equity_effect_rows"])


def rank_candidates(candidate_reports: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate_id, report in candidate_reports.items():
        stages = report["stages"]
        stage_checks = report["checks"]
        qualifier = all(all(checks.values()) for checks in stage_checks.values())
        test_primary = stages["test2024"]["primary"]
        eval_primary = stages["eval2025"]["primary"]
        min_ratio = min(float(test_primary["base"]["cagr_to_strict_mdd"]), float(eval_primary["base"]["cagr_to_strict_mdd"]))
        summed_log_return = log_return_from_primary(test_primary) + log_return_from_primary(eval_primary)
        worst_mdd = max(float(test_primary["base"]["strict_mdd_pct"]), float(eval_primary["base"]["strict_mdd_pct"]))
        rows.append({"candidate_id": candidate_id, "qualified": qualifier, "min_test_eval_cagr_to_strict_mdd": min_ratio, "summed_log_final_equity_return": summed_log_return, "worst_strict_mdd_pct": worst_mdd})
    rows.sort(key=lambda row: (not row["qualified"], -row["min_test_eval_cagr_to_strict_mdd"], -row["summed_log_final_equity_return"], row["worst_strict_mdd_pct"], row["candidate_id"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def verify_known_rank1_replay(
    freeze: Mapping[str, Any],
    candidate_reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_id = freeze["known_outcome_boundary"]["known_current_rank1_candidate_id"]
    report = candidate_reports[candidate_id]
    test_artifact = freeze_mod.load_hashed_json(
        freeze["frozen_inputs"]["test2024_terminal_artifact"]["path"]
    )
    eval_artifact = freeze_mod.load_hashed_json(
        freeze["frozen_inputs"]["current_rank1_eval2025_diagnostic"]["path"]
    )
    def comparable_primary(primary: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "base": primary["base"],
            "stress": primary["stress"],
            "calendar_halves": primary["calendar_halves"],
        }

    observed = {
        "test2024": canonical_hash(
            comparable_primary(report["stages"]["test2024"]["primary"])
        ),
        "eval2025": canonical_hash(
            comparable_primary(report["stages"]["eval2025"]["primary"])
        ),
    }
    expected = {
        "test2024": canonical_hash(comparable_primary(test_artifact["primary"])),
        "eval2025": canonical_hash(comparable_primary(eval_artifact["primary"])),
    }
    if observed != expected:
        raise RuntimeError(f"{POLICY_ID} known rank1 replay drift")
    return {
        "candidate_id": candidate_id,
        "primary_metrics_sha256": observed,
        "exact_match": True,
    }


def run(output: str | Path | None = None, *, freeze_path: Path = FREEZE) -> dict[str, Any]:
    freeze = load_freeze(freeze_path)
    chain = verify_terminal_override_chain(freeze)
    sources = load_all_stage_sources(freeze)
    sleeve_clocks = load_sleeve_clocks_once(freeze)
    candidate_reports: dict[str, dict[str, Any]] = {}
    weekly_by_stage: dict[str, dict[str, Mapping[tuple[int, int], float]]] = {stage: {} for stage in freeze["stages"]}

    known_candidate_id = freeze["known_outcome_boundary"][
        "known_current_rank1_candidate_id"
    ]
    family_by_id = {row["candidate_id"]: row for row in freeze["candidate_family"]}
    if known_candidate_id not in family_by_id:
        raise RuntimeError(f"{POLICY_ID} known rank1 missing from candidate family")
    ordered_family = [family_by_id[known_candidate_id]] + [
        row for row in freeze["candidate_family"] if row["candidate_id"] != known_candidate_id
    ]
    known_rank1_replay: dict[str, Any] | None = None
    for candidate_index, candidate in enumerate(ordered_family, start=1):
        candidate_id = candidate["candidate_id"]
        candidate_reports[candidate_id] = {
            "candidate_id": candidate_id,
            "kind": candidate["kind"],
            "weights": candidate["weights"],
            "candidate_provenance": {
                key: value
                for key, value in candidate.items()
                if key not in {"candidate_id", "kind", "weights"}
            },
            "stages": {},
            "checks": {},
        }
        for stage in freeze["stages"]:
            clock = candidate_clock(candidate, sleeve_clocks, stage)
            metrics, weekly = evaluate_candidate_stage(candidate, clock, sources[stage])
            candidate_reports[candidate_id]["stages"][stage] = metrics
            weekly_by_stage[stage][candidate_id] = weekly
        if candidate_id == known_candidate_id:
            known_rank1_replay = verify_known_rank1_replay(freeze, candidate_reports)
        if candidate_index % 10 == 0 or candidate_index == len(freeze["candidate_family"]):
            print(
                json.dumps(
                    {
                        "progress_candidates": candidate_index,
                        "candidate_total": len(freeze["candidate_family"]),
                    }
                ),
                flush=True,
            )

    max_t_by_stage = {stage: aligned_week_max_t(weekly) for stage, weekly in weekly_by_stage.items()}
    for candidate_id, report in candidate_reports.items():
        for stage in freeze["stages"]:
            metrics = report["stages"][stage]
            pvalue = max_t_by_stage[stage]["adjusted_pvalues"].get(candidate_id, 1.0)
            metrics["max_t_aligned_week_pvalue"] = pvalue
            checks = retained_checks(
                metrics["primary"],
                metrics["net_position_risk"],
                metrics["activity"]["intervals"],
                metrics["activity"]["active_iso_weeks"],
                metrics["activity"]["aggregate_net_signed_episodes"],
                pvalue,
            )
            report["checks"][stage] = checks
            report["stages"][stage]["passed_retained_checks"] = all(checks.values())
    if known_rank1_replay is None:
        raise RuntimeError(f"{POLICY_ID} known rank1 replay was not verified")
    ranking = rank_candidates(candidate_reports)
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "source_policy_id": freeze["source_policy_id"],
        "freeze": freeze_receipt(freeze_path, freeze),
        "override_chain": chain,
        "stages": freeze["stages"],
        "source_receipts": {stage: bundle["source"] for stage, bundle in sources.items()},
        "known_outcome_boundary": freeze["known_outcome_boundary"],
        "known_rank1_replay_consistency": known_rank1_replay,
        "candidate_count": len(candidate_reports),
        "candidate_metrics": candidate_reports,
        "max_t_by_stage": max_t_by_stage,
        "ranking": ranking,
        "qualifier_count": sum(1 for row in ranking if row["qualified"]),
        "ranking_rule": freeze["ranking_rule"],
        "waived_rejection_gates": freeze["gate_policy"]["waived_rejection_gates"],
        "final2026_opened": False,
        "diagnostic_post_outcome_ordering": True,
        "reranked_for_diagnostic_reporting": True,
        "selection_or_promotion_authorized": False,
        "repaired_or_substituted": False,
        "live_capital_authorized": False,
        "order_submission_enabled": False,
        "decision": "diagnostic_test2024_eval2025_revalidation_complete_no_final2026",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=FREEZE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_only:
        freeze = load_freeze(args.freeze)
        chain = verify_terminal_override_chain(freeze)
        print(json.dumps({"policy_id": POLICY_ID, "verified": True, "freeze": freeze_receipt(args.freeze, freeze), "override_chain": chain, "outcomes_opened": False}, ensure_ascii=False))
        return 0
    result = run(args.output, freeze_path=args.freeze)
    print(json.dumps({"policy_id": POLICY_ID, "output": str(args.output), "candidate_count": result["candidate_count"], "qualifier_count": result["qualifier_count"], "final2026_opened": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
