"""Pure BCTP transfer-stage family evaluator.

This module evaluates already-sealed, already-loaded BCTP transfer schedules on
caller-supplied stage-local market/funding frames.  It intentionally performs no
payload I/O, artifact writes, or payload hashing.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import math
from typing import Any, Literal

import pandas as pd

from training import bctp_stage_sources as stage_sources
from training import bctp_strict_economics as economics
from training import freeze_block_clearing_target_position_evaluator as freeze

Stage = Literal["2021", "2022"]

BASE_COST_RATE = 0.0006
STRESS_COST_RATE = 0.0010
RATIO_MDD_FLOOR = 1e-12
PRIMARY_IDS = tuple(stage_sources.PROMOTABLE_PRIMARY_IDS)
CONTROL_SUFFIXES = (
    "current_only",
    "reversed_sequence",
    "masked_source",
    "shuffled_reward",
    "circular_21_reward",
)


def _iso_z(timestamp: pd.Timestamp) -> str:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        raise ValueError("BCTP evaluator timestamps must be timezone aware")
    return ts.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _finite(value: Any, *, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    return value


def _validate_stage(stage: str) -> Stage:
    if stage not in ("2021", "2022"):
        raise ValueError("BCTP family evaluator supports transfer stages 2021 and 2022")
    return stage  # type: ignore[return-value]


def _stage_bounds(
    stage: str,
    *,
    calendar_start: pd.Timestamp | None,
    calendar_end: pd.Timestamp | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if calendar_start is None and calendar_end is None:
        spec = stage_sources.STAGE_SPECS[stage]
        return spec.start, spec.end
    if calendar_start is None or calendar_end is None:
        raise ValueError("BCTP compact-calendar evaluation requires both start and end")
    start = pd.Timestamp(calendar_start)
    end = pd.Timestamp(calendar_end)
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("BCTP compact-calendar bounds must be timezone aware")
    start = start.tz_convert("UTC")
    end = end.tz_convert("UTC")
    if end <= start + economics.FIVE_MINUTES:
        raise ValueError("BCTP compact calendar must include a final flat bar")
    return start, end


def _empty_schedule() -> pd.DataFrame:
    return pd.DataFrame(columns=stage_sources.SCHEDULE_COLUMNS)


def _split_schedules(frame: pd.DataFrame, ids: tuple[str, ...]) -> OrderedDict[str, pd.DataFrame]:
    if frame is None:
        frame = _empty_schedule()
    if tuple(frame.columns) != stage_sources.SCHEDULE_COLUMNS:
        raise ValueError("BCTP schedule combined frame schema mismatch")
    policy_ids = frame["policy_id"].astype(str)
    unknown = sorted(set(policy_ids) - set(ids))
    if unknown:
        raise ValueError(f"BCTP schedule contains unknown policy ids: {unknown}")
    observed_groups = tuple(
        policy_ids.loc[policy_ids.ne(policy_ids.shift())].tolist()
    )
    expected_groups = tuple(policy_id for policy_id in ids if policy_id in set(policy_ids))
    if observed_groups != expected_groups:
        raise ValueError("BCTP schedule combined policy order mismatch")
    out: OrderedDict[str, pd.DataFrame] = OrderedDict()
    for policy_id in ids:
        rows = frame.loc[frame["policy_id"].astype(str).eq(policy_id), list(stage_sources.SCHEDULE_COLUMNS)].copy()
        out[policy_id] = rows.reset_index(drop=True)
    return out


def _canonical_schedule_bytes(frame: pd.DataFrame) -> bytes:
    comparable = frame.loc[:, ["sequence_id", "entry_time", "target"]].copy()
    comparable["sequence_id"] = comparable["sequence_id"].astype(str)
    comparable["target"] = comparable["target"].astype(str)
    comparable["entry_time"] = [
        _iso_z(pd.Timestamp(value)) for value in comparable["entry_time"].tolist()
    ]
    return comparable.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _zero_weekly_returns(start: pd.Timestamp, end: pd.Timestamp) -> list[dict[str, Any]]:
    # Full-calendar Monday-Z intervals, matching bctp_strict_economics weekly keys.
    start = pd.Timestamp(start).tz_convert("UTC")
    end = pd.Timestamp(end).tz_convert("UTC")
    boundaries = [start]
    first_monday = (start - pd.Timedelta(days=start.weekday())).normalize()
    if first_monday <= start:
        first_monday += pd.Timedelta(days=7)
    current = first_monday
    while current < end:
        boundaries.append(current)
        current += pd.Timedelta(days=7)
    boundaries.append(end)
    rows: list[dict[str, Any]] = []
    for left, right in zip(boundaries, boundaries[1:]):
        if left == right:
            continue
        rows.append(
            {
                "week_start": _iso_z((left - pd.Timedelta(days=left.weekday())).normalize()),
                "start": _iso_z(left),
                "end": _iso_z(right),
                "log_return": 0.0,
            }
        )
    return rows


def _zero_metrics(start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    return {
        "start": _iso_z(start),
        "end": _iso_z(end),
        "cost_rate": BASE_COST_RATE,
        "absolute_return": 0.0,
        "total_return_pct": 0.0,
        "cagr": 0.0,
        "strict_mdd": 0.0,
        "cagr_to_strict_mdd": 0.0,
        "nonflat_interval_count": 0,
        "long_interval_count": 0,
        "short_interval_count": 0,
        "long_share_of_nonflat": 0.0,
        "short_share_of_nonflat": 0.0,
        "final_equity": 1.0,
        "weekly_log_returns": _zero_weekly_returns(start, end),
    }


def _summarise_simulation(result: Mapping[str, Any]) -> dict[str, Any]:
    intervals = list(result.get("intervals", []))
    long_count = 0
    short_count = 0
    for interval in intervals:
        target = _finite(interval.get("target", interval.get("new_target", 0.0)))
        if target > 0.0:
            long_count += 1
        elif target < 0.0:
            short_count += 1
    nonflat = long_count + short_count
    final_equity = _finite(result.get("final_equity"), default=1.0)
    absolute_return = final_equity - 1.0
    cagr = _finite(result.get("cagr"))
    mdd = max(0.0, _finite(result.get("max_drawdown")))
    ratio = cagr / max(mdd, RATIO_MDD_FLOOR)
    return {
        "start": str(result.get("start")),
        "end": str(result.get("end")),
        "cost_rate": _finite(result.get("cost_rate")),
        "absolute_return": float(absolute_return),
        "total_return_pct": float(100.0 * absolute_return),
        "cagr": float(cagr),
        "strict_mdd": float(mdd),
        "cagr_to_strict_mdd": float(_finite(ratio)),
        "nonflat_interval_count": int(nonflat),
        "long_interval_count": int(long_count),
        "short_interval_count": int(short_count),
        "long_share_of_nonflat": float(long_count / nonflat) if nonflat else 0.0,
        "short_share_of_nonflat": float(short_count / nonflat) if nonflat else 0.0,
        "final_equity": float(final_equity),
        "weekly_log_returns": [
            {
                "week_start": str(row["week_start"]),
                "start": str(row["start"]),
                "end": str(row["end"]),
                "log_return": _finite(row.get("log_return")),
            }
            for row in result.get("weekly_log_returns", [])
        ],
    }


def _evaluate_one(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost_rate: float,
) -> dict[str, Any]:
    raw = economics.simulate_target_schedule(
        market,
        funding,
        schedule,
        start=start,
        end=end,
        cost_rate=cost_rate,
    )
    metrics = _summarise_simulation(raw)
    metrics["cost_rate"] = float(cost_rate)
    return metrics


def _control_ids(primary_id: str) -> tuple[str, ...]:
    return (
        "exact_signature_memory",
        *(f"{primary_id}_{suffix}" for suffix in CONTROL_SUFFIXES),
    )


def _beats_controls(primary_id: str, base_metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    primary = base_metrics[primary_id]
    checks: dict[str, Any] = {}
    all_pass = True
    for control_id in _control_ids(primary_id):
        control = base_metrics[control_id]
        control_valid = "error" not in control
        beats_return = (
            control_valid
            and _finite(primary.get("absolute_return"))
            > _finite(control.get("absolute_return"))
        )
        beats_ratio = (
            control_valid
            and _finite(primary.get("cagr_to_strict_mdd"))
            > _finite(control.get("cagr_to_strict_mdd"))
        )
        passed = bool(control_valid and beats_return and beats_ratio)
        checks[control_id] = {
            "control_valid": bool(control_valid),
            "beats_return": bool(beats_return),
            "beats_cagr_to_strict_mdd": bool(beats_ratio),
            "passed": passed,
        }
        all_pass = all_pass and passed
    return {"passed": bool(all_pass), "controls": checks}


def _gate_thresholds(stage: str) -> dict[str, float]:
    if stage == "2021":
        return {"ratio": 1.0, "pmax": 0.25}
    return {"ratio": 1.5, "pmax": 0.10}


def _candidate_gate(
    primary_id: str,
    *,
    stage: str,
    base: Mapping[str, Any],
    stress: Mapping[str, Any],
    delay: Mapping[str, Any],
    pmax: float,
    control_defeat: Mapping[str, Any],
    action_code_identity: bool,
) -> dict[str, Any]:
    thresholds = _gate_thresholds(stage)
    checks = OrderedDict(
        (
            ("base_absolute_return_positive", _finite(base.get("absolute_return")) > 0.0),
            ("stress_absolute_return_positive", _finite(stress.get("absolute_return")) > 0.0),
            ("delay_absolute_return_positive", _finite(delay.get("absolute_return")) > 0.0),
            ("cagr_to_strict_mdd_minimum", _finite(base.get("cagr_to_strict_mdd")) >= thresholds["ratio"]),
            ("familywise_pmax", _finite(pmax, default=1.0) < thresholds["pmax"]),
            ("minimum_nonflat_intervals", int(base.get("nonflat_interval_count", 0)) >= 80),
            ("minimum_long_share", _finite(base.get("long_share_of_nonflat")) >= 0.20),
            ("minimum_short_share", _finite(base.get("short_share_of_nonflat")) >= 0.20),
            ("beat_required_controls_on_return_and_ratio", bool(control_defeat.get("passed"))),
            ("action_code_permutation_schedule_identity", bool(action_code_identity)),
        )
    )
    return {"passed": bool(all(checks.values())), "checks": checks}


def _winner_key(candidate: Mapping[str, Any]) -> tuple[float, float, float, float, str]:
    metrics = candidate["metrics"]
    minimum_ratio = min(
        _finite(metrics["base"].get("cagr_to_strict_mdd")),
        _finite(metrics["stress_10bp"].get("cagr_to_strict_mdd")),
        _finite(metrics["delayed_exact"].get("cagr_to_strict_mdd")),
    )
    return (
        minimum_ratio,
        _finite(metrics["base"].get("cagr_to_strict_mdd")),
        _finite(metrics["base"].get("absolute_return")),
        -_finite(metrics["base"].get("strict_mdd")),
        "" if not candidate.get("policy_id") else str(candidate["policy_id"]),
    )


def evaluate_bctp_transfer_stage(
    stage: Stage,
    market: pd.DataFrame,
    funding: pd.DataFrame,
    base_schedule_combined: pd.DataFrame,
    delayed_primary_combined: pd.DataFrame,
    *,
    selected_primary_id: str | None = None,
    calendar_start: pd.Timestamp | None = None,
    calendar_end: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Evaluate one sealed BCTP transfer stage on caller-supplied frames.

    The function is pure with respect to artifacts: it opens no payload path,
    hashes no payload bytes, and writes nothing.  ``calendar_start`` and
    ``calendar_end`` are test-only compact-calendar overrides; production calls
    should omit them to use the frozen annual stage bounds.
    """

    stage = _validate_stage(str(stage))
    if stage == "2021" and selected_primary_id is not None:
        raise ValueError("BCTP 2021 must select its primary from the frozen family")
    if stage == "2022" and selected_primary_id not in PRIMARY_IDS:
        raise ValueError(
            "BCTP 2022 selected_primary_id must be a promotable primary id"
        )
    start, end = _stage_bounds(stage, calendar_start=calendar_start, calendar_end=calendar_end)
    base_schedules = _split_schedules(base_schedule_combined, tuple(freeze.FAMILY_IDS))
    delayed_schedules = _split_schedules(delayed_primary_combined, PRIMARY_IDS)
    zero = _zero_metrics(start, end)

    errors: dict[str, list[str]] = {policy_id: [] for policy_id in freeze.FAMILY_IDS}
    base_metrics: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for policy_id in freeze.FAMILY_IDS:
        schedule = base_schedules[policy_id]
        if schedule.empty:
            metric = dict(zero)
            metric["error"] = "missing base schedule"
            errors[policy_id].append("missing base schedule")
        else:
            try:
                metric = _evaluate_one(market, funding, schedule, start=start, end=end, cost_rate=BASE_COST_RATE)
            except Exception as exc:  # keep failed family in familywise matrix
                metric = dict(zero)
                metric["error"] = f"base evaluation failed: {type(exc).__name__}: {exc}"
                errors[policy_id].append(metric["error"])
        base_metrics[policy_id] = metric

    weekly_input: OrderedDict[str, list[tuple[str, float]]] = OrderedDict()
    for policy_id, metric in base_metrics.items():
        weekly_input[policy_id] = [
            (str(row["week_start"]), _finite(row.get("log_return")))
            for row in metric["weekly_log_returns"]
        ]
    try:
        max_stat = freeze.shared_weekly_max_stat(weekly_input)
        pmax_by_policy = {policy_id: _finite(max_stat["p_max"].get(policy_id), default=1.0) for policy_id in freeze.FAMILY_IDS}
        stat_error = None
    except Exception as exc:
        max_stat = {
            "method": "failed",
            "draws": 0,
            "weeks": len(next(iter(weekly_input.values()), [])),
            "week_keys": [key for key, _ in next(iter(weekly_input.values()), [])],
            "family_ids": list(freeze.FAMILY_IDS),
            "observed_t": {policy_id: 0.0 for policy_id in freeze.FAMILY_IDS},
            "local_p": {policy_id: 1.0 for policy_id in freeze.FAMILY_IDS},
            "p_max": {policy_id: 1.0 for policy_id in freeze.FAMILY_IDS},
        }
        pmax_by_policy = {policy_id: 1.0 for policy_id in freeze.FAMILY_IDS}
        stat_error = f"familywise max-stat failed: {type(exc).__name__}: {exc}"

    action_identity: dict[str, bool] = {}
    stress_metrics: dict[str, dict[str, Any]] = {}
    delayed_metrics: dict[str, dict[str, Any]] = {}
    control_defeats: dict[str, Any] = {}
    candidates: OrderedDict[str, dict[str, Any]] = OrderedDict()

    for primary_id in PRIMARY_IDS:
        permutation_id = f"{primary_id}_action_code_permutation"
        try:
            action_identity[primary_id] = (
                _canonical_schedule_bytes(base_schedules[primary_id])
                == _canonical_schedule_bytes(base_schedules[permutation_id])
            )
        except Exception:
            action_identity[primary_id] = False

        try:
            stress_metrics[primary_id] = _evaluate_one(
                market,
                funding,
                base_schedules[primary_id],
                start=start,
                end=end,
                cost_rate=STRESS_COST_RATE,
            )
        except Exception as exc:
            stress_metrics[primary_id] = {**dict(zero), "cost_rate": STRESS_COST_RATE, "error": f"stress evaluation failed: {type(exc).__name__}: {exc}"}
        if delayed_schedules[primary_id].empty:
            delayed_metrics[primary_id] = {**dict(zero), "error": "missing delayed primary schedule"}
        else:
            try:
                delayed_metrics[primary_id] = _evaluate_one(
                    market,
                    funding,
                    delayed_schedules[primary_id],
                    start=start,
                    end=end,
                    cost_rate=BASE_COST_RATE,
                )
            except Exception as exc:
                delayed_metrics[primary_id] = {**dict(zero), "error": f"delayed evaluation failed: {type(exc).__name__}: {exc}"}

        control_defeats[primary_id] = _beats_controls(primary_id, base_metrics)
        gate = _candidate_gate(
            primary_id,
            stage=stage,
            base=base_metrics[primary_id],
            stress=stress_metrics[primary_id],
            delay=delayed_metrics[primary_id],
            pmax=pmax_by_policy[primary_id],
            control_defeat=control_defeats[primary_id],
            action_code_identity=action_identity[primary_id],
        )
        candidates[primary_id] = {
            "policy_id": primary_id,
            "pmax": pmax_by_policy[primary_id],
            "metrics": {
                "base": base_metrics[primary_id],
                "stress_10bp": stress_metrics[primary_id],
                "delayed_exact": delayed_metrics[primary_id],
            },
            "control_defeat": control_defeats[primary_id],
            "action_code_permutation_schedule_identity": action_identity[primary_id],
            "gate": gate,
            "passed": bool(gate["passed"]),
        }

    passing_ids = [policy_id for policy_id, candidate in candidates.items() if candidate["passed"]]
    frozen_winner: str | None
    if stage == "2022":
        assert selected_primary_id is not None
        frozen_winner = (
            selected_primary_id
            if candidates[selected_primary_id]["passed"]
            else None
        )
    elif passing_ids:
        # max uses lexical policy_id as final ascending tie-breaker by inverting explicitly.
        ranked = sorted(
            (candidates[policy_id] for policy_id in passing_ids),
            key=lambda candidate: (
                -_winner_key(candidate)[0],
                -_winner_key(candidate)[1],
                -_winner_key(candidate)[2],
                _finite(candidate["metrics"]["base"].get("strict_mdd")),
                str(candidate["policy_id"]),
            ),
        )
        frozen_winner = str(ranked[0]["policy_id"])
    else:
        frozen_winner = None

    stage_passed = frozen_winner is not None
    result = {
        "protocol_version": "bctp_family_evaluator_v1",
        "pure_calculation": True,
        "payload_io": False,
        "artifact_write": False,
        "payload_bytes_hashed": False,
        "stage": stage,
        "start": _iso_z(start),
        "end": _iso_z(end),
        "family_ids": list(freeze.FAMILY_IDS),
        "promotable_primary_ids": list(PRIMARY_IDS),
        "base_cost_rate": BASE_COST_RATE,
        "stress_cost_rate": STRESS_COST_RATE,
        "base_family_metrics": base_metrics,
        "weekly_log_returns_for_max_stat": weekly_input,
        "familywise_max_stat": max_stat,
        "pmax_by_policy": pmax_by_policy,
        "pmax_combined": {policy_id: pmax_by_policy[policy_id] for policy_id in PRIMARY_IDS},
        "primary_candidates": candidates,
        "passing_candidates": passing_ids,
        "frozen_lexicographic_winner": frozen_winner,
        "stage_passed": stage_passed,
        "selection_rule": [
            "largest_minimum_base_stress_delay_cagr_to_strict_mdd",
            "largest_base_cagr_to_strict_mdd",
            "largest_absolute_return",
            "lower_strict_mdd",
            "lexical_policy_id",
        ],
        "reselection_allowed": stage != "2022",
        "requested_primary_id": selected_primary_id,
        "selected_primary_id": frozen_winner,
        "errors": errors,
    }
    if stat_error is not None:
        result["statistical_error"] = stat_error
    return _json_safe(result)


evaluate_transfer_stage = evaluate_bctp_transfer_stage


__all__ = [
    "evaluate_bctp_transfer_stage",
    "evaluate_transfer_stage",
]
