"""Pure strict report-only transfer evaluation for the sealed PSIM family."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from training import bctp_strict_economics as economics
from training import (
    preregister_psim_d8_rllm2_s7_2021_report_only_transfer as prereg,
)

SCHEDULE_COLUMNS = ("policy_id", "sequence_id", "entry_time", "target")
ACTION_TARGET = {
    "TARGET_FLAT": 0.0,
    "TARGET_SHORT": -0.5,
    "TARGET_LONG": 0.5,
}
RATIO_MDD_FLOOR = 1e-12
FIVE_MINUTES = pd.Timedelta(minutes=5)


@dataclass(frozen=True)
class StatisticalConfig:
    draws: int = prereg.STATISTICAL_DRAWS
    seed: int = prereg.STATISTICAL_SEED
    batch_draws: int = prereg.STATISTICAL_BATCH_DRAWS


def _iso_z(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None or pd.isna(timestamp):
        raise ValueError("PSIM S7 timestamp must be timezone aware")
    return timestamp.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def combine_schedule_family(
    s4: pd.DataFrame,
    s5: pd.DataFrame,
    s6r1: pd.DataFrame,
    *,
    expected_rows_per_policy: int = 365,
) -> pd.DataFrame:
    groups = (
        (s4, prereg.s4.POLICY_FAMILY_IDS),
        (s5, prereg.s5.POLICY_FAMILY_IDS),
        (s6r1, prereg.s6r1.POLICY_FAMILY_IDS),
    )
    normalized: list[pd.DataFrame] = []
    for frame, expected_ids in groups:
        if tuple(frame.columns) != SCHEDULE_COLUMNS:
            raise ValueError("PSIM S7 schedule columns changed")
        observed_ids = tuple(frame["policy_id"].drop_duplicates())
        if observed_ids != expected_ids:
            raise ValueError("PSIM S7 schedule family order changed")
        counts = frame.groupby("policy_id", sort=False).size().to_dict()
        if counts != {
            policy_id: expected_rows_per_policy
            for policy_id in expected_ids
        }:
            raise ValueError("PSIM S7 schedule family rows changed")
        normalized.append(frame.copy())
    combined = pd.concat(normalized, ignore_index=True)
    if (
        tuple(combined["policy_id"].drop_duplicates()) != prereg.FAMILY_IDS
        or combined.duplicated(["policy_id", "sequence_id"]).any()
        or not combined["target"].isin(ACTION_TARGET).all()
    ):
        raise RuntimeError("PSIM S7 combined schedule family changed")
    return combined


def validate_delayed_primary(
    combined: pd.DataFrame,
    delayed: pd.DataFrame,
    *,
    expected_rows: int = 365,
) -> None:
    if (
        tuple(delayed.columns) != SCHEDULE_COLUMNS
        or len(delayed) != expected_rows
        or set(delayed["policy_id"]) != {prereg.PRIMARY_POLICY_ID}
        or delayed["sequence_id"].duplicated().any()
        or not delayed["target"].isin(ACTION_TARGET).all()
    ):
        raise ValueError("PSIM S7 delayed primary changed")
    primary = (
        combined.loc[
            combined["policy_id"].eq(prereg.PRIMARY_POLICY_ID)
        ]
        .sort_values("entry_time")
        .reset_index(drop=True)
    )
    shifted = delayed.sort_values("entry_time").reset_index(drop=True)
    if (
        shifted["sequence_id"].tolist()
        != [f"{value}:delay_5m" for value in primary["sequence_id"]]
        or not shifted["target"].equals(primary["target"])
        or not (
            pd.to_datetime(shifted["entry_time"], utc=True)
            == pd.to_datetime(primary["entry_time"], utc=True)
            + pd.Timedelta(minutes=5)
        ).all()
    ):
        raise ValueError("PSIM S7 delayed schedule identity changed")


def _policy_schedule(
    combined: pd.DataFrame,
    policy_id: str,
) -> pd.DataFrame:
    selected = combined.loc[
        combined["policy_id"].eq(policy_id),
        list(SCHEDULE_COLUMNS),
    ].copy()
    if selected.empty:
        raise ValueError(f"PSIM S7 policy schedule missing: {policy_id}")
    return selected.sort_values("entry_time").reset_index(drop=True)


def _trade_counts(
    schedule: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, int]:
    times = pd.to_datetime(schedule["entry_time"], utc=True)
    targets = schedule.loc[
        times.ge(start) & times.lt(end),
        "target",
    ].astype(str)
    prior = "TARGET_FLAT"
    directional_entries = 0
    target_changes = 0
    for target in targets:
        if target == prior:
            continue
        target_changes += 1
        if target != "TARGET_FLAT":
            directional_entries += 1
        prior = target
    if prior != "TARGET_FLAT":
        target_changes += 1
    return {
        "directional_entries_including_flips": directional_entries,
        "all_target_changes_including_terminal_flatten": target_changes,
    }


def _require_finite(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"PSIM S7 {name} is not numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"PSIM S7 {name} is not finite")
    return number


def _summarize(
    simulation: Mapping[str, Any],
    schedule: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    intervals = list(simulation["intervals"])
    long_count = sum(float(row["target"]) > 0.0 for row in intervals)
    short_count = sum(float(row["target"]) < 0.0 for row in intervals)
    nonflat = long_count + short_count
    final_equity = _require_finite(
        simulation["final_equity"],
        name="final equity",
    )
    cagr = _require_finite(simulation["cagr"], name="CAGR")
    strict_mdd = _require_finite(
        simulation["max_drawdown"],
        name="strict MDD",
    )
    if strict_mdd < 0.0:
        raise ValueError("PSIM S7 strict MDD is negative")
    ratio = cagr / max(strict_mdd, RATIO_MDD_FLOOR)
    return {
        "start": _iso_z(start),
        "end": _iso_z(end),
        "cost_rate": float(simulation["cost_rate"]),
        "absolute_return": final_equity - 1.0,
        "absolute_return_pct": 100.0 * (final_equity - 1.0),
        "cagr": cagr,
        "cagr_pct": 100.0 * cagr,
        "strict_mdd": strict_mdd,
        "strict_mdd_pct": 100.0 * strict_mdd,
        "cagr_to_strict_mdd": _require_finite(
            ratio,
            name="CAGR to strict MDD",
        ),
        "final_equity": final_equity,
        "nonflat_interval_count": nonflat,
        "long_interval_count": long_count,
        "short_interval_count": short_count,
        "long_share_of_nonflat": long_count / nonflat if nonflat else 0.0,
        "short_share_of_nonflat": short_count / nonflat if nonflat else 0.0,
        **_trade_counts(schedule, start=start, end=end),
        "weekly_log_returns": [
            {
                "week_start": str(row["week_start"]),
                "start": str(row["start"]),
                "end": str(row["end"]),
                "log_return": float(row["log_return"]),
            }
            for row in simulation["weekly_log_returns"]
        ],
    }


def evaluate_one(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost_rate: float,
) -> dict[str, Any]:
    simulation = economics.simulate_target_schedule(
        market,
        funding,
        schedule,
        start=start,
        end=end,
        cost_rate=cost_rate,
    )
    return _summarize(
        simulation,
        schedule,
        start=start,
        end=end,
    )


def _slice_half_open(
    frame: pd.DataFrame,
    *,
    kind: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if isinstance(frame.index, pd.DatetimeIndex):
        if frame.index.tz is None:
            raise ValueError(f"PSIM S7 {kind} timestamps must be aware")
        times = pd.Series(frame.index.tz_convert("UTC"), index=frame.index)
    else:
        aliases = {
            "market": ("timestamp", "open_time", "time", "date"),
            "funding": (
                "timestamp",
                "funding_time",
                "funding_time_utc",
                "time",
                "date",
            ),
            "schedule": ("timestamp", "entry_time", "time", "date"),
        }
        lowered = {str(column).lower(): column for column in frame.columns}
        column = next(
            (
                lowered[name]
                for name in aliases[kind]
                if name in lowered
            ),
            None,
        )
        if column is None:
            raise ValueError(f"PSIM S7 {kind} timestamp column missing")
        parsed: list[pd.Timestamp] = []
        for value in frame[column]:
            timestamp = pd.Timestamp(value)
            if timestamp.tzinfo is None or pd.isna(timestamp):
                raise ValueError(
                    f"PSIM S7 {kind} timestamps must be aware"
                )
            parsed.append(timestamp.tz_convert("UTC"))
        times = pd.Series(parsed, index=frame.index)
    return frame.loc[times.ge(start) & times.lt(end)].copy()


def _studentized_mean(values: np.ndarray) -> float:
    deviation = float(values.std(ddof=1))
    if deviation <= 0.0:
        return float("nan")
    return math.sqrt(len(values)) * float(values.mean()) / deviation


def shared_weekly_max_stat(
    weekly_returns: Mapping[str, Sequence[tuple[str, float]]],
    *,
    cfg: StatisticalConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or StatisticalConfig()
    if tuple(weekly_returns) != prereg.FAMILY_IDS:
        raise ValueError("PSIM S7 weekly family order changed")
    if cfg.draws <= 0 or cfg.batch_draws <= 0:
        raise ValueError("PSIM S7 statistical draws changed")
    week_keys: tuple[str, ...] | None = None
    rows: list[np.ndarray] = []
    for policy_id in prereg.FAMILY_IDS:
        observations = tuple(weekly_returns[policy_id])
        keys = tuple(str(key) for key, _ in observations)
        values = np.asarray([value for _, value in observations], dtype=float)
        if (
            len(keys) < 2
            or len(set(keys)) != len(keys)
            or keys != tuple(sorted(keys))
            or not np.isfinite(values).all()
        ):
            raise ValueError("PSIM S7 weekly observations changed")
        if week_keys is None:
            week_keys = keys
        elif keys != week_keys:
            raise ValueError("PSIM S7 weekly keys are misaligned")
        rows.append(values)
    assert week_keys is not None
    matrix = np.vstack(rows)
    observed = np.asarray(
        [_studentized_mean(row) for row in matrix],
        dtype=np.float64,
    )
    local_exceed = np.zeros(len(prereg.FAMILY_IDS), dtype=np.int64)
    max_exceed = np.zeros(len(prereg.FAMILY_IDS), dtype=np.int64)
    generator = np.random.default_rng(cfg.seed)
    completed = 0
    while completed < cfg.draws:
        current = min(cfg.batch_draws, cfg.draws - completed)
        signs = generator.choice(
            (-1.0, 1.0),
            size=(current, matrix.shape[1]),
        )
        signed = signs[:, None, :] * matrix[None, :, :]
        means = signed.mean(axis=2)
        deviations = signed.std(axis=2, ddof=1)
        null_t = np.full_like(means, float("-inf"))
        np.divide(
            math.sqrt(matrix.shape[1]) * means,
            deviations,
            out=null_t,
            where=deviations > 0.0,
        )
        family_max = null_t.max(axis=1)
        for index, statistic in enumerate(observed):
            if not math.isfinite(statistic):
                continue
            local_exceed[index] += int(
                np.count_nonzero(null_t[:, index] >= statistic - 1e-15)
            )
            max_exceed[index] += int(
                np.count_nonzero(family_max >= statistic - 1e-15)
            )
        completed += current
    denominator = cfg.draws + 1
    local = {}
    adjusted = {}
    for index, policy_id in enumerate(prereg.FAMILY_IDS):
        if not math.isfinite(observed[index]):
            local[policy_id] = 1.0
            adjusted[policy_id] = 1.0
        else:
            local[policy_id] = float(
                (local_exceed[index] + 1) / denominator
            )
            adjusted[policy_id] = float(
                (max_exceed[index] + 1) / denominator
            )
            if adjusted[policy_id] + 1e-15 < local[policy_id]:
                raise RuntimeError("PSIM S7 adjusted p below local p")
    return {
        "method": "monte_carlo_shared_sign_rademacher_max_stat",
        "draws": cfg.draws,
        "seed": cfg.seed,
        "weeks": matrix.shape[1],
        "week_keys": list(week_keys),
        "family_ids": list(prereg.FAMILY_IDS),
        "observed_t": {
            policy_id: (
                float(observed[index])
                if math.isfinite(observed[index])
                else None
            )
            for index, policy_id in enumerate(prereg.FAMILY_IDS)
        },
        "local_p": local,
        "p_max": adjusted,
    }


def _canonical_schedule_bytes(frame: pd.DataFrame) -> bytes:
    output = frame.loc[:, ["sequence_id", "entry_time", "target"]].copy()
    output["sequence_id"] = output["sequence_id"].astype(str)
    output["entry_time"] = [_iso_z(value) for value in output["entry_time"]]
    output["target"] = output["target"].astype(str)
    return output.to_csv(index=False, lineterminator="\n").encode("utf-8")


def evaluate_transfer(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    s4_schedules: pd.DataFrame,
    s5_schedules: pd.DataFrame,
    s6r1_schedules: pd.DataFrame,
    delayed_primary: pd.DataFrame,
    *,
    start: pd.Timestamp | None = None,
    half_split: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    expected_rows_per_policy: int = 365,
    statistical_config: StatisticalConfig | None = None,
) -> dict[str, Any]:
    start = pd.Timestamp(
        prereg.CALENDAR_START if start is None else start
    ).tz_convert("UTC")
    half_split = pd.Timestamp(
        prereg.HALF_SPLIT if half_split is None else half_split
    ).tz_convert("UTC")
    end = pd.Timestamp(
        prereg.CALENDAR_END if end is None else end
    ).tz_convert("UTC")
    if not start < half_split < end:
        raise ValueError("PSIM S7 calendar split changed")
    combined = combine_schedule_family(
        s4_schedules,
        s5_schedules,
        s6r1_schedules,
        expected_rows_per_policy=expected_rows_per_policy,
    )
    validate_delayed_primary(
        combined,
        delayed_primary,
        expected_rows=expected_rows_per_policy,
    )
    base_metrics: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for policy_id in prereg.FAMILY_IDS:
        base_metrics[policy_id] = evaluate_one(
            market,
            funding,
            _policy_schedule(combined, policy_id),
            start=start,
            end=end,
            cost_rate=prereg.BASE_COST_RATE,
        )
    weekly = OrderedDict(
        (
            policy_id,
            [
                (row["week_start"], row["log_return"])
                for row in metrics["weekly_log_returns"]
            ],
        )
        for policy_id, metrics in base_metrics.items()
    )
    max_stat = shared_weekly_max_stat(weekly, cfg=statistical_config)
    primary_schedule = _policy_schedule(combined, prereg.PRIMARY_POLICY_ID)
    primary = base_metrics[prereg.PRIMARY_POLICY_ID]
    stress = evaluate_one(
        market,
        funding,
        primary_schedule,
        start=start,
        end=end,
        cost_rate=prereg.STRESS_COST_RATE,
    )
    delayed = evaluate_one(
        market,
        funding,
        delayed_primary,
        start=start,
        end=end,
        cost_rate=prereg.BASE_COST_RATE,
    )
    first_market = _slice_half_open(
        market,
        kind="market",
        start=start,
        end=half_split,
    )
    first_funding = _slice_half_open(
        funding,
        kind="funding",
        start=start,
        end=half_split,
    )
    first_schedule = _slice_half_open(
        primary_schedule,
        kind="schedule",
        start=start,
        end=half_split - FIVE_MINUTES,
    )
    second_market = _slice_half_open(
        market,
        kind="market",
        start=half_split,
        end=end,
    )
    second_funding = _slice_half_open(
        funding,
        kind="funding",
        start=half_split,
        end=end,
    )
    second_schedule = _slice_half_open(
        primary_schedule,
        kind="schedule",
        start=half_split,
        end=end - FIVE_MINUTES,
    )
    first_half = evaluate_one(
        first_market,
        first_funding,
        first_schedule,
        start=start,
        end=half_split,
        cost_rate=prereg.BASE_COST_RATE,
    )
    second_half = evaluate_one(
        second_market,
        second_funding,
        second_schedule,
        start=half_split,
        end=end,
        cost_rate=prereg.BASE_COST_RATE,
    )
    strongest_control_id = min(
        prereg.NONSEMANTIC_CONTROL_IDS,
        key=lambda policy_id: (
            -base_metrics[policy_id]["cagr_to_strict_mdd"],
            -base_metrics[policy_id]["absolute_return"],
            base_metrics[policy_id]["strict_mdd"],
            policy_id,
        ),
    )
    strongest_control = base_metrics[strongest_control_id]
    permutation = _policy_schedule(
        combined,
        f"{prereg.PRIMARY_POLICY_ID}_action_code_permutation",
    )
    action_identity = (
        _canonical_schedule_bytes(primary_schedule)
        == _canonical_schedule_bytes(permutation)
    )
    checks = OrderedDict(
        (
            ("base_absolute_return_positive", primary["absolute_return"] > 0),
            (
                "stress_absolute_return_positive",
                stress["absolute_return"] > 0,
            ),
            (
                "delayed_absolute_return_positive",
                delayed["absolute_return"] > 0,
            ),
            (
                "first_half_absolute_return_positive",
                first_half["absolute_return"] > 0,
            ),
            (
                "second_half_absolute_return_positive",
                second_half["absolute_return"] > 0,
            ),
            (
                "base_cagr_to_strict_mdd_minimum",
                primary["cagr_to_strict_mdd"] >= prereg.MIN_RATIO,
            ),
            (
                "minimum_nonflat_intervals",
                primary["nonflat_interval_count"]
                >= prereg.MIN_NONFLAT_INTERVALS,
            ),
            (
                "minimum_long_share",
                primary["long_share_of_nonflat"]
                >= prereg.MIN_DIRECTION_SHARE,
            ),
            (
                "minimum_short_share",
                primary["short_share_of_nonflat"]
                >= prereg.MIN_DIRECTION_SHARE,
            ),
            (
                "beat_strongest_nonsemantic_absolute_return",
                primary["absolute_return"]
                > strongest_control["absolute_return"],
            ),
            (
                "beat_strongest_nonsemantic_cagr_to_strict_mdd",
                primary["cagr_to_strict_mdd"]
                > strongest_control["cagr_to_strict_mdd"],
            ),
            ("action_code_permutation_schedule_identity", action_identity),
            (
                "familywise_p_max_strictly_below",
                max_stat["p_max"][prereg.PRIMARY_POLICY_ID]
                < prereg.FAMILYWISE_P_MAX_EXCLUSIVE,
            ),
        )
    )
    return {
        "protocol_version": "psim_s7_report_only_transfer_calculation_v1",
        "stage": "2021",
        "calendar": {
            "start": _iso_z(start),
            "half_split": _iso_z(half_split),
            "end": _iso_z(end),
        },
        "robustness_semantics": {
            "half_metrics": (
                "standalone_reset_to_flat_equity_1_at_each_half_start"
            ),
            "continuous_full_path_subperiod_attribution": False,
        },
        "primary_policy_id": prereg.PRIMARY_POLICY_ID,
        "family_ids": list(prereg.FAMILY_IDS),
        "base_family_metrics": base_metrics,
        "familywise_max_stat": max_stat,
        "primary_metrics": {
            "base_6bp": primary,
            "stress_10bp": stress,
            "delayed_5m_6bp": delayed,
            "first_half_6bp": first_half,
            "second_half_6bp": second_half,
        },
        "strongest_nonsemantic_control": {
            "policy_id": strongest_control_id,
            "metrics": strongest_control,
        },
        "action_code_permutation_schedule_identity": action_identity,
        "gate": {
            "checks": checks,
            "passed": all(checks.values()),
        },
    }
