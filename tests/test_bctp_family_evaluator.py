from __future__ import annotations

import math
from collections import OrderedDict

import pandas as pd
import pytest

from training import bctp_stage_sources as stages
from training import freeze_block_clearing_target_position_evaluator as freeze
from training.bctp_family_evaluator import evaluate_bctp_transfer_stage

FIVE = pd.Timedelta(minutes=5)
START_2021 = pd.Timestamp("2021-01-04T00:00:00Z")
END_2021 = START_2021 + pd.Timedelta(days=22)
START_2022 = pd.Timestamp("2022-01-03T00:00:00Z")
END_2022 = START_2022 + pd.Timedelta(days=22)
DECISION_STEP = pd.Timedelta(hours=6)


def _alpha_target(index: int) -> str:
    return "TARGET_LONG" if index % 2 == 0 else "TARGET_SHORT"


def _flat_frame() -> pd.DataFrame:
    return pd.DataFrame({"timestamp": [], "settlement_mark": [], "funding_rate": []})


def _market(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    periods = int((end - start) / FIVE)
    opens: list[float] = []
    price = 100.0
    bars_per_decision = int(DECISION_STEP / FIVE)
    for i in range(periods):
        opens.append(price)
        decision = i // bars_per_decision
        week = i // int(pd.Timedelta(days=7) / FIVE)
        edge = 0.00011 + 0.000017 * (week % 6)
        if _alpha_target(decision) == "TARGET_LONG":
            price *= 1.0 + edge
        else:
            price *= 1.0 - edge
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=periods, freq="5min", tz="UTC"),
            "open": opens,
            "high": opens,
            "low": opens,
            "close": opens,
        }
    )


def _schedule(policy_id: str, start: pd.Timestamp, end: pd.Timestamp, *, alpha: bool) -> pd.DataFrame:
    terminal = end - FIVE
    rows = []
    current = start
    index = 0
    while current < terminal:
        rows.append(
            {
                "policy_id": policy_id,
                "sequence_id": f"seq-{index:04d}",
                "entry_time": current,
                "target": _alpha_target(index) if alpha else "TARGET_FLAT",
            }
        )
        current += DECISION_STEP
        index += 1
    return pd.DataFrame(rows, columns=stages.SCHEDULE_COLUMNS)


def _combined(start: pd.Timestamp = START_2021, end: pd.Timestamp = END_2021) -> tuple[pd.DataFrame, pd.DataFrame]:
    alpha_ids = set(stages.PROMOTABLE_PRIMARY_IDS)
    alpha_ids.update(f"{pid}_action_code_permutation" for pid in stages.PROMOTABLE_PRIMARY_IDS)
    frames = [
        _schedule(policy_id, start, end, alpha=policy_id in alpha_ids)
        for policy_id in freeze.FAMILY_IDS
    ]
    delayed_frames = []
    terminal = end - FIVE
    for primary_id in stages.PROMOTABLE_PRIMARY_IDS:
        frame = _schedule(primary_id, start, end, alpha=True)
        times = pd.to_datetime(frame["entry_time"], utc=True)
        keep = times + FIVE < terminal
        delayed = frame.loc[keep].copy().reset_index(drop=True)
        delayed["entry_time"] = pd.to_datetime(delayed["entry_time"], utc=True) + FIVE
        delayed["sequence_id"] = delayed["sequence_id"].astype(str) + ":delay_5m"
        delayed_frames.append(delayed)
    return pd.concat(frames, ignore_index=True), pd.concat(delayed_frames, ignore_index=True)


def _eval_2021(base: pd.DataFrame | None = None, delayed: pd.DataFrame | None = None) -> dict[str, object]:
    good_base, good_delayed = _combined(START_2021, END_2021)
    return evaluate_bctp_transfer_stage(
        "2021",
        _market(START_2021, END_2021),
        _flat_frame(),
        good_base if base is None else base,
        good_delayed if delayed is None else delayed,
        calendar_start=START_2021,
        calendar_end=END_2021,
    )


def test_2021_gate_passing_candidates_and_lexicographic_winner() -> None:
    result = _eval_2021()

    assert result["family_ids"] == list(freeze.FAMILY_IDS)
    assert result["passing_candidates"] == list(stages.PROMOTABLE_PRIMARY_IDS)
    assert result["frozen_lexicographic_winner"] == "categorical_linear_fqi"
    assert result["stage_passed"] is True
    assert result["selected_primary_id"] == "categorical_linear_fqi"
    for primary_id in stages.PROMOTABLE_PRIMARY_IDS:
        candidate = result["primary_candidates"][primary_id]
        assert candidate["passed"] is True
        assert candidate["metrics"]["base"]["absolute_return"] > 0.0
        assert candidate["metrics"]["stress_10bp"]["absolute_return"] > 0.0
        assert candidate["metrics"]["delayed_exact"]["absolute_return"] > 0.0
        assert candidate["metrics"]["base"]["nonflat_interval_count"] >= 80
        assert candidate["metrics"]["base"]["long_share_of_nonflat"] >= 0.20
        assert candidate["metrics"]["base"]["short_share_of_nonflat"] >= 0.20
        assert candidate["pmax"] < 0.25


def test_family_order_and_weekly_max_stat_alignment_are_frozen() -> None:
    result = _eval_2021()

    assert list(result["base_family_metrics"].keys()) == list(freeze.FAMILY_IDS)
    assert list(result["weekly_log_returns_for_max_stat"].keys()) == list(freeze.FAMILY_IDS)
    assert result["familywise_max_stat"]["family_ids"] == list(freeze.FAMILY_IDS)
    keys = [tuple(key for key, _ in rows) for rows in result["weekly_log_returns_for_max_stat"].values()]
    assert len(set(keys)) == 1
    assert all(key.endswith("T00:00:00Z") for key in keys[0])


def test_primary_defeats_exact_algorithm_matched_and_feature_controls() -> None:
    result = _eval_2021()
    controls = result["primary_candidates"]["categorical_linear_fqi"]["control_defeat"]["controls"]

    expected = {
        "exact_signature_memory",
        "categorical_linear_fqi_current_only",
        "categorical_linear_fqi_reversed_sequence",
        "categorical_linear_fqi_masked_source",
        "categorical_linear_fqi_shuffled_reward",
        "categorical_linear_fqi_circular_21_reward",
    }
    assert set(controls) == expected
    assert all(item["beats_return"] and item["beats_cagr_to_strict_mdd"] for item in controls.values())
    assert result["primary_candidates"]["categorical_linear_fqi"]["action_code_permutation_schedule_identity"] is True


def test_missing_family_remains_zero_with_error_and_json_safe_finite_output() -> None:
    base, delayed = _combined(START_2021, END_2021)
    missing_id = "categorical_ridge_fqi_masked_source"
    base = base.loc[~base["policy_id"].eq(missing_id)].reset_index(drop=True)

    result = _eval_2021(base=base, delayed=delayed)
    metric = result["base_family_metrics"][missing_id]

    assert missing_id in result["family_ids"]
    assert metric["absolute_return"] == 0.0
    assert metric["cagr_to_strict_mdd"] == 0.0
    assert metric["nonflat_interval_count"] == 0
    assert result["errors"][missing_id] == ["missing base schedule"]
    ridge = result["primary_candidates"]["categorical_ridge_fqi"]
    assert ridge["passed"] is False
    assert (
        ridge["control_defeat"]["controls"][missing_id]["control_valid"]
        is False
    )

    def walk(value):
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, float):
            assert math.isfinite(value)

    walk(result)


def test_2022_uses_selected_primary_without_reselection() -> None:
    base, delayed = _combined(START_2022, END_2022)
    result = evaluate_bctp_transfer_stage(
        "2022",
        _market(START_2022, END_2022),
        _flat_frame(),
        base,
        delayed,
        selected_primary_id="extra_trees_fqi",
        calendar_start=START_2022,
        calendar_end=END_2022,
    )

    assert set(result["passing_candidates"]) == set(stages.PROMOTABLE_PRIMARY_IDS)
    assert result["reselection_allowed"] is False
    assert result["stage_passed"] is True
    assert result["requested_primary_id"] == "extra_trees_fqi"
    assert result["selected_primary_id"] == "extra_trees_fqi"
    assert result["frozen_lexicographic_winner"] == "extra_trees_fqi"


def test_2022_requires_valid_selected_primary_id() -> None:
    base, delayed = _combined(START_2022, END_2022)
    with pytest.raises(ValueError, match="selected_primary_id"):
        evaluate_bctp_transfer_stage(
            "2022",
            _market(START_2022, END_2022),
            _flat_frame(),
            base,
            delayed,
            calendar_start=START_2022,
            calendar_end=END_2022,
        )
    with pytest.raises(ValueError, match="selected_primary_id"):
        evaluate_bctp_transfer_stage(
            "2022",
            _market(START_2022, END_2022),
            _flat_frame(),
            base,
            delayed,
            selected_primary_id="categorical_linear_fqi_current_only",
            calendar_start=START_2022,
            calendar_end=END_2022,
        )


def test_combined_schedule_rejects_unknown_or_reordered_policy_groups() -> None:
    base, delayed = _combined(START_2021, END_2021)
    unknown = base.iloc[[0]].copy()
    unknown["policy_id"] = "not_in_the_frozen_family"
    with pytest.raises(ValueError, match="unknown policy"):
        _eval_2021(base=pd.concat([base, unknown], ignore_index=True), delayed=delayed)

    first = base.loc[base["policy_id"].eq(freeze.FAMILY_IDS[0])]
    reordered = pd.concat(
        [base.loc[~base["policy_id"].eq(freeze.FAMILY_IDS[0])], first],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="policy order"):
        _eval_2021(base=reordered, delayed=delayed)
