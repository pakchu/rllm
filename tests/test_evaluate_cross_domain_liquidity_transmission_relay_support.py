from __future__ import annotations

# Pandas' ``to_dict('records')`` overload is incomplete in the installed stubs.
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportGeneralTypeIssues=false

import ast
from datetime import date, timedelta
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pytest

from training import (
    evaluate_cross_domain_liquidity_transmission_relay_support as support,
)


def _vote_frame(
    source: str,
    rows: Iterable[tuple[str, int, bool]],
) -> pd.DataFrame:
    records = []
    for index, (available, side, valid) in enumerate(rows):
        records.append(
            {
                "source": source,
                "observation_date": date(2021, 1, 1) + timedelta(days=index),
                "available_at": pd.Timestamp(available),
                "side": side,
                "valid": valid,
            }
        )
    return pd.DataFrame.from_records(records, columns=support.VOTE_COLUMNS)


def _clock_rows(clock: str = "primary") -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    counter = 0
    for window, starts in (
        (
            "train",
            (
                "2021-01-05T00:00:00Z",
                "2021-07-05T00:00:00Z",
                "2022-01-05T00:00:00Z",
                "2022-07-05T00:00:00Z",
            ),
        ),
        (
            "selection",
            ("2023-01-05T00:00:00Z", "2023-07-05T00:00:00Z"),
        ),
    ):
        for raw_start in starts:
            period_start = pd.Timestamp(raw_start)
            for offset in range(15):
                decision = period_start + pd.Timedelta(days=10 * offset)
                entry = decision + pd.Timedelta(minutes=5)
                records.append(
                    {
                        "clock": clock,
                        "window": window,
                        "decision_time_utc": decision,
                        "entry_time_utc": entry,
                        "exit_time_utc": entry + support.HOLD,
                        "side": 1 if counter % 2 == 0 else -1,
                    }
                )
                counter += 1
    return pd.DataFrame.from_records(records, columns=support.CLOCK_COLUMNS)


def test_rrp_uses_exact_fifth_prior_slot_and_quarantine_breaks_six_rows() -> None:
    dates = pd.bdate_range("2020-01-01", periods=13)
    complete = [True] * len(dates)
    complete[6] = False
    accepted: list[Any] = [100, 100, 100, 100, 100, 90, None, 120, 80, 80, 80, 80, 70]
    frame = pd.DataFrame(
        {
            "operation_date": dates.strftime("%Y-%m-%d"),
            "result_available_at_utc": dates.tz_localize("UTC")
            + pd.Timedelta(hours=18),
            "total_amount_accepted_usd": accepted,
            "source_complete": complete,
            "quarantine_reason": [
                "" if value else "archive_last_updated_after_operation_date"
                for value in complete
            ],
        }
    )

    votes = support.derive_rrp_votes(frame)

    assert votes.loc[5, ["side", "valid"]].tolist() == [1, True]
    assert votes.loc[6:11, "valid"].eq(False).all()
    assert votes.loc[6:11, "side"].eq(0).all()
    assert votes.loc[12, ["side", "valid"]].tolist() == [1, True]


def test_cboe_uses_previous_intersection_close_at_next_exact_date_0935_et() -> None:
    frame = pd.DataFrame(
        {
            "observation_date": ["2021-03-12", "2021-03-15", "2021-03-17"],
            "VIX9D_close": [20.0, 30.0, 25.0],
            "VIX3M_close": [25.0, 20.0, 25.0],
        }
    )

    votes = support.derive_cboe_votes(frame)

    assert votes["observation_date"].tolist() == [date(2021, 3, 12), date(2021, 3, 15)]
    assert votes["side"].tolist() == [1, -1]
    assert votes["available_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist() == [
        "2021-03-15T13:35:00Z",
        "2021-03-17T13:35:00Z",
    ]


def test_network_requires_exact_eight_calendar_dates_and_majority_log_sign() -> None:
    dates = pd.date_range("2021-01-01", periods=10, freq="D")
    frame = pd.DataFrame(
        {
            "observation_date": dates.strftime("%Y-%m-%d"),
            "available_at": dates.tz_localize("UTC") + pd.Timedelta(days=1),
            "AdrActCnt": [100] * 7 + [110, 90, 100],
            "TxCnt": [100] * 7 + [120, 80, 100],
            "TxTfrCnt": [100] * 7 + [90, 110, 100],
        }
    )

    votes = support.derive_network_votes(frame)

    assert votes.loc[:6, "valid"].eq(False).all()
    assert votes.loc[7:, "valid"].eq(True).all()
    assert votes.loc[7:, "side"].tolist() == [1, -1, 0]

    missing = frame.drop(index=3).reset_index(drop=True)
    missing_votes = support.derive_network_votes(missing)
    row = missing_votes.loc[
        missing_votes["observation_date"].eq(date(2021, 1, 8))
    ].iloc[0]
    assert row["valid"] is np.False_ or row["valid"] == False  # noqa: E712
    assert row["side"] == 0


def test_network_observation_unavailable_until_2024_cannot_emit_pre2024_vote() -> None:
    dates = pd.date_range("2023-12-24", periods=8, freq="D")
    frame = pd.DataFrame(
        {
            "observation_date": dates.strftime("%Y-%m-%d"),
            "available_at": dates.tz_localize("UTC") + pd.Timedelta(days=1),
            "AdrActCnt": range(100, 108),
            "TxCnt": range(200, 208),
            "TxTfrCnt": range(300, 308),
        }
    )

    votes = support.derive_network_votes(frame)

    assert votes["observation_date"].max() == date(2023, 12, 30)
    assert votes["available_at"].lt(support.EVALUATION_END).all()


def test_relay_requires_strictly_later_first_report_and_no_retry_until_reentry() -> (
    None
):
    rrp = _vote_frame(
        "rrp",
        [
            ("2021-01-01T00:00:00Z", 1, True),
            ("2021-01-01T04:00:00Z", -1, True),
            ("2021-01-01T05:00:00Z", 1, True),
        ],
    )
    cboe = _vote_frame("cboe", [("2021-01-01T01:00:00Z", 1, True)])
    network = _vote_frame(
        "network",
        [
            ("2021-01-01T01:00:00Z", 1, True),
            ("2021-01-01T02:00:00Z", 0, True),
            ("2021-01-01T03:00:00Z", 1, True),
            ("2021-01-01T06:00:00Z", 1, True),
        ],
    )

    candidates, onsets, audit = support._relay_candidates(rrp, cboe, network)

    assert onsets["decision_time"].tolist() == [
        pd.Timestamp("2021-01-01T01:00:00Z"),
        pd.Timestamp("2021-01-01T05:00:00Z"),
    ]
    assert candidates[["decision_time", "side"]].to_dict("records") == [
        {"decision_time": pd.Timestamp("2021-01-01T06:00:00Z"), "side": 1}
    ]
    assert audit["neutral_first_network_report"] == 1
    assert audit["confirmed"] == 1


def test_neutral_macro_update_does_not_replace_latest_non_neutral_vote() -> None:
    rrp = _vote_frame(
        "rrp",
        [
            ("2021-01-01T00:00:00Z", 1, True),
            ("2021-01-01T02:00:00Z", 0, True),
        ],
    )
    cboe = _vote_frame("cboe", [("2021-01-01T01:00:00Z", 1, True)])
    network = _vote_frame("network", [("2021-01-01T03:00:00Z", 1, True)])

    candidates, onsets, audit = support._relay_candidates(rrp, cboe, network)

    assert len(onsets) == 1
    assert candidates["decision_time"].tolist() == [
        pd.Timestamp("2021-01-01T03:00:00Z")
    ]
    assert audit["confirmed"] == 1


def test_exact_36h_deadline_is_inclusive_only_while_macro_votes_remain_live() -> None:
    onset = pd.Timestamp("2021-01-01T01:00:00Z")
    rrp = _vote_frame(
        "rrp",
        [
            ("2021-01-01T00:00:00Z", 1, True),
            ("2021-01-02T12:00:00Z", 1, True),
        ],
    )
    cboe = _vote_frame(
        "cboe",
        [
            (onset.isoformat(), 1, True),
            ("2021-01-02T12:00:00Z", 1, True),
        ],
    )
    exact = _vote_frame(
        "network", [((onset + support.RELAY_DEADLINE).isoformat(), 1, True)]
    )
    late = _vote_frame(
        "network",
        [
            (
                (onset + support.RELAY_DEADLINE + pd.Timedelta(minutes=5)).isoformat(),
                1,
                True,
            )
        ],
    )

    exact_candidates, _, exact_audit = support._relay_candidates(rrp, cboe, exact)
    late_candidates, _, late_audit = support._relay_candidates(rrp, cboe, late)

    assert exact_candidates["decision_time"].tolist() == [
        onset + support.RELAY_DEADLINE
    ]
    assert exact_audit["confirmed"] == 1
    assert late_candidates.empty
    assert late_audit["network_deadline_missed"] == 1

    expired_rrp = _vote_frame("rrp", [("2021-01-01T00:00:00Z", 1, True)])
    expired_cboe = _vote_frame("cboe", [(onset.isoformat(), 1, True)])
    expired_candidates, _, expired_audit = support._relay_candidates(
        expired_rrp, expired_cboe, exact
    )
    assert expired_candidates.empty
    assert expired_audit["macro_left_before_confirmation"] == 1


def test_reverse_order_consumes_first_macro_update_without_retry() -> None:
    rrp = _vote_frame(
        "rrp",
        [
            ("2021-01-01T01:00:00Z", 1, True),
            ("2021-01-01T05:00:00Z", 1, True),
        ],
    )
    cboe = _vote_frame("cboe", [("2021-01-01T02:00:00Z", 1, True)])
    network = _vote_frame(
        "network",
        [
            ("2021-01-01T00:00:00Z", 1, True),
            ("2021-01-01T03:00:00Z", 0, True),
            ("2021-01-01T04:00:00Z", 1, True),
        ],
    )

    candidates, audit = support._reverse_order_candidates(rrp, cboe, network)

    assert candidates[["decision_time", "side"]].to_dict("records") == [
        {"decision_time": pd.Timestamp("2021-01-01T05:00:00Z"), "side": 1}
    ]
    assert audit["first_macro_update_disagreed"] == 1
    assert audit["confirmed"] == 1


def test_entry_rounding_split_crossing_and_global_nonoverlap_are_exact() -> None:
    assert support.next_entry_time("2021-01-01T00:05:00Z") == pd.Timestamp(
        "2021-01-01T00:10:00Z"
    )
    assert support.next_entry_time("2021-01-01T00:06:00Z") == pd.Timestamp(
        "2021-01-01T00:15:00Z"
    )
    candidates = pd.DataFrame(
        {
            "decision_time": [
                "2021-01-01T00:00:00Z",
                "2021-01-02T00:00:00Z",
                "2022-12-31T23:58:00Z",
            ],
            "side": [1, -1, 1],
        }
    )

    clock, dropped = support.schedule_candidates(candidates, clock="primary")

    assert len(clock) == 1
    assert dropped == {"global_overlap": 1, "split_crossing": 1}
    assert (
        clock.iloc[0]["exit_time_utc"] - clock.iloc[0]["entry_time_utc"] == support.HOLD
    )


def test_adjacent_intervals_and_exact_split_exits_are_admissible() -> None:
    candidates = pd.DataFrame(
        {
            "decision_time": [
                "2021-01-01T00:00:00Z",
                "2021-01-04T00:00:00Z",
                "2022-12-28T23:55:00Z",
                "2023-01-01T00:00:00Z",
                "2023-12-28T23:55:00Z",
            ],
            "side": [1, -1, 1, -1, 1],
        }
    )

    clock, dropped = support.schedule_candidates(candidates, clock="primary")

    assert dropped == {}
    assert len(clock) == 5
    assert clock.iloc[1]["entry_time_utc"] == clock.iloc[0]["exit_time_utc"]
    assert clock.iloc[2]["exit_time_utc"] == support.TRAIN_END
    assert clock.iloc[3]["window"] == "selection"
    assert clock.iloc[4]["exit_time_utc"] == support.EVALUATION_END


def test_support_and_control_calendar_gates_pass_balanced_synthetic_clock() -> None:
    primary = _clock_rows()
    summary = support.support_summary(primary)
    assert summary["passed"] is True
    assert summary["counts"]["train"] == 60
    assert summary["counts"]["selection"] == 30
    assert summary["side_counts"]["train"] == {"long": 30, "short": 30}

    controls = pd.concat(
        [primary.assign(clock=name) for name in support.CONTROL_NAMES],
        ignore_index=True,
    )
    calendar = support.control_calendar_summary(controls)
    assert calendar["passed"] is True
    assert all(item["rows"] == 90 for item in calendar["controls"].values())

    missing_reverse = controls.loc[controls["clock"].ne("reverse_order")]
    rejected = support.control_calendar_summary(missing_reverse)
    assert rejected["passed"] is False
    assert rejected["controls"]["reverse_order"]["rows"] == 0
    assert rejected["controls"]["reverse_order"]["support"]["passed"] is False


def test_decision_date_overlap_uses_unique_utc_dates_and_one_day_tolerance() -> None:
    candidate = pd.Series(
        pd.to_datetime(
            ["2021-01-01T23:00:00Z", "2021-01-01T23:30:00Z", "2021-01-10T00:00:00Z"]
        )
    )
    comparator = pd.Series(
        pd.to_datetime(["2021-01-02T00:00:00Z", "2021-01-20T00:00:00Z"])
    )

    overlap = support.decision_date_overlap(candidate, comparator)

    assert overlap["candidate_unique_dates"] == 2
    assert overlap["decision_date_jaccard"] == 0.0
    assert overlap["candidate_dates_within_one_utc_day_fraction"] == 0.5


def test_signed_occupied_exposure_uses_entry_inclusive_exit_exclusive_grid() -> None:
    candidate = pd.DataFrame(
        {
            "entry_time_utc": ["2021-01-01T00:00:00Z", "2021-01-02T00:00:00Z"],
            "exit_time_utc": ["2021-01-01T06:00:00Z", "2021-01-02T06:00:00Z"],
            "side": [1, -1],
        }
    )
    same = pd.DataFrame(
        {
            "entry_time": candidate["entry_time_utc"],
            "exit_time": candidate["exit_time_utc"],
            "side": [1, -1],
        }
    )
    flipped = same.assign(side=[-1, 1])

    same_result = support.signed_occupied_exposure_correlation(candidate, same)
    flip_result = support.signed_occupied_exposure_correlation(candidate, flipped)

    assert same_result["signed_occupied_exposure_pearson"] == pytest.approx(1.0)
    assert flip_result["signed_occupied_exposure_pearson"] == pytest.approx(-1.0)
    assert same_result["candidate_nonflat_rows"] == 144


def test_zero_variance_exposure_is_a_failed_metric_not_an_evaluator_crash() -> None:
    candidate = pd.DataFrame(
        {
            "entry_time_utc": [support.EVALUATION_START.isoformat()],
            "exit_time_utc": [support.EVALUATION_END.isoformat()],
            "side": [1],
        }
    )
    comparator = pd.DataFrame(
        {
            "entry_time": [support.EVALUATION_START.isoformat()],
            "exit_time": [support.EVALUATION_END.isoformat()],
            "side": [-1],
        }
    )

    result = support.signed_occupied_exposure_correlation(candidate, comparator)

    assert result["defined"] is False
    assert result["failure_reason"] == "zero_variance"
    assert result["signed_occupied_exposure_pearson"] is None


def test_novelty_requires_complete_identity_and_never_invents_timestamp_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(support.prereg, "DIRECTIONAL_COMPARATORS", ("directional",))
    monkeypatch.setattr(support.prereg, "TIMESTAMP_ONLY_COMPARATORS", ("timestamp",))
    primary = _clock_rows().iloc[:2].copy()
    primary["decision_time_utc"] = pd.to_datetime(
        ["2021-01-10T00:00:00Z", "2021-01-20T00:00:00Z"]
    )
    primary["entry_time_utc"] = primary["decision_time_utc"] + pd.Timedelta(minutes=5)
    primary["exit_time_utc"] = primary["entry_time_utc"] + support.HOLD
    comparators = pd.DataFrame(
        [
            {
                "comparator": "directional",
                "capability": "directional_interval",
                "decision_time": "2021-02-01T00:00:00Z",
                "entry_time": "2021-02-01T00:05:00Z",
                "exit_time": "2021-02-04T00:05:00Z",
                "side": 1,
                "source_clock": "directional:primary",
            },
            {
                "comparator": "timestamp",
                "capability": "timestamp_only",
                "decision_time": "2021-03-01T00:00:00Z",
                "entry_time": "2021-03-01T00:05:00Z",
                "exit_time": "",
                "side": "",
                "source_clock": "timestamp:event_000",
            },
        ],
        columns=support.prereg.COMPARATOR_HEADER,
    )
    monkeypatch.setattr(
        support,
        "signed_occupied_exposure_correlation",
        lambda *_args: {
            "defined": True,
            "absolute_signed_occupied_exposure_pearson": 0.0,
            "signed_occupied_exposure_pearson": 0.0,
        },
    )

    summary = support.novelty_summary(primary, comparators)
    assert summary["passed"] is True
    assert set(summary["comparators"]) == {"directional", "timestamp"}

    leaked = comparators.copy()
    leaked.loc[leaked["comparator"].eq("timestamp"), "side"] = 1
    with pytest.raises(RuntimeError, match="invents direction or exit"):
        support.novelty_summary(primary, leaked)

    with pytest.raises(RuntimeError, match="identity set drift"):
        support.novelty_summary(
            primary, comparators.loc[comparators["comparator"].eq("directional")]
        )


def test_module_imports_no_market_simulator_or_outcome_module() -> None:
    source_path = support.REPOSITORY_ROOT / support.EVALUATOR_SOURCE
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    imports.update(
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert imports.isdisjoint({"evaluation", "execution", "envs", "models"})


def test_paths_fail_closed() -> None:
    for unsafe in ("/tmp/cdltr.json", "~/cdltr.json", "../cdltr.json"):
        with pytest.raises(RuntimeError, match="repository-relative"):
            support._repository_path(unsafe)


def test_preregistration_hashes_are_frozen_without_loading_source_values() -> None:
    artifact = support._load_preregistration()
    assert artifact["candidate"] == support.POLICY_ID
    assert artifact["manifest_hash"] == support.PREREGISTRATION_MANIFEST_HASH
    assert support.sha256_file(support.PREREGISTRATION_ARTIFACT) == (
        support.PREREGISTRATION_ARTIFACT_SHA256
    )


def test_preregistration_validation_never_calls_csv_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        support.pd,
        "read_csv",
        lambda *_args, **_kwargs: pytest.fail("source rows must remain unopened"),
    )
    assert support._load_preregistration()["candidate"] == support.POLICY_ID
