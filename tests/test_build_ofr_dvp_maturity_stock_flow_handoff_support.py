from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path

import pytest

from training import build_ofr_dvp_maturity_stock_flow_handoff_support as support
from training import preregister_ofr_dvp_maturity_stock_flow_handoff as prereg


UTC = timezone.utc


def _source_rows(
    day: date,
    available_at: datetime,
    *,
    disclosure_edit: bool = False,
    overrides: dict[str, Fraction | None] | None = None,
) -> dict[str, support.SourceRow]:
    values: dict[str, Fraction | None] = {
        "REPO-DVP_OV_OO-P": Fraction(2),
        "REPO-DVP_OV_LE30-P": Fraction(3),
        "REPO-DVP_OV_G30-P": Fraction(5),
        "REPO-DVP_TV_OO-P": Fraction(6),
        "REPO-DVP_TV_LE30-P": Fraction(1),
        "REPO-DVP_TV_G30-P": Fraction(3),
        "REPO-DVP_AR_OO-P": Fraction(4),
        "REPO-DVP_AR_LE30-P": Fraction(5),
        "REPO-DVP_AR_G30-P": Fraction(8),
    }
    values.update(overrides or {})
    return {
        mnemonic: support.SourceRow(
            mnemonic=mnemonic,
            observation_date=day,
            available_at=available_at,
            value=value,
            disclosure_edit=disclosure_edit and index == 0,
        )
        for index, (mnemonic, value) in enumerate(values.items())
    }


def _feature(
    index: int,
    *,
    flow: Fraction,
    curve: Fraction,
    available_at: datetime | None = None,
    epoch: int = 0,
    decision_allowed: bool = True,
    dominant: str = "LE30",
) -> support.FeatureRow:
    day = date(2019, 1, 1) + timedelta(days=index)
    available = available_at or datetime.combine(day, datetime.min.time(), UTC)
    return support.FeatureRow(
        observation_date=day,
        available_at=available,
        epoch=epoch,
        decision_allowed=decision_allowed,
        components={"maturity_flow_gap": flow, "curve_gap": curve},
        dominant_rate_bucket=dominant,
    )


def _state(
    index: int,
    flow_state: int,
    curve_state: int,
    *,
    epoch: int = 0,
    available_at: datetime | None = None,
) -> support.StateRow:
    feature = _feature(
        index,
        flow=Fraction(flow_state),
        curve=Fraction(curve_state),
        epoch=epoch,
        available_at=available_at,
    )
    units = {
        "maturity_flow_gap": Fraction(3 * flow_state, 4),
        "curve_gap": Fraction(3 * curve_state, 4),
    }
    return support.StateRow(
        rank=support.RankRow(feature=feature, units=units),
        epoch=epoch,
        flow_state=flow_state,
        curve_state=curve_state,
    )


def _candidate(
    precursor_day: date,
    confirmation_day: date,
    confirmation_available: datetime,
    *,
    polarity: int = 1,
    age: int = 1,
) -> support.Candidate:
    base = date(2019, 1, 1)
    precursor_index = (precursor_day - base).days
    confirmation_index = (confirmation_day - base).days
    precursor = _state(precursor_index, polarity, 0)
    confirmation = _state(
        confirmation_index,
        polarity,
        polarity,
        available_at=confirmation_available,
    )
    return support.Candidate(
        precursor=precursor,
        confirmation=confirmation,
        polarity=polarity,
        age=age,
    )


def _scheduled(entry: datetime, *, index: int) -> support.Scheduled:
    side = 1 if index % 2 == 0 else -1
    polarity = -side
    age = (2, 5, 8)[index % 3]
    bucket = "LE30" if index % 2 == 0 else "G30"
    split = "train" if entry.year < 2023 else "selection"
    return support.Scheduled(
        control="primary",
        precursor_observation_date=entry.date() - timedelta(days=age + 8),
        confirmation_observation_date=entry.date() - timedelta(days=8),
        signal_time=entry - support.BAR,
        entry_time=entry,
        exit_time=entry + support.HOLD,
        split=split,
        side=side,
        polarity=polarity,
        confirmation_age_rows=age,
        dominant_rate_bucket=bucket,
        components={
            "maturity_flow_gap": Fraction(polarity),
            "curve_gap": Fraction(polarity),
        },
        units={
            "maturity_flow_gap": Fraction(3 * polarity, 4),
            "curve_gap": Fraction(3 * polarity, 4),
        },
    )


def _passing_primary() -> list[support.Scheduled]:
    rows: list[support.Scheduled] = []
    index = 0
    for year in (2021, 2022, 2023):
        for month in range(1, 13):
            for day in (1, 15):
                rows.append(
                    _scheduled(
                        datetime(year, month, day, 0, 5, tzinfo=UTC),
                        index=index,
                    )
                )
                index += 1
    return rows


def _clock_set(primary: list[support.Scheduled]) -> dict[str, list[support.Scheduled]]:
    clocks = {
        name: [support.replace(row, control=name) for row in primary]
        for name in support.CLOCK_NAMES
    }
    clocks["exact_direction_flip"] = [
        support.replace(row, control="exact_direction_flip", side=-row.side)
        for row in primary
    ]
    clocks["deterministic_random_side"] = [
        support.replace(
            row,
            control="deterministic_random_side",
            side=support._random_side(row.entry_time),
        )
        for row in primary
    ]
    clocks["constant_long"] = [
        support.replace(row, control="constant_long", side=1) for row in primary
    ]
    clocks["constant_short"] = [
        support.replace(row, control="constant_short", side=-1) for row in primary
    ]
    return clocks


def test_exact_features_invalid_dates_and_equal_availability() -> None:
    shared = datetime(2021, 1, 10, tzinfo=UTC)
    first = date(2021, 1, 1)
    second = date(2021, 1, 2)
    partial = date(2021, 1, 3)
    edited = date(2021, 1, 4)
    negative = date(2021, 1, 5)
    zero_denominator = date(2021, 1, 6)
    partial_rows = dict(
        list(
            _source_rows(partial, datetime(2021, 1, 11, tzinfo=UTC)).items()
        )[:1]
    )
    features, audit = support.build_features(
        {
            first: _source_rows(first, shared),
            second: _source_rows(second, shared),
            partial: partial_rows,
            edited: _source_rows(
                edited, datetime(2021, 1, 11, tzinfo=UTC), disclosure_edit=True
            ),
            negative: _source_rows(
                negative,
                datetime(2021, 1, 12, tzinfo=UTC),
                overrides={"REPO-DVP_TV_G30-P": Fraction(-1)},
            ),
            zero_denominator: _source_rows(
                zero_denominator,
                datetime(2021, 1, 13, tzinfo=UTC),
                overrides={
                    "REPO-DVP_OV_OO-P": Fraction(0),
                    "REPO-DVP_OV_LE30-P": Fraction(0),
                    "REPO-DVP_OV_G30-P": Fraction(0),
                },
            ),
        }
    )

    assert len(features) == 2
    assert features[0].components == {
        "maturity_flow_gap": Fraction(2, 5),
        "curve_gap": Fraction(13, 4),
    }
    assert features[0].dominant_rate_bucket == "G30"
    assert [row.decision_allowed for row in features] == [False, True]
    assert audit.invalid_missing_null_or_edit_dates == 2
    assert audit.invalid_negative_volume_dates == 1
    assert audit.invalid_denominator_dates == 1
    assert audit.equal_availability_rows_suppressed == 1


def test_rank_rows_use_strict_prebatch_history() -> None:
    features = [
        _feature(index, flow=Fraction(index + 1), curve=Fraction(index + 1))
        for index in range(support.LOOKBACK)
    ]
    shared = datetime(2021, 1, 1, tzinfo=UTC)
    features.extend(
        [
            _feature(
                support.LOOKBACK,
                flow=Fraction(1_000),
                curve=Fraction(1_000),
                available_at=shared,
                decision_allowed=False,
            ),
            _feature(
                support.LOOKBACK + 1,
                flow=Fraction(1),
                curve=Fraction(1),
                available_at=shared,
            ),
        ]
    )
    ranks = support.build_rank_rows(features)
    assert len(ranks) == 2
    assert ranks[0].units["maturity_flow_gap"] == 1
    assert ranks[1].units["maturity_flow_gap"] == Fraction(-251, 252)


def test_contradiction_precedes_confirmation_and_prevents_same_row_rearm() -> None:
    rows = [
        _state(0, 0, 0),
        _state(1, 1, 0),
        _state(2, -1, 1),
        _state(3, -1, -1),
    ]
    assert support.derive_handoff_candidates(rows) == []

    rows.extend(
        [
            _state(4, 0, 0),
            _state(5, 1, 0),
            _state(6, 1, 1),
        ]
    )
    candidates = support.derive_handoff_candidates(rows)
    assert len(candidates) == 1
    assert candidates[0].precursor.rank.feature.observation_date == date(
        2019, 1, 6
    )
    assert candidates[0].age == 1


def test_age_ten_confirms_age_eleven_does_not_and_epoch_break_cancels() -> None:
    accepted = [_state(0, 0, 0), _state(1, 1, 0)]
    accepted.extend(_state(index, 1, 0) for index in range(2, 11))
    accepted.append(_state(11, 1, 1))
    candidates = support.derive_handoff_candidates(accepted)
    assert len(candidates) == 1
    assert candidates[0].age == 10

    expired = [_state(0, 0, 0), _state(1, 1, 0)]
    expired.extend(_state(index, 1, 0) for index in range(2, 12))
    expired.append(_state(12, 1, 1))
    assert support.derive_handoff_candidates(expired) == []

    broken = [
        _state(0, 0, 0, epoch=0),
        _state(1, 1, 0, epoch=0),
        _state(2, 1, 0, epoch=1),
        _state(3, 1, 1, epoch=1),
    ]
    assert support.derive_handoff_candidates(broken) == []


def test_stale_states_never_cross_epoch() -> None:
    rows = [
        _state(0, 0, 0, epoch=0),
        _state(1, 1, 0, epoch=0),
        _state(2, 1, 1, epoch=1),
        _state(3, -1, 1, epoch=1),
    ]
    stale = support.stale_state_rows(rows, 1)
    assert [(row.flow_state, row.curve_state) for row in stale] == [(0, 0), (1, 1)]
    assert stale[0].epoch != stale[1].epoch


def test_schedule_waits_one_bar_allows_touching_exit_and_contains_splits() -> None:
    first_signal = datetime(2021, 3, 1, 0, 0, tzinfo=UTC)
    first = _candidate(date(2021, 2, 20), date(2021, 2, 21), first_signal)
    touching_signal = first_signal + support.HOLD
    touching = _candidate(
        date(2021, 2, 27), date(2021, 2, 28), touching_signal
    )
    overlapping = _candidate(
        date(2021, 2, 26),
        date(2021, 2, 27),
        touching_signal - timedelta(minutes=1),
    )
    crossing = _candidate(
        date(2022, 12, 20),
        date(2022, 12, 21),
        datetime(2022, 12, 27, tzinfo=UTC),
    )
    rows = support.schedule("primary", [first, overlapping, touching, crossing])
    assert len(rows) == 2
    assert rows[0].entry_time == first_signal + support.BAR
    assert rows[0].exit_time == rows[1].entry_time
    assert all(row.split == "train" for row in rows)


def test_noncausal_placebos_are_deterministic_and_never_clock_controls() -> None:
    features = [
        _feature(
            index,
            flow=Fraction(index + 1),
            curve=Fraction((index * 7) % 31),
        )
        for index in range(700)
    ]
    first = support.permute_features(
        features, "year_curve_permutation_placebo"
    )
    second = support.permute_features(
        features, "year_curve_permutation_placebo"
    )
    assert first == second
    for year in {row.observation_date.year for row in features}:
        original = Counter(
            row.components["curve_gap"]
            for row in features
            if row.observation_date.year == year
        )
        permuted = Counter(
            row.components["curve_gap"]
            for row in first
            if row.observation_date.year == year
        )
        assert original == permuted
    report = support.placebo_incidence(features)
    assert set(report) == set(support.PLACEBO_NAMES)
    assert set(support.PLACEBO_NAMES).isdisjoint(support.CLOCK_NAMES)
    assert all(row["causal"] is False for row in report.values())
    assert all(row["economic_evaluation_forbidden"] for row in report.values())
    assert all(row["execution_clock_emitted"] is False for row in report.values())


def test_source_support_passes_full_synthetic_battery_and_fails_side_loss() -> None:
    primary = _passing_primary()
    clocks = _clock_set(primary)
    result = support.source_support(clocks)
    assert result["passed"] is True
    assert all(result["gates"].values())
    assert result["train"]["events"] == 48
    assert result["selection"]["events"] == 24

    failed = dict(clocks)
    failed["primary"] = [
        support.replace(row, side=1) if row.split == "selection" else row
        for row in primary
    ]
    result = support.source_support(failed)
    assert result["passed"] is False
    assert result["gates"]["selection_each_side"] is False


def test_support_ratios_and_elapsed_gaps_are_exact_rationals() -> None:
    result = support.source_support(_clock_set(_passing_primary()))
    assert result["train"]["maximum_month_share"] == "1/24"
    assert result["selection"]["maximum_month_share"] == "1/12"
    assert result["train"]["maximum_single_rate_bucket_share"] == "1/2"
    assert result["selection"]["maximum_single_rate_bucket_share"] == "1/2"
    assert "." not in result["train"]["maximum_entry_gap_elapsed_days"]


@pytest.mark.parametrize(
    ("precursor_delta", "confirmation_delta", "age"),
    ((8, 8, 1), (9, 8, 0), (19, 8, 11)),
)
def test_primary_chronology_and_age_fail_closed(
    precursor_delta: int, confirmation_delta: int, age: int
) -> None:
    primary = _passing_primary()
    malformed = support.replace(
        primary[0],
        precursor_observation_date=primary[0].entry_time.date()
        - timedelta(days=precursor_delta),
        confirmation_observation_date=primary[0].entry_time.date()
        - timedelta(days=confirmation_delta),
        confirmation_age_rows=age,
    )
    clocks = _clock_set([malformed, *primary[1:]])
    result = support.source_support(clocks)
    assert result["passed"] is False
    assert result["gates"]["exact_clock_integrity"] is False


def test_empty_support_is_serializable_and_fails_closed() -> None:
    clocks = {name: [] for name in support.CLOCK_NAMES}
    result = support.source_support(clocks)
    assert result["passed"] is False
    assert result["train"]["maximum_entry_gap_elapsed_days"] is None
    json.dumps(result, allow_nan=False)


def test_random_side_uses_canonical_utc_second() -> None:
    entry = datetime(2023, 1, 1, 0, 5, tzinfo=UTC)
    token = b"DMSH-168|deterministic_random_side|2023-01-01T00:05:00Z"
    expected = 1 if hashlib.sha256(token).digest()[0] < 128 else -1
    assert support._random_side(entry) == expected
    with pytest.raises(RuntimeError, match="whole-second"):
        support._random_side(entry.replace(microsecond=1))


def test_write_or_verify_is_immutable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(support, "REPOSITORY_ROOT", tmp_path)
    path = Path("results/artifact.bin")
    assert support._write_or_verify(path, b"first") == "created"
    assert support._write_or_verify(path, b"first") == "verified_existing"
    with pytest.raises(RuntimeError, match="artifact differs"):
        support._write_or_verify(path, b"second")
    assert (tmp_path / path).read_bytes() == b"first"


def test_registration_loader_does_not_reopen_comparator_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = {
        "candidate": prereg.POLICY_ID,
        "policy": {"frozen": True},
        "policy_hash": prereg.canonical_hash({"frozen": True}),
        "manifest_hash": support.PREREGISTRATION_MANIFEST_HASH,
        "candidate_features_or_incidence_opened": False,
        "comparator_rows_opened_during_preregistration": False,
        "outcomes_opened": False,
        "performance_values_opened": False,
    }
    path = Path("prereg.json")
    (tmp_path / path).write_text(json.dumps(artifact))
    monkeypatch.setattr(support, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(support, "PREREGISTRATION", path)
    monkeypatch.setattr(
        support,
        "PREREGISTRATION_SHA256",
        hashlib.sha256((tmp_path / path).read_bytes()).hexdigest(),
    )
    assert support._load_registration() == artifact


def test_protocol_guard_requires_tracked_clean_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for path in (support.SCRIPT_PATH, support.TEST_PATH):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("sealed\n")
    monkeypatch.setattr(support, "REPOSITORY_ROOT", tmp_path)
    calls: list[tuple[str, ...]] = []

    def clean_git(*args: str) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(("git", *args), 0, "", "")

    monkeypatch.setattr(support, "_git_check", clean_git)
    support._assert_protocol_committed()
    assert [call[0] for call in calls] == ["ls-files", "diff"]

    monkeypatch.setattr(
        support,
        "_git_check",
        lambda *args: subprocess.CompletedProcess(("git", *args), 1, "", ""),
    )
    with pytest.raises(RuntimeError, match="not committed"):
        support._assert_protocol_committed()

    def dirty_git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            ("git", *args), 0 if args[0] == "ls-files" else 1, "", ""
        )

    monkeypatch.setattr(support, "_git_check", dirty_git)
    with pytest.raises(RuntimeError, match="differs from HEAD"):
        support._assert_protocol_committed()


def test_run_checks_commit_before_opening_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def blocked() -> None:
        raise RuntimeError("protocol is not committed")

    monkeypatch.setattr(support, "_assert_protocol_committed", blocked)
    monkeypatch.setattr(
        support,
        "load_source",
        lambda: pytest.fail("source must remain unopened before commit check"),
    )
    with pytest.raises(RuntimeError, match="not committed"):
        support.run()
