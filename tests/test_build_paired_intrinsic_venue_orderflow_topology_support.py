from __future__ import annotations

import gzip
import io
import subprocess
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

from training import (
    build_paired_intrinsic_venue_orderflow_topology_support as b,
)
from training import (
    preregister_paired_intrinsic_venue_orderflow_topology as prereg,
)


def _raw_synthetic_source(days: int = 160) -> pd.DataFrame:
    grid = pd.date_range(
        b.SOURCE_START,
        b.SOURCE_START + days * b.DAY,
        freq=b.BAR,
        inclusive="left",
    )
    day_index = np.repeat(np.arange(days), b.ROWS_PER_DAY)
    bar_index = np.tile(np.arange(b.ROWS_PER_DAY), days)
    amplitude = 0.10 + 0.02 * (day_index % 5)
    front = np.where(bar_index < b.ROWS_PER_DAY // 2, 1.0, -1.0)
    leader_flip = np.where(day_index % 2 == 0, 1.0, -1.0)
    spot_quote = 1.0 + amplitude * front * leader_flip
    um_quote = 1.0 - amplitude * front * leader_flip
    spot_sign = np.choose(
        day_index % 4,
        (-0.55, -0.15, 0.0, 0.45),
    )
    um_sign = np.choose(
        (day_index + 1) % 4,
        (-0.50, 0.0, 0.20, 0.60),
    )
    intraday_wave = np.where(bar_index % 3 == 0, 1.0, 0.85)
    return pd.DataFrame(
        {
            "date": grid,
            "feature_available_time_utc": grid + b.BAR,
            "trade_earliest_time_utc": grid + b.BAR,
            "spot_quote_notional": spot_quote,
            "um_quote_notional": um_quote,
            "spot_signed_quote_notional": (
                spot_quote * spot_sign * intraday_wave
            ),
            "um_signed_quote_notional": (
                um_quote * um_sign * intraday_wave
            ),
            "source_complete": True,
        },
        columns=prereg.SOURCE_ALLOWLIST,
    )


def _synthetic_source(days: int = 160) -> pd.DataFrame:
    return b.validate_source_frame(
        _raw_synthetic_source(days),
        exact_grid=False,
    )


def _token_dict(row: object) -> dict[str, str]:
    return {
        name: str(getattr(row, name)) for name in prereg.TOKEN_COLUMNS
    }


@pytest.fixture(scope="module")
def real_prefix() -> pd.DataFrame:
    return b.load_source_prefix(160 * b.ROWS_PER_DAY)


@pytest.fixture(scope="module")
def real_base(real_prefix: pd.DataFrame) -> pd.DataFrame:
    base, _ = b.build_base_states(real_prefix)
    assert len(base) > 110
    return base


@pytest.fixture(scope="module")
def real_tokens(real_base: pd.DataFrame) -> pd.DataFrame:
    tokens, _ = b.tokenize_base_states(real_base)
    assert len(tokens) > 10
    return tokens


def test_source_validation_preserves_allowlist_and_exact_causal_times() -> None:
    source = _synthetic_source(2)
    assert list(source.columns[: len(prereg.SOURCE_ALLOWLIST)]) == list(
        prereg.SOURCE_ALLOWLIST
    )
    assert source["_row_valid"].all()
    broken = _raw_synthetic_source(2)
    broken.loc[0, "feature_available_time_utc"] = broken.loc[0, "date"]
    with pytest.raises(RuntimeError, match="availability"):
        b.validate_source_frame(broken, exact_grid=False)


def test_source_validation_marks_invalid_values_without_imputation() -> None:
    source = _raw_synthetic_source(2)
    source.loc[10, "spot_quote_notional"] = -1.0
    source.loc[11, "um_signed_quote_notional"] = 2.0
    validated = b.validate_source_frame(source, exact_grid=False)
    assert not bool(validated.loc[10, "_row_valid"])
    assert not bool(validated.loc[11, "_row_valid"])
    assert validated.loc[10, "spot_quote_notional"] == -1.0
    assert validated.loc[11, "um_signed_quote_notional"] == 2.0


def test_loader_uses_exact_usecols_and_never_loads_then_drops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw_synthetic_source(2)
    calls: list[dict[str, object]] = []

    def fake_read_csv(
        path: object,
        **kwargs: object,
    ) -> pd.DataFrame:
        calls.append(dict(kwargs))
        return raw.copy()

    monkeypatch.setattr(b.pd, "read_csv", fake_read_csv)
    loaded = b._read_source(nrows=len(raw))
    assert len(loaded) == len(raw)
    assert calls[0]["usecols"] == list(prereg.SOURCE_ALLOWLIST)
    assert set(calls[0]["dtype"]) == set(prereg.SOURCE_ALLOWLIST)


def test_pre_source_bindings_exclude_market_funding_and_comparators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []

    def fake_hash(path: str) -> str:
        opened.append(str(path))
        expected = {
            str(b.PREREGISTRATION): b.PREREGISTRATION_SHA256,
            prereg.BOUNDARY_DOCUMENT: prereg.BOUNDARY_DOCUMENT_SHA256,
            prereg.MECHANISM_DOCUMENT: prereg.MECHANISM_DOCUMENT_SHA256,
            prereg.SOURCE: prereg.SOURCE_SHA256,
            prereg.SOURCE_MANIFEST: prereg.SOURCE_MANIFEST_SHA256,
            prereg.SOURCE_AUDIT: prereg.SOURCE_AUDIT_SHA256,
        }
        return expected[str(path)]

    monkeypatch.setattr(b, "sha256_file", fake_hash)
    monkeypatch.setattr(
        prereg,
        "sha256_csv_header",
        lambda path: prereg.SOURCE_HEADER_SHA256,
    )
    monkeypatch.setattr(
        prereg,
        "csv_header",
        lambda path: list(prereg.SOURCE_ALLOWLIST),
    )
    b.verify_pre_source_bindings()
    assert prereg.MARKET_SOURCE not in opened
    assert prereg.FUNDING_SOURCE not in opened
    assert not set(prereg.FORBIDDEN_COMPARATOR_PATHS) & set(opened)


def test_base_state_uses_first_passage_buffer_and_real_inference_window() -> None:
    source = _synthetic_source(40)
    base, funnel = b.build_base_states(source)
    assert not base.empty
    row = next(base.itertuples(index=False))
    assert row.early_index < row.late_index <= b.LATEST_ANCHOR_INDEX
    assert row.state_completion_time == row.um_anchor_time + b.BAR or (
        row.state_completion_time == row.spot_anchor_time + b.BAR
    )
    assert row.buffer_completion_time == row.state_completion_time + b.BAR
    assert row.entry_time == row.buffer_completion_time + b.BAR
    assert row.exit_time == row.entry_time + prereg.Policy().hold_bars * b.BAR
    assert funnel["base_paired_states"] == len(base)


def test_exact_anchor_tie_is_ineligible() -> None:
    source = _synthetic_source(40)
    target_day = b.SOURCE_START + 30 * b.DAY
    mask = source["date"].dt.floor("D").eq(target_day)
    source.loc[mask, "um_quote_notional"] = source.loc[
        mask, "spot_quote_notional"
    ].to_numpy()
    source.loc[mask, "um_signed_quote_notional"] = 0.0
    base, funnel = b.build_base_states(source)
    assert target_day not in set(base["source_day"])
    assert funnel["exact_anchor_ties"] >= 1


def test_missing_prefix_row_cancels_state() -> None:
    source = _synthetic_source(45)
    base, _ = b.build_base_states(source)
    target = base.iloc[5]
    missing_time = pd.Timestamp(target["source_day"]) + int(
        target["early_index"]
    ) * b.BAR
    missing = source.loc[~source["date"].eq(missing_time)].copy()
    rebuilt, funnel = b.build_base_states(missing)
    assert target["source_day"] not in set(rebuilt["source_day"])
    assert funnel["invalid_or_missing_prefix"] >= 1


def test_current_value_is_excluded_from_every_prior_quartile() -> None:
    base, _ = b.build_base_states(_synthetic_source(150))
    index = 100
    before = b.prior_thresholds(base, index)
    changed = base.copy()
    changed.loc[index, "gap_bars"] = 1_000_000_000
    for raw_column in (
        "laggard_progress_at_early",
        "spot_abs_flow_late",
        "um_abs_flow_late",
    ):
        changed.loc[index, raw_column] = 1.0e30
    after = b.prior_thresholds(changed, index)
    assert before == after


def test_suppressed_state_remains_in_later_prior_history() -> None:
    base, _ = b.build_base_states(_synthetic_source(220))
    overlapping = base.copy()
    overlapping.loc[100, "entry_time"] = (
        overlapping.loc[99, "entry_time"] + b.BAR
    )
    overlapping.loc[100, "exit_time"] = (
        overlapping.loc[100, "entry_time"]
        + prereg.Policy().hold_bars * b.BAR
    )
    mask = b.reservation_mask(overlapping)
    assert bool(mask.iloc[99])
    assert not bool(mask.iloc[100])
    assert 100 in b.prior_reference_indices(181, len(base))


def test_exact_zero_sign_and_duplicate_quartiles_are_preserved() -> None:
    assert prereg.sign_token(-0.0) == "ZERO"
    assert prereg.sign_token(0.0) == "ZERO"
    prior = np.zeros(prereg.Policy().prior_base_states_min)
    assert prereg.prior_quartile_bucket(0.0, prior) == "Q3"


def test_token_serialization_is_action_option_order_independent() -> None:
    base, _ = b.build_base_states(_synthetic_source(150))
    tokens, _ = b.tokenize_base_states(base)
    row = next(tokens.itertuples(index=False))
    serialized = {
        order: b.serialize_token_state(_token_dict(row))
        for order in prereg.action_option_orders()
    }
    assert len(set(serialized.values())) == 1
    assert list(serialized)[0] == ("ABSTAIN", "LONG", "SHORT")


def test_reservation_is_action_independent_and_half_open() -> None:
    base, _ = b.build_base_states(_synthetic_source(150))
    states, _ = b.tokenize_base_states(base)
    first = states.iloc[:3].copy()
    first.loc[first.index[1], "entry_time"] = (
        first.loc[first.index[0], "entry_time"] + b.BAR
    )
    first.loc[first.index[1], "exit_time"] = (
        first.loc[first.index[1], "entry_time"]
        + prereg.Policy().hold_bars * b.BAR
    )
    first.loc[first.index[2], "entry_time"] = first.loc[
        first.index[0], "exit_time"
    ]
    first.loc[first.index[2], "exit_time"] = (
        first.loc[first.index[2], "entry_time"]
        + prereg.Policy().hold_bars * b.BAR
    )
    identity = first.assign(policy_action=["LONG", "ABSTAIN", "SHORT"])
    flipped = first.assign(policy_action=["ABSTAIN", "SHORT", "LONG"])
    assert b.reservation_mask(identity).tolist() == [True, False, True]
    assert b.reservation_mask(identity).equals(b.reservation_mask(flipped))


def test_split_crossing_state_stays_reserved_but_is_not_counted() -> None:
    base, _ = b.build_base_states(_synthetic_source(150))
    states, _ = b.tokenize_base_states(base)
    crossing = states.iloc[:1].copy()
    boundary = b.SPLITS["train"][1]
    crossing.loc[crossing.index[0], "source_day"] = boundary - b.DAY
    for column in b.TIME_COLUMNS:
        crossing.loc[crossing.index[0], column] = boundary - b.BAR
    crossing.loc[crossing.index[0], "exit_time"] = boundary
    clock = b.assign_primary_reservation(crossing)
    assert bool(clock.loc[clock.index[0], "primary_reserved"])
    assert clock.loc[clock.index[0], "primary_split"] == ""


def test_synthetic_venue_swap_and_sign_mirror_are_equivariant() -> None:
    source = _synthetic_source(160)
    original, _ = b.build_state_clock(source)
    swapped, _ = b.build_state_clock(b.swap_venues(source))
    mirrored, _ = b.build_state_clock(b.mirror_signed_flows(source))
    original_by_day = original.set_index("source_day")
    swapped_by_day = swapped.set_index("source_day")
    mirrored_by_day = mirrored.set_index("source_day")
    common = (
        original_by_day.index.intersection(swapped_by_day.index)
        .intersection(mirrored_by_day.index)
    )
    assert len(common) >= 20
    for day in common:
        expected_swap = prereg.venue_swap_tokens(
            _token_dict(original_by_day.loc[day])
        )
        expected_mirror = prereg.sign_mirror_tokens(
            _token_dict(original_by_day.loc[day])
        )
        assert _token_dict(swapped_by_day.loc[day]) == expected_swap
        assert _token_dict(mirrored_by_day.loc[day]) == expected_mirror


def test_future_append_leaves_prior_raw_states_and_tokens_byte_identical() -> None:
    source = _synthetic_source(170)
    cutoff = b.SOURCE_START + 150 * b.DAY
    prefix = source.loc[source["date"].lt(cutoff)].copy()
    first, _ = b.build_state_clock(prefix)
    second, _ = b.build_state_clock(source)
    second_prefix = second.loc[second["source_day"].lt(cutoff)].reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(
        first.reset_index(drop=True),
        second_prefix,
        check_exact=True,
    )
    assert b.deterministic_clock_bytes(first) == b.deterministic_clock_bytes(
        second_prefix
    )


def test_real_prefix_future_append_venue_swap_and_sign_mirror(
    real_prefix: pd.DataFrame,
) -> None:
    cutoff = b.SOURCE_START + 140 * b.DAY
    prefix = real_prefix.loc[real_prefix["date"].lt(cutoff)].copy()
    original, _ = b.build_state_clock(prefix)
    appended, _ = b.build_state_clock(real_prefix)
    appended_prefix = appended.loc[
        appended["source_day"].lt(cutoff)
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        original.reset_index(drop=True),
        appended_prefix,
        check_exact=True,
    )
    swapped, _ = b.build_state_clock(b.swap_venues(prefix))
    mirrored, _ = b.build_state_clock(b.mirror_signed_flows(prefix))
    original_by_day = original.set_index("source_day")
    swapped_by_day = swapped.set_index("source_day")
    mirrored_by_day = mirrored.set_index("source_day")
    common = (
        original_by_day.index.intersection(swapped_by_day.index)
        .intersection(mirrored_by_day.index)
    )
    assert len(common) >= 1
    for day in common:
        assert _token_dict(swapped_by_day.loc[day]) == (
            prereg.venue_swap_tokens(_token_dict(original_by_day.loc[day]))
        )
        assert _token_dict(mirrored_by_day.loc[day]) == (
            prereg.sign_mirror_tokens(_token_dict(original_by_day.loc[day]))
        )


def test_real_prefix_current_values_are_excluded_from_prior_quartiles(
    real_base: pd.DataFrame,
) -> None:
    index = 100
    before = b.prior_thresholds(real_base, index)
    changed = real_base.copy()
    changed.loc[index, "gap_bars"] = 1_000_000_000
    for raw_column in (
        "laggard_progress_at_early",
        "spot_abs_flow_late",
        "um_abs_flow_late",
    ):
        changed.loc[index, raw_column] = 1.0e30
    assert b.prior_thresholds(changed, index) == before


def test_real_prefix_suppressed_state_remains_in_prior_history(
    real_base: pd.DataFrame,
) -> None:
    overlapping = real_base.copy()
    suppressed_index = 100
    overlapping.loc[suppressed_index, "entry_time"] = (
        overlapping.loc[suppressed_index - 1, "entry_time"] + b.BAR
    )
    overlapping.loc[suppressed_index, "exit_time"] = (
        overlapping.loc[suppressed_index, "entry_time"]
        + prereg.Policy().hold_bars * b.BAR
    )
    reserved = b.reservation_mask(overlapping)
    assert bool(reserved.iloc[suppressed_index - 1])
    assert not bool(reserved.iloc[suppressed_index])
    later_index = len(real_base) - 1
    assert suppressed_index in b.prior_reference_indices(
        later_index,
        len(real_base),
    )


def test_real_prefix_missing_prefix_cancels_state(
    real_prefix: pd.DataFrame,
    real_base: pd.DataFrame,
) -> None:
    target = real_base.iloc[10]
    missing_time = pd.Timestamp(target["source_day"]) + int(
        target["early_index"]
    ) * b.BAR
    missing = real_prefix.loc[~real_prefix["date"].eq(missing_time)].copy()
    rebuilt, funnel = b.build_base_states(missing)
    assert target["source_day"] not in set(rebuilt["source_day"])
    assert funnel["invalid_or_missing_prefix"] >= 1


def _reference_target(
    source: pd.DataFrame,
    source_day: pd.Timestamp,
    venue: str,
) -> float:
    prior = source.loc[
        source["date"].ge(source_day - 28 * b.DAY)
        & source["date"].lt(source_day)
    ].copy()
    prior["_day"] = prior["date"].dt.floor("D")
    quote_column = f"{venue}_quote_notional"
    complete_totals = [
        float(group[quote_column].sum())
        for _, group in prior.groupby("_day", sort=True)
        if len(group) == b.ROWS_PER_DAY and group["_row_valid"].all()
    ]
    assert len(complete_totals) >= prereg.Policy().reference_complete_days_min
    return prereg.Policy().intrinsic_volume_fraction * float(
        np.median(np.asarray(complete_totals, dtype=np.float64))
    )


def test_real_prefix_exact_anchor_tie_is_ineligible(
    real_prefix: pd.DataFrame,
    real_base: pd.DataFrame,
) -> None:
    tied = real_prefix.copy()
    source_day = pd.Timestamp(real_base.iloc[10]["source_day"])
    day_mask = tied["date"].dt.floor("D").eq(source_day)
    for venue in ("spot", "um"):
        target = _reference_target(tied, source_day, venue)
        tied.loc[day_mask, f"{venue}_quote_notional"] = target / 99.5
        tied.loc[day_mask, f"{venue}_signed_quote_notional"] = 0.0
    rebuilt, funnel = b.build_base_states(tied)
    assert source_day not in set(rebuilt["source_day"])
    assert funnel["exact_anchor_ties"] >= 1


def test_real_prefix_exact_zero_sign_is_preserved(
    real_prefix: pd.DataFrame,
    real_tokens: pd.DataFrame,
) -> None:
    zeroed = real_prefix.copy()
    source_day = pd.Timestamp(real_tokens.iloc[-1]["source_day"])
    day_mask = zeroed["date"].dt.floor("D").eq(source_day)
    zeroed.loc[day_mask, "spot_signed_quote_notional"] = 0.0
    zeroed.loc[day_mask, "um_signed_quote_notional"] = 0.0
    rebuilt, _ = b.build_state_clock(zeroed)
    row = rebuilt.loc[rebuilt["source_day"].eq(source_day)].iloc[0]
    assert {
        str(row[token]) for token in prereg.SIGN_TOKEN_COLUMNS
    } == {"ZERO"}


def test_real_prefix_duplicate_quartiles_map_equal_values_upward(
    real_base: pd.DataFrame,
) -> None:
    duplicated = real_base.copy()
    current_index = prereg.Policy().prior_base_states_min
    prior_and_current = duplicated.index[: current_index + 1]
    duplicated.loc[prior_and_current, "gap_bars"] = 1
    duplicated.loc[
        prior_and_current,
        "laggard_progress_at_early",
    ] = 0.5
    duplicated.loc[
        prior_and_current,
        "spot_abs_flow_late",
    ] = 0.25
    duplicated.loc[
        prior_and_current,
        "um_abs_flow_late",
    ] = 0.25
    tokens, _ = b.tokenize_base_states(duplicated)
    source_day = duplicated.iloc[current_index]["source_day"]
    row = tokens.loc[tokens["source_day"].eq(source_day)].iloc[0]
    assert row["gap_q"] == "Q3"
    assert row["laggard_progress_q"] == "Q3"
    assert row["spot_late_abs_flow_q"] == "Q3"
    assert row["um_late_abs_flow_q"] == "Q3"


def test_real_prefix_option_order_does_not_change_token_serialization(
    real_tokens: pd.DataFrame,
) -> None:
    row = next(real_tokens.itertuples(index=False))
    serializations = [
        b.serialize_token_state(_token_dict(row))
        for _ in prereg.action_option_orders()
    ]
    assert len(set(serializations)) == 1


def test_real_prefix_reservation_is_action_independent(
    real_tokens: pd.DataFrame,
) -> None:
    states = real_tokens.iloc[:3].copy()
    states.loc[states.index[1], "entry_time"] = (
        states.loc[states.index[0], "entry_time"] + b.BAR
    )
    states.loc[states.index[1], "exit_time"] = (
        states.loc[states.index[1], "entry_time"]
        + prereg.Policy().hold_bars * b.BAR
    )
    first = states.assign(policy_action=["LONG", "ABSTAIN", "SHORT"])
    second = states.assign(policy_action=["SHORT", "LONG", "ABSTAIN"])
    assert b.reservation_mask(first).equals(b.reservation_mask(second))
    assert b.reservation_mask(first).tolist() == [True, False, True]


def test_support_clock_is_outcome_blind_deterministic_and_under_gated() -> None:
    report, first = b.build_support_from_frame(_synthetic_source(160))
    _, second = b.build_support_from_frame(_synthetic_source(160))
    assert first == second
    assert report["artifact_eligible"] is False
    assert report["outcomes_opened"] is False
    assert report["market_values_loaded"] is False
    assert report["funding_values_loaded"] is False
    assert report["comparator_rows_decoded"] == 0
    assert report["source_support_passed"] is False
    assert report["decision"] == "synthetic_build_cannot_authorize_next_stage"
    with gzip.GzipFile(fileobj=io.BytesIO(first), mode="rb") as handle:
        header = handle.readline().decode("utf-8").strip().split(",")
    assert header == list(b.CLOCK_COLUMNS)
    assert not any(
        token in column.lower()
        for column in header
        for token in b.FORBIDDEN_CLOCK_TOKENS
    )


def test_protocol_guard_requires_committed_clean_source_and_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: Iterator[subprocess.CompletedProcess[str]] = iter(
        (
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 0, "", ""),
        )
    )
    monkeypatch.setattr(b, "_git_check", lambda *args: next(calls))
    b._assert_protocol_committed()
    monkeypatch.setattr(
        b,
        "_git_check",
        lambda *args: subprocess.CompletedProcess([], 1, "", ""),
    )
    with pytest.raises(RuntimeError, match="not committed"):
        b._assert_protocol_committed()


def test_git_check_uses_an_absolute_resolved_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        observed.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(b.shutil, "which", lambda name: "/resolved/git")
    monkeypatch.setattr(b.subprocess, "run", fake_run)
    result = b._git_check("status", "--short")
    assert result.returncode == 0
    assert observed == [["/resolved/git", "status", "--short"]]


def test_write_once_is_reproducible_and_rejects_drift(tmp_path) -> None:
    path = tmp_path / "artifact.bin"
    assert b._write_once(path, b"one") == "created"
    assert b._write_once(path, b"one") == "verified_existing"
    with pytest.raises(RuntimeError, match="noncanonical"):
        b._write_once(path, b"two")
