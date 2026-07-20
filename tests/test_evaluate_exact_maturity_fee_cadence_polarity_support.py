from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import evaluate_exact_maturity_fee_cadence_polarity_support as support


def _brute_midrank(values: np.ndarray, reference: int) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    prior: list[float] = []
    for index, value in enumerate(values):
        if not np.isfinite(value):
            continue
        if len(prior) >= reference:
            window = np.asarray(prior[-reference:], dtype=np.float64)
            result[index] = (
                np.sum(window < value) + 0.5 * np.sum(window == value)
            ) / reference
        prior.append(float(value))
    return result


def _blocks(rows: int = 20) -> pd.DataFrame:
    heights = np.arange(1_000, 1_000 + rows, dtype=np.int64)
    return pd.DataFrame(
        {
            "height": heights,
            "id": [f"{value:064x}" for value in heights],
            "previousblockhash": [f"{value - 1:064x}" for value in heights],
            "timestamp": 1_600_000_000 + np.arange(rows) * 600,
            "mediantime": 1_599_996_400 + np.arange(rows) * 600,
            "total_fees": np.arange(rows, dtype=np.int64) * 100 + 1,
        }
    )


def test_strict_prior_midrank_matches_brute_force_and_is_prefix_invariant() -> None:
    values = np.asarray([1.0, 2.0, np.nan, 3.0, 2.0, 4.0, 0.0, 2.0])
    expected = _brute_midrank(values, reference=3)
    observed = support.strict_prior_midrank(values, reference=3)
    np.testing.assert_allclose(observed, expected, equal_nan=True)

    extended = np.concatenate([values, [1_000_000.0, -1_000_000.0]])
    extended_rank = support.strict_prior_midrank(extended, reference=3)
    np.testing.assert_allclose(extended_rank[: len(values)], observed, equal_nan=True)


def test_strict_prior_midrank_rejects_invalid_reference() -> None:
    with pytest.raises(ValueError, match="positive"):
        support.strict_prior_midrank(np.asarray([1.0]), reference=0)


@pytest.mark.parametrize(
    ("override", "failed_check"),
    [
        ({"upper_rank": 0.89}, "rank_tails"),
        ({"entry_delay_bars": 2}, "entry_delay"),
        ({"daily_reference_days": 179}, "daily_reference"),
        ({"maximum_month_share": 0.19}, "month_share"),
    ],
)
def test_policy_binding_rejects_evaluator_threshold_drift(
    override: dict[str, float | int],
    failed_check: str,
) -> None:
    artifact = json.loads(support.PREREGISTRATION.read_text(encoding="utf-8"))
    with pytest.raises(RuntimeError, match=failed_check):
        support._validate_policy_binding(
            artifact,
            replace(support.Policy(), **override),
        )


def test_support_loader_accepts_only_the_exact_frozen_preregistration() -> None:
    artifact = support.load_preregistration()
    assert artifact["manifest_hash"] == support.PREREGISTRATION_MANIFEST_HASH
    assert artifact["policy_hash"] == support.PREREGISTRATION_POLICY_HASH
    assert artifact["outcomes_opened"] is False


def test_state_onsets_do_not_let_invalid_rows_reset_state() -> None:
    state = np.asarray([0, -1, 1, -1, 0, -1, 1], dtype=np.int8)
    valid = np.asarray([True, True, False, True, True, True, True])
    observed = support.state_onsets(state, valid)
    assert observed.tolist() == [False, True, False, False, False, True, True]


def test_build_lag_features_uses_exact_origin_and_only_h_through_h_plus_six() -> None:
    blocks = _blocks()
    policy = replace(
        support.Policy(),
        maturity_lag=3,
        confirmation_blocks=2,
        historical_embargo_seconds=7_200,
        entry_delay_bars=1,
        reference_valid_heights=3,
    )
    features = support.build_lag_features(blocks, lag=3, policy=policy)
    assert len(features) == len(blocks) - 3 - 2
    first = features.iloc[0]
    assert first["maturity_height"] == blocks.iloc[3]["height"]
    assert first["confirmation_height"] == blocks.iloc[5]["height"]
    assert first["matured_fee_component"] == blocks.iloc[0]["total_fees"]
    assert first["maturity_elapsed_seconds"] == 1_800
    raw_available = int(blocks.iloc[5]["timestamp"]) + policy.historical_embargo_seconds
    expected_entry = (
        (raw_available + support.BAR_SECONDS - 1) // support.BAR_SECONDS
    ) * support.BAR_SECONDS + support.BAR_SECONDS
    assert first["entry_epoch"] == expected_entry


def test_lag_101_first_row_is_invalid_not_negative_index_future_reference() -> None:
    blocks = _blocks(rows=120)
    policy = replace(
        support.Policy(),
        maturity_lag=100,
        confirmation_blocks=6,
        reference_valid_heights=3,
    )
    features = support.build_lag_features(blocks, lag=101, policy=policy)
    assert bool(features.iloc[0]["raw_valid"]) is False
    assert np.isnan(features.iloc[0]["matured_fee_component"])
    assert bool(features.iloc[1]["raw_valid"]) is True
    assert features.iloc[1]["matured_fee_component"] == blocks.iloc[0]["total_fees"]


def test_clock_scheduler_uses_onsets_and_suppresses_every_overlapping_event() -> None:
    policy = replace(support.Policy(), hold_bars=12)
    start = int(support.GRID_START.timestamp())
    features = pd.DataFrame(
        {
            "origin_lag": [100] * 5,
            "maturity_height": np.arange(5),
            "confirmation_height": np.arange(5) + 6,
            "entry_epoch": [
                start,
                start + 300,
                start + 600,
                start + 3_900,
                start + 7_800,
            ],
            "fee_pressure": np.arange(5, dtype=float),
            "cadence_compression": np.arange(5, dtype=float),
            "fee_rank": np.linspace(0.1, 0.9, 5),
            "cadence_rank": np.linspace(0.1, 0.9, 5),
            "state": np.zeros(5, dtype=np.int8),
            "state_valid": True,
        }
    )
    state = np.asarray([-1, 0, 1, 0, -1], dtype=np.int8)
    clock = support._clock_from_features(
        "primary",
        features,
        state,
        np.ones(len(state), dtype=bool),
        policy,
    )
    assert clock["maturity_height"].tolist() == [0, 4]
    assert not (
        pd.to_datetime(clock["entry_time"], utc=True).iloc[1]
        < pd.to_datetime(clock["exit_time"], utc=True).iloc[0]
    )


def test_clock_scheduler_retains_pre_grid_event_with_carry_in_exposure() -> None:
    policy = support.Policy()
    entry = int((support.GRID_START - pd.Timedelta(hours=24)).timestamp())
    features = pd.DataFrame(
        {
            "origin_lag": [100],
            "maturity_height": [1],
            "confirmation_height": [7],
            "entry_epoch": [entry],
            "fee_pressure": [1.0],
            "cadence_compression": [1.0],
            "fee_rank": [0.95],
            "cadence_rank": [0.95],
        }
    )
    clock = support._clock_from_features(
        "primary",
        features,
        np.asarray([-1], dtype=np.int8),
        np.asarray([True]),
        policy,
    )
    assert len(clock) == 1
    exposure = support._clock_exposure(clock)
    assert exposure[0] == -1.0


def test_exposure_correlation_is_zero_for_flat_control_and_signed_otherwise() -> None:
    primary = np.asarray([0.0, 1.0, 1.0, 0.0, -1.0, -1.0])
    assert support._exposure_correlation(primary, np.zeros_like(primary)) == 0.0
    assert support._exposure_correlation(primary, primary) == pytest.approx(1.0)
    assert support._exposure_correlation(primary, -primary) == pytest.approx(-1.0)


def test_exposure_novelty_rejects_missing_frozen_shadow() -> None:
    preregistration = json.loads(support.PREREGISTRATION.read_text(encoding="utf-8"))
    with pytest.raises(RuntimeError, match="novelty shadows are missing"):
        support.exposure_novelty_summary(
            {"primary": pd.DataFrame(columns=support.CLOCK_COLUMNS)},
            preregistration,
            support.Policy(),
        )


def test_matched_random_clock_preserves_month_side_activity_and_nonoverlap() -> None:
    policy = replace(support.Policy(), hold_bars=288, random_seed=7)
    entries = pd.date_range(
        "2021-01-01T00:00:00Z",
        "2021-02-28T00:00:00Z",
        freq="2D",
    )
    features = pd.DataFrame(
        {
            "origin_lag": 100,
            "maturity_height": np.arange(len(entries)) + 10_000,
            "confirmation_height": np.arange(len(entries)) + 10_006,
            "entry_epoch": (entries.astype("int64") // 1_000_000_000).astype(int),
            "fee_pressure": np.linspace(1.0, 2.0, len(entries)),
            "cadence_compression": np.linspace(-1.0, 1.0, len(entries)),
            "fee_rank": np.linspace(0.01, 0.99, len(entries)),
            "cadence_rank": np.tile([0.10, 0.35, 0.60, 0.85], 8)[: len(entries)],
            "state": 0,
            "state_valid": True,
        }
    )
    selected_indexes = [0, 4, 17, 22]
    primary = features.iloc[selected_indexes].copy()
    primary.insert(0, "policy_id", policy.policy_id)
    primary.insert(1, "clock", "primary")
    primary.insert(2, "side", [-1, 1, -1, 1])
    primary.insert(
        3,
        "state_label",
        [support.STATE_LABELS[value] for value in primary["side"]],
    )
    primary["entry_time"] = pd.to_datetime(primary["entry_epoch"], unit="s", utc=True)
    primary["exit_time"] = primary["entry_time"] + pd.Timedelta(
        seconds=policy.hold_seconds
    )
    primary = primary.rename(columns={"state": "raw_state", "state_label": "state"})
    primary = primary[list(support.CLOCK_COLUMNS)]

    random_clock = support.matched_random_clock(features, primary, policy)

    assert len(random_clock) == len(primary)
    assert not set(random_clock["maturity_height"]) & set(primary["maturity_height"])
    random_entries = pd.to_datetime(random_clock["entry_time"], utc=True).sort_values()
    assert (
        random_entries.iloc[1:].reset_index(drop=True)
        >= random_entries.iloc[:-1].reset_index(drop=True)
        + pd.Timedelta(seconds=policy.hold_seconds)
    ).all()
    for clock in (primary, random_clock):
        frame = clock.copy()
        frame["month"] = pd.to_datetime(frame["entry_time"], utc=True).dt.strftime("%Y-%m")
        frame["quartile"] = support._cadence_quartile(frame["cadence_rank"])
        counts = frame.groupby(["month", "quartile", "side"]).size().to_dict()
        if clock is primary:
            expected = counts
        else:
            assert counts == expected


def test_atomic_json_writer_ignores_predictable_legacy_temp_symlink(
    tmp_path: Path,
) -> None:
    output = tmp_path / "support.json"
    protected = tmp_path / "protected.txt"
    protected.write_text("safe\n", encoding="utf-8")
    legacy = output.with_name(f".{output.name}.tmp")
    legacy.symlink_to(protected)
    support._atomic_write_json(output, {"ok": True})
    assert protected.read_text(encoding="utf-8") == "safe\n"
    assert legacy.is_symlink()
    assert json.loads(output.read_text(encoding="utf-8")) == {"ok": True}
