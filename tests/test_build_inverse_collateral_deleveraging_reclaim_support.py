from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import build_inverse_collateral_deleveraging_reclaim_support as support
from training import preregister_inverse_collateral_deleveraging_reclaim as prereg


def _small_policy(**changes: object) -> prereg.Policy:
    return replace(
        prereg.Policy(),
        baseline_bars=5,
        baseline_min_periods=3,
        oi_change_bars=2,
        taker_smoothing_bars=2,
        confirmation_window_bars=4,
        execution_delay_bars=2,
        hold_bars=5,
        **changes,
    )


def _raw_frame(rows: int = 50) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.date_range("2021-07-08", periods=rows, freq="5min"),
            "um_sum_open_interest_value": 1_000_000.0 + 100.0 * index,
            "um_sum_taker_long_short_vol_ratio": 1.0 + 0.01 * np.sin(index),
            "cm_sum_open_interest": 500_000.0 + 50.0 * index,
            "cm_sum_taker_long_short_vol_ratio": 1.0 + 0.01 * np.cos(index),
            "source_complete": True,
            "source_available": True,
            "quarantined": False,
        }
    )


def _event(
    setup: int,
    signal: int,
    *,
    confirmation: int | None = None,
    side: int = 1,
    branch: str = "primary",
    delay: int = 2,
    hold: int = 5,
) -> dict[str, object]:
    return {
        "setup_position": setup,
        "confirmation_position": signal if confirmation is None else confirmation,
        "signal_position": signal,
        "side": side,
        "branch": branch,
        "delay_bars": delay,
        "hold_bars": hold,
    }


def test_prior_quantile_excludes_current_and_quarantined_observations() -> None:
    values = pd.Series([1.0, 2.0, 100.0, 4.0, 5.0])
    eligible = pd.Series([True, True, False, True, True])
    result = support.prior_clean_quantile(
        values, eligible, quantile=0.5, window=4, min_periods=2
    )
    assert np.isnan(result.iloc[0])
    assert result.iloc[2] == pytest.approx(1.5)
    assert result.iloc[3] == pytest.approx(1.5)
    assert result.iloc[4] == pytest.approx(2.0)


def test_quarantine_carries_through_exact_following_bars() -> None:
    available = pd.Series([True, False, True, True, True, True])
    result = support.quarantine_mask(available, post_gap_bars=2)
    assert result.tolist() == [False, True, True, True, False, False]


def test_feature_prefix_is_invariant_to_future_source_mutation() -> None:
    raw = _raw_frame()
    policy = _small_policy()
    baseline = support.build_features(raw, policy)
    changed = raw.copy()
    changed.loc[40:, "um_sum_open_interest_value"] *= 100.0
    changed.loc[40:, "cm_sum_taker_long_short_vol_ratio"] *= 10.0
    replay = support.build_features(changed, policy)
    pd.testing.assert_frame_equal(baseline.loc[:39], replay.loc[:39])


def test_confirmation_takes_first_match_and_suppresses_replacement_setup() -> None:
    setup = pd.Series(
        [False, True, False, True, False, True, False, False, False, False]
    )
    confirmation = pd.Series(
        [False, False, False, True, False, False, True, False, False, False]
    )
    events, accepted = support.confirmed_events(
        setup, confirmation, policy=_small_policy(), branch="primary"
    )
    assert accepted.tolist() == [1, 5]
    assert events["setup_position"].tolist() == [1, 5]
    assert events["signal_position"].tolist() == [3, 6]


def test_expired_setup_does_not_read_beyond_confirmation_window() -> None:
    setup = pd.Series([False, True, *([False] * 8)])
    confirmation = pd.Series([False, *([False] * 5), True, False, False, False])
    events, accepted = support.confirmed_events(
        setup, confirmation, policy=_small_policy(), branch="primary"
    )
    assert accepted.tolist() == [1]
    assert events.empty


def test_no_reclaim_uses_primary_accepted_setups_not_replacement_onsets() -> None:
    policy = _small_policy()
    events, setups = support.immediate_events_from_setups(
        np.asarray([1, 5], dtype=np.int64), policy=policy, branch="no_reclaim"
    )
    assert setups.tolist() == [1, 5]
    assert events["setup_position"].tolist() == [1, 5]
    assert events["signal_position"].tolist() == [1, 5]
    assert events["confirmation_position"].tolist() == [-1, -1]


def test_nonoverlap_reserves_entry_to_exit_not_signal_to_exit() -> None:
    frame = pd.DataFrame({"date": pd.date_range("2021-07-08", periods=25, freq="5min")})
    events = pd.DataFrame(
        [_event(1, 1), _event(4, 4), _event(10, 10)],
        columns=support.EVENT_COLUMNS,
    )
    schedule = support.nonoverlapping_schedule(
        events, frame, policy_name="primary", purge_quantile=0.9
    )
    assert schedule["signal_position"].tolist() == [1, 10]
    assert schedule["entry_position"].tolist() == [3, 12]
    assert schedule["exit_position"].tolist() == [8, 17]


def test_classification_uses_relative_purge_then_source_only_reclaim() -> None:
    rows = 12
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2021-07-08", periods=rows, freq="5min"),
            "clean": True,
            "eligible": True,
            "um_taker_ratio": 0.9,
            "cm_taker_ratio": 0.8,
            "du": 0.0,
            "dc": 0.0,
            "du_one": 0.0,
            "dc_one": -0.01,
            "tu": 0.0,
            "tc": 0.0,
            "relative_purge": 0.0,
            "cm_sell_stress": 0.0,
            "cm_specific_sell_gap": 0.0,
            "um_sell_stress": 0.0,
            "cm_sell_q90": 0.5,
            "taker_gap_q90": 0.5,
            "um_sell_q90": 0.5,
            "purge_q900": 0.5,
            "cm_contraction_q900": 0.5,
            "um_contraction_q900": 0.5,
        }
    )
    frame.loc[2, ["dc", "relative_purge", "cm_sell_stress", "cm_specific_sell_gap"]] = [
        -0.6,
        0.8,
        0.8,
        0.8,
    ]
    frame.loc[4, ["cm_taker_ratio", "um_taker_ratio", "dc_one"]] = [1.2, 1.1, 0.0]
    events, setups, _ = support.classify_events(
        frame, _small_policy(), purge_quantile=0.9
    )
    assert setups["primary"].tolist() == [2]
    assert events["primary"]["signal_position"].tolist() == [4]
    assert events["primary"]["side"].tolist() == [1]
    assert events["direction_flip"]["side"].tolist() == [-1]
    assert events["one_hour_signal_delay"]["signal_position"].tolist() == [16]

    broken_anchor = frame.copy()
    broken_anchor.loc[3, "clean"] = False
    broken_events, _, _ = support.classify_events(
        broken_anchor, _small_policy(), purge_quantile=0.9
    )
    assert broken_events["primary"].empty


def test_policy_mutation_is_rejected_before_source_loading() -> None:
    changed = replace(prereg.Policy(), hold_bars=12)
    with pytest.raises(ValueError, match="policy is frozen"):
        support.run_support(changed)


def test_preregistration_and_loader_contract_are_outcome_blind() -> None:
    registration = support.verify_preregistration()
    assert registration["outcomes_opened"] is False
    assert set(support.METRIC_COLUMNS) == {
        "um_sum_open_interest_value",
        "um_sum_taker_long_short_vol_ratio",
        "cm_sum_open_interest",
        "cm_sum_taker_long_short_vol_ratio",
    }
    forbidden = {"open", "high", "low", "close", "return", "funding"}
    assert forbidden.isdisjoint(support.METRIC_COLUMNS)


def test_support_loader_does_not_open_market_or_funding_files(monkeypatch) -> None:
    registration = prereg.build_manifest()
    source = registration["source_contract"]
    opened_for_hash: list[str] = []

    def fake_hash(path: str | Path) -> str:
        value = str(path)
        opened_for_hash.append(value)
        expected = {
            source["metrics"]: source["metrics_sha256"],
            source["metrics_manifest"]: source["metrics_manifest_sha256"],
        }
        if value not in expected:
            raise AssertionError(f"sealed source opened during support: {value}")
        return expected[value]

    fake = pd.DataFrame(
        {
            "date": [support.START],
            "um_sum_open_interest_value": [1_000_000.0],
            "um_sum_taker_long_short_vol_ratio": [1.0],
            "cm_sum_open_interest": [500_000.0],
            "cm_sum_taker_long_short_vol_ratio": [1.0],
            "source_complete": [True],
        }
    )
    monkeypatch.setattr(support, "verify_preregistration", lambda: registration)
    monkeypatch.setattr(support, "_sha256", fake_hash)
    monkeypatch.setattr(support, "_assert_complete_grid", lambda dates: None)
    monkeypatch.setattr(support.pd, "read_csv", lambda *args, **kwargs: fake.copy())
    _, metadata = support.load_support_frame(_small_policy())
    assert set(opened_for_hash) == {source["metrics"], source["metrics_manifest"]}
    assert metadata["market_expected_sha256_not_opened"] == source["market_sha256"]
    assert metadata["funding_expected_sha256_not_opened"] == source["funding_sha256"]


def test_frozen_support_artifact_matches_clock_when_present() -> None:
    artifact = Path(support.DEFAULT_OUTPUT)
    clock = Path(support.DEFAULT_CLOCK)
    if not artifact.exists() or not clock.exists():
        pytest.skip("support artifact is generated after helper tests")
    payload = json.loads(artifact.read_text())
    assert payload["protocol"]["outcomes_opened"] is False
    assert payload["protocol"]["price_return_funding_loaded"] is False
    assert payload["source"]["price_or_outcome_columns_loaded"] == []
    assert payload["support_decision"] == "reject_before_returns_no_train_support_q"
    assert payload["selected_purge_quantile"] is None
    assert payload["selected_q_2023_support_validation"] is None
    candidates = payload["train_selection_candidates_descending_q"]
    assert [candidate["purge_quantile"] for candidate in candidates] == [
        0.95,
        0.925,
        0.9,
        0.85,
        0.8,
    ]
    assert [
        candidate["train_support"]["primary_nonoverlap_events"]
        for candidate in candidates
    ] == [38, 66, 89, 143, 193]
    assert payload["clock"]["rows"] == 0
    assert payload["clock"]["sha256"] == hashlib.sha256(clock.read_bytes()).hexdigest()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == support._canonical_hash(core)
