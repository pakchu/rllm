from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from training import preregister_stablecoin_denominator_dislocation as sddr


def _source_frame(rows: int = 674) -> pd.DataFrame:
    dates = pd.date_range("2023-08-01", periods=rows, freq="1h", tz="UTC")
    phase = np.arange(rows) % 4
    usdc = np.choose(phase, [-0.002, -0.001, 0.001, 0.002]).astype(float)
    fdusd = np.choose(phase, [-0.0022, -0.0011, 0.0011, 0.0022]).astype(float)
    disagreement = np.choose(phase, [0.0002, 0.0003, 0.0002, 0.0003]).astype(
        float
    )
    return pd.DataFrame(
        {
            "date": dates,
            "source_available_at": dates + pd.Timedelta("1h"),
            "usdc_vs_usdt": usdc,
            "fdusd_vs_usdt": fdusd,
            "alt_consensus": (usdc + fdusd) / 2.0,
            "alt_disagreement": disagreement,
            "source_complete": True,
        }
    )


def test_robust_z_uses_strictly_prior_distribution() -> None:
    values = pd.Series(np.tile([-2.0, -1.0, 1.0, 2.0], 169)[:673])
    changed = values.copy()
    changed.iloc[-1] = 100.0
    baseline = sddr.prior_robust_z(values, sddr.FROZEN_CONFIG)
    shocked = sddr.prior_robust_z(changed, sddr.FROZEN_CONFIG)
    assert shocked.iloc[-1]["median"] == baseline.iloc[-1]["median"]
    assert shocked.iloc[-1]["scale"] == baseline.iloc[-1]["scale"]
    assert shocked.iloc[-1]["z"] != baseline.iloc[-1]["z"]
    pd.testing.assert_frame_equal(baseline.iloc[:-1], shocked.iloc[:-1])


def test_primary_clock_requires_two_book_coherence_and_builds_stale_control() -> None:
    frame = _source_frame()
    frame.loc[672, ["usdc_vs_usdt", "fdusd_vs_usdt"]] = [0.02, 0.021]
    frame.loc[672, "alt_disagreement"] = 0.0001
    frame.loc[673, ["usdc_vs_usdt", "fdusd_vs_usdt"]] = [0.0, 0.0]
    frame.loc[673, "alt_disagreement"] = 0.0
    states = sddr.signal_states(frame)
    assert bool(states.loc[672, "primary_active"])
    assert states.loc[672, "primary_side"] == 1
    assert not bool(states.loc[672, "stale_1h_active"])
    assert bool(states.loc[673, "stale_1h_active"])
    assert states.loc[673, "stale_1h_side"] == 1

    incoherent = frame.copy()
    incoherent.loc[672, "alt_disagreement"] = 1.0
    controlled = sddr.signal_states(incoherent)
    assert not bool(controlled.loc[672, "primary_active"])
    assert bool(controlled.loc[672, "no_disagreement_active"])


def test_onset_and_schedule_are_conservative_and_nonoverlapping() -> None:
    dates = pd.date_range("2023-09-01", periods=5, freq="1h", tz="UTC")
    frame = pd.DataFrame(
        {
            "date": dates,
            "source_available_at": dates + pd.Timedelta("1h"),
            "alt_disagreement": np.zeros(5),
        }
    )
    states = pd.DataFrame(
        {
            "primary_active": [True, True, False, True, False],
            "primary_side": [1, 1, 0, -1, 0],
            "z_usdc": [2.0, 2.1, 0.0, -2.0, 0.0],
            "z_fdusd": [2.2, 2.0, 0.0, -2.2, 0.0],
            "min_abs_z": [2.0, 2.0, 0.0, 2.0, 0.0],
            "prior_disagreement_q80": np.ones(5),
        }
    )
    events = sddr.schedule(frame, states, control="primary")
    assert len(events) == 2
    assert events["source_hour_start"].tolist() == [dates[0], dates[3]]
    assert events.iloc[0]["decision_time"] == dates[0] + pd.Timedelta("1h")
    assert events.iloc[0]["entry_time"] == dates[0] + pd.Timedelta("65min")
    assert events.iloc[0]["exit_time"] == dates[0] + pd.Timedelta("125min")
    entries = events["entry_time"].iloc[1:].reset_index(drop=True)
    prior_exits = events["exit_time"].iloc[:-1].reset_index(drop=True)
    assert (entries >= prior_exits).all()


def test_frozen_config_rejects_repair() -> None:
    with pytest.raises(ValueError, match="configuration is frozen"):
        sddr.signal_states(
            _source_frame(), replace(sddr.FROZEN_CONFIG, z_threshold=0.75)
        )


def test_real_source_loader_is_hash_bound_and_outcome_blind() -> None:
    frame, audit = sddr.load_source()
    assert tuple(frame.columns) == sddr.SOURCE_COLUMNS
    assert len(frame) == audit["rows"] == 3_592
    assert frame["source_complete"].all()
    assert audit["panel_sha256"] == sddr.SOURCE_PANEL_SHA256


def test_preregistration_contains_no_realized_outcome(tmp_path) -> None:
    payload = sddr.preregistration_payload(tmp_path / "freeze.json")
    assert payload["outcomes_opened"] is False
    assert payload["outcome_sources_opened"] is False
    assert payload["post_2023_source_rows_opened"] is False
    assert payload["source_only_controls"] == [
        "no_disagreement",
        "usdc_only",
        "fdusd_only",
        "stale_1h",
    ]
    serialized = str(payload).lower()
    for forbidden in ("absolute_return", "trade_count", "realized_cagr"):
        assert forbidden not in serialized
