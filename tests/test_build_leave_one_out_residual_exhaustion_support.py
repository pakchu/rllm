from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training.build_leave_one_out_residual_exhaustion_support import (
    assert_clock_contract,
    candidate_frame,
    reserve_clock,
    support_stats,
)


def feature_fixture() -> dict[str, object]:
    index = pd.date_range("2023-03-01 01:00", periods=4, freq="1h")
    symbols = ["ADAUSDT", "BNBUSDT", "DOGEUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    rz = pd.DataFrame(0.0, index=index, columns=symbols)
    fz = pd.DataFrame(0.0, index=index, columns=symbols)
    beta = pd.DataFrame(1.0, index=index, columns=symbols)
    rz.loc[:, "ADAUSDT"] = -2.0
    rz.loc[:, "XRPUSDT"] = 2.0
    fz.loc[:, "ADAUSDT"] = 0.0
    fz.loc[:, "XRPUSDT"] = 0.0
    return {
        "symbols": symbols,
        "residual_z": rz,
        "flow_z": fz,
        "beta": beta,
        "source_clean": pd.Series(True, index=index),
        "finite": pd.Series(True, index=index),
    }


def test_candidate_direction_and_beta_neutral_weights() -> None:
    candidates = candidate_frame(feature_fixture(), 6)
    assert candidates["eligible"].all()
    assert candidates["long_symbol"].eq("ADAUSDT").all()
    assert candidates["short_symbol"].eq("XRPUSDT").all()
    assert np.allclose(candidates["long_weight"], 0.5)
    assert np.allclose(candidates["short_weight_abs"], 0.5)


def test_candidate_rejects_flow_confirmation_and_concentrated_weight() -> None:
    features = feature_fixture()
    features["flow_z"].loc[:, "XRPUSDT"] = 1.5  # type: ignore[index]
    assert not candidate_frame(features, 6)["eligible"].any()
    features = feature_fixture()
    features["beta"].loc[:, "ADAUSDT"] = 0.25  # type: ignore[index]
    features["beta"].loc[:, "XRPUSDT"] = 2.5  # type: ignore[index]
    assert not candidate_frame(features, 6)["eligible"].any()


def test_reservation_rejects_signal_formed_before_prior_exit() -> None:
    candidates = candidate_frame(feature_fixture(), 6)
    clock = reserve_clock(candidates, "L01", 2)
    assert len(clock) == 2
    assert clock.iloc[1]["signal_time"] >= clock.iloc[0]["exit_time"]
    assert_clock_contract(clock)


def test_clock_contract_rejects_outcome_like_column() -> None:
    clock = reserve_clock(candidate_frame(feature_fixture(), 6), "L01", 2)
    clock["future_return"] = 0.0
    with pytest.raises(RuntimeError, match="outcome-like"):
        assert_clock_contract(clock)


def test_support_stats_apply_all_preregistered_gates() -> None:
    dates = pd.date_range("2023-01-01", periods=160, freq="4D")
    pairs = [("ETHUSDT", "ADAUSDT"), ("SOLUSDT", "BNBUSDT"), ("XRPUSDT", "DOGEUSDT")]
    rows = []
    symbols = ["ADAUSDT", "BNBUSDT", "DOGEUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    for i, date in enumerate(dates):
        rows.append({
            "signal_time": date,
            "long_symbol": symbols[i % len(symbols)],
            "short_symbol": symbols[(i + 1) % len(symbols)],
        })
    stats = support_stats(pd.DataFrame(rows), 0.0)
    assert stats["events"] == 160
    assert stats["unique_ordered_pairs"] == 6
    assert stats["passes_support"] is False
