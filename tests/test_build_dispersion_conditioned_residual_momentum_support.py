from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training import build_dispersion_conditioned_residual_momentum_support as dcrm


def synthetic_close(weeks: int = 20) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=weeks * 7 * 24 * 12, freq="5min")
    x = np.arange(len(index), dtype=float)
    data = {}
    for number, symbol in enumerate(dcrm.SYMBOLS, start=1):
        data[symbol] = 100.0 * np.exp((number * 1e-7) * x + 0.001 * np.sin(x / (97 + number)))
    return pd.DataFrame(data, index=index)


def test_feature_panels_use_leave_one_out_factor_and_shifted_beta() -> None:
    close = synthetic_close(weeks=12)
    hourly_return, factor, factor_30d, beta = dcrm.feature_panels(close)
    timestamp = hourly_return.dropna().index[100]
    expected = hourly_return.loc[timestamp, list(dcrm.SYMBOLS[1:])].median()
    assert factor.loc[timestamp, dcrm.SYMBOLS[0]] == pytest.approx(expected)
    assert factor_30d.notna().any().any()
    assert beta.notna().any().any()


def test_clock_is_causal_weekly_and_beta_neutral() -> None:
    clock = dcrm.build_clock(synthetic_close())
    assert not clock.empty
    dcrm.assert_clock_contract(clock)
    assert (clock["last_feature_time"] < clock["decision_time"]).all()
    assert (clock["decision_time"] < clock["entry_time"]).all()
    assert set(clock["gross_scale"]).issubset({0.25, 1.0})


def test_clock_contract_rejects_current_bar_feature() -> None:
    clock = dcrm.build_clock(synthetic_close())
    clock.loc[0, "last_feature_time"] = clock.loc[0, "decision_time"]
    with pytest.raises(RuntimeError, match="feature cutoff"):
        dcrm.assert_clock_contract(clock)


def test_clock_contract_rejects_outcome_column() -> None:
    clock = dcrm.build_clock(synthetic_close())
    clock["trade_return"] = 0.0
    with pytest.raises(RuntimeError, match="schema changed"):
        dcrm.assert_clock_contract(clock)


def test_support_separates_gross_buckets() -> None:
    clock = dcrm.build_clock(synthetic_close())
    stats = dcrm.support_stats(clock)
    assert sum(stats["gross_scale_counts"].values()) == stats["events"]
    assert set(stats["gross_scale_counts"]).issubset({"0.25", "1.0"})
    assert set(stats["gross_scale_concentration"]) == set(stats["gross_scale_counts"])


def test_clock_overlap_uses_only_entry_and_position_intervals() -> None:
    candidate = pd.DataFrame(
        {"entry_time": ["2023-01-01 00:05"], "exit_time": ["2023-01-01 00:15"]}
    )
    reference = pd.DataFrame(
        {"entry_time": ["2023-01-01 00:10"], "exit_time": ["2023-01-01 00:20"]}
    )
    overlap = dcrm.clock_overlap(candidate, reference)
    assert overlap["exact_entry_jaccard"] == 0.0
    assert overlap["position_time_jaccard_5m"] == pytest.approx(1 / 3)
    assert overlap["post_entry_returns_or_pnl_read"] is False
