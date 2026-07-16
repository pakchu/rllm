from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from training import build_network_topology_broadening_support as support
from training import preregister_network_topology_broadening as prereg


def synthetic_network(rows: int = 220) -> pd.DataFrame:
    observation = pd.date_range("2021-01-01", periods=rows, freq="D")
    phase = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "observation_date": observation,
            "available_at": observation + pd.Timedelta(days=1, hours=4),
            "AdrActCnt": 100_000.0 + phase * 20.0,
            "TxCnt": 50_000.0 + phase * 10.0,
            "TxTfrCnt": 120_000.0 + phase * 15.0,
        }
    )


def test_strict_prior_z_ignores_not_yet_available_history() -> None:
    values = np.array([0.0, 1.0, 2.0, 50.0])
    available = np.array(
        [
            "2021-01-02",
            "2021-01-03",
            "2021-01-10",  # unavailable when the final row arrives
            "2021-01-04",
        ],
        dtype="datetime64[ns]",
    )
    zscore, counts = support._strict_prior_z(
        values, available, reference_days=10, minimum=2
    )
    assert counts[-1] == 2
    assert zscore[-1] == pytest.approx((50.0 - 0.5) / np.std([0.0, 1.0], ddof=1))


def test_build_features_uses_no_price_and_stale_rows_cannot_signal() -> None:
    network = synthetic_network()
    network.loc[150:, "AdrActCnt"] *= 1.3
    network.loc[150:, "TxTfrCnt"] *= 0.7
    network.loc[150, "available_at"] = network.loc[150, "observation_date"] + pd.Timedelta(days=4)
    features = support.build_features(network, prereg.Policy())
    assert not any("price" in column.lower() for column in features.columns)
    assert features.loc[150, "source_lag_days"] == 4.0
    assert not bool(features.loc[150, "eligible"])


def test_event_is_only_false_to_true_transition() -> None:
    network = synthetic_network()
    features = support.build_features(network, prereg.Policy())
    features["eligible"] = False
    features.loc[130:132, "eligible"] = True
    features["event"] = features["eligible"] & ~features["eligible"].shift(1, fill_value=False)
    assert features.index[features["event"]].tolist() == [130]


def test_schedule_is_delayed_and_nonoverlapping() -> None:
    policy = prereg.Policy()
    features = pd.DataFrame(
        {
            "observation_date": pd.to_datetime(["2021-03-01", "2021-03-03", "2021-03-10"]),
            "available_at": pd.to_datetime(
                ["2021-03-02 04:02", "2021-03-04 04:02", "2021-03-11 04:02"]
            ),
            "event": [True, True, True],
            "fanout": [1.0] * 3,
            "breadth": [1.0] * 3,
            "fanout_change": [-1.0] * 3,
            "breadth_change": [1.0] * 3,
            "fanout_z": [-1.0] * 3,
            "breadth_z": [1.0] * 3,
            "composite": [2.0] * 3,
            "source_lag_days": [1.2] * 3,
            "fanout_reference_count": [180] * 3,
            "breadth_reference_count": [180] * 3,
        }
    )
    clock = support.schedule_clock(features, policy)
    assert len(clock) == 2
    assert clock.iloc[0]["earliest_tradable_open"] == pd.Timestamp("2021-03-02 04:05")
    assert clock.iloc[0]["entry_date"] == pd.Timestamp("2021-03-02 04:10")
    assert clock.iloc[1]["entry_date"] >= clock.iloc[0]["exit_date"]


def test_load_preregistration_rejects_opened_outcomes(tmp_path) -> None:
    payload = prereg.build_manifest()
    payload["outcomes_opened"] = True
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"manifest_hash", "created_at"}
    }
    payload["manifest_hash"] = prereg.canonical_hash(core)
    path = tmp_path / "opened.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="cannot open outcomes"):
        support.load_preregistration(path)


def test_support_summary_uses_full_clock_month_concentration() -> None:
    clock = pd.DataFrame(
        {
            "entry_date": pd.to_datetime(
                ["2021-03-01", "2021-04-01", "2022-01-01", "2023-02-01"]
            )
        }
    )
    summary = support.support_summary(clock, prereg.build_manifest())
    assert summary["counts"]["train_2021"] == 2
    assert summary["counts"]["train_2022"] == 1
    assert summary["counts"]["selection_2023"] == 1
    assert summary["maximum_single_month_share"] == 0.25
    assert summary["passed"] is False
