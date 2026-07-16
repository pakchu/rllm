from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from training.search_chain_activity_impulse_momentum_alpha import (
    Config,
    build_daily_features,
    event_onset,
    execution_contract,
    fit_reference_mask,
    load_network,
    policy_masks,
    rolling_z,
    schedule_policy,
)
from training.search_inventory_purge_reclaim_alpha import (
    Config as ExecutionConfig,
    ExecutionEngine,
)


def _network_frame(rows: int = 220) -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=rows, freq="D")
    phase = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "observation_date": dates,
            "available_at": dates + pd.Timedelta(days=1, hours=3),
            "AdrActCnt": 100_000.0 + phase * 30.0 + np.sin(phase / 3.0) * 500.0,
            "TxCnt": 200_000.0 + phase * 20.0 + np.cos(phase / 5.0) * 700.0,
            "TxTfrCnt": 300_000.0 + phase * 40.0 + np.sin(phase / 7.0) * 900.0,
        }
    )


class _FeatureEngine:
    def __init__(self) -> None:
        self.dates = pd.Series(pd.date_range("2020-12-01", "2022-01-01", freq="5min", inclusive="left"))
        prices = 30_000.0 * np.exp(np.arange(len(self.dates)) * 1e-7)
        self.market = pd.DataFrame({"close": prices})


class TestChainActivityImpulseMomentum(unittest.TestCase):
    def test_rolling_z_prefix_is_unchanged_by_future_rows(self) -> None:
        base = pd.Series(np.sin(np.arange(240) / 8.0) + np.arange(240) / 1000.0)
        extended = pd.concat([base, pd.Series([1e9, -1e9])], ignore_index=True)
        np.testing.assert_allclose(
            rolling_z(base).to_numpy(),
            rolling_z(extended).iloc[: len(base)].to_numpy(),
            equal_nan=True,
        )

    def test_load_network_rejects_precompletion_availability(self) -> None:
        frame = _network_frame(3)
        frame.loc[1, "available_at"] = frame.loc[1, "observation_date"] + pd.Timedelta(hours=23)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "network.csv.gz"
            frame.to_csv(path, index=False, compression="gzip")
            with self.assertRaisesRegex(RuntimeError, "before the UTC day completed"):
                load_network(str(path), cutoff="2022-01-01")

    def test_feature_anchor_never_precedes_provider_completion(self) -> None:
        engine = _FeatureEngine()
        features = build_daily_features(_network_frame(), engine)  # type: ignore[arg-type]
        self.assertTrue((features["anchor_date"] >= features["available_at"]).all())
        # The first usable activity shock needs a causal rolling warmup.
        self.assertTrue(features["activity_shock_1d"].iloc[:80].isna().all())

    def test_fit_reference_uses_anchor_availability_not_observation_label(self) -> None:
        frame = pd.DataFrame(
            {
                "observation_date": pd.to_datetime(["2022-12-30", "2022-12-31"]),
                "anchor_date": pd.to_datetime(["2022-12-31 05:00", "2023-01-01 05:00"]),
            }
        )
        np.testing.assert_array_equal(fit_reference_mask(frame), [True, False])

    def test_execution_contract_binds_costs(self) -> None:
        base = Config("m", "f", "n", "o", "x")
        expensive = Config("m", "f", "n", "o", "x", fee_rate=0.001)
        self.assertNotEqual(execution_contract(base), execution_contract(expensive))

    def test_policy_masks_trigger_only_on_state_onset_and_use_completed_momentum(self) -> None:
        frame = pd.DataFrame(
            {
                "activity_shock_1d": [0.0, 2.0, 2.1, 0.0, 2.2, 0.0, 2.3],
                "price_ret_24h": [0.1, 0.2, 0.3, 0.1, -0.2, 0.1, -0.3],
                "price_ret_72h": [0.1] * 7,
            }
        )
        long_active, short_active = policy_masks(
            frame,
            event="activity_shock_1d",
            threshold=1.5,
            rule="momentum",
        )
        np.testing.assert_array_equal(long_active, [False, True, False, False, False, False, False])
        np.testing.assert_array_equal(short_active, [False, False, False, False, True, False, True])
        np.testing.assert_array_equal(event_onset(np.array([False, True, True, False, True])), [False, True, False, False, True])

    def test_schedule_enters_next_open_and_prevents_overlap(self) -> None:
        dates = pd.Series(pd.date_range("2023-01-01", periods=2_000, freq="5min"))
        price = 20_000.0 + np.arange(len(dates), dtype=float)
        market = pd.DataFrame(
            {"date": dates, "open": price, "high": price * 1.001, "low": price * 0.999, "close": price}
        )
        funding = pd.DataFrame({"date": pd.Series([], dtype="datetime64[ns]"), "funding_rate": pd.Series([], dtype=float)})
        exec_cfg = ExecutionConfig("", "", "", "", "", leverage=0.5, fee_rate=0.0005, slippage_rate=0.0001)
        engine = ExecutionEngine(market, funding, exec_cfg)
        frame = pd.DataFrame({"anchor": [100, 101, 500]})
        trades = schedule_policy(
            engine,
            frame,
            np.array([True, True, True]),
            np.array([False, False, False]),
            hold_days=1,
            start="2023-01-01",
            end="2023-01-08",
        )
        self.assertEqual([trade.signal_position for trade in trades], [100, 500])
        self.assertEqual([trade.entry_position for trade in trades], [101, 501])
        self.assertGreater(trades[1].entry_position, trades[0].exit_position)


if __name__ == "__main__":
    unittest.main()
