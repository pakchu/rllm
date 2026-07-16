from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from training.download_coinmetrics_stablecoin_supply_daily import ASSETS
from training.search_inventory_purge_reclaim_alpha import Config as ExecutionConfig, ExecutionEngine
from training.search_stablecoin_supply_breadth_absorption_alpha import (
    Config,
    build_daily_features,
    execution_contract,
    fit_reference_mask,
    json_hash,
    load_stablecoin,
    policy_masks,
    rolling_z,
    schedule_policy,
    validate_frozen_manifest,
    validate_selection_replay,
    vintage_contract,
)


def stablecoin_source(days: int = 240) -> pd.DataFrame:
    dates = pd.date_range("2020-12-01", periods=days, freq="1D")
    rows: list[dict[str, object]] = []
    for asset_index, asset in enumerate(ASSETS):
        for day_index, day in enumerate(dates):
            rows.append(
                {
                    "asset": asset,
                    "observation_date": day,
                    "available_at": day + pd.Timedelta(days=1, minutes=30 + asset_index),
                    "supply": 1_000_000.0 * (asset_index + 1) * (1.0 + 0.0002 * day_index),
                }
            )
    return pd.DataFrame(rows).sort_values(["observation_date", "asset"]).reset_index(drop=True)


def feature_engine(days: int = 300) -> SimpleNamespace:
    dates = pd.Series(pd.date_range("2020-11-01", periods=days * 288, freq="5min"))
    close = 10_000.0 * np.exp(np.arange(len(dates), dtype=float) * 1e-7)
    return SimpleNamespace(dates=dates, market=pd.DataFrame({"close": close}))


def real_engine(days: int = 30) -> ExecutionEngine:
    dates = pd.date_range("2021-06-01", periods=days * 288, freq="5min")
    close = 10_000.0 * np.exp(np.arange(len(dates), dtype=float) * 1e-6)
    market = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
        }
    )
    funding = pd.DataFrame(
        {"date": pd.Series(dtype="datetime64[ns]"), "funding_rate": pd.Series(dtype=float)}
    )
    cfg = ExecutionConfig(
        input_csv="", metrics_csv="", funding_csv="", output="", manifest_output="", leverage=0.5
    )
    return ExecutionEngine(market, funding, cfg)


class TestStablecoinSupplyBreadthAbsorption(unittest.TestCase):
    def test_rolling_z_uses_prior_rows_only(self) -> None:
        values = pd.Series(np.arange(200, dtype=float))
        before = rolling_z(values)
        changed = values.copy()
        changed.iloc[150:] += 100_000.0
        after = rolling_z(changed)
        pd.testing.assert_series_equal(before.iloc[:150], after.iloc[:150])
        self.assertAlmostEqual(before.iloc[100], (100.0 - 49.5) / np.std(np.arange(100), ddof=0))

    def test_build_features_rejects_late_row_for_signal_and_is_prefix_stable(self) -> None:
        source = stablecoin_source()
        late_day = source["observation_date"].drop_duplicates().iloc[180]
        late = (source["observation_date"] == late_day) & (source["asset"] == "usdt_eth")
        source.loc[late, "available_at"] = late_day + pd.Timedelta(days=10)
        engine = feature_engine()
        full = build_daily_features(source, engine)
        self.assertFalse(bool(full.loc[full["observation_date"] == late_day, "valid_today"].iloc[0]))
        self.assertTrue(
            np.isnan(full.loc[full["observation_date"] == late_day, "breadth_7d_z"].iloc[0])
        )
        prefix_source = source[source["observation_date"] < source["observation_date"].min() + pd.Timedelta(days=220)]
        prefix = build_daily_features(prefix_source, engine)
        columns = ["observation_date", "breadth_7d_z", "supply_growth_30d_z", "anchor_date"]
        pd.testing.assert_frame_equal(prefix[columns], full.loc[: len(prefix) - 1, columns])
        self.assertTrue((full["anchor_date"] >= full["available_at"]).all())

    def test_fit_reference_uses_anchor_clock(self) -> None:
        frame = pd.DataFrame(
            {
                "anchor_date": pd.to_datetime(
                    ["2021-05-31 23:55", "2021-06-01", "2022-12-31 23:55", "2023-01-01"],
                    format="mixed",
                )
            }
        )
        self.assertEqual(fit_reference_mask(frame).tolist(), [False, True, True, False])

    def test_absorption_policy_maps_supply_breadth_divergence(self) -> None:
        frame = pd.DataFrame(
            {
                "breadth_7d_z": [0.0, 2.0, 2.1, 0.0, -2.0, -2.1],
                "price_ret_24h": [0.0, -0.1, -0.1, 0.0, 0.1, 0.1],
            }
        )
        long_mask, short_mask = policy_masks(
            frame,
            feature="breadth_7d_z",
            lower_threshold=-1.0,
            upper_threshold=1.0,
            rule="absorb",
        )
        self.assertEqual(long_mask.tolist(), [False, True, False, False, False, False])
        self.assertEqual(short_mask.tolist(), [False, False, False, False, True, False])

    def test_schedule_enters_next_open_and_prevents_overlap(self) -> None:
        engine = real_engine()
        frame = pd.DataFrame({"anchor": [100, 101, 1_000]})
        trades = schedule_policy(
            engine,
            frame,
            np.array([True, True, False]),
            np.array([False, False, True]),
            hold_days=1,
            start="2021-06-01",
            end="2021-07-01",
        )
        self.assertEqual(len(trades), 2)
        self.assertEqual(trades[0].entry_position, trades[0].signal_position + 1)
        self.assertGreater(trades[1].signal_position, trades[0].exit_position)

    def test_load_stablecoin_fails_closed_on_incomplete_basket(self) -> None:
        source = stablecoin_source(days=2)
        source = source[~((source["asset"] == "pax") & (source["observation_date"] == source["observation_date"].max()))]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stable.csv"
            source.to_csv(path, index=False)
            with self.assertRaisesRegex(RuntimeError, "incomplete fixed-basket day"):
                load_stablecoin(str(path), cutoff="2022-01-01")

    def test_execution_contract_binds_costs(self) -> None:
        cfg = Config(
            input_csv="market",
            funding_csv="funding",
            stablecoin_csv="stablecoin",
            output="out",
            manifest_output="manifest",
        )
        base = execution_contract(cfg)
        changed = execution_contract(Config(**{**cfg.__dict__, "fee_rate": cfg.fee_rate * 2.0}))
        self.assertNotEqual(base, changed)
        self.assertTrue(base["realized_funding"])
        self.assertTrue(base["full_calendar_cagr"])

    def test_frozen_manifest_and_replay_integrity_fail_closed(self) -> None:
        cfg = Config(
            input_csv="market",
            funding_csv="funding",
            stablecoin_csv="stablecoin",
            output="out",
            manifest_output="manifest",
        )
        stats = {"fit": {"absolute_return_pct": 1.0}}
        core = {
            "fixed_asset_universe": list(ASSETS),
            "execution_contract": execution_contract(cfg),
            "vintage_contract": vintage_contract(),
            "source_prefix_hashes": {"stablecoin": "source"},
            "schedule_hashes": {"fit": "schedule"},
            "selection_stats_hash": json_hash(stats),
        }
        manifest = {**core, "manifest_hash": json_hash(core), "created_at": "ignored"}
        self.assertEqual(validate_frozen_manifest(manifest, cfg), manifest["manifest_hash"])
        validate_selection_replay(
            manifest,
            source_prefix_hashes={"stablecoin": "source"},
            schedule_hashes={"fit": "schedule"},
            stats=stats,
        )
        tampered = {**manifest, "manifest_hash": "bad"}
        with self.assertRaisesRegex(RuntimeError, "manifest hash mismatch"):
            validate_frozen_manifest(tampered, cfg)
        changed_cost = Config(**{**cfg.__dict__, "fee_rate": cfg.fee_rate * 2.0})
        with self.assertRaisesRegex(RuntimeError, "execution economics"):
            validate_frozen_manifest(manifest, changed_cost)
        with self.assertRaisesRegex(RuntimeError, "source prefix changed"):
            validate_selection_replay(
                manifest,
                source_prefix_hashes={"stablecoin": "changed"},
                schedule_hashes={"fit": "schedule"},
                stats=stats,
            )
        with self.assertRaisesRegex(RuntimeError, "schedule changed"):
            validate_selection_replay(
                manifest,
                source_prefix_hashes={"stablecoin": "source"},
                schedule_hashes={"fit": "changed"},
                stats=stats,
            )
        with self.assertRaisesRegex(RuntimeError, "performance accounting changed"):
            validate_selection_replay(
                manifest,
                source_prefix_hashes={"stablecoin": "source"},
                schedule_hashes={"fit": "schedule"},
                stats={"fit": {"absolute_return_pct": 2.0}},
            )


if __name__ == "__main__":
    unittest.main()
