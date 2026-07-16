from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from training.export_emfx_daily_from_postgres import SYMBOLS
from training.search_emfx_coherent_pressure_reversal_alpha import (
    FIT_END,
    FIT_START,
    MIN_OBSERVATIONS,
    RISK_SYMBOLS,
    Config,
    build_daily_features,
    execution_contract,
    fit_reference_mask,
    json_hash,
    load_emfx,
    policy_masks,
    rolling_z,
    schedule_policy,
    selection_eligible,
    source_vintage_contract,
    validate_frozen_manifest,
    validate_selection_replay,
)
from training.search_inventory_purge_reclaim_alpha import (
    Config as ExecutionConfig,
    ExecutionEngine,
)


def emfx_source(days: int = 800) -> pd.DataFrame:
    dates = pd.date_range("2019-01-01", periods=days, freq="1D")
    rows: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(SYMBOLS):
        minimum = MIN_OBSERVATIONS[symbol]
        trend = np.arange(days, dtype=float) * (0.00008 + symbol_index * 0.00001)
        cycle = np.sin(np.arange(days, dtype=float) / (7.0 + symbol_index)) * 0.01
        closes = np.exp(np.log(1.0 + symbol_index) + trend + cycle)
        for day, close in zip(dates, closes, strict=True):
            rows.append(
                {
                    "symbol": symbol,
                    "observation_date": day,
                    "observations": minimum,
                    "close": close,
                    "last_ts": day + pd.Timedelta(hours=23, minutes=59),
                    "max_updated_at": day + pd.Timedelta(days=100),
                }
            )
    return pd.DataFrame(rows).sort_values(["observation_date", "symbol"]).reset_index(drop=True)


def feature_engine(days: int = 900) -> SimpleNamespace:
    dates = pd.Series(pd.date_range("2018-12-15", periods=days * 288, freq="5min"))
    close = 10_000.0 * np.exp(np.arange(len(dates), dtype=float) * 1e-8)
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


class TestEmfxCoherentPressureReversal(unittest.TestCase):
    def test_rolling_z_uses_prior_sessions_only(self) -> None:
        values = pd.Series(np.arange(200, dtype=float))
        before = rolling_z(values)
        changed = values.copy()
        changed.iloc[150:] += 100_000.0
        after = rolling_z(changed)
        pd.testing.assert_series_equal(before.iloc[:150], after.iloc[:150])
        self.assertAlmostEqual(before.iloc[150], (150.0 - 74.5) / np.std(np.arange(150), ddof=0))

    def test_build_features_is_prefix_stable_and_formula_is_exact(self) -> None:
        source = emfx_source()
        engine = feature_engine()
        full = build_daily_features(source, engine)
        cutoff = source["observation_date"].drop_duplicates().iloc[740]
        prefix = build_daily_features(source[source["observation_date"] < cutoff], engine)
        columns = ["observation_date", "em_coherent_pressure_1d", "anchor_date"]
        pd.testing.assert_frame_equal(prefix[columns], full.loc[: len(prefix) - 1, columns])
        self.assertTrue((full["anchor_date"] >= full["available_at"]).all())

        close = source.pivot(index="observation_date", columns="symbol", values="close")[list(SYMBOLS)]
        returns = np.log(close).diff()
        standardized = pd.DataFrame({symbol: rolling_z(returns[symbol]) for symbol in SYMBOLS})
        core = standardized[list(RISK_SYMBOLS)]
        mean = core.mean(axis=1)
        coherence = mean / np.sqrt(core.pow(2).mean(axis=1))
        expected = mean * coherence.abs()
        merged = full.set_index("observation_date")["em_coherent_pressure_1d"]
        day = merged.dropna().index[20]
        self.assertAlmostEqual(float(merged.loc[day]), float(expected.loc[day]), places=12)

    def test_load_emfx_fails_closed_on_universe_and_utc_boundary(self) -> None:
        source = emfx_source(days=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "emfx.csv"
            source.to_csv(path, index=False)
            with self.assertRaisesRegex(RuntimeError, "backfilled rather than point-in-time"):
                load_emfx(str(path), cutoff="2020-01-01")
            loaded = load_emfx(str(path), cutoff="2020-01-01", allow_backfilled=True)
            self.assertEqual(set(loaded["symbol"]), set(SYMBOLS))
            source[source["symbol"] != "USDHKD"].to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "fixed universe mismatch"):
                load_emfx(str(path), cutoff="2020-01-01", allow_backfilled=True)
            bad = emfx_source(days=2)
            bad.loc[0, "last_ts"] = bad.loc[0, "observation_date"] + pd.Timedelta(days=1)
            bad.to_csv(path, index=False)
            with self.assertRaisesRegex(RuntimeError, "outside its labelled UTC day"):
                load_emfx(str(path), cutoff="2020-01-01", allow_backfilled=True)

    def test_load_emfx_rejects_duplicate_day_symbol_rows(self) -> None:
        source = emfx_source(days=2)
        source = pd.concat([source, source.iloc[[0]]], ignore_index=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "emfx.csv"
            source.to_csv(path, index=False)
            with self.assertRaisesRegex(RuntimeError, "duplicate EM-FX"):
                load_emfx(str(path), cutoff="2020-01-01", allow_backfilled=True)

    def test_fit_reference_uses_executable_anchor_clock(self) -> None:
        frame = pd.DataFrame(
            {
                "anchor_date": pd.to_datetime(
                    [
                        pd.Timestamp(FIT_START) - pd.Timedelta(minutes=5),
                        FIT_START,
                        pd.Timestamp(FIT_END) - pd.Timedelta(minutes=5),
                        FIT_END,
                    ],
                    format="mixed",
                )
            }
        )
        self.assertEqual(fit_reference_mask(frame).tolist(), [False, True, True, False])

    def test_price_reversal_uses_tail_onsets_only(self) -> None:
        frame = pd.DataFrame(
            {
                "em_coherent_pressure_1d": [0.0, 2.0, 2.1, 0.0, -2.0, -2.1],
                "price_ret_24h": [0.0, -0.1, -0.1, 0.0, 0.1, 0.1],
            }
        )
        long_mask, short_mask = policy_masks(
            frame,
            feature="em_coherent_pressure_1d",
            lower_threshold=-1.0,
            upper_threshold=1.0,
            rule="price_reversal",
        )
        self.assertEqual(long_mask.tolist(), [False, True, False, False, False, False])
        self.assertEqual(short_mask.tolist(), [False, False, False, False, True, False])

    def test_selection_requires_full_year_risk_adjusted_strength(self) -> None:
        stats = {
            name: {"trades": 5, "absolute_return_pct": 1.0, "cagr_to_strict_mdd": 1.0}
            for name in ("fit_2021", "fit_2022", "select_2023_h1", "select_2023_h2")
        }
        stats["select_2023"] = {
            "trades": 10,
            "absolute_return_pct": 2.0,
            "cagr_to_strict_mdd": 2.0,
        }
        self.assertTrue(selection_eligible(stats))
        stats["select_2023"]["cagr_to_strict_mdd"] = 1.99
        self.assertFalse(selection_eligible(stats))

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

    def test_frozen_manifest_binds_costs_source_schedule_and_stats(self) -> None:
        cfg = Config(
            input_csv="market",
            funding_csv="funding",
            emfx_csv="emfx",
            output="out",
            manifest_output="manifest",
        )
        stats = {"fit": {"absolute_return_pct": 1.0}}
        core = {
            "fixed_symbol_universe": list(SYMBOLS),
            "execution_contract": execution_contract(cfg),
            "source_vintage_contract": source_vintage_contract(cfg),
            "source_prefix_hashes": {"emfx": "source"},
            "schedule_hashes": {"fit": "schedule"},
            "selection_stats_hash": json_hash(stats),
        }
        manifest = {**core, "manifest_hash": json_hash(core), "created_at": "ignored"}
        self.assertEqual(validate_frozen_manifest(manifest, cfg), manifest["manifest_hash"])
        validate_selection_replay(
            manifest,
            source_prefix_hashes={"emfx": "source"},
            schedule_hashes={"fit": "schedule"},
            stats=stats,
        )
        with self.assertRaisesRegex(RuntimeError, "manifest hash mismatch"):
            validate_frozen_manifest({**manifest, "manifest_hash": "changed"}, cfg)
        changed_cost = Config(**{**cfg.__dict__, "fee_rate": cfg.fee_rate * 2.0})
        with self.assertRaisesRegex(RuntimeError, "execution economics"):
            validate_frozen_manifest(manifest, changed_cost)
        changed_vintage = Config(**{**cfg.__dict__, "allow_backfilled_emfx": True})
        with self.assertRaisesRegex(RuntimeError, "source-vintage mode"):
            validate_frozen_manifest(manifest, changed_vintage)
        with self.assertRaisesRegex(RuntimeError, "source prefix changed"):
            validate_selection_replay(
                manifest,
                source_prefix_hashes={"emfx": "changed"},
                schedule_hashes={"fit": "schedule"},
                stats=stats,
            )
        with self.assertRaisesRegex(RuntimeError, "schedule changed"):
            validate_selection_replay(
                manifest,
                source_prefix_hashes={"emfx": "source"},
                schedule_hashes={"fit": "changed"},
                stats=stats,
            )
        with self.assertRaisesRegex(RuntimeError, "performance accounting changed"):
            validate_selection_replay(
                manifest,
                source_prefix_hashes={"emfx": "source"},
                schedule_hashes={"fit": "schedule"},
                stats={"fit": {"absolute_return_pct": 2.0}},
            )


if __name__ == "__main__":
    unittest.main()
