import unittest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd

from execution.rex_llm_live import (
    RexLiveLoopConfig,
    RexLivePolicyConfig,
    RexLlmSelectorConfig,
    _seconds_until_next_interval,
    run_rex_live_loop,
    build_rex_live_policy_record,
)
from execution.wave_execution import WaveExecutionConfig, decision_from_policy_record, evaluate_execution_gate
from preprocessing.market_features import EXTENDED_MARKET_FEATURE_COLUMNS


def _base_features(n=120):
    df = pd.DataFrame(0.0, index=range(n), columns=list(EXTENDED_MARKET_FEATURE_COLUMNS))
    df["range_vol"] = 0.03
    df["kimchi_premium_change"] = -0.001
    df["trend_24"] = -0.01
    df["htf_4h_return_1"] = -0.01
    df["htf_1d_return_4"] = -0.10
    df["htf_3d_return_4"] = -0.03
    df["htf_1w_return_4"] = -0.02
    df["rex_36_range_pos"] = 0.80
    df["rex_144_range_pos"] = 0.75
    df["rex_576_range_pos"] = 0.50
    df["rex_2016_range_pos"] = 0.30
    df["rex_8640_range_pos"] = 0.20
    for w in (36, 144, 576, 2016, 8640):
        df[f"rex_{w}_range_width_pct"] = 0.05
        df[f"rex_{w}_max_to_cur_pct"] = 0.05
        df[f"rex_{w}_cur_to_min_pct"] = 0.20
    df["volume_zscore"] = 0.2
    df["taker_imbalance"] = -0.1
    # Make the final completed bar stronger than the historical 75% threshold.
    df.loc[n - 1, "htf_1d_return_4"] = -0.30
    df.loc[n - 1, "htf_3d_return_4"] = -0.10
    df.loc[n - 1, "htf_1w_return_4"] = -0.10
    return df


def _enriched(n=120):
    dates = pd.date_range("2026-01-01", periods=n, freq="5min")
    close = 100.0 + np.linspace(0.0, 1.0, n)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "dxy_available": 1.0,
            "kimchi_available": 1.0,
            "usdkrw_available": 1.0,
            "premium_available": 1.0,
            "funding_available": 1.0,
            "external_any_available": 1.0,
            "binance_aux_any_available": 1.0,
        }
    )


class TestRexLlmLive(unittest.TestCase):
    def test_builds_trade_record_and_execution_decision_for_short_candidate(self):
        features = _base_features()
        record = build_rex_live_policy_record(
            _enriched(),
            features,
            policy_cfg=RexLivePolicyConfig(min_positive_strengths=5),
        )
        self.assertEqual(record["prediction"], "TRADE")
        self.assertEqual(record["candidate_side"], "SHORT")
        decision = decision_from_policy_record(record)
        self.assertEqual(decision.signal, "SHORT")
        self.assertGreater(decision.current_atr, 0.0)

    def test_alternate_dual_regime_gate_matches_exported_rule(self):
        features = _base_features()
        features.loc[len(features) - 1, "range_vol"] = 0.01
        features.loc[len(features) - 1, "kimchi_premium_change"] = 0.001
        features.loc[len(features) - 1, "rex_8640_range_width_pct"] = 0.30
        features.loc[len(features) - 1, "usdkrw_zscore"] = 0.0

        record = build_rex_live_policy_record(
            _enriched(),
            features,
            policy_cfg=RexLivePolicyConfig(min_positive_strengths=5),
        )

        self.assertEqual(record["prediction"], "TRADE")
        self.assertEqual(record["matched_gate_set"], 1)
        self.assertFalse(record["gate_set_results"][0]["passed"])
        self.assertTrue(record["gate_set_results"][1]["passed"])

    def test_dual_regime_gate_blocks_when_both_branches_fail(self):
        features = _base_features()
        features.loc[len(features) - 1, "range_vol"] = 0.01
        features.loc[len(features) - 1, "kimchi_premium_change"] = 0.001
        features.loc[len(features) - 1, "rex_8640_range_width_pct"] = 0.20

        record = build_rex_live_policy_record(
            _enriched(),
            features,
            policy_cfg=RexLivePolicyConfig(min_positive_strengths=5),
        )

        self.assertEqual(record["prediction"], "ABSTAIN")
        self.assertIsNone(record["matched_gate_set"])
        self.assertIn("frozen_gate_block", record["reason"])

    def test_rex_selector_defaults_are_text_only(self):
        cfg = RexLlmSelectorConfig()
        self.assertEqual(cfg.model_name, "qwen2.5-1.5b-instruct")
        self.assertIn("qwen25_1p5b_lora_s80_2026-07-07", cfg.adapter_dir)

    def test_adapter_selector_can_block_otherwise_valid_rex_candidate(self):
        features = _base_features()
        with patch(
            "execution.rex_llm_live._score_rex_llm_selector",
            return_value={"enabled": True, "decision": "ABSTAIN", "scores": {"ABSTAIN": -1.0, "TRADE": -2.0}},
        ) as scorer:
            record = build_rex_live_policy_record(
                _enriched(),
                features,
                policy_cfg=RexLivePolicyConfig(min_positive_strengths=5),
                selector_cfg=RexLlmSelectorConfig(enabled=True, adapter_dir="dummy"),
            )
        scorer.assert_called_once()
        self.assertEqual(record["prediction"], "ABSTAIN")
        self.assertIn("llm_selector_block", record["reason"])
        self.assertEqual(record["llm_selector"]["decision"], "ABSTAIN")

    def test_adapter_selector_fail_closed_blocks_on_error(self):
        with patch("execution.rex_llm_live._score_rex_llm_selector", side_effect=RuntimeError("adapter missing")):
            record = build_rex_live_policy_record(
                _enriched(),
                _base_features(),
                policy_cfg=RexLivePolicyConfig(min_positive_strengths=5),
                selector_cfg=RexLlmSelectorConfig(enabled=True, adapter_dir="dummy", fail_closed=True),
            )
        self.assertEqual(record["prediction"], "ABSTAIN")
        self.assertIn("llm_selector_error_fail_closed", record["reason"])
        self.assertIn("adapter missing", record["llm_selector"]["error"])

    def test_gate_blocks_when_core_external_quality_missing_on_weekday(self):
        enriched = _enriched()
        enriched.loc[len(enriched) - 1, "date"] = pd.Timestamp("2026-01-02 12:00:00")  # Friday before FX close
        enriched.loc[len(enriched) - 1, "kimchi_available"] = 0.0
        record = build_rex_live_policy_record(
            enriched,
            _base_features(),
            policy_cfg=RexLivePolicyConfig(min_positive_strengths=5),
        )
        self.assertEqual(record["prediction"], "ABSTAIN")
        self.assertIn("missing_quality", record["reason"])

    def test_weekend_core_external_missing_matches_historical_neutral_fill(self):
        enriched = _enriched()
        enriched.loc[len(enriched) - 1, "date"] = pd.Timestamp("2026-01-03 12:00:00")  # Saturday FX closed
        for col in ["dxy_available", "kimchi_available", "usdkrw_available", "external_any_available"]:
            enriched.loc[len(enriched) - 1, col] = 0.0
        record = build_rex_live_policy_record(
            enriched,
            _base_features(),
            policy_cfg=RexLivePolicyConfig(min_positive_strengths=5),
        )
        self.assertEqual(record["prediction"], "TRADE")
        self.assertNotIn("missing_quality", record["reason"])


    def test_policy_record_age_uses_scorer_asof(self):
        enriched = _enriched()
        enriched.loc[len(enriched) - 1, "date"] = pd.Timestamp("2026-01-01 09:55:00")
        record = build_rex_live_policy_record(
            enriched,
            _base_features(),
            policy_cfg=RexLivePolicyConfig(min_positive_strengths=5),
            scorer_asof=pd.Timestamp("2026-01-01 10:10:00", tz="UTC"),
        )
        self.assertEqual(record["age_sec"], 900.0)
        decision = decision_from_policy_record(record)
        self.assertEqual(decision.age_sec, 900.0)

    def test_execution_config_still_blocks_live_candidate_without_bear_regime(self):
        record = build_rex_live_policy_record(
            _enriched(),
            _base_features(),
            policy_cfg=RexLivePolicyConfig(min_positive_strengths=5),
        )
        decision = decision_from_policy_record(record)
        cfg = WaveExecutionConfig(dry_run=True, allowed_signals=("SHORT",), manual_regime="UNKNOWN", require_bear_regime=True)
        gate = evaluate_execution_gate(cfg, decision)
        self.assertFalse(gate.allowed)
        self.assertIn("BEAR required", gate.reason)

    def test_short_only_execution_config_blocks_long_candidate(self):
        features = _base_features()
        features["htf_1d_return_4"] = 0.10
        features["htf_3d_return_4"] = 0.03
        features["htf_1w_return_4"] = 0.02
        features.loc[len(features) - 1, "htf_1d_return_4"] = 0.30
        features.loc[len(features) - 1, "htf_3d_return_4"] = 0.10
        features.loc[len(features) - 1, "htf_1w_return_4"] = 0.10
        features["trend_24"] = 0.01
        features["htf_4h_return_1"] = 0.01
        features["rex_36_range_pos"] = -0.80
        features["rex_144_range_pos"] = -0.75
        record = build_rex_live_policy_record(
            _enriched(),
            features,
            policy_cfg=RexLivePolicyConfig(min_positive_strengths=5),
        )
        self.assertEqual(record["candidate_side"], "LONG")
        decision = decision_from_policy_record(record)
        cfg = WaveExecutionConfig(dry_run=True, allowed_signals=("SHORT",), manual_regime="BEAR", require_bear_regime=True)
        gate = evaluate_execution_gate(cfg, decision)
        self.assertFalse(gate.allowed)
        self.assertIn("not in allowed_signals", gate.reason)

    def test_loop_sleep_aligns_to_next_interval_close_delay(self):
        wait = _seconds_until_next_interval(
            pd.Timestamp("2026-01-01 00:04:50", tz="UTC"),
            interval_minutes=5,
            close_delay_sec=15,
        )
        self.assertEqual(wait, 25.0)

        wait_after_boundary = _seconds_until_next_interval(
            pd.Timestamp("2026-01-01 00:05:16", tz="UTC"),
            interval_minutes=5,
            close_delay_sec=15,
        )
        self.assertEqual(wait_after_boundary, 299.0)

    def test_loop_skips_duplicate_signal_before_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "state.json"
            state_file.write_text(json.dumps({"last_signal_id": "same-signal"}))

            async_score = AsyncMock(return_value={"signal_id": "same-signal", "date": "2026-01-01 00:00:00"})
            with patch("execution.rex_llm_live.score_rex_live_once", async_score), patch(
                "execution.rex_llm_live.WaveExecutionBridge.from_env",
                side_effect=AssertionError("duplicate signal must not execute"),
            ):
                asyncio.run(
                    run_rex_live_loop(
                        execution_cfg=WaveExecutionConfig(dry_run=True, interval_minutes=5),
                        policy_cfg=RexLivePolicyConfig(min_positive_strengths=5),
                        loop_cfg=RexLiveLoopConfig(
                            state_file=state_file,
                            run_immediately=True,
                            max_iterations=1,
                        ),
                    )
                )

            async_score.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
