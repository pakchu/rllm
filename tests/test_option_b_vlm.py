import unittest

from models.option_b_vlm import (
    TradingPromptState,
    action_to_id,
    auto_select_vlm_model,
    resolve_vlm_model_alias,
    build_trading_prompt,
    detect_gpu_vram_gb,
    get_action_labels,
    make_action_system_prompt,
    parse_action_label,
    recommended_vlm_models,
)


class TestOptionBVLM(unittest.TestCase):
    def test_recommended_models(self):
        models = recommended_vlm_models()
        self.assertGreaterEqual(len(models), 2)
        self.assertEqual(models[0], "google/gemma-4-E4B-it")
        self.assertIn("Qwen/Qwen3-VL-8B-Instruct", models[1])

    def test_prompt_and_action_parse(self):
        prompt = build_trading_prompt(
            TradingPromptState(
                timeframe="1m",
                position_size_pct=0.0,
                last_entry_price=0.0,
                range_volatility_pct=0.12,
                extra_numeric_features=(("Close Z-Score (48)", -1.25),),
                extra_symbolic_features=(("Location", "DISCOUNT"),),
                context_tags=("DISCOUNT", "OVERSOLD"),
            ),
            prompt_style="hybrid",
        )
        self.assertIn("BUY", prompt)
        self.assertIn("Close Z-Score (48): -1.250", prompt)
        self.assertIn("Location: DISCOUNT", prompt)
        self.assertIn("Tags: DISCOUNT | OVERSOLD", prompt)
        self.assertEqual(parse_action_label("I choose SELL"), "SELL")
        self.assertEqual(parse_action_label("maybe hold here"), "HOLD")
        self.assertEqual(
            parse_action_label("Return BUY or HOLD or SELL. Final answer: SELL"),
            "SELL",
        )
        self.assertEqual(action_to_id("BUY"), 0)

    def test_trade_gate_prompt_and_parse(self):
        prompt = build_trading_prompt(
            TradingPromptState(
                timeframe="5m",
                position_size_pct=0.0,
                last_entry_price=0.0,
                range_volatility_pct=0.02,
            ),
            prompt_style="numeric",
            action_schema="trade_gate",
        )
        self.assertIn("TRADE/NO_TRADE", prompt)
        self.assertEqual(get_action_labels("trade_gate"), ("TRADE", "NO_TRADE"))
        self.assertIn("NO_TRADE", make_action_system_prompt("trade_gate"))
        self.assertEqual(
            parse_action_label("Final answer: NO_TRADE", default="NO_TRADE", labels=("TRADE", "NO_TRADE")),
            "NO_TRADE",
        )

    def test_trade_side_prompt_and_parse(self):
        prompt = build_trading_prompt(
            TradingPromptState(
                timeframe="5m",
                position_size_pct=0.0,
                last_entry_price=0.0,
                range_volatility_pct=0.02,
            ),
            prompt_style="symbolic",
            action_schema="trade_side",
        )
        self.assertIn("LONG/SHORT", prompt)
        self.assertIn("LONG", make_action_system_prompt("trade_side"))
        self.assertEqual(
            parse_action_label("SHORT", default="LONG", labels=("LONG", "SHORT")),
            "SHORT",
        )

    def test_multi_horizon_side_prompt_and_parse(self):
        prompt = build_trading_prompt(
            TradingPromptState(
                timeframe="5m",
                position_size_pct=0.0,
                last_entry_price=0.0,
                range_volatility_pct=0.02,
            ),
            prompt_style="symbolic",
            action_schema="multi_horizon_side",
        )
        self.assertIn("LONG_36", prompt)
        self.assertIn("SHORT_144", make_action_system_prompt("multi_horizon_side"))
        self.assertEqual(
            get_action_labels("multi_horizon_side"),
            (
                "NO_TRADE",
                "LONG_36",
                "LONG_72",
                "LONG_144",
                "SHORT_36",
                "SHORT_72",
                "SHORT_144",
            ),
        )
        self.assertEqual(
            parse_action_label(
                "Final answer: SHORT_72",
                default="NO_TRADE",
                labels=get_action_labels("multi_horizon_side"),
            ),
            "SHORT_72",
        )

    def test_symbolic_prompt_style(self):
        prompt = build_trading_prompt(
            TradingPromptState(
                timeframe="5m",
                position_size_pct=0.0,
                last_entry_price=0.0,
                range_volatility_pct=0.02,
                regime_label="UPTREND",
                volatility_label="HIGH",
                momentum_label="BULLISH",
                trend_strength_label="STRONG",
            ),
            prompt_style="symbolic",
        )
        self.assertIn("Regime: UPTREND", prompt)
        self.assertIn("Momentum: BULLISH", prompt)
        self.assertNotIn("Position Size (%)", prompt)

    def test_auto_model_helpers(self):
        vram = detect_gpu_vram_gb()
        if vram is not None:
            self.assertGreater(vram, 0.0)
        selected = auto_select_vlm_model(prefer_latest=True)
        self.assertIn(selected, recommended_vlm_models())
        self.assertEqual(resolve_vlm_model_alias("gemma4"), "google/gemma-4-E4B-it")


if __name__ == "__main__":
    unittest.main()
