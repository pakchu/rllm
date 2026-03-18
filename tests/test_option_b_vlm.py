import unittest

from models.option_b_vlm import (
    TradingPromptState,
    action_to_id,
    auto_select_vlm_model,
    build_trading_prompt,
    detect_gpu_vram_gb,
    parse_action_label,
    recommended_vlm_models,
)


class TestOptionBVLM(unittest.TestCase):
    def test_recommended_models(self):
        models = recommended_vlm_models()
        self.assertGreaterEqual(len(models), 2)
        self.assertIn("Qwen/Qwen3-VL-8B-Instruct", models[0])

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


if __name__ == "__main__":
    unittest.main()
