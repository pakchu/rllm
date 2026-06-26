import json
import unittest

from training.path_shape_token_policy_tte import _maybe_invert, _parse_side_modes, _summary_from_prompt, target_label, tokens_from_row


class TestPathShapeTokenPolicyTTE(unittest.TestCase):
    def test_summary_from_trader_prompt_extracts_past_json_only(self):
        prompt = 'x\nPast-only analyzer summary: {"regime":"UPTREND","context_tags":["A"]}\n\nAnalyzer path-shape output: {"direction_pressure":"LONG_FAVORED"}'
        self.assertEqual(_summary_from_prompt(prompt)["regime"], "UPTREND")

    def test_tokens_include_symbolic_context(self):
        row = {"prompt": 'Past-only analyzer summary: {"regime":"UPTREND","context_tags":["A"],"evidence":{"momentum_1h_pct":2.0}}'}
        toks = tokens_from_row(row)
        self.assertIn("regime=UPTREND", toks)
        self.assertIn("tag=A", toks)
        self.assertIn("ev.momentum_1h_pct=1.5..3", toks)

    def test_target_label_maps_trader_json(self):
        self.assertEqual(target_label({"target": json.dumps({"gate": "TRADE", "side": "SHORT", "max_hold_bars": 144})}), "SHORT")
        self.assertEqual(target_label({"target": json.dumps({"gate": "NO_TRADE", "side": "NONE"})}), "NO_TRADE")


if __name__ == "__main__":
    unittest.main()


class TestSideMode(unittest.TestCase):
    def test_parse_side_modes_and_invert(self):
        self.assertEqual(_parse_side_modes("normal,invert"), ["normal", "invert"])
        self.assertEqual(_maybe_invert("LONG", "invert"), "SHORT")
        self.assertEqual(_maybe_invert("NO_TRADE", "invert"), "NO_TRADE")
